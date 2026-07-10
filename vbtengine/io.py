"""Parallel archive reader: monthly ``.csv.gz`` partitions -> numpy arrays.

Design notes
------------
- **ThreadPoolExecutor, not ProcessPoolExecutor.** Decompressing and parsing
  csv.gz is I/O-plus-C-extension work: gzip and pyarrow's csv parser both
  release the GIL, so threads give real parallelism here without paying
  process spawn + pickling costs. Processes are reserved for CPU-bound
  Python-level compute (see ``studies/``).
- **Arrays-of-columns, not a DataFrame.** The loader returns a plain
  ``dict[str, np.ndarray]``. Downstream code indexes raw contiguous float64
  buffers -- no index alignment, no block manager, no accidental copies.
- **Validated once at the boundary.** Timestamps are checked strictly
  increasing at load time so every downstream routine can assume sorted
  input and use ``searchsorted`` fearlessly.
"""

from __future__ import annotations

import gzip
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.csv as pacsv

MAX_IO_WORKERS = 24

#: Canonical column set every partition must provide (case-insensitive).
REQUIRED_COLUMNS = ("ts_utc", "open", "high", "low", "close", "volume")

_DTYPES = {
    "ts": np.int64,
    "open": np.float64,
    "high": np.float64,
    "low": np.float64,
    "close": np.float64,
    "volume": np.int64,
}


class ArchiveError(RuntimeError):
    """Raised for malformed or inconsistent archive partitions."""


def _timestamps_to_epoch_ns(col: np.ndarray) -> np.ndarray:
    """Normalize a ts column to int64 epoch nanoseconds.

    Accepts integer epoch-ns directly, or ISO-8601 strings via a single
    vectorized ``datetime64[ns]`` cast.
    """
    if np.issubdtype(col.dtype, np.integer):
        return col.astype(np.int64)
    if np.issubdtype(col.dtype, np.datetime64):
        return col.astype("datetime64[ns]").astype(np.int64)
    return np.asarray(col, dtype="datetime64[ns]").astype(np.int64)


def read_partition(path: Path) -> dict[str, np.ndarray]:
    """Read one ``.csv.gz`` partition into a dict of numpy arrays.

    Header names are matched case-insensitively (``TS_UTC``/``Open``/...
    are all accepted) because real archives are rarely consistent.
    """
    with gzip.open(path, "rb") as fh:
        table = pacsv.read_csv(fh)

    lower_to_actual = {name.lower(): name for name in table.column_names}
    missing = [c for c in REQUIRED_COLUMNS if c not in lower_to_actual]
    if missing:
        raise ArchiveError(f"{path}: missing columns {missing}")

    out: dict[str, np.ndarray] = {}
    canonical_names = ("ts", "open", "high", "low", "close", "volume")
    for canonical, key in zip(canonical_names, REQUIRED_COLUMNS, strict=True):
        col = table.column(lower_to_actual[key]).to_numpy(zero_copy_only=False)
        if canonical == "ts":
            out["ts"] = _timestamps_to_epoch_ns(col)
        else:
            out[canonical] = col.astype(_DTYPES[canonical])
    return out


def list_partitions(archive_root: Path, symbol: str) -> list[Path]:
    """Monthly partition files for a symbol, in chronological (name) order."""
    sym_dir = Path(archive_root) / f"{symbol}_1m"
    if not sym_dir.is_dir():
        raise ArchiveError(f"no archive directory for symbol {symbol!r}: {sym_dir}")
    paths = sorted(sym_dir.glob("*.csv.gz"))
    if not paths:
        raise ArchiveError(f"no partitions under {sym_dir}")
    return paths


def load_symbol(
    archive_root: Path, symbol: str, max_workers: int = MAX_IO_WORKERS
) -> dict[str, np.ndarray]:
    """Load a symbol's full history into contiguous numpy arrays.

    Partitions are read in parallel (threads; see module docstring), then
    concatenated in filename order -- ``YYYY-MM`` sorts chronologically, so
    no post-hoc sort of millions of rows is needed. Timestamps are validated
    strictly increasing across the stitched result.
    """
    paths = list_partitions(archive_root, symbol)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        parts = list(pool.map(read_partition, paths))

    data = {
        key: np.concatenate([p[key] for p in parts])
        for key in ("ts", "open", "high", "low", "close", "volume")
    }

    ts = data["ts"]
    if ts.size > 1 and not np.all(ts[1:] > ts[:-1]):
        bad = int(np.flatnonzero(ts[1:] <= ts[:-1])[0])
        raise ArchiveError(
            f"{symbol}: timestamps not strictly increasing at row {bad + 1} "
            f"(ts[{bad}]={ts[bad]}, ts[{bad + 1}]={ts[bad + 1]})"
        )
    return data


def list_symbols(archive_root: Path) -> list[str]:
    """Symbols present in the archive (directories named ``{SYMBOL}_1m``)."""
    root = Path(archive_root)
    return sorted(p.name[: -len("_1m")] for p in root.glob("*_1m") if p.is_dir())
