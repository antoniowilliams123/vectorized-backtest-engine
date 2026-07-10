"""Granular parquet result writer.

Every study persists **row-per-event** output with all metric columns --
never just summary statistics. The payoff: future questions ("what does the
distribution look like above the median?", "how does metric A interact with
metric B?") become a parquet filter instead of a re-run of the whole
computation.

Naming convention: ``{study}_{YYYY-MM-DD}.parquet`` under a results
directory, so runs are date-stamped and reproducible side by side.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_results(
    columns: dict[str, np.ndarray],
    study: str,
    out_dir: Path,
    run_date: dt.date | None = None,
) -> Path:
    """Write a dict of equal-length arrays as ``{study}_{YYYY-MM-DD}.parquet``.

    Accepts numpy arrays (or lists) directly -- no DataFrame construction on
    the way out. Prints the saved path and row count, and returns the path.
    """
    if not columns:
        raise ValueError("columns must be non-empty")
    lengths = {name: len(col) for name, col in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"columns have differing lengths: {lengths}")

    run_date = run_date or dt.date.today()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{study}_{run_date.isoformat()}.parquet"

    table = pa.table({name: pa.array(col) for name, col in columns.items()})
    pq.write_table(table, path, compression="zstd")

    n_rows = next(iter(lengths.values()))
    print(f"[results] saved {path} ({n_rows:,} rows, {len(columns)} columns)")
    return path
