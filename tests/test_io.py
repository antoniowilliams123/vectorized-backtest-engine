"""Archive reader: header tolerance, ordering validation, roundtrips."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from vbtengine import io, synthetic

MINUTE_NS = 60_000_000_000


def _write_csv_gz(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(header + "\n")
        fh.write("\n".join(rows) + "\n")


def test_reads_capitalized_headers(tmp_path: Path) -> None:
    ts0 = 1_700_000_000_000_000_000
    _write_csv_gz(
        tmp_path / "TEST_1m" / "2023-11.csv.gz",
        "TS_UTC,Open,High,Low,Close,Volume",
        [
            f"{ts0},100.0,101.0,99.0,100.5,1000",
            f"{ts0 + MINUTE_NS},100.5,102.0,100.0,101.5,2000",
        ],
    )
    data = io.load_symbol(tmp_path, "TEST")
    np.testing.assert_array_equal(data["ts"], [ts0, ts0 + MINUTE_NS])
    np.testing.assert_allclose(data["high"], [101.0, 102.0])
    np.testing.assert_array_equal(data["volume"], [1000, 2000])
    assert data["close"].dtype == np.float64
    assert data["volume"].dtype == np.int64


def test_missing_column_raises(tmp_path: Path) -> None:
    _write_csv_gz(
        tmp_path / "BAD_1m" / "2023-11.csv.gz",
        "ts_utc,open,high,low,close",  # no volume
        ["1,1,1,1,1"],
    )
    with pytest.raises(io.ArchiveError, match="missing columns"):
        io.load_symbol(tmp_path, "BAD")


def test_non_monotonic_timestamps_raise(tmp_path: Path) -> None:
    _write_csv_gz(
        tmp_path / "OOO_1m" / "2023-11.csv.gz",
        "ts_utc,open,high,low,close,volume",
        ["200,1,1,1,1,1", "100,1,1,1,1,1"],
    )
    with pytest.raises(io.ArchiveError, match="not strictly increasing"):
        io.load_symbol(tmp_path, "OOO")


def test_multi_partition_stitch_order(tmp_path: Path) -> None:
    header = "ts_utc,open,high,low,close,volume"
    _write_csv_gz(tmp_path / "SYM_1m" / "2023-02.csv.gz", header, ["2000,2,2,2,2,2"])
    _write_csv_gz(tmp_path / "SYM_1m" / "2023-01.csv.gz", header, ["1000,1,1,1,1,1"])
    data = io.load_symbol(tmp_path, "SYM")
    np.testing.assert_array_equal(data["ts"], [1000, 2000])


def test_missing_symbol_raises(tmp_path: Path) -> None:
    with pytest.raises(io.ArchiveError, match="no archive directory"):
        io.load_symbol(tmp_path, "NOPE")


def test_synthetic_roundtrip_and_determinism(tmp_path: Path) -> None:
    data = synthetic.generate_symbol("RT", start_year=2023, years=1, seed=5)
    again = synthetic.generate_symbol("RT", start_year=2023, years=1, seed=5)
    np.testing.assert_array_equal(data["ts"], again["ts"])
    np.testing.assert_allclose(data["close"], again["close"])

    synthetic.write_symbol_archive(tmp_path, "RT", data)
    loaded = io.load_symbol(tmp_path, "RT")
    np.testing.assert_array_equal(loaded["ts"], data["ts"])
    # CSV serialization rounds prices to 4 decimals.
    np.testing.assert_allclose(loaded["close"], data["close"], atol=1e-4)
    assert np.all(loaded["high"] >= loaded["low"])
    assert io.list_symbols(tmp_path) == ["RT"]
