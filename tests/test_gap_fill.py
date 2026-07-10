"""Gap-fill demo logic on a tiny handcrafted archive with hand-known answers."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
from gap_fill_demo import compute_symbol_gap_fills

MINUTE_NS = 60_000_000_000
DAY_NS = 86_400_000_000_000

# Three 4-bar days, hand-built. Day boundaries are UTC days.
#
# Day 1: flat around 100; last close = 100.0.
# Day 2: opens 102.0 (gap +2.0%). Lows: 101.5, 100.8, 100.0, 100.2
#        -> touches prior close 100.0 at bar offset 2 => FILLED, 2 minutes.
#        Day-2 close = 100.9.
# Day 3: opens 99.0 vs prior close 100.9 (gap ~ -1.883%). Highs never
#        reach 100.9 => NOT filled.
_DAYS = [
    # (open, high, low, close) per bar
    [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
    ],
    [
        (102.0, 102.5, 101.5, 102.0),
        (102.0, 102.2, 100.8, 101.0),
        (101.0, 101.5, 100.0, 100.5),
        (100.5, 101.0, 100.2, 100.9),
    ],
    [
        (99.0, 99.5, 98.5, 99.2),
        (99.2, 100.0, 98.8, 99.5),
        (99.5, 100.5, 99.2, 100.0),
        (100.0, 100.8, 99.9, 100.4),
    ],
]


@pytest.fixture()
def tiny_archive(tmp_path: Path) -> Path:
    rows = []
    for d, bars in enumerate(_DAYS):
        for b, (o, h, lo, c) in enumerate(bars):
            ts = (19_723 + d) * DAY_NS + b * MINUTE_NS  # weekdays in Jan 2024
            rows.append(f"{ts},{o},{h},{lo},{c},{100 + b}")
    path = tmp_path / "TINY_1m" / "2024-01.csv.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as fh:
        fh.write("ts_utc,open,high,low,close,volume\n")
        fh.write("\n".join(rows) + "\n")
    return tmp_path


def test_gap_sizes(tiny_archive: Path) -> None:
    out = compute_symbol_gap_fills(tiny_archive, "TINY")
    assert out["gap_pct"].size == 2  # day 1 has no prior close
    np.testing.assert_allclose(out["gap_pct"], [0.02, (99.0 - 100.9) / 100.9])
    np.testing.assert_array_equal(out["gap_up"], [True, False])
    np.testing.assert_allclose(out["prior_close"], [100.0, 100.9])
    np.testing.assert_allclose(out["open_price"], [102.0, 99.0])


def test_fill_flags_and_timing(tiny_archive: Path) -> None:
    out = compute_symbol_gap_fills(tiny_archive, "TINY")
    np.testing.assert_array_equal(out["filled"], [True, False])
    np.testing.assert_array_equal(out["minutes_to_fill"], [2, -1])


def test_excursions(tiny_archive: Path) -> None:
    out = compute_symbol_gap_fills(tiny_archive, "TINY")
    # Day 2 (gap up, fill direction = down):
    #   toward fill: open 102.0 - day low 100.0 = 2.0
    #   away:        day high 102.5 - open 102.0 = 0.5
    np.testing.assert_allclose(out["max_toward_fill_pct"][0], 2.0 / 102.0)
    np.testing.assert_allclose(out["max_away_from_fill_pct"][0], 0.5 / 102.0)
    # Day 3 (gap down, fill direction = up):
    #   toward fill: day high 100.8 - open 99.0 = 1.8
    #   away:        open 99.0 - day low 98.5 = 0.5
    np.testing.assert_allclose(out["max_toward_fill_pct"][1], 1.8 / 99.0)
    np.testing.assert_allclose(out["max_away_from_fill_pct"][1], 0.5 / 99.0)


def test_session_bars_and_dates(tiny_archive: Path) -> None:
    out = compute_symbol_gap_fills(tiny_archive, "TINY")
    np.testing.assert_array_equal(out["session_bars"], [4, 4])
    assert list(out["date"]) == ["2024-01-02", "2024-01-03"]
    assert set(out["symbol"]) == {"TINY"}
