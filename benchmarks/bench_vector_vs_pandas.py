"""Honest micro-benchmark: numpy toolkit vs the naive pandas equivalents.

Three operations that dominate minute-bar research loops, on ~2M rows:

1. rolling mean (window=20):   ``np.convolve``  vs ``Series.rolling().mean()``
2. lag/shift (1 bar):          slice assignment vs ``Series.shift()``
3. per-day high:               ``maximum.reduceat`` vs ``groupby(day).max()``

"Naive pandas" means what a straightforward first implementation looks like
(including a datetime-normalized groupby key, which is how per-day stats are
usually first written). The pandas timings are best-of-``REPEATS`` -- this
is a fair fight, not a strawman: pandas is excellent at what it is for; it
simply carries machinery a sorted-array hot loop does not need.

Run: ``python benchmarks/bench_vector_vs_pandas.py``
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vbtengine import vector  # noqa: E402

N_ROWS = 2_000_000
WINDOW = 20
REPEATS = 5
BARS_PER_DAY = 390


def _best_of(fn: Callable[[], object], repeats: int = REPEATS) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def make_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(N_ROWS) * 5e-4))
    n_days = -(-N_ROWS // BARS_PER_DAY)
    day_starts = np.arange(n_days, dtype=np.int64) * vector.NS_PER_DAY
    minute = np.arange(N_ROWS, dtype=np.int64) % BARS_PER_DAY
    ts = day_starts.repeat(BARS_PER_DAY)[:N_ROWS] + minute * 60_000_000_000
    df = pd.DataFrame({"ts": pd.to_datetime(ts), "close": close})
    return ts, close, df


def main() -> None:
    ts, close, df = make_data()
    day_ids = ts // vector.NS_PER_DAY
    starts = vector.group_starts(day_ids)

    rows: list[tuple[str, float, float]] = []

    t_np = _best_of(lambda: vector.rolling_mean(close, WINDOW))
    t_pd = _best_of(lambda: df["close"].rolling(WINDOW).mean())
    rows.append((f"rolling mean (w={WINDOW})", t_np, t_pd))

    t_np = _best_of(lambda: vector.shift(close, 1))
    t_pd = _best_of(lambda: df["close"].shift(1))
    rows.append(("shift by 1 bar", t_np, t_pd))

    t_np = _best_of(lambda: vector.group_max(close, starts))
    t_pd = _best_of(lambda: df.groupby(df["ts"].dt.normalize())["close"].max())
    rows.append(("per-day high", t_np, t_pd))

    print(f"\n{N_ROWS:,} rows, best of {REPEATS} runs (seconds)\n")
    print(f"{'operation':<24} {'numpy':>10} {'pandas':>10} {'speedup':>9}")
    print("-" * 56)
    for name, t_np, t_pd in rows:
        print(f"{name:<24} {t_np:>10.4f} {t_pd:>10.4f} {t_pd / t_np:>8.1f}x")
    print()


if __name__ == "__main__":
    main()
