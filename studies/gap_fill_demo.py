"""Demo study: overnight gap-fill frequency on the synthetic archive.

**What it measures.** For every (symbol, day): the overnight gap -- today's
open versus yesterday's close -- and whether price traded back to
yesterday's close at some point during the day (the gap "filled"), plus how
long that took and how far price ran on either side of the open.

**Why this study.** Gap-fill frequency is a textbook descriptive statistic,
chosen deliberately because it is public knowledge and it exercises every
layer of the engine: parallel archive loading, day bucketing, per-day
first/last reductions, and the forward-path event study. It is a demo of
the *engine*, not tradeable research -- and it runs on synthetic data, so
the numbers mean nothing about any market.

**Execution model.** Symbols fan out across a ``ProcessPoolExecutor``
(CPU-bound numpy work in each worker); inside each worker the monthly
partitions load via threads. Granular row-per-event results go to parquet.

Run: ``python studies/gap_fill_demo.py``
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vbtengine import io, synthetic, vector  # noqa: E402
from vbtengine.engine import forward_event_study  # noqa: E402
from vbtengine.results import write_results  # noqa: E402

MAX_COMPUTE_WORKERS = 24
STUDY_NAME = "gap_fill_demo"


def compute_symbol_gap_fills(archive_root: Path, symbol: str) -> dict[str, np.ndarray]:
    """Row-per-day gap metrics for one symbol. Pure array ops throughout."""
    data = io.load_symbol(archive_root, symbol)
    ts, high, low = data["ts"], data["high"], data["low"]

    day_ids = ts // vector.NS_PER_DAY
    starts = vector.group_starts(day_ids)
    ends = vector.group_ends(starts, ts.size)

    day_open = vector.group_first(data["open"], starts)
    day_close = vector.group_last(data["close"], starts)
    bars_per_day = np.diff(np.append(starts, ts.size))

    # Events: every day after the first. Reference = today's open,
    # target = prior day's close.
    prior_close = day_close[:-1]
    entry_idx = starts[1:]
    open_px = day_open[1:]
    gap = open_px - prior_close
    gap_pct = gap / prior_close

    keep = gap != 0.0  # a zero gap is already "filled"; exclude it
    entry_idx, open_px, prior_close = entry_idx[keep], open_px[keep], prior_close[keep]
    gap_pct = gap_pct[keep]
    last_valid = ends[1:][keep]
    n_fwd_bars = bars_per_day[1:][keep]

    # Fill direction: gapped up -> price must come DOWN to prior close.
    direction = np.where(gap_pct > 0, -1, 1).astype(np.int64)

    res = forward_event_study(
        high=high,
        low=low,
        entry_idx=entry_idx,
        reference_price=open_px,
        target_price=prior_close,
        direction=direction,
        max_horizon=int(n_fwd_bars.max(initial=1)),
        last_valid_idx=last_valid,
    )

    dates = (ts[entry_idx].astype("datetime64[ns]").astype("datetime64[D]")).astype(str)
    return {
        "symbol": np.full(entry_idx.size, symbol, dtype=object),
        "date": dates,
        "gap_pct": gap_pct,
        "gap_abs_pct": np.abs(gap_pct),
        "gap_up": gap_pct > 0,
        "open_price": open_px,
        "prior_close": prior_close,
        "filled": res.hit,
        "minutes_to_fill": np.where(res.hit, res.bars_to_hit, -1).astype(np.int64),
        "max_toward_fill_pct": res.mfe / open_px,
        "max_away_from_fill_pct": res.mae / open_px,
        "session_bars": res.horizon_bars,
    }


def summarize(rows: dict[str, np.ndarray]) -> None:
    """Print fill rate by gap-size decile plus an events/day frequency line."""
    gap_abs = rows["gap_abs_pct"]
    filled = rows["filled"]
    mins = rows["minutes_to_fill"]

    edges = np.quantile(gap_abs, np.linspace(0.0, 1.0, 11))
    decile = np.clip(np.searchsorted(edges, gap_abs, side="right") - 1, 0, 9)

    print("\nGap-fill frequency by |gap| decile (synthetic data -- demo only)")
    cols = ("decile", "|gap| range (bps)", "events", "fill rate", "med mins")
    print(f"{cols[0]:>6} {cols[1]:>22} {cols[2]:>8} {cols[3]:>10} {cols[4]:>9}")
    for d in range(10):
        m = decile == d
        n = int(m.sum())
        rate = float(filled[m].mean()) if n else float("nan")
        filled_mins = mins[m & filled]
        med = float(np.median(filled_mins)) if filled_mins.size else float("nan")
        lo_bps, hi_bps = edges[d] * 1e4, edges[d + 1] * 1e4
        print(f"{d:>6} {lo_bps:>10.1f} - {hi_bps:>8.1f} {n:>8,} {rate:>9.1%} {med:>9.0f}")

    n_events = gap_abs.size
    n_symbols = len(np.unique(rows["symbol"]))
    n_days = len(np.unique(rows["date"]))
    overall = float(filled.mean())
    print(
        f"\nOverall: {overall:.1%} of {n_events:,} gaps filled | "
        f"{n_events / max(n_days, 1) / n_symbols:.2f} events per symbol-day "
        f"({n_symbols} symbols x {n_days:,} days)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--max-workers", type=int, default=MAX_COMPUTE_WORKERS)
    args = parser.parse_args()

    synthetic.ensure_archive(args.archive_root)
    symbols = io.list_symbols(args.archive_root)
    print(f"[demo] {len(symbols)} symbols: {', '.join(symbols)}")

    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        per_symbol = list(
            pool.map(compute_symbol_gap_fills, [args.archive_root] * len(symbols), symbols)
        )

    combined = {
        key: np.concatenate([p[key] for p in per_symbol]) for key in per_symbol[0]
    }
    combined["symbol"] = combined["symbol"].astype(str)

    write_results(combined, STUDY_NAME, args.out_dir)
    summarize(combined)


if __name__ == "__main__":
    main()
