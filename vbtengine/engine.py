"""Generic vectorized event study over forward bar paths.

Given per-event entry indices into a bar series, a per-event reference price
and target price, and a horizon, compute for every event:

- **MFE** (maximum favorable excursion): the best move toward the target
  over the forward window, in price units relative to the reference.
- **MAE** (maximum adverse excursion): the worst move away from it.
- **hit**: whether the target price traded within the window.
- **bars_to_hit**: bars elapsed until the first touch (0 if never).

Implementation: one gather-matrix of forward indices, ``events x horizon``,
built by broadcasting -- no Python loop over bars *or* events. For minute
data with day-bounded horizons the matrix is tiny (thousands of events x a
few hundred bars), so the O(events * horizon) memory is trivial and every
metric falls out of masked reductions along axis 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventStudyResult:
    """Column-per-metric result arrays, one row per event."""

    mfe: np.ndarray
    """Max favorable excursion (>= 0), price units toward the target."""

    mae: np.ndarray
    """Max adverse excursion (>= 0), price units away from the target."""

    hit: np.ndarray
    """Bool: target price traded within the horizon."""

    bars_to_hit: np.ndarray
    """Bars from entry to first touch; 0 where ``hit`` is False."""

    horizon_bars: np.ndarray
    """Actual forward bars evaluated per event (after end clipping)."""

    def as_columns(self, prefix: str = "") -> dict[str, np.ndarray]:
        """Dict-of-arrays view, ready for the parquet writer."""
        return {
            f"{prefix}mfe": self.mfe,
            f"{prefix}mae": self.mae,
            f"{prefix}hit": self.hit,
            f"{prefix}bars_to_hit": self.bars_to_hit,
            f"{prefix}horizon_bars": self.horizon_bars,
        }


def forward_event_study(
    high: np.ndarray,
    low: np.ndarray,
    entry_idx: np.ndarray,
    reference_price: np.ndarray,
    target_price: np.ndarray,
    direction: np.ndarray,
    max_horizon: int,
    last_valid_idx: np.ndarray | None = None,
) -> EventStudyResult:
    """Vectorized forward-path metrics for a batch of events.

    Parameters
    ----------
    high, low:
        Full bar series (1-D, sorted by time).
    entry_idx:
        Index of each event's entry bar. The forward window starts at the
        entry bar itself (``entry_idx + 0``) and spans ``max_horizon`` bars.
    reference_price:
        Per-event baseline from which excursions are measured.
    target_price:
        Per-event level whose first touch defines ``hit``/``bars_to_hit``.
    direction:
        Per-event ``+1`` if the favorable direction is up (target above
        reference), ``-1`` if down. Excursions are signed accordingly.
    max_horizon:
        Maximum forward bars to evaluate.
    last_valid_idx:
        Optional per-event index of the last bar the path may use (e.g. the
        last bar of the event's day). Bars past it are masked out.
    """
    entry_idx = np.asarray(entry_idx, dtype=np.int64)
    reference_price = np.asarray(reference_price, dtype=np.float64)
    target_price = np.asarray(target_price, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.int64)
    n_bars = high.shape[0]
    n_events = entry_idx.shape[0]

    if n_events == 0:
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        return EventStudyResult(empty_f, empty_f, np.empty(0, dtype=bool), empty_i, empty_i)

    if last_valid_idx is None:
        last_valid_idx = np.full(n_events, n_bars - 1, dtype=np.int64)
    else:
        last_valid_idx = np.asarray(last_valid_idx, dtype=np.int64)

    # (events, horizon) forward index matrix by broadcast -- the only 2-D
    # allocation in the engine.
    offsets = np.arange(max_horizon, dtype=np.int64)
    idx = entry_idx[:, None] + offsets[None, :]
    valid = idx <= np.minimum(last_valid_idx, n_bars - 1)[:, None]
    idx = np.minimum(idx, n_bars - 1)

    fwd_high = high[idx]
    fwd_low = low[idx]

    up = direction[:, None] > 0
    ref = reference_price[:, None]
    favorable = np.where(up, fwd_high - ref, ref - fwd_low)
    adverse = np.where(up, ref - fwd_low, fwd_high - ref)

    neg_inf = -np.inf
    mfe = np.max(np.where(valid, favorable, neg_inf), axis=1)
    mae = np.max(np.where(valid, adverse, neg_inf), axis=1)
    np.maximum(mfe, 0.0, out=mfe)
    np.maximum(mae, 0.0, out=mae)

    tgt = target_price[:, None]
    touched = np.where(up, fwd_high >= tgt, fwd_low <= tgt) & valid
    hit = touched.any(axis=1)
    bars_to_hit = np.where(hit, np.argmax(touched, axis=1), 0).astype(np.int64)

    horizon_bars = valid.sum(axis=1).astype(np.int64)
    return EventStudyResult(mfe, mae, hit, bars_to_hit, horizon_bars)
