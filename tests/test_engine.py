"""Forward-path event study on tiny handcrafted fixtures with known answers."""

from __future__ import annotations

import numpy as np

from vbtengine.engine import forward_event_study


def _series() -> tuple[np.ndarray, np.ndarray]:
    #        idx:    0      1      2      3      4      5
    high = np.array([101.0, 102.0, 104.0, 103.0, 106.0, 110.0])
    low = np.array([99.0, 100.0, 101.0, 98.0, 103.0, 108.0])
    return high, low


def test_upward_target_hit_and_timing() -> None:
    high, low = _series()
    res = forward_event_study(
        high=high,
        low=low,
        entry_idx=np.array([0]),
        reference_price=np.array([100.0]),
        target_price=np.array([103.5]),
        direction=np.array([1]),
        max_horizon=6,
    )
    assert res.hit[0]
    assert res.bars_to_hit[0] == 2  # high first reaches 103.5 at idx 2
    assert res.mfe[0] == 10.0  # 110 - 100
    assert res.mae[0] == 2.0  # 100 - 98
    assert res.horizon_bars[0] == 6


def test_downward_target_never_hit() -> None:
    high, low = _series()
    res = forward_event_study(
        high=high,
        low=low,
        entry_idx=np.array([0]),
        reference_price=np.array([100.0]),
        target_price=np.array([97.0]),
        direction=np.array([-1]),
        max_horizon=6,
    )
    assert not res.hit[0]
    assert res.bars_to_hit[0] == 0
    assert res.mfe[0] == 2.0  # best downward excursion: 100 - 98
    assert res.mae[0] == 10.0  # worst: 110 - 100


def test_last_valid_idx_masks_later_bars() -> None:
    high, low = _series()
    res = forward_event_study(
        high=high,
        low=low,
        entry_idx=np.array([0]),
        reference_price=np.array([100.0]),
        target_price=np.array([105.0]),
        direction=np.array([1]),
        max_horizon=6,
        last_valid_idx=np.array([3]),  # day ends at idx 3; 106/110 unseen
    )
    assert not res.hit[0]
    assert res.mfe[0] == 4.0  # capped at idx-2 high of 104
    assert res.horizon_bars[0] == 4


def test_batch_events_with_mixed_directions() -> None:
    high, low = _series()
    res = forward_event_study(
        high=high,
        low=low,
        entry_idx=np.array([1, 3]),
        reference_price=np.array([100.0, 100.0]),
        target_price=np.array([103.9, 99.0]),
        direction=np.array([1, -1]),
        max_horizon=3,
    )
    # Event 0 sees idx 1..3: max high 104 >= 103.9 at offset 1.
    assert res.hit[0] and res.bars_to_hit[0] == 1
    # Event 1 sees idx 3..5: min low 98 <= 99 at offset 0 (its entry bar).
    assert res.hit[1] and res.bars_to_hit[1] == 0


def test_horizon_clipped_at_series_end() -> None:
    high, low = _series()
    res = forward_event_study(
        high=high,
        low=low,
        entry_idx=np.array([4]),
        reference_price=np.array([104.0]),
        target_price=np.array([120.0]),
        direction=np.array([1]),
        max_horizon=10,
    )
    assert res.horizon_bars[0] == 2  # only idx 4 and 5 exist
    assert not res.hit[0]


def test_empty_event_batch() -> None:
    high, low = _series()
    res = forward_event_study(
        high=high,
        low=low,
        entry_idx=np.array([], dtype=np.int64),
        reference_price=np.array([]),
        target_price=np.array([]),
        direction=np.array([], dtype=np.int64),
        max_horizon=5,
    )
    assert res.mfe.size == 0 and res.hit.size == 0
