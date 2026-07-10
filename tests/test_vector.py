"""Correctness of the numpy toolkit against pandas reference implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vbtengine import vector

RNG = np.random.default_rng(3)


class TestRollingMean:
    def test_matches_pandas_reference(self) -> None:
        a = RNG.standard_normal(5_000).cumsum() + 100.0
        for window in (1, 2, 5, 20, 200):
            got = vector.rolling_mean(a, window)
            want = pd.Series(a).rolling(window).mean().to_numpy()
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9, equal_nan=True)

    def test_warmup_prefix_is_nan(self) -> None:
        out = vector.rolling_mean(np.arange(10.0), 4)
        assert np.isnan(out[:3]).all()
        assert not np.isnan(out[3:]).any()

    def test_window_larger_than_series(self) -> None:
        assert np.isnan(vector.rolling_mean(np.arange(3.0), 5)).all()

    def test_rejects_bad_window(self) -> None:
        with pytest.raises(ValueError):
            vector.rolling_mean(np.arange(3.0), 0)


class TestShift:
    def test_matches_pandas_reference(self) -> None:
        a = RNG.standard_normal(1_000)
        for periods in (-3, -1, 0, 1, 5):
            got = vector.shift(a, periods)
            want = pd.Series(a).shift(periods).to_numpy()
            np.testing.assert_allclose(got, want, equal_nan=True)

    def test_alignment(self) -> None:
        a = np.array([10.0, 20.0, 30.0, 40.0])
        out = vector.shift(a, 2)
        assert np.isnan(out[:2]).all()
        np.testing.assert_array_equal(out[2:], [10.0, 20.0])

    def test_shift_beyond_length_is_all_fill(self) -> None:
        assert np.isnan(vector.shift(np.arange(3.0), 4)).all()


class TestDayBucketing:
    def test_midnight_boundary_bar_starts_new_day(self) -> None:
        ns = vector.NS_PER_DAY
        minute = 60_000_000_000
        # 23:58, 23:59, exactly 00:00 of day 1, 00:01
        ts = np.array([ns - 2 * minute, ns - minute, ns, ns + minute], dtype=np.int64)
        boundaries = np.array([0, ns, 2 * ns], dtype=np.int64)
        ids = vector.bucket_ids(ts, boundaries)
        np.testing.assert_array_equal(ids, [0, 0, 1, 1])

    def test_bar_before_first_boundary_gets_minus_one(self) -> None:
        ids = vector.bucket_ids(np.array([5]), np.array([10, 20]))
        np.testing.assert_array_equal(ids, [-1])

    def test_group_starts_and_ends(self) -> None:
        ids = np.array([0, 0, 0, 1, 1, 4, 4, 4, 4])
        starts = vector.group_starts(ids)
        np.testing.assert_array_equal(starts, [0, 3, 5])
        np.testing.assert_array_equal(vector.group_ends(starts, ids.size), [2, 4, 8])


class TestGroupedAccumulate:
    def test_cummax_resets_at_group_starts(self) -> None:
        a = np.array([3.0, 5.0, 4.0, 1.0, 0.5, 2.0, 10.0, 9.0])
        starts = np.array([0, 3, 6])
        got = vector.grouped_cummax(a, starts)
        want = np.array([3.0, 5.0, 5.0, 1.0, 1.0, 2.0, 10.0, 10.0])
        np.testing.assert_allclose(got, want)

    def test_cummin_resets_at_group_starts(self) -> None:
        a = np.array([3.0, 1.0, 4.0, 9.0, 5.0, 7.0])
        starts = np.array([0, 3])
        want = np.array([3.0, 1.0, 1.0, 9.0, 5.0, 5.0])
        np.testing.assert_allclose(vector.grouped_cummin(a, starts), want)

    def test_matches_pandas_groupby_cummax(self) -> None:
        a = RNG.standard_normal(2_000).cumsum()
        gid = np.sort(RNG.integers(0, 40, size=2_000))
        starts = vector.group_starts(gid)
        want = pd.Series(a).groupby(gid).cummax().to_numpy()
        np.testing.assert_allclose(vector.grouped_cummax(a, starts), want)


class TestGroupReductions:
    def test_reduceat_matches_pandas_groupby(self) -> None:
        n = 10_000
        a = RNG.standard_normal(n).cumsum() + 50.0
        gid = np.sort(RNG.integers(0, 250, size=n))
        starts = vector.group_starts(gid)
        grouped = pd.Series(a).groupby(gid)

        np.testing.assert_allclose(vector.group_max(a, starts), grouped.max().to_numpy())
        np.testing.assert_allclose(vector.group_min(a, starts), grouped.min().to_numpy())
        np.testing.assert_allclose(vector.group_first(a, starts), grouped.first().to_numpy())
        np.testing.assert_allclose(vector.group_last(a, starts), grouped.last().to_numpy())
        np.testing.assert_allclose(vector.group_sum(a, starts), grouped.sum().to_numpy())

    def test_single_bar_groups(self) -> None:
        a = np.array([7.0, 3.0, 9.0])
        starts = np.array([0, 1, 2])
        np.testing.assert_allclose(vector.group_max(a, starts), a)
        np.testing.assert_allclose(vector.group_min(a, starts), a)
