"""The numpy vectorization toolkit.

Every function here replaces a common pandas idiom with a raw-array
equivalent, and each docstring says *why* the array version wins. The short
version: pandas operations drag an index, a block manager, and (for
``groupby``) hash tables through the hot path. On sorted minute data none of
that machinery is needed -- position *is* identity, and group boundaries are
just integer offsets.

Conventions
-----------
- All inputs are 1-D contiguous arrays over bars, sorted by timestamp.
- "Groups" (days, sessions, any bucket) are described by ``starts``: the
  integer index of each group's first bar. Contiguity is assumed -- group
  ``i`` spans ``starts[i] : starts[i+1]``.
"""

from __future__ import annotations

import numpy as np

NS_PER_DAY = 86_400_000_000_000


def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean via ``np.convolve``; NaN for the warm-up prefix.

    Why not ``pd.Series(a).rolling(window).mean()``: pandas' rolling engine
    is well-optimized Cython, so raw speed is comparable at short windows
    (see ``benchmarks/`` -- this repo does not pretend otherwise). The win
    here is compositional: the input and output are bare float64 buffers,
    so this call chains with slicing/reduceat pipelines without ever
    constructing a Series, an index, or paying alignment on every step.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    a = np.asarray(a, dtype=np.float64)
    if window > a.size:
        return np.full(a.size, np.nan)
    out = np.full(a.size, np.nan)
    kernel = np.full(window, 1.0 / window)
    out[window - 1 :] = np.convolve(a, kernel, mode="valid")
    return out


def shift(a: np.ndarray, periods: int, fill_value: float = np.nan) -> np.ndarray:
    """Shift by slicing: ``out[w:] = a[:-w]`` (positive = look back).

    Why not ``pd.Series(a).shift(periods)``: identical behavior, and both are a
    memcpy at heart, but this version touches each output byte exactly once
    (``np.empty`` + two slice writes, no full-array pre-fill) and neither
    builds nor returns index machinery -- it stays a bare buffer for the
    next array op.
    """
    a = np.asarray(a)
    w = abs(periods)
    if w == 0:
        return a.astype(np.float64, copy=True)
    out = np.empty(a.shape, dtype=np.float64)
    if w >= a.size:
        out[:] = fill_value
        return out
    if periods > 0:
        out[:w] = fill_value
        out[w:] = a[:-w]
    else:
        out[-w:] = fill_value
        out[:-w] = a[w:]
    return out


def day_start_timestamps(ts: np.ndarray) -> np.ndarray:
    """Unique UTC-day-start timestamps (epoch ns) present in ``ts``."""
    days = np.unique(ts // NS_PER_DAY)
    return days * NS_PER_DAY


def bucket_ids(ts: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Map each timestamp to a bucket via ``np.searchsorted``.

    ``boundaries`` are sorted bucket-start timestamps; a bar landing exactly
    on a boundary belongs to the bucket that starts there (``side='right'``
    then minus one). Bars before the first boundary get id ``-1``.

    Why not ``df.groupby(df.ts.dt.date)``: converting int64 nanoseconds to
    Python ``date`` objects allocates millions of PyObjects, then groupby
    hashes them. On *sorted* timestamps, bucket membership is a binary
    search: O(n log m), zero objects, and the result is a plain int64 array
    that downstream reduceat calls can use directly.
    """
    boundaries = np.asarray(boundaries, dtype=np.int64)
    return np.searchsorted(boundaries, np.asarray(ts, dtype=np.int64), side="right") - 1


def group_starts(ids: np.ndarray) -> np.ndarray:
    """First-bar index of each contiguous id run (for reduceat/accumulate).

    ``ids`` must be non-decreasing (guaranteed on sorted timestamps).
    """
    ids = np.asarray(ids)
    if ids.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(([0], np.flatnonzero(np.diff(ids)) + 1)).astype(np.int64)


def group_ends(starts: np.ndarray, n: int) -> np.ndarray:
    """Last-bar index of each group given its ``starts`` and total length."""
    starts = np.asarray(starts, dtype=np.int64)
    return np.append(starts[1:], n) - 1


def grouped_cummax(a: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-group running maximum with resets at group boundaries.

    The trick: add a per-group offset large enough that every value in group
    ``k`` dominates every value in groups ``< k``. A single global
    ``np.maximum.accumulate`` then can never carry a maximum across a
    boundary; subtracting the offset restores the values.

    Why not ``df.groupby(g).cummax()``: pandas runs the accumulation
    group-by-group through its groupby machinery. This is two vectorized
    passes over one float64 buffer regardless of how many groups exist.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return a.copy()
    gid = np.zeros(a.size, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    gid[starts[starts > 0]] = 1
    gid = np.cumsum(gid)
    span = float(np.max(a) - np.min(a)) + 1.0
    offset = gid * span
    return np.maximum.accumulate(a + offset) - offset


def grouped_cummin(a: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-group running minimum: ``grouped_cummax`` on the negated array."""
    return -grouped_cummax(-np.asarray(a, dtype=np.float64), starts)


def group_max(a: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-group max via ``np.maximum.reduceat``.

    Why not ``df.groupby(g)['x'].max()``: reduceat reduces each contiguous
    slice in one C call with no hash table, no key materialization, and no
    result index -- just ``len(starts)`` floats out.
    """
    return np.maximum.reduceat(np.asarray(a), np.asarray(starts, dtype=np.int64))


def group_min(a: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-group min via ``np.minimum.reduceat`` (see ``group_max``)."""
    return np.minimum.reduceat(np.asarray(a), np.asarray(starts, dtype=np.int64))


def group_first(a: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-group first value: fancy indexing, one gather."""
    return np.asarray(a)[np.asarray(starts, dtype=np.int64)]


def group_last(a: np.ndarray, starts: np.ndarray, n: int | None = None) -> np.ndarray:
    """Per-group last value: fancy indexing on the group end offsets."""
    a = np.asarray(a)
    return a[group_ends(starts, a.size if n is None else n)]


def group_sum(a: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-group sum via ``np.add.reduceat`` (see ``group_max``)."""
    return np.add.reduceat(np.asarray(a), np.asarray(starts, dtype=np.int64))
