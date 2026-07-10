"""Seeded synthetic 1-minute OHLCV archive generator.

Produces a realistic-looking but entirely artificial price archive so the
engine can be demonstrated, benchmarked, and tested without shipping any
licensed market data.

Model
-----
- Geometric random walk on 1-minute log returns.
- Regime-switching volatility: a two-state Markov chain (calm / stressed)
  scales the per-minute return sigma, producing the volatility clustering
  real intraday data exhibits.
- Overnight jump: an extra log-return is applied between the last bar of one
  day and the first bar of the next, so day opens gap away from prior closes.
- U-shaped intraday volume profile: heavy at the session open and close,
  light midday, with lognormal noise -- the classic equity intraday shape.

Archive layout
--------------
``{root}/{SYMBOL}_1m/{YYYY-MM}.csv.gz`` with columns
``ts_utc,open,high,low,close,volume`` where ``ts_utc`` is epoch nanoseconds
(UTC). One 390-minute session per weekday. This monthly-partitioned layout
mirrors a common archive convention for minute data.

Everything here is fully vectorized -- one numpy pass per symbol, no Python
loop over bars.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

NS_PER_MINUTE = 60_000_000_000
NS_PER_DAY = 86_400_000_000_000

#: Minutes per synthetic session (equity-style 6.5 hour day).
BARS_PER_DAY = 390

#: Session start offset from UTC midnight, in minutes (14:30 UTC).
SESSION_START_MINUTE = 14 * 60 + 30

DEFAULT_SYMBOLS = ("ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL")

CSV_HEADER = "ts_utc,open,high,low,close,volume"


def _weekday_session_starts(start_year: int, years: int) -> np.ndarray:
    """Epoch-ns timestamps of every weekday session start in the span.

    Uses numpy datetime64 arithmetic plus weekday filtering rather than any
    per-day Python loop.
    """
    d0 = np.datetime64(f"{start_year}-01-01", "D")
    d1 = np.datetime64(f"{start_year + years}-01-01", "D")
    days = np.arange(d0, d1)
    # Monday=0 .. Sunday=6; keep Mon-Fri.
    weekday = (days.astype("int64") + 3) % 7  # 1970-01-01 was a Thursday
    days = days[weekday < 5]
    day_ns = days.astype("datetime64[ns]").astype("int64")
    return day_ns + SESSION_START_MINUTE * NS_PER_MINUTE


def _regime_sigma(
    rng: np.random.Generator, n: int, calm: float, stressed: float, p_switch: float
) -> np.ndarray:
    """Per-bar sigma from a two-state Markov chain, vectorized.

    A state flips with probability ``p_switch`` each bar. The state path is
    the cumulative XOR of the flip draws -- computed as ``cumsum % 2``, a
    single vectorized pass instead of a stateful bar loop.
    """
    flips = rng.random(n) < p_switch
    state = np.cumsum(flips) % 2  # 0 = calm, 1 = stressed
    return np.where(state == 0, calm, stressed)


def _u_shaped_volume(rng: np.random.Generator, n_days: int, base: float) -> np.ndarray:
    """U-shaped intraday volume: quadratic in normalized session time.

    ``w(t) = 1 + 3*(2t-1)^2`` for t in [0, 1] gives ~4x weight at the open
    and close versus midday. Lognormal noise keeps it from looking sterile.
    """
    t = (np.arange(BARS_PER_DAY) + 0.5) / BARS_PER_DAY
    profile = 1.0 + 3.0 * (2.0 * t - 1.0) ** 2
    noise = rng.lognormal(mean=0.0, sigma=0.35, size=n_days * BARS_PER_DAY)
    vol = base * np.tile(profile, n_days) * noise
    return np.maximum(vol.astype("int64"), 1)


def generate_symbol(
    symbol: str,
    start_year: int = 2021,
    years: int = 3,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Generate one symbol's full history as a dict of contiguous arrays.

    Returns ``{"ts": int64 epoch-ns, "open"/"high"/"low"/"close": float64,
    "volume": int64}``, sorted by timestamp.
    """
    # Per-symbol deterministic stream: same (symbol, seed) -> same data.
    rng = np.random.default_rng([seed, *symbol.encode()])

    day_starts = _weekday_session_starts(start_year, years)
    n_days = day_starts.size
    n = n_days * BARS_PER_DAY

    ts = (day_starts[:, None] + np.arange(BARS_PER_DAY) * NS_PER_MINUTE).ravel()

    sigma = _regime_sigma(rng, n, calm=0.0004, stressed=0.0013, p_switch=0.004)
    rets = rng.standard_normal(n) * sigma

    # Overnight jump added to each day's first bar (except the very first).
    overnight = rng.standard_normal(n_days) * 0.006
    overnight[0] = 0.0
    first_bar_idx = np.arange(n_days) * BARS_PER_DAY
    rets[first_bar_idx] += overnight

    base_price = float(rng.uniform(40.0, 400.0))
    close = base_price * np.exp(np.cumsum(rets))

    open_ = np.empty_like(close)
    open_[1:] = close[:-1]
    open_[0] = base_price
    # The first bar of each day opens at prior close * overnight jump so the
    # gap shows up in the open, then the bar closes on the walk.
    open_[first_bar_idx[1:]] = close[first_bar_idx[1:] - 1] * np.exp(
        overnight[1:] * 0.9
    )

    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick = np.abs(rng.standard_normal(n)) * sigma * 0.7
    high = body_hi * np.exp(wick)
    low = body_lo * np.exp(-np.abs(rng.standard_normal(n)) * sigma * 0.7)

    volume = _u_shaped_volume(rng, n_days, base=2_500.0)

    return {
        "ts": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _month_keys(ts: np.ndarray) -> np.ndarray:
    """``YYYY-MM`` string key per bar, derived with datetime64 casts."""
    return ts.astype("datetime64[ns]").astype("datetime64[M]")


def write_symbol_archive(root: Path, symbol: str, data: dict[str, np.ndarray]) -> int:
    """Write one symbol as monthly ``{YYYY-MM}.csv.gz`` partitions.

    Returns the number of partition files written.
    """
    sym_dir = root / f"{symbol}_1m"
    sym_dir.mkdir(parents=True, exist_ok=True)

    months = _month_keys(data["ts"])
    unique_months, starts = np.unique(months, return_index=True)
    bounds = np.append(starts, months.size)

    for m, lo, hi in zip(unique_months, bounds[:-1], bounds[1:], strict=True):
        table = pa.table(
            {
                "ts_utc": data["ts"][lo:hi],
                "open": np.round(data["open"][lo:hi], 4),
                "high": np.round(data["high"][lo:hi], 4),
                "low": np.round(data["low"][lo:hi], 4),
                "close": np.round(data["close"][lo:hi], 4),
                "volume": data["volume"][lo:hi],
            }
        )
        path = sym_dir / f"{m}.csv.gz"
        with gzip.open(path, "wb", compresslevel=5) as fh:
            pacsv.write_csv(table, fh)
    return int(unique_months.size)


def ensure_archive(
    root: Path,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    start_year: int = 2021,
    years: int = 3,
    seed: int = 7,
) -> Path:
    """Generate the synthetic archive under ``root`` if not already present.

    Idempotent: a symbol directory that already contains partitions is
    skipped, so repeated demo runs cost nothing.
    """
    root = Path(root)
    for symbol in symbols:
        sym_dir = root / f"{symbol}_1m"
        if sym_dir.exists() and any(sym_dir.glob("*.csv.gz")):
            continue
        data = generate_symbol(symbol, start_year=start_year, years=years, seed=seed)
        n_parts = write_symbol_archive(root, symbol, data)
        print(f"[synthetic] {symbol}: {data['ts'].size:,} bars -> {n_parts} monthly partitions")
    return root


if __name__ == "__main__":
    ensure_archive(Path(__file__).resolve().parents[1] / "data")
