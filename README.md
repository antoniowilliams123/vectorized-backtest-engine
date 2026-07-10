# vectorized-backtest-engine

A high-performance research engine for 1-minute bar archives, written to stay
inside numpy from disk to parquet. It is built for the scale of a serious
personal research archive — 15 years × 7 symbols of 1-minute data, i.e.
tens of millions of bars and billions of bar-metric computations per study —
and keeps full-archive studies in the **seconds** range by never touching a
Python-level loop over bars.

This repo is a portfolio piece. It ships with a seeded synthetic archive
generator, one deliberately public demo study, tests, and an honest
benchmark. **The demo study is a well-known descriptive statistic chosen to
exercise the engine; research built on this engine is not published.**

## Architecture

```
 data/{SYMBOL}_1m/{YYYY-MM}.csv.gz          monthly-partitioned archive
        |
        v
 vbtengine.io          ThreadPoolExecutor(24) partition reads
        |              -> dict of contiguous numpy arrays
        v              (ts int64 ns, open/high/low/close float64, volume int64)
 vbtengine.vector      convolve / slicing / searchsorted / accumulate / reduceat
        |
        v
 vbtengine.engine      generic forward-path event study (MFE, MAE, hit, timing)
        |
        v
 vbtengine.results     row-per-event parquet: {study}_{YYYY-MM-DD}.parquet
```

Studies fan symbols out across a `ProcessPoolExecutor` (CPU-bound numpy work
per symbol); inside each worker, monthly partitions load in parallel threads.

## Quick start

```bash
pip install -e ".[dev]"
python studies/gap_fill_demo.py           # generates data/ on first run
python benchmarks/bench_vector_vs_pandas.py
pytest
```

The demo generates 8 synthetic symbols × 3 years of 1-minute bars
(~2.4M bars, 43 MB gzipped), then answers, per (symbol, day): *after the
overnight gap, did price trade back to the prior close, and how fast?* On
this machine the full study — parallel load of 288 csv.gz partitions,
day bucketing, forward-path metrics for 6,240 events, parquet write, summary
table — completes in **~0.8 s wall clock** (~4 s including first-time data
generation).

```
Gap-fill frequency by |gap| decile (synthetic data -- demo only)
decile      |gap| range (bps)   events  fill rate  med mins
     0        0.0 -      6.9      624     98.6%         0
     ...
     9       89.5 -    208.1      624     49.4%       132

Overall: 79.0% of 6,240 gaps filled | 1.00 events per symbol-day
```

(Numbers describe the synthetic random walk, not any market.)

## Benchmark (this machine, 2M rows, best of 5)

`python benchmarks/bench_vector_vs_pandas.py` — numpy toolkit vs the naive
pandas equivalents:

| operation           | numpy    | pandas   | speedup |
|---------------------|----------|----------|---------|
| rolling mean (w=20) | 0.0135 s | 0.0153 s | 1.1x    |
| shift by 1 bar      | 0.0012 s | 0.0010 s | 0.9x    |
| per-day high        | 0.0003 s | 0.0551 s | **182x** |

An honest reading: pandas' rolling and shift kernels are well-optimized
Cython, so element-wise ops are near parity in isolation. The decisive wins
are (a) **group operations** — `np.maximum.reduceat` on sorted data crushes
`groupby(day).max()` because there is no key materialization and no hash
table, and (b) **composition** — a study chains dozens of these ops, and the
numpy path never pays Series construction, index alignment, or block-manager
copies between steps. That compounding is why full studies finish in
seconds.

## Design decisions

**Arrays-of-columns, not DataFrames, in hot paths.** The loader returns
`dict[str, np.ndarray]`. On sorted minute data, position *is* identity:
day membership is a binary search (`searchsorted`), per-day stats are
`reduceat` over integer offsets, and lags are slice assignments. A DataFrame
adds an index, alignment checks, and copies to every one of those steps
while providing nothing the sorted invariant doesn't already guarantee.
pandas appears in this repo only as the reference implementation in tests
and the comparison baseline in benchmarks.

**Threads for I/O, processes for compute.** Reading csv.gz partitions is
gzip + pyarrow parsing — C code that releases the GIL — so a
`ThreadPoolExecutor(max_workers=24)` gets true parallelism with zero
serialization cost. Per-symbol metric computation is CPU-bound numpy, so
symbols fan out across a `ProcessPoolExecutor(max_workers=24)`. Using
processes for I/O would pay spawn + pickle for nothing; using threads for
compute would serialize on the GIL between numpy calls.

**Granular parquet rows, never summary stats.** Every study writes one row
per event with *all* metric columns (`{study}_{YYYY-MM-DD}.parquet`). Summary
tables are printed, not persisted, because summaries are cheap to recompute
and expensive to be wrong about: the next question ("only the top decile",
"only events before noon", "metric A conditioned on metric B") becomes a
parquet filter instead of a full re-run.

**Synthetic data ships instead of market data.** Market data is licensed;
redistributing it is not an option. `vbtengine/synthetic.py` generates a
seeded geometric random walk with regime-switching volatility, overnight
jumps, and a U-shaped intraday volume profile — enough realism to exercise
every code path (gaps, volatility clusters, monthly partitioning) while
being reproducible bit-for-bit from a seed.

**Validation at the boundary, assumptions in the core.** `io.load_symbol`
verifies strictly-increasing timestamps once at load time. Every downstream
function then *assumes* sorted input, which is exactly what makes
`searchsorted`/`reduceat`/`accumulate` legal. One check, then zero defensive
overhead in the hot path.

## Repo layout

```
vbtengine/
  synthetic.py   seeded synthetic OHLCV archive generator
  io.py          parallel csv.gz -> numpy loader with validation
  vector.py      the vectorization toolkit (documented op-by-op)
  engine.py      generic forward-path event study
  results.py     granular parquet writer
studies/
  gap_fill_demo.py            the end-to-end demo study
benchmarks/
  bench_vector_vs_pandas.py   honest micro-benchmark
tests/                        pytest suite (31 tests)
```

## License

MIT © 2026 Tony Williams
