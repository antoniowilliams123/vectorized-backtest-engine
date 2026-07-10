"""vbtengine: a numpy-vectorized event-study engine for 1-minute bar archives.

Layers:

- :mod:`vbtengine.synthetic` -- seeded synthetic OHLCV archive generator
- :mod:`vbtengine.io`        -- parallel archive reader (csv.gz -> numpy arrays)
- :mod:`vbtengine.vector`    -- the vectorization toolkit (convolve, slicing,
  searchsorted, accumulate, reduceat)
- :mod:`vbtengine.engine`    -- generic forward-path event study
- :mod:`vbtengine.results`   -- granular parquet result writer
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
