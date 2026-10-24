"""Summary statistics for sensor time series."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def cv_percent(values) -> float:
    """Coefficient of variation: 100 * sample std / |mean|. NaN if mean ~ 0."""
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(s) < 2 or abs(s.mean()) < 1e-12:
        return float("nan")
    return float(100 * s.std(ddof=1) / abs(s.mean()))


def descriptive(values) -> dict:
    """Descriptive statistics without silently imputing missing values."""
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if s.empty:
        # n is 0, not NaN: NaN is truthy in Python and would pass `if n:` checks.
        return {"n": 0, **{k: float("nan") for k in ["mean", "std", "median", "iqr", "cv_pct"]}}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) >= 2 else float("nan"),
        "median": float(s.median()),
        "iqr": float(s.quantile(.75) - s.quantile(.25)),
        "cv_pct": cv_percent(s),
    }


def relative_difference(new: float, baseline: float) -> float:
    """Percentage difference of `new` relative to `baseline`; NaN if baseline ~ 0."""
    if not np.isfinite(baseline) or not np.isfinite(new) or abs(baseline) < 1e-12:
        return float("nan")
    return float(100 * (new - baseline) / abs(baseline))


def _moving_block_resample(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Resample a series by stitching random blocks of consecutive values.

    Keeping neighbours together preserves short-range autocorrelation, which an
    ordinary (i.i.d.) bootstrap would destroy, giving intervals that are too narrow.
    """
    n = len(x)
    if n <= block:
        return rng.choice(x, size=n, replace=True)
    starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return x[idx]


def block_bootstrap_diff_ci(baseline, other, block: int = 6, n_boot: int = 2000,
                            level: float = .95, seed: int = 0) -> tuple[float, float]:
    """Confidence interval for mean(other) - mean(baseline) via moving-block bootstrap."""
    a = pd.to_numeric(pd.Series(baseline), errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(pd.Series(other), errors="coerce").dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.array([
        _moving_block_resample(b, block, rng).mean() - _moving_block_resample(a, block, rng).mean()
        for _ in range(n_boot)
    ])
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(diffs, [tail, 100 - tail])
    return float(lo), float(hi)
