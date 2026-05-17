"""Confidence-interval helpers for binomial proportions.

We use these to put error bars on respect rates: a 56% rate means very different things at n=40
vs n=400. Wilson's score interval is exact-ish for binomials and behaves well with small n;
bootstrap is distribution-free and lets us double-check the analytical CI."""
from __future__ import annotations
from typing import Iterable, Tuple
import math
import numpy as np


_Z_TABLE = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def wilson_score_interval(k: int, n: int, ci: float = 0.90) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    Returns (low, high) such that the true rate lies in [low, high] with probability `ci`.
    Wilson is preferred over the textbook normal approximation because it stays inside [0, 1]
    and behaves sensibly when k=0 or k=n.
    """
    if n == 0:
        return (0.0, 0.0)
    z = _Z_TABLE.get(ci, 1.645)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_proportion_ci(outcomes: Iterable[int], ci: float = 0.90,
                            n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """Bootstrap percentile CI for a proportion. `outcomes` is iterable of 0/1.

    Useful as a sanity check on the Wilson interval: if they disagree wildly the sample is
    probably weird (e.g. dominated by a few high-leverage outcomes)."""
    arr = np.asarray(list(outcomes), dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = float(np.percentile(samples, (1.0 - ci) / 2.0 * 100.0))
    hi = float(np.percentile(samples, (1.0 + ci) / 2.0 * 100.0))
    return (lo, hi)
