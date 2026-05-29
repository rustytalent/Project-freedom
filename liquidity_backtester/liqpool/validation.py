"""Statistical validation primitives for strategy/pocket selection.

Phase 4 keeps hitting the same wall: a pocket looks great in-sample, but once you
account for (a) how many configurations were tried before this one "won" and
(b) the small, fat-tailed, autocorrelated nature of trade returns, the edge is no
longer significant. This module centralises the honest tests that were previously
re-implemented ad hoc inside each analysis script:

  - Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR), after
    Bailey & López de Prado (2012/2014). DSR discounts the observed Sharpe by the
    Sharpe you would expect from the BEST of N random trials, so a pocket that only
    looks good because it was cherry-picked from a large sweep fails the test.
  - A bootstrap confidence interval on mean trade-R and P(mean R > 0), which is
    robust to the non-normality of per-trade returns.
  - Combinatorial Purged Cross-Validation (CPCV) group combinatorics plus a
    purge/embargo group assigner, so a caller can build many backtest paths from
    overlapping-label trades without leakage.

Everything here is a pure function of numbers/arrays — no model objects, no I/O —
so it is cheap to unit test and safe to call from any analysis or live-gate code.
All Gaussian CDF/PPF use the stdlib `statistics.NormalDist` (no SciPy dependency).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist
from typing import List, Optional, Sequence, Tuple

import numpy as np

_NORM = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Sharpe-based significance
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: Sequence[float], ddof: int = 1) -> float:
    """Per-observation Sharpe (mean / std). NOT annualised — callers annualise if
    they want to, but PSR/DSR below operate on this raw per-observation Sharpe."""
    arr = np.asarray(list(returns), dtype=float)
    if arr.size < 2:
        return 0.0
    sd = float(arr.std(ddof=ddof))
    if sd <= 0:
        return 0.0
    return float(arr.mean() / sd)


def _skew_kurtosis(returns: np.ndarray) -> Tuple[float, float]:
    """Sample skewness and Pearson kurtosis (normal == 3.0)."""
    n = returns.size
    if n < 3:
        return 0.0, 3.0
    mean = returns.mean()
    sd = returns.std(ddof=0)
    if sd <= 0:
        return 0.0, 3.0
    z = (returns - mean) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    return skew, kurt


def probabilistic_sharpe_ratio(observed_sr: float, n_obs: int,
                               skew: float = 0.0, kurtosis: float = 3.0,
                               sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark) given the observed Sharpe and return moments.

    `observed_sr`/`sr_benchmark` are per-observation Sharpe values. `kurtosis` is
    Pearson (normal == 3.0). Returns a probability in [0, 1]. Higher = more
    confident the edge is real. (Bailey & López de Prado, "The Sharpe Ratio
    Efficient Frontier".)
    """
    if n_obs < 2:
        return 0.0
    denom = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr ** 2
    if denom <= 0:
        # Degenerate higher-moment term; fall back to the Gaussian approximation.
        denom = 1.0
    z = (observed_sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)
    return float(_NORM.cdf(z))


def expected_max_sharpe(n_trials: int, variance_of_trial_sharpes: float) -> float:
    """Expected maximum of `n_trials` i.i.d. trial Sharpe estimates with the given
    cross-trial variance, under the false-strategy theorem. This is the benchmark
    Sharpe the deflated test must clear. Returns 0 for <= 1 trial."""
    if n_trials <= 1 or variance_of_trial_sharpes <= 0:
        return 0.0
    sigma = float(np.sqrt(variance_of_trial_sharpes))
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * np.e))
    return float(sigma * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(returns: Sequence[float], n_trials: int,
                          variance_of_trial_sharpes: Optional[float] = None
                          ) -> dict:
    """Deflated Sharpe Ratio for a returns series, discounting for `n_trials`
    configurations explored.

    If `variance_of_trial_sharpes` is None, it defaults to the sampling variance of
    a single Sharpe estimate, `(1 - skew*SR + (kurt-1)/4*SR^2) / (n-1)` — a
    reasonable proxy when the per-trial Sharpe spread is unknown.

    Returns a dict with the observed Sharpe, the deflation benchmark, and `dsr`
    (the probability the edge survives selection). A pocket is "DSR-confirmed" when
    `dsr` clears a chosen threshold (e.g. 0.95).
    """
    arr = np.asarray(list(returns), dtype=float)
    n = arr.size
    sr = sharpe_ratio(arr)
    skew, kurt = _skew_kurtosis(arr)
    if variance_of_trial_sharpes is None:
        denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
        variance_of_trial_sharpes = max(denom, 1e-9) / max(n - 1, 1)
    sr0 = expected_max_sharpe(n_trials, variance_of_trial_sharpes)
    dsr = probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=sr0)
    return {
        "n": int(n),
        "observed_sharpe": float(sr),
        "skew": float(skew),
        "kurtosis": float(kurt),
        "n_trials": int(n_trials),
        "deflation_benchmark_sharpe": float(sr0),
        "dsr": float(dsr),
    }


# ---------------------------------------------------------------------------
# Bootstrap on mean trade-R
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    n: int
    mean: float
    ci_low: float
    ci_high: float
    prob_positive: float


def bootstrap_mean_ci(returns: Sequence[float], ci: float = 0.95,
                      n_boot: int = 5000, seed: int = 0) -> BootstrapResult:
    """Percentile bootstrap CI for the mean of `returns`, plus the bootstrap
    probability that the true mean is > 0. Robust to fat tails / non-normality,
    which per-trade R distributions always have."""
    arr = np.asarray(list(returns), dtype=float)
    if arr.size == 0:
        return BootstrapResult(0, 0.0, 0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = float(np.percentile(means, (1.0 - ci) / 2.0 * 100.0))
    hi = float(np.percentile(means, (1.0 + ci) / 2.0 * 100.0))
    return BootstrapResult(
        n=int(arr.size),
        mean=float(arr.mean()),
        ci_low=lo,
        ci_high=hi,
        prob_positive=float((means > 0).mean()),
    )


# ---------------------------------------------------------------------------
# Combinatorial Purged Cross-Validation (CPCV) scaffolding
# ---------------------------------------------------------------------------

def cpcv_test_group_sets(n_groups: int, k_test_groups: int) -> List[Tuple[int, ...]]:
    """Every choice of `k_test_groups` test groups out of `n_groups`. Each set is one
    CPCV split; the remaining groups (minus purge/embargo) form its training set."""
    if not (1 <= k_test_groups < n_groups):
        raise ValueError("require 1 <= k_test_groups < n_groups")
    return list(combinations(range(n_groups), k_test_groups))


def n_cpcv_backtest_paths(n_groups: int, k_test_groups: int) -> int:
    """Number of distinct backtest paths CPCV can reconstruct.
    = C(n_groups, k) * k / n_groups  (each group appears in that many test sets)."""
    from math import comb
    return comb(n_groups, k_test_groups) * k_test_groups // n_groups


def assign_time_groups(start_times: Sequence, n_groups: int) -> np.ndarray:
    """Assign each event to one of `n_groups` contiguous, time-ordered groups of
    (near) equal size. Returns an int array of group ids aligned to input order."""
    starts = np.asarray([np.datetime64(t) for t in start_times])
    n = starts.size
    if n == 0:
        return np.empty(0, dtype=int)
    order = np.argsort(starts, kind="mergesort")
    group_of_sorted = np.minimum((np.arange(n) * n_groups) // n, n_groups - 1)
    groups = np.empty(n, dtype=int)
    groups[order] = group_of_sorted
    return groups


def purged_train_mask(groups: np.ndarray,
                      test_groups: Sequence[int],
                      start_times: Sequence,
                      end_times: Sequence,
                      embargo_td: "np.timedelta64 | None" = None) -> np.ndarray:
    """Boolean mask selecting TRAIN events for a CPCV split with the given test groups.

    An event is excluded from training if it is in a test group, if its label window
    [start, end] overlaps any test group's time span (purge), or if it starts within
    `embargo_td` after a test group's end (embargo). This is the leakage-safe training
    set for one combinatorial split.
    """
    groups = np.asarray(groups)
    starts = np.asarray([np.datetime64(t) for t in start_times])
    ends = np.asarray([np.datetime64(t) for t in end_times])
    test_set = set(int(g) for g in test_groups)
    in_test = np.array([g in test_set for g in groups])

    # Time spans of the test groups (for purge/embargo against all other events).
    overlaps = np.zeros(len(groups), dtype=bool)
    embargoed = np.zeros(len(groups), dtype=bool)
    for g in test_set:
        sel = groups == g
        if not sel.any():
            continue
        t_start = starts[sel].min()
        t_end = ends[sel].max()
        overlaps |= (starts <= t_end) & (ends >= t_start)
        if embargo_td is not None:
            embargoed |= (starts > t_end) & (starts <= t_end + embargo_td)

    return ~(in_test | overlaps | embargoed)
