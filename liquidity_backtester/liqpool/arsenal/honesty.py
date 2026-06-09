"""Statistical honesty utilities for the arsenal evaluator.

Three independent corrections to the headline numbers the arsenal
publishes today, each addressing a known source of overstated edge:

  1. Holm-Bonferroni step-down — controls the family-wise error rate
     across an alpha registry tested in one session. Strictly more
     powerful than plain Bonferroni at the same error guarantee.

  2. Probabilistic Sharpe Ratio (PSR) — Bailey & Lopez de Prado, 2012.
     Tests whether a strategy's observed Sharpe is significantly
     greater than a benchmark (default zero), accounting for skewness
     and kurtosis of the strategy's returns.

  3. Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado, 2014.
     Adjusts PSR for multiple testing by feeding in the *number* of
     trials and the *variance* of trial Sharpes. The DSR is the
     headline number that survives the search.

All three operate on plain numpy arrays of per-trade R or per-period
returns. No model assumptions beyond what the underlying papers
require (i.i.d.-ish returns for PSR/DSR; nothing for Holm).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------

@dataclass
class HolmResult:
    """Per-test verdict from a Holm-Bonferroni step-down."""
    name: str
    raw_p: float
    adjusted_threshold: float
    significant: bool


def holm_bonferroni(
    p_values: Sequence[Tuple[str, float]],
    alpha: float = 0.05,
) -> List[HolmResult]:
    """Step-down family-wise-error-rate control across `len(p_values)`
    hypotheses tested at overall confidence ``1 - alpha``.

    Algorithm:
      * Sort tests by p-value ascending.
      * Compare p_(k) against alpha / (N - k + 1) where k is the rank
        (1-indexed).
      * The first time a comparison fails, that test and ALL
        subsequent (larger-p) tests are non-significant.

    Returns one HolmResult per input tuple, **in the input order**,
    with the per-test ``adjusted_threshold`` the test was compared
    against (handy for the renderer to display the bar each alpha had
    to clear).

    Plain Bonferroni's behaviour is the special case where every test
    uses ``alpha / N`` — Holm is strictly more powerful (no test ever
    gets a tighter threshold).
    """
    if not p_values:
        return []
    n = len(p_values)
    # Sort by p ascending, keep original positions.
    indexed = sorted(
        enumerate(p_values), key=lambda kv: kv[1][1]
    )
    results: List[Optional[HolmResult]] = [None] * n
    rejected_chain = True  # we keep accepting until first failure
    for sorted_rank, (orig_idx, (name, p)) in enumerate(indexed, start=1):
        thresh = alpha / max(1, (n - sorted_rank + 1))
        sig = rejected_chain and (p <= thresh)
        if not sig:
            rejected_chain = False
        results[orig_idx] = HolmResult(
            name=name,
            raw_p=float(p),
            adjusted_threshold=float(thresh),
            significant=bool(sig),
        )
    # Drop the Optionals.
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Probabilistic / Deflated Sharpe
# ---------------------------------------------------------------------------

def _sharpe_stats(returns: np.ndarray) -> Tuple[float, float, float, int]:
    """Compute (sharpe, skew, kurtosis_excess, n) on a sample of
    per-period returns. Sharpe is in the same units as the returns;
    if returns are per-trade R, the Sharpe is per-trade — caller can
    annualise.

    Skew and kurtosis use the unbiased Fisher convention (kurtosis of
    a normal distribution = 0)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    # numpy's std on a "constant" array isn't exactly zero due to
    # floating-point cancellation (typically ~1e-17). Treat any value
    # below a sane absolute-scale floor as zero-variance — a strategy
    # with such tiny dispersion has no defined Sharpe.
    if not (std > 1e-12):
        return float("nan"), 0.0, 0.0, n
    sharpe = mean / std
    # Sample skewness (Fisher-Pearson, bias-corrected isn't necessary
    # for PSR's purpose — the original paper uses the simple sample
    # skewness g1).
    z = (r - mean) / std
    skew = float((z ** 3).mean())
    # Excess kurtosis (g2): population kurtosis minus 3.
    kurt = float((z ** 4).mean() - 3.0)
    return sharpe, skew, kurt, n


def _norm_cdf(x: float) -> float:
    """Standard normal CDF without scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def probabilistic_sharpe(
    returns: np.ndarray,
    sharpe_benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012).

    Returns the probability that the strategy's TRUE Sharpe is
    greater than ``sharpe_benchmark``, given the observed Sharpe and
    the third/fourth moments of returns. Range [0, 1]; higher is
    more confident in real edge.

    Formula::

        z = (sharpe_hat - sharpe_benchmark) * sqrt(n - 1)
            / sqrt(1 - skew * sharpe_hat + ((kurt) / 4) * sharpe_hat^2)
        PSR = Phi(z)

    where `kurt` is excess kurtosis. Reduces to the standard Sharpe
    significance test when skew = 0 and kurt = 0 (i.i.d. normal).
    """
    sharpe, skew, kurt, n = _sharpe_stats(returns)
    if not math.isfinite(sharpe) or n < 3:
        return float("nan")
    denom_sq = 1.0 - skew * sharpe + (kurt / 4.0) * sharpe * sharpe
    # Guard against the corrective denominator going non-positive
    # (extreme distributions). Fall back to the no-moment-correction
    # variance of 1.0.
    denom_sq = max(denom_sq, 1e-9)
    z = (sharpe - sharpe_benchmark) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return _norm_cdf(z)


# Euler-Mascheroni and other constants Bailey uses.
_EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(
    trials_sharpe_std: float,
    n_trials: int,
) -> float:
    """Expected maximum Sharpe under the null (no edge) after
    ``n_trials`` independent searches whose individual Sharpes have
    standard deviation ``trials_sharpe_std``.

    Bailey & LdP (2014) eq. 4. The closed form uses the Gumbel
    distribution's expected maximum order statistic.
    """
    if n_trials < 1 or trials_sharpe_std <= 0:
        return 0.0
    if n_trials == 1:
        return 0.0
    inv_n = 1.0 / float(n_trials)
    # Phi^{-1} via the rational approximation. For our purposes the
    # math.erf-based inverse-CDF would be overkill — use the linear
    # ApproxNormalQuantile from Bailey's paper.
    a = 1.0 - inv_n
    z_a = _norm_inv(a)
    z_a_minus = _norm_inv(1.0 - inv_n * math.exp(-1.0))
    e_max = trials_sharpe_std * (
        (1.0 - _EULER_GAMMA) * z_a + _EULER_GAMMA * z_a_minus
    )
    return e_max


def _norm_inv(p: float) -> float:
    """Inverse CDF of the standard normal. Beasley-Springer-Moro
    approximation; accurate to ~1e-9 across the central region we
    care about. Avoids a scipy dependency."""
    if not (0.0 < p < 1.0):
        if p <= 0:
            return -float("inf")
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return num / den
    if p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    return -num / den


def deflated_sharpe(
    returns: np.ndarray,
    trial_sharpes: Sequence[float],
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    The probability that the strategy's TRUE Sharpe is positive
    AFTER deflating the observed Sharpe by the maximum-of-N-trials
    null. Caller must pass the Sharpes of ALL trials run during the
    search (winners AND losers) — that's how the deflation honestly
    accounts for the search budget.

    Range [0, 1]; values > 0.95 mean "even after the search, this
    Sharpe is robust at 95%".

    Equivalent to PSR with benchmark = expected_max_sharpe(trials).
    """
    trials = np.asarray([s for s in trial_sharpes if math.isfinite(s)],
                        dtype=float)
    n_trials = len(trials)
    if n_trials < 2:
        # Without a search distribution, deflation reduces to PSR
        # against zero (no search penalty applicable).
        return probabilistic_sharpe(returns)
    trials_std = float(trials.std(ddof=1))
    benchmark = expected_max_sharpe(trials_std, n_trials)
    return probabilistic_sharpe(returns, sharpe_benchmark=benchmark)
