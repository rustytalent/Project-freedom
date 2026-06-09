"""Beta-binomial empirical Bayes bucket shrinkage.

Replaces the fixed Wilson-style shrinkage ``pull = n / (n + 20)`` in
``liqpool/ml_model.py`` with a data-driven prior strength.

The shrinkage problem in one paragraph:
  Each (TF, factor) bucket has its own historical hit rate. Small
  buckets carry high sampling noise — their observed rates should be
  pulled toward the global mean. Large buckets carry low sampling
  noise — their observed rates should be trusted. The fixed
  ``n/(n+20)`` arbitrarily declares "20 is the prior strength"; in
  practice the right number depends on how dispersed bucket rates
  actually are across the population. Beta-binomial fits that
  dispersion from the buckets themselves.

Method (method-of-moments estimator for the beta-binomial):
  Treat each bucket as draws from Binomial(n_b, p_b) where p_b ~
  Beta(alpha, beta). Estimate (alpha, beta) from the cross-bucket
  mean and variance via moments. The shrunk estimate for bucket b is

      p_hat_b = (k_b + alpha) / (n_b + alpha + beta)

  which is exactly Bayesian posterior mean under Beta(alpha, beta)
  prior. The prior strength ``alpha + beta`` is what replaces the
  fixed 20 — and it's now learned per training set.

Behaviour vs ``n/(n+20)``:
  * Buckets larger than the learned prior strength shrink less.
  * Buckets smaller shrink more.
  * If all buckets agree closely (low cross-bucket variance), the
    prior strength is HIGH and even moderately-sized buckets shrink
    hard — appropriate when buckets are statistically homogeneous.
  * If buckets disagree strongly, the prior strength is LOW and
    even small buckets keep most of their observed rate — appropriate
    when buckets are genuinely heterogeneous.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BetaBinomialPrior:
    """Fitted prior. ``alpha + beta`` is the learned 'prior strength'
    that replaces the hard-coded 20 in fixed-Wilson shrinkage."""
    alpha: float
    beta: float
    global_rate: float          # alpha / (alpha + beta)
    prior_strength: float       # alpha + beta
    n_buckets_used: int

    def shrink(self, hits: float, n: float) -> float:
        """Return the posterior-mean hit rate for one bucket. ``hits``
        is the observed count of successes; ``n`` is bucket size."""
        return (hits + self.alpha) / (n + self.alpha + self.beta)


def fit_beta_binomial_prior(
    counts: Sequence[Tuple[float, float]],
    min_buckets: int = 2,
    global_rate_floor: float = 1e-4,
    global_rate_ceil: float = 1.0 - 1e-4,
    fallback_prior_strength: float = 20.0,
) -> BetaBinomialPrior:
    """Method-of-moments fit of a Beta(alpha, beta) prior over the
    hit-rate distribution across buckets.

    ``counts`` is a sequence of ``(hits, n)`` per bucket. Buckets
    with ``n <= 0`` are skipped.

    The fallback path: if fewer than ``min_buckets`` valid buckets
    exist, or the moment-matching equations don't yield a positive
    (alpha, beta) — common when cross-bucket variance is at or
    below the binomial noise floor — we fall back to a flat prior
    centered on the global rate with strength
    ``fallback_prior_strength``. This is exactly the legacy
    ``n/(n+20)`` behaviour with the prior centered correctly.
    """
    valid = [(float(k), float(n)) for k, n in counts if n > 0]
    if len(valid) < min_buckets:
        # Fallback: flat prior at observed global rate.
        total_h = sum(k for k, _ in valid)
        total_n = sum(n for _, n in valid)
        rate = total_h / total_n if total_n > 0 else 0.5
        rate = min(max(rate, global_rate_floor), global_rate_ceil)
        ps = float(fallback_prior_strength)
        return BetaBinomialPrior(
            alpha=rate * ps,
            beta=(1.0 - rate) * ps,
            global_rate=rate,
            prior_strength=ps,
            n_buckets_used=len(valid),
        )

    hits = np.array([k for k, _ in valid])
    ns = np.array([n for _, n in valid])
    # Per-bucket observed rates, weighted by bucket size for the
    # moment estimates.
    rates = hits / ns
    # Sample mean and variance of rates, weighted by bucket size.
    weights = ns / ns.sum()
    mean_rate = float(np.sum(rates * weights))
    mean_rate = min(max(mean_rate, global_rate_floor), global_rate_ceil)
    var_rate = float(np.sum(weights * (rates - mean_rate) ** 2))

    # Binomial-noise component of the variance: E[var(p_hat | p)] =
    # p * (1 - p) / n averaged over buckets. The TRUE cross-bucket
    # variance is the observed variance minus this binomial floor.
    binomial_floor = float(np.sum(
        weights * (mean_rate * (1.0 - mean_rate)) / ns
    ))
    cross_bucket_var = max(0.0, var_rate - binomial_floor)

    # Beta has var = mu*(1-mu) / (1 + prior_strength). Solve.
    mu_term = mean_rate * (1.0 - mean_rate)
    if cross_bucket_var <= 0 or mu_term <= cross_bucket_var:
        # All variance explained by binomial noise -> buckets are
        # statistically identical -> infinite-strength prior. Use
        # the same fallback as too-few-buckets but centered on the
        # observed mean.
        ps = float(fallback_prior_strength)
    else:
        ps = mu_term / cross_bucket_var - 1.0
        # Guard: extremely tight buckets can push ps very high; cap
        # at a reasonable maximum so a single anomalous bucket
        # doesn't shrink everything to the mean.
        ps = float(max(1.0, min(ps, 10000.0)))

    return BetaBinomialPrior(
        alpha=mean_rate * ps,
        beta=(1.0 - mean_rate) * ps,
        global_rate=mean_rate,
        prior_strength=ps,
        n_buckets_used=len(valid),
    )


class BetaBinomialBucketShrinker:
    """Stateful interface compatible with the existing fit/apply
    pattern in ``ml_model.py``.

    Usage::

        sh = BetaBinomialBucketShrinker()
        sh.fit(bucket_counts)          # bucket_counts: {bucket_key: (hits, n)}
        rate = sh.apply(bucket_key, hits, n)
    """

    def __init__(self) -> None:
        self._prior: Optional[BetaBinomialPrior] = None

    def fit(self, bucket_counts: Dict[object, Tuple[float, float]]) -> "BetaBinomialBucketShrinker":
        self._prior = fit_beta_binomial_prior(list(bucket_counts.values()))
        return self

    def apply(self, bucket_key: object, hits: float, n: float) -> float:
        """Return the shrunk estimate for a specific bucket. If the
        prior wasn't fit (called pre-fit), return the raw observed
        rate. ``bucket_key`` is unused at v1 — every bucket uses the
        same prior; in v2 we could allow per-bucket priors for
        heterogeneous regimes."""
        if self._prior is None:
            return float(hits) / max(float(n), 1.0)
        return self._prior.shrink(hits, n)

    @property
    def prior(self) -> Optional[BetaBinomialPrior]:
        return self._prior

    @property
    def prior_strength(self) -> float:
        """The learned 'prior strength' that replaces the legacy
        fixed 20. Higher = more shrinkage; lower = less."""
        return self._prior.prior_strength if self._prior else 0.0
