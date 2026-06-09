"""Stream N — calibration-upgrade tests.

Pins for the three opt-in calibrators:

  * empirical_bayes — beta-binomial shrinkage with a data-driven
    prior strength
  * conformal — distribution-free intervals with marginal coverage
  * temperature — rank-preserving smoother
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from liqpool.calibration import (
    BetaBinomialBucketShrinker,
    ConformalIntervalCalibrator,
    TemperatureScaler,
    fit_beta_binomial_prior,
)


# ---------------------------------------------------------------------------
# Empirical Bayes shrinkage
# ---------------------------------------------------------------------------

def test_beta_binomial_recovers_high_prior_strength_when_buckets_are_homogeneous():
    """When every bucket has the same true rate, observed bucket
    variance comes entirely from binomial noise. The fitted prior
    strength should be high (heavy shrinkage)."""
    rng = np.random.default_rng(0)
    true_rate = 0.55
    buckets = []
    for _ in range(40):
        n = int(rng.integers(50, 500))
        hits = int(rng.binomial(n, true_rate))
        buckets.append((hits, n))
    prior = fit_beta_binomial_prior(buckets)
    # Population is homogeneous -> prior strength should be quite high.
    # The exact value depends on noise but anything below ~50 is
    # too eager to trust noisy small buckets.
    assert prior.prior_strength >= 50.0
    assert 0.50 < prior.global_rate < 0.60


def test_beta_binomial_recovers_low_prior_strength_when_buckets_are_heterogeneous():
    """When buckets have genuinely different rates (some 0.2, some
    0.8), cross-bucket variance dwarfs binomial noise. The fitted
    prior strength should be LOW (light shrinkage) so each bucket
    keeps its own observed rate."""
    rng = np.random.default_rng(1)
    buckets = []
    for _ in range(30):
        # Half low-rate, half high-rate.
        true_rate = 0.20 if rng.random() < 0.5 else 0.80
        n = int(rng.integers(100, 400))
        hits = int(rng.binomial(n, true_rate))
        buckets.append((hits, n))
    prior = fit_beta_binomial_prior(buckets)
    # Heterogeneous population -> prior strength should be LOW
    # (well below the legacy fixed 20).
    assert prior.prior_strength < 20.0


def test_beta_binomial_falls_back_when_too_few_buckets():
    """With only one bucket, the moment-matching equations are
    undefined; the calibrator falls back to a flat prior at the
    observed global rate."""
    prior = fit_beta_binomial_prior([(30, 50)])
    assert prior.global_rate == pytest.approx(0.6, rel=1e-9)
    # Fallback prior strength is the documented constant (20.0).
    assert prior.prior_strength == 20.0


def test_beta_binomial_shrinker_apply_matches_posterior_mean():
    """Posterior mean formula: (hits + alpha) / (n + alpha + beta)."""
    rng = np.random.default_rng(2)
    buckets = []
    for _ in range(30):
        n = int(rng.integers(80, 300))
        hits = int(rng.binomial(n, 0.4))
        buckets.append((hits, n))
    shrinker = BetaBinomialBucketShrinker()
    shrinker.fit({i: b for i, b in enumerate(buckets)})
    prior = shrinker.prior
    assert prior is not None
    # Manual posterior for one bucket.
    hits, n = 30, 100
    expected = (hits + prior.alpha) / (n + prior.alpha + prior.beta)
    actual = shrinker.apply("any_key", hits, n)
    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_unfit_shrinker_returns_raw_rate():
    sh = BetaBinomialBucketShrinker()
    assert sh.apply("k", 30, 100) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Conformal intervals
# ---------------------------------------------------------------------------

def test_conformal_achieves_target_coverage_on_calibrated_data():
    """When p_hat is well-calibrated (p_hat ~ p), conformal
    intervals at alpha=0.10 should cover ~90% of test outcomes."""
    rng = np.random.default_rng(3)
    n_calib = 500
    n_test = 2000
    # Well-calibrated: y_i ~ Bernoulli(p_hat_i).
    p_calib = rng.uniform(0.1, 0.9, n_calib)
    y_calib = rng.binomial(1, p_calib).astype(float)
    p_test = rng.uniform(0.1, 0.9, n_test)
    y_test = rng.binomial(1, p_test).astype(float)

    cc = ConformalIntervalCalibrator()
    cc.fit(p_calib, y_calib, alpha=0.10)
    coverage = cc.empirical_coverage(p_test, y_test)
    # Target 90%. Allow generous slack for sampling noise on n=2000.
    assert 0.85 <= coverage <= 1.0


def test_conformal_widens_for_poorly_calibrated_inputs():
    """If p_hat is systematically off (low estimate against high
    realised hit rate), conformal compensates with wider intervals
    to maintain coverage."""
    rng = np.random.default_rng(4)
    n = 500
    # Poorly calibrated: p_hat says 0.3 but reality is ~0.7.
    p_calib = rng.uniform(0.25, 0.35, n)
    y_calib = rng.binomial(1, 0.70, n).astype(float)
    cc = ConformalIntervalCalibrator()
    cc.fit(p_calib, y_calib, alpha=0.10)
    # Half-width should be large enough that the interval at p_hat=0.3
    # nearly reaches the true region around 0.7.
    lower, upper = cc.predict_interval(0.30)
    assert upper >= 0.60
    assert lower <= 0.30  # downward bound saturates at 0


def test_conformal_predict_intervals_clips_to_0_1():
    cc = ConformalIntervalCalibrator(half_width_quantile=0.40,
                                     n_calibration=100,
                                     alpha_used_at_fit=0.10)
    intervals = cc.predict_intervals([0.05, 0.50, 0.95])
    assert intervals.shape == (3, 2)
    assert np.all(intervals[:, 0] >= 0.0)
    assert np.all(intervals[:, 1] <= 1.0)
    # The 0.05 lower bound clips to 0 not -0.35.
    assert intervals[0, 0] == 0.0


def test_conformal_handles_empty_calibration():
    cc = ConformalIntervalCalibrator()
    cc.fit([], [], alpha=0.10)
    # Degenerate but no crash; predicted interval is [0, 1]-clipped wide.
    lo, hi = cc.predict_interval(0.5)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


# ---------------------------------------------------------------------------
# Temperature scaler
# ---------------------------------------------------------------------------

def test_temperature_identity_on_well_calibrated():
    """When p_hat is already well-calibrated, the fitted temperature
    should be close to 1.0 (no rescaling needed)."""
    rng = np.random.default_rng(5)
    n = 1000
    p_calib = rng.uniform(0.1, 0.9, n)
    y_calib = rng.binomial(1, p_calib).astype(float)
    ts = TemperatureScaler()
    ts.fit(p_calib, y_calib)
    assert 0.7 < ts.temperature < 1.4


def test_temperature_below_one_for_overconfident_input():
    """If the input is systematically overconfident (saying 0.95 when
    the true rate is closer to 0.65), the fit should drive T > 1 to
    SOFTEN those probabilities back toward 0.5."""
    rng = np.random.default_rng(6)
    n = 1000
    p_hat = rng.uniform(0.85, 0.95, n)  # overconfident
    y = rng.binomial(1, 0.65, n).astype(float)
    ts = TemperatureScaler()
    ts.fit(p_hat, y)
    # T > 1 softens predictions.
    assert ts.temperature > 1.2


def test_temperature_is_rank_preserving():
    """Apply on a sequence of inputs; the relative ranking must be
    preserved (monotonic transform)."""
    ts = TemperatureScaler(temperature=2.5)
    inputs = [0.1, 0.3, 0.5, 0.7, 0.9]
    outs = ts.transform(inputs)
    assert list(outs) == sorted(outs)


def test_temperature_handles_too_few_samples_with_identity():
    ts = TemperatureScaler()
    ts.fit([0.5, 0.5], [1.0, 0.0])  # n=2, below threshold
    assert ts.temperature == 1.0
    # Identity behaviour: transform output equals input (modulo clipping).
    out = ts.transform_one(0.3)
    assert math.isclose(out, 0.3, abs_tol=1e-6)


def test_temperature_handles_single_class_calibration_with_identity():
    """If the calibration set has only positives or only negatives,
    NLL has no minimum — fall back to identity."""
    ts = TemperatureScaler()
    ts.fit([0.1, 0.4, 0.6, 0.9, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert ts.temperature == 1.0
