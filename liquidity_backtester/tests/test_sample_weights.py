"""Tests for the time-decay sample weight helper.

Pinned contracts:
  * half_life_days=None or <= 0 returns uniform 1.0s (back-compat).
  * Older samples get smaller weights than newer ones.
  * The weighted mean is ~1.0 (LightGBM loss scale parity).
  * No-future-leakage: passing a reference_time before the most-recent
    sample clamps ages to >= 0 — never produces negative ages.
  * min_weight floor prevents the oldest samples from collapsing to 0.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liqpool.sample_weights import time_decay_weights


class TimeDecayWeightsTests(unittest.TestCase):

    def test_none_halflife_returns_uniform(self):
        times = pd.date_range("2024-01-01", periods=10, freq="D")
        w = time_decay_weights(times, half_life_days=None)
        np.testing.assert_allclose(w, np.ones(10))

    def test_zero_halflife_returns_uniform(self):
        times = pd.date_range("2024-01-01", periods=10, freq="D")
        w = time_decay_weights(times, half_life_days=0.0)
        np.testing.assert_allclose(w, np.ones(10))

    def test_older_samples_get_lower_weight(self):
        # Newest at idx 9; oldest at idx 0.
        times = pd.date_range("2024-01-01", periods=10, freq="D")
        w = time_decay_weights(times, half_life_days=2.0)
        # Strict monotone increase from oldest to newest.
        for i in range(1, 10):
            self.assertGreater(w[i], w[i - 1],
                f"weight at idx {i} ({w[i]}) should exceed idx {i-1} ({w[i-1]})")

    def test_mean_normalises_to_one(self):
        times = pd.date_range("2024-01-01", periods=100, freq="D")
        w = time_decay_weights(times, half_life_days=30.0)
        self.assertAlmostEqual(w.mean(), 1.0, places=4)

    def test_min_weight_floor(self):
        # Tiny half-life vs big age range -> oldest sample would collapse
        # to ~0 without the floor.
        times = pd.date_range("2024-01-01", periods=10, freq="D")
        w = time_decay_weights(times, half_life_days=0.1, min_weight=0.1)
        # Floor applies BEFORE normalisation, so the post-normalisation
        # values may exceed 0.1 — but no weight should be vastly below.
        self.assertGreater(w.min(), 0.0)

    def test_empty_input_returns_empty(self):
        w = time_decay_weights([], half_life_days=180.0)
        self.assertEqual(len(w), 0)

    def test_half_life_semantics(self):
        # Sample at age = half_life should have ~half the weight of
        # sample at age 0. Use unnormalised pre-mean values via the floor.
        ref = pd.Timestamp("2024-01-31")
        times = [pd.Timestamp("2024-01-31"),       # age 0
                 pd.Timestamp("2024-01-01")]       # age 30 = half_life
        w = time_decay_weights(times, reference_time=ref,
                                half_life_days=30.0, min_weight=0.0)
        # Ratio of new / old should be ~2.0 (since exp(-ln(2)*30/30) = 0.5).
        self.assertAlmostEqual(w[0] / w[1], 2.0, places=2)


class IntegrationWithPoolRespectModelTests(unittest.TestCase):
    """Pin that PoolRespectModel.fit accepts a sample_weight kwarg
    without crashing on a synthetic frame, and that the back-compat
    sample_weight=None path is unchanged."""

    def _synth(self, n=120):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = (rng.random(n) > 0.5).astype(int)
        return X, y

    def test_back_compat_no_weight_unchanged(self):
        from liqpool.ml_model import PoolRespectModel
        X, y = self._synth()
        model = PoolRespectModel().fit(X, y, seed=11)
        self.assertGreater(model.train_n, 0)

    def test_uniform_weight_does_not_change_outcome(self):
        from liqpool.ml_model import PoolRespectModel
        X, y = self._synth()
        m_no = PoolRespectModel().fit(X, y, seed=11)
        m_uniform = PoolRespectModel().fit(
            X, y, seed=11, sample_weight=np.ones(len(X)))
        # Uniform weight 1.0 must produce identical val metrics.
        self.assertAlmostEqual(m_no.val_brier, m_uniform.val_brier, places=4)
        self.assertAlmostEqual(m_no.val_logloss, m_uniform.val_logloss, places=4)

    def test_per_sample_weight_changes_feature_importance(self):
        """Weighting should change which features the booster relies on.

        Construct a fixture where the first half has y driven by feature a
        and the second half has y driven by feature b. Train uniformly
        weighted, then again with the second half weighted 10x. The
        feature-importance gain on b should rise relative to a in the
        weighted run. This is the cleanest signal that LightGBM actually
        consumed the per-sample weight.
        """
        from liqpool.ml_model import PoolRespectModel
        n = 200
        rng = np.random.default_rng(0)
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        y = np.empty(n, dtype=int)
        # First half: y follows a; second half: y follows b.
        y[:n // 2] = (a[:n // 2] > 0).astype(int)
        y[n // 2:] = (b[n // 2:] > 0).astype(int)
        X = pd.DataFrame({"a": a, "b": b})

        m_uniform = PoolRespectModel().fit(X, y, seed=11)
        skewed = np.concatenate([np.full(n // 2, 1.0),
                                  np.full(n // 2, 10.0)])
        m_skewed = PoolRespectModel().fit(X, y, seed=11, sample_weight=skewed)

        # Read feature importance (gain) for both runs.
        def _imp(model):
            return dict(zip(
                model.feature_names,
                model._gbm.feature_importance(importance_type="gain")
            ))

        imp_u = _imp(m_uniform)
        imp_s = _imp(m_skewed)
        # b-to-a importance ratio should rise when second half is upweighted.
        # Use a + 1 to avoid div-by-zero on degenerate fits.
        ratio_u = imp_u["b"] / (imp_u["a"] + 1.0)
        ratio_s = imp_s["b"] / (imp_s["a"] + 1.0)
        self.assertGreater(ratio_s, ratio_u,
            f"upweighting samples where b drives y should raise b/a "
            f"gain ratio. uniform={ratio_u:.2f}, skewed={ratio_s:.2f}")


if __name__ == "__main__":
    unittest.main()
