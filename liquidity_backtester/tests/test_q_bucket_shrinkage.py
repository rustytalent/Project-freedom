"""Tests for Q bucket-shrinkage decompression.

The Q (PoolRespectModel) bucket calibration historically used an unbounded
``pull = n/(n+20)`` weight. For large OOS buckets (n>=4000) this collapses
every prediction in the bucket to the bucket empirical mean, producing the
"Q compression" symptom (Q range 23-56% instead of 0-100%, gate at 70%
unreachable).

These tests pin the new ``shrinkage_max`` cap:

  * shrinkage_max=1.0 (legacy default) reproduces the old unbounded behaviour.
  * shrinkage_max=0.30 (new default in Config) caps pull at 0.30, preserving
    the underlying gbm's within-bucket variance.
"""
from __future__ import annotations

from collections import namedtuple
import unittest

import numpy as np
import pandas as pd

from liqpool.ml_model import BucketCalib, PoolRespectModel


_Contributor = namedtuple("_Contributor", ["source"])


class _PoolStub:
    """Minimal Pool-like object satisfying fit_bucket_calib's attribute reads."""
    def __init__(self, tfs, factor="EQH"):
        self.tfs = list(tfs)
        self.contributors = [_Contributor(factor)]


def _model() -> PoolRespectModel:
    """A minimal PoolRespectModel with a stubbed _gbm so predict_raw is callable."""
    m = PoolRespectModel()
    m.feature_names = ["x"]
    # We never invoke predict_raw in these tests; just exercise fit_bucket_calib.
    return m


class QBucketShrinkageCapTests(unittest.TestCase):

    def test_legacy_default_unbounded_pull(self) -> None:
        # 4000 samples in one bucket -> pull = 4000/4020 ≈ 0.9950
        # Legacy default shrinkage_max=1.0 should NOT cap it.
        pools = [_PoolStub(tfs=("base", "15min", "60min", "180min"), factor="EQH")
                 for _ in range(4000)]
        y_true = np.zeros(4000, dtype=int)
        p_pred = np.zeros(4000, dtype=float)
        m = _model()
        m.fit_bucket_calib(pools, y_true, p_pred, min_bucket_n=10,
                           shrinkage_max=1.0)
        # Exactly one bucket should be registered.
        self.assertEqual(len(m.bucket_calib), 1)
        bc = next(iter(m.bucket_calib.values()))
        self.assertAlmostEqual(bc.pull_weight, 4000 / 4020.0, places=4)
        self.assertGreater(bc.pull_weight, 0.99)

    def test_new_default_caps_pull_at_0_30(self) -> None:
        # Same large bucket, but shrinkage_max=0.30 caps pull weight.
        pools = [_PoolStub(tfs=("base", "15min", "60min", "180min"), factor="EQH")
                 for _ in range(4000)]
        y_true = np.zeros(4000, dtype=int)
        p_pred = np.zeros(4000, dtype=float)
        m = _model()
        m.fit_bucket_calib(pools, y_true, p_pred, min_bucket_n=10,
                           shrinkage_max=0.30)
        bc = next(iter(m.bucket_calib.values()))
        self.assertAlmostEqual(bc.pull_weight, 0.30, places=6)

    def test_small_bucket_below_cap_keeps_natural_pull(self) -> None:
        # 5 samples -> raw pull = 5/25 = 0.20. Below the 0.30 cap -> keep 0.20.
        # Min bucket size also matters; pass min_bucket_n=4 so this bucket fits.
        pools = [_PoolStub(tfs=("base",), factor="EQH") for _ in range(5)]
        y_true = np.zeros(5, dtype=int)
        p_pred = np.zeros(5, dtype=float)
        m = _model()
        m.fit_bucket_calib(pools, y_true, p_pred, min_bucket_n=4,
                           shrinkage_max=0.30)
        bc = next(iter(m.bucket_calib.values()))
        self.assertAlmostEqual(bc.pull_weight, 5 / 25.0, places=6)

    def test_shrinkage_max_zero_disables_bucket_recalibration(self) -> None:
        # shrinkage_max=0 -> pull = 0 -> bucket adjustment is a no-op.
        pools = [_PoolStub(tfs=("base", "15min"), factor="EQH") for _ in range(1000)]
        y_true = np.array([1] * 600 + [0] * 400, dtype=int)
        p_pred = np.full(1000, 0.5, dtype=float)
        m = _model()
        m.fit_bucket_calib(pools, y_true, p_pred, min_bucket_n=10,
                           shrinkage_max=0.0)
        bc = next(iter(m.bucket_calib.values()))
        self.assertEqual(bc.pull_weight, 0.0)

    def test_shrinkage_max_out_of_range_clamped(self) -> None:
        # Defensive: shrinkage_max>1 or <0 is clamped to [0,1].
        pools = [_PoolStub(tfs=("base",), factor="EQH") for _ in range(1000)]
        y_true = np.zeros(1000, dtype=int)
        p_pred = np.zeros(1000, dtype=float)
        m = _model()
        m.fit_bucket_calib(pools, y_true, p_pred, min_bucket_n=10,
                           shrinkage_max=10.0)
        bc = next(iter(m.bucket_calib.values()))
        # Cap is min(1.0, 1000/1020) = min(1.0, 0.980) = 0.980.
        self.assertLess(bc.pull_weight, 1.0)

        m2 = _model()
        m2.fit_bucket_calib(pools, y_true, p_pred, min_bucket_n=10,
                            shrinkage_max=-0.5)
        bc2 = next(iter(m2.bucket_calib.values()))
        self.assertEqual(bc2.pull_weight, 0.0)

    def test_decompression_preserves_variance_in_predictions(self) -> None:
        """The decisive test: with cap, the .predict path preserves more of the
        underlying p_raw range than without the cap.

        We construct a bucket where empirical_rate=0.5 but the model's raw
        predictions span 0.20..0.80. Under legacy unbounded pull, predictions
        collapse to ~0.5. Under cap=0.30, they remain spread out.
        """
        n = 1000
        pools = [_PoolStub(tfs=("base", "15min"), factor="EQH") for _ in range(n)]
        y_true = np.random.default_rng(0).integers(0, 2, n).astype(int)

        # We can't easily exercise PoolRespectModel.predict end-to-end without
        # a real gbm, but we CAN compute what the post-shrinkage value would
        # be from raw predictions and the bucket_calib record.
        p_raw = np.linspace(0.20, 0.80, n)

        m_legacy = _model()
        m_legacy.fit_bucket_calib(pools, y_true, p_raw, min_bucket_n=10,
                                    shrinkage_max=1.0)
        m_decompressed = _model()
        m_decompressed.fit_bucket_calib(pools, y_true, p_raw, min_bucket_n=10,
                                          shrinkage_max=0.30)

        bc_legacy = next(iter(m_legacy.bucket_calib.values()))
        bc_decomp = next(iter(m_decompressed.bucket_calib.values()))

        # Apply the predict-time formula manually.
        def _apply(p_raw_arr, bc):
            return (1.0 - bc.pull_weight) * p_raw_arr + bc.pull_weight * bc.empirical_rate

        legacy_post = _apply(p_raw, bc_legacy)
        decomp_post = _apply(p_raw, bc_decomp)

        # Decompressed predictions span more range than legacy.
        legacy_span = float(legacy_post.max() - legacy_post.min())
        decomp_span = float(decomp_post.max() - decomp_post.min())
        self.assertGreater(decomp_span, legacy_span * 5,
            f"decompression should preserve much more range: "
            f"legacy_span={legacy_span}, decomp_span={decomp_span}")


if __name__ == "__main__":
    unittest.main()
