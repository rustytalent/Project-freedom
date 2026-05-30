"""Tests for the Stage-C per-distance-bucket isotonic recalibration layer.

The invariants the layer must guarantee:

1. Sparse buckets fall back to identity — no bucket can *introduce* error.
2. The bucket mapping matches ``liqpool.timing.distance_bucket`` exactly.
3. Within a constant distance bucket the layer preserves argsort
   (isotonic regression is monotone non-decreasing).
4. ``transform`` on an unfit calibrator is identity.
5. Fitting on biased synthetic data actually reduces calibration error in
   the bucket that's biased.
"""
from __future__ import annotations

import numpy as np
import pytest

from liqpool.distance_calibration import DistanceCalibrator, distance_bucket
from liqpool.timing import distance_bucket as timing_distance_bucket


def test_bucket_assignment_matches_timing():
    # The audit table and the calibrator must label every row identically.
    for d in [-0.1, 0.0, 0.5, 0.99, 1.0, 2.5, 3.0, 4.9, 5.0, 9.99, 10.0, 50.0]:
        assert distance_bucket(d) == timing_distance_bucket(d)


def test_unfit_transform_is_identity():
    calib = DistanceCalibrator()
    raw = np.array([0.1, 0.3, 0.7, 0.95])
    dist = np.array([0.5, 2.0, 4.0, 11.0])
    assert calib.is_fitted is False
    out = calib.transform(raw, dist)
    np.testing.assert_array_equal(out, raw)


def test_length_mismatch_raises():
    calib = DistanceCalibrator()
    with pytest.raises(ValueError):
        calib.fit([0.1, 0.2], [1.0], [0, 1])
    calib.fit([0.1, 0.2], [1.0, 2.0], [0, 1], min_bucket_n=1)
    with pytest.raises(ValueError):
        calib.transform([0.1, 0.2], [1.0])


def test_sparse_bucket_falls_back_to_identity():
    rng = np.random.default_rng(7)
    # 0-1 ATR bucket has plenty of rows; 10+ ATR bucket has only 3.
    n_close = 600
    raw_close = rng.uniform(0.1, 0.9, n_close)
    y_close = (raw_close > 0.5).astype(int)
    dist_close = rng.uniform(0.0, 0.99, n_close)

    raw_far = np.array([0.10, 0.50, 0.90])
    y_far = np.array([0, 1, 0])
    dist_far = np.array([12.0, 15.0, 20.0])

    raw = np.concatenate([raw_close, raw_far])
    dist = np.concatenate([dist_close, dist_far])
    y = np.concatenate([y_close, y_far])

    calib = DistanceCalibrator(min_bucket_n=300).fit(raw, dist, y)
    assert "0-1 ATR" in calib.iso_by_bucket            # large bucket fitted
    assert "10+ ATR" not in calib.iso_by_bucket        # sparse bucket skipped
    stats = {s["bucket"]: s for s in calib.stats_rows()}
    assert stats["10+ ATR"]["fitted"] is False
    assert "n<300" in stats["10+ ATR"]["skip_reason"]

    # Sparse-bucket rows pass through unchanged.
    out = calib.transform(raw, dist)
    np.testing.assert_allclose(out[-3:], raw_far)


def test_only_one_class_in_bucket_is_skipped():
    raw = np.linspace(0.05, 0.95, 400)
    dist = np.full(400, 0.5)                  # all in 0-1 ATR
    y = np.zeros(400, dtype=int)              # only class 0 present
    calib = DistanceCalibrator(min_bucket_n=100).fit(raw, dist, y)
    assert "0-1 ATR" not in calib.iso_by_bucket
    stats = {s["bucket"]: s for s in calib.stats_rows()}
    assert "only one class" in stats["0-1 ATR"]["skip_reason"]
    np.testing.assert_array_equal(calib.transform(raw, dist), raw)


def test_within_bucket_preserves_argsort():
    rng = np.random.default_rng(11)
    n = 800
    raw = rng.uniform(0.0, 1.0, n)
    dist = rng.uniform(0.0, 0.99, n)          # all in 0-1 ATR
    y = (raw + rng.normal(0, 0.15, n) > 0.5).astype(int)
    calib = DistanceCalibrator(min_bucket_n=200).fit(raw, dist, y)
    assert calib.is_fitted

    # New rows at constant distance, sorted ascending in raw.
    test_dist = np.full(50, 0.5)
    test_raw = np.linspace(0.02, 0.98, 50)
    out = calib.transform(test_raw, test_dist)

    # Isotonic is non-decreasing → output must be non-decreasing.
    diffs = np.diff(out)
    assert (diffs >= -1e-9).all(), "isotonic must be non-decreasing within a bucket"


def test_fit_reduces_calibration_error_in_biased_bucket():
    """Synthetic raw predictions are biased high in the 3-5 ATR bucket.

    After fitting, the bucket's mean prediction should be much closer to the
    bucket's empirical positive rate than the raw input.
    """
    rng = np.random.default_rng(23)
    # Bucket 3-5 ATR: 1000 rows, raw is biased upward (mean ~0.70) but the
    # true rate is only 0.40.
    n_b = 1000
    raw_b = np.clip(rng.normal(0.70, 0.10, n_b), 0.01, 0.99)
    y_b = (rng.uniform(0, 1, n_b) < 0.40).astype(int)
    dist_b = rng.uniform(3.0, 4.9, n_b)

    # A second well-calibrated bucket so fit does something everywhere.
    n_a = 1000
    raw_a = np.clip(rng.normal(0.50, 0.15, n_a), 0.01, 0.99)
    y_a = (rng.uniform(0, 1, n_a) < raw_a).astype(int)
    dist_a = rng.uniform(0.0, 0.99, n_a)

    raw = np.concatenate([raw_a, raw_b])
    dist = np.concatenate([dist_a, dist_b])
    y = np.concatenate([y_a, y_b])

    calib = DistanceCalibrator(min_bucket_n=300).fit(raw, dist, y)
    out = calib.transform(raw, dist)

    raw_err_b = float(raw_b.mean() - y_b.mean())          # large +bias before
    cal_err_b = float(out[n_a:].mean() - y_b.mean())      # after recalibration
    assert abs(cal_err_b) < abs(raw_err_b) / 3.0, (
        f"expected per-bucket cal_err to shrink substantially "
        f"(raw={raw_err_b:+.3f}, cal={cal_err_b:+.3f})"
    )
    # And the well-calibrated bucket should not get worse.
    raw_err_a = float(raw_a.mean() - y_a.mean())
    cal_err_a = float(out[:n_a].mean() - y_a.mean())
    assert abs(cal_err_a) <= abs(raw_err_a) + 0.03


def test_stats_rows_shape_matches_buckets():
    raw = np.linspace(0.05, 0.95, 600)
    dist = np.linspace(0.0, 20.0, 600)
    y = (raw > 0.5).astype(int)
    calib = DistanceCalibrator(min_bucket_n=10).fit(raw, dist, y)
    rows = calib.stats_rows()
    assert {r["bucket"] for r in rows} == {
        "0-1 ATR", "1-3 ATR", "3-5 ATR", "5-10 ATR", "10+ ATR"
    }
    for r in rows:
        assert "fitted" in r and "n" in r and "base_rate" in r
