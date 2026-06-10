"""LGBMX tests — the project's enhanced LightGBM trainer.

Pins the five enhancements:
  1. purged chronological folds with embargo gaps
  2. focal loss objective trains and beats fallback on imbalance
  3. declarative constraints by name (and refusal on bad names)
  4. auto-calibration on out-of-fold predictions
  5. importance stability tracking across fits
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from liqpool.lgbm_x import (
    LGBMX,
    LGBMXConfig,
    purged_time_folds,
)


def _binary_frame(n: int = 800, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic binary problem with two informative features and one
    noise feature."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    logit = 1.2 * x1 - 0.8 * x2
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise": noise})
    return X, y


# ---------------------------------------------------------------------------
# Purged folds
# ---------------------------------------------------------------------------

def test_purged_folds_have_embargo_gaps():
    folds = purged_time_folds(n=400, n_folds=4, embargo_rows=10)
    assert len(folds) == 4
    for tr, va in folds:
        # No train index inside the embargoed zone around validation.
        va_lo, va_hi = va.min(), va.max()
        too_close = ((tr >= va_lo - 10) & (tr <= va_hi + 10)).sum()
        assert too_close == 0, "train rows leaked into the embargo zone"


def test_purged_folds_cover_all_validation_rows_once():
    folds = purged_time_folds(n=400, n_folds=4, embargo_rows=10)
    seen = np.concatenate([va for _, va in folds])
    assert len(seen) == 400
    assert len(np.unique(seen)) == 400


def test_purged_folds_degenerate_small_n():
    folds = purged_time_folds(n=30, n_folds=4, embargo_rows=5)
    # Falls back to one chronological split.
    assert len(folds) <= 1


# ---------------------------------------------------------------------------
# Fit / predict basics
# ---------------------------------------------------------------------------

def test_unfit_predicts_fallback():
    m = LGBMX(LGBMXConfig(objective="binary"))
    X, _ = _binary_frame(10)
    assert np.allclose(m.predict(X), 0.5)


def test_small_n_stays_unfit():
    X, y = _binary_frame(40)
    m = LGBMX().fit(X, y)
    assert not m.is_fitted


def test_binary_fit_learns_signal():
    X, y = _binary_frame(800)
    m = LGBMX(LGBMXConfig(objective="binary")).fit(X, y)
    assert m.is_fitted
    # Strong-positive vs strong-negative cases separate.
    hi = m.predict(pd.DataFrame({"x1": [2.0], "x2": [-2.0], "noise": [0.0]}))[0]
    lo = m.predict(pd.DataFrame({"x1": [-2.0], "x2": [2.0], "noise": [0.0]}))[0]
    assert hi > lo + 0.3
    assert 0.0 <= lo <= hi <= 1.0


def test_oof_metric_recorded():
    X, y = _binary_frame(800)
    m = LGBMX().fit(X, y)
    assert m.oof_metric is not None and 0.0 < m.oof_metric < 0.7


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def test_focal_objective_trains_on_imbalanced_target():
    rng = np.random.default_rng(1)
    n = 1000
    x1 = rng.normal(0, 1, n)
    # 8% positive rate, signal in x1.
    p = 1 / (1 + np.exp(-(x1 * 1.5 - 2.6)))
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"x1": x1, "noise": rng.normal(0, 1, n)})
    m = LGBMX(LGBMXConfig(objective="focal", focal_gamma=2.0,
                          calibrate=False)).fit(X, y)
    assert m.is_fitted
    hi = m.predict(pd.DataFrame({"x1": [2.5], "noise": [0.0]}))[0]
    lo = m.predict(pd.DataFrame({"x1": [-2.5], "noise": [0.0]}))[0]
    assert hi > lo          # signal recovered despite imbalance
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# Declarative constraints
# ---------------------------------------------------------------------------

def test_monotone_constraint_by_name_is_enforced():
    """+1 monotone on x1: predictions must be non-decreasing in x1
    holding everything else fixed."""
    X, y = _binary_frame(800)
    m = LGBMX(LGBMXConfig(
        objective="binary",
        monotone={"x1": +1},
        calibrate=False,    # isotonic is monotone in the SCORE, not x1;
                            # keep the raw path for a clean pin
    )).fit(X, y)
    grid = np.linspace(-2.5, 2.5, 9)
    preds = m.predict(pd.DataFrame({
        "x1": grid, "x2": np.zeros(9), "noise": np.zeros(9),
    }))
    diffs = np.diff(preds)
    assert (diffs >= -1e-9).all(), "monotone constraint violated"


def test_unknown_constraint_name_refuses_to_fit():
    X, y = _binary_frame(200)
    m = LGBMX(LGBMXConfig(monotone={"not_a_feature": 1}))
    with pytest.raises(ValueError, match="not in features"):
        m.fit(X, y)


def test_unknown_interaction_name_refuses_to_fit():
    X, y = _binary_frame(200)
    m = LGBMX(LGBMXConfig(interactions=[["x1", "ghost"]]))
    with pytest.raises(ValueError, match="not in features"):
        m.fit(X, y)


# ---------------------------------------------------------------------------
# Auto-calibration + conformal
# ---------------------------------------------------------------------------

def test_calibrated_predictions_have_low_calibration_error():
    X, y = _binary_frame(1500, seed=2)
    m = LGBMX(LGBMXConfig(objective="binary", calibrate=True)).fit(X, y)
    Xt, yt = _binary_frame(1500, seed=3)
    p = m.predict(Xt)
    # Bucketed calibration error on fresh data.
    df = pd.DataFrame({"p": p, "y": yt})
    df["bucket"] = pd.cut(df["p"], bins=[0, 0.25, 0.5, 0.75, 1.0])
    errs = []
    for _, g in df.groupby("bucket", observed=True):
        if len(g) >= 50:
            errs.append(abs(g["p"].mean() - g["y"].mean()))
    assert errs and max(errs) < 0.10


def test_conformal_intervals_emitted_and_cover():
    X, y = _binary_frame(1500, seed=4)
    m = LGBMX(LGBMXConfig(objective="binary", calibrate=True,
                          conformal_alpha=0.10)).fit(X, y)
    Xt, yt = _binary_frame(800, seed=5)
    intervals = m.predict_interval(Xt)
    assert intervals.shape == (800, 2)
    inside = ((yt >= intervals[:, 0]) & (yt <= intervals[:, 1])).mean()
    assert inside >= 0.80   # 90% nominal with sampling slack


def test_conformal_disabled_gives_degenerate_band():
    X, y = _binary_frame(800)
    m = LGBMX(LGBMXConfig(conformal_alpha=None)).fit(X, y)
    iv = m.predict_interval(X.head(5))
    assert np.allclose(iv[:, 0], iv[:, 1])


# ---------------------------------------------------------------------------
# Importance stability
# ---------------------------------------------------------------------------

def test_importance_sums_to_one_and_ranks_signal_over_noise():
    X, y = _binary_frame(800)
    m = LGBMX().fit(X, y)
    imp = m.importance
    assert abs(sum(imp.values()) - 1.0) < 1e-6
    assert imp["x1"] > imp["noise"]


def test_importance_drift_high_for_same_distribution():
    X1, y1 = _binary_frame(800, seed=6)
    X2, y2 = _binary_frame(800, seed=7)
    m1 = LGBMX().fit(X1, y1)
    m2 = LGBMX().fit(X2, y2)
    corr = m2.importance_drift(m1.importance)
    assert corr is not None and corr > 0.4


def test_importance_drift_none_when_too_few_common_features():
    X, y = _binary_frame(800)
    m = LGBMX().fit(X, y)
    assert m.importance_drift({"a": 1.0}) is None


# ---------------------------------------------------------------------------
# Time-decay weights integration
# ---------------------------------------------------------------------------

def test_time_decay_weights_path_runs():
    X, y = _binary_frame(800)
    times = pd.date_range("2026-01-01", periods=800, freq="1h")
    m = LGBMX(LGBMXConfig(half_life_days=10.0)).fit(X, y, sample_times=times)
    assert m.is_fitted
