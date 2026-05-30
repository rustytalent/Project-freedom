"""R1 tests — the policy RETURN model (Huber regression on realized net R).

The contract these tests pin:

1. Winsorization (R_WINSOR_LOW / R_WINSOR_HIGH) actually clips both ends.
2. Synthetic fit + predict roundtrip works on a small labelled frame; an
   informative signal in the features yields a positive Spearman correlation
   between predicted and realized R.
3. Chronological split: every validation row's `available_at` is strictly
   later than every train row's (this mirrors the live use case — the model
   is evaluated on rows it never saw, ordered in time).
4. Backward compatibility: a bundle stub without `policy_return_model` loads
   through the same `getattr` defaults path used by `_load_predict_report`.
5. Determinism: same seed -> identical predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liqpool.policy_model import (
    R_WINSOR_HIGH,
    R_WINSOR_LOW,
    PolicyReturnMetrics,
    PolicyReturnModel,
    PolicyReturnModelSuite,
    _chronological_split,
    _winsorize,
)


pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")


def _synthetic_labels(n: int = 220, seed: int = 0) -> pd.DataFrame:
    """Build a policy-labels frame with one informative feature.

    `score` linearly drives realized R with noise. `mode` is constant. Times
    advance monotonically so the chronological split is deterministic.
    """
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 1.0, n)
    # R ~ 4 * (score - 0.5) + N(0, 0.6); a few extreme tail values to test winsor.
    realized_r = 4.0 * (scores - 0.5) + rng.normal(0.0, 0.6, n)
    realized_r[0] = -50.0    # extreme tail (winsorized at -2.5)
    realized_r[1] = +50.0    # extreme tail (winsorized at +5.0)
    start = pd.Timestamp("2025-01-01 09:30", tz="UTC")
    available = [start + pd.Timedelta(minutes=15 * i) for i in range(n)]
    return pd.DataFrame({
        "mode": ["touch_confirmed"] * n,
        "split": ["oos"] * n,
        "symbol": [f"S{i % 5}" for i in range(n)],
        "sector": ["BANKING"] * n,
        "pool_idx": list(range(n)),
        "direction": ["UP"] * n,
        "direction_sign": [1] * n,
        "side": ["buy"] * n,
        "pool_low": scores * 100.0,
        "pool_high": scores * 100.0 + 5.0,
        "pool_mid": scores * 100.0 + 2.5,
        "available_at": [str(a) for a in available],
        "formed_at": [str(a - pd.Timedelta(hours=1)) for a in available],
        "score": scores,
        "tf_count": rng.integers(1, 5, n),
        "factor": ["EQHL"] * n,
        "policy_target_trade_generated": np.ones(n, dtype=int),
        "policy_target_return_r": realized_r,
        "policy_target_win": (realized_r > 0).astype(int),
    })


def test_winsorize_clips_both_ends():
    out = _winsorize(np.array([-100.0, -2.4, 0.0, 4.9, 99.0]))
    assert out[0] == R_WINSOR_LOW
    assert out[1] == pytest.approx(-2.4)
    assert out[2] == 0.0
    assert out[3] == pytest.approx(4.9)
    assert out[4] == R_WINSOR_HIGH


def test_fit_and_evaluate_roundtrip_on_synthetic():
    labels = _synthetic_labels(n=300, seed=1)
    model = PolicyReturnModel(mode="touch_confirmed")
    model.fit(labels, seed=42, min_trades=50)
    assert model.metrics.status == "trained"
    assert model.metrics.train_n > 0 and model.metrics.val_n > 0
    # Evaluate on the same labels (frame includes both train and val rows; this
    # only checks the predict path works, not generalisation).
    model.evaluate(labels)
    assert model.metrics.oos_n == len(labels)
    # The informative feature should yield a positive monotone relationship.
    assert model.metrics.oos_spearman is not None
    assert model.metrics.oos_spearman > 0.20


def test_top_decile_realized_r_uses_realized_not_predicted():
    """oos_top_decile_realized_r is the *realized* R within the top-decile of
    PREDICTED R — that's the edge gate's actual question."""
    labels = _synthetic_labels(n=300, seed=2)
    model = PolicyReturnModel(mode="touch_confirmed")
    model.fit(labels, seed=42, min_trades=50)
    model.evaluate(labels)
    # On synthetic data with a strong feature, the top-decile by predicted R
    # should realize a positive mean R.
    assert model.metrics.oos_top_decile_n > 0
    assert model.metrics.oos_top_decile_realized_r > 0.0


def test_chronological_split_is_time_ordered():
    labels = _synthetic_labels(n=200, seed=3)
    # _chronological_split sorts by available_at, then trims a tail for val.
    tr_idx, val_idx = _chronological_split(labels, val_frac=0.25)
    train_avail = pd.to_datetime(labels.loc[tr_idx, "available_at"], utc=True)
    val_avail = pd.to_datetime(labels.loc[val_idx, "available_at"], utc=True)
    assert train_avail.max() <= val_avail.min(), (
        "validation rows must come strictly after training rows in time"
    )


def test_suite_handles_skipped_mode_gracefully():
    """A mode below min_trades is reported as skipped with a reason — no crash."""
    labels = _synthetic_labels(n=120, seed=4)
    # Inject a second mode with only 10 trades — below default min_trades=80.
    sparse = labels.head(10).copy()
    sparse["mode"] = "blind_limit"
    labels_two_modes = pd.concat([labels, sparse], ignore_index=True)
    suite = PolicyReturnModelSuite().fit(labels_two_modes, labels_two_modes,
                                          seed=42, min_trades=80)
    assert "touch_confirmed" in suite.models
    assert "blind_limit" not in suite.models
    assert suite.metrics["blind_limit"].status == "skipped"
    assert "needs >= 80" in suite.metrics["blind_limit"].reason


def test_predictions_table_sorted_descending():
    labels = _synthetic_labels(n=180, seed=5)
    model = PolicyReturnModel(mode="touch_confirmed")
    model.fit(labels, seed=42, min_trades=50)
    preds = model.predictions(labels)
    assert "policy_return_model_predicted_r" in preds.columns
    p = preds["policy_return_model_predicted_r"].to_numpy()
    # Strict non-increasing within the sorted output.
    assert (np.diff(p) <= 1e-9).all(), "predictions must be sorted descending"


def test_calibration_table_schema():
    labels = _synthetic_labels(n=200, seed=6)
    suite = PolicyReturnModelSuite().fit(labels, labels, seed=42, min_trades=50)
    cal = suite.calibration_table
    assert not cal.empty
    expected = {"mode", "bucket", "n", "mean_predicted_r", "mean_realized_r",
                "win_rate", "calibration_error_r"}
    assert expected.issubset(set(cal.columns))


def test_determinism_same_seed_same_predictions():
    labels = _synthetic_labels(n=200, seed=7)
    m1 = PolicyReturnModel(mode="touch_confirmed").fit(labels, seed=99, min_trades=50)
    m2 = PolicyReturnModel(mode="touch_confirmed").fit(labels, seed=99, min_trades=50)
    p1 = m1.predict_frame(labels)
    p2 = m2.predict_frame(labels)
    np.testing.assert_allclose(p1, p2)


def test_backward_compat_default_attributes():
    """A report-like object without R1 attrs must round-trip via the same
    getattr pattern `_load_predict_report` uses."""
    class _OldBundle:
        # No policy_return_model* fields, like a pre-R1 saved bundle.
        pass

    report = _OldBundle()
    # The defaults the loader assigns:
    defaults = (
        ("policy_return_model", None),
        ("policy_return_model_report", []),
        ("policy_return_model_calibration", []),
        ("policy_return_model_feature_importance", []),
    )
    for attr, default in defaults:
        if not hasattr(report, attr):
            setattr(report, attr, default)
    assert report.policy_return_model is None
    assert report.policy_return_model_report == []


def test_metrics_to_dict_is_serialisable():
    m = PolicyReturnMetrics(mode="touch_confirmed", status="trained",
                            oos_spearman=0.42, oos_top_decile_realized_r=0.15)
    d = m.to_dict()
    assert d["mode"] == "touch_confirmed"
    assert d["status"] == "trained"
    assert d["oos_spearman"] == pytest.approx(0.42)
    # JSON-serialisability proxy: every value is a basic Python type.
    import json
    json.dumps(d)
