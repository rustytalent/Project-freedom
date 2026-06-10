"""Stream M — flywheel model tests (M.4 through M.10).

Per-model pins, all following the same discipline:
  * unfit model returns the documented fallback (never crashes)
  * fit on a synthetic frame with KNOWN signal recovers the signal
  * predict paths clamp / NaN-guard
  * persistence round-trips where implemented
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from liqpool.flywheel import (
    BriefConfidenceMetaCalibrator,
    BucketAgingModel,
    CrossAssetTransferMatrix,
    DetectorTrustRouter,
    DriftImminentModel,
    ReactionArchetypeModel,
    RegretEstimator,
)


# ---------------------------------------------------------------------------
# M.4 — RegretEstimator
# ---------------------------------------------------------------------------

def _shadow_joined_frame(n: int = 400) -> pd.DataFrame:
    """Synthetic joined shadow frame with a clear signal: high |lcs|
    SKIPs tend to be regretted (the gate was too tight); low |lcs|
    SKIPs tend to be vindicated."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        lcs = float(rng.uniform(0.0, 0.30))
        p_regret = 0.15 + 2.5 * lcs   # 0.15 at lcs=0 -> 0.9 at lcs=0.3
        regret = rng.random() < p_regret
        rows.append({
            "event_kind": "skip_options_executor",
            "decision_context": json.dumps({
                "lcs": lcs, "macro_score": float(rng.uniform(-1, 1)),
            }),
            "counterfactual_outcome": json.dumps({
                "verdict": "rejection_regret" if regret
                           else "rejection_vindicated",
            }),
        })
    return pd.DataFrame(rows)


def test_regret_unfit_returns_max_entropy():
    m = RegretEstimator()
    p = m.predict_regret("skip_options_executor", {"lcs": 0.2})
    assert p == pytest.approx(0.5)


def test_regret_learns_lcs_signal():
    m = RegretEstimator().fit(_shadow_joined_frame())
    assert m.is_fitted
    hi = m.predict_regret("skip_options_executor", {"lcs": 0.28})
    lo = m.predict_regret("skip_options_executor", {"lcs": 0.02})
    assert hi > lo + 0.1   # near-threshold SKIPs carry more regret


def test_regret_handles_unparseable_outcomes():
    df = pd.DataFrame([{
        "event_kind": "skip_options_executor",
        "decision_context": json.dumps({"lcs": 0.1}),
        "counterfactual_outcome": "not json at all",
    }])
    m = RegretEstimator().fit(df)   # all labels NaN -> stays unfit
    assert not m.is_fitted


# ---------------------------------------------------------------------------
# M.5 — DetectorTrustRouter
# ---------------------------------------------------------------------------

def _detector_outcomes() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    spec = {
        ("SWEEP", "trend"): 0.70,    # sweeps work in trends
        ("SWEEP", "range"): 0.45,
        ("HVN", "trend"): 0.50,
        ("HVN", "range"): 0.65,      # volume nodes work in ranges
    }
    for (factor, regime), rate in spec.items():
        for _ in range(300):
            rows.append({
                "factor": factor, "regime": regime,
                "respected": float(rng.random() < rate),
            })
    return pd.DataFrame(rows)


def test_trust_router_unfit_returns_fallback():
    r = DetectorTrustRouter()
    assert r.trust("SWEEP", "trend") == pytest.approx(0.5)


def test_trust_router_recovers_per_bucket_rates():
    r = DetectorTrustRouter().fit(_detector_outcomes())
    assert r.is_fitted
    assert r.trust("SWEEP", "trend") > r.trust("SWEEP", "range")
    assert r.trust("HVN", "range") > r.trust("HVN", "trend")
    # Sanity: trusts are within (0, 1) and near the generating rates.
    assert 0.6 < r.trust("SWEEP", "trend") < 0.8


def test_trust_router_unknown_bucket_gets_prior_mean():
    r = DetectorTrustRouter().fit(_detector_outcomes())
    unknown = r.trust("FVG", "chop")
    # Should equal the global prior mean (no observations).
    assert 0.4 < unknown < 0.7


def test_trust_router_save_load_round_trip(tmp_path):
    r = DetectorTrustRouter().fit(_detector_outcomes())
    p = tmp_path / "trust.json"
    r.save(p)
    loaded = DetectorTrustRouter.load(p)
    assert loaded.trust("SWEEP", "trend") == pytest.approx(
        r.trust("SWEEP", "trend"), rel=1e-9,
    )


def test_trust_table_emits_all_buckets():
    r = DetectorTrustRouter().fit(_detector_outcomes())
    t = r.trust_table()
    assert len(t) == 4
    assert {"factor", "regime", "n", "raw_rate", "trust"} <= set(t.columns)


# ---------------------------------------------------------------------------
# M.6 — DriftImminentModel
# ---------------------------------------------------------------------------

def _drift_history(n: int = 200) -> pd.DataFrame:
    """Synthetic run history: oos_broad_respect random-walks; when it
    has been falling for a few runs, a drift alert fires soon after."""
    rng = np.random.default_rng(2)
    respect = [0.45]
    for _ in range(n - 1):
        respect.append(respect[-1] + rng.normal(0, 0.01))
    respect = np.clip(respect, 0.25, 0.60)
    fired = np.zeros(n)
    for i in range(4, n):
        slope = respect[i] - respect[i - 3]
        if slope < -0.02 and rng.random() < 0.8:
            fired[i] = 1.0
    return pd.DataFrame({
        "oos_broad_respect": respect,
        "mean_overfit_gap": rng.normal(0.02, 0.005, n),
        "direction_auc": rng.normal(0.57, 0.01, n),
        "direction_top_quartile_confidence": rng.normal(0.58, 0.01, n),
        "drift_fired": fired,
    })


def test_drift_imminent_unfit_returns_fallback():
    m = DriftImminentModel()
    assert m.predict_imminence(_drift_history(10)) == pytest.approx(0.2)


def test_drift_imminent_learns_precursor():
    hist = _drift_history(400)
    m = DriftImminentModel(horizon=3).fit(hist)
    assert m.is_fitted
    # Build two synthetic tails: one with a sharp 3-run fall in
    # respect, one stable. The falling tail should score higher.
    base = {
        "mean_overfit_gap": 0.02, "direction_auc": 0.57,
        "direction_top_quartile_confidence": 0.58, "drift_fired": 0.0,
    }
    falling = pd.DataFrame([
        {**base, "oos_broad_respect": v}
        for v in [0.48, 0.46, 0.43, 0.40]
    ])
    stable = pd.DataFrame([
        {**base, "oos_broad_respect": v}
        for v in [0.48, 0.48, 0.48, 0.48]
    ])
    assert m.predict_imminence(falling) > m.predict_imminence(stable)


def test_drift_labels_nan_at_history_end():
    from liqpool.flywheel.drift_imminent import build_labels
    h = _drift_history(20)
    labels = build_labels(h, horizon=3)
    # The last `horizon` rows can't see their forward window.
    assert np.isnan(labels[-3:]).all()
    assert np.isfinite(labels[:-3]).all()


# ---------------------------------------------------------------------------
# M.7 — ReactionArchetypeModel
# ---------------------------------------------------------------------------

def _reaction_paths(n: int = 300) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Two clearly distinct shape families: V-bounce (down then up)
    and grind-through (steadily down). Context: vol_frac correlates
    with the family so the classifier has signal."""
    rng = np.random.default_rng(3)
    paths, sides, ctx_rows = [], [], []
    for _ in range(n):
        is_bounce = rng.random() < 0.5
        if is_bounce:
            base = np.array([-0.6, -0.9, -0.3, 0.4, 0.9, 1.2])
            vol = rng.uniform(0.002, 0.008)
        else:
            base = np.array([-0.3, -0.7, -1.1, -1.5, -1.8, -2.1])
            vol = rng.uniform(0.012, 0.020)
        paths.append(base + rng.normal(0, 0.1, 6))
        sides.append(1.0)
        ctx_rows.append({
            "distance_atr": rng.uniform(0.5, 3.0), "vol_frac": vol,
            "side_long": 1.0, "session_open": 0.0, "session_close": 0.0,
            "tf_count": 2.0, "score": 1.0,
        })
    return np.array(paths), np.array(sides), pd.DataFrame(ctx_rows)


def test_archetype_clustering_separates_shape_families():
    paths, sides, _ = _reaction_paths()
    m = ReactionArchetypeModel(n_archetypes=2, path_bars=6)
    labels = m.fit_clusters(paths, sides)
    assert m.is_fitted
    assert set(labels) <= {0, 1}
    # The two families should map to different clusters: check that
    # the terminal values of the two centroids differ in sign.
    profiles = m.archetype_profiles()
    terminals = sorted(profiles["terminal_atr"].tolist())
    assert terminals[0] < 0 < terminals[1]


def test_archetype_classifier_predicts_from_context():
    paths, sides, ctx = _reaction_paths()
    m = ReactionArchetypeModel(n_archetypes=2, path_bars=6)
    labels = m.fit_clusters(paths, sides)
    m.fit_classifier(ctx, labels)
    # Identify which cluster is the grind (negative terminal).
    profiles = m.archetype_profiles()
    grind_id = int(profiles.loc[profiles["terminal_atr"].idxmin(),
                                "archetype_id"])
    # High-vol context -> grind family by construction.
    high_vol_probs = m.predict_archetype_probs({
        "distance_atr": 1.5, "vol_frac": 0.018, "side_long": 1.0,
        "session_open": 0.0, "session_close": 0.0,
        "tf_count": 2.0, "score": 1.0,
    })
    low_vol_probs = m.predict_archetype_probs({
        "distance_atr": 1.5, "vol_frac": 0.003, "side_long": 1.0,
        "session_open": 0.0, "session_close": 0.0,
        "tf_count": 2.0, "score": 1.0,
    })
    assert high_vol_probs[grind_id] > low_vol_probs[grind_id]


def test_archetype_unfit_assign_returns_minus_one():
    m = ReactionArchetypeModel()
    assert m.assign(np.zeros(6), 1.0) == -1


def test_archetype_too_little_data_stays_unfit():
    m = ReactionArchetypeModel(n_archetypes=5)
    labels = m.fit_clusters(np.zeros((4, 6)), np.ones(4))
    assert (labels == -1).all()
    assert not m.is_fitted


# ---------------------------------------------------------------------------
# M.8 — BucketAgingModel
# ---------------------------------------------------------------------------

def _aging_history() -> pd.DataFrame:
    """Error grows ~0.01 per log-age unit for proximity; flat for
    avoidance."""
    rng = np.random.default_rng(4)
    rows = []
    for age in range(1, 60):
        rows.append({
            "prediction_type": "proximity", "age_days": age,
            "n_resolved": 50,
            "abs_calib_error": 0.02 + 0.015 * math.log1p(age)
                               + rng.normal(0, 0.003),
        })
        rows.append({
            "prediction_type": "avoidance", "age_days": age,
            "n_resolved": 50,
            "abs_calib_error": 0.03 + rng.normal(0, 0.003),
        })
    return pd.DataFrame(rows)


def test_bucket_aging_unfit_returns_fallback():
    m = BucketAgingModel()
    assert m.expected_error("proximity", 30) == pytest.approx(0.05)


def test_bucket_aging_learns_growth_curve():
    m = BucketAgingModel().fit(_aging_history())
    assert m.is_fitted
    # Proximity error grows with age; avoidance stays ~flat.
    assert m.expected_error("proximity", 50) > m.expected_error("proximity", 5) + 0.01
    flat_delta = abs(m.expected_error("avoidance", 50)
                     - m.expected_error("avoidance", 5))
    assert flat_delta < 0.01


def test_bucket_aging_staleness_multiplier():
    m = BucketAgingModel().fit(_aging_history())
    mult = m.staleness_multiplier("proximity", age_days=50)
    assert mult > 1.2          # 50-day-old proximity bucket is stale
    assert m.staleness_multiplier("avoidance", 50) < 1.2


def test_bucket_aging_filters_thin_cells():
    """Cells with n_resolved < 10 are too noisy to teach aging."""
    thin = _aging_history()
    thin["n_resolved"] = 5
    m = BucketAgingModel().fit(thin)
    assert not m.is_fitted


def test_bucket_aging_save_load(tmp_path):
    m = BucketAgingModel().fit(_aging_history())
    p = tmp_path / "aging.json"
    m.save(p)
    loaded = BucketAgingModel.load(p)
    assert loaded.expected_error("proximity", 30) == pytest.approx(
        m.expected_error("proximity", 30), rel=1e-9,
    )


# ---------------------------------------------------------------------------
# M.9 — CrossAssetTransferMatrix
# ---------------------------------------------------------------------------

def _co_occurrences() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    spec = {
        ("HDFCBANK", "ICICIBANK"): 0.75,   # same sector: transfers
        ("HDFCBANK", "TCS"): 0.45,          # cross sector: doesn't
    }
    for (a, b), rate in spec.items():
        for _ in range(400):
            rows.append({
                "asset_a": a, "asset_b": b,
                "agreed": float(rng.random() < rate),
            })
    return pd.DataFrame(rows)


def test_transfer_unfit_returns_fallback():
    m = CrossAssetTransferMatrix()
    assert m.transfer("HDFCBANK", "TCS") == pytest.approx(0.5)


def test_transfer_same_asset_is_one():
    m = CrossAssetTransferMatrix()
    assert m.transfer("TCS", "TCS") == 1.0


def test_transfer_recovers_pairwise_rates_and_is_symmetric():
    m = CrossAssetTransferMatrix().fit(_co_occurrences())
    assert m.is_fitted
    same_sector = m.transfer("HDFCBANK", "ICICIBANK")
    cross_sector = m.transfer("HDFCBANK", "TCS")
    assert same_sector > cross_sector + 0.1
    # Symmetry by construction.
    assert m.transfer("ICICIBANK", "HDFCBANK") == pytest.approx(same_sector)


def test_transfer_save_load(tmp_path):
    m = CrossAssetTransferMatrix().fit(_co_occurrences())
    p = tmp_path / "transfer.json"
    m.save(p)
    loaded = CrossAssetTransferMatrix.load(p)
    assert loaded.transfer("HDFCBANK", "ICICIBANK") == pytest.approx(
        m.transfer("HDFCBANK", "ICICIBANK"), rel=1e-9,
    )


# ---------------------------------------------------------------------------
# M.10 — BriefConfidenceMetaCalibrator
# ---------------------------------------------------------------------------

def _outcome_log_joined(overconfident: bool) -> pd.DataFrame:
    rng = np.random.default_rng(6)
    n = 300
    if overconfident:
        # Brief said ~0.85; reality was ~0.60.
        p = rng.uniform(0.80, 0.90, n)
        y = rng.binomial(1, 0.60, n).astype(float)
    else:
        p = rng.uniform(0.2, 0.8, n)
        y = rng.binomial(1, p).astype(float)
    return pd.DataFrame({
        "prediction_type": ["proximity"] * n,
        "predicted_value": p,
        "outcome_boolean": y,
    })


def test_meta_calibrator_unfit_passes_through():
    m = BriefConfidenceMetaCalibrator()
    assert m.adjust("proximity", 0.62) == pytest.approx(0.62)
    assert m.applied_temperature("proximity") == 1.0


def test_meta_calibrator_softens_overconfident_type():
    m = BriefConfidenceMetaCalibrator().fit(_outcome_log_joined(True))
    assert m.is_fitted
    # Overconfident -> T > 1 -> high probabilities get pulled down.
    assert m.applied_temperature("proximity") > 1.0
    assert m.adjust("proximity", 0.85) < 0.85


def test_meta_calibrator_near_identity_when_calibrated():
    m = BriefConfidenceMetaCalibrator().fit(_outcome_log_joined(False))
    t = m.applied_temperature("proximity")
    assert 0.7 < t < 1.4
    adj = m.adjust("proximity", 0.62)
    assert abs(adj - 0.62) < 0.10


def test_meta_calibrator_skips_thin_types():
    df = _outcome_log_joined(True).head(10)   # below MIN_RESOLVED_PER_TYPE
    m = BriefConfidenceMetaCalibrator().fit(df)
    assert not m.is_fitted


def test_meta_calibrator_temperature_is_bounded():
    """Even a catastrophic week can't push the correction beyond the
    [0.5, 3.0] bounds — the brief's probabilities can be softened
    but never inverted or flattened to coin flips."""
    rng = np.random.default_rng(7)
    n = 200
    # Maximal miscalibration: said 0.95, reality 0.05.
    df = pd.DataFrame({
        "prediction_type": ["proximity"] * n,
        "predicted_value": rng.uniform(0.93, 0.97, n),
        "outcome_boolean": rng.binomial(1, 0.05, n).astype(float),
    })
    m = BriefConfidenceMetaCalibrator().fit(df)
    assert m.applied_temperature("proximity") <= 3.0


def test_meta_calibrator_save_load(tmp_path):
    m = BriefConfidenceMetaCalibrator().fit(_outcome_log_joined(True))
    p = tmp_path / "meta.json"
    m.save(p)
    loaded = BriefConfidenceMetaCalibrator.load(p)
    assert loaded.applied_temperature("proximity") == pytest.approx(
        m.applied_temperature("proximity"), rel=1e-9,
    )
    assert loaded.adjust("proximity", 0.85) == pytest.approx(
        m.adjust("proximity", 0.85), rel=1e-6,
    )
