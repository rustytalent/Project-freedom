"""Phase 3B + 3C — tests for the per-pool learned dynamic gate.

These tests pin the discipline contract:

1. Target derivation is the closed-form weight that minimises blended
   squared error and is masked out when the two heads agree.
2. Feature frame schema is stable: same columns, same order, in the same
   shape every call — the model's predict path depends on this.
3. Gate clipping respects ``[GATE_WEIGHT_FLOOR, GATE_WEIGHT_CEILING]``.
4. Sparse / degenerate / signal-less audits return ``None`` (the
   fallback contract the orchestrator relies on for Phase 3E safety).
5. A clean signal-bearing synthetic audit actually trains, and the
   re-blended ``learned_blended_q`` has a lower Brier than the static
   blend the upstream MoE produced. That's the only real test of value —
   the gate must be useful, not just trainable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liqpool.learned_gate import (
    DISAGREEMENT_FLOOR,
    GATE_FEATURE_NAMES,
    GATE_WEIGHT_CEILING,
    GATE_WEIGHT_FLOOR,
    LearnedGateModel,
    build_gate_feature_frame,
    derive_target_weight,
)


pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")


def _signal_audit(seed: int, n: int = 400) -> pd.DataFrame:
    """Synthetic audit with a learnable signal: when global and sector
    expert disagree more, the sector expert is more trustworthy."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.3, 0.7, n)
    d = rng.uniform(-0.3, 0.3, n)
    s = np.clip(g + d, 0.02, 0.98)
    w = np.clip(0.20 + 1.5 * np.abs(d), GATE_WEIGHT_FLOOR, GATE_WEIGHT_CEILING)
    truth = w * s + (1 - w) * g
    a = (rng.uniform(0, 1, n) < truth).astype(int)
    return pd.DataFrame({
        "asset": ["A"] * (n // 2) + ["B"] * (n - n // 2),
        "sector": ["X"] * (n // 2) + ["Y"] * (n - n // 2),
        "global_q": g,
        "sector_q": s,
        "actual": a.astype(float),
        "pool_score": rng.uniform(0.5, 0.9, n),
        "pool_width": rng.uniform(1.0, 5.0, n),
        "n_tfs": rng.integers(1, 4, n).astype(float),
        "distance_atr": rng.uniform(0, 8, n),
        "available_at": pd.date_range("2025-01-01", periods=n, freq="1h"),
    })


def _side_channels():
    return (
        {"X": {"score": 0.15, "conviction": 0.40}, "Y": {"score": -0.10, "conviction": 0.20}},
        {"X": 0.48, "Y": 0.55},
        {"A": 0.50, "B": 0.55},
    )


def test_derive_target_weight_known_cases():
    actual = np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=float)
    g = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
    # Strong sector right, strong sector wrong, ambiguous, degenerate, degenerate
    s = np.array([0.8, 0.8, 0.6, 0.5005, 0.5005], dtype=float)
    target, mask = derive_target_weight(actual, g, s,
                                        disagreement_floor=DISAGREEMENT_FLOOR)
    # Row 0: w* = (1-0.5)/(0.8-0.5) = 1.67 -> clipped to 1.0
    assert mask[0] and target[0] == pytest.approx(1.0)
    # Row 1: w* = (0-0.5)/(0.8-0.5) = -1.67 -> clipped to 0.0
    assert mask[1] and target[1] == pytest.approx(0.0)
    # Row 2: w* = (1-0.5)/(0.6-0.5) = 5.0 -> clipped to 1.0
    assert mask[2] and target[2] == pytest.approx(1.0)
    # Rows 3,4: |s - g| = 0.0005 < floor 0.01 -> masked out
    assert not mask[3] and not mask[4]
    assert np.isnan(target[3]) and np.isnan(target[4])


def test_feature_frame_schema_is_stable():
    audit = _signal_audit(seed=11, n=80)
    regime, strict, reliability = _side_channels()
    feats = build_gate_feature_frame(audit, regime, strict, reliability)
    assert list(feats.columns) == list(GATE_FEATURE_NAMES), \
        "feature column order must match GATE_FEATURE_NAMES exactly"
    assert len(feats) == len(audit)
    # Missing sector / asset keys fall back to neutral values without crashing.
    audit_with_unknowns = audit.copy()
    audit_with_unknowns["sector"] = "MARS"
    audit_with_unknowns["asset"] = "UNKNOWN"
    feats_unknown = build_gate_feature_frame(audit_with_unknowns, regime, strict, reliability)
    assert (feats_unknown["sector_regime_score"] == 0.0).all()
    assert (feats_unknown["sector_oos_strict"] == 0.5).all()
    assert (feats_unknown["asset_reliability"] == 0.5).all()


def test_feature_frame_raises_on_missing_required_columns():
    audit = _signal_audit(seed=12, n=80).drop(columns=["pool_width"])
    regime, strict, reliability = _side_channels()
    with pytest.raises(ValueError, match="pool_width"):
        build_gate_feature_frame(audit, regime, strict, reliability)


def test_signal_audit_trains_and_beats_static_blend():
    audit = _signal_audit(seed=2, n=400)
    regime, strict, reliability = _side_channels()
    gate = LearnedGateModel.fit_from_audit(audit, regime, strict, reliability, seed=42)
    assert gate is not None, "clean signal should train"
    assert gate.stats.status == "trained"
    assert gate.stats.mae_improvement > 0.008
    assert gate.stats.shuffle_margin > 0.010
    applied = gate.apply(audit)
    assert "learned_gate_weight" in applied.columns
    assert "learned_blended_q" in applied.columns
    w = applied["learned_gate_weight"].to_numpy()
    assert (w >= GATE_WEIGHT_FLOOR - 1e-9).all()
    assert (w <= GATE_WEIGHT_CEILING + 1e-9).all()
    # The whole point: learned blend beats the static 70/30 default the MoE uses.
    static_blend = 0.70 * audit["sector_q"] + 0.30 * audit["global_q"]
    learned_brier = float(np.mean((applied["learned_blended_q"] - audit["actual"]) ** 2))
    static_brier = float(np.mean((static_blend - audit["actual"]) ** 2))
    assert learned_brier < static_brier, (
        f"learned blend Brier {learned_brier:.4f} should beat static 70/30 {static_brier:.4f}"
    )


def test_tiny_audit_falls_back():
    audit = _signal_audit(seed=3, n=20)
    regime, strict, reliability = _side_channels()
    gate = LearnedGateModel.fit_from_audit(audit, regime, strict, reliability, seed=42)
    assert gate is None, "<min_decisive rows must fall back to static (Phase 3E)"


def test_zero_disagreement_audit_falls_back():
    audit = _signal_audit(seed=4, n=200)
    # Force sector and global to be identical -> no disagreement signal at all.
    audit["sector_q"] = audit["global_q"]
    regime, strict, reliability = _side_channels()
    gate = LearnedGateModel.fit_from_audit(audit, regime, strict, reliability, seed=42)
    assert gate is None


def test_missing_sector_expert_routes_to_global():
    """Rows where sector_q is NaN — no trained expert for that sector —
    must end up with learned_gate_weight==0 so blended_q == global_q."""
    audit = _signal_audit(seed=5, n=400)
    regime, strict, reliability = _side_channels()
    gate = LearnedGateModel.fit_from_audit(audit, regime, strict, reliability, seed=42)
    assert gate is not None
    no_expert = audit.copy()
    no_expert.loc[no_expert.index[:50], "sector_q"] = np.nan
    applied = gate.apply(no_expert)
    weights = applied["learned_gate_weight"].to_numpy()
    assert (weights[:50] == 0.0).all(), "rows without sector_q must get weight 0"
    blended = applied["learned_blended_q"].to_numpy()
    global_q = applied["global_q"].to_numpy()
    np.testing.assert_allclose(
        blended[:50], np.clip(global_q[:50], 0.02, 0.98),
        err_msg="learned_blended_q must fall back to global_q when no expert exists",
    )


def test_determinism_same_seed_same_predictions():
    """Same seed + same data must produce identical weights. Uses the seed
    pair the signal-training test already proved fits cleanly so a flaky
    fluke-fit can't masquerade as non-determinism."""
    audit = _signal_audit(seed=2, n=400)
    regime, strict, reliability = _side_channels()
    g1 = LearnedGateModel.fit_from_audit(audit, regime, strict, reliability, seed=42)
    g2 = LearnedGateModel.fit_from_audit(audit, regime, strict, reliability, seed=42)
    assert g1 is not None and g2 is not None
    w1 = g1.predict_weights(audit)
    w2 = g2.predict_weights(audit)
    np.testing.assert_allclose(w1, w2)


def test_feature_importance_returns_known_names():
    audit = _signal_audit(seed=9, n=400)
    regime, strict, reliability = _side_channels()
    gate = LearnedGateModel.fit_from_audit(audit, regime, strict, reliability, seed=42)
    assert gate is not None
    fi = gate.feature_importance(top_k=5)
    assert 1 <= len(fi) <= 5
    names = {row["feature"] for row in fi}
    assert names.issubset(set(GATE_FEATURE_NAMES))


def test_blend_one_matches_apply_for_single_row():
    """Phase 3D wiring: ``blend_one`` is the per-candidate live API. The
    weight + blended q it returns for one row must match what ``apply``
    produces for the same row inside a frame."""
    audit = _signal_audit(seed=2, n=400)
    regime, strict, reliability = _side_channels()
    gate = LearnedGateModel.fit_from_audit(audit, regime, strict, reliability, seed=42)
    assert gate is not None
    sample = audit.iloc[3]
    applied = gate.apply(audit.iloc[[3]])
    expected_w = float(applied["learned_gate_weight"].iloc[0])
    expected_q = float(applied["learned_blended_q"].iloc[0])
    actual_w, actual_q = gate.blend_one(
        global_q=float(sample["global_q"]),
        sector_q=float(sample["sector_q"]),
        sector=str(sample["sector"]),
        asset=str(sample["asset"]),
        pool_score=float(sample["pool_score"]),
        pool_width=float(sample["pool_width"]),
        n_tfs=int(sample["n_tfs"]),
        distance_atr=float(sample["distance_atr"]),
    )
    assert actual_w == pytest.approx(expected_w)
    assert actual_q == pytest.approx(expected_q)


def test_blend_one_falls_back_to_global_when_no_expert():
    audit = _signal_audit(seed=2, n=400)
    regime, strict, reliability = _side_channels()
    gate = LearnedGateModel.fit_from_audit(audit, regime, strict, reliability, seed=42)
    assert gate is not None
    w, q = gate.blend_one(
        global_q=0.62, sector_q=None,           # no trained expert
        sector="X", asset="A",
        pool_score=0.7, pool_width=2.0, n_tfs=2, distance_atr=1.5,
    )
    assert w == 0.0
    assert q == pytest.approx(0.62)


def test_orchestrator_attach_path_train_and_fallback():
    """``multi_asset._fit_and_apply_learned_gate`` is the orchestrator entry
    point. With a signal-bearing audit it must attach the gate to the
    report AND augment the audit table; with a degenerate audit it must
    leave the report's gate field as None (Phase 3E fallback)."""
    from liqpool.multi_asset import (
        MultiAssetReport, OOSPredictionAudit, QualityModelAuditMetrics,
        _fit_and_apply_learned_gate,
    )

    class _StubMl:
        pass

    class _StubPool:
        def __init__(self, asset):
            self.asset = asset

    class _StubResult:
        def __init__(self, outcome="respected_strong"):
            self.outcome = outcome
            self.is_tested = True
            self.is_respect = outcome in ("respected_strong", "swept_and_reclaimed")

    pools = [_StubPool("A")] * 100 + [_StubPool("B")] * 100
    results = ([_StubResult("respected_strong")] * 70
               + [_StubResult("broken_strong")] * 30
               + [_StubResult("respected_strong")] * 60
               + [_StubResult("broken_strong")] * 40)
    dates = pd.date_range("2024-01-01", periods=120, freq="1D")
    asset_dfs = {
        "A": pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9,
                           "close": np.linspace(1, 1.1, len(dates))}, index=dates),
        "B": pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9,
                           "close": np.linspace(1, 1.05, len(dates))}, index=dates),
    }

    # Train case
    audit_df = _signal_audit(seed=2, n=400)
    audit_df["pool_index"] = np.arange(len(audit_df))
    audit_df["outcome"] = "respected_strong"
    audit_df["is_decisive"] = True
    audit_df["pool_mid"] = 100.0
    audit_df["distance_bucket"] = "1-3 ATR"
    audit_df["fallback_reason"] = ""
    audit_df["gate_weight"] = 0.5
    audit_df["blended_q"] = 0.5 * audit_df["sector_q"] + 0.5 * audit_df["global_q"]
    report = MultiAssetReport()
    report.pooled_oos_respect = 0.55
    report.unified_oos_audit = OOSPredictionAudit(
        table=audit_df,
        metrics={"blended": QualityModelAuditMetrics("blended", 400, 0.25, 0.7,
                                                    0.55, 0.55, 0.5, 0.5)},
    )
    _fit_and_apply_learned_gate(report, _StubMl(), pools, results, asset_dfs, seed=42)
    assert report.unified_learned_gate is not None
    assert "learned_blended_q" in report.unified_oos_audit.table.columns
    assert "learned_blended" in report.unified_oos_audit.metrics

    # Fallback case — too few rows. Build a fresh audit (not derived from
    # the mutated one above) so the assertion that the column was never
    # added is meaningful.
    fresh = _signal_audit(seed=11, n=20)
    fresh["pool_index"] = np.arange(len(fresh))
    fresh["outcome"] = "respected_strong"
    fresh["is_decisive"] = True
    fresh["pool_mid"] = 100.0
    fresh["distance_bucket"] = "1-3 ATR"
    fresh["fallback_reason"] = ""
    fresh["gate_weight"] = 0.5
    fresh["blended_q"] = 0.5 * fresh["sector_q"] + 0.5 * fresh["global_q"]
    report2 = MultiAssetReport()
    report2.pooled_oos_respect = 0.55
    report2.unified_oos_audit = OOSPredictionAudit(table=fresh, metrics={})
    _fit_and_apply_learned_gate(report2, _StubMl(), pools, results, asset_dfs, seed=42)
    assert report2.unified_learned_gate is None
    assert "learned_blended_q" not in report2.unified_oos_audit.table.columns


def test_stats_record_is_serialisable():
    audit = _signal_audit(seed=2, n=400)
    regime, strict, reliability = _side_channels()
    gate = LearnedGateModel.fit_from_audit(audit, regime, strict, reliability, seed=42)
    assert gate is not None
    d = gate.stats.to_dict()
    import json
    json.dumps(d)
    assert d["status"] == "trained"
    assert d["n_train"] > 0 and d["n_val"] > 0
    assert "shuffle_margin" in d
