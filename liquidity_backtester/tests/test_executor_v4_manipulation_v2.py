"""ManipulationV2 — seven detectors + Bayesian fusion + AUTHORITATIVE gate.

Founder 2026-06-22 Tier-3: this is the 95%+ engine. These tests verify
the building blocks, the fusion math, the operator override, the
self-calibration, and the end-to-end AUTHORITATIVE direction gate in
the aggregator.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PersistenceConfig,
    PortfolioManagerConfig,
    V4Runner,
    V4RunnerConfig,
)
from liqpool.research.belief.executor_v4.aggregator import (
    AggregatorConfig,
    DecisionAggregator,
)
from liqpool.research.belief.executor_v4.manipulation_v2 import (
    BayesianFusion,
    CrossRailAsymmetryDetector,
    DodSignatureDetector,
    AbnormalAcceptanceDetector,
    DetectorPosterior,
    DetectorMagnitude,
    ManipulationEngineConfig,
    ManipulationEngineV2,
    MMGammaProxyDetector,
    MMIntent,
    OutcomeCalibrator,
    PinRiskDetector,
    SweepDetector,
    TrendVsRangeDetector,
)

from tests.test_executor_v4_dead_market_and_web_viz import _snap


# ── DetectorPosterior dataclass ─────────────────────────────────


def test_posterior_quiet_factory_is_silent():
    p = DetectorPosterior.quiet()
    assert p.fire is False
    assert p.probability == 0.0
    assert p.direction == 0


def test_posterior_to_dict_round_trips_magnitude():
    p = DetectorPosterior(
        fire=True, probability=0.7, direction=+1, confidence=0.6,
        magnitude=DetectorMagnitude(z_score=2.5, raw_value=10.0,
                                       units="x"),
        evidence=["a"],
    )
    d = p.to_dict()
    assert d["magnitude"]["z_score"] == 2.5
    assert d["direction"] == 1
    assert d["evidence"] == ["a"]


# ── CrossRailAsymmetryDetector ──────────────────────────────────


def test_cross_rail_asymmetry_fires_on_persistent_imbalance():
    det = CrossRailAsymmetryDetector(persistence_bars=4)
    # Build a snapshot with strong positive asymmetry (ce_z high,
    # pe_z also high i.e. both rails biased bullish).
    snap_template = lambda ce_z, pe_z, ce_disp=0.20, pe_disp=0.20: {
        "battlefield": {
            "ce_rail": {"weighted_mean_signed_z": ce_z,
                          "dispersion_score": ce_disp},
            "pe_rail": {"weighted_mean_signed_z": pe_z,
                          "dispersion_score": pe_disp},
        },
        "slot_readings": [],
    }
    # Need to seed history with mixed values first so the std is non-zero.
    for i in range(8):
        det.observe(
            snapshot=snap_template(0.2 - 0.05 * i, -0.2 + 0.04 * i),
            bar_index=i)
    # Then a persistent asymmetric stretch.
    out = None
    for i in range(8, 16):
        out = det.observe(snapshot=snap_template(1.8, 1.5), bar_index=i)
    assert out is not None
    assert out.fire
    assert out.direction == 1
    assert out.confidence > 0.40


def test_cross_rail_quiet_when_dispersion_high():
    det = CrossRailAsymmetryDetector(
        persistence_bars=3, dispersion_ceiling=0.30)
    for i in range(10):
        det.observe(
            snapshot={
                "battlefield": {
                    "ce_rail": {"weighted_mean_signed_z": 2.0,
                                  "dispersion_score": 0.80},
                    "pe_rail": {"weighted_mean_signed_z": 2.0,
                                  "dispersion_score": 0.80},
                },
            },
            bar_index=i)
    out = det.observe(
        snapshot={
            "battlefield": {
                "ce_rail": {"weighted_mean_signed_z": 2.0,
                              "dispersion_score": 0.80},
                "pe_rail": {"weighted_mean_signed_z": 2.0,
                              "dispersion_score": 0.80},
            },
        },
        bar_index=10)
    assert out.fire is False


# ── DodSignatureDetector ───────────────────────────────────────


def _slots_wall_on_ce() -> List[Dict[str, Any]]:
    out = []
    for level in range(-5, 6):
        z = 2.5 if 1 <= level <= 4 else 0.1
        out.append({"strike": 23000 + 50 * level, "option_type": "CE",
                     "level": level, "dod_z": z, "mark_price": 80.0})
        out.append({"strike": 23000 + 50 * level, "option_type": "PE",
                     "level": level, "dod_z": 0.0, "mark_price": 60.0})
    return out


def test_dod_signature_detects_wall_and_signals_bearish():
    det = DodSignatureDetector(z_wall=1.5)
    out = det.observe(snapshot={"slot_readings": _slots_wall_on_ce()},
                        bar_index=10)
    assert out.fire
    # Wall on CE rail = ceiling = expect spot DOWN.
    assert out.direction == -1
    assert "wall_ce" in out.classification


def test_dod_signature_detects_single_strike_extreme():
    slots = [
        {"strike": 23000 + 50 * level, "option_type": "CE",
          "level": level, "dod_z": 0.1, "mark_price": 80.0}
        for level in range(-5, 6)
    ] + [
        {"strike": 23000 + 50 * level, "option_type": "PE",
          "level": level, "dod_z": 0.0, "mark_price": 60.0}
        for level in range(-5, 6)
    ]
    # Single extreme dod_z on CE level=3 with quiet neighbors.
    for s in slots:
        if s["option_type"] == "CE" and s["level"] == 3:
            s["dod_z"] = 3.0
    det = DodSignatureDetector(z_extreme=2.0)
    out = det.observe(snapshot={"slot_readings": slots}, bar_index=10)
    assert out.fire
    assert "single_strike_ce" in out.classification


def test_dod_signature_rail_tilt_after_persistence():
    det = DodSignatureDetector(rail_tilt_bars=4, rail_tilt_threshold=0.6)
    # CE rail persistently above baseline.
    slots = [
        {"strike": 23000 + 50 * level, "option_type": "CE",
          "level": level, "dod_z": 0.8, "mark_price": 80.0}
        for level in range(-5, 6)
    ] + [
        {"strike": 23000 + 50 * level, "option_type": "PE",
          "level": level, "dod_z": 0.0, "mark_price": 60.0}
        for level in range(-5, 6)
    ]
    out = None
    for i in range(8):
        out = det.observe(snapshot={"slot_readings": slots},
                            bar_index=i)
    assert out is not None
    assert out.fire
    assert "rail_tilt_ce" in out.classification


# ── AbnormalAcceptanceDetector ─────────────────────────────────


def test_abnormal_acceptance_fires_on_defended_ce_cluster():
    det = AbnormalAcceptanceDetector(defended_threshold=3)
    slots = []
    for level in range(-5, 6):
        slots.append({
            "strike": 23000 + 50 * level, "option_type": "CE",
            "acceptance": "defended" if 0 <= level <= 3 else "normal",
        })
        slots.append({
            "strike": 23000 + 50 * level, "option_type": "PE",
            "acceptance": "normal",
        })
    out = det.observe(snapshot={"slot_readings": slots}, bar_index=10)
    assert out.fire
    assert out.direction == -1     # CE defended → expect DOWN
    assert out.classification == "defended_ce_cluster"


# ── PinRiskDetector ────────────────────────────────────────────


def test_pin_risk_requires_bars_to_expiry():
    det = PinRiskDetector()
    out = det.observe(
        snapshot={
            "spot": 23000.0,
            "slot_readings": [{"strike": 23000, "option_type": "CE"}],
        },
        bar_index=10)
    assert out.fire is False


def test_pin_risk_fires_near_expiry_and_atm():
    det = PinRiskDetector(proximity_pct=0.005, max_bars_to_expiry=60)
    slots = [
        {"strike": 23000 + 50 * level, "option_type": "CE",
          "level": level, "dod_z": 1.5 if level == 0 else 0.1,
          "mark_price": 80.0}
        for level in range(-3, 4)
    ]
    out = det.observe(
        snapshot={
            "spot": 23000.0, "bars_to_expiry": 8,
            "slot_readings": slots,
            "battlefield": {"ce_rail": {"dispersion_score": 0.10},
                              "pe_rail": {"dispersion_score": 0.10}},
        },
        bar_index=10)
    assert out.fire
    assert out.classification == "pin_risk"


# ── SweepDetector ──────────────────────────────────────────────


def test_sweep_classifies_stop_hunt_on_reversal():
    det = SweepDetector(reversal_within_bars=4,
                          follow_through_window=6)
    # 30 ticks of background.
    for i in range(30):
        det.observe(
            snapshot={"spot": 23000.0 + 0.2 * i, "sweep_state": {},
                       "iv_state": {}},
            bar_index=i)
    # Sweep burst up: spot jumps, intent direction positive.
    det.observe(
        snapshot={"spot": 23080.0,
                   "sweep_state": {"state_index": 1},
                   "iv_state": {"direction": 1, "net_intent_z": 0.5}},
        bar_index=30)
    # Then reversal back down without sustained intent.
    out = det.observe(
        snapshot={"spot": 23010.0,
                   "sweep_state": {},
                   "iv_state": {"direction": 0, "net_intent_z": 0.1}},
        bar_index=32)
    assert out.fire
    assert out.classification == "stop_hunt"
    assert out.direction == -1


# ── MMGammaProxyDetector ───────────────────────────────────────


def test_mm_gamma_proxy_classifies_long_suppression():
    det = MMGammaProxyDetector(dispersion_low=0.20)

    class Rich:
        epicenter_migration_distance = 0.5
        net_intent_velocity = 0.2
        dispersion_velocity = -0.20
    out = det.observe(
        snapshot={"battlefield": {
            "ce_rail": {"dispersion_score": 0.10},
            "pe_rail": {"dispersion_score": 0.12},
        }},
        rich_context=Rich(),
        bar_index=10)
    assert out.fire
    assert out.classification == "long_gamma_suppression"


def test_mm_gamma_proxy_classifies_short_amplification():
    det = MMGammaProxyDetector(dispersion_high=0.40,
                                  intent_velocity_high=0.8)

    class Rich:
        epicenter_migration_distance = 3.0
        net_intent_velocity = 1.5
        dispersion_velocity = 0.30
    out = det.observe(
        snapshot={"battlefield": {
            "ce_rail": {"dispersion_score": 0.55},
            "pe_rail": {"dispersion_score": 0.55},
        }},
        rich_context=Rich(),
        bar_index=10)
    assert out.fire
    assert out.classification == "short_gamma_amplification"


# ── TrendVsRangeDetector ───────────────────────────────────────


def test_trend_vs_range_classifies_trending_bull():
    det = TrendVsRangeDetector(consensus_floor=0.20,
                                  persistence_bars=4)

    class Web:
        directional_consensus_horizon_weighted = 0.55

    class Rich:
        thesis_velocity_dominant_side = "bull"
        regime_stability_index = 0.6
        dispersion_velocity = 0.05
    out = None
    for i in range(6):
        out = det.observe(snapshot={}, rich_context=Rich(),
                            web_snapshot=Web(), bar_index=i)
    assert out is not None
    assert out.fire
    assert out.classification == "trending_bull"
    assert out.direction == 1


def test_trend_vs_range_classifies_ranging_when_stable():
    det = TrendVsRangeDetector(persistence_bars=3)

    class Web:
        directional_consensus_horizon_weighted = 0.05

    class Rich:
        thesis_velocity_dominant_side = "flat"
        regime_stability_index = 0.85
        dispersion_velocity = 0.01
    out = None
    for i in range(6):
        out = det.observe(snapshot={}, rich_context=Rich(),
                            web_snapshot=Web(), bar_index=i)
    assert out is not None
    assert out.fire
    assert out.classification == "ranging"
    assert out.direction == 0


# ── Fusion ────────────────────────────────────────────────────


def _post(direction=0, fire=False, confidence=0.0, probability=0.0,
            classification="", horizon=5):
    return DetectorPosterior(
        fire=fire, probability=probability, direction=direction,
        confidence=confidence, horizon_bars=horizon,
        classification=classification)


def test_fusion_combines_aligned_detectors_into_strong_direction():
    fusion = BayesianFusion()
    posteriors = {
        "sweep": _post(direction=1, fire=True, confidence=0.7,
                          probability=0.7),
        "cross_rail_asymmetry": _post(
            direction=1, fire=True, confidence=0.65, probability=0.65),
        "dod_signature": _post(
            direction=1, fire=True, confidence=0.55, probability=0.55,
            classification="wall_pe"),
        "abnormal_acceptance": DetectorPosterior.quiet(),
        "pin_risk": DetectorPosterior.quiet(),
        "mm_gamma_proxy": _post(
            direction=0, fire=True, confidence=0.5,
            classification="short_gamma_amplification"),
        "trend_vs_range": _post(
            direction=1, fire=True, confidence=0.7,
            classification="trending_bull"),
    }
    weights = {n: 1.0 for n in posteriors}
    intent = fusion.fuse(posteriors=posteriors, weights=weights)
    assert intent.direction == 1
    assert intent.confidence > 0.50
    assert intent.regime == "trending_bull"
    assert intent.gamma_regime == "short_gamma_amplification"


def test_fusion_flips_dod_rail_tilt_against_trend():
    """Founder's 'in a trend, cheap is a trap' rule. dod_signature says
    direction=+1 (cheap PE rail looks bullish) but trend says BEAR
    strongly — fusion should flip the dod contribution."""
    fusion = BayesianFusion()
    posteriors = {
        "sweep": DetectorPosterior.quiet(),
        "cross_rail_asymmetry": DetectorPosterior.quiet(),
        "dod_signature": _post(
            direction=1, fire=True, confidence=0.7, probability=0.7,
            classification="rail_tilt_pe_cheap"),
        "abnormal_acceptance": DetectorPosterior.quiet(),
        "pin_risk": DetectorPosterior.quiet(),
        "mm_gamma_proxy": DetectorPosterior.quiet(),
        "trend_vs_range": _post(
            direction=-1, fire=True, confidence=0.75,
            classification="trending_bear"),
    }
    weights = {n: 1.0 for n in posteriors}
    intent = fusion.fuse(posteriors=posteriors, weights=weights)
    # The flip + trend bias should drive intent BEAR overall.
    assert intent.direction == -1
    assert any("flipped" in n for n in intent.notes)


def test_fusion_quiet_when_no_detectors_fire():
    fusion = BayesianFusion()
    posteriors = {n: DetectorPosterior.quiet() for n in (
        "sweep", "cross_rail_asymmetry", "dod_signature",
        "abnormal_acceptance", "pin_risk", "mm_gamma_proxy",
        "trend_vs_range")}
    intent = fusion.fuse(posteriors=posteriors,
                            weights={n: 1.0 for n in posteriors})
    assert intent.direction == 0
    assert intent.confidence == 0.0


# ── Self-calibrator ────────────────────────────────────────────


def test_calibrator_updates_weights_after_enough_fires():
    calib = OutcomeCalibrator(
        detector_names=["good", "bad"],
        learning_rate=0.20,
        min_fires_before_learning=4,
    )
    good_post = _post(direction=1, fire=True, confidence=0.8,
                        probability=0.8)
    bad_post = _post(direction=-1, fire=True, confidence=0.8,
                       probability=0.8)
    for i in range(20):
        calib.snapshot_at_entry(
            position_id=f"p{i}", bar_index=i,
            intent_dict={},
            per_detector={"good": good_post, "bad": bad_post},
        )
        # Trade was profitable in DIR=+1, matching "good".
        calib.attribute_close(position_id=f"p{i}", realised_r=+0.5)
    weights = calib.weights
    assert weights["good"] > weights["bad"]
    assert weights["good"] > 1.0
    assert weights["bad"] < 1.0


# ── Engine + operator override ─────────────────────────────────


def test_engine_observe_runs_all_detectors():
    e = ManipulationEngineV2()
    for i in range(15):
        intent = e.observe(snapshot=_snap(100 + i), bar_index=100 + i)
    assert isinstance(intent, MMIntent)
    assert len(intent.per_detector) == 7


def test_engine_override_makes_intent_authoritative():
    e = ManipulationEngineV2()
    e.observe(snapshot=_snap(100), bar_index=100)
    e.set_operator_override(
        direction=-1, horizon_bars=10, reason="test", confidence=0.85)
    intent = e.observe(snapshot=_snap(101), bar_index=101)
    assert intent.direction == -1
    assert intent.operator_override is not None
    assert intent.confidence >= 0.85
    assert any("override" in n.lower() for n in intent.notes)


def test_engine_override_auto_decays():
    e = ManipulationEngineV2()
    e.observe(snapshot=_snap(100), bar_index=100)
    e.set_operator_override(direction=+1, horizon_bars=3,
                                reason="x")
    # After bar 103, override should have decayed.
    e.observe(snapshot=_snap(101), bar_index=101)
    e.observe(snapshot=_snap(102), bar_index=102)
    e.observe(snapshot=_snap(103), bar_index=103)
    intent_after = e.observe(snapshot=_snap(104), bar_index=104)
    assert intent_after.operator_override is None


def test_engine_clear_override():
    e = ManipulationEngineV2()
    e.observe(snapshot=_snap(100), bar_index=100)
    e.set_operator_override(direction=-1, horizon_bars=10, reason="x")
    assert e.is_overridden() is True
    e.clear_operator_override()
    assert e.is_overridden() is False


# ── Aggregator AUTHORITATIVE direction gate ────────────────────


class _FakeAggregatorInputs:
    """Minimum inputs to call DecisionAggregator.decide()."""
    @staticmethod
    def base_call_kwargs(proposed_direction: int = 1) -> Dict[str, Any]:
        return dict(
            base_confidence=0.70,
            mtf_alignment={"alignment_score": 0.7, "alignment_ok": True,
                            "l1_match": True, "l5_match": True,
                            "l15_match": True, "l60_match": True},
            projection=None,
            critique=None,
            ev_decision=None,
            web_snapshot=None,
            proposed_strategy_class="long_ce" if proposed_direction > 0 else "long_pe",
            proposed_direction=proposed_direction,
            open_positions_count=0,
            max_open_positions=3,
            daily_pnl_rupees=0.0,
            max_daily_bleed=5000.0,
            rehearsal_decision=None,
        )


def test_aggregator_refuses_when_mm_intent_contradicts_high_confidence():
    cfg = AggregatorConfig(mm_intent_contra_refuse_confidence=0.55)
    agg = DecisionAggregator(cfg)
    kwargs = _FakeAggregatorInputs.base_call_kwargs(proposed_direction=1)
    mm_intent = {"direction": -1, "confidence": 0.75,
                  "regime": "trending_bear", "gamma_regime": "long_gamma",
                  "fire_count": 3}
    d = agg.decide(**kwargs, mm_intent=mm_intent)
    assert d.decision == "REFUSE"
    assert any("manipulation_v2" in r for r in d.refuse_reasons)


def test_aggregator_haircuts_when_mm_intent_partial_contra():
    cfg = AggregatorConfig(
        mm_intent_contra_refuse_confidence=0.70,
        mm_intent_partial_contra_size_haircut=0.50,
    )
    agg = DecisionAggregator(cfg)
    kwargs = _FakeAggregatorInputs.base_call_kwargs(proposed_direction=1)
    mm_intent = {"direction": -1, "confidence": 0.50,
                  "regime": "transitioning", "gamma_regime": "unknown",
                  "fire_count": 1}
    d = agg.decide(**kwargs, mm_intent=mm_intent)
    # Not refused (under contra_refuse threshold).
    assert d.decision != "REFUSE" or "manipulation_v2" not in " ".join(
        d.refuse_reasons)
    # Size multiplier is reduced.
    assert d.mm_intent_size_multiplier == pytest.approx(0.50, abs=0.01)


def test_aggregator_boosts_size_when_mm_intent_aligned():
    cfg = AggregatorConfig(mm_intent_aligned_size_boost_max=1.80)
    agg = DecisionAggregator(cfg)
    kwargs = _FakeAggregatorInputs.base_call_kwargs(proposed_direction=1)
    mm_intent = {"direction": 1, "confidence": 0.85,
                  "regime": "trending_bull", "gamma_regime": "short_gamma",
                  "fire_count": 4, "targeted_strike": 23200.0}
    d = agg.decide(**kwargs, mm_intent=mm_intent)
    # Should boost > 1.0, but capped at boost_max.
    assert d.mm_intent_size_multiplier > 1.0
    assert d.mm_intent_size_multiplier <= 1.80
    assert d.mm_intent_direction == 1
    assert d.mm_intent_confidence == pytest.approx(0.85, abs=0.01)


def test_aggregator_no_op_when_mm_intent_absent():
    cfg = AggregatorConfig()
    agg = DecisionAggregator(cfg)
    kwargs = _FakeAggregatorInputs.base_call_kwargs(proposed_direction=1)
    d = agg.decide(**kwargs, mm_intent=None)
    assert d.mm_intent_size_multiplier == 1.0
    assert d.mm_intent_direction == 0


# ── End-to-end manager wiring ──────────────────────────────────


def _runner_with_state(state_dir: Path,
                         pm_cfg: PortfolioManagerConfig | None = None,
                         ) -> V4Runner:
    cfg = V4RunnerConfig(
        manager=pm_cfg,
        persistence=PersistenceConfig(state_dir=state_dir, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


def test_manager_creates_manipulation_v2_engine():
    tmp = Path(tempfile.mkdtemp(prefix="mv2_mgr_"))
    runner = _runner_with_state(tmp)
    assert runner.manager.manipulation_v2 is not None
    assert len(runner.manager.manipulation_v2.detectors) == 7


def test_cockpit_carries_manipulation_v2_panel():
    tmp = Path(tempfile.mkdtemp(prefix="mv2_panel_"))
    runner = _runner_with_state(tmp)
    res = None
    for i in range(20):
        res = runner.on_tick(_snap(100 + i))
    panel = res.cockpit.to_dict()["manipulation_v2_panel"]
    for key in ["direction", "confidence", "fire_count", "regime",
                  "gamma_regime", "per_detector", "calibrator_weights"]:
        assert key in panel
    assert len(panel["per_detector"]) == 7
    assert len(panel["calibrator_weights"]) == 7


def test_manipulation_v2_override_flows_to_cockpit():
    tmp = Path(tempfile.mkdtemp(prefix="mv2_ov_"))
    runner = _runner_with_state(tmp)
    for i in range(20):
        runner.on_tick(_snap(100 + i))
    runner.manager.manipulation_v2.set_operator_override(
        direction=+1, horizon_bars=15, reason="test")
    res = runner.on_tick(_snap(121))
    panel = res.cockpit.to_dict()["manipulation_v2_panel"]
    assert panel["direction"] == 1
    assert panel["operator_override"] is not None
    assert panel["operator_override"]["reason"] == "test"


def test_viewer_html_includes_manipulation_v2_panel():
    from liqpool.research.belief.executor_v4.cockpit_server import (
        _DEFAULT_VIEWER_HTML,
    )
    h = _DEFAULT_VIEWER_HTML
    for tok in ["MANIPULATION v2", "mv2_dir", "mv2_conf",
                  "mv2_detectors", "render_manipulation_v2",
                  "btn-mv2-override-bull", "btn-mv2-override-bear",
                  "/api/manipulation/override"]:
        assert tok in h, f"viewer HTML missing {tok}"
