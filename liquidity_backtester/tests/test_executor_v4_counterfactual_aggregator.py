"""Tests for executor_v4 counterfactual + aggregator (Sprint 2 brain)."""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4.aggregator import (
    AGGR_ACCEPT,
    AGGR_REFUSE,
    AGGR_SIZE_DOWN,
    AggregatorConfig,
    DecisionAggregator,
)
from liqpool.research.belief.executor_v4.counterfactual import (
    CounterfactualConfig,
    CounterfactualGenerator,
    SEVERITY_HARD,
    SEVERITY_SOFT,
)
from liqpool.research.belief.executor_v4.critic import (
    AdversarialCritic,
    CritiqueResult,
)
from liqpool.research.belief.executor_v4.flow_memory import FlowMemory
from liqpool.research.belief.executor_v4.scenario_web import (
    ScenarioWeb, WebSnapshot,
)
from liqpool.research.belief.executor_v4.substrate import (
    RichContext, SubstrateState,
)


def _mk(*, bars: int = 100, thesis: str = "HOLD_BULL",
        iv: str = "directional_bull", bf: str = "bullish_agreement",
        direction: int = 1, bull_score: float = 70.0, bear_score: float = 10.0,
        ce_signed: float = 1.5, pe_signed: float = -1.0,
        ce_disp: float = 0.10, pe_disp: float = 0.10,
        ce_epi_label: str = "CE_ATM", ce_epi_level: int = 0,
        ce_def_frac: float = 0.0, pe_def_frac: float = 0.0,
        winding: str = "NO_WINDING") -> dict:
    ts = pd.Timestamp("2026-06-19 10:00") + pd.Timedelta(seconds=bars)
    n_each = 11
    n_ce_def = int(ce_def_frac * n_each); n_pe_def = int(pe_def_frac * n_each)
    slot_readings = []
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "CE", "level": level,
            "label": f"CE_{level}",
            "acceptance": "defended" if n_ce_def > 0 else "normal",
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": ce_signed, "mark_price": 100.0,
        })
        if n_ce_def > 0:
            n_ce_def -= 1
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "PE", "level": level,
            "label": f"PE_{level}",
            "acceptance": "defended" if n_pe_def > 0 else "normal",
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": pe_signed, "mark_price": 100.0,
        })
        if n_pe_def > 0:
            n_pe_def -= 1
    return {
        "ts": ts, "spot": 23000.0, "bars_seen": bars, "is_warm": True,
        "thesis": {"composite_state": thesis,
                   "bull_thesis_score": bull_score,
                   "bear_thesis_score": bear_score,
                   "no_trade_score": 5.0},
        "iv_state": {"state": iv, "direction": direction, "confidence": 0.8,
                     "clean_mark_fraction": 0.98,
                     "net_intent_z": ce_signed - pe_signed},
        "battlefield": {
            "verdict": bf, "direction": direction, "confidence": 0.8,
            "ce_rail": {"weighted_mean_signed_z": ce_signed,
                        "dispersion_score": ce_disp,
                        "epicenter_label": ce_epi_label,
                        "epicenter_level": ce_epi_level},
            "pe_rail": {"weighted_mean_signed_z": pe_signed,
                        "dispersion_score": pe_disp,
                        "epicenter_label": "PE_ATM",
                        "epicenter_level": 0},
        },
        "decision": {"action": "ENTER_LONG", "direction": direction,
                     "confidence": 0.78, "trade_allowed": True},
        "winding": {"zone": winding},
        "bull_state": {"state_index": 2},
        "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


# ── Counterfactual ─────────────────────────────────────────────────


def test_counterfactual_emits_long_specific_killers():
    cf = CounterfactualGenerator()
    sub = SubstrateState(); fm = FlowMemory()
    snap = _mk(); rich = sub.observe(snap); ev = fm.observe(snap)
    plan = cf.generate(proposed_direction=+1, proposed_profile="INTRADAY",
                       proposed_strategy_class="long_ce",
                       snapshot=snap, rich_context=rich, flow_event=ev)
    names = {k.name for k in plan.kill_criteria}
    assert "thesis_flips_bear" in names
    assert "ce_rail_collapse" in names
    assert "pe_defended_rises" in names


def test_counterfactual_emits_short_specific_killers():
    cf = CounterfactualGenerator()
    sub = SubstrateState(); fm = FlowMemory()
    snap = _mk(direction=-1, thesis="HOLD_BEAR", bf="bearish_agreement",
                bull_score=8, bear_score=72,
                ce_signed=-1.0, pe_signed=1.0)
    rich = sub.observe(snap); ev = fm.observe(snap)
    plan = cf.generate(proposed_direction=-1, proposed_profile="INTRADAY",
                       proposed_strategy_class="long_pe",
                       snapshot=snap, rich_context=rich, flow_event=ev)
    names = {k.name for k in plan.kill_criteria}
    assert "thesis_flips_bull" in names
    assert "pe_rail_collapse" in names
    assert "ce_defended_rises" in names


def test_counterfactual_scalp_has_shorter_window():
    cf = CounterfactualGenerator(CounterfactualConfig(
        scalp_kill_window=4, intraday_kill_window=20))
    sub = SubstrateState(); fm = FlowMemory()
    snap = _mk()
    rich = sub.observe(snap); ev = fm.observe(snap)
    scalp_plan = cf.generate(proposed_direction=+1, proposed_profile="SCALP",
                              proposed_strategy_class="long_ce",
                              snapshot=snap, rich_context=rich, flow_event=ev)
    intraday_plan = cf.generate(proposed_direction=+1, proposed_profile="INTRADAY",
                                 proposed_strategy_class="long_ce",
                                 snapshot=snap, rich_context=rich, flow_event=ev)
    assert scalp_plan.kill_window_bars < intraday_plan.kill_window_bars


def test_counterfactual_adds_extra_killer_on_shaky_regime():
    cf = CounterfactualGenerator()
    sub = SubstrateState(); fm = FlowMemory()
    # Whip the bars to force regime instability — query while still in chop.
    rich = None; ev = None; snap = None
    for i in range(14):
        snap = _mk(bars=100 + i,
                   thesis="BULL_ENTRY" if i % 2 == 0 else "BEAR_ENTRY",
                   ce_signed=3.0 if i % 2 == 0 else -3.0,
                   bull_score=80 if i % 2 == 0 else 10,
                   bear_score=10 if i % 2 == 0 else 80)
        rich = sub.observe(snap); ev = fm.observe(snap)
    plan = cf.generate(proposed_direction=+1, proposed_profile="INTRADAY",
                       proposed_strategy_class="long_ce",
                       snapshot=snap, rich_context=rich, flow_event=ev)
    names = {k.name for k in plan.kill_criteria}
    assert "regime_destabilizes_further" in names


def test_counterfactual_serializable():
    cf = CounterfactualGenerator()
    sub = SubstrateState(); fm = FlowMemory()
    snap = _mk(); rich = sub.observe(snap); ev = fm.observe(snap)
    plan = cf.generate(proposed_direction=+1, proposed_profile="INTRADAY",
                       proposed_strategy_class="long_ce",
                       snapshot=snap, rich_context=rich, flow_event=ev)
    import json
    json.dumps(plan.to_dict())


def test_counterfactual_severity_hard_for_critical_killers():
    cf = CounterfactualGenerator()
    sub = SubstrateState(); fm = FlowMemory()
    snap = _mk(); rich = sub.observe(snap); ev = fm.observe(snap)
    plan = cf.generate(proposed_direction=+1, proposed_profile="INTRADAY",
                       proposed_strategy_class="long_ce",
                       snapshot=snap, rich_context=rich, flow_event=ev)
    thesis_flip = next(k for k in plan.kill_criteria
                       if k.name == "thesis_flips_bear")
    assert thesis_flip.severity == SEVERITY_HARD


# ── Aggregator ──────────────────────────────────────────────────────


def test_aggregator_weights_must_sum_to_one():
    import pytest
    with pytest.raises(ValueError):
        DecisionAggregator(AggregatorConfig(
            w_base_score=0.50, w_mtf_alignment=0.50,
            w_projection=0.50, w_fees_clearance=0.50,
            w_portfolio_capacity=0.50,
        ))


class _FakeEV:
    approve = True
    edge_multiple = 2.0

    def to_dict(self):
        return {"approve": True, "edge_multiple": 2.0}


def _good_mtf():
    return {
        "alignment_ok": True, "alignment_score": 0.80,
        "l1_match": True, "l5_match": True,
        "l15_match": True, "l60_match": False,
    }


def _bad_mtf():
    return {
        "alignment_ok": False, "alignment_score": 0.10,
        "l1_match": False, "l5_match": False,
        "l15_match": False, "l60_match": False,
    }


class _FakeWeb:
    def __init__(self, *, tail_mass: float = 0.05, chop_mass: float = 0.10,
                  consensus: float = 0.40,
                  dominant_strategy: str = "long_ce"):
        self.tail_mass = tail_mass
        self.chop_mass = chop_mass
        self.directional_consensus = consensus
        self.dominant_strategy_class = dominant_strategy


class _FakeProjection:
    def __init__(self, n_samples: int = 30,
                  p_target_hit_first: float = 0.60,
                  p_stop_hit_first: float = 0.30,
                  p_neither: float = 0.10):
        self.n_samples = n_samples
        self.p_target_hit_first = p_target_hit_first
        self.p_stop_hit_first = p_stop_hit_first
        self.p_neither = p_neither


def test_aggregator_accepts_clean_setup():
    agg = DecisionAggregator()
    critique = CritiqueResult(
        antithesis_score=0.10, antithesis_reasons=[],
        recommended_size_haircut=1.0,
        recommended_action="PROCEED",
    )
    result = agg.decide(
        base_confidence=0.80, mtf_alignment=_good_mtf(),
        projection=_FakeProjection(),
        critique=critique, ev_decision=_FakeEV(),
        web_snapshot=_FakeWeb(),
        proposed_strategy_class="long_ce", proposed_direction=+1,
        open_positions_count=0, max_open_positions=3,
        daily_pnl_rupees=0.0, max_daily_bleed=5000.0,
    )
    assert result.decision == AGGR_ACCEPT
    assert result.recommended_size_multiplier == 1.0


def test_aggregator_refuses_when_critic_refuses():
    agg = DecisionAggregator()
    critique = CritiqueResult(
        antithesis_score=0.85, antithesis_reasons=["very bad"],
        recommended_size_haircut=0.40,
        recommended_action="REFUSE",
    )
    result = agg.decide(
        base_confidence=0.80, mtf_alignment=_good_mtf(),
        projection=_FakeProjection(),
        critique=critique, ev_decision=_FakeEV(),
        web_snapshot=_FakeWeb(),
        proposed_strategy_class="long_ce", proposed_direction=+1,
        open_positions_count=0, max_open_positions=3,
        daily_pnl_rupees=0.0, max_daily_bleed=5000.0,
    )
    assert result.decision == AGGR_REFUSE
    assert any("critic REFUSE" in r for r in result.refuse_reasons)


def test_aggregator_refuses_when_mtf_failed():
    agg = DecisionAggregator()
    critique = CritiqueResult(
        antithesis_score=0.10, antithesis_reasons=[],
        recommended_size_haircut=1.0,
        recommended_action="PROCEED",
    )
    result = agg.decide(
        base_confidence=0.80, mtf_alignment=_bad_mtf(),
        projection=_FakeProjection(),
        critique=critique, ev_decision=_FakeEV(),
        web_snapshot=_FakeWeb(),
        proposed_strategy_class="long_ce", proposed_direction=+1,
        open_positions_count=0, max_open_positions=3,
        daily_pnl_rupees=0.0, max_daily_bleed=5000.0,
    )
    assert result.decision == AGGR_REFUSE
    assert any("MTF alignment failed" in r for r in result.refuse_reasons)


def test_aggregator_refuses_on_high_tail_mass():
    agg = DecisionAggregator()
    critique = CritiqueResult(
        antithesis_score=0.10, antithesis_reasons=[],
        recommended_size_haircut=1.0,
        recommended_action="PROCEED",
    )
    result = agg.decide(
        base_confidence=0.80, mtf_alignment=_good_mtf(),
        projection=_FakeProjection(),
        critique=critique, ev_decision=_FakeEV(),
        web_snapshot=_FakeWeb(tail_mass=0.50),
        proposed_strategy_class="long_ce", proposed_direction=+1,
        open_positions_count=0, max_open_positions=3,
        daily_pnl_rupees=0.0, max_daily_bleed=5000.0,
    )
    assert result.decision == AGGR_REFUSE
    assert any("tail_mass" in r for r in result.refuse_reasons)


def test_aggregator_refuses_on_contradicting_web_consensus():
    agg = DecisionAggregator()
    critique = CritiqueResult(
        antithesis_score=0.10, antithesis_reasons=[],
        recommended_size_haircut=1.0,
        recommended_action="PROCEED",
    )
    # Long proposal but web's directional_consensus is bearish.
    result = agg.decide(
        base_confidence=0.80, mtf_alignment=_good_mtf(),
        projection=_FakeProjection(),
        critique=critique, ev_decision=_FakeEV(),
        web_snapshot=_FakeWeb(consensus=-0.50),
        proposed_strategy_class="long_ce", proposed_direction=+1,
        open_positions_count=0, max_open_positions=3,
        daily_pnl_rupees=0.0, max_daily_bleed=5000.0,
    )
    assert result.decision == AGGR_REFUSE
    assert any("directional_consensus" in r for r in result.refuse_reasons)


def test_aggregator_size_downs_on_moderate_score():
    agg = DecisionAggregator(AggregatorConfig(
        size_down_threshold=0.40, accept_threshold=0.70,
    ))
    critique = CritiqueResult(
        antithesis_score=0.10, antithesis_reasons=[],
        recommended_size_haircut=1.0,
        recommended_action="PROCEED",
    )
    # Modest confidence + projection neutral → final ~ mid.
    result = agg.decide(
        base_confidence=0.60, mtf_alignment=_good_mtf(),
        projection=_FakeProjection(n_samples=5),  # low confidence → neutral
        critique=critique,
        ev_decision=_FakeEV(),
        web_snapshot=_FakeWeb(),
        proposed_strategy_class="long_ce", proposed_direction=+1,
        open_positions_count=1, max_open_positions=3,
        daily_pnl_rupees=0.0, max_daily_bleed=5000.0,
    )
    assert result.decision in (AGGR_ACCEPT, AGGR_SIZE_DOWN)
    if result.decision == AGGR_SIZE_DOWN:
        assert 0.5 <= result.recommended_size_multiplier < 1.0


def test_aggregator_refuses_on_unfavorable_projection():
    agg = DecisionAggregator(AggregatorConfig(
        favorable_target_minus_stop=0.05,
    ))
    critique = CritiqueResult(
        antithesis_score=0.10, antithesis_reasons=[],
        recommended_size_haircut=1.0,
        recommended_action="PROCEED",
    )
    # P(stop) >> P(target) → margin very negative.
    bad_proj = _FakeProjection(n_samples=30,
                                 p_target_hit_first=0.20,
                                 p_stop_hit_first=0.70,
                                 p_neither=0.10)
    result = agg.decide(
        base_confidence=0.80, mtf_alignment=_good_mtf(),
        projection=bad_proj,
        critique=critique, ev_decision=_FakeEV(),
        web_snapshot=_FakeWeb(),
        proposed_strategy_class="long_ce", proposed_direction=+1,
        open_positions_count=0, max_open_positions=3,
        daily_pnl_rupees=0.0, max_daily_bleed=5000.0,
    )
    assert result.decision == AGGR_REFUSE
    assert any("projection unfavorable" in r for r in result.refuse_reasons)


def test_aggregator_refuses_on_daily_bleed():
    agg = DecisionAggregator()
    critique = CritiqueResult(
        antithesis_score=0.10, antithesis_reasons=[],
        recommended_size_haircut=1.0,
        recommended_action="PROCEED",
    )
    result = agg.decide(
        base_confidence=0.80, mtf_alignment=_good_mtf(),
        projection=_FakeProjection(),
        critique=critique, ev_decision=_FakeEV(),
        web_snapshot=_FakeWeb(),
        proposed_strategy_class="long_ce", proposed_direction=+1,
        open_positions_count=0, max_open_positions=3,
        daily_pnl_rupees=-5100.0, max_daily_bleed=5000.0,
    )
    assert result.decision == AGGR_REFUSE
    assert any("daily bleed" in r for r in result.refuse_reasons)


def test_aggregator_to_dict_serializable():
    agg = DecisionAggregator()
    critique = CritiqueResult(
        antithesis_score=0.10, antithesis_reasons=[],
        recommended_size_haircut=1.0,
        recommended_action="PROCEED",
    )
    result = agg.decide(
        base_confidence=0.80, mtf_alignment=_good_mtf(),
        projection=_FakeProjection(),
        critique=critique, ev_decision=_FakeEV(),
        web_snapshot=_FakeWeb(),
        proposed_strategy_class="long_ce", proposed_direction=+1,
        open_positions_count=0, max_open_positions=3,
        daily_pnl_rupees=0.0, max_daily_bleed=5000.0,
    )
    import json
    json.dumps(result.to_dict())


def test_aggregator_haircut_compounds_with_critic():
    agg = DecisionAggregator()
    critique = CritiqueResult(
        antithesis_score=0.45, antithesis_reasons=["moderate"],
        recommended_size_haircut=0.60,
        recommended_action="SIZE_DOWN",
    )
    result = agg.decide(
        base_confidence=0.80, mtf_alignment=_good_mtf(),
        projection=_FakeProjection(),
        critique=critique, ev_decision=_FakeEV(),
        web_snapshot=_FakeWeb(),
        proposed_strategy_class="long_ce", proposed_direction=+1,
        open_positions_count=0, max_open_positions=3,
        daily_pnl_rupees=0.0, max_daily_bleed=5000.0,
    )
    # Should have compounded critic's 0.60 with whatever the final_score says.
    assert result.recommended_size_multiplier <= 0.60
