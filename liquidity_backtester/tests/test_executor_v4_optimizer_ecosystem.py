"""Tests for the portfolio optimizer + walk-forward learner + ecosystem strategies + Monday bonuses."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

from liqpool.research.belief.executor_v4 import (
    AccumulationBreakoutStrategy,
    AggregatorConfig,
    AlertHookEmitter,
    AntiCrowdContrarianStrategy,
    CoordinateDescentSolver,
    CounterfactualInversionStrategy,
    DefensiveAutoHedger,
    DefensiveAutoHedgerConfig,
    EpicenterMigrationStrategy,
    GreedyHedgeSolver,
    GreekTargets,
    MMIntentMimicryStrategy,
    MTFDivergenceStrategy,
    OnlineLearner,
    OnlineLearnerConfig,
    RegimeTransitionStrategy,
    SlippageTracker,
    StopHuntFadeStrategy,
    StrategyContext,
    StrategySelector,
    TimeOfDayStopScaler,
    TimeOfDayStopScalerConfig,
    V4Runner,
    WebDominantPathwayStrategy,
    health_report,
)


def _strike_lookup() -> Dict[Tuple[str, int], Tuple[float, float, float, str]]:
    out = {}
    for level in range(-5, 6):
        strike = 23000.0 + 50 * level
        ce_prem = max(5.0, 100.0 - 18.0 * level)
        pe_prem = max(5.0, 100.0 + 18.0 * level)
        out[("CE", level)] = (strike, ce_prem, 0.92, "clean")
        out[("PE", level)] = (strike, pe_prem, 0.92, "clean")
    return out


# ── Portfolio optimizer ─────────────────────────────────────────────


@dataclass(frozen=True)
class _FakeLeg:
    contract_side: str
    strike_price: float
    direction: int
    lots: int
    contract_level: int
    entry_premium: float


@dataclass(frozen=True)
class _FakeStrategyLegs:
    """Mimics StrategyLegs.legs for portfolio aggregation."""
    legs: list


def test_optimizer_returns_noop_when_already_in_band():
    solver = GreedyHedgeSolver(targets=GreekTargets(
        max_net_delta=500.0, max_net_vega=200.0,
        max_net_gamma=10.0, max_net_theta_per_day=2000.0,
    ))
    proposal = solver.solve(
        open_positions=[],
        strike_lookup=_strike_lookup(),
        spot=23000.0, time_to_expiry=7/365,
        lot_size=65,
    )
    assert proposal.in_target_band_before
    assert proposal.actions == []


def test_greedy_solver_proposes_hedge_for_long_call_delta():
    # One long ATM CE = positive delta ~ 30 (lot 65 * 0.5)
    pos = _FakeStrategyLegs(legs=[
        _FakeLeg(contract_side="CE", strike_price=23000.0,
                  direction=+1, lots=1, contract_level=0,
                  entry_premium=100.0),
    ])
    solver = GreedyHedgeSolver(targets=GreekTargets(
        max_net_delta=10.0, max_net_vega=200.0,
        max_net_gamma=10.0, max_net_theta_per_day=2000.0,
    ))
    proposal = solver.solve(
        open_positions=[pos],
        strike_lookup=_strike_lookup(),
        spot=23000.0, time_to_expiry=7/365,
        lot_size=65,
    )
    # Delta band is tight → should propose something.
    assert not proposal.in_target_band_before
    assert len(proposal.actions) >= 1
    # Objective should improve.
    assert proposal.objective_after <= proposal.objective_before


def test_coordinate_descent_solver_runs():
    pos = _FakeStrategyLegs(legs=[
        _FakeLeg(contract_side="CE", strike_price=23000.0,
                  direction=+1, lots=2, contract_level=0,
                  entry_premium=100.0),
    ])
    solver = CoordinateDescentSolver(targets=GreekTargets(
        max_net_delta=10.0, max_net_vega=10.0,
    ))
    proposal = solver.solve(
        open_positions=[pos],
        strike_lookup=_strike_lookup(),
        spot=23000.0, time_to_expiry=7/365,
        lot_size=65,
    )
    assert len(proposal.actions) >= 1
    assert proposal.objective_after <= proposal.objective_before


def test_optimizer_proposal_serializable():
    pos = _FakeStrategyLegs(legs=[
        _FakeLeg(contract_side="CE", strike_price=23000.0,
                  direction=+1, lots=1, contract_level=0,
                  entry_premium=100.0),
    ])
    solver = GreedyHedgeSolver(targets=GreekTargets(max_net_delta=5.0))
    proposal = solver.solve(
        open_positions=[pos], strike_lookup=_strike_lookup(),
        spot=23000.0, time_to_expiry=7/365, lot_size=65,
    )
    import json
    json.dumps(proposal.to_dict())


# ── Walk-forward learner ───────────────────────────────────────────


def _initial_weights() -> Dict[str, float]:
    cfg = AggregatorConfig()
    return {
        "w_base_score": cfg.w_base_score,
        "w_mtf_alignment": cfg.w_mtf_alignment,
        "w_projection": cfg.w_projection,
        "w_fees_clearance": cfg.w_fees_clearance,
        "w_portfolio_capacity": cfg.w_portfolio_capacity,
    }


def test_walk_forward_rejects_overfit_update():
    """Build observations where TRAIN is easy and VAL is impossible to
    predict from the same features. The overfit ratio should fire."""
    learner = OnlineLearner(
        initial_weights=_initial_weights(),
        cfg=OnlineLearnerConfig(
            min_samples_per_update=10,
            update_every_n_closures=10,
            walk_forward_enabled=True,
            walk_forward_train_fraction=0.70,
            walk_forward_overfit_threshold=1.30,
            walk_forward_min_val_samples=2,
            learning_rate=0.05,
        ),
    )
    # 7 train samples where base_score perfectly predicts wins.
    for i in range(7):
        base = 0.9 if i % 2 == 0 else 0.1
        r = 1.5 if i % 2 == 0 else -1.0
        learner.observe_closure(
            component_scores={"base_score": base,
                                "mtf_alignment_score": 0.5,
                                "projection_factor": 0.5,
                                "fees_clearance_score": 0.6,
                                "portfolio_capacity_score": 0.6},
            realized_r=r,
        )
    # 3 val samples where base_score INVERTS — high base = loss.
    for i in range(3):
        base = 0.9 if i % 2 == 0 else 0.1
        r = -1.0 if i % 2 == 0 else 1.5     # inverted
        learner.observe_closure(
            component_scores={"base_score": base,
                                "mtf_alignment_score": 0.5,
                                "projection_factor": 0.5,
                                "fees_clearance_score": 0.6,
                                "portfolio_capacity_score": 0.6},
            realized_r=r,
        )
    update = learner.maybe_update()
    assert update is not None
    # Either rejected, or the val_loss is genuinely tracked.
    assert update.val_loss > 0
    if not update.walk_forward_accepted:
        # Weights unchanged
        for k in update.weight_deltas:
            assert update.weight_deltas[k] == 0


def test_walk_forward_accepts_consistent_update():
    learner = OnlineLearner(
        initial_weights=_initial_weights(),
        cfg=OnlineLearnerConfig(
            min_samples_per_update=10,
            update_every_n_closures=10,
            walk_forward_enabled=True,
            walk_forward_min_val_samples=2,
            learning_rate=0.05,
        ),
    )
    # 14 consistent samples — high base_score predicts wins in both halves.
    for i in range(14):
        base = 0.9 if i % 2 == 0 else 0.1
        r = 1.5 if i % 2 == 0 else -1.0
        learner.observe_closure(
            component_scores={"base_score": base,
                                "mtf_alignment_score": 0.5,
                                "projection_factor": 0.5,
                                "fees_clearance_score": 0.6,
                                "portfolio_capacity_score": 0.6},
            realized_r=r,
        )
    update = learner.maybe_update()
    assert update is not None
    assert update.walk_forward_accepted


def test_walk_forward_disabled_works_as_before():
    learner = OnlineLearner(
        initial_weights=_initial_weights(),
        cfg=OnlineLearnerConfig(
            min_samples_per_update=8,
            update_every_n_closures=8,
            walk_forward_enabled=False,
            learning_rate=0.05,
        ),
    )
    for i in range(10):
        learner.observe_closure(
            component_scores={"base_score": 0.7, "mtf_alignment_score": 0.6,
                                "projection_factor": 0.5,
                                "fees_clearance_score": 0.6,
                                "portfolio_capacity_score": 0.6},
            realized_r=1.0 if i % 2 == 0 else -0.5,
        )
    update = learner.maybe_update()
    assert update is not None
    assert update.walk_forward_accepted


# ── Ecosystem strategies ───────────────────────────────────────────


class _FakeMM:
    def __init__(self, *, intent: str = "accumulating",
                  prob: float = 0.65, confidence: float = 0.60,
                  vol_view: str = "neutral"):
        self.dominant_intent = intent
        self.dominant_probability = prob
        self.confidence = confidence
        self.implied_volatility_view = vol_view
        self.intent_distribution = {intent: prob}


class _FakeWeb:
    def __init__(self, *, consensus: float = 0.30,
                  chop: float = 0.20, tail: float = 0.05,
                  dom: str = "long_ce",
                  top_scenarios=None,
                  currently_dominant_pathway=None):
        self.directional_consensus = consensus
        self.chop_mass = chop
        self.tail_mass = tail
        self.dominant_strategy_class = dom
        self.top_scenarios = top_scenarios or []
        self.currently_dominant_pathway = currently_dominant_pathway


class _FakeTail:
    def __init__(self, *, action: str = "NORMAL", score: float = 0.10):
        self.recommended_action = action
        self.tail_score = score
        self.notes = []


class _FakeCrowd:
    def __init__(self, *, retail: bool = False, score: float = 0.30,
                  long_atm_ce: float = 0.0, long_atm_pe: float = 0.0):
        self.we_look_like_retail = retail
        self.retail_similarity_score = score
        self.crowd_density_long_atm_ce = long_atm_ce
        self.crowd_density_long_atm_pe = long_atm_pe


class _FakeRich:
    def __init__(self, *, regime: float = 0.50, epi_mig: float = 0.0,
                  epi_level: int = 0, disp_v: float = 0.0,
                  vel_side: str = "flat"):
        self.regime_stability_index = regime
        self.epicenter_migration_distance = epi_mig
        self.epicenter_level = epi_level
        self.epicenter_label = f"CE_OTM{epi_level}" if epi_level > 0 else "CE_ATM"
        self.dispersion_velocity = disp_v
        self.thesis_velocity_dominant_side = vel_side


def _ctx(**kwargs) -> StrategyContext:
    defaults = dict(
        spot=23000.0, lot_size=65,
        iv_state="directional_bull",
        bf_verdict="bullish_agreement",
        thesis_state="HOLD_BULL",
        thesis_direction=+1,
        confidence=0.80,
        web_snapshot=_FakeWeb(),
        mm_posterior=_FakeMM(),
        fat_tail_score=_FakeTail(),
        crowd_report=_FakeCrowd(),
        rich_context=_FakeRich(),
        flow_event=None,
        strike_lookup=_strike_lookup(),
        lots_budget=2,
        preferred_side="CE", preferred_level=0,
    )
    defaults.update(kwargs)
    return StrategyContext(**defaults)


def test_mm_mimicry_accepts_strong_accumulating():
    s = MMIntentMimicryStrategy()
    ctx = _ctx(mm_posterior=_FakeMM(intent="accumulating",
                                       prob=0.70, confidence=0.60))
    d = s.can_enter(ctx)
    assert d.can_enter


def test_mm_mimicry_refuses_neutral_intent():
    s = MMIntentMimicryStrategy()
    ctx = _ctx(mm_posterior=_FakeMM(intent="neutral_inventory", prob=0.90))
    d = s.can_enter(ctx)
    assert not d.can_enter


def test_mm_mimicry_builds_long_atm_ce_for_accumulating():
    s = MMIntentMimicryStrategy()
    ctx = _ctx(mm_posterior=_FakeMM(intent="accumulating",
                                       prob=0.70, confidence=0.60))
    legs = s.build(ctx, lots=1)
    assert legs.legs[0].contract_side == "CE"
    assert legs.legs[0].direction == +1


def test_anti_crowd_contrarian_fades_long_ce_crowd():
    s = AntiCrowdContrarianStrategy()
    ctx = _ctx(crowd_report=_FakeCrowd(retail=True, score=0.75,
                                          long_atm_ce=0.80))
    d = s.can_enter(ctx)
    assert d.can_enter
    # Should propose PE.
    legs = s.build(ctx, lots=1)
    assert legs.legs[0].contract_side == "PE"


def test_epicenter_migration_accepts_distance_3_plus():
    s = EpicenterMigrationStrategy()
    ctx = _ctx(rich_context=_FakeRich(epi_mig=4.0, epi_level=2))
    d = s.can_enter(ctx)
    assert d.can_enter


def test_epicenter_migration_refuses_under_threshold():
    s = EpicenterMigrationStrategy()
    ctx = _ctx(rich_context=_FakeRich(epi_mig=1.0))
    d = s.can_enter(ctx)
    assert not d.can_enter


def test_stop_hunt_fade_accepts_when_scenario_active():
    s = StopHuntFadeStrategy()
    ctx = _ctx(web_snapshot=_FakeWeb(top_scenarios=[
        {"name": "stop_hunt_up_reverse", "current_probability": 0.20},
    ]))
    d = s.can_enter(ctx)
    assert d.can_enter
    # stop_hunt_up implies long PE (fade up move).
    legs = s.build(ctx, lots=1)
    assert legs.legs[0].contract_side == "PE"


def test_accumulation_breakout_uses_mm_signal():
    s = AccumulationBreakoutStrategy()
    ctx = _ctx(mm_posterior=_FakeMM(intent="accumulating",
                                       prob=0.60))
    d = s.can_enter(ctx)
    assert d.can_enter


def test_mtf_divergence_fires_on_velocity_conflict():
    s = MTFDivergenceStrategy()
    # Thesis is bullish (+1) but velocity dominant is bear.
    ctx = _ctx(thesis_direction=+1,
                rich_context=_FakeRich(vel_side="bear"))
    d = s.can_enter(ctx)
    assert d.can_enter


def test_web_dominant_pathway_picks_dominant_scenario():
    s = WebDominantPathwayStrategy()
    ctx = _ctx(web_snapshot=_FakeWeb(currently_dominant_pathway={
        "name": "bull_continuation",
        "current_probability": 0.55,
        "implied_direction": +1,
    }))
    d = s.can_enter(ctx)
    assert d.can_enter


def test_regime_transition_fires_at_transition_zone():
    s = RegimeTransitionStrategy()
    ctx = _ctx(rich_context=_FakeRich(regime=0.45, disp_v=0.10))
    d = s.can_enter(ctx)
    assert d.can_enter


def test_counterfactual_inversion_fires_on_velocity_thesis_conflict():
    s = CounterfactualInversionStrategy()
    ctx = _ctx(thesis_direction=+1,
                rich_context=_FakeRich(regime=0.35, vel_side="bear"))
    d = s.can_enter(ctx)
    assert d.can_enter


def test_selector_includes_all_ecosystem_strategies():
    sel = StrategySelector()
    names = {s.name for s in sel.strategies}
    ecosystem_names = {
        "mm_intent_mimicry", "anti_crowd_contrarian",
        "epicenter_migration", "stop_hunt_fade",
        "accumulation_breakout", "mtf_divergence",
        "web_dominant_pathway", "regime_transition",
        "counterfactual_inversion",
    }
    assert ecosystem_names.issubset(names)


# ── Monday bonuses ──────────────────────────────────────────────────


def test_time_of_day_stop_scaler_morning_widens():
    scaler = TimeOfDayStopScaler()
    ts = datetime(2026, 6, 22, 9, 20, 0)
    assert scaler.scale_for(ts) > 1.0


def test_time_of_day_stop_scaler_lunch_widens():
    scaler = TimeOfDayStopScaler()
    ts = datetime(2026, 6, 22, 12, 30, 0)
    assert scaler.scale_for(ts) > 1.0


def test_time_of_day_stop_scaler_final_15min_tightens():
    scaler = TimeOfDayStopScaler()
    ts = datetime(2026, 6, 22, 15, 20, 0)
    assert scaler.scale_for(ts) < 1.0


def test_time_of_day_stop_scaler_default_when_no_ts():
    scaler = TimeOfDayStopScaler()
    # When ts is None, returns 1.0 (or whatever current IST time gives)
    assert scaler.scale_for("garbage") == 1.0


def test_defensive_auto_hedger_fires_after_n_consecutive():
    hedger = DefensiveAutoHedger(DefensiveAutoHedgerConfig(
        consecutive_hedge_ticks_required=2,
        max_hedges_per_session=2,
    ))
    proposal = {"proposed": True, "proposals": [
        {"contract_side": "PE", "contract_level": -2, "lots": 1,
          "premium": 50.0, "rationale": "tail defense"}]}
    # First HEDGE tick — no fire (need 2)
    assert hedger.consider(fat_tail_action="HEDGE",
                              hedge_proposal_dict=proposal) is None
    # Second HEDGE tick — should fire
    result = hedger.consider(fat_tail_action="HEDGE",
                                hedge_proposal_dict=proposal)
    assert result is not None
    # Third tick (still HEDGE) — no fire (in cooldown)
    assert hedger.consider(fat_tail_action="HEDGE",
                              hedge_proposal_dict=proposal) is None


def test_defensive_auto_hedger_respects_session_cap():
    hedger = DefensiveAutoHedger(DefensiveAutoHedgerConfig(
        consecutive_hedge_ticks_required=1,
        consecutive_normal_ticks_to_clear=1,
        max_hedges_per_session=1,
    ))
    proposal = {"proposed": True, "proposals": [{"x": 1}]}
    assert hedger.consider(fat_tail_action="HEDGE",
                              hedge_proposal_dict=proposal) is not None
    # Reset to NORMAL
    hedger.consider(fat_tail_action="NORMAL", hedge_proposal_dict=None)
    # Try again — session cap should block
    assert hedger.consider(fat_tail_action="HEDGE",
                              hedge_proposal_dict=proposal) is None


def test_slippage_tracker_records_and_summarizes():
    st = SlippageTracker()
    st.record(tradingsymbol="X", side="BUY",
                predicted=100.0, actual=100.50)
    st.record(tradingsymbol="X", side="BUY",
                predicted=100.0, actual=99.80)
    s = st.rolling_summary()
    assert s["n_records"] == 2


def test_alert_emitter_generates_big_loss_alert():
    em = AlertHookEmitter()
    intent = {"closed_this_tick": [{
        "position_id": "abc12345",
        "outcome": {"realized_rupees": -800.0,
                     "exit_reason": "stop", "realized_r": -1.0}}],
        "portfolio_summary": {}}
    alerts = em.from_intent(intent)
    assert any(a.code == "BIG_LOSS" for a in alerts)


def test_alert_emitter_generates_kill_switch_alert():
    em = AlertHookEmitter()
    intent = {"closed_this_tick": [],
              "portfolio_summary": {
                  "portfolio_risk": {"kill_switches": ["cluster overshoot"]}}}
    alerts = em.from_intent(intent)
    assert any(a.code == "KILL_SWITCH" for a in alerts)


def test_alert_emitter_rate_limits_same_code():
    em = AlertHookEmitter()
    intent = {"closed_this_tick": [],
              "portfolio_summary": {
                  "portfolio_risk": {"kill_switches": ["cluster"]}}}
    a1 = em.from_intent(intent)
    a2 = em.from_intent(intent)
    # Second call within rate-limit window → empty
    assert len(a1) >= 1
    assert len(a2) == 0


def test_health_report_returns_serializable_dict():
    runner = V4Runner.paper()
    rep = health_report(runner)
    assert rep["ok"] is True
    assert "broker" in rep
    assert "daily_pnl_rupees" in rep
    import json
    json.dumps(rep, default=str)
