"""Tests for the new advanced strategies + online learner."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from liqpool.research.belief.executor_v4 import (
    AggregatorConfig,
    BullRatioSpreadStrategy,
    JadeLizardStrategy,
    LongStrangleStrategy,
    OnlineLearner,
    OnlineLearnerConfig,
    StrategyContext,
    StrategySelector,
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


class _FakeWeb:
    def __init__(self, *, consensus: float = 0.30, chop: float = 0.20,
                  tail: float = 0.05, dom: str = "long_ce"):
        self.directional_consensus = consensus
        self.chop_mass = chop
        self.tail_mass = tail
        self.dominant_strategy_class = dom


class _FakeMM:
    def __init__(self, *, intent: str = "neutral_inventory",
                  vol_view: str = "neutral"):
        self.dominant_intent = intent
        self.implied_volatility_view = vol_view


class _FakeTail:
    def __init__(self, *, action: str = "NORMAL", score: float = 0.10):
        self.recommended_action = action
        self.tail_score = score
        self.notes = []


class _FakeCrowd:
    def __init__(self, *, retail: bool = False, score: float = 0.30):
        self.we_look_like_retail = retail
        self.retail_similarity_score = score


class _FakeRich:
    def __init__(self, *, regime: float = 0.50):
        self.regime_stability_index = regime


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


# ── Strangle ───────────────────────────────────────────────────────


def test_strangle_accepts_low_consensus_low_regime():
    s = LongStrangleStrategy()
    ctx = _ctx(web_snapshot=_FakeWeb(consensus=0.05),
                rich_context=_FakeRich(regime=0.30),
                mm_posterior=_FakeMM(vol_view="expansion"),
                thesis_direction=0)
    d = s.can_enter(ctx)
    assert d.can_enter


def test_strangle_builds_two_otm_long_legs():
    s = LongStrangleStrategy()
    ctx = _ctx(thesis_direction=0, web_snapshot=_FakeWeb(consensus=0.05),
                rich_context=_FakeRich(regime=0.30),
                mm_posterior=_FakeMM(vol_view="expansion"))
    legs = s.build(ctx, lots=1)
    assert len(legs.legs) == 2
    assert all(l.direction == +1 for l in legs.legs)


# ── Ratio spread ───────────────────────────────────────────────────


def test_ratio_spread_accepts_mild_bull():
    s = BullRatioSpreadStrategy()
    ctx = _ctx(web_snapshot=_FakeWeb(consensus=0.25),
                mm_posterior=_FakeMM(intent="accumulating"))
    d = s.can_enter(ctx)
    assert d.can_enter


def test_ratio_spread_refuses_strong_consensus():
    s = BullRatioSpreadStrategy()
    ctx = _ctx(web_snapshot=_FakeWeb(consensus=0.60))
    d = s.can_enter(ctx)
    assert not d.can_enter


def test_ratio_spread_refuses_fat_tail_hedge():
    s = BullRatioSpreadStrategy()
    ctx = _ctx(fat_tail_score=_FakeTail(action="HEDGE"))
    d = s.can_enter(ctx)
    assert not d.can_enter


def test_ratio_spread_builds_1x2_with_naked_short():
    s = BullRatioSpreadStrategy()
    ctx = _ctx(web_snapshot=_FakeWeb(consensus=0.25),
                mm_posterior=_FakeMM(intent="accumulating"))
    legs = s.build(ctx, lots=1)
    assert len(legs.legs) == 2
    short_leg = next(l for l in legs.legs if l.direction == -1)
    assert short_leg.lots == 2
    assert math.isinf(legs.max_loss_rupees)


# ── Jade lizard ────────────────────────────────────────────────────


def test_jade_lizard_accepts_mild_bull_chop():
    s = JadeLizardStrategy()
    ctx = _ctx(web_snapshot=_FakeWeb(consensus=0.20, chop=0.40),
                mm_posterior=_FakeMM(intent="accumulating"))
    d = s.can_enter(ctx)
    assert d.can_enter


def test_jade_lizard_builds_three_legs():
    s = JadeLizardStrategy()
    ctx = _ctx(web_snapshot=_FakeWeb(consensus=0.20, chop=0.40),
                mm_posterior=_FakeMM(intent="accumulating"))
    legs = s.build(ctx, lots=1)
    assert len(legs.legs) == 3
    # One PE short + one CE short + one CE long
    sides = [(l.contract_side, l.direction) for l in legs.legs]
    assert ("PE", -1) in sides
    assert ("CE", -1) in sides
    assert ("CE", +1) in sides


# ── Selector with new strategies ───────────────────────────────────


def test_selector_includes_all_new_strategies():
    sel = StrategySelector()
    names = {s.name for s in sel.strategies}
    assert "long_strangle" in names
    assert "bull_ratio_spread" in names
    assert "jade_lizard" in names


# ── Online learner ─────────────────────────────────────────────────


def _initial_weights() -> Dict[str, float]:
    cfg = AggregatorConfig()
    return {
        "w_base_score": cfg.w_base_score,
        "w_mtf_alignment": cfg.w_mtf_alignment,
        "w_projection": cfg.w_projection,
        "w_fees_clearance": cfg.w_fees_clearance,
        "w_portfolio_capacity": cfg.w_portfolio_capacity,
    }


def test_online_learner_initial_weights_sum_to_one():
    learner = OnlineLearner(initial_weights=_initial_weights())
    assert abs(sum(learner.weights.values()) - 1.0) < 1e-6


def test_online_learner_requires_min_samples_before_update():
    learner = OnlineLearner(
        initial_weights=_initial_weights(),
        cfg=OnlineLearnerConfig(min_samples_per_update=10,
                                  update_every_n_closures=3),
    )
    for i in range(4):
        learner.observe_closure(
            component_scores={"base_score": 0.7, "mtf_alignment_score": 0.6,
                                "projection_factor": 0.5,
                                "fees_clearance_score": 0.7,
                                "portfolio_capacity_score": 0.8},
            realized_r=1.2,
        )
    # Threshold of update_every_n_closures=3 is met, but min_samples=10 isn't.
    assert learner.maybe_update() is None


def test_online_learner_updates_weights_after_n_closures():
    learner = OnlineLearner(
        initial_weights=_initial_weights(),
        cfg=OnlineLearnerConfig(min_samples_per_update=10,
                                  update_every_n_closures=10,
                                  learning_rate=0.05),
    )
    # Pretend positions with high base_score → wins; low base_score → losses.
    for i in range(12):
        base = 0.8 if i % 2 == 0 else 0.3
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
    # Weights still sum to 1 after update.
    assert abs(sum(update.post_update_weights.values()) - 1.0) < 1e-6


def test_online_learner_apply_to_aggregator_config_round_trip():
    learner = OnlineLearner(initial_weights=_initial_weights())
    cfg = AggregatorConfig()
    learner.apply_to_aggregator_config(cfg)
    assert abs(cfg.w_base_score - learner.weights["w_base_score"]) < 1e-9


def test_online_learner_summary_serializable():
    learner = OnlineLearner(initial_weights=_initial_weights())
    import json
    json.dumps(learner.summary())


def test_online_learner_weight_update_serializable():
    learner = OnlineLearner(
        initial_weights=_initial_weights(),
        cfg=OnlineLearnerConfig(min_samples_per_update=4,
                                  update_every_n_closures=4),
    )
    for i in range(4):
        learner.observe_closure(
            component_scores={"base_score": 0.7, "mtf_alignment_score": 0.6,
                                "projection_factor": 0.5,
                                "fees_clearance_score": 0.7,
                                "portfolio_capacity_score": 0.8},
            realized_r=1.0 if i % 2 == 0 else -0.5,
        )
    update = learner.maybe_update()
    assert update is not None
    import json
    json.dumps(update.to_dict())
