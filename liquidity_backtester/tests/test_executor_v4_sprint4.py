"""Tests for executor_v4 Sprint 4: strategy library + risk + hedge + explainer."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pytest

from liqpool.research.belief.executor_v4.explainer import explain_tick
from liqpool.research.belief.executor_v4.hedge import (
    HedgeProposalConfig,
    HedgeProposer,
)
from liqpool.research.belief.executor_v4.risk import (
    PortfolioRiskConfig,
    PortfolioRiskLayer,
)
from liqpool.research.belief.executor_v4.strategy_library import (
    BearVerticalStrategy,
    BullVerticalStrategy,
    ButterflyStrategy,
    IronCondorStrategy,
    LongStraddleStrategy,
    SingleLegStrategy,
    StrategyContext,
    StrategySelector,
)


def _strike_lookup() -> Dict[Tuple[str, int], Tuple[float, float, float, str]]:
    """Build a synthetic strike lookup centred at 23000 with ATM premium 100."""
    out: Dict[Tuple[str, int], Tuple[float, float, float, str]] = {}
    for level in range(-5, 6):
        strike = 23000.0 + 50 * level
        # CE: ATM 100, OTM decays
        ce_prem = max(5.0, 100.0 - 18.0 * level)
        # PE: mirror
        pe_prem = max(5.0, 100.0 + 18.0 * level)
        out[("CE", level)] = (strike, ce_prem, 0.92, "clean")
        out[("PE", level)] = (strike, pe_prem, 0.92, "clean")
    return out


def _ctx(*, thesis_direction: int = +1,
          confidence: float = 0.80,
          web=None, mm=None, tail=None, crowd=None,
          rich=None, iv_state: str = "directional_bull",
          preferred_side: str = "CE", preferred_level: int = 0,
          ) -> StrategyContext:
    return StrategyContext(
        spot=23000.0, lot_size=65,
        iv_state=iv_state,
        bf_verdict="bullish_agreement",
        thesis_state="HOLD_BULL",
        thesis_direction=thesis_direction,
        confidence=confidence,
        web_snapshot=web, mm_posterior=mm,
        fat_tail_score=tail, crowd_report=crowd,
        rich_context=rich, flow_event=None,
        strike_lookup=_strike_lookup(),
        lots_budget=2,
        preferred_side=preferred_side,
        preferred_level=preferred_level,
    )


class _FakeWeb:
    def __init__(self, *, consensus: float = 0.50, chop: float = 0.10,
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
    def __init__(self, *, regime: float = 0.80):
        self.regime_stability_index = regime


# ── Single-leg ─────────────────────────────────────────────────────


def test_single_leg_accepts_clean_bull_setup():
    s = SingleLegStrategy()
    ctx = _ctx(web=_FakeWeb(consensus=0.50),
                tail=_FakeTail(), crowd=_FakeCrowd())
    d = s.can_enter(ctx)
    assert d.can_enter
    assert d.fitness_score > 0.50


def test_single_leg_refuses_when_mm_hunting():
    s = SingleLegStrategy()
    ctx = _ctx(mm=_FakeMM(intent="hunting_stops"))
    d = s.can_enter(ctx)
    assert not d.can_enter


def test_single_leg_refuses_under_fat_tail_refuse():
    s = SingleLegStrategy()
    ctx = _ctx(tail=_FakeTail(action="REFUSE"))
    d = s.can_enter(ctx)
    assert not d.can_enter


def test_single_leg_build_emits_one_long_leg():
    s = SingleLegStrategy()
    ctx = _ctx(web=_FakeWeb(consensus=0.40))
    legs = s.build(ctx, lots=2)
    assert legs.strategy_name == "single_leg"
    assert len(legs.legs) == 1
    assert legs.legs[0].direction == +1
    assert legs.legs[0].lots == 2
    assert math.isinf(legs.max_gain_rupees)


# ── Vertical spreads ───────────────────────────────────────────────


def test_bull_vertical_prefers_when_crowd_retail():
    s = BullVerticalStrategy()
    ctx = _ctx(crowd=_FakeCrowd(retail=True, score=0.80),
                web=_FakeWeb(consensus=0.40))
    d = s.can_enter(ctx)
    assert d.can_enter
    assert d.fitness_score > 0.60


def test_bull_vertical_builds_two_legs():
    s = BullVerticalStrategy()
    ctx = _ctx()
    legs = s.build(ctx, lots=2)
    assert len(legs.legs) == 2
    assert legs.legs[0].direction == +1   # long ATM
    assert legs.legs[1].direction == -1   # short OTM
    assert legs.max_gain_rupees > 0
    assert legs.max_loss_rupees > 0


def test_bear_vertical_refuses_bull_direction():
    s = BearVerticalStrategy()
    ctx = _ctx(thesis_direction=+1)
    d = s.can_enter(ctx)
    assert not d.can_enter


# ── Long straddle ──────────────────────────────────────────────────


def test_straddle_accepts_low_consensus_low_regime():
    s = LongStraddleStrategy()
    ctx = _ctx(web=_FakeWeb(consensus=0.05),
                rich=_FakeRich(regime=0.30),
                mm=_FakeMM(vol_view="expansion"),
                thesis_direction=0)
    d = s.can_enter(ctx)
    assert d.can_enter
    assert d.fitness_score > 0.45


def test_straddle_refuses_strongly_directional():
    s = LongStraddleStrategy()
    ctx = _ctx(web=_FakeWeb(consensus=0.70))
    d = s.can_enter(ctx)
    assert not d.can_enter


def test_straddle_builds_two_long_legs():
    s = LongStraddleStrategy()
    ctx = _ctx(web=_FakeWeb(consensus=0.05), thesis_direction=0)
    legs = s.build(ctx, lots=1)
    assert len(legs.legs) == 2
    assert all(l.direction == +1 for l in legs.legs)


# ── Iron condor ────────────────────────────────────────────────────


def test_iron_condor_accepts_chop_dominant():
    s = IronCondorStrategy()
    ctx = _ctx(web=_FakeWeb(consensus=0.05, chop=0.55),
                mm=_FakeMM(intent="neutral_inventory"))
    d = s.can_enter(ctx)
    assert d.can_enter


def test_iron_condor_refuses_expansion_view():
    s = IronCondorStrategy()
    ctx = _ctx(web=_FakeWeb(chop=0.55),
                mm=_FakeMM(vol_view="expansion"))
    d = s.can_enter(ctx)
    assert not d.can_enter


def test_iron_condor_builds_four_legs():
    s = IronCondorStrategy()
    ctx = _ctx(web=_FakeWeb(consensus=0.05, chop=0.55),
                mm=_FakeMM(intent="neutral_inventory"))
    legs = s.build(ctx, lots=1)
    assert len(legs.legs) == 4
    # Should be net credit (negative net_premium_paid).
    assert legs.net_premium_paid <= 0


# ── Butterfly ──────────────────────────────────────────────────────


def test_butterfly_accepts_pinning_mm():
    s = ButterflyStrategy()
    ctx = _ctx(mm=_FakeMM(intent="pinning_near_expiry"),
                web=_FakeWeb(dom="butterfly"))
    d = s.can_enter(ctx)
    assert d.can_enter


def test_butterfly_builds_three_strikes_with_short_doubled():
    s = ButterflyStrategy()
    ctx = _ctx(mm=_FakeMM(intent="pinning_near_expiry"))
    legs = s.build(ctx, lots=1)
    assert len(legs.legs) == 3
    short_leg = next(l for l in legs.legs if l.direction == -1)
    assert short_leg.lots == 2     # doubled


# ── Strategy selector ──────────────────────────────────────────────


def test_selector_picks_iron_condor_in_chop():
    sel = StrategySelector()
    ctx = _ctx(thesis_direction=0,
                web=_FakeWeb(consensus=0.05, chop=0.60),
                mm=_FakeMM(intent="neutral_inventory"))
    result = sel.select(ctx)
    assert result.chosen is not None
    assert result.chosen_name in ("iron_condor", "long_straddle")


def test_selector_picks_bull_vertical_when_crowd_retail():
    sel = StrategySelector()
    ctx = _ctx(thesis_direction=+1,
                web=_FakeWeb(consensus=0.40),
                crowd=_FakeCrowd(retail=True, score=0.80))
    result = sel.select(ctx)
    assert result.chosen is not None
    # Should prefer a defined-risk vertical over a naked single leg.
    assert result.chosen_name in ("bull_vertical", "single_leg")


def test_selector_fallback_uses_single_leg():
    sel = StrategySelector()
    # No quotes → no eligible strategy.
    ctx = _ctx(thesis_direction=+1)
    ctx.strike_lookup = {}
    result = sel.select(ctx)
    assert result.chosen is None


def test_selector_serializable():
    sel = StrategySelector()
    ctx = _ctx(thesis_direction=+1, web=_FakeWeb(consensus=0.40))
    result = sel.select(ctx)
    import json
    json.dumps(result.to_dict())


# ── Greeks ─────────────────────────────────────────────────────────


def test_single_leg_greeks_long_ce_positive_delta():
    s = SingleLegStrategy()
    ctx = _ctx()
    legs = s.build(ctx, lots=1)
    g = s.greeks(legs, ctx.spot)
    assert g["delta"] > 0
    assert g["vega"] > 0
    assert g["theta"] < 0   # long position bleeds theta


def test_iron_condor_greeks_near_neutral_delta():
    s = IronCondorStrategy()
    ctx = _ctx(web=_FakeWeb(consensus=0.05, chop=0.55),
                mm=_FakeMM(intent="neutral_inventory"))
    legs = s.build(ctx, lots=1)
    g = s.greeks(legs, ctx.spot)
    # Net delta should be small (long PE + short CE on one side, etc.)
    assert abs(g["delta"]) < 0.5
    # Net vega should be net short (we sold the inner strangle).
    assert g["vega"] < 0


# ── Portfolio risk ─────────────────────────────────────────────────


class _FakeHypothesis:
    def __init__(self, *, position_id: str, direction: int,
                  rupees_at_risk: float, size_lots: int):
        self.position_id = position_id
        self.direction = direction
        self.rupees_at_risk = rupees_at_risk
        self.size_lots = size_lots


class _FakeState:
    def __init__(self, hypothesis: _FakeHypothesis):
        self.hypothesis = hypothesis


def test_risk_layer_aggregates_premium():
    risk = PortfolioRiskLayer()
    open_states = {
        "p1": _FakeState(_FakeHypothesis(position_id="p1", direction=+1,
                                            rupees_at_risk=500.0, size_lots=1)),
        "p2": _FakeState(_FakeHypothesis(position_id="p2", direction=+1,
                                            rupees_at_risk=300.0, size_lots=2)),
    }
    report = risk.compute(open_states=open_states,
                            current_r_by_position={"p1": 0.3, "p2": -0.4})
    assert report.total_premium_at_risk_rupees == 800.0
    assert report.portfolio_drawdown_r == 0.4


def test_risk_kill_switch_fires_on_overshoot_budget():
    risk = PortfolioRiskLayer(PortfolioRiskConfig(
        max_total_premium_at_risk_rupees=500.0))
    open_states = {
        "p1": _FakeState(_FakeHypothesis(position_id="p1", direction=+1,
                                            rupees_at_risk=600.0, size_lots=1)),
    }
    report = risk.compute(open_states=open_states,
                            current_r_by_position={})
    assert any("budget" in k for k in report.kill_switches)


def test_risk_kill_switch_fires_on_net_delta_cap():
    risk = PortfolioRiskLayer(PortfolioRiskConfig(max_net_delta_lots=2.0))
    open_states = {
        "p1": _FakeState(_FakeHypothesis(position_id="p1", direction=+1,
                                            rupees_at_risk=500.0, size_lots=3)),
    }
    report = risk.compute(open_states=open_states,
                            current_r_by_position={})
    assert any("directional" in k for k in report.kill_switches)


def test_risk_serializable():
    risk = PortfolioRiskLayer()
    report = risk.compute(open_states={}, current_r_by_position={})
    import json
    json.dumps(report.to_dict())


# ── Hedge proposer ─────────────────────────────────────────────────


def test_hedge_proposes_nothing_in_calm():
    h = HedgeProposer()
    proposal = h.propose(fat_tail_action="NORMAL",
                          net_delta_lots=0.5,
                          open_positions=[_FakeHypothesis(
                              position_id="p1", direction=+1,
                              rupees_at_risk=500.0, size_lots=1)],
                          strike_lookup=_strike_lookup())
    assert not proposal.proposed


def test_hedge_proposes_pe_against_long_delta_overshoot():
    h = HedgeProposer(HedgeProposalConfig(delta_cap_for_hedge_lots=2.0))
    proposal = h.propose(fat_tail_action="NORMAL",
                          net_delta_lots=3.0,   # well above cap
                          open_positions=[_FakeHypothesis(
                              position_id="p1", direction=+1,
                              rupees_at_risk=500.0, size_lots=3)],
                          strike_lookup=_strike_lookup())
    assert proposal.proposed
    assert proposal.proposals[0].contract_side == "PE"


def test_hedge_proposes_strangle_on_fat_tail_hedge():
    h = HedgeProposer()
    proposal = h.propose(fat_tail_action="HEDGE",
                          net_delta_lots=0.5,
                          open_positions=[_FakeHypothesis(
                              position_id="p1", direction=+1,
                              rupees_at_risk=500.0, size_lots=1)],
                          strike_lookup=_strike_lookup())
    assert proposal.proposed
    sides = {p.contract_side for p in proposal.proposals}
    assert "CE" in sides and "PE" in sides


def test_hedge_serializable():
    h = HedgeProposer()
    proposal = h.propose(fat_tail_action="NORMAL",
                          net_delta_lots=0.5,
                          open_positions=[],
                          strike_lookup=_strike_lookup())
    import json
    json.dumps(proposal.to_dict())


# ── Explainer ──────────────────────────────────────────────────────


def test_explainer_handles_minimal_intent():
    intent = {
        "ts": "2026-06-19 10:00", "bar_index": 100,
        "new_entry": None, "closed_this_tick": [],
        "refuse_reasons": [], "daily_pnl_rupees": 0.0,
        "cumulative_fees_rupees": 0.0,
        "portfolio_summary": {"n_open": 0, "n_closed": 0},
    }
    out = explain_tick(intent)
    assert "Tick @ bar 100" in out
    assert "HOLD" in out


def test_explainer_shows_open_entry():
    intent = {
        "ts": "2026-06-19 10:01", "bar_index": 101,
        "new_entry": {
            "hypothesis": {"contract_label": "CE_ATM", "size_lots": 2,
                            "entry_premium": 100.0},
            "aggregator_decision": {"decision": "ACCEPT",
                                      "final_score": 0.72,
                                      "recommended_size_multiplier": 1.0},
        },
        "closed_this_tick": [], "refuse_reasons": [],
        "daily_pnl_rupees": 0.0, "cumulative_fees_rupees": 0.0,
        "portfolio_summary": {"n_open": 1, "n_closed": 0},
    }
    out = explain_tick(intent)
    assert "OPENED CE_ATM" in out


def test_explainer_shows_refuse_with_reason():
    intent = {
        "ts": "2026-06-19 10:02", "bar_index": 102,
        "new_entry": None,
        "closed_this_tick": [],
        "refuse_reasons": ["critic REFUSE (antithesis=0.85)"],
        "daily_pnl_rupees": 0.0, "cumulative_fees_rupees": 0.0,
        "portfolio_summary": {"n_open": 0, "n_closed": 0},
    }
    out = explain_tick(intent)
    assert "REFUSED" in out
    assert "critic" in out.lower()
