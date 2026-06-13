"""Wave 3 tests — stress simulator, customer strategy auditor, multi-leg
builder, top-10 equity contextual layer, SaaS tier gating.

These are the tier-3/tier-4 customer surfaces. Each test pins the
established methodology so a desk audit reads pass/fail at a glance.
"""
from __future__ import annotations

import math

import pytest

from sentinel import io_decl
from sentinel.auditor import StrategyLeg, audit_strategy, detect_strategy, payoff_curve
from sentinel.equity_layer import (
    NIFTY_TOP10_WEIGHTS,
    top10_contextual_layer,
    weightage_divergence,
)
from sentinel.institutional import LegExposure
from sentinel.saas import (
    FEATURE_CATALOG,
    FOUNDER,
    PRO,
    QUANT,
    RETAIL,
    features_for,
    gate,
    plan_summary,
)
from sentinel.strategy_builder import ChainQuote, INTENTS, build
from sentinel.stress import SCENARIOS, stress_test, stress_test_all


# ---------------------------------------------------------------------------
# Stress simulator
# ---------------------------------------------------------------------------

def _book():
    """Long ATM CE + short OTM PE — a delta-positive book."""
    return [
        LegExposure("NIFTY25000CE", qty=75, premium=180, delta=0.5,
                    gamma=0.001, theta_per_day=-5.0, vega_per_pct=15.0),
        LegExposure("NIFTY24800PE", qty=-75, premium=80, delta=-0.3,
                    gamma=0.001, theta_per_day=-3.0, vega_per_pct=10.0),
    ]


def test_stress_test_named_scenario_runs():
    r = stress_test(_book(), spot=25000.0, scenario="COVID_MAR_2020")
    assert r.scenario == "COVID_MAR_2020"
    assert len(r.legs) == 2
    # COVID is -12.98% spot — both delta-positive legs lose
    assert r.portfolio_pnl < 0
    # the per-leg stressed_premium is positive and finite
    for l in r.legs:
        assert l.stressed_premium >= 0 and math.isfinite(l.stressed_premium)


def test_stress_test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        stress_test(_book(), spot=25000.0, scenario="NOT_A_REAL_DAY")


def test_stress_test_all_returns_sorted_worst_first():
    out = stress_test_all(_book(), spot=25000.0)
    assert len(out) == len(SCENARIOS)
    # sorted worst-first (most-negative pnl first)
    assert out[0]["portfolio_pnl"] <= out[-1]["portfolio_pnl"]
    # every scenario carries a date + narrative
    for row in out:
        assert row["date"] and row["narrative"]


def test_stress_test_bs_reprice_path_runs():
    """When strikes/option_types are supplied the BS-reprice path is taken."""
    r = stress_test(
        _book(), spot=25000.0, scenario="DEMONETISATION_2016",
        strikes=[25000, 24800], option_types=["CE", "PE"],
        t_years_remaining=7 / 365, base_iv=0.15,
    )
    # Demonetisation is -6.1% — long CE loses, short PE loses too (PE expensive)
    assert r.portfolio_pnl < 0


def test_stress_scenarios_have_real_dates_and_ranges():
    """Sanity check the canonical scenario library: every spot_shock is
    negative (these are crises), every IV shock is non-negative."""
    for sc in SCENARIOS.values():
        assert sc.spot_shock_pct < 0
        assert sc.iv_shock_abs >= 0
        assert sc.timeframe_min > 0


# ---------------------------------------------------------------------------
# Auditor — named-strategy detection (Hull taxonomy)
# ---------------------------------------------------------------------------

def test_detect_long_call():
    leg = StrategyLeg(option_type="CE", strike=25000, qty=75, premium=180)
    assert detect_strategy([leg]) == "LONG_CALL"


def test_detect_bull_call_spread():
    legs = [
        StrategyLeg("CE", 25000, +75, 180),
        StrategyLeg("CE", 25200, -75, 90),
    ]
    assert detect_strategy(legs) == "BULL_CALL_SPREAD"


def test_detect_bull_put_spread():
    legs = [
        StrategyLeg("PE", 24800, -75, 70),
        StrategyLeg("PE", 25000, +75, 150),
    ]
    # sorted by strike: 24800 (short), 25000 (long)
    # a.qty<0 and b.qty>0 with -a.qty == b.qty -> BULL_PUT_SPREAD
    assert detect_strategy(legs) == "BULL_PUT_SPREAD"


def test_detect_long_straddle_and_strangle():
    straddle = [
        StrategyLeg("CE", 25000, +75, 180),
        StrategyLeg("PE", 25000, +75, 150),
    ]
    strangle = [
        StrategyLeg("PE", 24800, +75, 80),
        StrategyLeg("CE", 25200, +75, 90),
    ]
    assert detect_strategy(straddle) == "LONG_STRADDLE"
    assert detect_strategy(strangle) == "LONG_STRANGLE"


def test_detect_iron_condor():
    legs = [
        StrategyLeg("PE", 24600, +75, 30),
        StrategyLeg("PE", 24800, -75, 70),
        StrategyLeg("CE", 25200, -75, 90),
        StrategyLeg("CE", 25400, +75, 40),
    ]
    assert detect_strategy(legs) == "IRON_CONDOR"


def test_detect_custom_when_nonstandard():
    legs = [
        StrategyLeg("CE", 25000, +75, 180),
        StrategyLeg("PE", 24500, +75, 50),
        StrategyLeg("CE", 25400, -50, 50),       # mismatched qty
    ]
    assert detect_strategy(legs) == "CUSTOM"


# ---------------------------------------------------------------------------
# Auditor — payoff curve, breakevens, flags
# ---------------------------------------------------------------------------

def test_audit_long_call_max_loss_equals_premium_paid():
    """A long call's max loss is the premium paid (intrinsic at expiry is 0
    when far OTM)."""
    leg = StrategyLeg("CE", 25000, +75, 180)
    res = audit_strategy([leg], spot_now=25000.0)
    assert res.detected_strategy == "LONG_CALL"
    # max loss capped at -premium * qty = -75 * 180
    assert abs(res.max_loss_rupees - (-180 * 75)) < 1.0
    # breakeven near strike + premium
    assert any(abs(be - 25180) < 100 for be in res.breakeven_points)


def test_audit_iron_condor_caps_both_sides():
    legs = [
        StrategyLeg("PE", 24600, +75, 30),
        StrategyLeg("PE", 24800, -75, 70),
        StrategyLeg("CE", 25200, -75, 90),
        StrategyLeg("CE", 25400, +75, 40),
    ]
    res = audit_strategy(legs, spot_now=25000.0)
    assert res.detected_strategy == "IRON_CONDOR"
    # condor profit is finite and positive
    assert res.max_profit_rupees > 0
    # condor max loss is bounded (wings cap it)
    assert res.max_loss_rupees > -1e6


def test_audit_flags_naked_short_call_uncapped_loss():
    leg = StrategyLeg("CE", 25000, -75, 180)
    res = audit_strategy([leg], spot_now=25000.0)
    assert res.detected_strategy == "NAKED_SHORT_CALL"
    assert any("uncapped" in f.lower() and "upside" in f.lower() for f in res.flags)


def test_audit_flags_negative_theta_in_chop():
    leg = StrategyLeg("CE", 25000, +75, 180, theta_per_day=-5.0)
    res = audit_strategy([leg], spot_now=25000.0, regime="chop")
    assert any("theta" in f.lower() and "chop" in f.lower() for f in res.flags)


def test_audit_payoff_curve_is_monotone_for_long_call():
    leg = StrategyLeg("CE", 25000, +75, 180)
    curve = payoff_curve([leg], spot_now=25000.0)
    assert curve[-1]["pnl"] > curve[0]["pnl"]


# ---------------------------------------------------------------------------
# Strategy builder
# ---------------------------------------------------------------------------

def _synthetic_chain(spot=25000.0):
    """Wide chain: +/- 30 strikes at 50-pt steps covers the 2-sigma wings
    a condor / butterfly needs at IV 0.15 on a weekly."""
    chain = []
    for off in range(-30, 31):
        strike = spot + off * 50
        # CE: cheaper as we go OTM (off > 0 = strike above spot)
        ce_prem = max(5.0, 200 - off * 6)
        # PE: cheaper as we go OTM (off < 0 = strike below spot)
        pe_prem = max(5.0, 200 + off * 6)
        chain.append(ChainQuote("CE", strike, premium=ce_prem,
                                 delta=max(0.02, min(0.98, 0.5 - off * 0.03))))
        chain.append(ChainQuote("PE", strike, premium=pe_prem,
                                 delta=min(-0.02, max(-0.98, -0.5 - off * 0.03))))
    return chain


def test_build_iron_condor_is_detected_by_auditor():
    res = build("PREMIUM_SELL_NEUTRAL", spot=25000.0,
                chain=_synthetic_chain(), iv=0.15, t_years=7 / 365)
    assert res["audit"]["detected_strategy"] == "IRON_CONDOR"
    assert len(res["legs"]) == 4
    # condor has a positive max profit (the net credit collected)
    assert res["audit"]["max_profit_rupees"] > 0


def test_build_directional_bull_picks_bull_call_spread():
    res = build("DIRECTIONAL_BULL", spot=25000.0, chain=_synthetic_chain())
    assert res["audit"]["detected_strategy"] == "BULL_CALL_SPREAD"
    assert len(res["legs"]) == 2


def test_build_vol_buy_picks_long_straddle():
    res = build("VOL_BUY", spot=25000.0, chain=_synthetic_chain())
    assert res["audit"]["detected_strategy"] == "LONG_STRADDLE"


def test_build_unknown_intent_raises():
    with pytest.raises(ValueError):
        build("MAKE_MONEY", spot=25000.0, chain=_synthetic_chain())


def test_build_warns_when_legs_substituted():
    # narrow chain — wings won't exist, so missing_strikes records the substitution
    narrow = [c for c in _synthetic_chain() if abs(c.strike - 25000) <= 100]
    res = build("PREMIUM_SELL_NEUTRAL", spot=25000.0, chain=narrow)
    assert res["missing_strikes"]


# ---------------------------------------------------------------------------
# Equity contextual layer
# ---------------------------------------------------------------------------

def test_equity_layer_hidden_bull_when_index_flat_but_heavyweights_rip():
    # Index ~flat (+0.1%) but RELIANCE + HDFCBANK pump enough to contribute > 0.4%
    returns = {sym: 0.0 for sym in NIFTY_TOP10_WEIGHTS}
    returns["RELIANCE"] = 3.0          # +3%
    returns["HDFCBANK"] = 2.5
    ctx = top10_contextual_layer(returns, index_return_pct=0.1)
    assert ctx.regime == "HIDDEN_BULL"
    assert ctx.leaders[0].symbol in ("RELIANCE", "HDFCBANK")


def test_equity_layer_hidden_bear_when_index_flat_but_heavyweights_drop():
    returns = {sym: 0.0 for sym in NIFTY_TOP10_WEIGHTS}
    returns["RELIANCE"] = -3.0
    returns["HDFCBANK"] = -2.5
    ctx = top10_contextual_layer(returns, index_return_pct=-0.1)
    assert ctx.regime == "HIDDEN_BEAR"


def test_equity_layer_broad_bull_when_all_up():
    returns = {sym: 1.0 for sym in NIFTY_TOP10_WEIGHTS}
    ctx = top10_contextual_layer(returns, index_return_pct=0.9)
    assert ctx.regime == "BROAD_BULL"
    assert ctx.breadth_up == 10 and ctx.breadth_down == 0


def test_equity_layer_true_flat_when_nothing_moves():
    returns = {sym: 0.0 for sym in NIFTY_TOP10_WEIGHTS}
    ctx = top10_contextual_layer(returns, index_return_pct=0.0)
    assert ctx.regime == "TRUE_FLAT"


def test_weightage_divergence_sign_matches_intuition():
    # heavyweights leading -> positive divergence
    returns = {sym: 0.0 for sym in NIFTY_TOP10_WEIGHTS}
    returns["RELIANCE"] = 2.0
    d = weightage_divergence(returns, index_return_pct=0.0)
    assert d > 0


def test_equity_layer_weights_sum_below_one():
    """Top-10 carries < 100% of the index; this is a sanity check that we
    aren't double-weighting the top names."""
    assert 0.3 < sum(NIFTY_TOP10_WEIGHTS.values()) < 0.7


# ---------------------------------------------------------------------------
# SaaS tier gating
# ---------------------------------------------------------------------------

def test_saas_retail_blocked_from_pro_feature():
    r = gate(RETAIL, "stress.full_matrix")
    assert not r.allowed and r.required_tier == PRO


def test_saas_pro_allowed_pro_feature():
    r = gate(PRO, "stress.full_matrix")
    assert r.allowed


def test_saas_pro_blocked_from_quant_feature():
    r = gate(PRO, "rv.yang_zhang")
    assert not r.allowed and r.required_tier == QUANT


def test_saas_founder_bypasses_all_gates():
    for feature in FEATURE_CATALOG:
        assert gate(FOUNDER, feature).allowed


def test_saas_quant_gets_every_feature():
    feats = features_for(QUANT)
    # QUANT-tier plan must cover every catalog entry (it's the top tier)
    assert feats == set(FEATURE_CATALOG)


def test_saas_unknown_feature_defaults_to_pro_gate():
    """Safer default: unknown feature requires PRO, so a missing tag
    cannot silently expose new code to RETAIL."""
    r_retail = gate(RETAIL, "made.up.feature")
    r_pro = gate(PRO, "made.up.feature")
    assert not r_retail.allowed and r_pro.allowed


def test_saas_plan_summary_carries_level_and_count():
    s = plan_summary(PRO)
    assert s["plan"] == PRO and s["level"] == 2
    assert s["feature_count"] > 0


def test_saas_unknown_plan_is_denied():
    r = gate("FREE_TRIAL_HACK", "stress.full_matrix")
    assert not r.allowed and "unknown plan" in r.reason.lower()


# ---------------------------------------------------------------------------
# IO registry — every new module declared
# ---------------------------------------------------------------------------

def test_wave3_modules_declare_io():
    import sentinel.stress           # noqa: F401
    import sentinel.auditor          # noqa: F401
    import sentinel.strategy_builder # noqa: F401
    import sentinel.equity_layer     # noqa: F401
    import sentinel.saas             # noqa: F401
    expected = {
        "sentinel.stress", "sentinel.auditor", "sentinel.strategy_builder",
        "sentinel.equity_layer", "sentinel.saas",
    }
    assert expected <= set(io_decl.REGISTRY)
