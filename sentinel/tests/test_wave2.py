"""Wave 2 tests — scientists, curator, calibration, scenario engine,
leakage guard, io map, reports. The laboratory loop, end to end."""
from __future__ import annotations

from pathlib import Path

import pytest

from sentinel import io_decl
from sentinel.calibration import CalibrationEngine
from sentinel.curator import Curator
from sentinel.leakage_guard import LeakageGuard
from sentinel.moneyness import ContractRef, moneyness_key
from sentinel.reports import write_artifacts, connection_map
from sentinel.scenario_engine import (
    ScenarioInput, evaluate, payoff_curve, pnl_heatmap, premium_sensitivity,
)
from sentinel.scientists import (
    ContractLive, MarketSnapshot, ScientistPool, SecondPullbackScientist,
)
from sentinel.shadow_ledger import (
    Context, Hypothesis, Identity, ShadowLedger, new_event, read_session,
    KIND_VIRTUAL,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _live(otype, offset, strike, spot, premium, delta, gamma=0.001, theta=-5.0):
    ref = ContractRef(f"NIFTY{int(strike)}{otype}", "NIFTY", otype, strike,
                      moneyness_key("NIFTY", otype, strike, spot, 50))
    return ContractLive(ref=ref, premium=premium, bid=premium-0.5,
                        ask=premium+0.5, spread=1.0, volume=1e6, oi=1e6,
                        iv=0.14, delta=delta, gamma=gamma, theta=theta)


def _snapshot(spot=25000.0, **ctx):
    uni = {}
    for off in range(-5, 6):
        ce_strike = 25000 + off * 50
        pe_strike = 25000 - off * 50
        uni_ce = _live("CE", off, ce_strike, spot, max(5, 180 - off*30),
                       delta=max(0.05, 0.5 - off*0.08))
        uni_pe = _live("PE", off, pe_strike, spot, max(5, 150 - off*25),
                       delta=min(-0.05, -0.5 + off*0.08))
        uni[uni_ce.ref.key.as_str()] = uni_ce
        uni[uni_pe.ref.key.as_str()] = uni_pe
    return MarketSnapshot(session="2026-06-13", spot=spot, universe=uni, **ctx)


# ---------------------------------------------------------------------------
# Scientists
# ---------------------------------------------------------------------------

def test_second_pullback_fires_only_with_context():
    sci = SecondPullbackScientist()
    # no context -> nothing
    assert sci.generate(_snapshot()) == []
    # uptrend + 2 pullbacks -> one validated candidate
    out = sci.generate(_snapshot(trend="up", pullback_count=2))
    assert len(out) == 1
    ev = out[0]
    assert ev.hypothesis.reason_codes              # carries its validation
    assert ev.hypothesis.confidence > 0
    assert ev.identity.moneyness_key == "NIFTY:CE:+0"


def test_scientist_pool_runs_all_and_carries_reasons():
    pool = ScientistPool()
    snap = _snapshot(trend="up", pullback_count=2, gamma_expansion_early=True,
                     dist_from_vwap_atr=0.0, minutes_to_expiry=300)
    events = pool.run(snap)
    assert len(events) >= 2
    # every emitted trade MUST explain itself (the founder's rule)
    assert all(e.hypothesis.reason_codes for e in events)


def test_chop_scientist_needs_stretch():
    from sentinel.scientists import ChopMeanRevScientist
    sci = ChopMeanRevScientist()
    assert sci.generate(_snapshot(trend="chop", dist_from_vwap_atr=0.3)) == []
    out = sci.generate(_snapshot(trend="chop", dist_from_vwap_atr=1.5))
    assert len(out) == 1 and out[0].identity.option_type == "PE"  # stretched up -> fade with PE


# ---------------------------------------------------------------------------
# Full laboratory loop: scientists -> ledger -> journey -> curator -> calib
# ---------------------------------------------------------------------------

def _seed_completed_session(tmp_path, n=40, target_too_far=True):
    """Write n completed journeys with a KNOWN target bias so the curator's
    finding is deterministic. With target_too_far, the suggested target is
    always above the achieved MFE -> curator should flag 'target too far'."""
    led = ShadowLedger(tmp_path)
    for i in range(n):
        entry = 100.0
        ident = Identity(instrument=f"S{i}", option_type="CE", strike=25000,
                         expiry=None, moneyness_key="NIFTY:CE:+0",
                         underlying_price=25000, premium=entry, delta=0.5)
        # MFE reaches 130; target set at 160 (too far) or 120 (reachable)
        target = 160.0 if target_too_far else 120.0
        hyp = Hypothesis(expected_underlying_move=60, expected_premium=target,
                         expected_horizon_min=15, suggested_entry=entry,
                         suggested_stop=80.0, suggested_target=target,
                         confidence=0.6, reason_codes=["test"])
        ev = new_event(KIND_VIRTUAL, "2026-06-13", ident, Context(), hyp)
        ShadowLedger.stamp_journey(ev, 1.0, 110)
        ShadowLedger.stamp_journey(ev, 5.0, 130)    # MFE = 130
        ShadowLedger.stamp_journey(ev, 60.0, 115)   # completes
        led.write(ev)
    return read_session(tmp_path, "2026-06-13")


def test_curator_detects_target_too_far(tmp_path):
    rows = _seed_completed_session(tmp_path, n=40, target_too_far=True)
    rep = Curator().judge_rows("2026-06-13", rows)
    assert rep.n_complete == 40
    bias = next(f for f in rep.findings if f.key == "target_distance_bias")
    assert bias.value > 0          # target sits beyond the achieved peak
    assert bias.confidence == "firm"   # n=40 >= 30


def test_calibration_accepts_when_rerun_improves(tmp_path):
    """Target too far -> pulling it in should improve the re-run expectancy
    (MFE 130 reachable at a 120-ish target but not at 160). Calibration
    must ACCEPT the tightening."""
    rows = _seed_completed_session(tmp_path, n=40, target_too_far=True)
    rep = Curator().judge_rows("2026-06-13", rows)
    eng = CalibrationEngine(knobs_path=tmp_path / "knobs.json")
    res = eng.calibrate(rep, rows)
    assert res.after_expectancy >= res.before_expectancy
    # The tightened target knob should have been accepted.
    assert eng.knobs.target_distance_mult < 1.0
    assert any("ACCEPTED" in n for n in res.notes)


def test_calibration_rejects_when_no_improvement(tmp_path):
    """Targets already reachable -> no firm bias to act on -> nothing
    accepted; knobs unchanged."""
    rows = _seed_completed_session(tmp_path, n=40, target_too_far=False)
    rep = Curator().judge_rows("2026-06-13", rows)
    eng = CalibrationEngine(knobs_path=tmp_path / "knobs.json")
    res = eng.calibrate(rep, rows)
    assert eng.knobs.target_distance_mult == 1.0    # untouched


def test_calibration_ignores_noise_findings(tmp_path):
    """With only 5 journeys, the target bias is 'noise' and must NOT act,
    even though the bias is present."""
    rows = _seed_completed_session(tmp_path, n=5, target_too_far=True)
    rep = Curator().judge_rows("2026-06-13", rows)
    eng = CalibrationEngine(knobs_path=tmp_path / "knobs.json")
    res = eng.calibrate(rep, rows)
    assert eng.knobs.target_distance_mult == 1.0
    assert any("noise" in n for n in res.notes)


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

def test_leakage_clean_then_earned_then_suspicious(tmp_path):
    g = LeakageGuard(history_path=tmp_path / "leak.json")
    v1 = g.check("hit_rate", 0.52, expectancy_improved=False)
    assert v1.verdict == "CLEAN"                     # first obs
    v2 = g.check("hit_rate", 0.55, expectancy_improved=False)
    assert v2.verdict == "CLEAN"                     # small move
    # big jump + expectancy improved -> EARNED
    v3 = g.check("hit_rate", 0.80, expectancy_improved=True)
    assert v3.verdict == "EARNED"
    # big jump + expectancy NOT improved -> SUSPICIOUS
    v4 = g.check("hit_rate", 0.40, expectancy_improved=False)
    assert v4.verdict == "SUSPICIOUS"


# ---------------------------------------------------------------------------
# Scenario engine
# ---------------------------------------------------------------------------

def test_scenario_evaluate_outputs_full_block():
    inp = ScenarioInput(spot=25000, strike=25000, option_type="CE",
                        entry_premium=180, quantity=75, t_years=7/365,
                        underlying_move=100, time_passed_min=15, iv_move=0.0,
                        target=230, stop=150)
    out = evaluate(inp)
    # spot +100 on a call -> positive expected P&L
    assert out["expected_pnl"] > 0
    # waterfall decomposition sums (approximately) to total P&L
    wf_sum = sum(x["pnl"] for x in out["risk_waterfall"])
    assert abs(wf_sum - out["expected_pnl"]) < 5
    # ITM/ATM/OTM comparison present
    zones = {z["zone"] for z in out["itm_atm_otm"]}
    assert {"ITM", "ATM", "OTM"} <= zones
    assert isinstance(out["best_suggestion"], str)


def test_scenario_theta_bleed_negative_for_long():
    inp = ScenarioInput(spot=25000, strike=25000, option_type="CE",
                        entry_premium=180, quantity=75, t_years=2/365,
                        underlying_move=0, time_passed_min=120, iv_move=0.0)
    out = evaluate(inp)
    assert out["theta_bleed_pnl"] < 0      # time decay hurts a long


def test_scenario_payoff_curve_monotone_for_call():
    inp = ScenarioInput(spot=25000, strike=25000, option_type="CE",
                        entry_premium=180, quantity=75, t_years=7/365)
    curve = payoff_curve(inp, span_pts=300, steps=31)
    pnls = [p["pnl"] for p in curve]
    assert pnls[-1] > pnls[0]               # calls gain as spot rises


def test_scenario_heatmap_and_sensitivity_shapes():
    inp = ScenarioInput(spot=25000, strike=25000, option_type="CE",
                        entry_premium=180, quantity=75, t_years=7/365)
    hm = pnl_heatmap(inp, spot_steps=9, time_steps=6)
    assert len(hm["spots"]) == 9 and len(hm["minutes"]) == 6
    assert len(hm["pnl_grid"]) == 6 and len(hm["pnl_grid"][0]) == 9
    sens = premium_sensitivity(inp)
    drivers = {s["driver"] for s in sens}
    assert any("spot" in d for d in drivers) and any("vol" in d for d in drivers)


# ---------------------------------------------------------------------------
# IO map + reports
# ---------------------------------------------------------------------------

def test_io_registry_has_all_modules():
    import sentinel.reports  # noqa: F401  ensures all declare() ran
    names = set(io_decl.REGISTRY)
    expected = {
        "sentinel.greeks", "sentinel.moneyness", "sentinel.shadow_ledger",
        "sentinel.orchestration", "sentinel.profit_lock", "sentinel.portfolio",
        "sentinel.trails", "sentinel.advisor", "sentinel.kite_client",
        "sentinel.scientists", "sentinel.curator", "sentinel.calibration",
        "sentinel.scenario_engine", "sentinel.leakage_guard",
    }
    assert expected <= names


def test_connection_map_renders():
    cm = connection_map()
    assert "SENTINEL CONNECTION MAP" in cm["text"]
    assert cm["dot"].startswith("digraph Sentinel")
    # every produces_for edge should appear in the dot graph
    assert "->" in cm["dot"]


def test_write_artifacts_produces_three_files(tmp_path):
    _seed_completed_session(tmp_path, n=35, target_too_far=True)
    out = write_artifacts(tmp_path, "2026-06-13", tmp_path / "art")
    assert out["system_report"].exists()
    assert out["connection_map_txt"].exists()
    assert out["connection_map_dot"].exists()
    report = out["system_report"].read_text()
    # with real data the report shows findings + a self-assessment
    assert "Curator findings" in report
    assert "Self-assessment" in report
    assert "target_distance_bias" in report
