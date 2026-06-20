"""Tests for executor_v4.scenario_web — the probability web."""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4.flow_memory import FlowMemory
from liqpool.research.belief.executor_v4.scenario_web import (
    FAMILY_CHOP,
    FAMILY_DIRECTIONAL,
    FAMILY_FAT_TAIL,
    FAMILY_MANIPULATION,
    ScenarioWeb,
    ScenarioWebConfig,
    STRAT_LONG_CE,
    STRAT_LONG_PE,
    STRAT_IRON_CONDOR,
    STRAT_WAIT,
)
from liqpool.research.belief.executor_v4.substrate import SubstrateState


def _mk(*, bars: int = 100, spot: float = 23000.0,
        thesis: str = "HOLD_BULL", iv: str = "directional_bull",
        bf: str = "bullish_agreement", direction: int = 1,
        bull_score: float = 70.0, bear_score: float = 10.0,
        no_trade: float = 5.0,
        ce_signed: float = 1.5, pe_signed: float = -1.0,
        ce_disp: float = 0.10, pe_disp: float = 0.10,
        ce_epi_label: str = "CE_ATM", ce_epi_level: int = 0,
        ce_def_frac: float = 0.0, pe_def_frac: float = 0.0,
        winding: str = "NO_WINDING",
        ts: pd.Timestamp | None = None) -> dict:
    if ts is None:
        ts = pd.Timestamp("2026-06-19 10:00") + pd.Timedelta(seconds=bars)
    n_each = 11
    n_ce_def = int(ce_def_frac * n_each)
    n_pe_def = int(pe_def_frac * n_each)
    slot_readings = []
    for level in range(-5, 6):
        acc_ce = "defended" if n_ce_def > 0 else "normal"
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "CE", "level": level,
            "label": f"CE_{level}", "moneyness_label": f"CE_{level}",
            "behavior": "gamma_atm" if abs(level) <= 1 else "convex_otm",
            "acceptance": acc_ce, "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "dod_z": ce_signed, "mark_price": 100.0, "is_abnormal": False,
        })
        if n_ce_def > 0:
            n_ce_def -= 1
    for level in range(-5, 6):
        acc_pe = "defended" if n_pe_def > 0 else "normal"
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "PE", "level": level,
            "label": f"PE_{level}", "moneyness_label": f"PE_{level}",
            "behavior": "gamma_atm" if abs(level) <= 1 else "convex_otm",
            "acceptance": acc_pe, "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "dod_z": pe_signed, "mark_price": 100.0, "is_abnormal": False,
        })
        if n_pe_def > 0:
            n_pe_def -= 1
    return {
        "ts": ts, "spot": spot, "bars_seen": bars, "is_warm": True,
        "thesis": {"composite_state": thesis,
                   "bull_thesis_score": bull_score,
                   "bear_thesis_score": bear_score,
                   "no_trade_score": no_trade},
        "iv_state": {"state": iv, "direction": direction, "confidence": 0.8,
                     "clean_mark_fraction": 0.98,
                     "net_intent_z": ce_signed - pe_signed},
        "battlefield": {
            "verdict": bf, "direction": direction, "confidence": 0.8,
            "ce_rail": {"weighted_mean_signed_z": ce_signed,
                        "weighted_mean_abs_z": abs(ce_signed),
                        "dispersion_score": ce_disp,
                        "epicenter_label": ce_epi_label,
                        "epicenter_level": ce_epi_level},
            "pe_rail": {"weighted_mean_signed_z": pe_signed,
                        "weighted_mean_abs_z": abs(pe_signed),
                        "dispersion_score": pe_disp,
                        "epicenter_label": "PE_ATM",
                        "epicenter_level": 0},
        },
        "decision": {"action": "HOLD", "direction": direction,
                     "confidence": 0.8, "trade_allowed": True},
        "winding": {"zone": winding},
        "bull_state": {"state_index": 2},
        "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


def _drive(web: ScenarioWeb, snap: dict,
            substrate: SubstrateState, flow: FlowMemory):
    rich = substrate.observe(snap)
    ev = flow.observe(snap)
    return web.observe(snap, rich, ev, mtf_views=None)


def test_scenario_web_starts_empty():
    web = ScenarioWeb()
    assert web.active() == []
    snap = web.snapshot()
    assert snap.n_active == 0
    assert snap.directional_consensus == 0.0


def test_bull_scenario_spawns_under_bull_thesis():
    web = ScenarioWeb()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(8):
        _drive(web, _mk(bars=100 + i, thesis="HOLD_BULL",
                          bull_score=70, bear_score=8), sub, fm)
    active = web.active()
    names = {s.name for s in active}
    assert "bull_continuation" in names
    bulls = [s for s in active if s.name == "bull_continuation"]
    assert bulls[0].implied_direction == +1
    assert bulls[0].implied_strategy_class == STRAT_LONG_CE


def test_consensus_positive_under_bull_history():
    web = ScenarioWeb()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(15):
        _drive(web, _mk(bars=100 + i, thesis="HOLD_BULL",
                          bull_score=72, bear_score=8,
                          ce_signed=1.5, pe_signed=-0.8), sub, fm)
    snap = web.snapshot()
    assert snap.directional_consensus > 0.05, (
        f"expected positive consensus, got {snap.directional_consensus}"
    )


def test_bear_scenario_spawns_under_bear_thesis():
    web = ScenarioWeb()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(8):
        _drive(web, _mk(bars=100 + i, thesis="HOLD_BEAR",
                          bull_score=8, bear_score=72,
                          ce_signed=-0.5, pe_signed=1.5,
                          iv="directional_bear", bf="bearish_agreement",
                          direction=-1), sub, fm)
    active = web.active()
    names = {s.name for s in active}
    assert "bear_continuation" in names


def test_chop_scenario_spawns_under_neutral_and_unstable():
    web = ScenarioWeb()
    sub = SubstrateState(); fm = FlowMemory()
    # Whip thesis state and bull/bear scores → low stability.
    for i in range(12):
        thesis = "BULL_ENTRY" if i % 2 == 0 else "BEAR_ENTRY"
        bull = 70.0 if i % 2 == 0 else 8.0
        bear = 8.0 if i % 2 == 0 else 70.0
        ce_s = 2.0 if i % 2 == 0 else -2.0
        _drive(web, _mk(bars=100 + i, thesis=thesis,
                          bull_score=bull, bear_score=bear,
                          ce_signed=ce_s, pe_signed=-ce_s), sub, fm)
    names = {s.name for s in web.active()}
    assert any("chop" in n for n in names), (
        f"expected chop family; got {names}"
    )


def test_manipulation_spawns_on_epicenter_migration():
    web = ScenarioWeb()
    sub = SubstrateState(); fm = FlowMemory()
    _drive(web, _mk(bars=100, ce_epi_label="CE_ATM", ce_epi_level=0), sub, fm)
    _drive(web, _mk(bars=101, ce_epi_label="CE_OTM2", ce_epi_level=2), sub, fm)
    _drive(web, _mk(bars=102, ce_epi_label="CE_OTM4", ce_epi_level=4), sub, fm)
    names = {s.name for s in web.active()}
    assert "single_strike_distortion" in names


def test_fat_tail_spawns_on_dirty_iv():
    web = ScenarioWeb()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(5):
        _drive(web, _mk(bars=100 + i, iv="dirty_data", bf="vol_expansion"),
                sub, fm)
    names = {s.name for s in web.active()}
    assert "vol_expansion_or_shock" in names
    assert web.tail_mass() > 0.10


def test_killer_retires_directional_when_thesis_flips():
    web = ScenarioWeb()
    sub = SubstrateState(); fm = FlowMemory()
    # First grow bull continuation.
    for i in range(8):
        _drive(web, _mk(bars=100 + i, thesis="HOLD_BULL", bull_score=70,
                          bear_score=8), sub, fm)
    assert any(s.name == "bull_continuation" for s in web.active())
    # Then flip to bearish + battlefield bear → killer "thesis_state contains BEAR".
    for i in range(5):
        _drive(web, _mk(bars=108 + i, thesis="HOLD_BEAR",
                          bull_score=8, bear_score=72,
                          ce_signed=-1.0, pe_signed=1.0,
                          iv="directional_bear", bf="bearish_agreement",
                          direction=-1), sub, fm)
    # The bull continuation should be retired (gone from active).
    assert not any(s.name == "bull_continuation" for s in web.active())


def test_dominant_pathway_returns_highest_probability():
    web = ScenarioWeb()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(10):
        _drive(web, _mk(bars=100 + i, thesis="HOLD_BULL", bull_score=72,
                          bear_score=8), sub, fm)
    dom = web.currently_dominant_pathway()
    assert dom is not None
    # Top scenarios all sorted descending by probability
    tops = web.top_k(3)
    assert tops[0].current_probability >= tops[-1].current_probability


def test_scenario_web_snapshot_serializable():
    web = ScenarioWeb()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(10):
        _drive(web, _mk(bars=100 + i), sub, fm)
    import json
    snap = web.snapshot()
    json.dumps(snap.to_dict(), default=str)


def test_scenario_max_cap_enforced():
    web = ScenarioWeb(ScenarioWebConfig(max_scenarios=8,
                                          spawn_cooldown_ticks=1))
    sub = SubstrateState(); fm = FlowMemory()
    # Stir through many different conditions to try to spawn many distinct
    # scenarios.
    for i in range(40):
        thesis = ("HOLD_BULL" if i % 4 == 0
                  else "HOLD_BEAR" if i % 4 == 1
                  else "NEUTRAL" if i % 4 == 2 else "BULL_ENTRY")
        iv = "directional_bull" if i % 3 == 0 else "dirty_data"
        _drive(web, _mk(bars=100 + i, thesis=thesis, iv=iv,
                          ce_epi_level=(i % 5 - 2),
                          ce_disp=0.20 + 0.05 * (i % 6)), sub, fm)
    assert len(web.active()) <= 8


def test_tail_mass_alarm_flagged_in_snapshot():
    web = ScenarioWeb(ScenarioWebConfig(tail_mass_alarm=0.20))
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(10):
        _drive(web, _mk(bars=100 + i, iv="common_shock",
                          bf="vol_expansion"), sub, fm)
    snap = web.snapshot()
    assert snap.tail_mass > 0
    if snap.tail_mass >= 0.20:
        assert any("tail" in n for n in snap.notes)


def test_dominant_strategy_class_matches_family():
    web = ScenarioWeb()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(10):
        _drive(web, _mk(bars=100 + i, thesis="HOLD_BULL",
                          bull_score=72, bear_score=8), sub, fm)
    assert web.dominant_strategy_class() == STRAT_LONG_CE
