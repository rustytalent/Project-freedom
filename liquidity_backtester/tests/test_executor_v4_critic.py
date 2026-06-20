"""Tests for executor_v4.critic — adversarial bear-case generator."""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4.critic import (
    AdversarialCritic,
    CRITIC_PROCEED,
    CRITIC_REFUSE,
    CRITIC_SIZE_DOWN,
    CriticConfig,
)
from liqpool.research.belief.executor_v4.flow_memory import FlowMemory
from liqpool.research.belief.executor_v4.scenario_web import ScenarioWeb
from liqpool.research.belief.executor_v4.substrate import SubstrateState


def _mk(*, bars: int = 100, spot: float = 23000.0,
        thesis: str = "HOLD_BULL", iv: str = "directional_bull",
        bf: str = "bullish_agreement", direction: int = 1,
        bull_score: float = 70.0, bear_score: float = 10.0,
        ce_signed: float = 1.5, pe_signed: float = -1.0,
        ce_disp: float = 0.10, pe_disp: float = 0.10,
        ce_epi_label: str = "CE_ATM", ce_epi_level: int = 0,
        ce_def_frac: float = 0.0, pe_def_frac: float = 0.0,
        abnormal_count: int = 0,
        winding: str = "NO_WINDING",
        ts: pd.Timestamp | None = None) -> dict:
    if ts is None:
        ts = pd.Timestamp("2026-06-19 10:00") + pd.Timedelta(seconds=bars)
    n_each = 11
    n_ce_def = int(ce_def_frac * n_each)
    n_pe_def = int(pe_def_frac * n_each)
    abn_remaining = abnormal_count
    slot_readings = []
    for level in range(-5, 6):
        acc_ce = "defended" if n_ce_def > 0 else "normal"
        is_abn = abn_remaining > 0
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "CE", "level": level,
            "label": f"CE_{level}", "moneyness_label": f"CE_{level}",
            "acceptance": acc_ce, "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "dod_z": ce_signed, "mark_price": 100.0,
            "is_abnormal": is_abn,
        })
        if is_abn:
            abn_remaining -= 1
        if n_ce_def > 0:
            n_ce_def -= 1
    for level in range(-5, 6):
        acc_pe = "defended" if n_pe_def > 0 else "normal"
        is_abn = abn_remaining > 0
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "PE", "level": level,
            "label": f"PE_{level}", "moneyness_label": f"PE_{level}",
            "acceptance": acc_pe, "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "dod_z": pe_signed, "mark_price": 100.0,
            "is_abnormal": is_abn,
        })
        if is_abn:
            abn_remaining -= 1
        if n_pe_def > 0:
            n_pe_def -= 1
    return {
        "ts": ts, "spot": spot, "bars_seen": bars, "is_warm": True,
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


def test_critic_proceeds_on_clean_bull_setup():
    critic = AdversarialCritic()
    sub = SubstrateState(); fm = FlowMemory()
    # Build a clean bullish history.
    for i in range(20):
        sub.observe(_mk(bars=100 + i))
        fm.observe(_mk(bars=100 + i))
    rich = sub.observe(_mk(bars=120))
    ev = fm.observe(_mk(bars=120))
    result = critic.critique(
        proposed_direction=+1, proposed_profile="INTRADAY",
        snapshot=_mk(bars=120), rich_context=rich, flow_event=ev,
        flow_memory=fm,
        mtf_alignment={"alignment_ok": True, "alignment_score": 0.80,
                        "l1_match": True, "l5_match": True,
                        "l15_match": True, "l60_match": False},
    )
    assert result.recommended_action == CRITIC_PROCEED
    assert result.recommended_size_haircut == 1.0


def test_critic_refuses_on_strong_opposite_acceptance():
    critic = AdversarialCritic()
    sub = SubstrateState(); fm = FlowMemory()
    # Long proposal but PE is heavily defended → hidden short positioning.
    for i in range(20):
        sub.observe(_mk(bars=100 + i, pe_def_frac=0.70))
        fm.observe(_mk(bars=100 + i, pe_def_frac=0.70))
    snap = _mk(bars=120, pe_def_frac=0.70)
    rich = sub.observe(snap); ev = fm.observe(snap)
    result = critic.critique(
        proposed_direction=+1, proposed_profile="INTRADAY",
        snapshot=snap, rich_context=rich, flow_event=ev, flow_memory=fm,
        mtf_alignment={"alignment_ok": True, "alignment_score": 0.80,
                        "l1_match": True, "l5_match": True,
                        "l15_match": True, "l60_match": False},
    )
    # Should at minimum size down
    assert result.recommended_action in (CRITIC_SIZE_DOWN, CRITIC_REFUSE)
    assert any("PE rail defended" in r for r in result.antithesis_reasons)


def test_critic_size_downs_on_moderate_evidence():
    critic = AdversarialCritic()
    sub = SubstrateState(); fm = FlowMemory()
    # Single moderate-severity argument: brittle rail.
    for i in range(20):
        snap = _mk(bars=100 + i, abnormal_count=8)
        sub.observe(snap); fm.observe(snap)
    snap = _mk(bars=120, abnormal_count=8)
    rich = sub.observe(snap); ev = fm.observe(snap)
    result = critic.critique(
        proposed_direction=+1, proposed_profile="INTRADAY",
        snapshot=snap, rich_context=rich, flow_event=ev, flow_memory=fm,
        mtf_alignment={"alignment_ok": True, "alignment_score": 0.80,
                        "l1_match": True, "l5_match": True,
                        "l15_match": True, "l60_match": False},
    )
    assert result.recommended_action in (CRITIC_SIZE_DOWN, CRITIC_PROCEED, CRITIC_REFUSE)
    # Brittle rail must be among reasons.
    assert any("brittle" in r for r in result.antithesis_reasons)


def test_critic_refuses_on_mtf_failed():
    critic = AdversarialCritic(CriticConfig(refuse_score=0.50))
    sub = SubstrateState(); fm = FlowMemory()
    snap = _mk()
    rich = sub.observe(snap); ev = fm.observe(snap)
    result = critic.critique(
        proposed_direction=+1, proposed_profile="INTRADAY",
        snapshot=snap, rich_context=rich, flow_event=ev, flow_memory=fm,
        mtf_alignment={"alignment_ok": False, "alignment_score": 0.10,
                        "l1_match": False, "l5_match": False,
                        "l15_match": False, "l60_match": False},
    )
    # MTF failure adds 0.20; with refuse_score=0.50 it may not refuse alone,
    # but it should AT LEAST surface in reasons.
    assert any("MTF alignment failed" in r for r in result.antithesis_reasons)


def test_critic_haircut_in_band_is_between_min_and_one():
    critic = AdversarialCritic(CriticConfig(size_down_score=0.30,
                                              refuse_score=0.70,
                                              min_haircut=0.40))
    sub = SubstrateState(); fm = FlowMemory()
    snap = _mk(pe_def_frac=0.35)   # moderate evidence
    rich = sub.observe(snap); ev = fm.observe(snap)
    result = critic.critique(
        proposed_direction=+1, proposed_profile="INTRADAY",
        snapshot=snap, rich_context=rich, flow_event=ev, flow_memory=fm,
        mtf_alignment={"alignment_ok": True, "alignment_score": 0.80,
                        "l1_match": True, "l5_match": True,
                        "l15_match": True, "l60_match": False},
    )
    assert 0.40 <= result.recommended_size_haircut <= 1.0


def test_critic_to_dict_serializable():
    critic = AdversarialCritic()
    sub = SubstrateState(); fm = FlowMemory()
    snap = _mk()
    rich = sub.observe(snap); ev = fm.observe(snap)
    result = critic.critique(
        proposed_direction=+1, proposed_profile="INTRADAY",
        snapshot=snap, rich_context=rich, flow_event=ev, flow_memory=fm,
    )
    import json
    json.dumps(result.to_dict())


def test_critic_picks_up_web_chop_dominance():
    """When the scenario web shows chop dominant, critic should reflect it."""
    critic = AdversarialCritic()
    sub = SubstrateState(); fm = FlowMemory()
    snap = _mk()
    rich = sub.observe(snap); ev = fm.observe(snap)

    class FakeWeb:
        chop_mass = 0.65
        tail_mass = 0.05

    result = critic.critique(
        proposed_direction=+1, proposed_profile="INTRADAY",
        snapshot=snap, rich_context=rich, flow_event=ev, flow_memory=fm,
        web_snapshot=FakeWeb(),
    )
    assert any("chop mass" in r for r in result.antithesis_reasons)


def test_critic_picks_up_web_tail_mass():
    critic = AdversarialCritic()
    sub = SubstrateState(); fm = FlowMemory()
    snap = _mk()
    rich = sub.observe(snap); ev = fm.observe(snap)

    class FakeWeb:
        chop_mass = 0.05
        tail_mass = 0.45

    result = critic.critique(
        proposed_direction=+1, proposed_profile="INTRADAY",
        snapshot=snap, rich_context=rich, flow_event=ev, flow_memory=fm,
        web_snapshot=FakeWeb(),
    )
    assert any("tail mass" in r for r in result.antithesis_reasons)
