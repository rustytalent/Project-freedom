"""Tests for Belief Engine Phase 5 — multi-strike battlefield + IV state classifier.

Pins the founder's six battlefield scenarios:
  * Bullish agreement: CE rail broadly strong, PE rail broadly weak.
  * Bearish agreement: PE rail broadly strong, CE rail broadly weak.
  * Vol expansion (common shock): BOTH rails strong same direction.
  * Single-strike distortion: one strike screaming while the rest sits quiet.
  * Liquidity distortion: clean direction but spreads collapsed.
  * Dirty data: too many invalid marks.
And the acceptance-skew read: broad CE-defended + PE-rejected pattern is
itself the directional belief read even when the rail-mean is quiet.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import pytest

from liqpool.research.belief.battlefield import (
    FIELD_BEARISH_AGREEMENT,
    FIELD_BULLISH_AGREEMENT,
    FIELD_QUIET,
    FIELD_SINGLE_DISTORTION,
    FIELD_VOL_EXPANSION,
    RAIL_QUIET,
    RAIL_STRONG_BEAR,
    RAIL_STRONG_BULL,
    SlotReading,
    battlefield_snapshot,
    make_readings,
)
from liqpool.research.belief.iv_state import (
    IV_COMMON_SHOCK,
    IV_DIRECTIONAL_BEAR,
    IV_DIRECTIONAL_BULL,
    IV_DIRTY_DATA,
    IV_LIQUIDITY_DISTORTION,
    IV_NEUTRAL,
    IV_VOL_CONTRACTION,
    classify_iv_state,
)
from liqpool.research.belief.moneyness import build_chain_slots


SPOT = 23000.0
STEP = 50.0


def _battlefield_slots():
    strikes = [SPOT + STEP * i for i in range(-5, 6)]
    contracts = [(k, "CE") for k in strikes] + [(k, "PE") for k in strikes]
    slots = build_chain_slots(SPOT, contracts, STEP)
    ce = [s for s in slots if s.option_type == "CE"]
    pe = [s for s in slots if s.option_type == "PE"]
    return ce, pe


def _readings(ce_z: Sequence[float], pe_z: Sequence[float],
              *, friend: float = 1.0,
              ce_acc: Optional[Sequence[str]] = None,
              pe_acc: Optional[Sequence[str]] = None) -> List[SlotReading]:
    ce, pe = _battlefield_slots()
    ce_acc = list(ce_acc) if ce_acc else ["normal"] * len(ce)
    pe_acc = list(pe_acc) if pe_acc else ["normal"] * len(pe)
    out = []
    for s, z, a in zip(ce, ce_z, ce_acc):
        out.append(SlotReading(slot=s, dod_z=float(z),
                                is_abnormal=abs(z) > 1.0,
                                friendliness=friend, acceptance=a))
    for s, z, a in zip(pe, pe_z, pe_acc):
        out.append(SlotReading(slot=s, dod_z=float(z),
                                is_abnormal=abs(z) > 1.0,
                                friendliness=friend, acceptance=a))
    return out


# ─────────────────────────────────────────────────────────────────
# Battlefield core scenarios
# ─────────────────────────────────────────────────────────────────

def test_bullish_agreement_when_ce_strong_pe_weak():
    snap = battlefield_snapshot(_readings([2.0] * 11, [-2.0] * 11))
    assert snap.verdict == FIELD_BULLISH_AGREEMENT
    assert snap.direction == +1
    assert snap.confidence > 0.5
    # PE rail signed mean negative reads as bullish belief (puts being sold).
    assert snap.ce_rail.state == RAIL_STRONG_BULL
    assert snap.pe_rail.state == RAIL_STRONG_BULL


def test_bearish_agreement_when_pe_strong_ce_weak():
    snap = battlefield_snapshot(_readings([-2.0] * 11, [2.0] * 11))
    assert snap.verdict == FIELD_BEARISH_AGREEMENT
    assert snap.direction == -1
    assert snap.ce_rail.state == RAIL_STRONG_BEAR
    assert snap.pe_rail.state == RAIL_STRONG_BEAR


def test_vol_expansion_when_both_rails_strong_same_way():
    snap = battlefield_snapshot(_readings([2.5] * 11, [2.5] * 11))
    assert snap.verdict == FIELD_VOL_EXPANSION
    assert snap.direction == 0


def test_quiet_when_both_rails_silent():
    snap = battlefield_snapshot(_readings([0.0] * 11, [0.0] * 11))
    assert snap.verdict == FIELD_QUIET
    assert snap.direction == 0
    assert snap.ce_rail.state == RAIL_QUIET
    assert snap.pe_rail.state == RAIL_QUIET


def test_single_strike_distortion_detected():
    """One ATM CE spike while the rest of the rail is silent — the quiet
    rail-mean is itself the distortion fingerprint."""
    ce_z = [0.1] * 11
    ce_z[5] = 6.0    # ATM CE
    snap = battlefield_snapshot(_readings(ce_z, [0.0] * 11))
    assert snap.verdict == FIELD_SINGLE_DISTORTION
    assert snap.direction == 0
    assert "CE_ATM" in snap.note


def test_dispersion_blocks_agreement_call():
    """If the CE rail is otherwise silent and one strike screams, the
    dispersion should refuse to call clean agreement even when the
    opposite rail looks weak."""
    ce_z = [0.1] * 11
    ce_z[3] = 5.0
    snap = battlefield_snapshot(_readings(ce_z, [-2.0] * 11))
    # CE rail is dominated by one slot → distortion takes priority.
    assert snap.verdict == FIELD_SINGLE_DISTORTION


# ─────────────────────────────────────────────────────────────────
# Rail summary internals
# ─────────────────────────────────────────────────────────────────

def test_rail_epicenter_is_max_abs_z():
    ce_z = [0.0] * 11
    ce_z[1] = -3.5   # CE_ITM4 (level -4)
    snap = battlefield_snapshot(_readings(ce_z, [0.0] * 11))
    # The epicenter slot label should reflect the strike with the max |z|.
    assert snap.ce_rail.epicenter_label.startswith("CE_")
    # That spike is the only abnormal slot → fraction_abnormal is small.
    assert snap.ce_rail.fraction_abnormal < 0.2


def test_make_readings_accepts_tuples_and_objects():
    ce, _ = _battlefield_slots()
    rs = make_readings([(ce[0], 1.5), SlotReading(slot=ce[1], dod_z=0.2)])
    assert len(rs) == 2
    assert rs[0].is_abnormal
    assert not rs[1].is_abnormal


# ─────────────────────────────────────────────────────────────────
# IV state classifier
# ─────────────────────────────────────────────────────────────────

def test_iv_state_dirty_data_dominates():
    r = _readings([2.0] * 11, [-2.0] * 11)
    iv = classify_iv_state(battlefield_snapshot(r), r, clean_mark_fraction=0.3)
    assert iv.state == IV_DIRTY_DATA
    assert iv.direction == 0


def test_iv_state_liquidity_distortion_overrides_direction():
    r = _readings([2.0] * 11, [-2.0] * 11, friend=0.2)
    iv = classify_iv_state(battlefield_snapshot(r), r, clean_mark_fraction=1.0)
    assert iv.state == IV_LIQUIDITY_DISTORTION
    assert iv.direction == 0


def test_iv_state_common_shock_on_vol_expansion():
    r = _readings([2.5] * 11, [2.5] * 11)
    iv = classify_iv_state(battlefield_snapshot(r), r, clean_mark_fraction=1.0)
    assert iv.state == IV_COMMON_SHOCK
    assert iv.direction == 0


def test_iv_state_directional_bull_on_bullish_agreement():
    r = _readings([2.0] * 11, [-2.0] * 11)
    iv = classify_iv_state(battlefield_snapshot(r), r, clean_mark_fraction=1.0)
    assert iv.state == IV_DIRECTIONAL_BULL
    assert iv.direction == +1


def test_iv_state_directional_bear_on_bearish_agreement():
    r = _readings([-2.0] * 11, [2.0] * 11)
    iv = classify_iv_state(battlefield_snapshot(r), r, clean_mark_fraction=1.0)
    assert iv.state == IV_DIRECTIONAL_BEAR
    assert iv.direction == -1


def test_iv_state_acceptance_skew_fires_even_when_rail_mean_quiet():
    """The founder's exact case: rail-mean z is small (≈0.5) but broad
    acceptance pattern (CE defended in many strikes, PE rejected in many)
    is itself the bullish belief read."""
    ce_acc = ["defended"] * 5 + ["normal"] * 6
    pe_acc = ["rejected"] * 5 + ["normal"] * 6
    r = _readings([0.5] * 11, [0.0] * 11,
                  ce_acc=ce_acc, pe_acc=pe_acc)
    iv = classify_iv_state(battlefield_snapshot(r), r, clean_mark_fraction=1.0)
    assert iv.state == IV_DIRECTIONAL_BULL
    assert iv.direction == +1


def test_iv_state_acceptance_skew_bearish():
    ce_acc = ["rejected"] * 5 + ["normal"] * 6
    pe_acc = ["defended"] * 5 + ["normal"] * 6
    r = _readings([0.0] * 11, [0.5] * 11,
                  ce_acc=ce_acc, pe_acc=pe_acc)
    iv = classify_iv_state(battlefield_snapshot(r), r, clean_mark_fraction=1.0)
    assert iv.state == IV_DIRECTIONAL_BEAR
    assert iv.direction == -1


def test_iv_state_vol_contraction_on_quiet_clean():
    r = _readings([0.0] * 11, [0.0] * 11, friend=0.95)
    iv = classify_iv_state(battlefield_snapshot(r), r, clean_mark_fraction=1.0)
    assert iv.state == IV_VOL_CONTRACTION


def test_iv_state_empty_readings_returns_dirty():
    """Defensive — no slot data must not crash; treat as dirty data."""
    iv = classify_iv_state(battlefield_snapshot([]), [], clean_mark_fraction=1.0)
    assert iv.state == IV_DIRTY_DATA


def test_iv_state_dict_is_serializable():
    r = _readings([2.0] * 11, [-2.0] * 11)
    iv = classify_iv_state(battlefield_snapshot(r), r, clean_mark_fraction=1.0)
    import json
    json.dumps(iv.to_dict())


def test_battlefield_snapshot_dict_is_serializable():
    r = _readings([2.0] * 11, [-2.0] * 11)
    snap = battlefield_snapshot(r)
    import json
    json.dumps(snap.to_dict())
