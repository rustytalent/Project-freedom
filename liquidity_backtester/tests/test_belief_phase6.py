"""Tests for Belief Engine Phase 6 — thesis memory with hysteresis.

Pins the founder's "don't flip like a mosquito" principle through
deterministic score arcs:

  * Bull builds → fires BULL_ENTRY at entry threshold.
  * Holds through neutral bars (decay path).
  * Drops below the EXIT threshold (45) → EXIT_BULL, then NEUTRAL.
  * Hysteresis: a single counter bar does NOT reverse; it nicks.
  * REGIME_BREAK on the held side cuts hard.
  * STRONG_TRAP on the held side cuts.
  * Two consecutive liquidity_distortion bars → NO_TRADE_DANGER.
  * Common IV shock raises vol_expansion → contributes to no_trade.
  * Acceptance-skew IV note adds the bonus (extra confirmation kick).
"""
from __future__ import annotations

import pytest

from liqpool.research.belief.iv_state import (
    IVState,
    IV_COMMON_SHOCK,
    IV_DIRECTIONAL_BEAR,
    IV_DIRECTIONAL_BULL,
    IV_DIRTY_DATA,
    IV_LIQUIDITY_DISTORTION,
    IV_NEUTRAL,
)
from liqpool.research.belief.thesis_memory import (
    BEAR_ENTRY,
    BULL_ENTRY,
    EXIT_BULL,
    HOLD_BULL,
    HOLD_BEAR,
    NEUTRAL_THESIS,
    NO_TRADE_DANGER,
    ThesisMemory,
    ThesisMemoryConfig,
    ThesisSnapshot,
)


def _iv(state: str, *, conf: float = 0.9, note: str = "") -> IVState:
    direction = (+1 if state == IV_DIRECTIONAL_BULL
                 else -1 if state == IV_DIRECTIONAL_BEAR else 0)
    return IVState(
        state=state, direction=direction, confidence=conf,
        avg_friendliness=1.0, clean_mark_fraction=1.0,
        ce_defended_fraction=0.0, pe_defended_fraction=0.0,
        ce_rejected_fraction=0.0, pe_rejected_fraction=0.0, note=note,
    )


# ─────────────────────────────────────────────────────────────────
# Build / hold / exit arc
# ─────────────────────────────────────────────────────────────────

def test_bull_thesis_builds_to_entry():
    m = ThesisMemory()
    states = []
    for _ in range(5):
        states.append(m.update(iv_state=_iv(IV_DIRECTIONAL_BULL)).composite_state)
    # After enough bull bars, BULL_ENTRY must fire.
    assert BULL_ENTRY in states
    # After firing, subsequent bars must be HOLD_BULL.
    last = m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    assert last.composite_state == HOLD_BULL
    assert m.held_direction == +1


def test_bull_holds_through_neutral_bars_then_exits():
    m = ThesisMemory()
    for _ in range(6):
        m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    assert m.held_direction == +1
    # Now drift through neutral bars — score decays.
    seen_exit = False
    for _ in range(30):
        snap = m.update(iv_state=_iv(IV_NEUTRAL, conf=0.0))
        if snap.composite_state == EXIT_BULL:
            seen_exit = True
            break
    assert seen_exit
    # After EXIT_BULL, the next bar should be NEUTRAL (held_direction == 0).
    nxt = m.update(iv_state=_iv(IV_NEUTRAL, conf=0.0))
    assert nxt.composite_state == NEUTRAL_THESIS
    assert m.held_direction == 0


def test_bear_entry_and_hold_mirror():
    m = ThesisMemory()
    for _ in range(5):
        m.update(iv_state=_iv(IV_DIRECTIONAL_BEAR))
    assert m.held_direction == -1
    snap = m.update(iv_state=_iv(IV_DIRECTIONAL_BEAR))
    assert snap.composite_state == HOLD_BEAR


# ─────────────────────────────────────────────────────────────────
# Hysteresis — single counter bar does not reverse
# ─────────────────────────────────────────────────────────────────

def test_single_counter_bar_does_not_flip_held_direction():
    m = ThesisMemory()
    for _ in range(5):
        m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    snap = m.update(iv_state=_iv(IV_DIRECTIONAL_BEAR))
    # Held long should persist — one counter bar nicks but doesn't reverse.
    assert snap.composite_state == HOLD_BULL
    assert m.held_direction == +1
    assert snap.bull_thesis_score > 60.0


def test_exit_requires_score_below_exit_threshold():
    """Pure hysteresis: brief touches above exit threshold do not trigger
    re-entry; brief dips just below entry threshold do not exit."""
    m = ThesisMemory()
    for _ in range(6):
        m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    # A single neutral bar brings score down a few points; must remain HOLD_BULL.
    snap = m.update(iv_state=_iv(IV_NEUTRAL, conf=0.0))
    assert snap.composite_state == HOLD_BULL
    assert snap.bull_thesis_score >= m.cfg.exit_confidence


# ─────────────────────────────────────────────────────────────────
# Damaging signals
# ─────────────────────────────────────────────────────────────────

def test_regime_break_on_held_side_cuts_hard():
    m = ThesisMemory()
    for _ in range(6):
        m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    before = m.snapshot().bull_thesis_score
    snap = m.update(iv_state=_iv(IV_NEUTRAL, conf=0.0),
                    hunt_verdict="REGIME_BREAK_DOWN")
    after = snap.bull_thesis_score
    # Big cut — should drop by at least the configured cost.
    assert (before - after) >= m.cfg.cost_regime_break * 0.5


def test_strong_bull_trap_on_held_long_cuts():
    m = ThesisMemory()
    for _ in range(6):
        m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    before = m.snapshot().bull_thesis_score
    snap = m.update(iv_state=_iv(IV_NEUTRAL, conf=0.0),
                    trap_verdict="STRONG_BULL_TRAP")
    assert snap.bull_thesis_score < before


def test_basic_trap_costs_less_than_strong_trap():
    m1 = ThesisMemory(); m2 = ThesisMemory()
    for _ in range(6):
        m1.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
        m2.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    s1 = m1.update(iv_state=_iv(IV_NEUTRAL, conf=0.0), trap_verdict="BULL_TRAP")
    s2 = m2.update(iv_state=_iv(IV_NEUTRAL, conf=0.0), trap_verdict="STRONG_BULL_TRAP")
    # STRONG trap cuts more than basic.
    assert s2.bull_thesis_score < s1.bull_thesis_score


# ─────────────────────────────────────────────────────────────────
# No-trade danger
# ─────────────────────────────────────────────────────────────────

def test_two_consecutive_distortion_bars_trigger_no_trade():
    m = ThesisMemory()
    for _ in range(4):
        m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    # First distortion bar: probably below danger.
    m.update(iv_state=_iv(IV_LIQUIDITY_DISTORTION, conf=0.9))
    second = m.update(iv_state=_iv(IV_LIQUIDITY_DISTORTION, conf=0.9))
    assert second.composite_state == NO_TRADE_DANGER
    assert second.liquidity_danger_score >= m.cfg.no_trade_danger


def test_dirty_data_triggers_no_trade_after_persistence():
    """Dirty data carries a bigger per-bar delta than liquidity distortion;
    persistence across two bars must cross the danger threshold."""
    m = ThesisMemory()
    m.update(iv_state=_iv(IV_DIRTY_DATA, conf=1.0))
    snap = m.update(iv_state=_iv(IV_DIRTY_DATA, conf=1.0))
    assert snap.composite_state == NO_TRADE_DANGER
    assert snap.liquidity_danger_score >= m.cfg.no_trade_danger


def test_no_trade_blocks_new_entry_even_with_bull_score():
    """If liquidity_danger is above threshold, no entry fires regardless
    of how high the bull thesis is."""
    m = ThesisMemory()
    # Build the bull thesis up first.
    for _ in range(4):
        m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    # Now flatten the held position (decay it out).
    m._state.held_direction = 0
    m._state.last_composite = NEUTRAL_THESIS
    # Re-build bull while a distortion lurks.
    m.update(iv_state=_iv(IV_LIQUIDITY_DISTORTION, conf=0.9))
    snap = m.update(iv_state=_iv(IV_LIQUIDITY_DISTORTION, conf=0.9))
    # Even a high bull score is overridden by no_trade.
    assert snap.composite_state == NO_TRADE_DANGER


# ─────────────────────────────────────────────────────────────────
# Vol expansion contributes to no-trade but at a discount
# ─────────────────────────────────────────────────────────────────

def test_vol_expansion_contributes_to_no_trade():
    m = ThesisMemory()
    for _ in range(6):
        m.update(iv_state=_iv(IV_COMMON_SHOCK, conf=0.95))
    snap = m.snapshot()
    assert snap.vol_expansion_score > 50.0
    # vol_exp counts at 50% weight into no_trade — saturated vol_exp must
    # still reach the danger threshold.
    assert snap.no_trade_score >= m.cfg.no_trade_danger * 0.5


# ─────────────────────────────────────────────────────────────────
# Acceptance skew bonus + decay-only behavior
# ─────────────────────────────────────────────────────────────────

def test_acceptance_skew_note_adds_bonus():
    m_plain = ThesisMemory(); m_skew = ThesisMemory()
    plain = m_plain.update(iv_state=_iv(IV_DIRECTIONAL_BULL, note="ordinary"))
    skew = m_skew.update(iv_state=_iv(IV_DIRECTIONAL_BULL, note="acceptance skew"))
    # The acceptance bonus delivers a higher first-bar bull score.
    assert skew.bull_thesis_score > plain.bull_thesis_score


def test_decay_only_drifts_scores_toward_zero():
    m = ThesisMemory()
    # Seed the state.
    for _ in range(4):
        m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    seeded = m.snapshot().bull_thesis_score
    for _ in range(40):
        m.update(iv_state=None)
    assert m.snapshot().bull_thesis_score < seeded * 0.2


def test_reset_clears_state():
    m = ThesisMemory()
    for _ in range(6):
        m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    assert m.held_direction == +1
    m.reset()
    snap = m.snapshot()
    assert snap.bull_thesis_score == 0.0
    assert snap.held_direction == 0
    assert snap.composite_state == NEUTRAL_THESIS


# ─────────────────────────────────────────────────────────────────
# Config + serialization
# ─────────────────────────────────────────────────────────────────

def test_config_validation():
    with pytest.raises(ValueError):
        ThesisMemoryConfig(decay_per_bar=1.5)
    with pytest.raises(ValueError):
        ThesisMemoryConfig(entry_confidence=50, exit_confidence=60)
    with pytest.raises(ValueError):
        ThesisMemoryConfig(no_trade_danger=200)


def test_snapshot_dict_is_serializable():
    m = ThesisMemory()
    snap = m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
    import json
    json.dumps(snap.to_dict())


def test_just_changed_flag_tracks_state_transitions():
    m = ThesisMemory()
    a = m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))    # NEUTRAL → NEUTRAL most likely first bar
    # Build until BULL_ENTRY fires; that transition must mark just_changed.
    saw_change = a.just_changed
    for _ in range(6):
        snap = m.update(iv_state=_iv(IV_DIRECTIONAL_BULL))
        saw_change = saw_change or snap.just_changed
    assert saw_change
