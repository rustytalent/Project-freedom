"""Tests for Belief Engine Phase 7 — winding zones + state machines.

Pins all four winding-zone types and the three state machines walking
through their numbered narratives.
"""
from __future__ import annotations

import pytest

from liqpool.research.belief.battlefield import (
    BattlefieldSnapshot,
    FIELD_BEARISH_AGREEMENT,
    FIELD_BULLISH_AGREEMENT,
    FIELD_QUIET,
    FIELD_VOL_EXPANSION,
    RailSummary,
)
from liqpool.research.belief.iv_state import (
    IVState,
    IV_DIRECTIONAL_BEAR,
    IV_DIRECTIONAL_BULL,
    IV_NEUTRAL,
)
from liqpool.research.belief.spread import CLEAN as SPREAD_CLEAN
from liqpool.research.belief.state_machines import (
    BearContinuationMachine,
    BullContinuationMachine,
    LiquiditySweepMachine,
    StateUpdate,
)
from liqpool.research.belief.winding import (
    BEAR_TRAP_WINDING,
    BEARISH_WINDING_DOWN,
    BULL_TRAP_WINDING,
    BULLISH_WINDING_UP,
    NO_WINDING,
    SCALP_CE,
    SCALP_PE,
    WindingConfig,
    WindingDetector,
    WindingZone,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _rail(side: str) -> RailSummary:
    return RailSummary(
        side=side, n_slots=11, weighted_mean_signed_z=0.0,
        weighted_mean_abs_z=0.0, fraction_abnormal=0.0,
        dispersion_score=0.0, n_abnormal_strong=0, n_abnormal_weak=0,
        epicenter_label="", epicenter_level=0, state="strong_bull",
    )


def _bf(verdict: str, *, conf: float = 0.8, direction: int = 0) -> BattlefieldSnapshot:
    if direction == 0:
        if verdict == FIELD_BULLISH_AGREEMENT:
            direction = +1
        elif verdict == FIELD_BEARISH_AGREEMENT:
            direction = -1
    return BattlefieldSnapshot(
        ce_rail=_rail("CE"), pe_rail=_rail("PE"),
        verdict=verdict, direction=direction, confidence=conf, note="",
    )


def _iv(state: str, *, conf: float = 0.9) -> IVState:
    direction = (+1 if state == IV_DIRECTIONAL_BULL
                 else -1 if state == IV_DIRECTIONAL_BEAR else 0)
    return IVState(
        state=state, direction=direction, confidence=conf,
        avg_friendliness=1.0, clean_mark_fraction=1.0,
        ce_defended_fraction=0.0, pe_defended_fraction=0.0,
        ce_rejected_fraction=0.0, pe_rejected_fraction=0.0, note="",
    )


# ─────────────────────────────────────────────────────────────────
# Winding zones
# ─────────────────────────────────────────────────────────────────

def _push_rising_then_pause(d: WindingDetector, **kw):
    """Push 10 rising bars then 5 paused bars at the top. Returns the
    last zone reading."""
    last = None
    prices = list(range(100, 110)) + [110.2, 110.0, 110.3, 110.1, 110.2]
    for p in prices:
        last = d.observe(p, **kw)
    return last


def _push_falling_then_pause(d: WindingDetector, **kw):
    last = None
    prices = list(range(110, 100, -1)) + [99.8, 100.0, 99.7, 99.9, 99.8]
    for p in prices:
        last = d.observe(p, **kw)
    return last


def test_bullish_winding_up_at_upper_excursion():
    d = WindingDetector()
    z = _push_rising_then_pause(
        d,
        battlefield=_bf(FIELD_BULLISH_AGREEMENT),
        iv_state=_iv(IV_DIRECTIONAL_BULL),
        spread_friendly=True,
    )
    assert z.zone == BULLISH_WINDING_UP
    assert z.scalp_direction == SCALP_CE
    assert z.confidence > 0.4


def test_bull_trap_winding_when_battlefield_disagrees_or_spread_bad():
    """Same upper excursion, but battlefield says bearish (or spread is
    bad) → BULL_TRAP_WINDING, scalp put."""
    d = WindingDetector()
    z = _push_rising_then_pause(
        d,
        battlefield=_bf(FIELD_BEARISH_AGREEMENT),
        iv_state=_iv(IV_NEUTRAL, conf=0.0),
        spread_friendly=False,
    )
    assert z.zone == BULL_TRAP_WINDING
    assert z.scalp_direction == SCALP_PE


def test_bearish_winding_down_at_lower_excursion():
    d = WindingDetector()
    z = _push_falling_then_pause(
        d,
        battlefield=_bf(FIELD_BEARISH_AGREEMENT),
        iv_state=_iv(IV_DIRECTIONAL_BEAR),
        spread_friendly=True,
    )
    assert z.zone == BEARISH_WINDING_DOWN
    assert z.scalp_direction == SCALP_PE


def test_bear_trap_winding_at_lower_excursion_with_bull_battlefield():
    d = WindingDetector()
    z = _push_falling_then_pause(
        d,
        battlefield=_bf(FIELD_BULLISH_AGREEMENT),
        iv_state=_iv(IV_NEUTRAL, conf=0.0),
        spread_friendly=False,
    )
    assert z.zone == BEAR_TRAP_WINDING
    assert z.scalp_direction == SCALP_CE


def test_no_winding_during_fast_one_way_move():
    """If price is still moving one-way (not paused), no winding zone."""
    d = WindingDetector()
    # 15 strictly rising bars — never paused.
    last = None
    for p in range(100, 115):
        last = d.observe(p,
                          battlefield=_bf(FIELD_BULLISH_AGREEMENT),
                          iv_state=_iv(IV_DIRECTIONAL_BULL))
    assert last.zone == NO_WINDING


def test_no_winding_when_history_too_short():
    d = WindingDetector()
    z = d.observe(100.0)
    assert z.zone == NO_WINDING


def test_no_winding_when_range_too_tight():
    d = WindingDetector(cfg=WindingConfig(min_range_pct=0.05))
    for _ in range(15):
        z = d.observe(100.0)
    # Flatlined price → range too tight.
    assert z.zone == NO_WINDING


def test_winding_to_dict_serializable():
    d = WindingDetector()
    z = _push_rising_then_pause(d,
                                battlefield=_bf(FIELD_BULLISH_AGREEMENT),
                                iv_state=_iv(IV_DIRECTIONAL_BULL))
    import json
    json.dumps(z.to_dict())


# ─────────────────────────────────────────────────────────────────
# Bull continuation state machine
# ─────────────────────────────────────────────────────────────────

def test_bull_machine_walks_0_to_2():
    m = BullContinuationMachine()
    u = m.update(spot=100, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
                 iv_state=_iv(IV_NEUTRAL, conf=0.0))
    assert u.state_index == 1   # impulse forming
    u = m.update(spot=101, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
                 iv_state=_iv(IV_DIRECTIONAL_BULL))
    assert u.state_index == 2   # impulse confirmed
    assert u.just_entered


def test_bull_machine_pullback_then_survives():
    m = BullContinuationMachine()
    m.update(spot=100, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BULL))
    m.update(spot=101, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BULL))
    # First drop → pullback test (state 3).
    u = m.update(spot=100.5, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
                 iv_state=_iv(IV_DIRECTIONAL_BULL))
    assert u.state_index == 3
    # Second consecutive drop → pullback survived (state 4).
    u = m.update(spot=100.2, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
                 iv_state=_iv(IV_DIRECTIONAL_BULL))
    assert u.state_index == 4


def test_bull_machine_invalidates_on_regime_break():
    m = BullContinuationMachine()
    m.update(spot=100, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BULL))
    m.update(spot=101, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BULL))
    u = m.update(spot=101, hunt_verdict="REGIME_BREAK_DOWN")
    assert u.state_index == 8
    assert "invalidated" in u.state_label


def test_bull_machine_invalidates_on_strong_bull_trap():
    m = BullContinuationMachine()
    m.update(spot=100, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BULL))
    m.update(spot=101, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BULL))
    u = m.update(spot=100, trap_verdict="STRONG_BULL_TRAP")
    assert u.state_index == 8


def test_bull_machine_damaged_on_bear_agreement():
    m = BullContinuationMachine()
    m.update(spot=100, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BULL))
    m.update(spot=101, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BULL))
    u = m.update(spot=100, battlefield=_bf(FIELD_BEARISH_AGREEMENT),
                 iv_state=_iv(IV_DIRECTIONAL_BEAR))
    assert u.state_index == 7    # damaged (not yet broken)


def test_bull_machine_reset_clears_state():
    m = BullContinuationMachine()
    m.update(spot=100, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BULL))
    m.update(spot=101, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BULL))
    m.reset()
    u = m.update(spot=100)
    assert u.state_index == 0


# ─────────────────────────────────────────────────────────────────
# Bear continuation state machine
# ─────────────────────────────────────────────────────────────────

def test_bear_machine_confirms_and_invalidates():
    m = BearContinuationMachine()
    m.update(spot=100, battlefield=_bf(FIELD_BEARISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BEAR))
    u = m.update(spot=99, battlefield=_bf(FIELD_BEARISH_AGREEMENT),
                 iv_state=_iv(IV_DIRECTIONAL_BEAR))
    assert u.state_index == 2
    # Regime break up → invalidated.
    u = m.update(spot=100, hunt_verdict="REGIME_BREAK_UP")
    assert u.state_index == 8


def test_bear_machine_bounce_then_failed():
    m = BearContinuationMachine()
    m.update(spot=101, battlefield=_bf(FIELD_BEARISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BEAR))
    m.update(spot=100, battlefield=_bf(FIELD_BEARISH_AGREEMENT),
             iv_state=_iv(IV_DIRECTIONAL_BEAR))
    u = m.update(spot=100.5, battlefield=_bf(FIELD_BEARISH_AGREEMENT),
                 iv_state=_iv(IV_DIRECTIONAL_BEAR))
    assert u.state_index == 3   # bounce test
    u = m.update(spot=100.8, battlefield=_bf(FIELD_BEARISH_AGREEMENT),
                 iv_state=_iv(IV_DIRECTIONAL_BEAR))
    assert u.state_index == 4   # bounce failed (positioning intact)


# ─────────────────────────────────────────────────────────────────
# Liquidity sweep state machine
# ─────────────────────────────────────────────────────────────────

def test_sweep_machine_walks_all_5_states_bullish():
    sm = LiquiditySweepMachine()
    u = sm.update(spot=100, hunt_verdict="LIQUIDITY_HUNT_UP")
    assert u.state_index == 1
    # bar 2: spot moves down for the hunt, IV NOT bear → opposite premium fails
    u = sm.update(spot=99, iv_state=_iv(IV_DIRECTIONAL_BULL))
    assert u.state_index == 2
    # bar 3: CE rail strong → same-side survives
    u = sm.update(spot=99, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
                  iv_state=_iv(IV_DIRECTIONAL_BULL))
    assert u.state_index == 3
    # bar 4: spread clean → normalises
    u = sm.update(spot=99.5, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
                  iv_state=_iv(IV_DIRECTIONAL_BULL), spread_state=SPREAD_CLEAN)
    assert u.state_index == 4
    # bar 5: bull + IV bull → reversal trigger
    u = sm.update(spot=100, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
                  iv_state=_iv(IV_DIRECTIONAL_BULL), spread_state=SPREAD_CLEAN)
    assert u.state_index == 5
    assert "take CE" in u.note


def test_sweep_machine_walks_bearish_sweep():
    sm = LiquiditySweepMachine()
    sm.update(spot=100, hunt_verdict="LIQUIDITY_HUNT_DOWN")
    sm.update(spot=101, iv_state=_iv(IV_DIRECTIONAL_BEAR))
    sm.update(spot=101, battlefield=_bf(FIELD_BEARISH_AGREEMENT),
              iv_state=_iv(IV_DIRECTIONAL_BEAR))
    sm.update(spot=100.5, battlefield=_bf(FIELD_BEARISH_AGREEMENT),
              iv_state=_iv(IV_DIRECTIONAL_BEAR), spread_state=SPREAD_CLEAN)
    u = sm.update(spot=100, battlefield=_bf(FIELD_BEARISH_AGREEMENT),
                  iv_state=_iv(IV_DIRECTIONAL_BEAR), spread_state=SPREAD_CLEAN)
    assert u.state_index == 5
    assert "take PE" in u.note


def test_sweep_machine_expires_without_progress():
    sm = LiquiditySweepMachine()
    sm.update(spot=100, hunt_verdict="LIQUIDITY_HUNT_UP")
    # Stay at state 1 (no positioning to fail) for many bars.
    for _ in range(10):
        u = sm.update(spot=99, iv_state=_iv(IV_DIRECTIONAL_BEAR))
    assert u.state_index == 0     # reset / aborted


def test_state_update_serializable():
    m = BullContinuationMachine()
    u = m.update(spot=100, battlefield=_bf(FIELD_BULLISH_AGREEMENT),
                 iv_state=_iv(IV_DIRECTIONAL_BULL))
    import json
    json.dumps(u.to_dict())
