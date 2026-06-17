"""Bullish, bearish, and liquidity-sweep state machines (Belief Engine, Phase 7).

The founder's exact specification, three independent machines that run
in parallel on every bar. Each one's job is to track a specific narrative
and walk through its numbered states — so the engine "knows where it is"
in each story, not just what the current bar says.

BULLISH CONTINUATION:
  0: neutral
  1: bull_impulse_forming      — first signal of bull positioning
  2: bull_impulse_confirmed    — IV directional bull + bullish agreement
  3: pullback_test              — spot drops while we're in a bull regime
  4: pullback_survived          — pullback ends, positioning still bull
  5: winding_up                 — at upper excursion, paused, bull regime
  6: continuation_trigger       — bull confirmed again after winding
  7: bull_thesis_damaged        — counter signal, but regime not broken
  8: bull_thesis_invalidated    — REGIME_BREAK on the bull regime

BEARISH CONTINUATION: mirror.

LIQUIDITY SWEEP:
  1: local_extreme_taken        — spot pierces a local high (bullish setup
                                   for a bearish sweep) or local low
                                   (bearish setup for a bullish sweep)
  2: opposite_premium_fails     — the leg that "should" be expanding doesn't
  3: same_side_premium_survives — the leg that should be dying doesn't
  4: spread_normalizes          — the book becomes execution-friendly again
  5: reversal_trigger           — the sweep completes, take the contra-trade

Each machine emits per-bar:
  * its current state index + label
  * just_entered: True only on the bar the state changes
  * a brief note explaining the transition

The machines are STRICTLY causal and streaming — every transition is
based only on the current bar's inputs + the prior state. ``reset()``
clears them.

The decision layer (Phase 8) consumes these states. A trader holding
LONG at state 5 (winding_up) is in a "scalp the continuation" zone; a
trader holding LONG at state 8 (invalidated) must exit immediately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .battlefield import (
    BattlefieldSnapshot,
    FIELD_BEARISH_AGREEMENT,
    FIELD_BULLISH_AGREEMENT,
)
from .iv_state import (
    IVState,
    IV_DIRECTIONAL_BEAR,
    IV_DIRECTIONAL_BULL,
)
from .spread import CLEAN as SPREAD_CLEAN, DANGEROUS as SPREAD_DANGEROUS
from .winding import (
    BEAR_TRAP_WINDING,
    BEARISH_WINDING_DOWN,
    BULL_TRAP_WINDING,
    BULLISH_WINDING_UP,
    WindingZone,
)


# Bull continuation state labels.
_BULL_STATE_LABELS = [
    "neutral",
    "bull_impulse_forming",
    "bull_impulse_confirmed",
    "pullback_test",
    "pullback_survived",
    "winding_up",
    "continuation_trigger",
    "bull_thesis_damaged",
    "bull_thesis_invalidated",
]

_BEAR_STATE_LABELS = [
    "neutral",
    "bear_impulse_forming",
    "bear_impulse_confirmed",
    "bounce_test",
    "bounce_failed",
    "winding_down",
    "continuation_trigger",
    "bear_thesis_damaged",
    "bear_thesis_invalidated",
]

_SWEEP_STATE_LABELS = [
    "neutral",
    "local_extreme_taken",
    "opposite_premium_fails",
    "same_side_premium_survives",
    "spread_normalizes",
    "reversal_trigger",
]


@dataclass(frozen=True)
class StateUpdate:
    """One state-machine output for one bar."""
    machine: str          # "bull" / "bear" / "sweep"
    state_index: int
    state_label: str
    just_entered: bool
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# ─────────────────────────────────────────────────────────────────
# Bull continuation
# ─────────────────────────────────────────────────────────────────

@dataclass
class BullContinuationMachine:
    """Tracks the bull-continuation narrative across bars."""
    state: int = 0
    _prior_spot: Optional[float] = None
    _bars_in_pullback: int = 0

    def reset(self) -> None:
        self.state = 0
        self._prior_spot = None
        self._bars_in_pullback = 0

    def update(self, *,
               spot: float,
               battlefield: Optional[BattlefieldSnapshot] = None,
               iv_state: Optional[IVState] = None,
               winding: Optional[WindingZone] = None,
               hunt_verdict: str = "",
               trap_verdict: str = "",
               ) -> StateUpdate:
        prev = self.state
        note = ""
        bf_verdict = battlefield.verdict if battlefield else ""
        iv = iv_state.state if iv_state else ""

        # Invalidation overrides everything.
        if hunt_verdict == "REGIME_BREAK_DOWN" or trap_verdict == "STRONG_BULL_TRAP":
            self.state = 8
            note = "bull thesis invalidated — regime broken or strong bull trap"
        # Bearish agreement on what was a bull → damaged, not yet broken.
        elif (self.state >= 2 and bf_verdict == FIELD_BEARISH_AGREEMENT
              and self.state != 8):
            self.state = 7
            note = "bull thesis damaged — battlefield turned bearish"
        # Continuation trigger from winding-up + IV bull.
        elif (self.state == 5 and winding is not None
              and winding.zone == BULLISH_WINDING_UP
              and iv == IV_DIRECTIONAL_BULL):
            self.state = 6
            note = "winding-up resolved up; continuation trigger fired"
        # Enter winding-up if we're already confirmed/pullback-survived
        # and the winding detector says so.
        elif (winding is not None and winding.zone == BULLISH_WINDING_UP
              and self.state in (2, 4, 6)):
            self.state = 5
            note = "in bullish winding-up zone — continuation likely"
        # Pullback test: in a bull-active state and spot drops. State 3
        # is included so a sustained pullback can promote to state 4
        # once survival conditions are met.
        elif (self.state in (2, 3, 4, 5, 6) and self._prior_spot is not None
              and spot < self._prior_spot):
            self._bars_in_pullback += 1
            if self.state != 3:
                self.state = 3
                note = "pullback test — spot dropping, regime still bullish"
            # Pullback survived → bull positioning intact after 2 bars of pullback.
            if (self._bars_in_pullback >= 2
                    and bf_verdict == FIELD_BULLISH_AGREEMENT
                    and iv == IV_DIRECTIONAL_BULL):
                self.state = 4
                self._bars_in_pullback = 0
                note = "pullback survived — positioning still bullish"
        # Impulse confirmed.
        elif (bf_verdict == FIELD_BULLISH_AGREEMENT
              and iv == IV_DIRECTIONAL_BULL):
            self._bars_in_pullback = 0
            if self.state < 2:
                self.state = 2
                note = "bull impulse confirmed — battlefield + IV bull"
            elif self.state == 7:
                # Recovered from damage.
                self.state = 4
                note = "bull thesis recovered after damage"
        # Impulse forming.
        elif (bf_verdict == FIELD_BULLISH_AGREEMENT
              or iv == IV_DIRECTIONAL_BULL):
            if self.state == 0:
                self.state = 1
                note = "bull impulse forming"
        # Otherwise sit / decay.

        self._prior_spot = spot
        return StateUpdate(
            machine="bull",
            state_index=self.state,
            state_label=_BULL_STATE_LABELS[self.state],
            just_entered=(self.state != prev),
            note=note or "no transition",
        )


# ─────────────────────────────────────────────────────────────────
# Bear continuation — mirror of bull
# ─────────────────────────────────────────────────────────────────

@dataclass
class BearContinuationMachine:
    state: int = 0
    _prior_spot: Optional[float] = None
    _bars_in_bounce: int = 0

    def reset(self) -> None:
        self.state = 0
        self._prior_spot = None
        self._bars_in_bounce = 0

    def update(self, *,
               spot: float,
               battlefield: Optional[BattlefieldSnapshot] = None,
               iv_state: Optional[IVState] = None,
               winding: Optional[WindingZone] = None,
               hunt_verdict: str = "",
               trap_verdict: str = "",
               ) -> StateUpdate:
        prev = self.state
        note = ""
        bf_verdict = battlefield.verdict if battlefield else ""
        iv = iv_state.state if iv_state else ""

        if hunt_verdict == "REGIME_BREAK_UP" or trap_verdict == "STRONG_BEAR_TRAP":
            self.state = 8
            note = "bear thesis invalidated — regime broken or strong bear trap"
        elif (self.state >= 2 and bf_verdict == FIELD_BULLISH_AGREEMENT
              and self.state != 8):
            self.state = 7
            note = "bear thesis damaged — battlefield turned bullish"
        elif (self.state == 5 and winding is not None
              and winding.zone == BEARISH_WINDING_DOWN
              and iv == IV_DIRECTIONAL_BEAR):
            self.state = 6
            note = "winding-down resolved down; continuation trigger fired"
        elif (winding is not None and winding.zone == BEARISH_WINDING_DOWN
              and self.state in (2, 4, 6)):
            self.state = 5
            note = "in bearish winding-down zone — continuation likely"
        elif (self.state in (2, 3, 4, 5, 6) and self._prior_spot is not None
              and spot > self._prior_spot):
            self._bars_in_bounce += 1
            if self.state != 3:
                self.state = 3
                note = "bounce test — spot rising, regime still bearish"
            if (self._bars_in_bounce >= 2
                    and bf_verdict == FIELD_BEARISH_AGREEMENT
                    and iv == IV_DIRECTIONAL_BEAR):
                self.state = 4
                self._bars_in_bounce = 0
                note = "bounce failed — positioning still bearish"
        elif (bf_verdict == FIELD_BEARISH_AGREEMENT
              and iv == IV_DIRECTIONAL_BEAR):
            self._bars_in_bounce = 0
            if self.state < 2:
                self.state = 2
                note = "bear impulse confirmed — battlefield + IV bear"
            elif self.state == 7:
                self.state = 4
                note = "bear thesis recovered after damage"
        elif (bf_verdict == FIELD_BEARISH_AGREEMENT
              or iv == IV_DIRECTIONAL_BEAR):
            if self.state == 0:
                self.state = 1
                note = "bear impulse forming"

        self._prior_spot = spot
        return StateUpdate(
            machine="bear",
            state_index=self.state,
            state_label=_BEAR_STATE_LABELS[self.state],
            just_entered=(self.state != prev),
            note=note or "no transition",
        )


# ─────────────────────────────────────────────────────────────────
# Liquidity-sweep machine
# ─────────────────────────────────────────────────────────────────

@dataclass
class LiquiditySweepMachine:
    """Detects the stop-sweep narrative.

    The bullish-sweep setup: spot pierces a local LOW (state 1), the put
    rail fails to expand (state 2 — the contrarian leg doesn't confirm),
    the call rail survives instead of collapsing (state 3), spread
    normalizes (state 4), and reversal triggers (state 5) — operator takes
    a CE scalp. Mirror for bearish sweep at a local high.
    """
    state: int = 0
    direction: int = 0     # +1 expecting a bullish sweep, -1 bearish
    _recent_high: Optional[float] = None
    _recent_low: Optional[float] = None
    _bars_since_extreme: int = 0
    _max_bars: int = 6     # if state hasn't progressed in this many bars, reset

    def reset(self) -> None:
        self.state = 0
        self.direction = 0
        self._recent_high = None
        self._recent_low = None
        self._bars_since_extreme = 0

    def update(self, *,
               spot: float,
               battlefield: Optional[BattlefieldSnapshot] = None,
               iv_state: Optional[IVState] = None,
               spread_state: str = SPREAD_CLEAN,
               hunt_verdict: str = "",
               ) -> StateUpdate:
        prev = self.state
        note = ""
        bf_verdict = battlefield.verdict if battlefield else ""
        iv = iv_state.state if iv_state else ""

        # Maintain rolling extremes from spot.
        if self._recent_high is None or spot > self._recent_high:
            self._recent_high = spot
        if self._recent_low is None or spot < self._recent_low:
            self._recent_low = spot

        # State machine.
        if self.state == 0:
            # Look for a local extreme being taken — for now use the
            # founder's primary hunt verdict as the trigger (LIQUIDITY_HUNT_UP
            # = a downside hunt expecting up reversal; LIQUIDITY_HUNT_DOWN
            # = an upside hunt expecting down reversal).
            if hunt_verdict == "LIQUIDITY_HUNT_UP":
                self.state = 1
                self.direction = +1
                self._bars_since_extreme = 0
                note = "local low taken; expecting bullish sweep reversal"
            elif hunt_verdict == "LIQUIDITY_HUNT_DOWN":
                self.state = 1
                self.direction = -1
                self._bars_since_extreme = 0
                note = "local high taken; expecting bearish sweep reversal"
        elif self.state == 1:
            self._bars_since_extreme += 1
            # Opposite premium fails — the leg that "should" be expanding
            # given the spot move doesn't. For a bullish sweep: spot moved
            # DOWN to take the low, so the put rail should be strong; if
            # the IV state is NOT IV_DIRECTIONAL_BEAR, the puts are not
            # confirming the move down → state 2.
            if (self.direction == +1 and iv != IV_DIRECTIONAL_BEAR):
                self.state = 2
                note = "puts not expanding into the down-take → opposite leg fails"
            elif (self.direction == -1 and iv != IV_DIRECTIONAL_BULL):
                self.state = 2
                note = "calls not expanding into the up-take → opposite leg fails"
            elif self._bars_since_extreme >= self._max_bars:
                self.state = 0
                self.direction = 0
                note = "sweep watch expired"
        elif self.state == 2:
            # Same-side premium survives — for bullish sweep, the calls
            # are still showing strength (battlefield bullish or IV bull).
            if (self.direction == +1
                    and (bf_verdict == FIELD_BULLISH_AGREEMENT
                         or iv == IV_DIRECTIONAL_BULL)):
                self.state = 3
                note = "call rail surviving the down-take"
            elif (self.direction == -1
                    and (bf_verdict == FIELD_BEARISH_AGREEMENT
                         or iv == IV_DIRECTIONAL_BEAR)):
                self.state = 3
                note = "put rail surviving the up-take"
            elif self._bars_since_extreme >= self._max_bars:
                self.state = 0; self.direction = 0
                note = "sweep watch expired before same-side survived"
            self._bars_since_extreme += 1
        elif self.state == 3:
            self._bars_since_extreme += 1
            if spread_state == SPREAD_CLEAN:
                self.state = 4
                note = "spread normalised — execution window opening"
            elif self._bars_since_extreme >= self._max_bars:
                self.state = 0; self.direction = 0
                note = "spread did not normalise; sweep aborted"
        elif self.state == 4:
            # Reversal trigger: positioning confirms the sweep direction.
            if (self.direction == +1
                    and bf_verdict == FIELD_BULLISH_AGREEMENT
                    and iv == IV_DIRECTIONAL_BULL):
                self.state = 5
                note = "bullish sweep reversal trigger — take CE"
            elif (self.direction == -1
                    and bf_verdict == FIELD_BEARISH_AGREEMENT
                    and iv == IV_DIRECTIONAL_BEAR):
                self.state = 5
                note = "bearish sweep reversal trigger — take PE"
            else:
                self._bars_since_extreme += 1
                if self._bars_since_extreme >= self._max_bars + 2:
                    self.state = 0; self.direction = 0
                    note = "reversal failed to trigger; sweep aborted"
        elif self.state == 5:
            # Latch — caller resets() when the trade is taken / aborted.
            note = "reversal already triggered — operator should act"

        return StateUpdate(
            machine="sweep",
            state_index=self.state,
            state_label=_SWEEP_STATE_LABELS[self.state],
            just_entered=(self.state != prev),
            note=note or "no transition",
        )
