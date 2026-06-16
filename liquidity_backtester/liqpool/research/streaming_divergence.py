"""Mid-candle streaming engine for the divergence + hunt stack.

The founder's concern (2026-06-16): "a 15-min candle, 5 min in, spot
+₹5, 10 min left — the model can't wait for the close." The detectors
in ``premium_divergence`` and ``liquidity_hunt`` are written against a
sealed bar series; this engine adapts them to live operation, where the
current candle is *forming* and the operator must read it in real time
without flip-flopping on intra-candle wiggle.

The model is the standard streaming-vs-confirmed split:

  * CONFIRMED history: every sealed (closed) bar feeds the history. The
    rolling computations (deltas, residuals, pressure, regime) use the
    confirmed history exclusively for stability — a forming bar can't
    contaminate the rolling stats.

  * PROVISIONAL forming bar: the current bar is updated tick-by-tick
    with the current spot, call, put. We append it to the confirmed
    history, run the same compute_divergence_frame + classify_liquidity_hunts
    pipeline, and read off the LAST row's verdict — that's the
    PROVISIONAL read. On bar close, the operator calls ``seal_bar()``
    and the forming bar becomes confirmed.

  * STABILITY GATE: the founder explicitly called intra-candle wiggle
    "buffer". We don't emit a verdict change until the same verdict has
    held for ``stability_ticks`` consecutive updates. This damps the
    flip-flop you'd otherwise get when net_intent_z is hovering near
    the threshold.

Action precedence (which the operator should act on):

  HUNT  > BREAK  > TRAP  > NEUTRAL

So when the hunt classifier says LIQUIDITY_HUNT_UP and the trap detector
says BULL_TRAP on the same bar, the engine reports HUNT — it's the
higher-level read ("the move's positioning is intact even though spot
moved against") and the trap signal is subsumed by it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

import numpy as np
import pandas as pd

from .liquidity_hunt import (
    ACTION_EXIT,
    ACTION_HOLD_THROUGH,
    ACTION_NONE,
    ACTION_RIDE,
    LiquidityHuntConfig,
    NO_REGIME,
    classify_liquidity_hunts,
)
from .premium_divergence import (
    NEUTRAL,
    PremiumDivergenceConfig,
    STRONG_TRAP_VERDICTS,
    TRAP_VERDICTS,
    compute_divergence_frame,
)


# Maximum confirmed-history size kept in memory. Larger means more stable
# rolling stats but more memory; 600 bars = 10 hours of 1-min bars =
# comfortably more than any rolling window the detectors use.
DEFAULT_HISTORY_BARS: int = 600

# Number of consecutive updates the same verdict must hold before the
# engine reports it as the "stable read". Damps intra-candle flicker.
DEFAULT_STABILITY_TICKS: int = 3


@dataclass(frozen=True)
class StreamingRead:
    """One snapshot from the streaming engine.

    ``provisional`` is the verdict computed on the forming bar (changes
    as the bar evolves). ``stable`` is the last verdict that held for
    ``stability_ticks`` consecutive updates — that's the read the
    operator should act on.

    The single ``action`` field collapses the trap + hunt layers per the
    module's precedence rule (HUNT > BREAK > TRAP > NEUTRAL).
    """
    bar_index: int                 # which forming-bar index this read belongs to
    ts: Any
    spot: float
    provisional_verdict: str
    stable_verdict: str
    action: str
    direction: int                 # +1 expect up, -1 expect down, 0 none
    hunt_confidence: float
    trap_tier: str                 # "strong" / "basic" / "" if no trap
    net_intent_z: float
    spot_move_norm: float
    bars_held: int                 # how many consecutive updates the stable verdict held

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _combined_action(trap_verdict: str, hunt_verdict: str,
                     hunt_action: str) -> str:
    """Precedence: hunt's action wins when it's actionable; trap is only
    used when the hunt layer reports nothing."""
    if hunt_action in (ACTION_HOLD_THROUGH, ACTION_EXIT, ACTION_RIDE):
        return hunt_action
    if trap_verdict in TRAP_VERDICTS:
        return "fade"            # trap verdicts signal a fade trade
    return ACTION_NONE


def _direction_from_verdicts(trap_verdict: str, hunt_verdict: str,
                              hunt_direction: int) -> int:
    if hunt_direction != 0:
        return int(hunt_direction)
    if trap_verdict in TRAP_VERDICTS:
        # BULL_TRAP / STRONG_BULL_TRAP → expect spot down (take put);
        # BEAR_TRAP / STRONG_BEAR_TRAP → expect spot up (take call).
        return -1 if "BULL" in trap_verdict else 1
    return 0


@dataclass
class StreamingDivergenceEngine:
    """Stateful mid-candle reader for the divergence + hunt stack.

    Usage:
        eng = StreamingDivergenceEngine()
        # On every tick of the forming bar:
        read = eng.update(ts, spot, call_premium, put_premium)
        # On bar close:
        eng.seal_bar(ts, spot, call_premium, put_premium)
    """
    divergence_cfg: PremiumDivergenceConfig = field(
        default_factory=PremiumDivergenceConfig)
    hunt_cfg: LiquidityHuntConfig = field(default_factory=LiquidityHuntConfig)
    history_bars: int = DEFAULT_HISTORY_BARS
    stability_ticks: int = DEFAULT_STABILITY_TICKS

    _confirmed: Deque[Dict[str, Any]] = field(default_factory=deque, init=False)
    _last_stable: str = field(default=NEUTRAL, init=False)
    _last_stable_action: str = field(default=ACTION_NONE, init=False)
    _last_stable_direction: int = field(default=0, init=False)
    _candidate: str = field(default=NEUTRAL, init=False)
    _candidate_holds: int = field(default=0, init=False)
    _bar_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.history_bars < 60:
            raise ValueError("history_bars must be >= 60 — detectors need rolling stats")
        if self.stability_ticks < 1:
            raise ValueError("stability_ticks must be >= 1")

    @property
    def history_size(self) -> int:
        return len(self._confirmed)

    def seal_bar(self, ts: Any, spot: float,
                 call_premium: float, put_premium: float) -> None:
        """Append a CLOSED bar to the confirmed history. Resets the
        forming-bar stability tracker so the next forming bar starts
        from a clean candidate."""
        self._confirmed.append({
            "ts": ts, "spot": float(spot),
            "call_premium": float(call_premium),
            "put_premium": float(put_premium),
        })
        while len(self._confirmed) > self.history_bars:
            self._confirmed.popleft()
        self._bar_counter += 1
        # Reset candidate — the new forming bar gets a fresh stability run.
        self._candidate = NEUTRAL
        self._candidate_holds = 0

    def update(self, ts: Any, spot: float,
               call_premium: float, put_premium: float) -> StreamingRead:
        """Read the FORMING bar with current tick values. The forming bar
        is appended to the confirmed history transiently — the rolling
        stats see it as the last row; on the next ``update`` it's replaced.

        Returns a :class:`StreamingRead` whose ``stable_verdict`` is the
        last verdict that held for ``stability_ticks`` consecutive
        updates. ``provisional_verdict`` is the raw current-tick read.
        """
        forming = {
            "ts": ts, "spot": float(spot),
            "call_premium": float(call_premium),
            "put_premium": float(put_premium),
        }
        rows = list(self._confirmed) + [forming]
        df = pd.DataFrame(rows)
        # Detectors need enough history to be meaningful — degrade gracefully
        # to NEUTRAL when the history is too short.
        min_required = max(self.divergence_cfg.delta_window
                           + self.divergence_cfg.accumulation_window,
                           self.hunt_cfg.regime_window) + 4
        if len(df) < min_required:
            return StreamingRead(
                bar_index=self._bar_counter,
                ts=ts, spot=float(spot),
                provisional_verdict=NEUTRAL,
                stable_verdict=NEUTRAL,
                action=ACTION_NONE,
                direction=0,
                hunt_confidence=0.0,
                trap_tier="",
                net_intent_z=0.0,
                spot_move_norm=0.0,
                bars_held=0,
            )
        enriched = compute_divergence_frame(df, self.divergence_cfg)
        classified = classify_liquidity_hunts(enriched, self.hunt_cfg)
        last = classified.iloc[-1]
        trap_v = str(last["verdict"])
        hunt_v = str(last["hunt_verdict"])
        hunt_action = str(last["hunt_action"])
        trap_tier = str(last.get("tier", ""))

        # The PROVISIONAL verdict per the precedence rule. We pack both
        # layers into one label so the operator sees the dominant read.
        provisional = hunt_v if hunt_v != NO_REGIME else trap_v
        action = _combined_action(trap_v, hunt_v, hunt_action)
        direction = _direction_from_verdicts(trap_v, hunt_v,
                                              int(last.get("regime", 0) or 0)
                                              if hunt_v in (
                                                  "LIQUIDITY_HUNT_UP", "REGIME_BREAK_UP")
                                              else (-int(last.get("regime", 0) or 0)
                                                    if hunt_v in (
                                                        "LIQUIDITY_HUNT_DOWN",
                                                        "REGIME_BREAK_DOWN")
                                                    else 0))

        # Stability gate: only promote the candidate to stable when it has
        # held for ``stability_ticks`` consecutive updates.
        if provisional == self._candidate:
            self._candidate_holds += 1
        else:
            self._candidate = provisional
            self._candidate_holds = 1
        if self._candidate_holds >= self.stability_ticks:
            if provisional != self._last_stable:
                self._last_stable = provisional
                self._last_stable_action = action
                self._last_stable_direction = direction

        return StreamingRead(
            bar_index=self._bar_counter,
            ts=ts, spot=float(spot),
            provisional_verdict=provisional,
            stable_verdict=self._last_stable,
            action=self._last_stable_action,
            direction=self._last_stable_direction,
            hunt_confidence=float(last.get("hunt_confidence", 0.0) or 0.0),
            trap_tier=trap_tier,
            net_intent_z=float(last.get("net_intent_z", 0.0) or 0.0),
            spot_move_norm=float(last.get("spot_move_norm", 0.0) or 0.0),
            bars_held=int(self._candidate_holds),
        )

    def reset(self) -> None:
        """Clear all state — useful between sessions or in tests."""
        self._confirmed.clear()
        self._last_stable = NEUTRAL
        self._last_stable_action = ACTION_NONE
        self._last_stable_direction = 0
        self._candidate = NEUTRAL
        self._candidate_holds = 0
        self._bar_counter = 0
