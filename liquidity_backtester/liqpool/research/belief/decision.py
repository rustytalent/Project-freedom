"""Decision layer (Belief Engine, Phase 8).

The founder's "decision layer should not only say buy/sell — it should
output:

    trade_allowed?  direction?  confidence?  best_strike?  exit_rule?
    spread_friendliness?  thesis_state?  invalidation?  no_trade_reason?
"

This module consumes the outputs of every upstream phase and produces a
single :class:`Decision` per bar. Precedence is sharp:

    1. NO-TRADE rules (highest)        — dirty data, dangerous spread,
                                          common shock, thesis no-trade.
    2. EXIT rules                       — invalidation states 8 on the
                                          held machine.
    3. SCALP windings                   — bull/bear-trap winding zones.
    4. ENTRY rules                      — BULL_ENTRY/BEAR_ENTRY from
                                          thesis memory + winding-up
                                          continuation triggers.
    5. HOLD rules                       — sitting and the thesis is intact.
    6. NEUTRAL                          — nothing to do.

The decision also recommends the BEST STRIKE for a directional trade
based on the founder's moneyness identity: ATM gamma for entries (most
informative + maximum convexity per rupee), 1-ITM for scalps where we
want directional confidence with less theta bleed, and contrarian-leg
ATM for trap-winding scalps. The recommendation is a label + a strike
offset relative to ATM, ready for the executor to translate into a
real contract.

Everything is per-bar and deterministic from its inputs — the upstream
phases carry the state, so the decision layer itself is pure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .battlefield import BattlefieldSnapshot, FIELD_SINGLE_DISTORTION
from .iv_state import (
    IVState,
    IV_COMMON_SHOCK,
    IV_DIRECTIONAL_BEAR,
    IV_DIRECTIONAL_BULL,
    IV_DIRTY_DATA,
    IV_LIQUIDITY_DISTORTION,
)
from .state_machines import StateUpdate
from .thesis_memory import (
    BEAR_ENTRY,
    BULL_ENTRY,
    EXIT_BEAR,
    EXIT_BULL,
    HOLD_BEAR,
    HOLD_BULL,
    NEUTRAL_THESIS,
    NO_TRADE_DANGER,
    ThesisSnapshot,
)
from .winding import (
    BEAR_TRAP_WINDING,
    BEARISH_WINDING_DOWN,
    BULL_TRAP_WINDING,
    BULLISH_WINDING_UP,
    NO_WINDING,
    WindingZone,
)


# Action labels — what the decision tells the operator.
ACTION_ENTER_LONG = "ENTER_LONG"
ACTION_ENTER_SHORT = "ENTER_SHORT"
ACTION_HOLD = "HOLD"
ACTION_EXIT = "EXIT"
ACTION_SCALP_CALL = "SCALP_CALL"     # trap-winding bear → CE; bearish winding → CE
ACTION_SCALP_PUT = "SCALP_PUT"       # trap-winding bull → PE; bullish winding → PE
ACTION_NO_TRADE = "NO_TRADE"
ACTION_WAIT = "WAIT"


@dataclass
class DecisionConfig:
    """Knobs for the decision layer."""
    min_directional_confidence: float = 0.30   # floor on IV / battlefield confidence
    min_scalp_winding_confidence: float = 0.45 # winding zones require this to scalp
    prefer_atm_for_entries: bool = True        # entry strike = ATM (max gamma)
    scalp_use_itm_buffer: int = 0              # +1 for slightly ITM scalp (less theta)
    eps: float = 1e-9


@dataclass(frozen=True)
class StrikeRecommendation:
    """Which contract slot the decision recommends."""
    side: str                  # "CE" / "PE" / ""
    level: int                 # signed moneyness level (0 = ATM)
    label: str                 # e.g. "CE_ATM", "PE_ITM1"; "" if N/A

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Decision:
    """The decision layer's per-bar output."""
    action: str
    trade_allowed: bool
    direction: int                            # +1 long, -1 short, 0 none
    confidence: float                         # [0,1]
    strike: StrikeRecommendation
    thesis_state: str                         # the thesis-memory composite state
    invalidation_rule: str                    # how this trade dies
    exit_rule: str                            # short text describing the exit
    no_trade_reason: str                      # populated when action == NO_TRADE / WAIT
    spread_friendliness: float                # 0..1 (from the upstream battlefield)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["strike"] = self.strike.to_dict()
        d["notes"] = list(self.notes)
        return d


def _strike_for_directional(direction: int, cfg: DecisionConfig) -> StrikeRecommendation:
    """ATM CE for long, ATM PE for short — max gamma per rupee."""
    if direction > 0:
        return StrikeRecommendation(side="CE", level=0, label="CE_ATM")
    if direction < 0:
        return StrikeRecommendation(side="PE", level=0, label="PE_ATM")
    return StrikeRecommendation(side="", level=0, label="")


def _strike_for_scalp(side: str, cfg: DecisionConfig) -> StrikeRecommendation:
    """ATM by default; can shift slightly ITM if configured (less theta
    bleed for very short holding periods)."""
    if side not in ("CE", "PE"):
        return StrikeRecommendation(side="", level=0, label="")
    level = -cfg.scalp_use_itm_buffer if side == "CE" else cfg.scalp_use_itm_buffer
    if level == 0:
        label = f"{side}_ATM"
    elif level < 0:
        label = f"{side}_ITM{abs(level)}"
    else:
        label = f"{side}_OTM{level}"
    return StrikeRecommendation(side=side, level=level, label=label)


def decide(*,
           thesis: ThesisSnapshot,
           iv_state: IVState,
           battlefield: BattlefieldSnapshot,
           bull_state: Optional[StateUpdate] = None,
           bear_state: Optional[StateUpdate] = None,
           sweep_state: Optional[StateUpdate] = None,
           winding: Optional[WindingZone] = None,
           avg_friendliness: float = 1.0,
           cfg: Optional[DecisionConfig] = None,
           ) -> Decision:
    """Produce the per-bar :class:`Decision` from the upstream readings."""
    cfg = cfg or DecisionConfig()
    notes: List[str] = []

    # ── 1. NO-TRADE rules ─────────────────────────────────────────────
    if iv_state.state == IV_DIRTY_DATA:
        return Decision(
            action=ACTION_NO_TRADE, trade_allowed=False, direction=0,
            confidence=0.0,
            strike=StrikeRecommendation("", 0, ""),
            thesis_state=thesis.composite_state,
            invalidation_rule="", exit_rule="",
            no_trade_reason="dirty data — too many invalid marks",
            spread_friendliness=avg_friendliness, notes=notes,
        )

    if iv_state.state == IV_LIQUIDITY_DISTORTION or thesis.composite_state == NO_TRADE_DANGER:
        reason = ("liquidity distortion — spread/book unreliable"
                  if iv_state.state == IV_LIQUIDITY_DISTORTION
                  else "thesis memory hit no-trade danger")
        return Decision(
            action=ACTION_NO_TRADE, trade_allowed=False, direction=0,
            confidence=0.0,
            strike=StrikeRecommendation("", 0, ""),
            thesis_state=thesis.composite_state,
            invalidation_rule="", exit_rule="",
            no_trade_reason=reason,
            spread_friendliness=avg_friendliness, notes=notes,
        )

    if iv_state.state == IV_COMMON_SHOCK:
        # Vol expansion: refuse to read direction; sit out.
        return Decision(
            action=ACTION_WAIT, trade_allowed=False, direction=0,
            confidence=0.0,
            strike=StrikeRecommendation("", 0, ""),
            thesis_state=thesis.composite_state,
            invalidation_rule="", exit_rule="",
            no_trade_reason="common IV shock — straddle bid; no directional read",
            spread_friendliness=avg_friendliness, notes=notes,
        )

    if battlefield.verdict == FIELD_SINGLE_DISTORTION:
        return Decision(
            action=ACTION_WAIT, trade_allowed=False, direction=0,
            confidence=0.0,
            strike=StrikeRecommendation("", 0, ""),
            thesis_state=thesis.composite_state,
            invalidation_rule="", exit_rule="",
            no_trade_reason=f"single-strike distortion: {battlefield.note}",
            spread_friendliness=avg_friendliness, notes=notes,
        )

    # ── 2. EXIT — invalidation states on held machine ─────────────────
    if thesis.held_direction == +1 and bull_state is not None and bull_state.state_index == 8:
        return Decision(
            action=ACTION_EXIT, trade_allowed=True, direction=0,
            confidence=1.0,
            strike=StrikeRecommendation("CE", 0, "CE_ATM"),
            thesis_state=thesis.composite_state,
            invalidation_rule="bull thesis invalidated",
            exit_rule="exit now — bull thesis invalidated (state 8)",
            no_trade_reason="",
            spread_friendliness=avg_friendliness, notes=notes,
        )
    if thesis.held_direction == -1 and bear_state is not None and bear_state.state_index == 8:
        return Decision(
            action=ACTION_EXIT, trade_allowed=True, direction=0,
            confidence=1.0,
            strike=StrikeRecommendation("PE", 0, "PE_ATM"),
            thesis_state=thesis.composite_state,
            invalidation_rule="bear thesis invalidated",
            exit_rule="exit now — bear thesis invalidated (state 8)",
            no_trade_reason="",
            spread_friendliness=avg_friendliness, notes=notes,
        )

    # Thesis memory EXIT_* also forces an exit (the scores decayed past
    # the exit threshold).
    if thesis.composite_state == EXIT_BULL:
        return Decision(
            action=ACTION_EXIT, trade_allowed=True, direction=0,
            confidence=0.6,
            strike=StrikeRecommendation("CE", 0, "CE_ATM"),
            thesis_state=thesis.composite_state,
            invalidation_rule="bull thesis confidence decayed below exit",
            exit_rule="exit — bull score crossed exit threshold",
            no_trade_reason="",
            spread_friendliness=avg_friendliness, notes=notes,
        )
    if thesis.composite_state == EXIT_BEAR:
        return Decision(
            action=ACTION_EXIT, trade_allowed=True, direction=0,
            confidence=0.6,
            strike=StrikeRecommendation("PE", 0, "PE_ATM"),
            thesis_state=thesis.composite_state,
            invalidation_rule="bear thesis confidence decayed below exit",
            exit_rule="exit — bear score crossed exit threshold",
            no_trade_reason="",
            spread_friendliness=avg_friendliness, notes=notes,
        )

    # ── 3. SCALP windings (trap zones) ────────────────────────────────
    if winding is not None and winding.zone != NO_WINDING:
        if winding.confidence >= cfg.min_scalp_winding_confidence:
            if winding.zone == BULL_TRAP_WINDING:
                return Decision(
                    action=ACTION_SCALP_PUT, trade_allowed=True,
                    direction=-1, confidence=winding.confidence,
                    strike=_strike_for_scalp("PE", cfg),
                    thesis_state=thesis.composite_state,
                    invalidation_rule=("spot breaks above upper band OR "
                                       "battlefield turns clean-bullish"),
                    exit_rule="exit when winding zone resolves up or spread worsens",
                    no_trade_reason="",
                    spread_friendliness=avg_friendliness,
                    notes=["bull-trap winding scalp"],
                )
            if winding.zone == BEAR_TRAP_WINDING:
                return Decision(
                    action=ACTION_SCALP_CALL, trade_allowed=True,
                    direction=+1, confidence=winding.confidence,
                    strike=_strike_for_scalp("CE", cfg),
                    thesis_state=thesis.composite_state,
                    invalidation_rule=("spot breaks below lower band OR "
                                       "battlefield turns clean-bearish"),
                    exit_rule="exit when winding zone resolves down or spread worsens",
                    no_trade_reason="",
                    spread_friendliness=avg_friendliness,
                    notes=["bear-trap winding scalp"],
                )

    # ── 4. ENTRY rules ───────────────────────────────────────────────
    # Thesis memory's BULL_ENTRY / BEAR_ENTRY is the primary entry trigger.
    # The decision can also fire on the continuation_trigger state (state 6)
    # of the held machine when the operator is already in.
    if thesis.composite_state == BULL_ENTRY:
        conf = max(iv_state.confidence or 0.0, battlefield.confidence or 0.0)
        if conf >= cfg.min_directional_confidence:
            return Decision(
                action=ACTION_ENTER_LONG, trade_allowed=True, direction=+1,
                confidence=conf,
                strike=_strike_for_directional(+1, cfg),
                thesis_state=thesis.composite_state,
                invalidation_rule=("REGIME_BREAK_DOWN or STRONG_BULL_TRAP "
                                   "→ exit; battlefield bearish → reassess"),
                exit_rule="exit when bull score < exit threshold OR state 8 fires",
                no_trade_reason="",
                spread_friendliness=avg_friendliness,
                notes=["bull entry from thesis memory"],
            )
    if thesis.composite_state == BEAR_ENTRY:
        conf = max(iv_state.confidence or 0.0, battlefield.confidence or 0.0)
        if conf >= cfg.min_directional_confidence:
            return Decision(
                action=ACTION_ENTER_SHORT, trade_allowed=True, direction=-1,
                confidence=conf,
                strike=_strike_for_directional(-1, cfg),
                thesis_state=thesis.composite_state,
                invalidation_rule=("REGIME_BREAK_UP or STRONG_BEAR_TRAP "
                                   "→ exit; battlefield bullish → reassess"),
                exit_rule="exit when bear score < exit threshold OR state 8 fires",
                no_trade_reason="",
                spread_friendliness=avg_friendliness,
                notes=["bear entry from thesis memory"],
            )

    # Continuation trigger from the held machine — extends a held position.
    if (thesis.held_direction == +1 and bull_state is not None
            and bull_state.state_index == 6):
        return Decision(
            action=ACTION_HOLD, trade_allowed=True, direction=+1,
            confidence=0.85,
            strike=_strike_for_directional(+1, cfg),
            thesis_state=thesis.composite_state,
            invalidation_rule="bull thesis damage / break",
            exit_rule="standard bull exit rules",
            no_trade_reason="",
            spread_friendliness=avg_friendliness,
            notes=["bull continuation trigger — extend held long"],
        )
    if (thesis.held_direction == -1 and bear_state is not None
            and bear_state.state_index == 6):
        return Decision(
            action=ACTION_HOLD, trade_allowed=True, direction=-1,
            confidence=0.85,
            strike=_strike_for_directional(-1, cfg),
            thesis_state=thesis.composite_state,
            invalidation_rule="bear thesis damage / break",
            exit_rule="standard bear exit rules",
            no_trade_reason="",
            spread_friendliness=avg_friendliness,
            notes=["bear continuation trigger — extend held short"],
        )

    # Liquidity sweep reversal trigger (state 5).
    if sweep_state is not None and sweep_state.state_index == 5:
        if sweep_state.note and "take CE" in sweep_state.note:
            return Decision(
                action=ACTION_SCALP_CALL, trade_allowed=True, direction=+1,
                confidence=0.7,
                strike=_strike_for_scalp("CE", cfg),
                thesis_state=thesis.composite_state,
                invalidation_rule="new low taken / call rail collapses",
                exit_rule="exit at first weakness on call rail",
                no_trade_reason="",
                spread_friendliness=avg_friendliness,
                notes=["bullish liquidity-sweep reversal"],
            )
        if sweep_state.note and "take PE" in sweep_state.note:
            return Decision(
                action=ACTION_SCALP_PUT, trade_allowed=True, direction=-1,
                confidence=0.7,
                strike=_strike_for_scalp("PE", cfg),
                thesis_state=thesis.composite_state,
                invalidation_rule="new high taken / put rail collapses",
                exit_rule="exit at first weakness on put rail",
                no_trade_reason="",
                spread_friendliness=avg_friendliness,
                notes=["bearish liquidity-sweep reversal"],
            )

    # ── 5. HOLD rules ────────────────────────────────────────────────
    if thesis.composite_state == HOLD_BULL:
        return Decision(
            action=ACTION_HOLD, trade_allowed=True, direction=+1,
            confidence=min(1.0, thesis.bull_thesis_score / 100.0),
            strike=_strike_for_directional(+1, cfg),
            thesis_state=thesis.composite_state,
            invalidation_rule="bull thesis damage / break",
            exit_rule="exit when bull score < exit threshold",
            no_trade_reason="",
            spread_friendliness=avg_friendliness,
            notes=["holding long"],
        )
    if thesis.composite_state == HOLD_BEAR:
        return Decision(
            action=ACTION_HOLD, trade_allowed=True, direction=-1,
            confidence=min(1.0, thesis.bear_thesis_score / 100.0),
            strike=_strike_for_directional(-1, cfg),
            thesis_state=thesis.composite_state,
            invalidation_rule="bear thesis damage / break",
            exit_rule="exit when bear score < exit threshold",
            no_trade_reason="",
            spread_friendliness=avg_friendliness,
            notes=["holding short"],
        )

    # ── 6. NEUTRAL / WAIT ────────────────────────────────────────────
    return Decision(
        action=ACTION_WAIT, trade_allowed=False, direction=0,
        confidence=0.0,
        strike=StrikeRecommendation("", 0, ""),
        thesis_state=thesis.composite_state,
        invalidation_rule="", exit_rule="",
        no_trade_reason="no actionable read on this bar",
        spread_friendliness=avg_friendliness, notes=notes,
    )
