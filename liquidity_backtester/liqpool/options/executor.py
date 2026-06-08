"""Options executor — pre-trade decision + in-trade state machine.

Implements §3.5 (force-entry bootstrap) and §4 (hold-vs-exit
decision tree) of ``docs/options_executor_layered_conviction.md``.

The executor is rule-based at v1. The rules are:

  Pre-trade
  ─────────
  ENTER  when |LCS| >= FORCE_ENTRY_THRESHOLD
  WAIT   when |LCS| in (WAIT_THRESHOLD, FORCE_ENTRY_THRESHOLD)
  SKIP   otherwise (macro gate closed or signal too weak)

  In-trade (recomputed every 5-min bar)
  ─────────────────────────────────────
  EXIT_KILL          — if any kill condition triggered
  EXIT_INVALIDATION  — if LCS dropped by >= INVALIDATION_LCS_DROP
                       AND agreeing layer count <= 2
  EXIT_TARGET        — if realized R reached or exceeded the
                       per-bucket predicted target
  TRAIL_STOP         — if LCS still high AND profitable
  HOLD               — default

Kill conditions are events the 5-minute layer recomputation cannot
react to fast enough:
  * exchange halt (operator flag from the broker / NSE feed)
  * broker connection loss > 60 s
  * VIX intra-bar spike of more than VIX_KILL_SPIKE points
  * Underlying intra-bar move of more than KILL_UNDERLYING_ATR ATRs

There is NO drawdown floor in production. See D13 for the rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .ccv import CCV, MACRO_GATE_THRESHOLD


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

FORCE_ENTRY_THRESHOLD: float = 0.30
WAIT_THRESHOLD: float = 0.10

INVALIDATION_LCS_DROP: float = 0.40
INVALIDATION_AGREEING_LAYERS_MAX: int = 2

TRAIL_LCS_MIN: float = 0.60
TRAIL_REALIZED_R_MIN: float = 1.0

# Kill conditions
VIX_KILL_SPIKE: float = 3.0
KILL_UNDERLYING_ATR: float = 4.0
BROKER_OUTAGE_KILL_SECONDS: float = 60.0


# ---------------------------------------------------------------------------
# Pre-trade
# ---------------------------------------------------------------------------

@dataclass
class PreTradeDecision:
    """Output of the pre-trade gate. ``side_thesis`` is what direction
    the executor recommends the position be opened on (``buy`` or
    ``sell``) — derived from sign(LCS). ``None`` when SKIP."""
    action: str                     # "ENTER" | "WAIT" | "SKIP"
    reason: str
    lcs: float
    side_thesis: Optional[str]


def pre_trade_decision(ccv: CCV) -> PreTradeDecision:
    """Bootstrap rule from §3.5 of the executor doc.

    No model-derived threshold at v1. Once enough trades have been
    observed to fit the force-entry classifier, the threshold becomes
    data-driven; this function is the structural fallback when the
    classifier hasn't been trained yet for this bucket.
    """
    lcs = ccv.lcs
    if abs(ccv.macro_score) < MACRO_GATE_THRESHOLD:
        return PreTradeDecision(
            action="SKIP",
            reason="macro_gate_closed_no_possibility_space",
            lcs=lcs,
            side_thesis=None,
        )
    if abs(lcs) >= FORCE_ENTRY_THRESHOLD:
        return PreTradeDecision(
            action="ENTER",
            reason=f"lcs_at_or_above_bootstrap_threshold_{FORCE_ENTRY_THRESHOLD}",
            lcs=lcs,
            side_thesis="buy" if lcs > 0 else "sell",
        )
    if abs(lcs) > WAIT_THRESHOLD:
        return PreTradeDecision(
            action="WAIT",
            reason="lcs_in_wait_band",
            lcs=lcs,
            side_thesis=None,
        )
    return PreTradeDecision(
        action="SKIP",
        reason="lcs_below_signal_floor",
        lcs=lcs,
        side_thesis=None,
    )


# ---------------------------------------------------------------------------
# Kill conditions
# ---------------------------------------------------------------------------

@dataclass
class KillConditions:
    """Snapshot of the kill-trigger inputs at one in-trade bar.

    The executor module does not produce these signals — the live-
    runner does (broker connection state from the broker SDK, halt
    status from the exchange feed, VIX from the macro feed, intra-bar
    underlying move from the bar stream). The state machine just
    consumes them.
    """
    exchange_halt: bool = False
    broker_outage_seconds: float = 0.0
    vix_intrabar_spike: float = 0.0
    underlying_intrabar_move_atr: float = 0.0

    def triggered(self) -> Optional[str]:
        """Return the name of the first triggered kill condition, or
        ``None`` when the bar is calm."""
        if self.exchange_halt:
            return "exchange_halt"
        if self.broker_outage_seconds >= BROKER_OUTAGE_KILL_SECONDS:
            return "broker_outage_over_60s"
        if abs(self.vix_intrabar_spike) > VIX_KILL_SPIKE:
            return f"vix_intrabar_spike_{self.vix_intrabar_spike:+.2f}"
        if abs(self.underlying_intrabar_move_atr) > KILL_UNDERLYING_ATR:
            return (
                "underlying_intrabar_move_"
                f"{self.underlying_intrabar_move_atr:+.2f}_atr"
            )
        return None


# ---------------------------------------------------------------------------
# In-trade state and decision
# ---------------------------------------------------------------------------

def agreeing_layer_count(ccv: CCV) -> int:
    """Number of non-macro layers whose sign agrees with the macro
    direction. Counts each of {regime, pool_holding, options,
    micro_organic, manipulation}. Silent (zero-sign) layers do NOT
    count — they aren't disagreeing, but they aren't agreeing either.
    """
    if abs(ccv.macro_score) < MACRO_GATE_THRESHOLD:
        return 0
    macro_dir = 1 if ccv.macro_score > 0 else -1
    layers = [
        ccv.regime_score, ccv.pool_holding_strength, ccv.options_score,
        ccv.micro_organic_score, ccv.manipulation_score,
    ]
    count = 0
    for s in layers:
        if s == 0.0:
            continue
        layer_dir = 1 if s > 0 else -1
        if layer_dir == macro_dir:
            count += 1
    return count


@dataclass
class InTradeState:
    """Inputs the in-trade decision tree consumes every 5 min."""
    ccv_at_entry: CCV
    ccv_now: CCV
    realized_r_atr: float            # signed; negative = against us
    predicted_target_r_atr: float    # the model's predicted_R
    kill: KillConditions = field(default_factory=KillConditions)


@dataclass
class InTradeDecision:
    action: str                      # HOLD | EXIT_KILL | EXIT_INVALIDATION
                                      # | EXIT_TARGET | TRAIL_STOP
    reason: str
    lcs_now: float
    lcs_drop: float
    agreeing_layers_now: int


def in_trade_decision(state: InTradeState) -> InTradeDecision:
    """The §4 decision tree.

    Order matters: kill conditions fire FIRST so a flash event always
    wins; then layer-invalidation; then target; then trail; then HOLD.
    """
    triggered = state.kill.triggered()
    lcs_now = state.ccv_now.lcs
    lcs_drop = state.ccv_at_entry.lcs - lcs_now
    agreeing = agreeing_layer_count(state.ccv_now)

    if triggered is not None:
        return InTradeDecision(
            action="EXIT_KILL",
            reason=triggered,
            lcs_now=lcs_now,
            lcs_drop=lcs_drop,
            agreeing_layers_now=agreeing,
        )
    if (lcs_drop >= INVALIDATION_LCS_DROP
            and agreeing <= INVALIDATION_AGREEING_LAYERS_MAX):
        return InTradeDecision(
            action="EXIT_INVALIDATION",
            reason=(
                f"lcs_drop_{lcs_drop:.2f}_agreeing_layers_{agreeing}"
            ),
            lcs_now=lcs_now,
            lcs_drop=lcs_drop,
            agreeing_layers_now=agreeing,
        )
    if state.realized_r_atr >= state.predicted_target_r_atr:
        return InTradeDecision(
            action="EXIT_TARGET",
            reason=(
                f"realized_r_{state.realized_r_atr:.2f}_at_target_"
                f"{state.predicted_target_r_atr:.2f}"
            ),
            lcs_now=lcs_now,
            lcs_drop=lcs_drop,
            agreeing_layers_now=agreeing,
        )
    if (abs(lcs_now) >= TRAIL_LCS_MIN
            and state.realized_r_atr >= TRAIL_REALIZED_R_MIN):
        return InTradeDecision(
            action="TRAIL_STOP",
            reason=(
                f"lcs_{lcs_now:.2f}_realized_r_{state.realized_r_atr:.2f}"
            ),
            lcs_now=lcs_now,
            lcs_drop=lcs_drop,
            agreeing_layers_now=agreeing,
        )
    return InTradeDecision(
        action="HOLD",
        reason="default_hold",
        lcs_now=lcs_now,
        lcs_drop=lcs_drop,
        agreeing_layers_now=agreeing,
    )


# ---------------------------------------------------------------------------
# Convenience: derive a list of human-readable causal warnings
# ---------------------------------------------------------------------------

def causal_warnings(ccv: CCV) -> List[str]:
    """Build short user-facing strings describing active causal flags.

    These appear next to the LCS scalar in the brief's per-strike
    block (e.g. "⚠ MICRO FORCED"). The customer reads not just the
    number but the why.
    """
    out: List[str] = []
    if ccv.micro_forced_flag:
        out.append(
            "⚠ MICRO FORCED — recent stop-run activity is partly "
            "responsible for the microstructure read; organic "
            f"contribution is {ccv.micro_organic_score:+.2f}."
        )
    if ccv.pool_distortion_flag:
        out.append(
            "⚠ POOL DISTORTED — aggressive flow against the pool "
            "reduces its holding strength to "
            f"{ccv.pool_holding_strength:+.2f}."
        )
    return out
