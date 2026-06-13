"""Orchestration spine — the missing layer.

The founder's diagnosis: "we have organs; the missing layer is live
orchestration — what data comes in, where it goes, which model consumes
it, which output is trusted, which is logged, which can affect
execution, which is research/shadow only."

This module is that contract, as code. Every output produced anywhere
in Sentinel is wrapped in a ``Signal`` carrying a ``Tier``. The
``Orchestrator`` routes each signal to the sinks its tier permits — and
**only EXECUTION-tier signals are ever handed to the order path.** This
is the single rule that guarantees a research model can never silently
move real money, no matter how confident it is.

Graduation: a signal source starts at SHADOW. The curator promotes it
(SHADOW -> LOGGED -> TRUSTED) as its ledger-validated track record
earns it. Nothing reaches EXECUTION except the two hard-wired exit
actors (trailing stop, profit lock), and they only ever exit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

LOG = logging.getLogger("sentinel.orchestration")


class Tier(IntEnum):
    """Ordered: higher = more authority. Comparisons are meaningful."""
    SHADOW = 0       # research only; never surfaced, only ledgered
    LOGGED = 1       # recorded for research; optionally shown
    TRUSTED = 2      # shown to the operator as a recommendation
    EXECUTION = 3    # may touch the order path (exits only, v1)


@dataclass
class Signal:
    source: str                  # which organ emitted it
    tier: Tier
    kind: str                    # "exit" | "recommendation" | "context" | ...
    payload: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""             # why — never empty for TRUSTED+

    def __post_init__(self) -> None:
        if self.tier >= Tier.TRUSTED and not self.reason:
            raise ValueError(
                f"{self.source}: TRUSTED+ signals must carry a reason "
                f"(the founder's rule: every actionable output is explained)")


# Sink signatures
ExecutionSink = Callable[[Signal], Any]      # only EXECUTION signals reach this
SurfaceSink = Callable[[Signal], None]       # operator-facing
LedgerSink = Callable[[Signal], None]        # research record


class Orchestrator:
    """Routes signals by tier. The sinks are injected so the spine is
    testable in isolation and the server wires the real ones.

    Routing table (the contract, enforced):
        SHADOW     -> ledger
        LOGGED     -> ledger (+ surface if show_logged)
        TRUSTED    -> ledger + surface
        EXECUTION  -> ledger + surface + execution   (and ONLY this tier
                      ever reaches execution)
    """

    def __init__(self,
                 ledger_sink: Optional[LedgerSink] = None,
                 surface_sink: Optional[SurfaceSink] = None,
                 execution_sink: Optional[ExecutionSink] = None,
                 show_logged: bool = False,
                 allow_execution: bool = True) -> None:
        self.ledger_sink = ledger_sink
        self.surface_sink = surface_sink
        self.execution_sink = execution_sink
        self.show_logged = show_logged
        self.allow_execution = allow_execution
        # graduation registry: source -> max tier it's allowed to emit at.
        # A source can request EXECUTION, but if it hasn't graduated, the
        # spine CLAMPS it down. This is the curator's lever.
        self._ceilings: Dict[str, Tier] = {}
        self.routed: Dict[str, int] = {t.name: 0 for t in Tier}
        self.clamped = 0

    def set_ceiling(self, source: str, tier: Tier) -> None:
        """Curator promotes/demotes a source's maximum tier."""
        self._ceilings[source] = tier

    def _effective_tier(self, sig: Signal) -> Tier:
        # Hard-wired exit actors are allowed EXECUTION by identity.
        if sig.source in ("trailing_stop", "profit_lock"):
            ceiling = Tier.EXECUTION
        else:
            # Everything else defaults to SHADOW until graduated.
            ceiling = self._ceilings.get(sig.source, Tier.SHADOW)
        if sig.tier > ceiling:
            self.clamped += 1
            LOG.info("clamped %s from %s to %s (not graduated)",
                     sig.source, sig.tier.name, ceiling.name)
            return ceiling
        return sig.tier

    def route(self, sig: Signal) -> Optional[Any]:
        """Route one signal. Returns the execution sink's result when the
        signal both requests AND is permitted EXECUTION; else None."""
        tier = self._effective_tier(sig)
        self.routed[tier.name] += 1
        # Ledger always receives everything (the substrate sees all).
        if self.ledger_sink is not None:
            self.ledger_sink(sig)
        # Surface
        if tier >= Tier.TRUSTED or (tier == Tier.LOGGED and self.show_logged):
            if self.surface_sink is not None:
                self.surface_sink(sig)
        # Execution — the one gated path.
        if tier == Tier.EXECUTION:
            if not self.allow_execution:
                LOG.warning("execution signal from %s blocked: "
                            "allow_execution is False", sig.source)
                return None
            if self.execution_sink is None:
                return None
            return self.execution_sink(sig)
        return None

    def stats(self) -> Dict[str, Any]:
        return {"routed": dict(self.routed), "clamped": self.clamped,
                "ceilings": {k: v.name for k, v in self._ceilings.items()}}

    def ceiling_of(self, source: str) -> Tier:
        """The source's current max tier (hard-wired actors are EXECUTION
        by identity; everything else defaults to SHADOW)."""
        if source in ("trailing_stop", "profit_lock"):
            return Tier.EXECUTION
        return self._ceilings.get(source, Tier.SHADOW)


# ---------------------------------------------------------------------------
# Trust promotion record — the curator's graduation decision, logged.
# (Codex §5.1 TrustPromotionRecord.) Every promotion/demotion is an
# auditable event: who, from what, to what, on what evidence, when.
# ---------------------------------------------------------------------------

@dataclass
class TrustPromotionRecord:
    source: str
    from_tier: str
    to_tier: str
    decision: str                       # "promoted" | "demoted" | "held"
    ts_utc: str = ""
    evidence_window: str = ""           # e.g. "2026-06-01..2026-06-13"
    n: Optional[int] = None
    hit_rate: Optional[float] = None
    expectancy: Optional[float] = None
    drawdown: Optional[float] = None
    calibration_error: Optional[float] = None
    leakage_status: Optional[str] = None
    note: str = ""

    @staticmethod
    def build(source: str, from_tier: Tier, to_tier: Tier,
              ts_utc: str = "", **evidence: Any) -> "TrustPromotionRecord":
        decision = ("promoted" if to_tier > from_tier else
                    "demoted" if to_tier < from_tier else "held")
        allowed = {"evidence_window", "n", "hit_rate", "expectancy",
                   "drawdown", "calibration_error", "leakage_status", "note"}
        ev = {k: v for k, v in evidence.items() if k in allowed}
        return TrustPromotionRecord(
            source=source, from_tier=from_tier.name, to_tier=to_tier.name,
            decision=decision, ts_utc=ts_utc, **ev)

    def to_row(self) -> Dict[str, Any]:
        from dataclasses import asdict as _asdict
        return _asdict(self)


from .io_decl import IOSpec, declare
declare(IOSpec(
    module="sentinel.orchestration",
    purpose="trust-tier spine: routes Signals; ONLY EXECUTION tier reaches orders",
    inputs=["Signal(source, tier, kind, payload, reason) from any organ"],
    outputs=["routed effects: ledger_sink / surface_sink / execution_sink"],
    consumes_from=["sentinel.scientists", "sentinel.curator", "sentinel.calibration",
                   "sentinel.trails", "sentinel.profit_lock", "sentinel.leakage_guard"],
    produces_for=["sentinel.server (execution + surface)", "sentinel.shadow_ledger"],
    tier="EXECUTION",
))
