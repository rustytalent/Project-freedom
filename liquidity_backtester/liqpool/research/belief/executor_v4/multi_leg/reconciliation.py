"""Partial-fill reconciliation for multi-leg bundles.

Rule: a multi-leg structure is only valid if ALL legs fill. If we
submit four legs and the broker fills three before rejecting the
fourth, we'd be left holding an unintended directional spread — a
disaster for an iron condor whose whole point is being delta-neutral.

This module owns the logic that decides what to do in that case:

  1. Inspect the per-leg fill results.
  2. If every leg filled, the bundle opens cleanly.
  3. If a leg was rejected (or partially filled in a way that breaks
     the bundle), reverse every leg that successfully filled. We don't
     try to retry the broken leg — by the time we noticed, the price
     has likely moved away and racing to chase the fill is precisely
     the slippage trap the founder spent the live session catching.

The reconciler does NOT itself submit reverse orders — it returns the
list of reverse LegSpec instances and the caller (broker adapter or
manager) decides how to fan them out. This keeps reconciliation pure
and unit-testable; only the broker integration is stateful.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .bundle import (
    BUNDLE_OPEN, BUNDLE_PROPOSED, BUNDLE_REJECTED, BUNDLE_ROLLED_BACK,
    MultiLegBundle,
)
from .leg_spec import LegSpec


# Reconciliation outcomes.
RECON_OK = "OK"
RECON_ROLLED_BACK = "ROLLED_BACK"
RECON_REJECTED = "REJECTED"


@dataclass
class _FillReport:
    """What we know about one leg's broker result, normalised."""
    leg_index: int
    leg: LegSpec
    filled: bool
    filled_quantity: int
    avg_fill_price: float
    rejection_reason: Optional[str]


@dataclass
class BundleReconciliationOutcome:
    """The reconciler's verdict + the actions the caller must take."""
    outcome: str                              # RECON_OK / RECON_ROLLED_BACK / RECON_REJECTED
    bundle: MultiLegBundle
    filled_legs: List[int] = field(default_factory=list)
    rejected_legs: List[int] = field(default_factory=list)
    reverse_orders: List[LegSpec] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "bundle_id": self.bundle.bundle_id,
            "filled_legs": list(self.filled_legs),
            "rejected_legs": list(self.rejected_legs),
            "reverse_orders": [l.to_dict() for l in self.reverse_orders],
            "notes": list(self.notes),
        }


def _interpret(results: Sequence[Any], legs: Sequence[LegSpec]
                ) -> List[_FillReport]:
    """Convert the broker-result list into our normalised view. The
    broker result type is the executor's ``BrokerOrderResult`` but we
    accept any duck-typed object with ``is_filled``,
    ``filled_quantity``, ``avg_fill_price``, ``rejection_reason``."""
    out: List[_FillReport] = []
    for i, leg in enumerate(legs):
        r = results[i] if i < len(results) else None
        if r is None:
            out.append(_FillReport(
                leg_index=i, leg=leg, filled=False,
                filled_quantity=0, avg_fill_price=0.0,
                rejection_reason="no broker reply",
            ))
            continue
        filled = bool(getattr(r, "is_filled", False))
        out.append(_FillReport(
            leg_index=i, leg=leg, filled=filled,
            filled_quantity=int(getattr(r, "filled_quantity", 0) or 0),
            avg_fill_price=float(getattr(r, "avg_fill_price", 0.0) or 0.0),
            rejection_reason=(str(getattr(r, "rejection_reason", "") or "")
                                or None),
        ))
    return out


def reconcile_partial_fills(*,
                              bundle: MultiLegBundle,
                              results: Sequence[Any],
                              lot_size: int,
                              ) -> BundleReconciliationOutcome:
    """Inspect per-leg fill results; either confirm the bundle as OPEN
    or generate reverse orders for the legs that filled."""
    reports = _interpret(results, bundle.legs)

    filled_legs = [r.leg_index for r in reports if r.filled]
    rejected_legs = [r.leg_index for r in reports if not r.filled]
    notes: List[str] = []

    if not rejected_legs:
        # All legs filled. The bundle is officially OPEN.
        bundle.state = BUNDLE_OPEN
        # Realised net credit from actual fills replaces the estimate.
        actual_net = 0.0
        for r in reports:
            sign = 1.0 if r.leg.is_short else -1.0
            actual_net += sign * r.avg_fill_price * r.leg.lots
        bundle.net_credit_at_entry = actual_net
        notes.append(
            f"bundle filled cleanly: net entry premium ₹{actual_net:.2f}/lot")
        return BundleReconciliationOutcome(
            outcome=RECON_OK, bundle=bundle,
            filled_legs=filled_legs, rejected_legs=[], notes=notes,
        )

    if not filled_legs:
        # No leg filled — the bundle is rejected outright. No reverse
        # orders needed (nothing to undo).
        bundle.state = BUNDLE_REJECTED
        reasons = [r.rejection_reason or "unknown"
                   for r in reports if not r.filled]
        notes.append(f"bundle rejected by broker: {reasons[:2]}")
        return BundleReconciliationOutcome(
            outcome=RECON_REJECTED, bundle=bundle,
            filled_legs=[], rejected_legs=rejected_legs, notes=notes,
        )

    # Partial fill — generate reverse orders for the legs that filled.
    reverse_orders: List[LegSpec] = []
    for r in reports:
        if r.filled:
            reverse_orders.append(r.leg.reversed())
    bundle.state = BUNDLE_ROLLED_BACK
    notes.append(
        f"partial fill: legs {filled_legs} filled, legs {rejected_legs} "
        f"rejected → reversing {len(reverse_orders)} legs to unwind")
    return BundleReconciliationOutcome(
        outcome=RECON_ROLLED_BACK, bundle=bundle,
        filled_legs=filled_legs, rejected_legs=rejected_legs,
        reverse_orders=reverse_orders, notes=notes,
    )
