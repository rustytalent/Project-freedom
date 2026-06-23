"""MultiLegBundle — a structured option position (the unit of trade
for iron condor / jade lizard / butterfly / strangle / vertical
spreads).

Lifecycle:

    PROPOSED → submitted to broker
       │
       ├── all legs filled → OPEN
       ├── partial fills, reconciler unwinds → ROLLED_BACK
       └── broker rejection on first leg → REJECTED

    OPEN → exit hits → CLOSING → CLOSED

The bundle is *the* identity at the manager level: P&L, Greeks, exit
policy, the cockpit row — all keyed on bundle_id. Individual legs are
addressable for inspection and for the partial-fill reconciler, but
nothing else in the system thinks per-leg.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .leg_spec import LegSpec


# Bundle states.
BundleState = str
BUNDLE_PROPOSED = "PROPOSED"
BUNDLE_OPEN = "OPEN"
BUNDLE_CLOSING = "CLOSING"
BUNDLE_CLOSED = "CLOSED"
BUNDLE_REJECTED = "REJECTED"
BUNDLE_ROLLED_BACK = "ROLLED_BACK"


@dataclass
class MultiLegBundle:
    """A group of LegSpec instances submitted (and managed) atomically.

    Combined P&L semantics: ``net_credit_at_entry`` is the rupees the
    book received (short legs credit, long legs debit) when the bundle
    opened. ``current_combined_premium`` (set by the bundle ledger from
    live marks) lets the exit policy compute combined R.

    Strategy class lives on the bundle so the cockpit and ledger can
    label rows; the legs themselves don't need to know the structure
    they belong to.
    """
    bundle_id: str
    structure_class: str                    # iron_condor / jade_lizard / butterfly / ...
    legs: List[LegSpec]
    state: BundleState = BUNDLE_PROPOSED
    # Combined economics.
    net_credit_at_entry: float = 0.0        # +ve = credit, -ve = debit
    max_loss_estimate: float = 0.0          # rupees per bundle
    max_profit_estimate: float = 0.0        # rupees per bundle
    # Exit policy (per-bundle, not per-leg).
    target_credit_pct: float = 0.50         # close when credit_decay >= 50%
    stop_loss_pct: float = 1.50             # close when loss >= 1.5x credit
    max_bars_in_position: int = 90
    # Audit.
    notes: List[str] = field(default_factory=list)
    ts_proposed: float = 0.0
    ts_opened: Optional[float] = None
    ts_closed: Optional[float] = None
    bar_opened: Optional[int] = None
    # Carries the regime tag the bundle was opened in (for after-the-
    # fact attribution — "iron condors performed great in chop today
    # but were a disaster in tail").
    regime_tag_at_entry: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bundle_id:
            object.__setattr__(self, "bundle_id", f"b{uuid.uuid4().hex[:11]}")
        if self.ts_proposed == 0.0:
            object.__setattr__(self, "ts_proposed", time.time())

    # ── Combined views ─────────────────────────────────────────────

    def total_lots(self) -> int:
        """Sum of |lots| across legs — useful for capital/margin
        gating."""
        return sum(abs(l.lots) for l in self.legs)

    def long_lots(self) -> int:
        return sum(l.lots for l in self.legs if l.is_long)

    def short_lots(self) -> int:
        return sum(l.lots for l in self.legs if l.is_short)

    def has_balanced_size(self) -> bool:
        """Iron condors / strangles / butterflies have equal long and
        short lot counts. A True here means we won't end up directional
        if all legs fill at the planned size."""
        return self.long_lots() == self.short_lots()

    def estimated_combined_premium(self) -> float:
        """Net entry premium across legs in *rupees per single lot*.

        Positive ⇒ net credit received; negative ⇒ net debit paid.
        Long legs cost the book, short legs credit it.
        """
        total = 0.0
        for leg in self.legs:
            est = float(leg.estimated_premium or 0.0)
            # Per-leg per-single-lot. The lot multiplier is applied at
            # the bundle level when we compute rupees.
            if leg.is_short:
                total += est * leg.lots
            else:
                total -= est * leg.lots
        return total

    def leg_roles(self) -> List[str]:
        return [l.role for l in self.legs]

    # ── Serialisation ───────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "structure_class": self.structure_class,
            "legs": [l.to_dict() for l in self.legs],
            "state": self.state,
            "net_credit_at_entry": round(self.net_credit_at_entry, 2),
            "max_loss_estimate": round(self.max_loss_estimate, 2),
            "max_profit_estimate": round(self.max_profit_estimate, 2),
            "target_credit_pct": self.target_credit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "max_bars_in_position": self.max_bars_in_position,
            "notes": list(self.notes),
            "ts_proposed": self.ts_proposed,
            "ts_opened": self.ts_opened,
            "ts_closed": self.ts_closed,
            "bar_opened": self.bar_opened,
            "regime_tag_at_entry": dict(self.regime_tag_at_entry),
            "total_lots": self.total_lots(),
            "leg_roles": self.leg_roles(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultiLegBundle":
        return cls(
            bundle_id=str(d.get("bundle_id", "")),
            structure_class=str(d.get("structure_class", "")),
            legs=[LegSpec.from_dict(l) for l in (d.get("legs") or [])],
            state=str(d.get("state", BUNDLE_PROPOSED)),
            net_credit_at_entry=float(d.get("net_credit_at_entry", 0.0)),
            max_loss_estimate=float(d.get("max_loss_estimate", 0.0)),
            max_profit_estimate=float(d.get("max_profit_estimate", 0.0)),
            target_credit_pct=float(d.get("target_credit_pct", 0.50)),
            stop_loss_pct=float(d.get("stop_loss_pct", 1.50)),
            max_bars_in_position=int(d.get("max_bars_in_position", 90)),
            notes=list(d.get("notes") or []),
            ts_proposed=float(d.get("ts_proposed", 0.0)),
            ts_opened=(float(d["ts_opened"])
                         if d.get("ts_opened") is not None else None),
            ts_closed=(float(d["ts_closed"])
                         if d.get("ts_closed") is not None else None),
            bar_opened=(int(d["bar_opened"])
                         if d.get("bar_opened") is not None else None),
            regime_tag_at_entry=dict(d.get("regime_tag_at_entry") or {}),
        )
