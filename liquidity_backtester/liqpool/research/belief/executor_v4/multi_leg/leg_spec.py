"""LegSpec — one option leg within a multi-leg structure.

A structured option position (iron condor, jade lizard, butterfly,
strangle, vertical spread) is just a vector of LegSpec instances. Every
downstream component — broker submission, ledger tracking, Greeks
aggregation, exit logic — operates on this vector, so the structure
type itself is just a label attached to the bundle.

Why side strings rather than signed quantity: the broker adapter
already speaks BUY/SELL, and keeping that vocabulary at the leg level
avoids a class of sign-flip bugs when constructing reverse orders
(rollback after a partial fill). Quantity stays positive everywhere;
direction is a property of the side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Sides.
LEG_LONG = "BUY"
LEG_SHORT = "SELL"


# Option types.
OPT_CE = "CE"
OPT_PE = "PE"


@dataclass(frozen=True)
class LegSpec:
    """One option leg.

    Fields:
      * ``role``           — human label like ``"long_otm_ce"``, used by
                              the cockpit and the bundle ledger so the
                              operator can see what each leg's job is in
                              the structure.
      * ``strike``         — option strike (rupees, integer except for
                              fractional-strike instruments which NIFTY
                              isn't).
      * ``option_type``    — ``CE`` / ``PE``
      * ``side``           — ``BUY`` / ``SELL`` (mapped to broker side)
      * ``lots``           — positive integer; shares = lots × lot_size
      * ``tradingsymbol``  — exchange-recognised symbol (e.g.
                              ``NIFTY24JUL24000CE``). Optional at construct
                              time; the structure builder fills it from
                              the strike-grid lookup.
      * ``limit_price``    — caller-determined limit price; the
                              policy/strategy_library sets this from
                              the current mid-mark + the configured
                              slippage budget.
      * ``estimated_premium`` — what the caller priced this leg at when
                              constructing the bundle (used for combined
                              R and combined premium projection).
    """
    role: str
    strike: float
    option_type: str          # CE / PE
    side: str                 # BUY / SELL
    lots: int
    tradingsymbol: str = ""
    limit_price: Optional[float] = None
    estimated_premium: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience views.

    @property
    def is_long(self) -> bool:
        return self.side == LEG_LONG

    @property
    def is_short(self) -> bool:
        return self.side == LEG_SHORT

    @property
    def signed_lots(self) -> int:
        """Lots with sign: +long, -short. Useful for combined-Greeks
        accumulation."""
        return self.lots if self.is_long else -self.lots

    def reversed(self) -> "LegSpec":
        """Same leg but with the side flipped. Used by the partial-fill
        reconciler to unwind a successfully-filled leg when a later leg
        in the same bundle was rejected."""
        return LegSpec(
            role=self.role + "_reverse",
            strike=self.strike,
            option_type=self.option_type,
            side=LEG_SHORT if self.is_long else LEG_LONG,
            lots=self.lots,
            tradingsymbol=self.tradingsymbol,
            limit_price=self.limit_price,
            estimated_premium=self.estimated_premium,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "strike": self.strike,
            "option_type": self.option_type,
            "side": self.side,
            "lots": self.lots,
            "tradingsymbol": self.tradingsymbol,
            "limit_price": self.limit_price,
            "estimated_premium": self.estimated_premium,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LegSpec":
        return cls(
            role=str(d.get("role", "")),
            strike=float(d.get("strike", 0.0)),
            option_type=str(d.get("option_type", "")),
            side=str(d.get("side", "")),
            lots=int(d.get("lots", 0)),
            tradingsymbol=str(d.get("tradingsymbol") or ""),
            limit_price=(float(d["limit_price"])
                          if d.get("limit_price") is not None else None),
            estimated_premium=(float(d["estimated_premium"])
                                  if d.get("estimated_premium") is not None
                                  else None),
            metadata=dict(d.get("metadata") or {}),
        )
