"""Hedge layer — proposes protective legs when tail risk rises.

When ``fat_tail_score`` lands in the HEDGE band, OR when the portfolio's
net delta exceeds the risk layer's cap, the hedge layer proposes one or
more protective legs that the operator (or future auto-hedger) can take.

Sprint 4 ships a heuristic proposer — not a full Greek-neutral solver.
The proposals are concrete (side, level, lots, intended reduction)
and serialized to the ledger so post-mortems can grade whether the
hedge actually helped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HedgeProposalConfig:
    """Knobs for the hedge proposer."""
    delta_cap_for_hedge_lots: float = 2.5
    # Default protective wing level — OTM3 for cheap insurance.
    default_wing_level: int = 3
    min_premium_to_propose_rupees: float = 50.0


@dataclass(frozen=True)
class HedgeLeg:
    """One protective leg."""
    contract_side: str          # CE / PE
    contract_level: int
    direction: int              # +1 long, -1 short
    lots: int
    strike_price: float
    premium: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class HedgeProposal:
    """Per-tick hedge proposal."""
    proposed: bool                  # True = open this leg
    proposals: List[HedgeLeg]
    reasons: List[str]
    total_premium_rupees: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposed": self.proposed,
            "proposals": [p.to_dict() for p in self.proposals],
            "reasons": list(self.reasons),
            "total_premium_rupees": round(self.total_premium_rupees, 2),
            "notes": list(self.notes),
        }


class HedgeProposer:
    """Proposes protective legs given current state."""

    def __init__(self, cfg: Optional[HedgeProposalConfig] = None) -> None:
        self.cfg = cfg or HedgeProposalConfig()

    def propose(self, *,
                  fat_tail_action: str,
                  net_delta_lots: float,
                  open_positions: List[Any],
                  strike_lookup: Dict[Any, Any],
                  lot_size: int = 65,
                  ) -> HedgeProposal:
        """Look at current portfolio state and propose hedges.

        Returns ``proposed=False`` when no hedge needed.
        """
        cfg = self.cfg
        legs: List[HedgeLeg] = []
        reasons: List[str] = []
        notes: List[str] = []
        total_premium = 0.0

        if fat_tail_action == "HEDGE":
            reasons.append("fat-tail action HEDGE — propose protective wings")
        if abs(net_delta_lots) > cfg.delta_cap_for_hedge_lots:
            reasons.append(
                f"net delta {net_delta_lots:+.1f} lots > cap "
                f"{cfg.delta_cap_for_hedge_lots:.1f} — neutralize"
            )

        # Only act if we have a reason.
        if not reasons or not open_positions:
            return HedgeProposal(proposed=False, proposals=[],
                                  reasons=reasons,
                                  total_premium_rupees=0.0,
                                  notes=["no hedge needed"])

        # Strategy: if net delta is long, buy OTM PE; if short, buy OTM CE.
        # If symmetric exposure but tail-risk HEDGE, buy a strangle.
        if abs(net_delta_lots) > cfg.delta_cap_for_hedge_lots:
            direction_of_exposure = +1 if net_delta_lots > 0 else -1
            protective_side = "PE" if direction_of_exposure > 0 else "CE"
            protective_level = (-cfg.default_wing_level
                                if direction_of_exposure > 0
                                else cfg.default_wing_level)
            info = strike_lookup.get((protective_side, protective_level))
            if info is None:
                notes.append(
                    f"no quote for protective {protective_side} "
                    f"level {protective_level}"
                )
            else:
                strike, premium, _, _ = info
                lots_needed = max(1, int(round(abs(net_delta_lots) / 2)))
                cost = premium * lots_needed * lot_size
                if cost >= cfg.min_premium_to_propose_rupees:
                    legs.append(HedgeLeg(
                        contract_side=protective_side,
                        contract_level=protective_level,
                        direction=+1, lots=lots_needed,
                        strike_price=strike, premium=premium,
                        rationale=f"protect against opposite move "
                                  f"({lots_needed} lot)",
                    ))
                    total_premium += cost
        elif fat_tail_action == "HEDGE":
            # Symmetric strangle — long OTM CE + long OTM PE.
            for side, level in (
                ("CE", cfg.default_wing_level),
                ("PE", -cfg.default_wing_level),
            ):
                info = strike_lookup.get((side, level))
                if info is None:
                    continue
                strike, premium, _, _ = info
                cost = premium * 1 * lot_size
                if cost < cfg.min_premium_to_propose_rupees:
                    continue
                legs.append(HedgeLeg(
                    contract_side=side, contract_level=level,
                    direction=+1, lots=1,
                    strike_price=strike, premium=premium,
                    rationale=f"strangle wing against tail risk",
                ))
                total_premium += cost

        return HedgeProposal(
            proposed=bool(legs),
            proposals=legs,
            reasons=reasons,
            total_premium_rupees=total_premium,
            notes=notes,
        )
