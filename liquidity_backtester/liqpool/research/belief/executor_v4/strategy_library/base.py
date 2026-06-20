"""Base classes for the strategy library.

Every strategy is a stateless functor with the four methods:

  * ``can_enter(ctx)`` — entry conditions returning ``(bool, reasons)``
  * ``build(ctx, lots)`` — produce the structured ``StrategyLegs``
  * ``greeks(legs)`` — delta/vega/theta/gamma profile at entry
  * ``early_warning(legs, ctx)`` — strategy-specific invalidation phrases

Each strategy carries a single name + family tag used by the selector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Strategy family tags.
FAMILY_DIRECTIONAL = "directional"
FAMILY_NEUTRAL = "neutral"
FAMILY_VOL_LONG = "vol_long"
FAMILY_VOL_SHORT = "vol_short"
FAMILY_PIN = "pin"


@dataclass(frozen=True)
class StrategyLeg:
    """One leg of a structured position."""
    contract_side: str             # CE / PE
    contract_level: int            # ATM offset (0 = ATM)
    strike_price: float
    direction: int                 # +1 = long, -1 = short
    lots: int
    entry_premium: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class StrategyLegs:
    """A complete structured position."""
    strategy_name: str
    family: str
    legs: List[StrategyLeg]
    primary_direction: int          # +1 / -1 / 0
    max_loss_rupees: float          # ₹ (bounded structures); inf for naked
    max_gain_rupees: float          # ₹ (bounded structures); inf for outright
    breakeven_low: float            # underlying spot at lower breakeven
    breakeven_high: float            # underlying spot at higher breakeven
    net_premium_paid: float          # +ve = paid, -ve = received
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "family": self.family,
            "legs": [l.to_dict() for l in self.legs],
            "primary_direction": self.primary_direction,
            "max_loss_rupees": (None if math.isinf(self.max_loss_rupees)
                                  else round(self.max_loss_rupees, 2)),
            "max_gain_rupees": (None if math.isinf(self.max_gain_rupees)
                                  else round(self.max_gain_rupees, 2)),
            "breakeven_low": round(self.breakeven_low, 2),
            "breakeven_high": round(self.breakeven_high, 2),
            "net_premium_paid": round(self.net_premium_paid, 2),
            "notes": list(self.notes),
        }


@dataclass
class StrategyContext:
    """Everything a strategy needs to make its entry decision.

    A struct of the latest Sprint 2/3 outputs (plus the spot/strike
    lookup helpers).
    """
    spot: float
    lot_size: int
    # Snapshot-driven
    iv_state: str
    bf_verdict: str
    thesis_state: str
    thesis_direction: int
    confidence: float
    # Sprint 2/3
    web_snapshot: Any = None        # WebSnapshot
    mm_posterior: Any = None        # MMPosterior
    fat_tail_score: Any = None      # FatTailScore
    crowd_report: Any = None        # CrowdMirrorReport
    rich_context: Any = None        # RichContext
    flow_event: Any = None
    # Per-strike lookup: side, level → (strike, premium, friendliness, spread_state)
    strike_lookup: Optional[Dict[Tuple[str, int], Tuple[float, float, float, str]]] = None
    # Caller's lots budget
    lots_budget: int = 1
    # Engine-specified preferred strike (CE / PE) when available
    preferred_side: str = ""
    preferred_level: int = 0


@dataclass(frozen=True)
class StrategyEntryDecision:
    """Per-strategy entry verdict."""
    strategy_name: str
    family: str
    can_enter: bool
    reasons: List[str]                # if can_enter False, why; else supporting
    fitness_score: float              # 0..1, higher = better fit for current state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "family": self.family,
            "can_enter": self.can_enter,
            "reasons": list(self.reasons),
            "fitness_score": round(self.fitness_score, 3),
        }


@dataclass
class StrategyOutcome:
    """Per-strategy realized outcome (filled in at close)."""
    strategy_name: str
    realized_rupees: float
    fees_rupees: float
    bars_held: int
    closed_via: str                  # "target" / "stop" / "max_hold" / "kill"

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class BaseStrategy:
    """Abstract base. Subclasses must override.

    The base class provides helpers (premium lookup, lot-size math) so
    every concrete strategy is short.
    """
    name: str = "base"
    family: str = ""

    # ── Required overrides ────────────────────────────────────────────

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        raise NotImplementedError

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        raise NotImplementedError

    # ── Default helpers (can be overridden) ───────────────────────────

    def greeks(self, legs: StrategyLegs, spot: float) -> Dict[str, float]:
        """Approximate per-position greeks.

        We don't have a full Black-Scholes calibration here — instead we
        use simple per-leg approximations sufficient for the risk layer:
          * delta: CE +0.5 at ATM (signed by direction), PE -0.5 at ATM
          * vega: |position|; long vega = +1, short vega = -1 per leg
          * theta: -1 per long leg, +1 per short leg (time decay direction)
          * gamma: 0.5 at ATM, decays |level|/10 from ATM
        """
        delta = vega = theta = gamma = 0.0
        for leg in legs.legs:
            atm_distance = abs(leg.contract_level)
            # Delta
            d = 0.5 if leg.contract_side == "CE" else -0.5
            d *= max(0.10, 1.0 - atm_distance * 0.18)
            delta += d * leg.direction * leg.lots
            # Vega decays with distance from ATM (max at ATM, fades OTM/ITM).
            vega_per_lot = max(0.10, 1.0 - atm_distance * 0.20)
            vega += leg.direction * leg.lots * vega_per_lot
            # Theta (long = negative theta) — also peaks at ATM.
            theta_per_lot = max(0.10, 1.0 - atm_distance * 0.20)
            theta -= leg.direction * leg.lots * theta_per_lot
            # Gamma (long = positive gamma) — peaks sharply at ATM.
            gamma += leg.direction * leg.lots * max(0.10, 1.0
                                                     - atm_distance * 0.25)
        return {
            "delta": round(delta, 3),
            "vega": round(vega, 3),
            "theta": round(theta, 3),
            "gamma": round(gamma, 3),
        }

    def early_warning(self, legs: StrategyLegs,
                       ctx: StrategyContext) -> List[str]:
        """Strategy-specific kill phrases. Override in subclasses."""
        return []

    # ── Internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _lookup(ctx: StrategyContext, side: str,
                  level: int) -> Optional[Tuple[float, float, float, str]]:
        if not ctx.strike_lookup:
            return None
        return ctx.strike_lookup.get((side, level))

    @staticmethod
    def _spot_to_strike(spot: float, step: float = 50.0) -> float:
        return round(spot / step) * step

    @staticmethod
    def _format_strike(side: str, level: int) -> str:
        if level == 0:
            return f"{side}_ATM"
        if level > 0:
            return f"{side}_OTM{level}"
        return f"{side}_ITM{abs(level)}"
