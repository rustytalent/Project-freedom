"""ExitThesis — structured exit plan per open position.

Every open position carries an ExitThesis that captures:

  * Four price tiers (long perspective; symmetric for shorts)
    - ideal_exit_price       — target if the thesis plays out perfectly
    - acceptable_exit_price  — still a "win" even in shade mode
    - minimum_profit_exit    — last line above breakeven
    - invalidation_price     — premium stop
  * Revisit expectations
    - expected_revisit_window_bars
    - expected_revisit_price (the next local high we expect)
  * Operating envelope
    - max_hold_seconds
    - thesis_confidence_at_entry
  * Current state
    - current_ghost_exit_price (model's ideal RIGHT NOW)
    - current_broker_exit_price (live limit order on the broker)

The ghost vs broker split is the heart of the modification-budget logic:
the model can re-cost the ghost every tick; the broker order only moves
when the ghost moves enough to justify spending one of the 25 mods.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExitThesisConfig:
    """Knobs for constructing the initial exit thesis from a hypothesis."""
    # Fraction-of-stop for each tier (long: above entry; short: below)
    acceptable_fraction_of_target: float = 0.70   # 70% of target → still happy
    minimum_fraction_of_target: float = 0.30      # 30% → break-even-ish
    # Operating envelope
    default_max_hold_seconds: float = 3600.0      # 1 hour for intraday default
    scalp_max_hold_seconds: float = 600.0         # 10 minutes for scalps
    # Revisit window — how long until we expect to see the next favorable revisit
    default_revisit_window_bars: int = 12
    # Noise filter — micro-reductions of ghost don't reduce broker
    min_meaningful_change_pct: float = 0.005      # 0.5% of premium


@dataclass
class ExitThesis:
    """Structured exit plan for one position."""
    position_id: str
    direction: int                     # +1 long, -1 short
    entry_premium: float
    # Four-tier targets (in premium space)
    ideal_exit_price: float
    acceptable_exit_price: float
    minimum_profit_exit: float
    invalidation_price: float
    # Operating envelope
    max_hold_seconds: float
    thesis_confidence_at_entry: float
    expected_revisit_window_bars: int
    expected_revisit_price: float
    # Current state (mutable through the position's life)
    current_ghost_exit_price: float
    current_broker_exit_price: float
    # Bookkeeping
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "direction": self.direction,
            "entry_premium": round(self.entry_premium, 4),
            "ideal_exit_price": round(self.ideal_exit_price, 4),
            "acceptable_exit_price": round(self.acceptable_exit_price, 4),
            "minimum_profit_exit": round(self.minimum_profit_exit, 4),
            "invalidation_price": round(self.invalidation_price, 4),
            "max_hold_seconds": round(self.max_hold_seconds, 2),
            "thesis_confidence_at_entry": round(self.thesis_confidence_at_entry, 3),
            "expected_revisit_window_bars": self.expected_revisit_window_bars,
            "expected_revisit_price": round(self.expected_revisit_price, 4),
            "current_ghost_exit_price": round(self.current_ghost_exit_price, 4),
            "current_broker_exit_price": round(self.current_broker_exit_price, 4),
            "notes": list(self.notes),
        }

    # ── Profit-tier helpers ──────────────────────────────────────────

    def profit_at_price(self, price: float, *, lots: int = 1,
                          lot_size: int = 65) -> float:
        """Expected ₹ profit per the founder's lot model if exited at price."""
        if self.direction > 0:
            return (price - self.entry_premium) * lots * lot_size
        return (self.entry_premium - price) * lots * lot_size

    def is_above_minimum(self, price: float) -> bool:
        """For long: price ≥ minimum_profit_exit. For short: ≤."""
        if self.direction > 0:
            return price >= self.minimum_profit_exit
        return price <= self.minimum_profit_exit

    def is_at_or_better_than_ideal(self, price: float) -> bool:
        if self.direction > 0:
            return price >= self.ideal_exit_price
        return price <= self.ideal_exit_price

    def reduce_ghost_toward(self, new_price: float, *,
                              cfg: Optional[ExitThesisConfig] = None) -> bool:
        """Tighten the ghost exit price toward the current market. Returns
        True if a meaningful change occurred."""
        cfg = cfg or ExitThesisConfig()
        old = self.current_ghost_exit_price
        if self.direction > 0:
            # For long, ghost reducing means moving DOWN.
            if new_price >= old:
                return False
            change_pct = (old - new_price) / max(1e-6, old)
            if change_pct < cfg.min_meaningful_change_pct:
                return False
            # Never reduce below minimum.
            new_price = max(new_price, self.minimum_profit_exit)
        else:
            # Short: ghost reducing means moving UP.
            if new_price <= old:
                return False
            change_pct = (new_price - old) / max(1e-6, old)
            if change_pct < cfg.min_meaningful_change_pct:
                return False
            new_price = min(new_price, self.minimum_profit_exit)
        self.current_ghost_exit_price = new_price
        return True

    def raise_ghost_toward(self, new_price: float, *,
                             cfg: Optional[ExitThesisConfig] = None) -> bool:
        """Raise the ghost exit price further into profit territory."""
        cfg = cfg or ExitThesisConfig()
        old = self.current_ghost_exit_price
        if self.direction > 0:
            if new_price <= old:
                return False
            change_pct = (new_price - old) / max(1e-6, old)
            if change_pct < cfg.min_meaningful_change_pct:
                return False
            new_price = min(new_price, self.ideal_exit_price * 1.10)
        else:
            if new_price >= old:
                return False
            change_pct = (old - new_price) / max(1e-6, old)
            if change_pct < cfg.min_meaningful_change_pct:
                return False
            new_price = max(new_price, self.ideal_exit_price * 0.90)
        self.current_ghost_exit_price = new_price
        return True


# ── Construction ──────────────────────────────────────────────────


def build_exit_thesis(*,
                        position_id: str,
                        direction: int,
                        entry_premium: float,
                        target_premium: float,
                        stop_premium: float,
                        thesis_confidence: float,
                        profile: str = "INTRADAY",
                        expected_revisit_window_bars: Optional[int] = None,
                        cfg: Optional[ExitThesisConfig] = None,
                        ) -> ExitThesis:
    """Construct an ExitThesis from a PositionHypothesis."""
    cfg = cfg or ExitThesisConfig()
    if entry_premium <= 0 or target_premium <= 0 or stop_premium <= 0:
        raise ValueError("entry/target/stop premiums must be positive")

    # Compute the four tiers (long perspective; mirror for shorts)
    target_gap = abs(target_premium - entry_premium)
    if direction > 0:
        ideal = target_premium
        acceptable = entry_premium + cfg.acceptable_fraction_of_target * target_gap
        minimum = entry_premium + cfg.minimum_fraction_of_target * target_gap
        invalidation = stop_premium
    else:
        ideal = target_premium
        acceptable = entry_premium - cfg.acceptable_fraction_of_target * target_gap
        minimum = entry_premium - cfg.minimum_fraction_of_target * target_gap
        invalidation = stop_premium

    # Operating envelope
    max_hold = (cfg.scalp_max_hold_seconds if profile.upper() == "SCALP"
                 else cfg.default_max_hold_seconds)
    revisit_window = (expected_revisit_window_bars
                       if expected_revisit_window_bars is not None
                       else cfg.default_revisit_window_bars)
    expected_revisit_price = acceptable    # safe initial estimate

    return ExitThesis(
        position_id=position_id,
        direction=direction,
        entry_premium=entry_premium,
        ideal_exit_price=ideal,
        acceptable_exit_price=acceptable,
        minimum_profit_exit=minimum,
        invalidation_price=invalidation,
        max_hold_seconds=max_hold,
        thesis_confidence_at_entry=thesis_confidence,
        expected_revisit_window_bars=revisit_window,
        expected_revisit_price=expected_revisit_price,
        current_ghost_exit_price=ideal,
        current_broker_exit_price=ideal,
        notes=[f"thesis built at entry for {profile.lower()} profile"],
    )
