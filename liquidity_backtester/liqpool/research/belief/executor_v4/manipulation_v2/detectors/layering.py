"""LayeringDetector — stacked depth on one side, thin on the other.

Layering signature: a participant stacks 3+ price levels deep on one
side of the book to suggest demand/supply that doesn't exist, intending
to push price the OTHER way. Real (non-spoofed) layering is rare;
spoofed layering is what we have to filter out.

Detection:

  1. Look at each slot's ``depth_buy_levels`` + ``depth_sell_levels``
     (lists of {price, quantity, orders} dicts).
  2. Score each side by how many levels carry "meaningful" quantity
     (above a per-instrument median).
  3. Layering fires when one side has ≥ 3 meaningful levels while the
     other has ≤ 1, AND the slot's ``persistence_score`` ≥ threshold
     (so we don't catch spoofs).
  4. Direction = OPPOSITE the layered side (layering on buy side =
     intent is to sell into the false demand = direction DOWN).

Aggregates across all slots; fires only when ≥ N slots agree on the
side being layered.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class LayeringDetector(DetectorBase):
    name = "layering"

    def __init__(self, *,
                  min_slots_layered: int = 2,
                  min_persistence_score: float = 0.50,
                  meaningful_level_floor: int = 50,
                  ) -> None:
        self.min_slots_layered = int(min_slots_layered)
        self.min_persistence_score = float(min_persistence_score)
        self.meaningful_level_floor = int(meaningful_level_floor)

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        slots = list(snapshot.get("slot_readings") or [])
        if not slots:
            return DetectorPosterior.quiet()

        layered_buy = 0
        layered_sell = 0
        targeted_strikes: List[float] = []
        magnitudes: List[float] = []
        for raw in slots:
            slot = raw if isinstance(raw, dict) else {}
            persistence = float(slot.get("persistence_score") or 1.0)
            if persistence < self.min_persistence_score:
                # Spoof-likely book — skip this slot.
                continue
            buy_levels = list(slot.get("depth_buy_levels") or [])
            sell_levels = list(slot.get("depth_sell_levels") or [])
            buy_meaningful = sum(
                1 for d in buy_levels
                if int((d or {}).get("quantity") or 0)
                >= self.meaningful_level_floor)
            sell_meaningful = sum(
                1 for d in sell_levels
                if int((d or {}).get("quantity") or 0)
                >= self.meaningful_level_floor)
            if buy_meaningful >= 3 and sell_meaningful <= 1:
                layered_buy += 1
                targeted_strikes.append(float(slot.get("strike") or 0.0))
                magnitudes.append(float(buy_meaningful))
            elif sell_meaningful >= 3 and buy_meaningful <= 1:
                layered_sell += 1
                targeted_strikes.append(float(slot.get("strike") or 0.0))
                magnitudes.append(float(sell_meaningful))

        if max(layered_buy, layered_sell) < self.min_slots_layered:
            return DetectorPosterior.quiet()

        if layered_buy > layered_sell:
            # Buy-side layering = intent is to push DOWN.
            direction = -1
            classification = "buy_side_layering"
            count = layered_buy
        else:
            direction = 1
            classification = "sell_side_layering"
            count = layered_sell

        avg_magnitude = (statistics.mean(magnitudes)
                          if magnitudes else 3.0)
        targeted = (statistics.mean(targeted_strikes)
                     if targeted_strikes else None)
        probability = min(0.80, 0.45 + 0.10 * (count - 2))
        confidence = min(0.75, 0.40 + 0.10 * (count - 2)
                          + 0.05 * (avg_magnitude - 3.0))
        magnitude = DetectorMagnitude(
            z_score=float(avg_magnitude),
            raw_value=float(count),
            units="layered slots")
        return DetectorPosterior(
            fire=True, probability=probability,
            direction=direction, confidence=confidence,
            horizon_bars=6,
            targeted_strike=(float(targeted) if targeted is not None
                              else None),
            magnitude=magnitude,
            evidence=[
                f"{count} slots layered ({classification})",
                f"avg meaningful levels per layered side: "
                f"{avg_magnitude:.1f}",
                f"persistence gate ≥ {self.min_persistence_score:.2f} "
                f"(spoof filter)",
            ],
            classification=classification,
        )
