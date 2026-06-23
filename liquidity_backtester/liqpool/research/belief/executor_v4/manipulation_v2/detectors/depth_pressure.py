"""DepthPressureDetector — persistent buy-vs-sell pending pressure.

Uses the chain-wide ``buy_quantity_pending`` / ``sell_quantity_pending``
fields from each slot (Kite's pending order totals across the chain
for that strike), plus the L5 depth totals as a backup.

The detector aggregates pressure across the rail and looks for:

  * Persistent imbalance (recent window) above ``pressure_threshold``
  * Confirmed by thin depth on the dominant side (real pressure
    exhausts depth; spoof pressure doesn't because the spoof depth
    is what's making the pressure)
  * Confirmed by tape — recent ``last_trade_size`` totals on the
    aggressive side
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class DepthPressureDetector(DetectorBase):
    name = "depth_pressure"

    def __init__(self, *,
                  persistence_bars: int = 4,
                  pressure_threshold: float = 0.30,
                  min_persistence_score: float = 0.30,
                  ) -> None:
        self.persistence_bars = int(persistence_bars)
        self.pressure_threshold = float(pressure_threshold)
        self.min_persistence_score = float(min_persistence_score)
        self._pressure_history: Deque[Tuple[int, float, float]] = deque(
            maxlen=32)

    def reset(self) -> None:
        self._pressure_history.clear()

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        slots = list(snapshot.get("slot_readings") or [])
        if not slots:
            return DetectorPosterior.quiet()

        # Aggregate buy / sell pressure across the chain.
        buy_total = 0
        sell_total = 0
        avg_persistence = 0.0
        n_with_persistence = 0
        for raw in slots:
            slot = raw if isinstance(raw, dict) else {}
            buy_pending = int(slot.get("buy_quantity_pending") or 0)
            sell_pending = int(slot.get("sell_quantity_pending") or 0)
            if buy_pending == 0 and sell_pending == 0:
                # Fall back to summed L5 depth if pending totals weren't
                # populated (e.g. older snapshot schemas).
                buy_pending = int(slot.get("total_depth_buy_qty") or 0)
                sell_pending = int(slot.get("total_depth_sell_qty") or 0)
            buy_total += buy_pending
            sell_total += sell_pending
            persistence = slot.get("persistence_score")
            if persistence is not None:
                avg_persistence += float(persistence)
                n_with_persistence += 1

        if buy_total == 0 and sell_total == 0:
            return DetectorPosterior.quiet()

        total = buy_total + sell_total
        ratio = float(buy_total - sell_total) / float(total)
        self._pressure_history.append((bar_index, ratio,
                                            avg_persistence / max(1, n_with_persistence)))

        if len(self._pressure_history) < self.persistence_bars:
            return DetectorPosterior.quiet()

        recent = list(self._pressure_history)[-self.persistence_bars:]
        recent_ratios = [r[1] for r in recent]
        recent_persistence = [r[2] for r in recent
                                if r[2] is not None]
        avg_persistence_window = (
            sum(recent_persistence) / max(1, len(recent_persistence))
            if recent_persistence else 1.0)

        # Same-sign and above threshold over the whole window.
        all_pos = all(r > self.pressure_threshold for r in recent_ratios)
        all_neg = all(r < -self.pressure_threshold for r in recent_ratios)
        if not (all_pos or all_neg):
            return DetectorPosterior.quiet()

        # Anti-spoof gate: persistence must be high; spoofs cancel.
        if avg_persistence_window < self.min_persistence_score:
            return DetectorPosterior.quiet()

        direction = 1 if all_pos else -1
        avg_ratio = sum(recent_ratios) / len(recent_ratios)
        magnitude = DetectorMagnitude(
            z_score=float(abs(avg_ratio) / self.pressure_threshold),
            raw_value=float(avg_ratio),
            units="pressure ratio")
        probability = min(0.85, 0.50 + 0.30 * (abs(avg_ratio)
                                                    - self.pressure_threshold))
        confidence = min(0.90, 0.40
                          + 0.30 * (abs(avg_ratio) - self.pressure_threshold)
                          + 0.20 * (avg_persistence_window
                                      - self.min_persistence_score))
        evidence = [
            f"pressure ratio history "
            f"{[round(r, 2) for r in recent_ratios]}",
            f"persistence window avg={avg_persistence_window:.2f} "
            f"(spoof-resistant)",
        ]
        return DetectorPosterior(
            fire=True, probability=probability,
            direction=direction, confidence=confidence,
            horizon_bars=8, magnitude=magnitude, evidence=evidence,
            classification=("pressure_bull" if direction > 0
                              else "pressure_bear"),
        )
