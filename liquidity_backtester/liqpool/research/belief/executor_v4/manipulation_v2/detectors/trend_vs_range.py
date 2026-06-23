"""TrendVsRangeDetector — regime moderator the founder explicitly asked for.

"In a trend, the put will become cheaper and cheaper and cheaper and
we will enter and enter and enter and we will lose." That is the trap:
naive dod-cheap signals look like opportunity but are actually
relentless trend pressure marking the wrong side ever cheaper.

This detector classifies the current regime as **trending**, **ranging**,
or **transitioning**, using:
  * Spot velocity persistence (consecutive same-sign bars)
  * Substrate's thesis velocity dominance (bull vs bear)
  * Dispersion velocity (rising → trending, falling → ranging)
  * Web's directional_consensus_horizon_weighted (above floor →
    trending in that direction)

When the regime is TRENDING, the fusion layer is told to flip
interpretation of the dod-signature detector: a persistently "cheap"
put rail in a clear bear trend is NOT a buy signal; it's a trap. The
trend regime is itself the MM intent — it tells us direction with
confidence ∝ persistence.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class TrendVsRangeDetector(DetectorBase):
    name = "trend_vs_range"

    def __init__(self, *,
                  consensus_floor: float = 0.20,
                  persistence_bars: int = 4,
                  ) -> None:
        self.consensus_floor = float(consensus_floor)
        self.persistence_bars = int(persistence_bars)
        self._consensus_history: Deque[Tuple[int, float]] = deque(maxlen=32)

    def reset(self) -> None:
        self._consensus_history.clear()

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        consensus = 0.0
        if web_snapshot is not None:
            consensus = float(getattr(
                web_snapshot,
                "directional_consensus_horizon_weighted", 0.0) or 0.0)
        self._consensus_history.append((bar_index, consensus))

        if rich_context is None:
            return DetectorPosterior.quiet()

        if len(self._consensus_history) < self.persistence_bars:
            return DetectorPosterior.quiet()

        recent = list(self._consensus_history)[-self.persistence_bars:]
        recent_values = [r[1] for r in recent]

        # Persistent strong consensus = TRENDING in consensus direction.
        all_pos = all(v > self.consensus_floor for v in recent_values)
        all_neg = all(v < -self.consensus_floor for v in recent_values)

        thesis_side = str(getattr(
            rich_context, "thesis_velocity_dominant_side", "flat") or "flat")
        regime_stability = float(getattr(
            rich_context, "regime_stability_index", 0.5) or 0.5)
        dispersion_velocity = float(getattr(
            rich_context, "dispersion_velocity", 0.0) or 0.0)

        if all_pos and thesis_side == "bull":
            direction = 1
            classification = "trending_bull"
        elif all_neg and thesis_side == "bear":
            direction = -1
            classification = "trending_bear"
        elif regime_stability >= 0.7 and abs(dispersion_velocity) < 0.05:
            direction = 0
            classification = "ranging"
        else:
            classification = "transitioning"
            direction = (1 if recent_values[-1] > self.consensus_floor
                          else (-1 if recent_values[-1] < -self.consensus_floor
                                else 0))

        if classification == "ranging":
            # In a ranging regime the detector contributes
            # direction-neutral context — it modulates other detectors
            # via the fusion layer.
            confidence = min(0.80, regime_stability)
            probability = 0.55
            horizon = 16
        elif classification.startswith("trending"):
            avg_consensus = sum(recent_values) / len(recent_values)
            confidence = min(0.92,
                              0.50 + 0.20 * (abs(avg_consensus)
                                              - self.consensus_floor))
            probability = min(0.90, 0.55 + 0.30 * abs(avg_consensus))
            horizon = 14
        else:
            confidence = 0.40
            probability = 0.45
            horizon = 6
            if direction == 0:
                return DetectorPosterior.quiet()

        magnitude = DetectorMagnitude(
            z_score=float(recent_values[-1]),
            raw_value=float(recent_values[-1]),
            units="consensus")
        return DetectorPosterior(
            fire=True, probability=probability,
            direction=int(direction), confidence=confidence,
            horizon_bars=horizon,
            magnitude=magnitude,
            evidence=[
                f"consensus history {[round(v, 2) for v in recent_values]}",
                f"thesis side={thesis_side}, regime_stability="
                f"{regime_stability:.2f}, dispersion_velocity="
                f"{dispersion_velocity:+.3f}",
            ],
            classification=classification,
        )
