"""CancelRateDetector — spoofing regime modulator.

High aggregate cancel rate + low persistence score = spoofing-rich
environment. Direction-neutral; the role is to SUPPRESS other
detectors that read book depth in that regime (the fusion layer
reads this detector's classification + confidence and damps the
contribution of layering / depth_pressure / iceberg).

Like the MMGammaProxy this detector emits direction=0 and the fusion
layer uses ``classification`` to gain-modulate other contributions.

Two classifications:

  * ``spoofing_dominant`` — high cancel rate, low persistence. Damp
    other book-reading detectors.
  * ``clean_book`` — high persistence, low cancel rate. Amplify
    other book-reading detectors.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class CancelRateDetector(DetectorBase):
    name = "cancel_rate"

    def __init__(self, *,
                  spoof_persistence_ceiling: float = 0.30,
                  clean_persistence_floor: float = 0.70,
                  spoof_cancel_floor: float = 1.5,
                  ) -> None:
        self.spoof_persistence_ceiling = float(spoof_persistence_ceiling)
        self.clean_persistence_floor = float(clean_persistence_floor)
        self.spoof_cancel_floor = float(spoof_cancel_floor)

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        slots = list(snapshot.get("slot_readings") or [])
        if not slots:
            return DetectorPosterior.quiet()
        persistence_values: List[float] = []
        cancel_values: List[float] = []
        for raw in slots:
            slot = raw if isinstance(raw, dict) else {}
            p = slot.get("persistence_score")
            c = slot.get("cancel_rate_per_tick")
            if p is not None:
                persistence_values.append(float(p))
            if c is not None:
                cancel_values.append(float(c))
        if not persistence_values:
            return DetectorPosterior.quiet()
        avg_persistence = statistics.mean(persistence_values)
        avg_cancel = (statistics.mean(cancel_values)
                       if cancel_values else 0.0)

        classification = ""
        evidence: List[str] = []
        if (avg_persistence <= self.spoof_persistence_ceiling
                or avg_cancel >= self.spoof_cancel_floor):
            classification = "spoofing_dominant"
            confidence = min(0.85,
                              0.40 + 0.30 * (1.0 - avg_persistence)
                              + 0.15 * min(1.0,
                                            avg_cancel
                                            / max(1.0, self.spoof_cancel_floor)))
            evidence.append(
                f"avg persistence={avg_persistence:.2f} "
                f"(≤ {self.spoof_persistence_ceiling:.2f}) and/or "
                f"avg cancel rate={avg_cancel:.2f}/tick "
                f"(≥ {self.spoof_cancel_floor:.2f}) — book reads spoof-heavy")
        elif avg_persistence >= self.clean_persistence_floor:
            classification = "clean_book"
            confidence = min(0.80, 0.40 + 0.40 * avg_persistence)
            evidence.append(
                f"avg persistence={avg_persistence:.2f} "
                f"(≥ {self.clean_persistence_floor:.2f}) — book reads "
                f"clean; book detectors trustworthy")
        else:
            return DetectorPosterior.quiet()

        magnitude = DetectorMagnitude(
            z_score=float(1.0 - avg_persistence),
            raw_value=float(avg_persistence),
            units="persistence")
        return DetectorPosterior(
            fire=True,
            probability=min(0.80, 0.40 + 0.30 * (1.0 - avg_persistence)),
            direction=0, confidence=confidence,
            horizon_bars=14,
            magnitude=magnitude,
            evidence=evidence,
            classification=classification,
        )
