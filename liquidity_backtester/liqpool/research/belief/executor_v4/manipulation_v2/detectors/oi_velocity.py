"""OIVelocityDetector — open interest velocity + redistribution.

Real OI velocity from the snapshot's per-slot ``oi_velocity_per_min``
field (populated by the live data v2 pipeline). The detector
interprets:

  * **CE rail OI rising while spot moves up**: writers piling in to
    defend the ceiling → expect spot DOWN.
  * **CE rail OI collapsing while spot moves up**: writers covering
    (running for cover) → expect spot UP.
  * **PE rail OI rising while spot moves down**: writers defending
    floor → expect spot UP.
  * **PE rail OI collapsing while spot moves down**: floor giving way
    → expect spot DOWN.

The detector aggregates per-rail velocity and combines with a recent
spot move to classify which of the four states we're in.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class OIVelocityDetector(DetectorBase):
    name = "oi_velocity"

    def __init__(self, *,
                  spot_window_bars: int = 5,
                  velocity_threshold_per_min: float = 500.0,
                  ) -> None:
        self.spot_window_bars = int(spot_window_bars)
        self.velocity_threshold_per_min = float(velocity_threshold_per_min)
        self._spot_history: Deque[Tuple[int, float]] = deque(maxlen=64)

    def reset(self) -> None:
        self._spot_history.clear()

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        spot = float(snapshot.get("spot") or 0.0)
        self._spot_history.append((bar_index, spot))
        slots = list(snapshot.get("slot_readings") or [])
        if not slots:
            return DetectorPosterior.quiet()
        # Per-rail OI velocity sum.
        ce_velocity = 0.0
        pe_velocity = 0.0
        n_ce = 0
        n_pe = 0
        max_strike_ce = (None, 0.0)
        max_strike_pe = (None, 0.0)
        for raw in slots:
            slot = raw if isinstance(raw, dict) else {}
            opt = str(slot.get("option_type") or "")
            v = float(slot.get("oi_velocity_per_min") or 0.0)
            if opt == "CE":
                ce_velocity += v
                n_ce += 1
                if abs(v) > abs(max_strike_ce[1]):
                    max_strike_ce = (slot.get("strike"), v)
            elif opt == "PE":
                pe_velocity += v
                n_pe += 1
                if abs(v) > abs(max_strike_pe[1]):
                    max_strike_pe = (slot.get("strike"), v)

        if n_ce == 0 and n_pe == 0:
            return DetectorPosterior.quiet()

        avg_ce = ce_velocity / max(1, n_ce)
        avg_pe = pe_velocity / max(1, n_pe)
        threshold = self.velocity_threshold_per_min
        ce_strong_pos = avg_ce > threshold
        ce_strong_neg = avg_ce < -threshold
        pe_strong_pos = avg_pe > threshold
        pe_strong_neg = avg_pe < -threshold

        # Spot move over the window.
        if len(self._spot_history) >= self.spot_window_bars:
            window = list(self._spot_history)[-self.spot_window_bars:]
            spot_delta = window[-1][1] - window[0][1]
        else:
            spot_delta = 0.0

        evidence = []
        direction = 0
        classification = ""
        probability = 0.0
        confidence = 0.0
        targeted_strike: Optional[float] = None

        # Defended ceilings — CE writers pile in as spot rises.
        if ce_strong_pos and spot_delta > 0:
            direction = -1
            classification = "ce_defended_ceiling"
            probability = 0.65
            confidence = min(0.85,
                              0.45 + 0.05 * (abs(avg_ce)
                                              / threshold - 1.0))
            targeted_strike = max_strike_ce[0]
            evidence.append(
                f"CE OI velocity {avg_ce:+.0f}/min while spot up "
                f"{spot_delta:+.1f} → writers defending ceiling")
        # CE writers running for cover — spot rises and CE OI collapses.
        elif ce_strong_neg and spot_delta > 0:
            direction = 1
            classification = "ce_unwinding_cover"
            probability = 0.70
            confidence = min(0.85,
                              0.50 + 0.05 * (abs(avg_ce)
                                              / threshold - 1.0))
            targeted_strike = max_strike_ce[0]
            evidence.append(
                f"CE OI velocity {avg_ce:+.0f}/min while spot up "
                f"{spot_delta:+.1f} → CE writers covering")
        # Defended floor — PE writers pile in as spot drops.
        elif pe_strong_pos and spot_delta < 0:
            direction = 1
            classification = "pe_defended_floor"
            probability = 0.65
            confidence = min(0.85,
                              0.45 + 0.05 * (abs(avg_pe)
                                              / threshold - 1.0))
            targeted_strike = max_strike_pe[0]
            evidence.append(
                f"PE OI velocity {avg_pe:+.0f}/min while spot down "
                f"{spot_delta:+.1f} → writers defending floor")
        # PE floor giving way.
        elif pe_strong_neg and spot_delta < 0:
            direction = -1
            classification = "pe_floor_breaking"
            probability = 0.70
            confidence = min(0.85,
                              0.50 + 0.05 * (abs(avg_pe)
                                              / threshold - 1.0))
            targeted_strike = max_strike_pe[0]
            evidence.append(
                f"PE OI velocity {avg_pe:+.0f}/min while spot down "
                f"{spot_delta:+.1f} → PE floor breaking")
        else:
            return DetectorPosterior.quiet()

        magnitude = DetectorMagnitude(
            z_score=float(max(abs(avg_ce), abs(avg_pe)) / threshold),
            raw_value=float(max(abs(avg_ce), abs(avg_pe))),
            units="OI per min")
        return DetectorPosterior(
            fire=True, probability=probability,
            direction=direction, confidence=confidence,
            horizon_bars=10,
            targeted_strike=(float(targeted_strike)
                              if targeted_strike is not None else None),
            magnitude=magnitude, evidence=evidence,
            classification=classification,
        )
