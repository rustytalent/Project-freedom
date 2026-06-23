"""AbnormalAcceptanceDetector — clusters of 'defended' or 'ignored' acceptance.

Each slot reading has an ``acceptance`` field {normal, defended, ignored}.
This detector looks at distributions across the slot grid:

  * A **cluster of 'defended' slots on one rail** signals coordinated
    MM defense at those strikes. Direction = away from the defended
    side (MM defends CE → they expect spot down).
  * A **cluster of 'ignored' slots** means premiums are getting marked
    through without resistance — directional flow that MM isn't
    fighting. Direction = toward the side being ignored.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class AbnormalAcceptanceDetector(DetectorBase):
    name = "abnormal_acceptance"

    def __init__(self, *,
                  defended_threshold: int = 3,
                  ignored_threshold: int = 4,
                  ) -> None:
        self.defended_threshold = int(defended_threshold)
        self.ignored_threshold = int(ignored_threshold)

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        slots = list(snapshot.get("slot_readings") or [])
        if not slots:
            return DetectorPosterior.quiet()

        # Count defended / ignored per rail.
        per_rail: Dict[str, Counter] = {
            "CE": Counter(), "PE": Counter()}
        for raw in slots:
            slot = raw if isinstance(raw, dict) else {}
            opt = str(slot.get("option_type") or "")
            if opt not in per_rail:
                continue
            acc = str(slot.get("acceptance") or "normal").lower()
            per_rail[opt][acc] += 1

        ce_defended = per_rail["CE"].get("defended", 0)
        pe_defended = per_rail["PE"].get("defended", 0)
        ce_ignored = per_rail["CE"].get("ignored", 0)
        pe_ignored = per_rail["PE"].get("ignored", 0)

        evidence: List[str] = []
        fires: List = []     # list of (direction, prob, conf, magnitude, classification)

        if ce_defended >= self.defended_threshold:
            evidence.append(
                f"{ce_defended} CE slots show 'defended' acceptance")
            fires.append((-1, 0.55 + 0.05 * (ce_defended - self.defended_threshold),
                            min(0.85, 0.50 + 0.05 * (ce_defended - 2)),
                            ce_defended, "defended_ce_cluster"))
        if pe_defended >= self.defended_threshold:
            evidence.append(
                f"{pe_defended} PE slots show 'defended' acceptance")
            fires.append((1, 0.55 + 0.05 * (pe_defended - self.defended_threshold),
                            min(0.85, 0.50 + 0.05 * (pe_defended - 2)),
                            pe_defended, "defended_pe_cluster"))
        if ce_ignored >= self.ignored_threshold:
            # CE rail ignored = premium being marked through up = spot up.
            evidence.append(
                f"{ce_ignored} CE slots show 'ignored' acceptance")
            fires.append((1, 0.50 + 0.04 * (ce_ignored - self.ignored_threshold),
                            min(0.75, 0.45 + 0.04 * (ce_ignored - 3)),
                            ce_ignored, "ignored_ce_cluster"))
        if pe_ignored >= self.ignored_threshold:
            evidence.append(
                f"{pe_ignored} PE slots show 'ignored' acceptance")
            fires.append((-1, 0.50 + 0.04 * (pe_ignored - self.ignored_threshold),
                            min(0.75, 0.45 + 0.04 * (pe_ignored - 3)),
                            pe_ignored, "ignored_pe_cluster"))

        if not fires:
            return DetectorPosterior.quiet()

        # Pick the strongest fire.
        fires.sort(key=lambda f: f[2], reverse=True)
        direction, prob, conf, count, classification = fires[0]
        magnitude = DetectorMagnitude(
            z_score=float(count) / 2.0,    # crude normalisation
            raw_value=float(count), units="slot count")
        return DetectorPosterior(
            fire=True, probability=min(0.92, prob),
            direction=int(direction),
            confidence=min(0.92, conf),
            horizon_bars=8,
            magnitude=magnitude,
            evidence=evidence,
            classification=classification,
        )
