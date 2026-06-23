"""CrossRailAsymmetryDetector — persistent imbalance between CE and PE rails.

When dealers are positioning asymmetrically — defending one side while
relinquishing the other — the CE rail's signed-z and the PE rail's
signed-z diverge persistently. This detector picks up that signature.

Signals:
  * **magnitude** = |ce_signed_z − (−pe_signed_z)|  ↑ when rails diverge
  * **persistence** = rolling correlation > threshold for K bars
  * **direction** = sign of the rail that's stronger (i.e. ce dominant
    bullish, pe dominant bearish)
  * **dispersion floor** = both rails must have low dispersion (focused
    positioning, not spray)

Strong fires require ALL three: large magnitude, persistence across at
least ``persistence_bars``, and tight per-rail dispersion.
"""
from __future__ import annotations

import statistics
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class CrossRailAsymmetryDetector(DetectorBase):
    name = "cross_rail_asymmetry"

    def __init__(self, *,
                  persistence_bars: int = 5,
                  asymmetry_z_threshold: float = 1.2,
                  dispersion_ceiling: float = 0.35,
                  ) -> None:
        self.persistence_bars = int(persistence_bars)
        self.asymmetry_z_threshold = float(asymmetry_z_threshold)
        self.dispersion_ceiling = float(dispersion_ceiling)
        # Each entry: (bar_index, ce_signed_z, pe_signed_z, asymmetry,
        # ce_dispersion, pe_dispersion).
        self._history: Deque[Tuple[int, float, float, float, float, float]] = (
            deque(maxlen=64))

    def reset(self) -> None:
        self._history.clear()

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        bf = _map(snapshot.get("battlefield"))
        ce = _map(bf.get("ce_rail"))
        pe = _map(bf.get("pe_rail"))

        ce_z = float(ce.get("weighted_mean_signed_z") or 0.0)
        pe_z = float(pe.get("weighted_mean_signed_z") or 0.0)
        ce_disp = float(ce.get("dispersion_score") or 0.0)
        pe_disp = float(pe.get("dispersion_score") or 0.0)

        # Asymmetry: ce_z is "buy CE / sell PE" pressure (+), pe_z is
        # "buy PE / sell CE" pressure (+). Symmetric flow has
        # ce_z ≈ −pe_z. The deviation from symmetry is the asymmetry.
        symmetric_pe = -pe_z
        asymmetry = ce_z - symmetric_pe
        self._history.append(
            (bar_index, ce_z, pe_z, asymmetry, ce_disp, pe_disp))

        # Need enough history to assess persistence.
        if len(self._history) < self.persistence_bars + 1:
            return DetectorPosterior.quiet()

        recent = list(self._history)[-self.persistence_bars:]
        recent_asym = [r[3] for r in recent]
        recent_ce_disp = [r[4] for r in recent]
        recent_pe_disp = [r[5] for r in recent]

        # Standardise the asymmetry vs the long-window null.
        long_window = list(self._history)[-min(40, len(self._history)):]
        long_asym = [r[3] for r in long_window]
        mean_asym = (statistics.mean(long_asym)
                     if len(long_asym) > 1 else 0.0)
        std_asym = (statistics.pstdev(long_asym)
                    if len(long_asym) > 2 else 1.0) or 1e-6
        avg_recent = statistics.mean(recent_asym)
        z = (avg_recent - mean_asym) / std_asym

        # All recent bars must show same-sign asymmetry above floor.
        same_sign = all(
            (a > self.asymmetry_z_threshold * std_asym)
            for a in recent_asym
        ) or all(
            (a < -self.asymmetry_z_threshold * std_asym)
            for a in recent_asym
        )
        avg_ce_disp = statistics.mean(recent_ce_disp)
        avg_pe_disp = statistics.mean(recent_pe_disp)
        tight_dispersion = (avg_ce_disp <= self.dispersion_ceiling
                            and avg_pe_disp <= self.dispersion_ceiling)

        if not same_sign or not tight_dispersion:
            return DetectorPosterior.quiet()

        # MM intent direction: positive asymmetry = MM net buying CE
        # (dealer short call gamma = pushing spot UP). Negative
        # asymmetry = MM net buying PE = pushing DOWN.
        direction = 1 if avg_recent > 0 else -1
        probability = min(0.95, 0.55 + 0.10 * max(0.0, abs(z) - 1.0))
        confidence = min(0.95, 0.55 + 0.08 * max(0.0, abs(z) - 1.0)
                          + 0.10 * (1.0 - max(avg_ce_disp, avg_pe_disp)))
        evidence = [
            f"persistent asymmetry z={z:+.2f} for "
            f"{self.persistence_bars}b",
            f"ce_disp={avg_ce_disp:.2f} pe_disp={avg_pe_disp:.2f} "
            f"(focused positioning)",
            f"ce_signed_z={ce_z:+.2f}, pe_signed_z={pe_z:+.2f}",
        ]
        magnitude = DetectorMagnitude(
            z_score=float(z), raw_value=float(avg_recent),
            units="signed_z asymmetry")
        return DetectorPosterior(
            fire=True, probability=probability, direction=direction,
            confidence=confidence,
            horizon_bars=10,
            magnitude=magnitude,
            evidence=evidence,
            classification="rail_asymmetry",
        )


def _map(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
