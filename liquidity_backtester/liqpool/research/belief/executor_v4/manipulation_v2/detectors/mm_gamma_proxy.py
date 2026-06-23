"""MMGammaProxyDetector — proxy net dealer gamma without OI feed.

Real net dealer gamma is computed from open interest at strikes around
spot. We don't have OI here. We proxy it from:

  * **Dispersion velocity**: collapsing dispersion across strikes (low
    and decreasing) typically reflects long-gamma suppression — moves
    get absorbed. Expanding dispersion reflects short-gamma — moves
    accelerate.
  * **Epicenter migration distance**: tight epicenter (small migration)
    + low dispersion = pinned regime = LONG gamma. Big migration with
    rising dispersion = SHORT gamma.
  * **Net intent velocity**: high |net_intent_velocity| with constant
    direction = short-gamma amplification.

The detector emits not a direction (gamma is a state, not a side) but
a **regime modifier** that the fusion layer reads. We encode this by
returning direction=0 with a confidence > 0 and a classification of
either ``long_gamma_suppression`` or ``short_gamma_amplification``.
The fusion uses ``classification`` to know whether to amplify or mute
the other detectors.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class MMGammaProxyDetector(DetectorBase):
    name = "mm_gamma_proxy"

    def __init__(self, *,
                  dispersion_low: float = 0.15,
                  dispersion_high: float = 0.45,
                  migration_high: float = 2.5,
                  intent_velocity_high: float = 1.0,
                  ) -> None:
        self.dispersion_low = float(dispersion_low)
        self.dispersion_high = float(dispersion_high)
        self.migration_high = float(migration_high)
        self.intent_velocity_high = float(intent_velocity_high)

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        battlefield = _map(snapshot.get("battlefield"))
        ce_disp = float(_map(battlefield.get("ce_rail")).get(
            "dispersion_score") or 0.0)
        pe_disp = float(_map(battlefield.get("pe_rail")).get(
            "dispersion_score") or 0.0)
        avg_disp = (ce_disp + pe_disp) / 2.0

        migration = 0.0
        net_intent_velocity = 0.0
        dispersion_velocity = 0.0
        if rich_context is not None:
            migration = float(getattr(
                rich_context, "epicenter_migration_distance", 0.0) or 0.0)
            net_intent_velocity = float(getattr(
                rich_context, "net_intent_velocity", 0.0) or 0.0)
            dispersion_velocity = float(getattr(
                rich_context, "dispersion_velocity", 0.0) or 0.0)

        score = 0.0     # > 0 → short-gamma; < 0 → long-gamma
        evidence: list[str] = []
        if avg_disp <= self.dispersion_low and migration <= 1.5:
            score -= 0.5
            evidence.append(
                f"dispersion low ({avg_disp:.2f}), migration "
                f"{migration:.1f} — suppressed regime")
        if avg_disp >= self.dispersion_high:
            score += 0.5
            evidence.append(
                f"dispersion high ({avg_disp:.2f}) — amplified regime")
        if migration >= self.migration_high:
            score += 0.3
            evidence.append(
                f"epicenter migration {migration:.1f} high — chasing flows")
        if dispersion_velocity > 0.1:
            score += 0.2
            evidence.append(
                f"dispersion velocity rising ({dispersion_velocity:+.3f})")
        elif dispersion_velocity < -0.1:
            score -= 0.2
            evidence.append(
                f"dispersion velocity falling ({dispersion_velocity:+.3f})")
        if abs(net_intent_velocity) >= self.intent_velocity_high:
            score += 0.2
            evidence.append(
                f"|net_intent_velocity|={abs(net_intent_velocity):.2f} "
                f"— directional pressure outpacing absorption")

        if abs(score) < 0.4:
            return DetectorPosterior.quiet()

        classification = ("short_gamma_amplification" if score > 0
                          else "long_gamma_suppression")
        confidence = min(0.85, 0.40 + 0.20 * (abs(score) - 0.4))
        magnitude = DetectorMagnitude(
            z_score=float(score),
            raw_value=float(avg_disp),
            units="gamma regime score")
        # Gamma is direction-neutral; horizon depends on regime.
        horizon = 12 if score > 0 else 18
        return DetectorPosterior(
            fire=True,
            probability=min(0.80, 0.40 + 0.20 * abs(score)),
            direction=0, confidence=confidence,
            horizon_bars=horizon,
            magnitude=magnitude,
            evidence=evidence,
            classification=classification,
        )


def _map(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
