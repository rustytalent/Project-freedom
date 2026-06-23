"""PinRiskDetector — ATM gravity near expiry.

Near Thursday weekly expiry, MMs hedge net gamma at the strikes closest
to spot. When net dealer gamma is short there, every move accelerates;
when long, moves are suppressed and spot tends to pin a strike.

We can't see real OI here, but we can proxy pin risk from:
  1. Spot's proximity to the nearest strike (close = high pin risk).
  2. Time-to-expiry proxy: if ``bars_to_expiry`` is available on the
     snapshot (set by the live bridge), use it; otherwise we fall back
     to a fixed bias of "any tick on Thursday afternoon = elevated pin
     risk".
  3. Dispersion is suppressed — strikes near spot show muted dod_z
     variance (the pin is suppressing chain-wide vol).

When pin risk fires, the implied DIRECTION is toward the nearest strike
that is currently above the spot if dealers are long-gamma there
(toward the magnet), or away if short-gamma (anti-gravity). Without OI
we default to magnetism — spot gravitates to the nearest high-OI proxy
strike, which we approximate as the nearest strike with the LARGEST
absolute dod_z (where MM is positioned).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class PinRiskDetector(DetectorBase):
    name = "pin_risk"

    def __init__(self, *,
                  proximity_pct: float = 0.0025,
                  max_bars_to_expiry: int = 60,
                  dispersion_ceiling: float = 0.20,
                  ) -> None:
        self.proximity_pct = float(proximity_pct)
        self.max_bars_to_expiry = int(max_bars_to_expiry)
        self.dispersion_ceiling = float(dispersion_ceiling)

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        spot = float(snapshot.get("spot") or 0.0)
        if spot <= 0:
            return DetectorPosterior.quiet()
        slots = list(snapshot.get("slot_readings") or [])
        if not slots:
            return DetectorPosterior.quiet()

        # If the bridge supplied bars_to_expiry, gate on it; otherwise
        # we don't enable pin risk (too noisy mid-week).
        bars_to_expiry = snapshot.get("bars_to_expiry")
        if bars_to_expiry is None:
            return DetectorPosterior.quiet()
        try:
            bte = int(bars_to_expiry)
        except (TypeError, ValueError):
            return DetectorPosterior.quiet()
        if bte > self.max_bars_to_expiry or bte < 0:
            return DetectorPosterior.quiet()

        # Nearest strike.
        ce_strikes = sorted({float(s.get("strike") or 0.0)
                              for s in slots
                              if str(s.get("option_type") or "") == "CE"})
        if not ce_strikes:
            return DetectorPosterior.quiet()
        nearest = min(ce_strikes, key=lambda k: abs(k - spot))
        proximity = abs(nearest - spot) / max(1.0, spot)
        if proximity > self.proximity_pct:
            return DetectorPosterior.quiet()

        # Suppression check: dispersion at near-ATM strikes must be tight.
        battlefield = _map(snapshot.get("battlefield"))
        ce_disp = float(_map(battlefield.get("ce_rail")).get(
            "dispersion_score") or 0.0)
        pe_disp = float(_map(battlefield.get("pe_rail")).get(
            "dispersion_score") or 0.0)
        if max(ce_disp, pe_disp) > self.dispersion_ceiling:
            return DetectorPosterior.quiet()

        # Direction: which side of spot has the larger absolute dod_z
        # — that's where MM has positioned and where spot gravitates to.
        target_strike = nearest
        max_z = 0.0
        for raw in slots:
            slot = raw if isinstance(raw, dict) else {}
            strike = float(slot.get("strike") or 0.0)
            z = abs(float(slot.get("dod_z") or 0.0))
            if z > max_z and abs(strike - spot) <= spot * 0.005:
                max_z = z
                target_strike = strike
        direction = 1 if target_strike > spot else (
            -1 if target_strike < spot else 0)

        # Confidence scales with proximity + how few bars remain.
        time_factor = 1.0 - (bte / max(1, self.max_bars_to_expiry))
        proximity_factor = 1.0 - (
            proximity / max(1e-6, self.proximity_pct))
        confidence = max(0.30, min(0.92,
            0.40 + 0.30 * time_factor + 0.20 * proximity_factor))
        probability = min(0.85, 0.45 + 0.20 * time_factor)
        magnitude = DetectorMagnitude(
            z_score=float(max_z),
            raw_value=float(abs(nearest - spot)),
            units="points to strike")
        return DetectorPosterior(
            fire=True, probability=probability,
            direction=int(direction), confidence=confidence,
            horizon_bars=max(1, bte),
            targeted_strike=float(target_strike),
            magnitude=magnitude,
            evidence=[
                f"spot {spot:.1f} within "
                f"{abs(nearest - spot):.1f} of strike {nearest:.0f}",
                f"bars_to_expiry={bte}, dispersions tight "
                f"(ce={ce_disp:.2f}, pe={pe_disp:.2f})",
            ],
            classification="pin_risk",
        )


def _map(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
