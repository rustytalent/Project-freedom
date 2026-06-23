"""BayesianFusion — combine seven detector posteriors into one MMIntent.

For each detector we have:
  * fire ∈ {True, False}
  * probability ∈ [0, 1]
  * direction ∈ {-1, 0, +1}
  * confidence ∈ [0, 1]
  * horizon_bars
  * weight (from the SelfCalibrator)

The fusion computes a directional vote:
  vote_dir = Σ (weight_i × confidence_i × direction_i × P_i)
  for fired detectors with direction != 0

A normalised log-odds is then mapped to a confidence score in [0, 1].
The MM gamma proxy is direction-neutral (direction=0) but its
classification ('short_gamma_amplification' vs 'long_gamma_suppression')
gain-modulates the directional vote: amplification multiplies the
magnitude of the vote; suppression damps it.

The TrendVsRange detector is treated specially: when it classifies the
regime as TRENDING, the dod_signature contribution is FLIPPED if its
direction contradicts the trend — that's the founder's "in a trend,
cheap is a trap" rule baked into the fusion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .detectors.base import DetectorPosterior


@dataclass
class MMIntent:
    """The fused output of the manipulation engine.

    direction:        +1 / 0 / -1
    confidence:       in [0, 1]
    horizon_bars:     typical bars over which intent persists
    targeted_strike:  strike most-likely targeted (if any)
    per_detector:     individual posteriors for transparency
    regime:           'trending_bull' / 'trending_bear' / 'ranging' /
                       'transitioning' / 'unknown' — from TrendVsRange
    gamma_regime:     'short_gamma_amplification' / 'long_gamma_suppression' /
                       'unknown' — from MMGammaProxy
    fire_count:       number of detectors that fired
    composite_score:  signed directional score (raw, pre-confidence map)
    notes:            human-readable summary lines
    operator_override:dict if a manual override is active for this tick
    """
    direction: int = 0
    confidence: float = 0.0
    horizon_bars: int = 5
    targeted_strike: Optional[float] = None
    per_detector: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regime: str = "unknown"
    gamma_regime: str = "unknown"
    spoof_regime: str = "unknown"
    fire_count: int = 0
    composite_score: float = 0.0
    notes: List[str] = field(default_factory=list)
    operator_override: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": int(self.direction),
            "confidence": round(self.confidence, 4),
            "horizon_bars": int(self.horizon_bars),
            "targeted_strike": (round(float(self.targeted_strike), 2)
                                  if self.targeted_strike is not None
                                  else None),
            "per_detector": dict(self.per_detector),
            "regime": self.regime,
            "gamma_regime": self.gamma_regime,
            "spoof_regime": self.spoof_regime,
            "fire_count": int(self.fire_count),
            "composite_score": round(self.composite_score, 4),
            "notes": list(self.notes),
            "operator_override": (dict(self.operator_override)
                                     if self.operator_override is not None
                                     else None),
        }


class BayesianFusion:
    """Combine detector posteriors into one MMIntent."""

    def __init__(self) -> None:
        pass

    def fuse(self,
              posteriors: Dict[str, DetectorPosterior],
              weights: Dict[str, float],
              ) -> MMIntent:
        notes: List[str] = []

        regime = "unknown"
        gamma_regime = "unknown"
        spoof_regime = "unknown"
        trend_direction = 0
        trend_confidence = 0.0
        gamma_factor = 1.0
        # Book-detector damping factor — applied to layering /
        # depth_pressure / iceberg when CancelRateDetector says the book
        # is spoof-heavy.
        book_damp_factor = 1.0

        trend_post = posteriors.get("trend_vs_range")
        if trend_post is not None and trend_post.fire:
            regime = trend_post.classification or "unknown"
            if regime.startswith("trending"):
                trend_direction = trend_post.direction
                trend_confidence = trend_post.confidence
                notes.append(
                    f"trend regime {regime} (dir={trend_direction:+d}, "
                    f"conf={trend_confidence:.2f})")

        gamma_post = posteriors.get("mm_gamma_proxy")
        if gamma_post is not None and gamma_post.fire:
            gamma_regime = gamma_post.classification or "unknown"
            if gamma_regime == "short_gamma_amplification":
                gamma_factor = 1.0 + 0.30 * gamma_post.confidence
            elif gamma_regime == "long_gamma_suppression":
                gamma_factor = 1.0 - 0.40 * gamma_post.confidence
            notes.append(
                f"gamma regime {gamma_regime} (factor={gamma_factor:.2f})")

        cancel_post = posteriors.get("cancel_rate")
        if cancel_post is not None and cancel_post.fire:
            spoof_regime = cancel_post.classification or "unknown"
            if spoof_regime == "spoofing_dominant":
                book_damp_factor = max(0.0,
                                            1.0 - 0.70 * cancel_post.confidence)
                notes.append(
                    f"spoof regime: damping book detectors × "
                    f"{book_damp_factor:.2f}")
            elif spoof_regime == "clean_book":
                book_damp_factor = 1.0 + 0.15 * cancel_post.confidence
                notes.append(
                    f"clean book: book detectors × "
                    f"{book_damp_factor:.2f}")

        composite = 0.0
        fire_count = 0
        targeted: List = []     # (strike, weight) for averaging

        # Detectors that read the order book — damped when CancelRate
        # says the book is spoof-dominant.
        book_reading = {"layering", "depth_pressure", "iceberg"}
        for name, post in posteriors.items():
            if not post.fire or name in (
                    "trend_vs_range", "mm_gamma_proxy", "cancel_rate"):
                continue
            fire_count += 1
            weight = float(weights.get(name, 1.0))
            direction = float(post.direction)
            # Sweep / dod / acceptance / pin / cross_rail all carry
            # direction. The dod-signature contribution gets *flipped*
            # when the trend regime contradicts it strongly (the
            # founder's "cheap in a trend is a trap" rule).
            if (name == "dod_signature"
                    and trend_direction != 0
                    and direction == -trend_direction
                    and post.classification.startswith("rail_tilt")
                    and trend_confidence >= 0.55):
                direction = float(trend_direction)
                notes.append(
                    f"flipped dod rail_tilt direction (trend "
                    f"{trend_direction:+d} overrides cheap-side trap)")
            # Book-reading detectors get damped under spoof regime so a
            # fake L2 picture doesn't trigger spurious intent.
            detector_damp = (book_damp_factor
                             if name in book_reading else 1.0)
            contribution = (direction * post.probability
                             * post.confidence * weight * detector_damp)
            composite += contribution
            if post.targeted_strike is not None:
                targeted.append((post.targeted_strike,
                                  post.confidence * weight
                                  * detector_damp))

        # Apply gamma factor + trend bias.
        composite *= gamma_factor
        if trend_direction != 0:
            composite += (trend_direction * trend_confidence
                            * 0.40 * gamma_factor)

        # Map composite score → direction + confidence.
        if abs(composite) < 0.20:
            direction = 0
            confidence = 0.0
        else:
            direction = 1 if composite > 0 else -1
            confidence = min(0.95, 1.0 / (1.0 + math.exp(-1.8 * abs(composite))))
            confidence = max(confidence, min(0.95,
                              0.20 + 0.40 * (abs(composite) - 0.20)))

        # Horizon: probability-weighted average of contributing detectors.
        horizons = []
        h_weights = []
        for name, post in posteriors.items():
            if not post.fire or post.direction == 0:
                continue
            w = float(weights.get(name, 1.0)) * post.confidence
            horizons.append(post.horizon_bars * w)
            h_weights.append(w)
        total_w = sum(h_weights) or 1.0
        horizon = max(1, int(round(sum(horizons) / total_w)))

        # Targeted strike: weighted average of fires that supplied one.
        target_strike: Optional[float] = None
        if targeted:
            ws = sum(w for _, w in targeted) or 1.0
            target_strike = sum(s * w for s, w in targeted) / ws

        per_detector = {n: p.to_dict() for n, p in posteriors.items()}

        return MMIntent(
            direction=direction, confidence=confidence,
            horizon_bars=horizon, targeted_strike=target_strike,
            per_detector=per_detector,
            regime=regime, gamma_regime=gamma_regime,
            spoof_regime=spoof_regime,
            fire_count=fire_count, composite_score=composite,
            notes=notes,
        )
