"""Detector base + posterior + self-calibrator.

Every manipulation detector returns a ``DetectorPosterior`` — a
probability of fire, a signed direction, a magnitude in standardised
units, and a horizon in bars. The fusion layer combines these with
detector-specific *weights* that the ``SelfCalibrator`` updates from
realised outcomes after each closed trade.

A detector that is consistently right gets its weight raised; a
detector that is consistently wrong gets it lowered. Over time the
fusion auto-adjusts to which detectors actually work in the current
market regime.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


# ── Posterior dataclasses ──────────────────────────────────────────


@dataclass
class DetectorMagnitude:
    """Strength of the signal expressed in standardised units.

    ``z_score`` should be how many standard deviations above the
    detector's null distribution the current observation sits. Magnitudes
    above ~2.0 are interesting; above 4.0 are alarming.
    """
    z_score: float = 0.0
    raw_value: float = 0.0
    units: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "z_score": round(self.z_score, 3),
            "raw_value": round(self.raw_value, 4),
            "units": self.units,
        }


@dataclass
class DetectorPosterior:
    """One detector's posterior over manipulation intent.

    Fields:
      * ``fire``                 — boolean: did anything notable happen?
      * ``probability``          — P(MM intent in some direction)
      * ``direction``            — +1 / 0 / -1; which way MM appears to push
      * ``confidence``           — calibrated self-trust in this fire
      * ``horizon_bars``         — how long the intent should persist
      * ``targeted_strike``      — strike being defended/attacked (None if not applicable)
      * ``magnitude``            — DetectorMagnitude(z_score=…, raw_value=…)
      * ``evidence``             — list of human-readable evidence strings
      * ``classification``       — detector-specific label (e.g. 'stop_hunt'
                                     vs 'real_flow' for the sweep detector)
    """
    fire: bool
    probability: float
    direction: int
    confidence: float
    horizon_bars: int = 5
    targeted_strike: Optional[float] = None
    magnitude: DetectorMagnitude = field(default_factory=DetectorMagnitude)
    evidence: List[str] = field(default_factory=list)
    classification: str = ""

    @staticmethod
    def quiet() -> "DetectorPosterior":
        return DetectorPosterior(
            fire=False, probability=0.0, direction=0, confidence=0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fire": self.fire,
            "probability": round(self.probability, 4),
            "direction": int(self.direction),
            "confidence": round(self.confidence, 4),
            "horizon_bars": int(self.horizon_bars),
            "targeted_strike": (round(float(self.targeted_strike), 2)
                                  if self.targeted_strike is not None
                                  else None),
            "magnitude": self.magnitude.to_dict(),
            "evidence": list(self.evidence)[:8],
            "classification": self.classification,
        }


# ── Detector base ─────────────────────────────────────────────────


class DetectorBase:
    """All detectors implement ``observe(snapshot, rich_context,
    web_snapshot) -> DetectorPosterior``.

    Sub-classes typically also maintain a small rolling window of
    feature values for z-score normalisation (the null distribution).
    """

    name: str = "base"

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        raise NotImplementedError

    # Optional: detectors that maintain rolling stats override this.
    def reset(self) -> None:
        pass


# ── Self-calibrator ────────────────────────────────────────────────


@dataclass
class _DetectorScore:
    n_fires: int = 0
    n_correct_direction: int = 0
    weighted_r: float = 0.0           # Σ realised_r when this detector fired
    weighted_r_squared: float = 0.0
    sum_confidence_x_outcome: float = 0.0


class SelfCalibrator:
    """Per-detector weight self-tuning from realised outcomes.

    Each detector's *fusion weight* lives in [w_floor, w_ceiling]. It
    starts at 1.0 and drifts toward higher values for detectors whose
    predictions actually matched realised outcomes, lower for those
    that didn't.

    Update rule per closed trade:
        score_i  = detector_i.direction × realised_r × detector_i.confidence
        weight_i ← weight_i + learning_rate × (score_i − weight_i × baseline)
        weight_i ← clip(weight_i, w_floor, w_ceiling)

    Empty-handed start: weights at 1.0; minimum sample count before any
    learning kicks in (so a single freak trade doesn't blow up).
    """

    def __init__(self, *,
                  detector_names: List[str],
                  learning_rate: float = 0.04,
                  w_floor: float = 0.10,
                  w_ceiling: float = 2.50,
                  min_fires_before_learning: int = 10,
                  ) -> None:
        self.learning_rate = float(learning_rate)
        self.w_floor = float(w_floor)
        self.w_ceiling = float(w_ceiling)
        self.min_fires_before_learning = int(min_fires_before_learning)
        self.weights: Dict[str, float] = {n: 1.0 for n in detector_names}
        self._scores: Dict[str, _DetectorScore] = {
            n: _DetectorScore() for n in detector_names
        }
        # Bounded outcome history for diagnostics / cockpit.
        self._recent_outcomes: Deque[Dict[str, Any]] = deque(maxlen=64)

    def record_outcome(self, *,
                          per_detector_posteriors: Dict[str, DetectorPosterior],
                          realised_r: float,
                          ) -> None:
        """Update each detector's score based on the realised outcome.

        The "right" direction is the sign of realised_r. A detector that
        fired with the right direction and high confidence gets a big
        score bump; wrong direction with high confidence gets a hit.
        """
        if realised_r == 0.0:
            return
        true_direction = 1 if realised_r > 0 else -1
        for name, post in per_detector_posteriors.items():
            if not post.fire:
                continue
            score = self._scores.setdefault(name, _DetectorScore())
            score.n_fires += 1
            score.weighted_r += abs(realised_r)
            score.weighted_r_squared += realised_r * realised_r
            correct = (post.direction == true_direction)
            if correct:
                score.n_correct_direction += 1
            score.sum_confidence_x_outcome += (
                post.direction * realised_r * post.confidence)
        # Update weights once each detector has enough fires.
        for name, score in self._scores.items():
            if score.n_fires < self.min_fires_before_learning:
                continue
            avg_signal = (score.sum_confidence_x_outcome
                           / max(1, score.n_fires))
            cur = self.weights.get(name, 1.0)
            # Drift toward avg_signal (positive = good detector, negative =
            # bad). The (cur − 1.0) term anchors weights to the neutral 1.0.
            new = cur + self.learning_rate * (avg_signal - (cur - 1.0))
            self.weights[name] = max(self.w_floor,
                                          min(self.w_ceiling, new))
        self._recent_outcomes.append({
            "realised_r": realised_r,
            "fires": {n: p.fire for n, p in per_detector_posteriors.items()},
        })

    def summary(self) -> Dict[str, Any]:
        per_det: Dict[str, Dict[str, Any]] = {}
        for name, score in self._scores.items():
            per_det[name] = {
                "weight": round(self.weights.get(name, 1.0), 4),
                "n_fires": score.n_fires,
                "directional_accuracy": (
                    round(score.n_correct_direction / max(1, score.n_fires),
                            3) if score.n_fires > 0 else 0.0),
                "avg_abs_r_when_fired": (
                    round(score.weighted_r / max(1, score.n_fires), 3)
                    if score.n_fires > 0 else 0.0),
            }
        return {
            "weights": dict(self.weights),
            "per_detector": per_det,
            "min_fires_before_learning": self.min_fires_before_learning,
            "n_recent_outcomes": len(self._recent_outcomes),
        }
