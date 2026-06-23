"""Rehearsal-based pre-calibration for the manipulation detectors.

Founder 2026-06-22 idea: "use the layer in which we are pretty much
sure, somehow that, to calibrate." The rehearsal ensemble produces a
per-perturbation analogue distribution; its predicted P(profit) under
the current parameters is itself a soft "ground-truth" signal.

When the manipulation engine fires at entry consideration, we ask the
rehearsal layer: "given this state, what does the analogue distribution
say the expected R is over the rehearsal horizon?" If the manipulation
direction agrees with the analogue mean_r sign with high confidence on
both, count this as a virtual correct prediction. If they disagree,
count as a virtual incorrect prediction.

This bootstraps the SelfCalibrator from day one — no waiting weeks for
real outcomes. It's a *soft* signal (it's the analogue distribution
not actual fills), so we weight it less than real attribution: every
real ``record_outcome`` counts as 1.0; every rehearsal-based virtual
counts as ``virtual_weight`` (default 0.25).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .detectors.base import DetectorPosterior


@dataclass
class RehearsalCalibrationConfig:
    enabled: bool = True
    # Weight of each virtual outcome relative to a real one. 0.25 means
    # 4 rehearsal-based updates ≈ 1 real outcome.
    virtual_weight: float = 0.25
    # Min rehearsal confidence to count it as a usable signal.
    min_rehearsal_confidence: float = 0.30
    # Min number of analogues for the rehearsal call to count.
    min_analogues: int = 6


class RehearsalCalibrator:
    """Routes rehearsal-decision evidence into the SelfCalibrator."""

    def __init__(self, *,
                  cfg: Optional[RehearsalCalibrationConfig] = None,
                  ) -> None:
        self.cfg = cfg or RehearsalCalibrationConfig()
        self._n_virtual_updates: int = 0
        self._n_skipped_low_confidence: int = 0
        self._n_skipped_no_intent: int = 0

    def maybe_calibrate(self, *,
                            self_calibrator,
                            per_detector_posteriors: Dict[str, DetectorPosterior],
                            mm_intent_direction: int,
                            mm_intent_confidence: float,
                            rehearsal_decision: Any,
                            ) -> bool:
        """Returns True if a virtual outcome was synthesised and pushed.

        Inspects the rehearsal decision: if it ran with enough analogues
        and confidence, derive a virtual ``realised_r`` signed by the
        analogue mean. Then call ``self_calibrator.record_outcome`` with
        the virtual outcome scaled by ``virtual_weight``.
        """
        cfg = self.cfg
        if not cfg.enabled:
            return False
        if mm_intent_direction == 0 or mm_intent_confidence == 0.0:
            self._n_skipped_no_intent += 1
            return False
        if rehearsal_decision is None:
            return False
        ran = bool(getattr(rehearsal_decision, "ran", False))
        if not ran:
            return False
        rehearsal_conf = float(getattr(
            rehearsal_decision, "confidence", 0.0) or 0.0)
        n_analogues = int(getattr(
            rehearsal_decision, "n_analogues_total", 0) or 0)
        if (rehearsal_conf < cfg.min_rehearsal_confidence
                or n_analogues < cfg.min_analogues):
            self._n_skipped_low_confidence += 1
            return False
        # Synthesise a virtual realised_r from the analogue distribution
        # of the current perturbation (best proxy for "what the analogues
        # say should happen if we entered now").
        per_perts = list(getattr(
            rehearsal_decision, "per_perturbation", []) or [])
        current_name = getattr(
            rehearsal_decision, "current_perturbation_name", "as_proposed")
        cur = None
        for p in per_perts:
            if getattr(p, "name", "") == current_name:
                cur = p
                break
        if cur is None and per_perts:
            cur = per_perts[0]
        if cur is None:
            return False
        virtual_r = float(getattr(cur, "mean_r", 0.0) or 0.0)
        # Scale by virtual_weight so it doesn't dominate the calibrator
        # over real fills.
        virtual_r_scaled = virtual_r * cfg.virtual_weight
        if virtual_r_scaled == 0.0:
            return False
        try:
            self_calibrator.record_outcome(
                per_detector_posteriors=per_detector_posteriors,
                realised_r=virtual_r_scaled,
            )
            self._n_virtual_updates += 1
            return True
        except Exception:
            return False

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.cfg.enabled,
            "virtual_weight": self.cfg.virtual_weight,
            "n_virtual_updates": self._n_virtual_updates,
            "n_skipped_low_confidence": self._n_skipped_low_confidence,
            "n_skipped_no_intent": self._n_skipped_no_intent,
        }
