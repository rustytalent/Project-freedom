"""ManipulationEngineV2 — the orchestrator + operator-override surface.

Public API used by the manager:

    engine = ManipulationEngineV2()
    intent = engine.observe(snapshot, rich_context, web_snapshot,
                              bar_index)
    # intent: MMIntent dataclass

    engine.snapshot_at_entry(position_id, intent, bar_index)
    engine.attribute_close(position_id, realised_r)

    engine.set_operator_override(direction=+1, horizon_bars=10,
                                  reason="news catalyst")
    engine.clear_operator_override()
    engine.is_overridden() → bool

Operator override:
    The cockpit lets the operator declare "I see something the model
    doesn't, allow direction X for N bars." When active, ``observe()``
    still computes the fused MMIntent, but the override is annotated
    on it (operator_override field), the aggregator uses the override
    direction in its gate. The override decays after horizon_bars and
    is logged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .calibrator import OutcomeCalibrator
from .detectors.abnormal_acceptance import AbnormalAcceptanceDetector
from .detectors.base import DetectorBase, DetectorPosterior
from .detectors.cancel_rate import CancelRateDetector
from .detectors.cross_rail import CrossRailAsymmetryDetector
from .detectors.depth_pressure import DepthPressureDetector
from .detectors.dod_signature import DodSignatureDetector
from .detectors.iceberg import IcebergDetector
from .detectors.layering import LayeringDetector
from .detectors.mm_gamma_proxy import MMGammaProxyDetector
from .detectors.oi_velocity import OIVelocityDetector
from .detectors.pin_risk import PinRiskDetector
from .detectors.sweep import SweepDetector
from .detectors.trend_vs_range import TrendVsRangeDetector
from .fusion import BayesianFusion, MMIntent


@dataclass
class ManipulationEngineConfig:
    """Knobs."""
    detector_learning_rate: float = 0.04
    min_fires_before_learning: int = 10
    enable_self_calibration: bool = True
    # When an override is active without a fresh refresh, it auto-decays
    # this many bars after set_operator_override.
    override_auto_decay_bars: int = 30


class ManipulationEngineV2:
    """The orchestrator."""

    def __init__(self,
                  cfg: Optional[ManipulationEngineConfig] = None,
                  ) -> None:
        self.cfg = cfg or ManipulationEngineConfig()
        self.detectors: Dict[str, DetectorBase] = {
            "sweep": SweepDetector(),
            "cross_rail_asymmetry": CrossRailAsymmetryDetector(),
            "dod_signature": DodSignatureDetector(),
            "abnormal_acceptance": AbnormalAcceptanceDetector(),
            "pin_risk": PinRiskDetector(),
            "mm_gamma_proxy": MMGammaProxyDetector(),
            "trend_vs_range": TrendVsRangeDetector(),
            # Microstructure detectors (real L2 + OI from Kite).
            "oi_velocity": OIVelocityDetector(),
            "depth_pressure": DepthPressureDetector(),
            "layering": LayeringDetector(),
            "iceberg": IcebergDetector(),
            "cancel_rate": CancelRateDetector(),
        }
        self.fusion = BayesianFusion()
        self.calibrator = OutcomeCalibrator(
            detector_names=list(self.detectors.keys()),
            learning_rate=self.cfg.detector_learning_rate,
            min_fires_before_learning=self.cfg.min_fires_before_learning,
        )
        self._operator_override: Optional[Dict[str, Any]] = None
        self._last_intent: Optional[MMIntent] = None
        self._last_per_detector_posteriors: Dict[str, DetectorPosterior] = {}
        self._last_bar_index: int = -1

    # ── Observation ────────────────────────────────────────────────

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> MMIntent:
        posteriors: Dict[str, DetectorPosterior] = {}
        for name, det in self.detectors.items():
            try:
                p = det.observe(
                    snapshot=snapshot,
                    rich_context=rich_context,
                    web_snapshot=web_snapshot,
                    bar_index=bar_index,
                )
            except Exception:
                p = DetectorPosterior.quiet()
            posteriors[name] = p

        weights = (self.calibrator.weights
                   if self.cfg.enable_self_calibration
                   else {n: 1.0 for n in self.detectors.keys()})

        intent = self.fusion.fuse(posteriors=posteriors, weights=weights)

        # Operator override.
        override = self._operator_override
        if override is not None:
            set_bar = int(override.get("set_at_bar", bar_index))
            horizon = int(override.get("horizon_bars",
                                          self.cfg.override_auto_decay_bars))
            if bar_index - set_bar >= horizon:
                # Auto-decay.
                self._operator_override = None
            else:
                override = dict(override)
                override["bars_remaining"] = max(
                    0, horizon - (bar_index - set_bar))
                intent.operator_override = override
                # The override is AUTHORITATIVE — replace direction +
                # confidence accordingly.
                intent.direction = int(override.get("direction", 0))
                intent.confidence = max(intent.confidence,
                                              float(override.get(
                                                  "confidence", 0.75)))
                intent.notes.append(
                    f"operator override active "
                    f"(dir={intent.direction:+d}, "
                    f"bars_remaining={override['bars_remaining']})")

        self._last_intent = intent
        self._last_per_detector_posteriors = posteriors
        self._last_bar_index = bar_index
        return intent

    # ── Operator override ──────────────────────────────────────────

    def set_operator_override(self, *,
                                 direction: int,
                                 horizon_bars: int = 10,
                                 reason: str = "",
                                 confidence: float = 0.75,
                                 ) -> Dict[str, Any]:
        self._operator_override = {
            "direction": int(direction),
            "horizon_bars": int(horizon_bars),
            "reason": str(reason or ""),
            "confidence": float(confidence),
            "set_at_bar": int(self._last_bar_index),
            "set_at_ts": time.time(),
        }
        return dict(self._operator_override)

    def clear_operator_override(self) -> bool:
        had = self._operator_override is not None
        self._operator_override = None
        return had

    def is_overridden(self) -> bool:
        return self._operator_override is not None

    # ── Outcome attribution ────────────────────────────────────────

    def snapshot_at_entry(self, *,
                              position_id: str,
                              bar_index: int,
                              ) -> None:
        intent = self._last_intent
        if intent is None:
            return
        self.calibrator.snapshot_at_entry(
            position_id=position_id,
            bar_index=bar_index,
            intent_dict=intent.to_dict(),
            per_detector=self._last_per_detector_posteriors,
        )

    def attribute_close(self, *,
                            position_id: str,
                            realised_r: float,
                            ) -> Optional[Dict[str, Any]]:
        return self.calibrator.attribute_close(
            position_id=position_id, realised_r=realised_r,
        )

    # ── Summary ────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        last = self._last_intent
        return {
            "last_intent": last.to_dict() if last is not None else None,
            "calibrator": self.calibrator.summary(),
            "operator_override": (dict(self._operator_override)
                                     if self._operator_override is not None
                                     else None),
            "detector_count": len(self.detectors),
        }
