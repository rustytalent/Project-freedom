"""ManipulationV2 — state-of-the-art manipulation detection engine.

Founder 2026-06-22 directive: "Make this layer at 95%+. The executor
shouldn't take the past, should use only the new modules. Manipulation
detection is the EDGE."

This is the upgrade. Replaces the legacy ``manipulation_patterns`` +
``market_maker_mind`` federation with one coherent Bayesian fusion of
seven independent detectors, each self-calibrated from realised
outcomes, each emitting a posterior + magnitude + horizon. The fused
``MMIntent`` becomes an AUTHORITATIVE input to the aggregator —
contradicting it refuses the trade outright, not a soft penalty.

Architecture:

    detectors/                # Seven independent signature detectors
      sweep.py                # Stop-hunt vs real flow classification
      cross_rail.py           # Persistent CE↔PE rail asymmetry
      dod_signature.py        # Per-slot dod_z patterns (defense walls,
                                single-strike distortions)
      abnormal_acceptance.py  # 'defended' / 'ignored' acceptance clusters
      pin_risk.py             # ATM gravity near expiry
      mm_gamma_proxy.py       # Net dealer gamma proxied from dispersion
                                + epicenter migration (no OI feed needed)
      trend_vs_range.py       # Regime moderator — in a trend, dod-cheap
                                is a trap, not opportunity
    fusion.py                 # BayesianFusion + MMIntent dataclass
    calibrator.py             # OutcomeCalibrator — self-tuning weights
    engine.py                 # ManipulationEngineV2 (the orchestrator)

Public API:

    engine = ManipulationEngineV2()
    intent = engine.observe(snapshot, rich_context, web_snapshot)
    # intent.direction ∈ {+1, 0, -1}
    # intent.confidence ∈ [0, 1]
    # intent.horizon_bars ∈ [1, 30]
    # intent.targeted_strike: Optional[float]
    # intent.per_detector: Dict[name, DetectorPosterior]

Override:

    engine.set_operator_override(direction=+1, horizon_bars=10,
                                  reason="news catalyst, MM hidden")
    # Override is logged + visible in cockpit + decays after horizon_bars

Calibration:

    engine.record_outcome(intent_at_open, realised_r, bars_to_resolve)
    # Each detector's weight is updated by how much its prediction
    # matched the realised outcome.
"""
from .detectors.base import (
    DetectorBase, DetectorPosterior, DetectorMagnitude, SelfCalibrator,
)
from .detectors.sweep import SweepDetector
from .detectors.cross_rail import CrossRailAsymmetryDetector
from .detectors.dod_signature import DodSignatureDetector
from .detectors.abnormal_acceptance import AbnormalAcceptanceDetector
from .detectors.pin_risk import PinRiskDetector
from .detectors.mm_gamma_proxy import MMGammaProxyDetector
from .detectors.trend_vs_range import TrendVsRangeDetector
from .fusion import BayesianFusion, MMIntent
from .calibrator import OutcomeCalibrator
from .engine import ManipulationEngineV2, ManipulationEngineConfig

__all__ = [
    "DetectorBase", "DetectorPosterior", "DetectorMagnitude",
    "SelfCalibrator",
    "SweepDetector", "CrossRailAsymmetryDetector",
    "DodSignatureDetector", "AbnormalAcceptanceDetector",
    "PinRiskDetector", "MMGammaProxyDetector",
    "TrendVsRangeDetector",
    "BayesianFusion", "MMIntent",
    "OutcomeCalibrator",
    "ManipulationEngineV2", "ManipulationEngineConfig",
]
