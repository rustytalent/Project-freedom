"""Manipulation detectors — each emits an independent posterior."""
from .base import (
    DetectorBase, DetectorPosterior, DetectorMagnitude, SelfCalibrator,
)
from .sweep import SweepDetector
from .cross_rail import CrossRailAsymmetryDetector
from .dod_signature import DodSignatureDetector
from .abnormal_acceptance import AbnormalAcceptanceDetector
from .pin_risk import PinRiskDetector
from .mm_gamma_proxy import MMGammaProxyDetector
from .trend_vs_range import TrendVsRangeDetector

__all__ = [
    "DetectorBase", "DetectorPosterior", "DetectorMagnitude",
    "SelfCalibrator",
    "SweepDetector", "CrossRailAsymmetryDetector",
    "DodSignatureDetector", "AbnormalAcceptanceDetector",
    "PinRiskDetector", "MMGammaProxyDetector",
    "TrendVsRangeDetector",
]
