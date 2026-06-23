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
# Microstructure detectors (founder 2026-06-22 follow-up: use the L2 +
# OI data Kite already gives us).
from .oi_velocity import OIVelocityDetector
from .depth_pressure import DepthPressureDetector
from .layering import LayeringDetector
from .iceberg import IcebergDetector
from .cancel_rate import CancelRateDetector

__all__ = [
    "DetectorBase", "DetectorPosterior", "DetectorMagnitude",
    "SelfCalibrator",
    "SweepDetector", "CrossRailAsymmetryDetector",
    "DodSignatureDetector", "AbnormalAcceptanceDetector",
    "PinRiskDetector", "MMGammaProxyDetector",
    "TrendVsRangeDetector",
    "OIVelocityDetector", "DepthPressureDetector",
    "LayeringDetector", "IcebergDetector", "CancelRateDetector",
]
