"""Flywheel models — Stream M.

Seven models that train on data the engine generates about itself
(shadow log, outcome log, walkforward results, calibration history).
M.1-M.3 (slippage / fill / survival) live in ``liqpool/execution``;
this package holds the rest:

  M.4  regret_model          — P(regret | declined decision context)
  M.5  detector_trust        — per-(factor, regime) trust scores
  M.6  drift_imminent        — P(drift fires within k sessions)
  M.7  reaction_archetypes   — cluster + classify post-touch shapes
  M.8  bucket_aging          — calibration error vs bucket age
  M.9  cross_asset           — signal transferability matrix
  M.10 brief_meta_calibrator — per-type rolling confidence recalibration

Common discipline (mirrors liqpool/execution):
  * deterministic fallback when unfit — attaching an untrained model
    is always a no-op, never a crash
  * fit() returns self; predict paths clamp/NaN-guard
  * each module documents the exact data shape it trains on and
    which Stream L event kind / engine artifact produces it
"""
from .regret_model import RegretEstimator
from .detector_trust import DetectorTrustRouter
from .drift_imminent import DriftImminentModel
from .reaction_archetypes import ReactionArchetypeModel
from .bucket_aging import BucketAgingModel
from .cross_asset import CrossAssetTransferMatrix
from .brief_meta_calibrator import BriefConfidenceMetaCalibrator
from .hub import FlywheelHub
from .extractors import (
    bucket_aging_history_from_joined,
    co_occurrences_from_report,
    detector_outcomes_from_report,
    reaction_paths_from_report,
)

__all__ = [
    "RegretEstimator",
    "DetectorTrustRouter",
    "DriftImminentModel",
    "ReactionArchetypeModel",
    "BucketAgingModel",
    "CrossAssetTransferMatrix",
    "BriefConfidenceMetaCalibrator",
    "FlywheelHub",
    "bucket_aging_history_from_joined",
    "co_occurrences_from_report",
    "detector_outcomes_from_report",
    "reaction_paths_from_report",
]
