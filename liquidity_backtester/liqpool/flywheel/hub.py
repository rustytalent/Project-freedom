"""FlywheelHub — the circulatory system.

One object that carries every trained flywheel model to its
consumers. The consumers (brief generator, options executor, drift
monitor, pool scorer) each accept an OPTIONAL hub; an absent or
partially-trained hub degrades to today's behaviour at every site.
This is the primitive form of the roadmap's T2.3 cross-organ
communication protocol: organs don't import each other — they all
plug into the hub.

    train side                         serve side
    ----------                         ----------
    scripts/train_flywheel.py          generate_brief(flywheel_hub=...)
      extractors -> fit_from_artifacts   pre_trade_decision(regret advisory)
      hub.save(dir)                      check_drift imminence
                                         trust_weighted_score()

Persistence: one directory, one file per organ. Missing files load
as unfit models (safe fallbacks built into each).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .brief_meta_calibrator import BriefConfidenceMetaCalibrator
from .bucket_aging import BucketAgingModel
from .cross_asset import CrossAssetTransferMatrix
from .detector_trust import DetectorTrustRouter
from .drift_imminent import DriftImminentModel
from .reaction_archetypes import ReactionArchetypeModel
from .regret_model import RegretEstimator


class FlywheelHub:
    """Holds the seven flywheel organs; all optional, all safe-unfit."""

    def __init__(self) -> None:
        self.regret = RegretEstimator()
        self.trust = DetectorTrustRouter()
        self.drift_imminent = DriftImminentModel()
        self.archetypes = ReactionArchetypeModel()
        self.aging = BucketAgingModel()
        self.transfer = CrossAssetTransferMatrix()
        self.meta_calibrator = BriefConfidenceMetaCalibrator()

    # -- training ------------------------------------------------------

    def fit_from_artifacts(
        self,
        report: Any = None,
        shadow_joined: Optional[pd.DataFrame] = None,
        outcome_joined: Optional[pd.DataFrame] = None,
        drift_history: Optional[pd.DataFrame] = None,
        bundle_fit_date: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Feed every organ whatever food is available. Returns
        {organ: is_fitted} so the train job logs exactly which organs
        starved (insufficient data) vs ate."""
        from .extractors import (
            bucket_aging_history_from_joined,
            co_occurrences_from_report,
            detector_outcomes_from_report,
            reaction_paths_from_report,
        )
        if report is not None:
            try:
                outcomes = detector_outcomes_from_report(report)
                if not outcomes.empty:
                    self.trust.fit(outcomes)
            except Exception:
                pass
            try:
                paths, sides, ctx = reaction_paths_from_report(report)
                if len(paths):
                    labels = self.archetypes.fit_clusters(paths, sides)
                    if (labels >= 0).any() and not ctx.empty:
                        self.archetypes.fit_classifier(ctx, labels)
            except Exception:
                pass
            try:
                co = co_occurrences_from_report(report)
                if not co.empty:
                    self.transfer.fit(co)
            except Exception:
                pass
        if shadow_joined is not None and not shadow_joined.empty:
            try:
                self.regret.fit(shadow_joined)
            except Exception:
                pass
        if outcome_joined is not None and not outcome_joined.empty:
            try:
                self.meta_calibrator.fit(outcome_joined)
            except Exception:
                pass
            if bundle_fit_date:
                try:
                    hist = bucket_aging_history_from_joined(
                        outcome_joined, bundle_fit_date)
                    if not hist.empty:
                        self.aging.fit(hist)
                except Exception:
                    pass
        if drift_history is not None and not drift_history.empty:
            try:
                self.drift_imminent.fit(drift_history)
            except Exception:
                pass
        return self.fitted_map()

    def fitted_map(self) -> Dict[str, bool]:
        return {
            "regret": self.regret.is_fitted,
            "trust": self.trust.is_fitted,
            "drift_imminent": self.drift_imminent.is_fitted,
            "archetypes": self.archetypes.is_fitted,
            "aging": self.aging.is_fitted,
            "transfer": self.transfer.is_fitted,
            "meta_calibrator": self.meta_calibrator.is_fitted,
        }

    # -- serving helpers -------------------------------------------------

    def adjust_probability(self, prediction_type: str, p: float) -> float:
        """Meta-calibrated probability (identity when unfit)."""
        try:
            return self.meta_calibrator.adjust(prediction_type, p)
        except Exception:
            return float(p)

    def trust_weighted_score(self, base_score: float, factor: str,
                             regime: str) -> float:
        """Pool score x detector trust, rescaled so a neutral-trust
        (0.5) bucket leaves the score unchanged. Trust 0.7 boosts the
        score 1.4x; trust 0.3 cuts it to 0.6x. Unfit router = no-op."""
        if not self.trust.is_fitted:
            return float(base_score)
        try:
            return float(base_score) * (self.trust.trust(factor, regime) / 0.5)
        except Exception:
            return float(base_score)

    def staleness_note(self, prediction_type: str,
                       age_days: float) -> Optional[str]:
        """Confidence-notes line when a bucket has aged badly; None
        when fresh or the aging model is unfit."""
        if not self.aging.is_fitted:
            return None
        try:
            mult = self.aging.staleness_multiplier(prediction_type, age_days)
        except Exception:
            return None
        if mult < 1.25:
            return None
        return (
            f"{prediction_type}: calibration is ~{mult:.1f}x its "
            f"fresh-bundle error at age {int(age_days)}d - treat "
            f"confidence buckets as one notch lower"
        )

    # -- persistence -------------------------------------------------------

    def save(self, root: Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.regret.save(root / "regret.txt")
        self.trust.save(root / "trust.json")
        self.drift_imminent.save(root / "drift_imminent.txt")
        self.aging.save(root / "aging.json")
        self.transfer.save(root / "transfer.json")
        self.meta_calibrator.save(root / "meta_calibrator.json")
        # archetypes: kmeans persistence deferred — the clusterer
        # refits cheaply from the bundle at train time. v2: joblib.

    @classmethod
    def load(cls, root: Path) -> "FlywheelHub":
        """Load whatever exists; everything missing stays safe-unfit."""
        root = Path(root)
        hub = cls()
        loaders = [
            ("regret", lambda: RegretEstimator.load(root / "regret.txt"),
             root / "regret.meta.json"),
            ("trust", lambda: DetectorTrustRouter.load(root / "trust.json"),
             root / "trust.json"),
            ("drift_imminent",
             lambda: DriftImminentModel.load(root / "drift_imminent.txt"),
             root / "drift_imminent.meta.json"),
            ("aging", lambda: BucketAgingModel.load(root / "aging.json"),
             root / "aging.json"),
            ("transfer",
             lambda: CrossAssetTransferMatrix.load(root / "transfer.json"),
             root / "transfer.json"),
            ("meta_calibrator",
             lambda: BriefConfidenceMetaCalibrator.load(
                 root / "meta_calibrator.json"),
             root / "meta_calibrator.json"),
        ]
        for attr, loader, marker in loaders:
            try:
                if marker.exists():
                    setattr(hub, attr, loader())
            except Exception:
                continue
        return hub
