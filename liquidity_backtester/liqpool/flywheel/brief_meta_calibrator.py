"""M.10 — Brief confidence meta-calibrator.

The brief's confidence buckets ("very_high"/"high"/"moderate"/"low")
are static p-thresholds today. This meta-calibrator closes the loop:
it reads the outcome log's joined prediction/resolution history and
fits a per-prediction-type temperature correction so that tomorrow's
published probabilities reflect HOW MISCALIBRATED the recent window
actually was — recalibrating from yesterday's miss directions, not
just absolute error.

Reuses the Stream N TemperatureScaler per prediction_type. The
correction is rank-preserving (so the watchlist ordering never
changes), bounded (temperature clamped into [0.5, 3.0] so one bad
week can't invert the brief), and fully disclosed (the brief's
confidence_notes can print the applied temperature).

Data shape: the joined outcome-log frame (one row per resolved
prediction):
    prediction_type   — proximity / avoidance / options_strike
    predicted_value   — the published calibrated probability
    outcome_boolean   — what actually happened
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..calibration.temperature import TemperatureScaler

TEMPERATURE_FLOOR = 0.5
TEMPERATURE_CEIL = 3.0
MIN_RESOLVED_PER_TYPE = 30


class BriefConfidenceMetaCalibrator:
    """Per-type rolling temperature recalibration of brief outputs."""

    def __init__(self) -> None:
        self._scalers: Dict[str, TemperatureScaler] = {}

    def fit(self, joined: pd.DataFrame) -> "BriefConfidenceMetaCalibrator":
        """``joined``: the outcome-log join. Only resolved rows train;
        types with fewer than MIN_RESOLVED_PER_TYPE resolved rows are
        skipped (identity transform) — a temperature fit on 12 rows
        is noise, not signal."""
        if joined.empty or "outcome_boolean" not in joined.columns:
            return self
        sub = joined[joined["outcome_boolean"].notna()]
        for ptype, g in sub.groupby("prediction_type"):
            if len(g) < MIN_RESOLVED_PER_TYPE:
                continue
            ts = TemperatureScaler()
            ts.fit(
                g["predicted_value"].astype(float).to_numpy(),
                g["outcome_boolean"].astype(float).to_numpy(),
            )
            # Bound the correction. A catastrophic week must not be
            # able to invert or flatten the brief's probabilities.
            ts.temperature = float(
                min(max(ts.temperature, TEMPERATURE_FLOOR), TEMPERATURE_CEIL)
            )
            self._scalers[str(ptype)] = ts
        return self

    def adjust(self, prediction_type: str, p: float) -> float:
        """Recalibrated probability for tomorrow's brief. Types with
        no fitted scaler pass through unchanged."""
        ts = self._scalers.get(str(prediction_type))
        if ts is None:
            return float(p)
        return float(ts.transform_one(float(p)))

    def applied_temperature(self, prediction_type: str) -> float:
        """For the confidence_notes disclosure line. 1.0 = no
        adjustment in effect."""
        ts = self._scalers.get(str(prediction_type))
        return float(ts.temperature) if ts is not None else 1.0

    @property
    def is_fitted(self) -> bool:
        return bool(self._scalers)

    # -- persistence ---------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            k: {"temperature": v.temperature,
                "n_calibration": v.n_calibration}
            for k, v in self._scalers.items()
        }))

    @classmethod
    def load(cls, path: Path) -> "BriefConfidenceMetaCalibrator":
        blob = json.loads(Path(path).read_text())
        obj = cls()
        for k, v in blob.items():
            ts = TemperatureScaler(temperature=float(v["temperature"]))
            ts.n_calibration = int(v.get("n_calibration", 0))
            obj._scalers[k] = ts
        return obj
