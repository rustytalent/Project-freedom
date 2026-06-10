"""M.6 — Drift-imminent model.

The static drift detector (liqpool/drift.py) fires when a metric
crosses an absolute threshold — by which point the calibration has
already degraded. This model learns the PRECURSOR pattern: given the
trajectory of headline metrics over the last few runs, what is the
probability a drift alert fires within the next ``horizon`` runs?

Training data: the drift-metrics history. Two sources feed it:
  * the shadow log's drift_flag_fired events (Stream L wiring in
    liqpool/drift.record_drift_alerts_to_shadow), and
  * the saved DriftMetrics baselines accumulated across retrains.

Feature engineering on a metric history frame (one row per run,
chronological): current level + 1-run delta + 3-run slope for each
headline metric. Label: did any drift alert fire within the next
``horizon`` rows?

v1 keeps the feature set deliberately small (the history is short —
tens of runs, not thousands) and leans on LightGBM's NaN handling
for missing early-history deltas.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..execution._base import LGBMConfig, LGBMRegressorWrapper

METRICS: List[str] = [
    "oos_broad_respect",
    "mean_overfit_gap",
    "direction_auc",
    "direction_top_quartile_confidence",
]

FEATURE_COLUMNS: List[str] = (
    [f"{m}_level" for m in METRICS]
    + [f"{m}_delta1" for m in METRICS]
    + [f"{m}_slope3" for m in METRICS]
)

FALLBACK_PROB: float = 0.2     # base rate guess when unfit
PROB_FLOOR, PROB_CEIL = 0.02, 0.98


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """``history``: chronological frame with the METRICS columns.
    Emits level/delta/slope features per row (NaN where history is
    too short — LightGBM handles those)."""
    out = pd.DataFrame(index=history.index)
    for m in METRICS:
        s = history.get(m, pd.Series(np.nan, index=history.index)).astype(float)
        out[f"{m}_level"] = s
        out[f"{m}_delta1"] = s.diff(1)
        out[f"{m}_slope3"] = (s - s.shift(3)) / 3.0
    return out[FEATURE_COLUMNS]


def build_labels(history: pd.DataFrame, horizon: int = 3) -> np.ndarray:
    """Binary: does ``drift_fired`` become true within the NEXT
    ``horizon`` rows (strictly after the current row)? Rows whose
    forward window runs off the end get NaN (dropped by the trainer)
    — labelling them 0 would teach the model that recent history is
    always safe."""
    fired = history.get("drift_fired")
    if fired is None:
        return np.full(len(history), np.nan)
    fired = fired.astype(float).to_numpy()
    n = len(fired)
    labels = np.full(n, np.nan)
    for i in range(n):
        end = i + 1 + horizon
        if end > n:
            break
        labels[i] = 1.0 if np.nansum(fired[i + 1:end]) > 0 else 0.0
    return labels


class DriftImminentModel:
    """P(drift alert within `horizon` runs | metric trajectory)."""

    def __init__(self, horizon: int = 3,
                 config: Optional[LGBMConfig] = None) -> None:
        self.horizon = int(horizon)
        self._wrapper = LGBMRegressorWrapper(
            feature_names=FEATURE_COLUMNS,
            fallback=FALLBACK_PROB,
            config=config,
        )

    def fit(self, history: pd.DataFrame) -> "DriftImminentModel":
        if history.empty:
            return self
        X = build_features(history)
        y = build_labels(history, self.horizon)
        mask = np.isfinite(y)
        if mask.sum() == 0:
            return self
        self._wrapper.fit(X.loc[mask].reset_index(drop=True), y[mask])
        return self

    def predict_imminence(self, recent_history: pd.DataFrame) -> float:
        """Score the LATEST row of a chronological history frame."""
        if recent_history.empty:
            return FALLBACK_PROB
        X = build_features(recent_history).tail(1)
        raw = float(self._wrapper.predict(X)[0])
        if not math.isfinite(raw):
            raw = FALLBACK_PROB
        return max(PROB_FLOOR, min(raw, PROB_CEIL))

    @property
    def is_fitted(self) -> bool:
        return self._wrapper.is_fitted

    def save(self, path) -> None:
        self._wrapper.save(path)

    @classmethod
    def load(cls, path, horizon: int = 3) -> "DriftImminentModel":
        obj = cls(horizon=horizon)
        obj._wrapper = LGBMRegressorWrapper.load(path)
        return obj
