"""M.8 — Bucket-aging model.

Each (prediction_type, confidence_bucket) cell of the public
calibration table degrades at its own rate as the market drifts away
from the regime the bucket was fit in. The current pipeline treats
bucket calibration as static between retrains; this model learns the
AGING CURVE — expected |calibration_error| as a function of bucket
age — so the brief's confidence notes can pre-emptively widen or
demote stale buckets instead of waiting for the drift detector.

Training data: the calibration-by-bucket history, one row per
(trading_date, prediction_type, confidence_bucket):

    age_days        — days since the bucket's model was (re)fit
    n_resolved      — resolved predictions in the cell that day
    abs_calib_error — |mean_predicted_p - hit_rate| that day

v1 model: parametric — a robust linear fit of abs_calib_error on
log1p(age_days), per prediction_type. Tiny data (hundreds of rows)
makes trees overkill; the parametric slope is also directly
interpretable ("error grows ~0.8% per doubling of age"), which is
exactly what the confidence-notes copy needs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

FALLBACK_ERROR: float = 0.05


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust slope/intercept via the Theil-Sen median-of-slopes
    estimator. Resistant to the outlier days a single bad session
    creates in calibration history. O(n^2) pairs — fine for the
    hundreds-of-rows scale this model lives at."""
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if abs(dx) > 1e-12:
                slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return 0.0, float(np.median(y)) if n else FALLBACK_ERROR
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


class BucketAgingModel:
    """Aging curve per prediction_type: E|calib_error| ~ f(age)."""

    def __init__(self) -> None:
        # prediction_type -> (slope, intercept) on log1p(age_days)
        self._curves: Dict[str, tuple[float, float]] = {}

    def fit(self, history: pd.DataFrame) -> "BucketAgingModel":
        """``history`` columns: prediction_type, age_days,
        abs_calib_error (n_resolved optional, used as a weight filter
        — cells with n_resolved < 10 are too noisy to teach aging)."""
        if history.empty:
            return self
        h = history.copy()
        if "n_resolved" in h.columns:
            h = h[h["n_resolved"].astype(float) >= 10]
        for ptype, g in h.groupby("prediction_type"):
            x = np.log1p(g["age_days"].astype(float).to_numpy())
            y = g["abs_calib_error"].astype(float).to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                continue
            self._curves[str(ptype)] = _theil_sen(x[mask], y[mask])
        return self

    def expected_error(self, prediction_type: str, age_days: float) -> float:
        """Expected |calibration error| at the given age. Unknown
        types / unfit model fall back to the audit's working
        tolerance midpoint."""
        curve = self._curves.get(str(prediction_type))
        if curve is None:
            return FALLBACK_ERROR
        slope, intercept = curve
        pred = intercept + slope * math.log1p(max(0.0, float(age_days)))
        return float(min(max(pred, 0.0), 0.5))

    def staleness_multiplier(self, prediction_type: str,
                             age_days: float,
                             fresh_age_days: float = 5.0) -> float:
        """How much worse is the bucket now vs freshly fit? >= 1.0.
        The confidence-notes generator can use this to demote stale
        buckets ('moderate' -> 'low') when the multiplier crosses a
        threshold."""
        fresh = self.expected_error(prediction_type, fresh_age_days)
        now = self.expected_error(prediction_type, age_days)
        if fresh <= 1e-9:
            return 1.0
        return float(max(1.0, now / fresh))

    @property
    def is_fitted(self) -> bool:
        return bool(self._curves)

    # -- persistence ---------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {k: list(v) for k, v in self._curves.items()}
        ))

    @classmethod
    def load(cls, path: Path) -> "BucketAgingModel":
        obj = cls()
        blob = json.loads(Path(path).read_text())
        obj._curves = {k: tuple(v) for k, v in blob.items()}
        return obj
