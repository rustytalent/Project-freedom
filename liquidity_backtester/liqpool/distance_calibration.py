"""Per-distance-bucket isotonic recalibration (Stage C).

Quality, proximity, direction, and reaction models all already apply *global*
isotonic calibration. OOS reporting (`evaluate_timing` and the quality
distance-binned table) shows that calibration error is uneven across distance
buckets — most visibly the 10+ ATR bucket where mean predicted Q is well below
the actual rate, and the 0-1 ATR bucket where proximity is over-confident.

This module adds a small post-hoc layer:

  raw_p (already globally calibrated)  ─┐
                                        ├─► distance_bucket  ─► per-bucket isotonic ─► p
  distance_atr                          ─┘

The mapping is fit ONCE on held-out OOS data (independent of training), keyed on
the distance bucket the existing :func:`liqpool.timing.distance_bucket` already
publishes (0-1, 1-3, 3-5, 5-10, 10+ ATR). Buckets with fewer than `min_bucket_n`
samples fall back to identity, so a sparse bucket can never *introduce* error.

Calibration is *monotone* per bucket (isotonic regression preserves argsort
within the bucket); since the bucket assignment is a pure function of
``distance_atr`` and not of the prediction, the layer also preserves argsort
across the full population *within a constant distance*. (Across buckets it
shifts levels — that is the whole point of the recalibration.)

The layer is small and pickle-friendly: each bucket holds one
``sklearn.isotonic.IsotonicRegression``. Callers that want to keep an old model
bundle working without the layer simply do not call :meth:`DistanceCalibrator.fit`
— :meth:`transform` returns its input unchanged when the calibrator is empty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# Bucket boundaries match `liqpool.timing.distance_bucket` exactly so the
# reporting tables and this calibrator agree on every row's bucket assignment.
_DISTANCE_BUCKETS: tuple[str, ...] = (
    "0-1 ATR", "1-3 ATR", "3-5 ATR", "5-10 ATR", "10+ ATR",
)


def distance_bucket(distance_atr: float) -> str:
    """Same bucketing as :func:`liqpool.timing.distance_bucket`.

    Duplicated here (rather than imported) so this module has no circular
    dependency on ``timing`` — ``timing`` will import this module.
    """
    d = float(distance_atr)
    if d < 1.0:
        return "0-1 ATR"
    if d < 3.0:
        return "1-3 ATR"
    if d < 5.0:
        return "3-5 ATR"
    if d < 10.0:
        return "5-10 ATR"
    return "10+ ATR"


@dataclass
class _BucketCalibStat:
    """Diagnostic record per bucket. Persisted in the model bundle for audit."""
    bucket: str
    n: int
    base_rate: float
    raw_mean: float
    cal_mean: float
    fitted: bool
    skip_reason: str = ""


@dataclass
class DistanceCalibrator:
    """Per-distance-bucket post-hoc isotonic recalibration layer.

    Use:
        calib = DistanceCalibrator()
        calib.fit(raw_p, distance_atr, y_true, min_bucket_n=300)
        p_out = calib.transform(raw_p, distance_atr)

    Both :meth:`fit` and :meth:`transform` accept array-like inputs of equal
    length. ``transform`` on an unfit calibrator returns its input unchanged.
    """

    min_bucket_n: int = 300
    iso_by_bucket: Dict[str, object] = field(default_factory=dict)
    stats: List[_BucketCalibStat] = field(default_factory=list)

    # The buckets used. Held as an attribute so a future config can override
    # without subclassing.
    buckets: tuple[str, ...] = _DISTANCE_BUCKETS

    @property
    def is_fitted(self) -> bool:
        return bool(self.iso_by_bucket)

    def fit(self,
            raw_p: Sequence[float],
            distance_atr: Sequence[float],
            y_true: Sequence[int],
            min_bucket_n: Optional[int] = None) -> "DistanceCalibrator":
        """Fit one isotonic regressor per bucket with enough samples.

        Buckets below the threshold are left as identity (their predictions
        pass through unchanged), so sparse buckets cannot make things worse.
        """
        from sklearn.isotonic import IsotonicRegression

        raw = np.asarray(raw_p, dtype=float)
        dist = np.asarray(distance_atr, dtype=float)
        y = np.asarray(y_true, dtype=float)
        if not (len(raw) == len(dist) == len(y)):
            raise ValueError("raw_p, distance_atr, y_true must have equal length")
        if min_bucket_n is not None:
            self.min_bucket_n = int(min_bucket_n)

        bucket_labels = np.array([distance_bucket(d) for d in dist], dtype=object)
        self.iso_by_bucket = {}
        self.stats = []
        for b in self.buckets:
            mask = bucket_labels == b
            n = int(mask.sum())
            if n == 0:
                self.stats.append(_BucketCalibStat(
                    bucket=b, n=0, base_rate=0.0, raw_mean=0.0,
                    cal_mean=0.0, fitted=False, skip_reason="no rows in bucket",
                ))
                continue
            r_b = raw[mask]
            y_b = y[mask]
            base = float(y_b.mean())
            raw_mean = float(r_b.mean())
            if n < self.min_bucket_n:
                self.stats.append(_BucketCalibStat(
                    bucket=b, n=n, base_rate=base, raw_mean=raw_mean,
                    cal_mean=raw_mean, fitted=False,
                    skip_reason=f"n<{self.min_bucket_n}, fall back to global",
                ))
                continue
            if len(np.unique(y_b)) < 2:
                self.stats.append(_BucketCalibStat(
                    bucket=b, n=n, base_rate=base, raw_mean=raw_mean,
                    cal_mean=raw_mean, fitted=False,
                    skip_reason="only one class present",
                ))
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(r_b, y_b)
            cal_b = iso.transform(r_b)
            self.iso_by_bucket[b] = iso
            self.stats.append(_BucketCalibStat(
                bucket=b, n=n, base_rate=base, raw_mean=raw_mean,
                cal_mean=float(cal_b.mean()), fitted=True,
            ))
        return self

    def transform(self,
                  raw_p: Sequence[float],
                  distance_atr: Sequence[float]) -> np.ndarray:
        """Apply per-bucket isotonic; identity for buckets that were not fit."""
        raw = np.asarray(raw_p, dtype=float).copy()
        if not self.is_fitted or len(raw) == 0:
            return raw
        dist = np.asarray(distance_atr, dtype=float)
        if len(raw) != len(dist):
            raise ValueError("raw_p and distance_atr must have equal length")
        out = raw.copy()
        # Bucketing is a vectorised pure function of dist — apply per bucket.
        for b, iso in self.iso_by_bucket.items():
            mask = np.array([distance_bucket(d) == b for d in dist], dtype=bool)
            if not mask.any():
                continue
            out[mask] = iso.transform(raw[mask])
        return out

    def stats_rows(self) -> List[Dict]:
        """Audit rows for the model report."""
        return [
            {
                "bucket": s.bucket,
                "n": s.n,
                "base_rate": s.base_rate,
                "raw_mean": s.raw_mean,
                "calibrated_mean": s.cal_mean,
                "fitted": s.fitted,
                "skip_reason": s.skip_reason,
            }
            for s in self.stats
        ]
