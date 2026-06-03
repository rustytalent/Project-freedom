"""Time-decay sample weights for model training.

Treating 3-year-old data the same as last month's is wrong when the
underlying regime drifts. Stale samples can drag a model toward
patterns that no longer work. Standard fix (Lopez de Prado AFML
chapter 4): weight each sample by a decay function of its age.

We use exponential decay with a configurable half-life, normalised so
the mean weight stays close to 1.0 — this keeps the LightGBM training
loss roughly comparable across runs with different decay settings.

Usage::

    from liqpool.sample_weights import time_decay_weights
    w = time_decay_weights(
        sample_times=pool_available_at_array,
        reference_time=most_recent_oos_bar_time,
        half_life_days=180.0,
    )
    model.fit(..., sample_weight=w)

When ``half_life_days`` is None or <= 0, returns a uniform array of
1.0s (no decay) — opt-in by setting Config.sample_decay_halflife_days.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def time_decay_weights(sample_times: Sequence[pd.Timestamp],
                       reference_time: Optional[pd.Timestamp] = None,
                       half_life_days: Optional[float] = None,
                       min_weight: float = 0.05) -> np.ndarray:
    """Per-sample weights that decay with sample age.

    Args:
      sample_times: timestamps for each training row. Typically a pool's
                    ``available_at`` (the bar at which the pool became
                    actionable).
      reference_time: the "now" against which ages are measured. Default
                      is max(sample_times), making the most recent sample
                      weight = 1.0.
      half_life_days: weight halves every ``half_life_days`` of age.
                      None or <= 0 returns uniform 1.0s (no decay).
      min_weight: floor so very old samples don't collapse to 0 and
                  fall out of training. Default 0.05.

    Returns:
      np.ndarray of length len(sample_times), in [min_weight, 1.0].
      Normalised so mean weight is ~1.0 (preserves loss-scale parity
      with the no-decay case).
    """
    n = len(sample_times)
    if n == 0:
        return np.zeros(0, dtype=float)
    if half_life_days is None or half_life_days <= 0.0:
        return np.ones(n, dtype=float)

    ts = pd.to_datetime(pd.Series(sample_times))
    if ts.isna().all():
        return np.ones(n, dtype=float)
    if reference_time is None:
        reference_time = ts.max()
    age_seconds = (pd.Timestamp(reference_time) - ts).dt.total_seconds()
    age_days = age_seconds.to_numpy() / 86400.0
    age_days = np.where(np.isnan(age_days), 0.0, age_days)
    age_days = np.maximum(age_days, 0.0)

    decay_constant = np.log(2.0) / float(half_life_days)
    raw = np.exp(-decay_constant * age_days)
    raw = np.maximum(raw, float(min_weight))

    # Normalise so the mean weight is 1.0 — keeps LightGBM loss
    # comparable to the no-decay run.
    mean = raw.mean()
    if mean > 0:
        raw = raw / mean
    return raw.astype(float)
