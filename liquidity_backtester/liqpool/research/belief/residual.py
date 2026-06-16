"""Deviation-of-deviation residual layer (Belief Engine, Phase 4).

The founder's sharpest insight: the residual (actual − fair premium
change) is itself noisy — every contract deviates a little all the time.
The real signal is when the residual is ABNORMAL relative to its OWN
normal band — the deviation of the deviation.

    residual            = actual_change − fair_change          (fair_response)
    normal_residual_band = rolling median ± IQR of residual     (this module)
    deviation_of_deviation = (residual − median) / robust_sigma

And — critically — this band is SLOT-SPECIFIC: each contract carries its
own rolling band, so a ₹0.70 residual is judged against what residuals
normally look like for THAT strike. (Time-of-day and spread conditioning
are layered on top downstream; here the band adapts within the session.)

Median + IQR (not mean + std) because option residuals are heavy-tailed:
a single fat spike would blow up a standard deviation and desensitize the
z-score. The robust band keeps the abnormality measure honest. The z is
clipped so a flatlined contract (IQR≈0) can't produce an infinite value.

`acceptance_state` formalizes the founder's "₹43/₹44" idea at the
residual level: a residual that is persistently and abnormally positive
on a leg that *should* be decaying = the market is defending that premium
(acceptance failure of the fair value).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .fair_response import FairResponseConfig, fair_response_frame
from .moneyness import MoneynessSlot


# IQR → σ for a normal distribution.
_IQR_TO_SIGMA = 1.349


@dataclass
class ResidualConfig:
    """Knobs for the deviation-of-deviation band."""
    band_window: int = 60        # rolling window for the normal residual band
    anomaly_z_clip: float = 8.0  # cap on |deviation_of_deviation|
    abnormal_threshold: float = 1.5   # |z| beyond this is "abnormal"
    persistence_bars: int = 3    # consecutive abnormal bars for acceptance state
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.band_window < 8:
            raise ValueError("band_window must be >= 8")
        if self.persistence_bars < 1:
            raise ValueError("persistence_bars must be >= 1")


def deviation_of_deviation(residual: pd.Series,
                           cfg: Optional[ResidualConfig] = None) -> pd.DataFrame:
    """Robust z-score of the residual against its own rolling band.

    Returns a frame with columns: ``resid_median``, ``resid_sigma``,
    ``band_lower``, ``band_upper``, ``dod_z`` (deviation of deviation)."""
    cfg = cfg or ResidualConfig()
    r = pd.to_numeric(residual, errors="coerce")
    w = cfg.band_window
    min_p = max(8, w // 2)
    median = r.rolling(w, min_periods=min_p).median()
    q75 = r.rolling(w, min_periods=min_p).quantile(0.75)
    q25 = r.rolling(w, min_periods=min_p).quantile(0.25)
    sigma = ((q75 - q25) / _IQR_TO_SIGMA).clip(lower=cfg.eps)
    z = ((r - median) / sigma).clip(lower=-cfg.anomaly_z_clip,
                                    upper=cfg.anomaly_z_clip).fillna(0.0)
    return pd.DataFrame({
        "resid_median": median.values,
        "resid_sigma": sigma.values,
        "band_lower": (median - sigma).values,
        "band_upper": (median + sigma).values,
        "dod_z": z.values,
    }, index=r.index)


def slot_residual_frame(df: pd.DataFrame,
                        slot: MoneynessSlot,
                        fair_cfg: Optional[FairResponseConfig] = None,
                        resid_cfg: Optional[ResidualConfig] = None,
                        ) -> pd.DataFrame:
    """Full per-slot residual pipeline: fair response → residual →
    deviation-of-deviation → acceptance state.

    ``df`` needs columns ``spot`` and ``mark``. Returns the enriched frame
    with the fair-response columns plus:
      * ``dod_z``         — deviation of deviation (robust z)
      * ``resid_median`` / ``resid_sigma`` / ``band_lower`` / ``band_upper``
      * ``is_abnormal``   — |dod_z| > abnormal_threshold
      * ``acceptance``    — "defended" / "rejected" / "normal" per the
                            founder's ₹43/₹44 logic (see below)
    """
    resid_cfg = resid_cfg or ResidualConfig()
    fr = fair_response_frame(df, slot, fair_cfg)
    dod = deviation_of_deviation(fr["residual"], resid_cfg)
    out = pd.concat([fr, dod], axis=1)

    z = out["dod_z"]
    is_abnormal = z.abs() > resid_cfg.abnormal_threshold
    out["is_abnormal"] = is_abnormal.values

    # Acceptance state — the residual-level ₹43/₹44 read.
    #   "defended": residual persistently ABNORMALLY POSITIVE — the premium
    #     is being held above its fair value (the market refuses the lower
    #     price). For a leg that "should" be decaying, that is hidden
    #     support on that side.
    #   "rejected": residual persistently ABNORMALLY NEGATIVE — the premium
    #     is being pushed below fair (the market refuses the higher price).
    pos_abnormal = (z > resid_cfg.abnormal_threshold)
    neg_abnormal = (z < -resid_cfg.abnormal_threshold)
    pb = resid_cfg.persistence_bars
    pos_run = pos_abnormal.rolling(pb, min_periods=pb).sum()
    neg_run = neg_abnormal.rolling(pb, min_periods=pb).sum()
    acceptance = np.full(len(out), "normal", dtype=object)
    acceptance = np.where(pos_run >= pb, "defended", acceptance)
    acceptance = np.where(neg_run >= pb, "rejected", acceptance)
    out["acceptance"] = acceptance
    return out
