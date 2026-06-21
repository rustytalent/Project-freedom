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
    """Knobs for the deviation-of-deviation band.

    The advanced fields below are ALL opt-in (default = off → exactly
    reproduces the original Phase-4 behavior). They were added in the
    Sun 2026-06-22 upgrade pass after the founder's audit flagged the
    original primitives:
      1. Single fixed band window (no time-of-day awareness, no
         multi-horizon view).
      2. Binary acceptance label with no confidence carry.
    """
    band_window: int = 60        # rolling window for the normal residual band
    anomaly_z_clip: float = 8.0  # cap on |deviation_of_deviation|
    abnormal_threshold: float = 1.5   # |z| beyond this is "abnormal"
    persistence_bars: int = 3    # consecutive abnormal bars for acceptance state
    eps: float = 1e-9

    # ── Advanced (opt-in) ─────────────────────────────────────────
    # Multi-horizon view: when enabled, compute three bands
    # (tactical / current / session) and report cross-horizon agreement.
    enable_multi_horizon: bool = False
    horizon_tactical: int = 30
    horizon_current: int = 60     # the default band; matches band_window
    horizon_session: int = 240

    # Time-of-day band scaling: when enabled, multiplies the IQR-derived
    # sigma by a per-bar scale factor (passed in via slot_residual_frame
    # arg `time_of_day_scale`). Wider bands during opens/lunch/close
    # reduce false abnormals in known-noisy windows.
    enable_time_of_day_scaling: bool = False
    time_of_day_scale_floor: float = 0.50
    time_of_day_scale_cap: float = 3.0

    # Acceptance confidence: a continuous [0,1] alongside the discrete
    # label, blending recent-bar fraction × magnitude × recency. The
    # discrete label is preserved unchanged for downstream readers.
    enable_acceptance_confidence: bool = False
    acceptance_confidence_window: int = 8
    acceptance_confidence_recency_weight: float = 1.6

    def __post_init__(self) -> None:
        if self.band_window < 8:
            raise ValueError("band_window must be >= 8")
        if self.persistence_bars < 1:
            raise ValueError("persistence_bars must be >= 1")
        if self.enable_multi_horizon:
            if self.horizon_tactical < 8 or self.horizon_session < 30:
                raise ValueError(
                    "multi-horizon: tactical>=8 and session>=30 required")
            if not (self.horizon_tactical
                     <= self.horizon_current
                     <= self.horizon_session):
                raise ValueError(
                    "horizons must satisfy tactical<=current<=session")
        if self.enable_time_of_day_scaling:
            if not (0 < self.time_of_day_scale_floor
                     < self.time_of_day_scale_cap):
                raise ValueError(
                    "time_of_day_scale: 0 < floor < cap required")
        if self.enable_acceptance_confidence:
            if self.acceptance_confidence_window < 2:
                raise ValueError(
                    "acceptance_confidence_window must be >= 2")


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
                        time_of_day_scale: Optional[pd.Series] = None,
                        ) -> pd.DataFrame:
    """Full per-slot residual pipeline: fair response → residual →
    deviation-of-deviation → acceptance state.

    ``df`` needs columns ``spot`` and ``mark``. ``time_of_day_scale`` is
    an optional Series (aligned with ``df.index``) carrying the per-bar
    band scale factor — used only when ``resid_cfg.enable_time_of_day_scaling``
    is True; values are clipped to the [floor, cap] band defined in cfg.

    Always returns the legacy columns (unchanged shape/semantics):
      * ``dod_z`` — deviation of deviation (robust z)
      * ``resid_median`` / ``resid_sigma`` / ``band_lower`` / ``band_upper``
      * ``is_abnormal`` — |dod_z| > abnormal_threshold
      * ``acceptance`` — "defended" / "rejected" / "normal"

    When the advanced flags are enabled, ADDITIONAL columns appear:
      * ``dod_z_tactical`` / ``dod_z_current`` / ``dod_z_session`` and
        ``horizon_agreement`` (0..1): cross-horizon vote on abnormality.
      * ``acceptance_confidence`` (0..1): continuous strength of the
        discrete acceptance label.
      * ``tod_scale_applied`` (float): the realized scale factor used
        (only when time-of-day scaling is on).

    All advanced columns have safe defaults when their flag is off, so
    downstream code that doesn't look for them is unaffected.
    """
    resid_cfg = resid_cfg or ResidualConfig()
    fr = fair_response_frame(df, slot, fair_cfg)

    # The PRIMARY band — the original column shape downstream depends on.
    # When time-of-day scaling is on, we widen sigma per bar (which softens
    # `dod_z` and reduces false-abnormal classifications during noisy
    # session phases). When off, this path is bit-identical to the original.
    if (resid_cfg.enable_time_of_day_scaling
            and time_of_day_scale is not None):
        dod = _deviation_of_deviation_with_scale(
            fr["residual"], resid_cfg, time_of_day_scale)
    else:
        dod = deviation_of_deviation(fr["residual"], resid_cfg)
    out = pd.concat([fr, dod], axis=1)

    z = out["dod_z"]
    is_abnormal = z.abs() > resid_cfg.abnormal_threshold
    out["is_abnormal"] = is_abnormal.values

    # Acceptance state — the residual-level ₹43/₹44 read.
    pos_abnormal = (z > resid_cfg.abnormal_threshold)
    neg_abnormal = (z < -resid_cfg.abnormal_threshold)
    pb = resid_cfg.persistence_bars
    pos_run = pos_abnormal.rolling(pb, min_periods=pb).sum()
    neg_run = neg_abnormal.rolling(pb, min_periods=pb).sum()
    acceptance = np.full(len(out), "normal", dtype=object)
    acceptance = np.where(pos_run >= pb, "defended", acceptance)
    acceptance = np.where(neg_run >= pb, "rejected", acceptance)
    out["acceptance"] = acceptance

    # ── Advanced (opt-in) layers ──────────────────────────────────

    if resid_cfg.enable_time_of_day_scaling and time_of_day_scale is not None:
        scale = _clipped_scale(time_of_day_scale, resid_cfg)
        out["tod_scale_applied"] = scale.reindex(out.index).values

    if resid_cfg.enable_multi_horizon:
        # Compute the three horizons against the SAME residual stream.
        residual = fr["residual"]
        tactical = _dod_z_at_window(residual, resid_cfg.horizon_tactical,
                                      resid_cfg)
        current = _dod_z_at_window(residual, resid_cfg.horizon_current,
                                     resid_cfg)
        session = _dod_z_at_window(residual, resid_cfg.horizon_session,
                                     resid_cfg)
        out["dod_z_tactical"] = tactical.values
        out["dod_z_current"] = current.values
        out["dod_z_session"] = session.values
        # Horizon agreement: fraction of the three bands that agree on
        # abnormality AND direction at this bar.
        thr = resid_cfg.abnormal_threshold
        votes = pd.DataFrame({
            "t": tactical.abs() > thr,
            "c": current.abs() > thr,
            "s": session.abs() > thr,
        })
        # Direction-aware vote: count only when sign matches the current.
        sign_c = np.sign(current).replace(0, np.nan)
        sign_match = pd.DataFrame({
            "t": np.sign(tactical) == sign_c,
            "c": pd.Series(True, index=current.index),
            "s": np.sign(session) == sign_c,
        }).fillna(False)
        directional_votes = (votes & sign_match).sum(axis=1)
        out["horizon_agreement"] = (directional_votes / 3.0).values

    if resid_cfg.enable_acceptance_confidence:
        out["acceptance_confidence"] = _acceptance_confidence(
            z=z, abnormal_threshold=resid_cfg.abnormal_threshold,
            window=resid_cfg.acceptance_confidence_window,
            recency_weight=resid_cfg.acceptance_confidence_recency_weight,
        ).values

    return out


# ── Internals supporting the advanced layers ─────────────────────


def _deviation_of_deviation_with_scale(
        residual: pd.Series,
        cfg: ResidualConfig,
        tod_scale: pd.Series,
) -> pd.DataFrame:
    """Same as ``deviation_of_deviation`` but multiplies the IQR-derived
    sigma by the per-bar time-of-day scale (clipped to [floor, cap])."""
    r = pd.to_numeric(residual, errors="coerce")
    w = cfg.band_window
    min_p = max(8, w // 2)
    median = r.rolling(w, min_periods=min_p).median()
    q75 = r.rolling(w, min_periods=min_p).quantile(0.75)
    q25 = r.rolling(w, min_periods=min_p).quantile(0.25)
    base_sigma = ((q75 - q25) / _IQR_TO_SIGMA).clip(lower=cfg.eps)
    scale = _clipped_scale(tod_scale, cfg).reindex(r.index).fillna(1.0)
    sigma = (base_sigma * scale).clip(lower=cfg.eps)
    z = ((r - median) / sigma).clip(
        lower=-cfg.anomaly_z_clip, upper=cfg.anomaly_z_clip).fillna(0.0)
    return pd.DataFrame({
        "resid_median": median.values,
        "resid_sigma": sigma.values,
        "band_lower": (median - sigma).values,
        "band_upper": (median + sigma).values,
        "dod_z": z.values,
    }, index=r.index)


def _clipped_scale(scale: pd.Series, cfg: ResidualConfig) -> pd.Series:
    s = pd.to_numeric(scale, errors="coerce").fillna(1.0)
    return s.clip(lower=cfg.time_of_day_scale_floor,
                   upper=cfg.time_of_day_scale_cap)


def _dod_z_at_window(residual: pd.Series, window: int,
                       cfg: ResidualConfig) -> pd.Series:
    """Compute dod_z at an arbitrary window; used by multi-horizon view."""
    r = pd.to_numeric(residual, errors="coerce")
    min_p = max(8, window // 2)
    median = r.rolling(window, min_periods=min_p).median()
    q75 = r.rolling(window, min_periods=min_p).quantile(0.75)
    q25 = r.rolling(window, min_periods=min_p).quantile(0.25)
    sigma = ((q75 - q25) / _IQR_TO_SIGMA).clip(lower=cfg.eps)
    z = ((r - median) / sigma).clip(
        lower=-cfg.anomaly_z_clip, upper=cfg.anomaly_z_clip).fillna(0.0)
    return z


def _acceptance_confidence(z: pd.Series, abnormal_threshold: float,
                              window: int, recency_weight: float) -> pd.Series:
    """Continuous confidence in [0,1] alongside the discrete label.

    Blends three pieces:
      * fraction of the last `window` bars over the abnormal threshold
      * average MAGNITUDE above the threshold when abnormal (so a stack
        of strong abnormals scores higher than a stack of borderline)
      * recency weight that gives the latest bars more pull, so the
        score reacts within the window rather than averaging away signal.
    """
    thr = float(abnormal_threshold)
    # Per-bar excess over the threshold (0 when below); use signed magnitude.
    abnormal = (z.abs() - thr).clip(lower=0.0)
    is_abn = (abnormal > 0.0).astype(float)

    n = int(window)
    if n < 2:
        return pd.Series(0.0, index=z.index)

    # Recency-weighted rolling fraction + magnitude.
    weights = np.array([recency_weight ** i for i in range(n)])
    weights = weights[::-1]
    weights = weights / weights.sum()

    def _w_apply(s: pd.Series) -> pd.Series:
        # Causal rolling weighted mean.
        # NOTE: rolling.apply with raw=True is fast enough here.
        return s.rolling(n, min_periods=2).apply(
            lambda arr: float(np.dot(arr[-len(weights):],
                                       weights[-len(arr):]
                                       / weights[-len(arr):].sum())),
            raw=True,
        )

    frac = _w_apply(is_abn)
    avg_mag = _w_apply(abnormal).fillna(0.0)
    # Squash magnitude so a single huge spike doesn't pin confidence at 1.
    mag_factor = (avg_mag / (avg_mag + 1.0)).fillna(0.0)
    confidence = (0.55 * frac + 0.45 * mag_factor).fillna(0.0)
    return confidence.clip(lower=0.0, upper=1.0)
