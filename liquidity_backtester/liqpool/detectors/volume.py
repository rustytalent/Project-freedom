"""Volume-aware detectors — institutional footprint signals.

Two new ``LevelCandidate`` sources that exploit the volume column the
existing fractal-style detectors ignore:

  * ``volume_weighted_swings`` — the plain ``swing_candidates``
    detector emits every fractal at equal strength. Empirically, the
    swings worth defending are the ones formed on high-volume bars
    (the institutional footprint). This detector emits an ADDITIONAL
    candidate at the same swing price but with boosted strength when
    the swing bar's volume exceeded ``vw_swing_multiplier`` x the
    rolling mean volume over the prior ``vw_swing_window`` bars.

  * ``cumulative_delta_divergences`` — a tick-data proxy. Real delta
    (signed-volume from the tick stream) is unavailable in bar data,
    but ``signed_volume = volume * sign(close - open)`` (close-vs-open
    direction) approximates it well enough for divergence detection.
    A divergence occurs when price prints a NEW SWING EXTREME but the
    rolling cumulative signed-volume does NOT confirm: e.g. price
    makes a higher high but cum_delta makes a lower high. The new
    swing extreme becomes a pool candidate at boosted strength — the
    classical "exhaustion" signature, marking levels that institutions
    are quietly distributing into.

Both detectors are causal: ``known_at`` matches the swing detector's
confirmation rule (swing bar + ``swing_right`` future bars + one
period for the close), since divergence cannot be confirmed until the
swing itself is.

Configuration knobs (added to ``DetectionParams`` via ``getattr``
defaults so older Configs work unchanged):

  vw_swing_window         default 30    — bars used for rolling-mean volume
  vw_swing_multiplier     default 1.8   — swing volume must exceed this x mean
  vw_swing_strength_cap   default 3.0   — clamp on emitted strength
  cum_delta_window        default 30    — bars in the cumulative-delta window
  cum_delta_lookback      default 50    — bars in which to find a prior swing
                                          for divergence comparison
  cum_delta_strength_cap  default 3.0   — clamp on emitted strength
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..config import DetectionParams
from ..features import LevelCandidate, _infer_tf_period, detect_swings
from ..indicators import atr


def _volume_params(params: DetectionParams) -> dict:
    return {
        "vw_window":            int(getattr(params, "vw_swing_window", 30)),
        "vw_multiplier":        float(getattr(params,
                                              "vw_swing_multiplier", 1.8)),
        "vw_cap":               float(getattr(params,
                                              "vw_swing_strength_cap", 3.0)),
        "cd_window":            int(getattr(params, "cum_delta_window", 30)),
        "cd_lookback":          int(getattr(params, "cum_delta_lookback", 50)),
        "cd_cap":               float(getattr(params,
                                              "cum_delta_strength_cap", 3.0)),
    }


def _rolling_mean_volume(vol: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean of volume — uses ONLY past bars.

    Returns an array of the same length as ``vol`` where index i holds
    the mean of vol[max(0, i-window):i] (strictly the prior window,
    NOT including i itself, so a high-volume swing isn't compared to
    its own inflated mean).
    """
    n = len(vol)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    cs = np.cumsum(vol)
    for i in range(n):
        lo = max(0, i - window)
        hi = i
        if hi <= lo:
            out[i] = 1e-9
            continue
        total = cs[hi - 1] - (cs[lo - 1] if lo > 0 else 0.0)
        out[i] = total / max(hi - lo, 1)
    out[out <= 0] = 1e-9
    return out


def volume_weighted_swings(df: pd.DataFrame, params: DetectionParams,
                            tf: str = "base") -> List[LevelCandidate]:
    """Emit a boosted-strength LevelCandidate at each high-volume swing.

    The plain ``swing_candidates`` already emits a base candidate at
    every swing; this detector ADDS a parallel candidate at the same
    price for the institutional subset (swing volume exceeds the
    multiplier x rolling-mean volume of the prior window).

    The pool builder merges candidates within ``merge_atr`` so the two
    don't double-count price-wise, but the higher-strength VW_SWING
    contributor lifts the pool's score for the institutional set.

    ``known_at`` matches the plain swing detector's rule: confirmation
    happens at bar (i + swing_right), reported as the close of that
    bar (i + swing_right + period).
    """
    if "volume" not in df.columns:
        return []
    sh, sl = detect_swings(df, params.swing_left, params.swing_right)
    a_arr = atr(df, params.atr_period).bfill().values
    period = _infer_tf_period(df)
    right = params.swing_right
    ts = df.index
    h_arr = df["high"].values
    l_arr = df["low"].values
    vol = df["volume"].values.astype(float)
    hw = params.pool_halfwidth_atr

    cfg = _volume_params(params)
    window = cfg["vw_window"]
    multiplier = cfg["vw_multiplier"]
    cap = cfg["vw_cap"]

    mean_vol = _rolling_mean_volume(vol, window)
    out: List[LevelCandidate] = []

    for i in np.where(sh)[0]:
        if i + right >= len(df):
            continue
        if mean_vol[i] <= 0:
            continue
        ratio = float(vol[i]) / float(mean_vol[i])
        if ratio < multiplier:
            continue
        known = ts[i + right] + period
        strength = min(cap, 1.0 + (ratio - multiplier) * 0.5 + 0.5)
        out.append(LevelCandidate(
            side="high", price=float(h_arr[i]), ts=ts[i],
            source=f"VW_SWING_H@{tf}", known_at=known,
            strength=float(strength),
            half_width=hw * float(a_arr[i]), tf=tf,
            meta={"volume_ratio": ratio,
                  "swing_volume": float(vol[i]),
                  "mean_volume": float(mean_vol[i])},
        ))

    for i in np.where(sl)[0]:
        if i + right >= len(df):
            continue
        if mean_vol[i] <= 0:
            continue
        ratio = float(vol[i]) / float(mean_vol[i])
        if ratio < multiplier:
            continue
        known = ts[i + right] + period
        strength = min(cap, 1.0 + (ratio - multiplier) * 0.5 + 0.5)
        out.append(LevelCandidate(
            side="low", price=float(l_arr[i]), ts=ts[i],
            source=f"VW_SWING_L@{tf}", known_at=known,
            strength=float(strength),
            half_width=hw * float(a_arr[i]), tf=tf,
            meta={"volume_ratio": ratio,
                  "swing_volume": float(vol[i]),
                  "mean_volume": float(mean_vol[i])},
        ))

    return out


def cumulative_delta_divergences(df: pd.DataFrame,
                                  params: DetectionParams,
                                  tf: str = "base") -> List[LevelCandidate]:
    """Price-vs-cumulative-delta divergence at confirmed swings.

    Per-bar signed-volume proxy:
      ``sv[i] = volume[i] * sign(close[i] - open[i])``
    Rolling cumulative over ``cum_delta_window``:
      ``cd[i] = sum(sv[i-window+1 : i+1])``

    Bearish divergence at a swing HIGH at bar ``i``: there exists a
    prior swing high at bar ``j`` (within ``cum_delta_lookback``) where
      ``high[i] > high[j]`` (new price high)
    BUT
      ``cd[i] < cd[j]``    (lower cumulative-delta high)
    The new high becomes a pool candidate at boosted strength —
    institutional distribution is leaking through.

    Mirror for swing LOWs (new lower low + higher cum-delta low =
    bullish divergence, a demand pool).

    ``known_at`` is the close of the swing-confirmation bar
    ``i + swing_right + period``, same as the plain swing detector.
    """
    if "volume" not in df.columns:
        return []
    sh, sl = detect_swings(df, params.swing_left, params.swing_right)
    a_arr = atr(df, params.atr_period).bfill().values
    period = _infer_tf_period(df)
    right = params.swing_right
    ts = df.index
    h_arr = df["high"].values
    l_arr = df["low"].values
    o_arr = df["open"].values
    c_arr = df["close"].values
    vol = df["volume"].values.astype(float)
    hw = params.pool_halfwidth_atr

    cfg = _volume_params(params)
    window = cfg["cd_window"]
    lookback = cfg["cd_lookback"]
    cap = cfg["cd_cap"]

    direction = np.sign(c_arr - o_arr)
    sv = vol * direction
    # Rolling cumulative delta over a window of `window` bars.
    n = len(df)
    cd = np.zeros(n, dtype=float)
    cs = np.cumsum(sv)
    for i in range(n):
        lo = max(0, i - window + 1)
        cd[i] = cs[i] - (cs[lo - 1] if lo > 0 else 0.0)

    sh_idxs = list(np.where(sh)[0])
    sl_idxs = list(np.where(sl)[0])

    out: List[LevelCandidate] = []

    # ---- Bearish divergence at swing HIGH -----------------------------
    for k, i in enumerate(sh_idxs):
        if i + right >= n:
            continue
        # Find prior swing high within lookback bars.
        best_j = None
        for j in sh_idxs[:k][::-1]:
            if i - j > lookback:
                break
            if h_arr[i] > h_arr[j] and cd[i] < cd[j]:
                best_j = j
                break
        if best_j is None:
            continue
        known = ts[i + right] + period
        delta_gap = float(cd[best_j] - cd[i])
        norm = max(abs(cd[best_j]), abs(cd[i]), 1.0)
        strength = min(cap, 1.4 + min(2.0, delta_gap / norm))
        out.append(LevelCandidate(
            side="high", price=float(h_arr[i]), ts=ts[i],
            source=f"CD_DIV_H@{tf}", known_at=known,
            strength=float(strength),
            half_width=hw * float(a_arr[i]), tf=tf,
            meta={"prior_swing_idx": int(best_j),
                  "prior_swing_price": float(h_arr[best_j]),
                  "cum_delta_current": float(cd[i]),
                  "cum_delta_prior": float(cd[best_j])},
        ))

    # ---- Bullish divergence at swing LOW ------------------------------
    for k, i in enumerate(sl_idxs):
        if i + right >= n:
            continue
        best_j = None
        for j in sl_idxs[:k][::-1]:
            if i - j > lookback:
                break
            if l_arr[i] < l_arr[j] and cd[i] > cd[j]:
                best_j = j
                break
        if best_j is None:
            continue
        known = ts[i + right] + period
        delta_gap = float(cd[i] - cd[best_j])
        norm = max(abs(cd[best_j]), abs(cd[i]), 1.0)
        strength = min(cap, 1.4 + min(2.0, delta_gap / norm))
        out.append(LevelCandidate(
            side="low", price=float(l_arr[i]), ts=ts[i],
            source=f"CD_DIV_L@{tf}", known_at=known,
            strength=float(strength),
            half_width=hw * float(a_arr[i]), tf=tf,
            meta={"prior_swing_idx": int(best_j),
                  "prior_swing_price": float(l_arr[best_j]),
                  "cum_delta_current": float(cd[i]),
                  "cum_delta_prior": float(cd[best_j])},
        ))

    return out
