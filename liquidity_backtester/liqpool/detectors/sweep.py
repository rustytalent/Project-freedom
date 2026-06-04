"""Liquidity-sweep + stop-run-reclaim detectors.

Two new ``LevelCandidate`` sources that capture the canonical
"institutional manipulation" patterns the audit + the user's brain-dump
flagged as missing from the original detector library:

  * ``liquidity_sweeps`` — a swing high/low is pierced by a wick whose
    extent exceeds ``sweep_min_atr`` × ATR, AND price closes back inside
    within ``reclaim_window`` bars. This is the classic "stop hunt" —
    the previous swing's stop cluster is harvested, then price refuses
    to follow through. The level the sweep originated FROM becomes a
    high-value pool candidate (the institutions who triggered the hunt
    are now defending that side).

  * ``stop_run_reclaims`` — a stronger variant. Price not only pierces
    a swing extreme, it CLOSES beyond it on at least one bar
    (registering as a "break" to anyone watching closing prices), and
    THEN closes back inside the original range within
    ``reclaim_window`` bars. This is the canonical SMC sweep-and-go
    setup. It produces a LevelCandidate at the swing extreme that was
    swept (the new pool), AND records the reclaim bar as ``known_at``.

Both detectors are causal: ``known_at`` is the timestamp at the close
of the bar where the reclaim is CONFIRMED (not the swing-formation bar
which is in the past). A trader watching live can only act on these
patterns at the reclaim-confirmation bar, never earlier.

Configuration knobs (added to ``DetectionParams`` in the consuming
``Config``; sensible defaults baked in here so the detectors run on
older Configs via ``getattr``):

  sweep_min_atr             default 0.20  — pierce must exceed this ATR
                                            fraction beyond the swing
  sweep_reclaim_window      default 12    — bars in which reclaim must occur
  sweep_strength_cap        default 3.0   — clamp on emitted strength
  stop_run_close_atr        default 0.15  — close-beyond minimum for the
                                            stop_run variant (must close
                                            past the swing, not just wick)
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..config import DetectionParams
from ..features import LevelCandidate, _infer_tf_period, detect_swings
from ..indicators import atr


def _sweep_params(params: DetectionParams) -> dict:
    """Read sweep parameters off ``params``, with safe defaults so older
    Configs that pre-date these knobs still work."""
    return {
        "min_atr":          float(getattr(params, "sweep_min_atr", 0.20)),
        "reclaim_window":   int(getattr(params, "sweep_reclaim_window", 12)),
        "strength_cap":     float(getattr(params, "sweep_strength_cap", 3.0)),
        "stop_run_close":   float(getattr(params, "stop_run_close_atr", 0.15)),
    }


def liquidity_sweeps(df: pd.DataFrame, params: DetectionParams,
                     tf: str = "base") -> List[LevelCandidate]:
    """Wick-only sweep: swing pierced by a wick whose extent exceeds
    ``sweep_min_atr * ATR``, and price closes back inside the original
    swing range within ``reclaim_window`` bars (the wick bar itself
    closes back inside is the typical case).

    Emits a LevelCandidate at the SWEPT swing level. Side semantics:
      * Swing HIGH swept (wick above, close back below) → the HIGH side
        of price is being defended. Pool side = "high".
      * Swing LOW swept (wick below, close back above) → the LOW side
        is being defended. Pool side = "low".

    Strength scales with (a) how far the wick pierced beyond the swing
    (in ATR), and (b) how quickly the reclaim happened (faster = stronger).
    """
    sh, sl = detect_swings(df, params.swing_left, params.swing_right)
    a_arr = atr(df, params.atr_period).bfill().values
    period = _infer_tf_period(df)
    right = params.swing_right
    ts = df.index
    h_arr = df["high"].values
    l_arr = df["low"].values
    c_arr = df["close"].values
    hw = params.pool_halfwidth_atr

    cfg = _sweep_params(params)
    min_pierce = cfg["min_atr"]
    window = cfg["reclaim_window"]
    cap = cfg["strength_cap"]

    n = len(df)
    out: List[LevelCandidate] = []

    swing_high_idxs = list(np.where(sh)[0])
    swing_low_idxs = list(np.where(sl)[0])

    # ---- Swept swing-HIGH (defending the high side) -------------------
    for i in swing_high_idxs:
        swing_price = float(h_arr[i])
        # Swing is confirmed at bar (i + right). The earliest CANDIDATE
        # sweep bar is one bar after confirmation.
        scan_start = i + right + 1
        scan_end = min(n, scan_start + window)
        a_swing = max(float(a_arr[i]), 1e-9)
        for j in range(scan_start, scan_end):
            wick = float(h_arr[j]) - swing_price
            if wick < min_pierce * a_swing:
                continue
            # Reclaim check: same bar's close back below the swing.
            if c_arr[j] >= swing_price:
                continue
            # Sweep + reclaim both confirmed at bar j's close.
            pierce_atr = wick / a_swing
            reclaim_bars = j - i
            speed_bonus = max(0.0, 1.0 - reclaim_bars / max(window, 1))
            strength = min(cap, 1.0 + pierce_atr + speed_bonus)
            known = ts[j] + period
            out.append(LevelCandidate(
                side="high", price=swing_price, ts=ts[i],
                source=f"SWEEP_H@{tf}", known_at=known,
                strength=float(strength),
                half_width=hw * a_swing, tf=tf,
                meta={"pierce_atr": float(pierce_atr),
                      "reclaim_bars": int(reclaim_bars),
                      "swept_bar_idx": int(j)},
            ))
            break    # one sweep event per swing

    # ---- Swept swing-LOW (defending the low side) ---------------------
    for i in swing_low_idxs:
        swing_price = float(l_arr[i])
        scan_start = i + right + 1
        scan_end = min(n, scan_start + window)
        a_swing = max(float(a_arr[i]), 1e-9)
        for j in range(scan_start, scan_end):
            wick = swing_price - float(l_arr[j])
            if wick < min_pierce * a_swing:
                continue
            if c_arr[j] <= swing_price:
                continue
            pierce_atr = wick / a_swing
            reclaim_bars = j - i
            speed_bonus = max(0.0, 1.0 - reclaim_bars / max(window, 1))
            strength = min(cap, 1.0 + pierce_atr + speed_bonus)
            known = ts[j] + period
            out.append(LevelCandidate(
                side="low", price=swing_price, ts=ts[i],
                source=f"SWEEP_L@{tf}", known_at=known,
                strength=float(strength),
                half_width=hw * a_swing, tf=tf,
                meta={"pierce_atr": float(pierce_atr),
                      "reclaim_bars": int(reclaim_bars),
                      "swept_bar_idx": int(j)},
            ))
            break
    return out


def stop_run_reclaims(df: pd.DataFrame, params: DetectionParams,
                      tf: str = "base") -> List[LevelCandidate]:
    """Stronger sweep variant: price CLOSES beyond the swing on at
    least one bar (the "stop run" — registers as a break to anyone
    watching closes), AND THEN closes back inside within
    ``reclaim_window`` bars.

    This is the canonical SMC swept-and-reclaimed setup. Distinct from
    ``liquidity_sweeps`` (which is wick-only): here the close-past
    happened, so a less-sophisticated trader would have stopped out on
    the break. The reclaim is the institutional re-entry. The level
    that was breached becomes a higher-conviction pool than a plain
    wick sweep.

    Emits a LevelCandidate with source ``SR_H`` / ``SR_L``, slightly
    higher base strength than ``liquidity_sweeps``, and a meta dict
    recording how many bars closed past before the reclaim.
    """
    sh, sl = detect_swings(df, params.swing_left, params.swing_right)
    a_arr = atr(df, params.atr_period).bfill().values
    period = _infer_tf_period(df)
    right = params.swing_right
    ts = df.index
    h_arr = df["high"].values
    l_arr = df["low"].values
    c_arr = df["close"].values
    hw = params.pool_halfwidth_atr

    cfg = _sweep_params(params)
    close_min = cfg["stop_run_close"]
    window = cfg["reclaim_window"]
    cap = cfg["strength_cap"]

    n = len(df)
    out: List[LevelCandidate] = []

    # ---- Swing-HIGH break-and-reclaim ---------------------------------
    for i in np.where(sh)[0]:
        swing_price = float(h_arr[i])
        scan_start = i + right + 1
        scan_end = min(n, scan_start + window)
        a_swing = max(float(a_arr[i]), 1e-9)
        close_threshold = swing_price + close_min * a_swing
        bars_closed_past = 0
        for j in range(scan_start, scan_end):
            if c_arr[j] >= close_threshold:
                bars_closed_past += 1
                continue
            # bar j closes back BELOW the swing (reclaim candidate)
            if bars_closed_past >= 1 and c_arr[j] < swing_price:
                # confirmed: at least one close-past + this close back inside
                strength = min(cap,
                                1.4 + 0.4 * bars_closed_past
                                + (float(h_arr[j - 1]) - swing_price) / a_swing)
                known = ts[j] + period
                out.append(LevelCandidate(
                    side="high", price=swing_price, ts=ts[i],
                    source=f"SR_H@{tf}", known_at=known,
                    strength=float(strength),
                    half_width=hw * a_swing, tf=tf,
                    meta={"closes_past": int(bars_closed_past),
                          "reclaim_bar_idx": int(j),
                          "swing_bar_idx": int(i)},
                ))
                break
            # bar didn't close past and didn't reclaim — keep scanning.

    # ---- Swing-LOW break-and-reclaim ----------------------------------
    for i in np.where(sl)[0]:
        swing_price = float(l_arr[i])
        scan_start = i + right + 1
        scan_end = min(n, scan_start + window)
        a_swing = max(float(a_arr[i]), 1e-9)
        close_threshold = swing_price - close_min * a_swing
        bars_closed_past = 0
        for j in range(scan_start, scan_end):
            if c_arr[j] <= close_threshold:
                bars_closed_past += 1
                continue
            if bars_closed_past >= 1 and c_arr[j] > swing_price:
                strength = min(cap,
                                1.4 + 0.4 * bars_closed_past
                                + (swing_price - float(l_arr[j - 1])) / a_swing)
                known = ts[j] + period
                out.append(LevelCandidate(
                    side="low", price=swing_price, ts=ts[i],
                    source=f"SR_L@{tf}", known_at=known,
                    strength=float(strength),
                    half_width=hw * a_swing, tf=tf,
                    meta={"closes_past": int(bars_closed_past),
                          "reclaim_bar_idx": int(j),
                          "swing_bar_idx": int(i)},
                ))
                break
    return out
