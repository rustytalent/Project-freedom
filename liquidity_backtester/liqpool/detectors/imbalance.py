"""Multi-bar imbalance + premium/discount midpoint detectors.

Two new ``LevelCandidate`` sources that fill specific gaps the existing
detector library does not capture:

  * ``multi_bar_imbalances`` — extends the classic 3-bar fair-value-gap
    detector to displacement RUNS spanning N >= 3 consecutive bars.
    Where the existing ``fair_value_gaps`` catches the canonical 3-bar
    gap (bar i-1 high < bar i+1 low), this detector finds longer
    no-retracement legs: K consecutive bullish bars whose individual
    ranges DON'T meaningfully overlap and whose cumulative displacement
    exceeds a configurable ATR threshold. The midpoint of the spanned
    range becomes a high-conviction return-to-mean pool — these are
    the "thick" imbalance zones that institutional impulse legs carve
    on real charts but that fall outside the strict 3-bar FVG window.

  * ``premium_discount_midpoints`` — the SMC "equilibrium" line. Given
    the most recent confirmed swing high + confirmed swing low pair,
    the midpoint of that range is the dealing-range equilibrium.
    Premium (above midpoint) is statistically a sell zone in the
    institutional taxonomy; discount (below) is a buy zone. The
    midpoint itself is the highest-confluence pool inside the range —
    smart-money entries cluster here.

Both detectors are causal: every ``known_at`` is the close timestamp of
the bar at which a real-time trader could first observe the pattern
(the last bar of the imbalance run, or the confirmation bar of the
later of the two swings forming the dealing range).

Configuration knobs (added to ``DetectionParams`` in the consuming
``Config``; safe defaults baked in here so older Configs work via
``getattr``):

  imbalance_min_run        default 3     — minimum consecutive bars in run
  imbalance_max_run        default 7     — cap so runaway trends don't
                                           spam the candidate stream
  imbalance_min_atr        default 0.50  — cumulative displacement, ATRs
  imbalance_max_overlap_atr default 0.05 — per-bar range overlap tolerated
                                           (in ATR) before the run breaks
  dealing_range_max_bars   default 200   — max bars between the swing
                                           high and swing low forming
                                           the dealing range
  dealing_range_min_atr    default 1.0   — minimum swing-pair range, ATRs
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..config import DetectionParams
from ..features import LevelCandidate, _infer_tf_period, detect_swings
from ..indicators import atr


def _imbalance_params(params: DetectionParams) -> dict:
    """Read imbalance params off ``params``, with safe defaults so older
    Configs that pre-date these knobs still work."""
    return {
        "min_run":              int(getattr(params, "imbalance_min_run", 3)),
        "max_run":              int(getattr(params, "imbalance_max_run", 7)),
        "min_atr":              float(getattr(params, "imbalance_min_atr", 0.50)),
        "max_overlap_atr":      float(getattr(params,
                                              "imbalance_max_overlap_atr",
                                              0.05)),
        "range_max_bars":       int(getattr(params,
                                            "dealing_range_max_bars", 200)),
        "range_min_atr":        float(getattr(params,
                                              "dealing_range_min_atr", 1.0)),
    }


def multi_bar_imbalances(df: pd.DataFrame, params: DetectionParams,
                          tf: str = "base") -> List[LevelCandidate]:
    """Displacement run: N consecutive bars marching in one direction
    with negligible per-bar retracement.

    Bullish run starting at bar ``i``:
      * For ``k`` in ``[i+1, i+max_run]``: every bar's low is >= previous
        bar's high - ``max_overlap_atr * ATR`` (no meaningful pullback
        into the prior bar's range).
      * The cumulative displacement (last_high - first_low) must exceed
        ``min_atr * ATR`` for the run to qualify.
      * The run must span at least ``min_run`` bars.

    The pool candidate is the MIDPOINT of the spanned range. Side =
    "low" for a bullish run (the imbalance is below the current price;
    a return to the midpoint is a demand-zone test).

    ``known_at`` is the close timestamp of the LAST bar in the run.
    Earlier disclosure would be a look-ahead.
    """
    h_arr = df["high"].values
    l_arr = df["low"].values
    a_arr = atr(df, params.atr_period).bfill().values
    period = _infer_tf_period(df)
    ts = df.index
    hw = params.pool_halfwidth_atr

    cfg = _imbalance_params(params)
    min_run = cfg["min_run"]
    max_run = cfg["max_run"]
    min_disp = cfg["min_atr"]
    max_overlap = cfg["max_overlap_atr"]

    n = len(df)
    out: List[LevelCandidate] = []

    i = 0
    while i < n - min_run:
        a_i = max(float(a_arr[i]), 1e-9)
        overlap_tol = max_overlap * a_i

        # ---- Bullish run starting at bar i --------------------------------
        run_end = i
        for k in range(i + 1, min(n, i + max_run + 1)):
            # No retracement: this bar's low must sit at or above the
            # previous bar's high (minus tiny overlap_tol). Equivalently,
            # the bars are stacking with no real wick overlap.
            if l_arr[k] < h_arr[k - 1] - overlap_tol:
                break
            run_end = k
        bull_len = run_end - i + 1
        if bull_len >= min_run:
            displacement = float(h_arr[run_end]) - float(l_arr[i])
            if displacement >= min_disp * a_i:
                mid = (float(h_arr[run_end]) + float(l_arr[i])) / 2.0
                half_w = max(displacement / 2.0, hw * a_i)
                known = ts[run_end] + period
                out.append(LevelCandidate(
                    side="low", price=mid, ts=ts[i],
                    source=f"MBI_bull@{tf}", known_at=known,
                    strength=min(2.5, displacement / a_i),
                    half_width=half_w, tf=tf,
                    meta={"run_len": int(bull_len),
                          "displacement_atr": float(displacement / a_i),
                          "start_idx": int(i),
                          "end_idx": int(run_end),
                          "low": float(l_arr[i]),
                          "high": float(h_arr[run_end])},
                ))
                # advance past the consumed run so we don't double-count
                i = run_end + 1
                continue

        # ---- Bearish run starting at bar i --------------------------------
        run_end = i
        for k in range(i + 1, min(n, i + max_run + 1)):
            if h_arr[k] > l_arr[k - 1] + overlap_tol:
                break
            run_end = k
        bear_len = run_end - i + 1
        if bear_len >= min_run:
            displacement = float(h_arr[i]) - float(l_arr[run_end])
            if displacement >= min_disp * a_i:
                mid = (float(h_arr[i]) + float(l_arr[run_end])) / 2.0
                half_w = max(displacement / 2.0, hw * a_i)
                known = ts[run_end] + period
                out.append(LevelCandidate(
                    side="high", price=mid, ts=ts[i],
                    source=f"MBI_bear@{tf}", known_at=known,
                    strength=min(2.5, displacement / a_i),
                    half_width=half_w, tf=tf,
                    meta={"run_len": int(bear_len),
                          "displacement_atr": float(displacement / a_i),
                          "start_idx": int(i),
                          "end_idx": int(run_end),
                          "low": float(l_arr[run_end]),
                          "high": float(h_arr[i])},
                ))
                i = run_end + 1
                continue

        i += 1

    return out


def premium_discount_midpoints(df: pd.DataFrame, params: DetectionParams,
                                tf: str = "base") -> List[LevelCandidate]:
    """The SMC dealing-range equilibrium line.

    For each consecutive (confirmed swing high, confirmed swing low)
    pair within ``dealing_range_max_bars`` whose range is at least
    ``dealing_range_min_atr * ATR``, emit a LevelCandidate at the
    range MIDPOINT. The midpoint is the equilibrium between premium
    (above) and discount (below) — institutional re-entries cluster
    here regardless of which side price is on.

    Two LevelCandidates per qualifying pair: one ``side="high"`` (the
    midpoint defending against premium-to-discount sells) and one
    ``side="low"`` (defending against discount-to-premium buys). The
    pool builder treats them as a confluence zone via merge_atr; the
    duplication is intentional so the level is recognized whichever
    direction price approaches from.

    ``known_at`` is the close of the LATER of the two swings'
    confirmation bars — both swings must be confirmed before a real-
    time trader could draw this equilibrium line.
    """
    sh, sl = detect_swings(df, params.swing_left, params.swing_right)
    a_arr = atr(df, params.atr_period).bfill().values
    period = _infer_tf_period(df)
    right = params.swing_right
    ts = df.index
    h_arr = df["high"].values
    l_arr = df["low"].values
    hw = params.pool_halfwidth_atr

    cfg = _imbalance_params(params)
    max_bars = cfg["range_max_bars"]
    min_atr = cfg["range_min_atr"]

    n = len(df)
    out: List[LevelCandidate] = []

    swing_high_idxs = list(np.where(sh)[0])
    swing_low_idxs = list(np.where(sl)[0])
    if not swing_high_idxs or not swing_low_idxs:
        return out

    # Build a sorted timeline of (idx, kind, price) so we can pair each
    # swing with the closest opposite-side swing that precedes or follows
    # it within the window.
    events: list = []
    for i in swing_high_idxs:
        events.append((i, "high", float(h_arr[i])))
    for i in swing_low_idxs:
        events.append((i, "low", float(l_arr[i])))
    events.sort(key=lambda e: e[0])

    emitted_pairs: set = set()
    for k in range(len(events) - 1):
        i_a, kind_a, price_a = events[k]
        for m in range(k + 1, len(events)):
            i_b, kind_b, price_b = events[m]
            if kind_a == kind_b:
                continue
            if i_b - i_a > max_bars:
                break
            # Confirmation: the LATER swing must be fully confirmed
            # (i.e. the dataframe must have at least `right` more bars
            # past i_b). Otherwise the pair isn't observable yet.
            if i_b + right >= n:
                break
            pair_key = (min(i_a, i_b), max(i_a, i_b))
            if pair_key in emitted_pairs:
                continue
            range_size = abs(price_a - price_b)
            a_ref = max(float(a_arr[i_b]), 1e-9)
            if range_size < min_atr * a_ref:
                continue
            mid = (price_a + price_b) / 2.0
            known = ts[i_b + right] + period
            half_w = max(range_size * 0.10, hw * a_ref)
            strength = min(2.5, 1.0 + range_size / a_ref / 4.0)
            common = dict(
                price=mid, ts=ts[i_b],
                source=f"PD_MID@{tf}", known_at=known,
                strength=float(strength),
                half_width=float(half_w), tf=tf,
                meta={"swing_high_price": float(
                          price_a if kind_a == "high" else price_b),
                      "swing_low_price": float(
                          price_b if kind_b == "low" else price_a),
                      "range_atr": float(range_size / a_ref),
                      "earlier_idx": int(min(i_a, i_b)),
                      "later_idx": int(max(i_a, i_b))},
            )
            out.append(LevelCandidate(side="high", **common))
            out.append(LevelCandidate(side="low", **common))
            emitted_pairs.add(pair_key)
            # Only emit one PD_MID per (earlier swing) — the dealing
            # range is defined by the most recent opposite swing.
            break

    return out
