"""Feature detectors that contribute candidate liquidity points to the pool builder.

Each detector returns a list of `LevelCandidate` records with: side (high/low), price, ts,
source (which detector), strength (raw 0-1), and optional zone width.
The pool builder clusters these into zones and applies factor weights to get a final score.

Concepts implemented:
  - Swing highs/lows (fractals) – the bedrock of "where stops sit".
  - Equal Highs / Equal Lows (EQH/EQL) – repeated swings at the same price = stop cluster.
  - Previous Day / Week / Month High & Low (PDH/PDL/PWH/PWL/PMH/PML) – session liquidity.
  - Fair Value Gaps (FVG) – three-candle imbalance, both bullish & bearish.
  - Order Blocks (OB) – last opposite-color candle before a displacement.
  - In-candle imbalance / rejection wicks – single-bar liquidity grabs.
  - Opening Range Breakout extremes (ORB high/low) – session pivots.
  - Volume nodes (HVN) – rolling volume-by-price concentration peaks.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd

from .indicators import atr, candle_features
from .config import DetectionParams


@dataclass
class LevelCandidate:
    side: str               # "high" or "low" — which side of price the stops likely sit on
    price: float            # central price of the candidate level
    ts: pd.Timestamp        # when the level was formed
    source: str             # detector name (e.g. "EQH", "PWH", "FVG_bull", "OB_bull", ...)
    strength: float = 1.0   # detector-local strength in [0, ~5]
    half_width: float = 0.0 # zone half-width in price units; 0 means a thin line
    tf: str = "base"        # source timeframe label
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Swings (fractals)
# ---------------------------------------------------------------------------

def detect_swings(df: pd.DataFrame, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (swing_high, swing_low) aligned to df index."""
    h, l = df["high"].values, df["low"].values
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        wh = h[i - left:i + right + 1]
        wl = l[i - left:i + right + 1]
        # Strict: must be unique max/min in the window.
        if h[i] >= wh.max() and (wh == h[i]).sum() == 1:
            sh[i] = True
        if l[i] <= wl.min() and (wl == l[i]).sum() == 1:
            sl[i] = True
    return sh, sl


def swing_candidates(df: pd.DataFrame, params: DetectionParams, tf: str) -> List[LevelCandidate]:
    sh, sl = detect_swings(df, params.swing_left, params.swing_right)
    a = atr(df, params.atr_period).bfill().values
    out: List[LevelCandidate] = []
    h, l = df["high"].values, df["low"].values
    ts = df.index
    hw = params.pool_halfwidth_atr
    for i in np.where(sh)[0]:
        out.append(LevelCandidate("high", float(h[i]), ts[i], f"SWING_H@{tf}", 1.0,
                                  half_width=hw * a[i], tf=tf))
    for i in np.where(sl)[0]:
        out.append(LevelCandidate("low", float(l[i]), ts[i], f"SWING_L@{tf}", 1.0,
                                  half_width=hw * a[i], tf=tf))
    return out


# ---------------------------------------------------------------------------
# Equal Highs / Equal Lows -- the heaviest external-liquidity pools.
# ---------------------------------------------------------------------------

def equal_levels(df: pd.DataFrame, params: DetectionParams, tf: str) -> List[LevelCandidate]:
    sh, sl = detect_swings(df, params.swing_left, params.swing_right)
    a = atr(df, params.atr_period).bfill().values
    h, l, ts = df["high"].values, df["low"].values, df.index
    out: List[LevelCandidate] = []

    def cluster(idx: np.ndarray, prices: np.ndarray, side: str):
        used = np.zeros(len(idx), dtype=bool)
        for a_i, ix in enumerate(idx):
            if used[a_i]:
                continue
            group = [a_i]
            base_price = prices[ix]
            tol = params.eqhl_tol_atr * a[ix]
            for b_i in range(a_i + 1, len(idx)):
                jx = idx[b_i]
                if used[b_i]:
                    continue
                if jx - ix > params.eqhl_max_bars:
                    break
                if abs(prices[jx] - base_price) <= tol:
                    group.append(b_i)
            if len(group) >= params.eqhl_min_touches:
                for g in group:
                    used[g] = True
                group_idx = idx[group]
                avg_price = float(prices[group_idx].mean())
                # Strength grows with number of touches and shrinks with span (recent stacks heavier).
                span = group_idx[-1] - group_idx[0]
                strength = min(3.0, 0.5 * len(group) + (1.0 if span < params.eqhl_max_bars / 2 else 0.0))
                out.append(LevelCandidate(
                    side, avg_price, ts[group_idx[-1]],
                    f"EQ{'H' if side == 'high' else 'L'}@{tf}",
                    strength=strength,
                    half_width=max(params.pool_halfwidth_atr * a[group_idx[-1]],
                                   abs(prices[group_idx].max() - prices[group_idx].min()) / 2),
                    tf=tf,
                    meta={"touches": int(len(group)), "indices": [int(x) for x in group_idx]},
                ))

    cluster(np.where(sh)[0], h, "high")
    cluster(np.where(sl)[0], l, "low")
    return out


# ---------------------------------------------------------------------------
# Previous Day / Week / Month highs & lows
# ---------------------------------------------------------------------------

def previous_period_levels(df: pd.DataFrame, tf_label: str = "base") -> List[LevelCandidate]:
    out: List[LevelCandidate] = []
    for rule, tag in [("1D", "PD"), ("1W", "PW"), ("1ME", "PM")]:
        try:
            grp = df.resample(rule, label="left", closed="left").agg({"high": "max", "low": "min"}).dropna()
        except Exception:
            continue
        # For each completed period i, project its high/low as a level valid starting at period i+1.
        for i in range(len(grp) - 1):
            ts_start_next = grp.index[i + 1]
            out.append(LevelCandidate("high", float(grp["high"].iloc[i]), ts_start_next,
                                      f"{tag}H", strength=1.5 if tag == "PD" else 2.0 if tag == "PW" else 2.5,
                                      tf=tag))
            out.append(LevelCandidate("low", float(grp["low"].iloc[i]), ts_start_next,
                                      f"{tag}L", strength=1.5 if tag == "PD" else 2.0 if tag == "PW" else 2.5,
                                      tf=tag))
    return out


# ---------------------------------------------------------------------------
# Fair Value Gap (3-candle imbalance)
# ---------------------------------------------------------------------------

def fair_value_gaps(df: pd.DataFrame, params: DetectionParams, tf: str) -> List[LevelCandidate]:
    h, l = df["high"].values, df["low"].values
    a = atr(df, params.atr_period).bfill().values
    ts = df.index
    out: List[LevelCandidate] = []
    for i in range(1, len(df) - 1):
        # Bullish FVG: prior high < next low → gap [prior_high, next_low].
        if h[i - 1] < l[i + 1]:
            gap = l[i + 1] - h[i - 1]
            if gap >= params.fvg_min_atr * a[i]:
                mid = (h[i - 1] + l[i + 1]) / 2
                # A bullish FVG is a *demand* zone — stops/orders sit on the low side.
                out.append(LevelCandidate("low", float(mid), ts[i + 1], f"FVG_bull@{tf}",
                                          strength=min(2.5, gap / max(a[i], 1e-9)),
                                          half_width=float(gap / 2), tf=tf,
                                          meta={"low": float(h[i - 1]), "high": float(l[i + 1])}))
        # Bearish FVG: prior low > next high → gap [next_high, prior_low].
        if l[i - 1] > h[i + 1]:
            gap = l[i - 1] - h[i + 1]
            if gap >= params.fvg_min_atr * a[i]:
                mid = (l[i - 1] + h[i + 1]) / 2
                out.append(LevelCandidate("high", float(mid), ts[i + 1], f"FVG_bear@{tf}",
                                          strength=min(2.5, gap / max(a[i], 1e-9)),
                                          half_width=float(gap / 2), tf=tf,
                                          meta={"low": float(h[i + 1]), "high": float(l[i - 1])}))
    return out


# ---------------------------------------------------------------------------
# Order Blocks: last opposite-color candle before a displacement move.
# ---------------------------------------------------------------------------

def order_blocks(df: pd.DataFrame, params: DetectionParams, tf: str) -> List[LevelCandidate]:
    o, c, h, l = df["open"].values, df["close"].values, df["high"].values, df["low"].values
    a = atr(df, params.atr_period).bfill().values
    ts = df.index
    out: List[LevelCandidate] = []
    n = len(df)
    for i in range(2, n - 1):
        disp = (c[i] - o[i]) / max(a[i], 1e-9)
        if abs(disp) < params.ob_displacement_atr:
            continue
        # bullish displacement → find most recent bearish candle within lookback
        if disp > 0:
            for j in range(i - 1, max(i - params.ob_lookback, 0), -1):
                if c[j] < o[j]:
                    out.append(LevelCandidate("low", float((o[j] + l[j]) / 2), ts[j],
                                              f"OB_bull@{tf}",
                                              strength=min(2.5, abs(disp)),
                                              half_width=float((o[j] - l[j]) / 2), tf=tf,
                                              meta={"low": float(l[j]), "high": float(o[j])}))
                    break
        else:
            for j in range(i - 1, max(i - params.ob_lookback, 0), -1):
                if c[j] > o[j]:
                    out.append(LevelCandidate("high", float((o[j] + h[j]) / 2), ts[j],
                                              f"OB_bear@{tf}",
                                              strength=min(2.5, abs(disp)),
                                              half_width=float((h[j] - o[j]) / 2), tf=tf,
                                              meta={"low": float(o[j]), "high": float(h[j])}))
                    break
    return out


# ---------------------------------------------------------------------------
# In-candle imbalance — rejection wicks / liquidity grabs at a single bar.
# ---------------------------------------------------------------------------

def in_candle_imbalance(df: pd.DataFrame, params: DetectionParams, tf: str) -> List[LevelCandidate]:
    cf = candle_features(df)
    a = atr(df, params.atr_period).bfill().values
    ts = df.index
    h, l = df["high"].values, df["low"].values
    out: List[LevelCandidate] = []
    upper = cf["upper_wick_ratio"].values
    lower = cf["lower_wick_ratio"].values
    body = cf["body_ratio"].values
    rng = cf["range"].values
    for i in range(len(df)):
        # Long upper wick + small body → swept highs, sellers absorbed → liquidity *was* taken at h[i].
        # The high itself becomes a recently-grabbed pool (often re-tested before truly invalidating).
        if upper[i] >= params.wick_dominance and body[i] <= params.body_max_ratio and rng[i] > 0.2 * a[i]:
            out.append(LevelCandidate("high", float(h[i]), ts[i], f"REJ_high@{tf}",
                                      strength=min(2.0, upper[i] * 2),
                                      half_width=0.10 * a[i], tf=tf))
        if lower[i] >= params.wick_dominance and body[i] <= params.body_max_ratio and rng[i] > 0.2 * a[i]:
            out.append(LevelCandidate("low", float(l[i]), ts[i], f"REJ_low@{tf}",
                                      strength=min(2.0, lower[i] * 2),
                                      half_width=0.10 * a[i], tf=tf))
    return out


# ---------------------------------------------------------------------------
# Opening Range Breakout extremes (per session).
# ---------------------------------------------------------------------------

def orb_levels(df: pd.DataFrame, params: DetectionParams) -> List[LevelCandidate]:
    out: List[LevelCandidate] = []
    if len(df) == 0:
        return out
    minutes = pd.Timedelta(minutes=params.orb_minutes)
    by_day = df.groupby(df.index.normalize())
    for day, day_df in by_day:
        if day_df.empty:
            continue
        session_start = day_df.index[0]
        orb_window = day_df.loc[:session_start + minutes]
        if orb_window.empty:
            continue
        hi = float(orb_window["high"].max())
        lo = float(orb_window["low"].min())
        t = orb_window.index[-1]
        out.append(LevelCandidate("high", hi, t, "ORB_H", strength=1.2))
        out.append(LevelCandidate("low", lo, t, "ORB_L", strength=1.2))
    return out


# ---------------------------------------------------------------------------
# Volume-by-price (HVN) on a rolling window.
# ---------------------------------------------------------------------------

def volume_nodes(df: pd.DataFrame, params: DetectionParams, tf: str) -> List[LevelCandidate]:
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return []
    out: List[LevelCandidate] = []
    window = max(params.vp_bins * 4, min(params.vp_window_bars, len(df)))
    a = atr(df, params.atr_period).bfill().values
    # Compute one VP per non-overlapping window then take top-N peaks per window.
    for start in range(0, len(df) - window, window // 2):
        end = start + window
        chunk = df.iloc[start:end]
        prices = ((chunk["high"] + chunk["low"]) / 2).values
        vols = chunk["volume"].values
        lo, hi = prices.min(), prices.max()
        if hi <= lo:
            continue
        bins = np.linspace(lo, hi, params.vp_bins + 1)
        hist, _ = np.histogram(prices, bins=bins, weights=vols)
        # Find top-N bins.
        if hist.sum() == 0:
            continue
        top = np.argsort(hist)[-params.vp_top_n:]
        ts = chunk.index[-1]
        for b in top:
            center = (bins[b] + bins[b + 1]) / 2
            strength = float(hist[b] / hist.sum()) * params.vp_top_n
            # Volume node is side-agnostic — treat as both: the pool builder will dedupe.
            out.append(LevelCandidate("high", float(center), ts, f"HVN@{tf}",
                                      strength=min(2.0, strength * 3),
                                      half_width=0.15 * a[end - 1], tf=tf))
            out.append(LevelCandidate("low", float(center), ts, f"HVN@{tf}",
                                      strength=min(2.0, strength * 3),
                                      half_width=0.15 * a[end - 1], tf=tf))
    return out


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def detect_all(df: pd.DataFrame, params: DetectionParams, tf: str = "base",
               include_orb: bool = True) -> List[LevelCandidate]:
    """Run every detector on a single timeframe's OHLCV and return the merged candidate list."""
    cands: List[LevelCandidate] = []
    cands.extend(swing_candidates(df, params, tf))
    cands.extend(equal_levels(df, params, tf))
    cands.extend(fair_value_gaps(df, params, tf))
    cands.extend(order_blocks(df, params, tf))
    cands.extend(in_candle_imbalance(df, params, tf))
    cands.extend(volume_nodes(df, params, tf))
    if include_orb and tf == "base":
        cands.extend(orb_levels(df, params))
    return cands
