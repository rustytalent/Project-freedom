"""Extractors — the digestive system of the flywheel.

Each flywheel model (M.4-M.10) eats a specific training frame. The
engine already EXCRETES the raw material (walkforward results, shadow
log, outcome log) but nothing was converting artifacts into food.
These extractors are those missing enzymes:

    engine artifact          extractor                       feeds
    -----------------        ---------------------------     -----
    wf.oos_pools/results  -> detector_outcomes_from_report -> M.5 trust
    wf + base_df          -> reaction_paths_from_report    -> M.7 archetypes
    wf across assets      -> co_occurrences_from_report    -> M.9 transfer
    outcome-log joined    -> bucket_aging_history           -> M.8 aging
    shadow joined         -> (already model-shaped)         -> M.4 regret
    drift history         -> (already model-shaped)         -> M.6 imminent
    outcome-log joined    -> (already model-shaped)         -> M.10 meta-cal

Every extractor is defensive: missing attributes, empty frames, and
malformed rows yield smaller outputs, never exceptions — an extractor
that crashes the nightly train job starves every downstream model.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..indicators import atr as compute_atr
from ..pools import _factor_of


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dominant_factor(pool: Any) -> str:
    """Most-represented factor family among the pool's contributors."""
    counts: Dict[str, int] = {}
    for c in getattr(pool, "contributors", []) or []:
        f = _factor_of(getattr(c, "source", "")) or "OTHER"
        counts[f] = counts.get(f, 0) + 1
    if not counts:
        return "OTHER"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _regime_label(df: pd.DataFrame, ts: pd.Timestamp,
                  adx_series: Optional[pd.Series] = None) -> str:
    """Coarse trend/range label at a timestamp. v1 uses ADX(14) >= 25
    as 'trend' — conventional and good enough for the 2-regime trust
    buckets; the registry can host the 25 once this is validated."""
    try:
        if adx_series is None:
            from ..regime import adx
            adx_series = adx(df, 14)
        pos = adx_series.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            return "range"
        val = float(adx_series.iloc[pos])
        return "trend" if val >= 25.0 else "range"
    except Exception:
        return "range"


def _iter_asset_walkforwards(report: Any):
    assets = getattr(report, "assets", {}) or {}
    for symbol, ad in assets.items():
        wf = getattr(ad, "walkforward", None)
        if wf is None:
            continue
        pools = getattr(wf, "oos_pools", []) or []
        results = getattr(wf, "oos_results", []) or []
        df = getattr(ad, "base_df", None)
        yield str(symbol), pools, results, df


# ---------------------------------------------------------------------------
# M.5 — detector trust food
# ---------------------------------------------------------------------------

def detector_outcomes_from_report(report: Any) -> pd.DataFrame:
    """(factor, regime, respected) — one row per DECISIVE tested pool.

    Pools that were never touched carry no respect information and are
    skipped; including them as 0 would teach the router that quiet
    factors are untrustworthy when they were merely far from price.
    """
    rows: List[Dict[str, Any]] = []
    for symbol, pools, results, df in _iter_asset_walkforwards(report):
        adx_series = None
        if df is not None and not df.empty:
            try:
                from ..regime import adx
                adx_series = adx(df, 14)
            except Exception:
                adx_series = None
        for result in results:
            try:
                if not getattr(result, "is_tested", False):
                    continue
                idx = int(getattr(result, "pool_idx", -1))
                pool = pools[idx] if 0 <= idx < len(pools) else None
                factor = _dominant_factor(pool) if pool is not None else "OTHER"
                formed = getattr(result, "formed_at", None)
                regime = (
                    _regime_label(df, formed, adx_series)
                    if df is not None and formed is not None else "range"
                )
                rows.append({
                    "symbol": symbol,
                    "factor": factor,
                    "regime": regime,
                    "respected": float(bool(getattr(result, "is_respect", False))),
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M.7 — reaction archetype food
# ---------------------------------------------------------------------------

def reaction_paths_from_report(
    report: Any, path_bars: int = 6,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(paths, sides, context) for every touched pool with enough
    post-touch bars.

    path[i] = (close[t+1..t+path_bars] - touch_level) / ATR(touch)
    side    = +1 for demand pools below price (respect bounces UP),
              -1 for supply pools above (respect bounces DOWN)
    context = the M.7 classifier features at touch time.
    """
    paths: List[np.ndarray] = []
    sides: List[float] = []
    ctx_rows: List[Dict[str, float]] = []
    for symbol, pools, results, df in _iter_asset_walkforwards(report):
        if df is None or df.empty:
            continue
        try:
            atr_series = compute_atr(df, 14)
        except Exception:
            continue
        closes = df["close"]
        for result in results:
            try:
                touched = getattr(result, "touched_at", None)
                if touched is None:
                    continue
                pos = df.index.searchsorted(touched, side="right") - 1
                if pos < 0 or pos + path_bars >= len(df):
                    continue
                level = (float(getattr(result, "price_low", np.nan))
                         + float(getattr(result, "price_high", np.nan))) / 2.0
                a = float(atr_series.iloc[pos])
                if not math.isfinite(level) or not math.isfinite(a) or a <= 0:
                    continue
                seg = closes.iloc[pos + 1: pos + 1 + path_bars].to_numpy(dtype=float)
                if len(seg) < path_bars:
                    continue
                side = 1.0 if str(getattr(result, "side", "low")) == "low" else -1.0
                paths.append((seg - level) / a)
                sides.append(side)
                last_close = float(closes.iloc[pos])
                ctx_rows.append({
                    "distance_atr": abs(last_close - level) / a,
                    "vol_frac": a / max(last_close, 1e-9),
                    "side_long": 1.0 if side > 0 else 0.0,
                    "session_open": 0.0,   # refined later; touch ToD needs IST mapping
                    "session_close": 0.0,
                    "tf_count": float(len(getattr(result, "tfs", []) or [])),
                    "score": float(getattr(result, "score", 0.0)),
                })
            except Exception:
                continue
    if not paths:
        return (np.zeros((0, path_bars)), np.zeros(0), pd.DataFrame())
    return np.vstack(paths), np.asarray(sides), pd.DataFrame(ctx_rows)


# ---------------------------------------------------------------------------
# M.9 — cross-asset transfer food
# ---------------------------------------------------------------------------

def co_occurrences_from_report(report: Any) -> pd.DataFrame:
    """(asset_a, asset_b, agreed) — same-IST-DATE decisive pool pairs.

    'agreed' = both respected or both broke. One row per cross-asset
    pool pair that resolved on the same trading date; the empirical
    matrix shrinks these counts."""
    per_asset: Dict[str, List[Tuple[str, bool]]] = {}
    for symbol, pools, results, df in _iter_asset_walkforwards(report):
        outcomes: List[Tuple[str, bool]] = []
        for result in results:
            try:
                if not getattr(result, "is_tested", False):
                    continue
                ts = (getattr(result, "touched_at", None)
                      or getattr(result, "broken_at", None))
                if ts is None:
                    continue
                date_key = str(pd.Timestamp(ts).date())
                outcomes.append((date_key, bool(getattr(result, "is_respect", False))))
            except Exception:
                continue
        if outcomes:
            per_asset[symbol] = outcomes
    rows: List[Dict[str, Any]] = []
    symbols = sorted(per_asset)
    for i, a in enumerate(symbols):
        by_date_a: Dict[str, List[bool]] = {}
        for d, r in per_asset[a]:
            by_date_a.setdefault(d, []).append(r)
        for b in symbols[i + 1:]:
            for d, r_b in per_asset[b]:
                for r_a in by_date_a.get(d, []):
                    rows.append({
                        "asset_a": a, "asset_b": b,
                        "agreed": float(r_a == r_b),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M.8 — bucket aging food
# ---------------------------------------------------------------------------

def bucket_aging_history_from_joined(
    joined: pd.DataFrame,
    bundle_fit_date: str,
) -> pd.DataFrame:
    """(prediction_type, age_days, n_resolved, abs_calib_error) — one
    row per (trading_date, prediction_type) cell of the outcome log,
    with age measured from the model bundle's fit date.

    ``joined`` is the outcome-log join (one row per prediction);
    ``bundle_fit_date`` is the YYYY-MM-DD the serving bundle was
    trained. Cells aggregate per-date so each row is one day's
    realised calibration error at one age."""
    if joined.empty or "outcome_boolean" not in joined.columns:
        return pd.DataFrame()
    sub = joined[joined["outcome_boolean"].notna()].copy()
    if sub.empty or "trading_date_ist" not in sub.columns:
        return pd.DataFrame()
    fit = pd.Timestamp(bundle_fit_date)
    rows: List[Dict[str, Any]] = []
    grouped = sub.groupby(["trading_date_ist", "prediction_type"])
    for (date, ptype), g in grouped:
        try:
            age = (pd.Timestamp(str(date)) - fit).days
            if age < 0:
                continue
            hit = g["outcome_boolean"].astype(float).mean()
            mean_p = g["predicted_value"].astype(float).mean()
            rows.append({
                "prediction_type": str(ptype),
                "age_days": float(age),
                "n_resolved": int(len(g)),
                "abs_calib_error": abs(float(mean_p) - float(hit)),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)
