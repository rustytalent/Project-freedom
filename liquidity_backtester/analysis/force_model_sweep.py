"""Force model sweep — triple-barrier labels × grid sweep × learned R-policy.

The central problem the user identified: high model AUCs (proximity 0.97,
direction 0.62-0.72, reaction 0.77) but every execution mode is negative net
R. The gap is path-dependence — those models predict endpoint labels
(touched? direction? respected?), while a trade's outcome depends on the
PATH the price takes between entry and barrier. A direction-correct trade
can still hit stop before target.

This script tests two specific hypotheses:

  Part 1 (STATIC GRID): for each (stop, target, horizon) configuration in a
  4 x 4 x 3 = 48-cell grid, compute realized R via triple-barrier labels
  across every touched OOS pool. Report per-config mean R, win rate, PF, 95%
  CI on mean R. The verdict: is there a fixed barrier config with
  statistically positive net R? This is the "what's the best fixed policy"
  question.

  Part 2 (FORCE MODEL): train a LightGBM regressor on
  (state features + pool features + barrier-config features) -> realized R.
  Chronological train/val split. For each grid config, slice val predictions
  to that config setting, take the top-decile by predicted R, report the
  realized R within that subset with 95% CI. The verdict: can a learned
  policy pick better trades than the best fixed config?

Features come in two modes (--use-extended-features flag):

  basic (default): pool geometry + state at entry, all derivable directly
    from df_base.

  extended: basic + non-geometry microstructure --
    volume_ratio_5b, volume_zscore_20b, gap_atr, range_compression,
    time_of_day_quartile, vol_ratio_regime.

Run order on the user's Mac (when they're back):

  PYTHONPATH=. .venv/bin/python analysis/force_model_sweep.py \\
      --model-dir output_models/core25_latest \\
      --use-extended-features false \\
      --out output_audit/force_model_basic

  PYTHONPATH=. .venv/bin/python analysis/force_model_sweep.py \\
      --model-dir output_models/core25_latest \\
      --use-extended-features true \\
      --out output_audit/force_model_extended

Compare verdicts to see whether the non-geometry features close the gap.
"""
from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from liqpool.execution_backtest import (
    NO_NEW_ENTRY_AFTER_IST_MIN,
    SESSION_CLOSE_IST_MIN,
    SESSION_OPEN_IST_MIN,
    _headline_factor,
    _ist_minute_of_day,
)
from liqpool.indicators import atr
from liqpool.sectors import sector_of
from liqpool.triple_barrier import triple_barrier_label


# 4 stops × 4 targets × 3 horizons = 48 configs. Covers tight intraday
# (12 bars = 1 hour) to whole-session (78 bars). Matches the user's grid.
GRID: List[Tuple[float, float, int]] = [
    (s, t, h)
    for s in (0.3, 0.5, 0.75, 1.0)
    for t in (1.0, 1.5, 2.0, 3.0)
    for h in (12, 24, 78)
]

# Categorical buckets for one-hot.
FACTOR_BUCKETS = ("OB", "FVG", "EQHL", "REJ", "ORB", "SWING", "OTHER")
SECTOR_BUCKETS = ("AUTO", "BANKING", "FMCG", "IT", "PHARMA", "OTHER")


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _log_ret(close_arr: np.ndarray, idx: int, k: int) -> float:
    i0 = max(0, idx - k)
    if close_arr[i0] <= 0:
        return 0.0
    return float(np.log(close_arr[idx] / close_arr[i0]))


def _basic_features(df: pd.DataFrame, entry_idx: int, pool, atr_val: float,
                    symbol: str) -> Dict[str, float]:
    """Pool geometry + state-at-entry features. Self-contained — no
    StateFeaturizer dependency, so we can compute on bundle-loaded base_df
    directly without rebuilding active-pool snapshots."""
    n = entry_idx
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    feats: Dict[str, float] = {}

    # Pool geometry.
    feats["pool_score"] = float(pool.score)
    feats["pool_tf_count"] = float(len(set(pool.tfs)))
    feats["pool_width_atr"] = float(pool.width) / max(atr_val, 1e-9)
    feats["pool_n_contributors"] = float(len(pool.contributors))
    feats["side_high"] = 1.0 if pool.side == "high" else 0.0
    headline = _headline_factor(pool)
    for f in FACTOR_BUCKETS:
        feats[f"factor_{f}"] = 1.0 if (
            headline == f or (f == "OTHER" and headline not in FACTOR_BUCKETS[:-1])
        ) else 0.0

    # Sector one-hot.
    sec = sector_of(symbol)
    for s in SECTOR_BUCKETS:
        feats[f"sector_{s}"] = 1.0 if (
            sec == s or (s == "OTHER" and sec not in SECTOR_BUCKETS[:-1])
        ) else 0.0

    # State at entry.
    feats["ret_1"] = _log_ret(close, n, 1)
    feats["ret_6"] = _log_ret(close, n, 6)
    feats["ret_24"] = _log_ret(close, n, 24)
    feats["ret_78"] = _log_ret(close, n, 78)

    if n >= 6:
        rng = high[max(0, n - 6):n + 1].max() - low[max(0, n - 6):n + 1].min()
        feats["range_6_atr"] = float(rng / max(atr_val, 1e-9))
    else:
        feats["range_6_atr"] = 0.0

    feats["atr_value"] = float(atr_val)
    feats["minutes_since_session_open"] = float(max(
        0, _ist_minute_of_day(df.index[entry_idx]) - SESSION_OPEN_IST_MIN
    ))

    return feats


def _extended_features(df: pd.DataFrame, entry_idx: int,
                       atr_val: float) -> Dict[str, float]:
    """Non-geometry features: volume, gap, range compression, regime,
    time-of-day quartile."""
    n = entry_idx
    vol = df["volume"].values if "volume" in df.columns else np.ones(len(df))
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_ = df["open"].values

    feats: Dict[str, float] = {}

    # Volume ratio: last 5 bars vs last 20 bars average.
    if n >= 20:
        v5 = float(vol[max(0, n - 5):n].mean())
        v20 = float(vol[max(0, n - 20):n].mean())
        feats["volume_ratio_5b"] = v5 / max(v20, 1e-9)
    else:
        feats["volume_ratio_5b"] = 1.0

    # Volume z-score: current bar vs last 20 bars distribution.
    if n >= 20:
        recent = vol[max(0, n - 20):n]
        mu = float(recent.mean())
        sigma = float(recent.std(ddof=1)) if len(recent) > 1 else 1.0
        feats["volume_zscore_20b"] = (float(vol[n]) - mu) / max(sigma, 1e-9)
    else:
        feats["volume_zscore_20b"] = 0.0

    # Gap from prior bar's close to current bar's open, in ATR units.
    # This captures both intraday "micro gaps" and session-open gaps.
    if n > 0:
        feats["gap_atr"] = float(
            (open_[n] - close[n - 1]) / max(atr_val, 1e-9)
        )
    else:
        feats["gap_atr"] = 0.0

    # Range compression: last 6 bars' avg range vs last 30 bars' avg range.
    # < 1.0 means recent bars are tighter — often precedes expansion.
    if n >= 30:
        ranges = high[max(0, n - 30):n] - low[max(0, n - 30):n]
        recent6 = float(ranges[-6:].mean()) if len(ranges) >= 6 else float(ranges.mean())
        avg30 = float(ranges.mean())
        feats["range_compression"] = recent6 / max(avg30, 1e-9)
    else:
        feats["range_compression"] = 1.0

    # Time-of-day quartile within the NSE session.
    minutes = _ist_minute_of_day(df.index[entry_idx])
    session_len = SESSION_CLOSE_IST_MIN - SESSION_OPEN_IST_MIN
    if minutes < SESSION_OPEN_IST_MIN or minutes > SESSION_CLOSE_IST_MIN:
        feats["time_quartile"] = -1.0
    else:
        position = (minutes - SESSION_OPEN_IST_MIN) / max(session_len, 1)
        feats["time_quartile"] = float(min(3, int(position * 4)))

    # Volatility regime: current ATR vs longer-window average range.
    if n >= 60:
        avg_range_60 = float(
            (high[max(0, n - 60):n] - low[max(0, n - 60):n]).mean()
        )
        feats["vol_ratio_regime"] = float(atr_val / max(avg_range_60, 1e-9))
    else:
        feats["vol_ratio_regime"] = 1.0

    return feats


def _features_for_pool(df: pd.DataFrame, entry_idx: int, pool, atr_val: float,
                       symbol: str, use_extended: bool) -> Dict[str, float]:
    feats = _basic_features(df, entry_idx, pool, atr_val, symbol)
    if use_extended:
        feats.update(_extended_features(df, entry_idx, atr_val))
    return feats


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _entry_idx_at_touch(df: pd.DataFrame, touched_at) -> Optional[int]:
    if touched_at is None:
        return None
    ts = pd.Timestamp(touched_at)
    idx = int(df.index.searchsorted(ts, side="left"))
    if idx >= len(df) or idx < 0:
        return None
    return idx


def build_dataset(report, use_extended_features: bool,
                  atr_period: int = 14, verbose: bool = True) -> pd.DataFrame:
    """For each touched OOS pool × each grid config, compute features + label.

    Returns one row per (pool, config). Rows include feature columns plus
    ``stop_atr``, ``target_atr``, ``horizon_bars`` (as model inputs), the
    realized_r label, and bookkeeping columns (symbol, entry_ts, exit_reason).
    """
    all_rows: List[Dict] = []
    for symbol, ad in report.assets.items():
        df = ad.base_df
        if df is None or df.empty:
            continue
        atr_series = atr(df, atr_period).bfill()
        oos_pools = ad.walkforward.oos_pools
        oos_results = ad.walkforward.oos_results
        n_pool = 0
        for pool, result in zip(oos_pools, oos_results):
            entry_idx = _entry_idx_at_touch(df, result.touched_at)
            if entry_idx is None:
                continue
            entry_min = _ist_minute_of_day(df.index[entry_idx])
            if not (SESSION_OPEN_IST_MIN <= entry_min <= SESSION_CLOSE_IST_MIN):
                continue
            if entry_min > NO_NEW_ENTRY_AFTER_IST_MIN:
                continue
            atr_val = float(atr_series.iloc[entry_idx])
            entry_price = float(df["close"].iloc[entry_idx])
            feats = _features_for_pool(df, entry_idx, pool, atr_val, symbol,
                                        use_extended_features)
            side = "long" if pool.side == "low" else "short"
            for stop_atr, target_atr, horizon in GRID:
                r, reason, exit_idx = triple_barrier_label(
                    df, entry_idx, side, entry_price,
                    stop_atr=stop_atr, target_atr=target_atr,
                    horizon_bars=horizon, atr_value=atr_val,
                )
                row = dict(feats)
                row.update({
                    "symbol": symbol,
                    "pool_idx": int(result.pool_idx),
                    "entry_ts": str(df.index[entry_idx]),
                    "side_long": 1.0 if side == "long" else 0.0,
                    "stop_atr": float(stop_atr),
                    "target_atr": float(target_atr),
                    "horizon_bars": float(horizon),
                    "realized_r": float(r),
                    "exit_reason": reason,
                })
                all_rows.append(row)
            n_pool += 1
        if verbose:
            print(f"  [{symbol}] {n_pool} touched OOS pools × {len(GRID)} configs"
                  f" = {n_pool * len(GRID)} rows")
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Per-config aggregation (static grid sweep)
# ---------------------------------------------------------------------------

@dataclass
class ConfigRow:
    stop_atr: float
    target_atr: float
    horizon_bars: int
    trades: int
    win_rate: float
    mean_R: float
    median_R: float
    PF: float
    se: float
    ci95_lo: float
    ci95_hi: float


def _aggregate_per_config(dataset: pd.DataFrame) -> List[ConfigRow]:
    rows: List[ConfigRow] = []
    if dataset.empty:
        return rows
    for (s, t, h), g in dataset.groupby(["stop_atr", "target_atr", "horizon_bars"]):
        r_arr = g["realized_r"].to_numpy(dtype=float)
        r_arr = r_arr[np.isfinite(r_arr)]
        n = len(r_arr)
        if n == 0:
            continue
        wins = float((r_arr > 0).mean())
        mean_R = float(r_arr.mean())
        med_R = float(np.median(r_arr))
        se = float(r_arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        gross_win = float(r_arr[r_arr > 0].sum())
        gross_loss = float(-r_arr[r_arr < 0].sum())
        pf = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
        rows.append(ConfigRow(
            stop_atr=float(s), target_atr=float(t), horizon_bars=int(h),
            trades=n, win_rate=wins, mean_R=mean_R, median_R=med_R,
            PF=pf, se=se,
            ci95_lo=mean_R - 1.96 * se, ci95_hi=mean_R + 1.96 * se,
        ))
    return rows


def _print_config_table(rows: List[ConfigRow], title: str) -> None:
    print(f"\n================ {title} ================")
    if not rows:
        print("  (no data)")
        return
    print(f"  {'stop':>5} {'target':>6} {'horiz':>5} {'n':>6} "
          f"{'win':>6} {'mean_R':>7} {'med_R':>6} {'PF':>5} "
          f"{'95% CI mean_R':>18}")
    for r in sorted(rows, key=lambda x: -x.mean_R):
        pf_s = "inf" if not np.isfinite(r.PF) else f"{r.PF:.2f}"
        ci = f"[{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}]"
        print(f"  {r.stop_atr:>5.2f} {r.target_atr:>6.2f} {r.horizon_bars:>5} "
              f"{r.trades:>6} {r.win_rate:>5.1%} {r.mean_R:>+6.2f} "
              f"{r.median_R:>+5.2f} {pf_s:>5} {ci:>18}")


# ---------------------------------------------------------------------------
# Force model (LightGBM regressor)
# ---------------------------------------------------------------------------

@dataclass
class ForcedConfigRow:
    stop_atr: float
    target_atr: float
    horizon_bars: int
    val_n: int
    top_decile_n: int
    top_decile_realized_r: float
    se: float
    ci95_lo: float
    ci95_hi: float


def _train_force_model(dataset: pd.DataFrame, seed: int = 71
                       ) -> Tuple[Optional[object], pd.DataFrame, List[str]]:
    """Chronological train/val split + LightGBM regression.

    Returns (model, val_df_with_predictions, feature_names). val_df_with_predictions
    has all dataset columns PLUS ``predicted_r``."""
    import lightgbm as lgb

    if dataset.empty:
        return None, dataset, []

    # Chronological split on entry_ts. First 70% -> train; last 30% -> val.
    ts = pd.to_datetime(dataset["entry_ts"])
    order = np.argsort(ts.values, kind="stable")
    n = len(dataset)
    cut = int(round(n * 0.70))
    train_idx = order[:cut]
    val_idx = order[cut:]

    drop = {"symbol", "pool_idx", "entry_ts", "exit_reason", "realized_r"}
    feature_cols = [c for c in dataset.columns if c not in drop]

    X_tr = dataset.iloc[train_idx][feature_cols].fillna(0.0).to_numpy(dtype=float)
    y_tr = dataset.iloc[train_idx]["realized_r"].astype(float).to_numpy()
    X_val = dataset.iloc[val_idx][feature_cols].fillna(0.0).to_numpy(dtype=float)
    y_val = dataset.iloc[val_idx]["realized_r"].astype(float).to_numpy()

    if len(X_tr) == 0 or len(X_val) == 0:
        return None, dataset, feature_cols

    params = dict(
        objective="huber",         # robust to the heavy realized-R tails
        alpha=0.9,
        metric="l1",
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        min_data_in_leaf=80,
        feature_fraction=0.70,
        bagging_fraction=0.80,
        bagging_freq=5,
        lambda_l2=10.0,
        verbose=-1,
        seed=seed,
    )
    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr, feature_name=feature_cols)
    model = lgb.train(
        params, dtr, num_boost_round=400, valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    val_df = dataset.iloc[val_idx].copy()
    val_df["predicted_r"] = pred_val
    return model, val_df, feature_cols


def _evaluate_force_per_config(val_df: pd.DataFrame,
                                top_pct: float = 0.10
                                ) -> List[ForcedConfigRow]:
    rows: List[ForcedConfigRow] = []
    if val_df.empty or "predicted_r" not in val_df.columns:
        return rows
    for (s, t, h), g in val_df.groupby(["stop_atr", "target_atr", "horizon_bars"]):
        val_n = len(g)
        if val_n < 20:                                # too few to meaningfully decile
            continue
        top_n = max(1, int(np.ceil(val_n * top_pct)))
        top = g.sort_values("predicted_r", ascending=False).head(top_n)
        r_arr = top["realized_r"].to_numpy(dtype=float)
        r_arr = r_arr[np.isfinite(r_arr)]
        if len(r_arr) == 0:
            continue
        mean_R = float(r_arr.mean())
        se = float(r_arr.std(ddof=1) / math.sqrt(len(r_arr))) if len(r_arr) > 1 else 0.0
        rows.append(ForcedConfigRow(
            stop_atr=float(s), target_atr=float(t), horizon_bars=int(h),
            val_n=val_n, top_decile_n=len(r_arr),
            top_decile_realized_r=mean_R, se=se,
            ci95_lo=mean_R - 1.96 * se, ci95_hi=mean_R + 1.96 * se,
        ))
    return rows


def _print_force_table(rows: List[ForcedConfigRow], title: str) -> None:
    print(f"\n================ {title} ================")
    if not rows:
        print("  (no data)")
        return
    print(f"  {'stop':>5} {'target':>6} {'horiz':>5} {'val_n':>6} "
          f"{'topD_n':>7} {'topD_R':>7} {'95% CI':>18}")
    for r in sorted(rows, key=lambda x: -x.top_decile_realized_r):
        ci = f"[{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}]"
        print(f"  {r.stop_atr:>5.2f} {r.target_atr:>6.2f} {r.horizon_bars:>5} "
              f"{r.val_n:>6} {r.top_decile_n:>7} "
              f"{r.top_decile_realized_r:>+6.2f} {ci:>18}")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _verdict(static_rows: List[ConfigRow],
             forced_rows: List[ForcedConfigRow]) -> List[str]:
    out: List[str] = []
    static_positive = [r for r in static_rows if r.ci95_lo > 0]
    static_borderline = [r for r in static_rows
                          if r.mean_R > 0 and r.ci95_lo <= 0 < r.ci95_hi]

    out.append("STATIC GRID:")
    out.append(f"  Statistically positive configs (ci95_lo > 0): {len(static_positive)}")
    for r in sorted(static_positive, key=lambda x: -x.mean_R)[:5]:
        out.append(f"    + s={r.stop_atr} t={r.target_atr} h={r.horizon_bars}  "
                   f"n={r.trades} mean_R={r.mean_R:+.2f} "
                   f"CI=[{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}] PF={r.PF:.2f}")
    out.append(f"  Borderline (positive mean, CI crosses zero): {len(static_borderline)}")
    if static_rows:
        best = max(static_rows, key=lambda x: x.mean_R)
        out.append(f"  Best fixed config (by mean_R): "
                   f"s={best.stop_atr} t={best.target_atr} h={best.horizon_bars}  "
                   f"mean_R={best.mean_R:+.2f} CI=[{best.ci95_lo:+.2f},{best.ci95_hi:+.2f}]")

    forced_positive = [r for r in forced_rows if r.ci95_lo > 0]
    forced_borderline = [r for r in forced_rows
                          if r.top_decile_realized_r > 0
                          and r.ci95_lo <= 0 < r.ci95_hi]

    out.append("")
    out.append("FORCE MODEL (top-decile predicted R, val set):")
    out.append(f"  Statistically positive configs: {len(forced_positive)}")
    for r in sorted(forced_positive,
                    key=lambda x: -x.top_decile_realized_r)[:5]:
        out.append(f"    + s={r.stop_atr} t={r.target_atr} h={r.horizon_bars}  "
                   f"topD_n={r.top_decile_n} R={r.top_decile_realized_r:+.2f} "
                   f"CI=[{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}]")
    out.append(f"  Borderline: {len(forced_borderline)}")
    if forced_rows:
        best = max(forced_rows, key=lambda x: x.top_decile_realized_r)
        out.append(f"  Best learned cell: s={best.stop_atr} t={best.target_atr} "
                   f"h={best.horizon_bars}  R={best.top_decile_realized_r:+.2f} "
                   f"CI=[{best.ci95_lo:+.2f},{best.ci95_hi:+.2f}]")

    # Cross-comparison: does learned beat best fixed?
    if static_rows and forced_rows:
        best_static = max(static_rows, key=lambda x: x.mean_R).mean_R
        best_force = max(forced_rows, key=lambda x: x.top_decile_realized_r
                          ).top_decile_realized_r
        out.append("")
        out.append(f"Best fixed config mean_R:     {best_static:+.3f}")
        out.append(f"Best learned top-decile R:    {best_force:+.3f}")
        out.append(f"Learned lift over best fixed: {best_force - best_static:+.3f}R")
        if best_force > 0 and best_static <= 0:
            out.append("=> Learned policy turns a non-positive fixed config into "
                       "a positive top-decile subset. ALPHA EXISTS but requires "
                       "selective trading.")
        elif best_force > best_static > 0:
            out.append("=> Both fixed and learned positive; learned adds selection lift "
                       "on top of an already-positive fixed config.")
        elif best_force <= 0:
            out.append("=> Neither fixed nor learned finds positive R. The current pool "
                       "detection + features do not contain extractable edge under "
                       "MIS arithmetic. Try extended features (-use-extended-features), "
                       "different pool inputs, or alternative alphas.")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_bundle(model_dir: Path):
    p = model_dir / "multi_asset_report.pkl"
    if not p.exists():
        raise SystemExit(f"missing bundle: {p}. Run --mode train first.")
    with p.open("rb") as f:
        return pickle.load(f)


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="output_models/core25_latest")
    ap.add_argument("--use-extended-features", type=_str2bool, default=False,
                    help="If true, add volume/gap/range/regime/time features.")
    ap.add_argument("--out", default="output_audit/force_model")
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--top-decile-pct", type=float, default=0.10)
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[force-model] loading bundle: {model_dir/'multi_asset_report.pkl'}")
    print(f"[force-model] use_extended_features={args.use_extended_features}")
    report = _load_bundle(model_dir)
    print(f"[force-model] {len(report.assets)} assets in bundle")
    print(f"[force-model] grid size: {len(GRID)} configs "
          f"(stop x target x horizon)")
    print(f"[force-model] building dataset (one row per pool × config)...")

    dataset = build_dataset(report, use_extended_features=args.use_extended_features,
                             atr_period=args.atr_period, verbose=True)
    if dataset.empty:
        raise SystemExit("[force-model] dataset is empty — no touched OOS pools "
                          "matched MIS constraints.")
    print(f"[force-model] dataset: {len(dataset):,} (pool, config) rows")

    dataset_path = out_dir / "force_model_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    print(f"[force-model] wrote {dataset_path}")

    # Part 1: static grid sweep.
    static_rows = _aggregate_per_config(dataset)
    _print_config_table(static_rows, "PART 1: STATIC GRID SWEEP (per-config mean R)")
    static_df = pd.DataFrame([r.__dict__ for r in static_rows])
    static_path = out_dir / "static_grid_sweep.csv"
    static_df.to_csv(static_path, index=False)

    # Part 2: force model.
    print("\n[force-model] training LightGBM regressor "
          "(70/30 chronological val split)...")
    model, val_df, feature_cols = _train_force_model(dataset)
    if model is None:
        print("[force-model] training failed (no usable data).")
        return 1
    forced_rows = _evaluate_force_per_config(val_df, top_pct=args.top_decile_pct)
    _print_force_table(forced_rows,
                        f"PART 2: FORCE MODEL "
                        f"(top-{int(args.top_decile_pct*100)}% predicted R per config, val set)")
    forced_df = pd.DataFrame([r.__dict__ for r in forced_rows])
    forced_path = out_dir / "force_model_per_config.csv"
    forced_df.to_csv(forced_path, index=False)

    val_path = out_dir / "force_model_val_predictions.parquet"
    val_df.to_parquet(val_path, index=False)
    print(f"\n[force-model] wrote {static_path}")
    print(f"[force-model] wrote {forced_path}")
    print(f"[force-model] wrote {val_path}")

    print("\n================ VERDICT ================")
    for line in _verdict(static_rows, forced_rows):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
