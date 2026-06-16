"""Track 4: multi-asset walk-forward with unified Q + direction + proximity models, plus
cross-asset ranked trading plan.

Run:
    python examples/multi_asset_run.py \
        --symbols HDFCBANK.NS,ICICIBANK.NS,KOTAKBANK.NS,SBIN.NS,AXISBANK.NS \
        --period 60d --folds 5 --iters 60 --final-iters 120
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from liqpool import Config, plot_chart
from liqpool.costs import (
    ZerodhaEquityCostConfig,
    estimate_round_trip_charges,
    expected_trade_value,
    trade_levels,
)
from liqpool.data import ParquetProvider
from liqpool.execution_backtest import (
    build_execution_backtest_for_report,
    summarise_execution_by_direction,
)
from liqpool.execution_simulator_v2 import (
    ExecutionV2Config,
    build_execution_backtest_v2_for_report,
    compare_execution_summaries_v1_v2,
)
from liqpool.featurize import MultiAssetFeaturizer
from liqpool.feature_store import FeatureStore, post_touch_event_row
from liqpool.indicators import atr
from liqpool.leakage_audit import (
    build_leakage_audit_for_report,
    max_active_label_horizon,
)
from liqpool.multi_asset import (AssetData, run_multi_asset, print_multi_asset_summary,
                                 distance_bucket)
from liqpool.policy_labels import (
    build_policy_labels_for_report,
    summarise_policy_labels,
)
from liqpool.policy_model import PolicyOutcomeModelSuite, PolicyReturnModelSuite
from liqpool.pools import build_pools, project_to_base
from liqpool.products.brief_renderer import render_email
from liqpool.products.daily_brief import generate_brief
from liqpool.products.outcome_log import OutcomeLogWriter
from liqpool.sectors import (sector_of, compute_sector_metrics, sector_regime_signal,
                             sector_execution_filter,
                             sector_correlation_matrix, detect_rotation, per_sector_oos,
                             serialise_sector_intel)
from liqpool.tester import test_pools
from liqpool.timing import StateFeaturizer
from liqpool.universe import symbols_for_universe
from liqpool.stratified import _headline_factor


def _bundle_path(model_dir: str) -> Path:
    return Path(model_dir).expanduser() / "multi_asset_report.pkl"


def _next_trading_date_after_latest_bar(report) -> str:
    """Return the next weekday IST date after the latest asset bar."""
    latest_ist = None
    for ad in getattr(report, "assets", {}).values():
        base = getattr(ad, "base_df", None)
        if base is None or base.empty:
            continue
        ts = pd.Timestamp(base.index[-1])
        if ts.tzinfo is not None:
            ts_ist = ts.tz_convert("Asia/Kolkata")
        else:
            ts_ist = ts + pd.Timedelta(hours=5, minutes=30)
        if latest_ist is None or ts_ist > latest_ist:
            latest_ist = ts_ist
    if latest_ist is None:
        return pd.Timestamp.utcnow().tz_convert("Asia/Kolkata").strftime("%Y-%m-%d")
    target = latest_ist.normalize() + pd.Timedelta(days=1)
    while target.weekday() >= 5:
        target += pd.Timedelta(days=1)
    return target.strftime("%Y-%m-%d")


def _brief_bundle_label(model_dir: str) -> str:
    label = Path(model_dir).expanduser().name
    return label or "multi_asset_bundle"


def _save_model_bundle(report, model_dir: str, args, cfg: Config) -> None:
    model_path = _bundle_path(model_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(report, f)
    metadata = {
        "symbols": list(report.assets.keys()),
        "regularization_preset": cfg.regularization_preset,
        "validation_method": cfg.validation_method,
        "embargo_bars": cfg.embargo_bars,
        "requested_embargo_bars": getattr(args, "requested_embargo_bars", args.embargo_bars),
        "effective_embargo_bars": cfg.embargo_bars,
        "max_active_label_horizon": getattr(args, "max_active_label_horizon", None),
        "allow_short_embargo": getattr(args, "allow_short_embargo", False),
        "data_source": args.data_source,
        "data_dir": args.data_dir,
        "universe": args.universe,
        "feature_store_dir": args.feature_store_dir,
    }
    (model_path.parent / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"[model] saved bundle: {model_path}")


def _load_predict_report(model_dir: str, data_provider: ParquetProvider, cfg: Config):
    model_path = _bundle_path(model_dir)
    if not model_path.exists():
        raise SystemExit(f"missing model bundle: {model_path}. Run --mode train first.")
    with model_path.open("rb") as f:
        report = pickle.load(f)
    for attr, default in (
        ("reaction_model", None),
        ("reaction_model_report", []),
        ("reaction_model_calibration", []),
        ("reaction_feature_importance", []),
        ("policy_model", None),
        ("policy_model_report", []),
        ("policy_model_calibration", []),
        ("policy_model_feature_importance", []),
        ("policy_return_model", None),
        ("policy_return_model_report", []),
        ("policy_return_model_calibration", []),
        ("policy_return_model_feature_importance", []),
        # Phase 3B/3C learned dynamic gate. Older bundles predate this so
        # the default is None and the predict path falls back to the static
        # per-sector blend (the Phase 3E safety rail).
        ("unified_learned_gate", None),
        ("learned_gate_metrics", None),
    ):
        if not hasattr(report, attr):
            setattr(report, attr, default)
    if report.unified_ml is None or not report.unified_proximity:
        raise SystemExit(f"model bundle is incomplete: {model_path}")

    asset_dfs = {}
    refreshed_assets = {}
    for symbol, ad in report.assets.items():
        try:
            tf = data_provider.load_timeframes(
                symbol, cfg.base_interval, cfg.higher_tfs,
                period=cfg.period, start=cfg.start, end=cfg.end,
            )
        except Exception as e:
            print(f"  [{symbol}] SKIPPED in predict mode — parquet failed: {e}")
            continue
        best = ad.final_cfg
        pools = project_to_base(build_pools(tf, best), tf["base"].index)
        results = test_pools(tf["base"], pools, best)
        for p in pools:
            p.asset = symbol
        asset_dfs[symbol] = tf["base"]
        refreshed_assets[symbol] = AssetData(
            symbol=symbol,
            base_df=tf["base"],
            tf_data=tf,
            walkforward=ad.walkforward,
            final_cfg=best,
            final_pools=pools,
            final_results=results,
        )
    if not refreshed_assets:
        raise SystemExit("predict mode could not load any symbols from parquet")
    report.assets = refreshed_assets
    report.unified_featurizer = MultiAssetFeaturizer(asset_dfs)
    return report


def _reaction_prior_for_candidate(cand: dict, post_touch_metrics: dict,
                                  min_bucket_n: int) -> dict:
    """Empirical-Bayes post-touch prior for the candidate's reaction quality.

    Quality Q is a pre-touch prior. This function pulls it back toward the
    historical post-touch bucket rate so a liquidity magnet is not treated as
    a reversal trade just because touch probability is high.
    """
    overall = (post_touch_metrics or {}).get("overall", {}) or {}
    by = (post_touch_metrics or {}).get("by", {}) or {}
    global_rate = float(overall.get("strict_respect_rate", 0.0) or 0.0)
    global_n = int(overall.get("n", 0) or 0)
    prior_n = max(20, min_bucket_n)

    choices = [
        ("factor_type", cand.get("headline_factor")),
        ("tf_count", str(cand.get("tf_count"))),
        ("sector", cand.get("sector")),
    ]
    chosen_name = "overall"
    chosen_bucket = "overall"
    chosen = overall
    for group_name, bucket_name in choices:
        bucket = (by.get(group_name, {}) or {}).get(bucket_name)
        if bucket and int(bucket.get("n", 0) or 0) >= min_bucket_n:
            chosen_name = group_name
            chosen_bucket = str(bucket_name)
            chosen = bucket
            break

    n = int(chosen.get("n", 0) or 0)
    raw_rate = float(chosen.get("strict_respect_rate", 0.0) or 0.0)
    if n > 0:
        shrunk = ((raw_rate * n) + (global_rate * prior_n)) / (n + prior_n)
    else:
        shrunk = global_rate
    return {
        "bucket_group": chosen_name,
        "bucket": chosen_bucket,
        "n": n,
        "raw_strict_rate": raw_rate,
        "shrunk_strict_rate": float(shrunk),
        "global_strict_rate": global_rate,
        "global_n": global_n,
    }


def _model_health_warning(report) -> str:
    parts = []
    auc = getattr(report.unified_ml, "val_auc", None) if report.unified_ml else None
    if auc is not None and auc < 0.55:
        parts.append(f"quality validation AUC is weak ({auc:.3f})")
    pt = (report.post_touch_reaction_metrics or {}).get("overall", {})
    if pt:
        broken = float(pt.get("broken_strong_rate", 0.0) or 0.0)
        strict = float(pt.get("strict_respect_rate", 0.0) or 0.0)
        if broken > strict:
            parts.append(
                f"post-touch breaks dominate strict reactions "
                f"({broken:.1%} broken vs {strict:.1%} strict)"
            )
    return "; ".join(parts)


def _finite_or_none(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _candidate_predicted_r(candidates, return_suite, execution_mode: str):
    """Thin wrapper around :func:`policy_model.score_candidates_with_return_model`
    that maps the live-candidate dict shape onto the model's feature schema.

    Returns ``{id(cand): predicted_r}`` for each candidate the suite can score,
    or ``{}`` when no suite is loaded / no model exists for the mode."""
    from liqpool.policy_model import score_candidates_with_return_model

    if return_suite is None or not candidates:
        return {}
    payload = [{
        "key": id(cand),
        "mode": execution_mode,
        "symbol": cand.get("symbol"),
        "sector": cand.get("sector"),
        "side": cand.get("side"),
        "direction": cand.get("direction"),
        "direction_sign": int(cand.get("direction_sign") or 0),
        "pool_low": float(cand["pool"].price_low),
        "pool_high": float(cand["pool"].price_high),
        "pool_mid": float(cand["pool"].mid),
        "available_at": str(getattr(cand["pool"], "available_at", "")),
        "formed_at": str(getattr(cand["pool"], "formed_at", "")),
        "score": float(cand.get("q") or 0.0),
        "tf_count": int(cand.get("tf_count") or 0),
        "factor": cand.get("headline_factor") or "",
    } for cand in candidates]
    try:
        return score_candidates_with_return_model(payload, return_suite, execution_mode)
    except Exception as exc:
        print(f"  [r_policy] predict failed for mode={execution_mode}: {exc}")
        return {}


def _fmt_pct(value) -> str:
    value = _finite_or_none(value)
    return "n/a" if value is None else f"{value:.0%}"


def _fmt_price(value) -> str:
    value = _finite_or_none(value)
    return "n/a" if value is None else f"₹{value:.2f}"


TRACK_A_ALLOWED_SECTORS = {"AUTO", "PHARMA", "FMCG"}
TRACK_A_MIN_TOUCH = 0.75
TRACK_A_MIN_DIRECTION = 0.65
TRACK_A_MIN_DISTANCE_ATR = 3.0
TRACK_A_MAX_DISTANCE_ATR = 8.0
TRACK_A_TARGET_FRACTION = 1.0
TRACK_A_STOP_ATR_MULT = 2.0
TRACK_A_MAX_HOLD_BARS = 60
TRACK_A_OUTPUT_COLUMNS = [
    "symbol", "sector", "direction", "side", "current", "pool_low", "pool_high",
    "pool_mid", "entry_reference", "target", "stop", "target_fraction",
    "stop_atr_mult", "max_hold_bars", "distance_atr", "p_touch",
    "p_direction_to_pool", "p_up", "q", "headline_factor", "tf_count",
    "gross_rr", "net_target_r", "net_stop_r", "target_cost_per_share",
    "stop_cost_per_share", "pocket_score", "research_status", "note",
]


def _track_a_pretouch_setups(candidates, cost_cfg: ZerodhaEquityCostConfig,
                             quantity: int) -> list[dict]:
    """Research-only Track A pre-touch pocket from Phase 4 constrained validation.

    This is not wired into the old post-touch live gate. It surfaces the
    validated pocket as a separate watch panel:
    long-only, pool above spot, AUTO/PHARMA/FMCG, T>=0.75, D>=0.65, 3-8 ATR,
    target = pool lower boundary, stop = 2 ATR.
    """
    rows = []
    for c in candidates:
        p_direction_to_pool = c.get("p_direction_to_pool")
        if p_direction_to_pool is None:
            p_up = c.get("dir_p_up")
            if p_up is None:
                p_direction_to_pool = 0.0
            else:
                p_direction_to_pool = p_up if c["side"] == "above" else (1.0 - p_up)
        p_direction_to_pool = float(p_direction_to_pool or 0.0)

        if c["side"] != "above":
            continue
        if c.get("sector") not in TRACK_A_ALLOWED_SECTORS:
            continue
        if c.get("p_touch", 0.0) < TRACK_A_MIN_TOUCH:
            continue
        if p_direction_to_pool < TRACK_A_MIN_DIRECTION:
            continue
        if not (TRACK_A_MIN_DISTANCE_ATR <= c["dist_atr"] < TRACK_A_MAX_DISTANCE_ATR):
            continue

        pool = c["pool"]
        entry = float(c["current"])
        atr_value = max(float(c["atr_proxy"]), 1e-9)
        target = entry + TRACK_A_TARGET_FRACTION * (float(pool.price_low) - entry)
        stop = entry - TRACK_A_STOP_ATR_MULT * atr_value
        reward = target - entry
        risk = entry - stop
        if reward <= 0.0 or risk <= 0.0:
            continue

        target_cost = estimate_round_trip_charges(entry, target, quantity, cost_cfg)
        stop_cost = estimate_round_trip_charges(entry, stop, quantity, cost_cfg)
        net_target_r = (reward - target_cost["cost_per_share"]) / risk
        net_stop_r = -((risk + stop_cost["cost_per_share"]) / risk)
        pocket_score = c["p_touch"] * p_direction_to_pool * net_target_r

        rows.append({
            "symbol": c["symbol"],
            "sector": c["sector"],
            "direction": "UP",
            "side": c["side"],
            "current": entry,
            "pool_low": float(pool.price_low),
            "pool_high": float(pool.price_high),
            "pool_mid": float(pool.mid),
            "entry_reference": entry,
            "target": float(target),
            "stop": float(stop),
            "target_fraction": TRACK_A_TARGET_FRACTION,
            "stop_atr_mult": TRACK_A_STOP_ATR_MULT,
            "max_hold_bars": TRACK_A_MAX_HOLD_BARS,
            "distance_atr": float(c["dist_atr"]),
            "p_touch": float(c["p_touch"]),
            "p_direction_to_pool": p_direction_to_pool,
            "p_up": _finite_or_none(c.get("dir_p_up")),
            "q": float(c.get("q", 0.0)),
            "headline_factor": c.get("headline_factor"),
            "tf_count": c.get("tf_count"),
            "gross_rr": float(reward / risk),
            "net_target_r": float(net_target_r),
            "net_stop_r": float(net_stop_r),
            "target_cost_per_share": float(target_cost["cost_per_share"]),
            "stop_cost_per_share": float(stop_cost["cost_per_share"]),
            "pocket_score": float(pocket_score),
            "research_status": "PRETOUCH_RESEARCH_ONLY",
            "note": (
                "Phase 4 constrained pocket: long-only AUTO/PHARMA/FMCG; "
                "not live-approved until CPCV/null baselines pass"
            ),
        })

    return sorted(
        rows,
        key=lambda r: (r["pocket_score"], r["p_touch"], r["p_direction_to_pool"]),
        reverse=True,
    )


def _build_reaction_alerts(report, feature_bars: int, lookback_bars: int,
                           confirm_threshold: float,
                           break_risk_threshold: float):
    """Score recent touched pools with the Phase 3 post-touch model.

    This intentionally does not promote anything into a live trade. It is a
    confirmation layer for "price has reached the pool; now evaluate reaction
    candles".
    """
    suite = getattr(report, "reaction_model", None)
    if suite is None or not getattr(suite, "models", None):
        return [], "reaction model unavailable in this model bundle"

    event_rows = []
    for symbol, ad in report.assets.items():
        base = ad.base_df
        if base is None or base.empty:
            continue
        cfg_for_asset = ad.final_cfg
        now_ts = base.index[-1]
        atr_values = atr(base, cfg_for_asset.detect.atr_period).bfill().ffill()

        for pool_idx, (pool, result) in enumerate(zip(ad.final_pools, ad.final_results)):
            if result.touched_at is None:
                continue
            try:
                touch_ts = pd.Timestamp(result.touched_at)
            except Exception:
                continue
            if touch_ts > now_ts:
                continue

            touch_idx = int(base.index.searchsorted(touch_ts, side="left"))
            if touch_idx < 0 or touch_idx >= len(base):
                continue
            bars_since_touch = int(len(base) - 1 - touch_idx)
            if bars_since_touch < feature_bars:
                continue
            if bars_since_touch > lookback_bars:
                continue

            row = post_touch_event_row(
                symbol, "live", pool_idx, pool, result,
                df_base=base, cfg=cfg_for_asset, atr_values=atr_values,
                feature_bars=feature_bars,
            )
            row["now_ts"] = str(now_ts)
            row["bars_since_touch"] = bars_since_touch
            event_rows.append(row)

    if not event_rows:
        return [], (
            f"no touched pools with at least {feature_bars} confirmation bars "
            f"inside the last {lookback_bars} bars"
        )

    events = pd.DataFrame(event_rows)
    try:
        preds = suite.predict_event_frame(events)
    except Exception as exc:
        return [], f"reaction alert scoring failed: {exc}"
    scored = pd.concat([events.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    alerts = []
    for _, row in scored.iterrows():
        p_strict = _finite_or_none(row.get("p_strict_reaction"))
        p_reclaim = _finite_or_none(row.get("p_reclaim_success"))
        p_break = _finite_or_none(row.get("p_break_continuation"))
        p_reaction = _finite_or_none(row.get("p_reaction_model"))
        strict_for_gate = p_strict if p_strict is not None else (p_reaction or 0.0)
        break_for_gate = p_break if p_break is not None else 0.0
        reclaim_for_gate = p_reclaim if p_reclaim is not None else 0.0

        direction = "UP" if row.get("side") == "low" else "DOWN"
        if break_for_gate >= break_risk_threshold and break_for_gate >= strict_for_gate:
            action = "AVOID_BREAK_CONTINUATION"
            reason = (
                f"break risk {_fmt_pct(break_for_gate)} >= "
                f"{break_risk_threshold:.0%}"
            )
        elif strict_for_gate >= confirm_threshold and break_for_gate < break_risk_threshold:
            action = f"CONFIRM_{direction}"
            reason = (
                f"strict reaction {_fmt_pct(strict_for_gate)} >= "
                f"{confirm_threshold:.0%}"
            )
        elif reclaim_for_gate >= confirm_threshold and break_for_gate < break_risk_threshold:
            action = f"RECLAIM_WATCH_{direction}"
            reason = (
                f"reclaim probability {_fmt_pct(reclaim_for_gate)} >= "
                f"{confirm_threshold:.0%}"
            )
        else:
            action = "WATCH_REACTION"
            reason = "confirmation model is not strong enough yet"

        tfs = str(row.get("tfs", ""))
        tf_count = len([x for x in tfs.split("+") if x]) if tfs else None
        alerts.append({
            "symbol": row.get("symbol"),
            "sector": row.get("sector"),
            "direction": direction,
            "side": "below" if row.get("side") == "low" else "above",
            "pool_low": _finite_or_none(row.get("price_low")),
            "pool_high": _finite_or_none(row.get("price_high")),
            "mid": _finite_or_none(row.get("mid")),
            "score": _finite_or_none(row.get("score")),
            "headline_factor": row.get("headline_factor"),
            "tf_bucket": row.get("tf_bucket"),
            "tf_count": tf_count,
            "touched_at": row.get("touched_at"),
            "now_ts": row.get("now_ts"),
            "bars_since_touch": int(row.get("bars_since_touch", 0)),
            "observed_outcome_so_far": row.get("outcome"),
            "reaction_label_so_far": row.get("reaction_label"),
            "p_strict_reaction": p_strict,
            "p_reclaim_success": p_reclaim,
            "p_break_continuation": p_break,
            "p_reaction_model": p_reaction,
            "pt_close_back_inside_6": _finite_or_none(row.get("pt_close_back_inside_6")),
            "pt_n_closes_inside_6": _finite_or_none(row.get("pt_n_closes_inside_6")),
            "pt_n_closes_through_6": _finite_or_none(row.get("pt_n_closes_through_6")),
            "pt_max_reaction_6_atr": _finite_or_none(row.get("pt_max_reaction_6_atr")),
            "pt_max_close_through_6_atr": _finite_or_none(
                row.get("pt_max_close_through_6_atr")
            ),
            "pt_wick_rejection_ratio": _finite_or_none(row.get("pt_wick_rejection_ratio")),
            "pt_touch_bar_volume_ratio_20": _finite_or_none(
                row.get("pt_touch_bar_volume_ratio_20")
            ),
            "action": action,
            "reason": reason,
        })

    alerts.sort(
        key=lambda x: (
            x["action"] == "AVOID_BREAK_CONTINUATION",
            -(x.get("p_reaction_model") or x.get("p_strict_reaction") or 0.0),
            x.get("p_break_continuation") or 0.0,
            x["bars_since_touch"],
        )
    )
    return alerts, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols",
                    default=None,
                    help="comma-separated NSE symbols")
    ap.add_argument("--universe", default="custom", choices=("custom", "core25"),
                    help="Named universe. core25 is the Mac-safe local development basket.")
    ap.add_argument("--base", default="5m")
    ap.add_argument("--period", default="60d")
    ap.add_argument("--tfs", default="15min,60min,180min,1D,1W")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.6, dest="train_frac")
    ap.add_argument("--iters", type=int, default=60,
                    help="optimizer iterations per fold (lowered vs single-asset since we have "
                         "5x assets)")
    ap.add_argument("--final-iters", type=int, default=120, dest="final_iters")
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--min-score", type=float, default=2.0, dest="min_score")
    ap.add_argument("--min-train-days", type=int, default=10, dest="min_train_days")
    ap.add_argument("--regularization-preset", default="default",
                    choices=("default", "conservative_finml"))
    ap.add_argument("--data-source", default="yfinance",
                    choices=("yfinance", "parquet"),
                    help="Market data source. parquet mode never calls yfinance.")
    ap.add_argument("--data-dir", default="",
                    help="Directory containing all_5m/all_15m/... parquet files")
    ap.add_argument("--mode", default="train", choices=("train", "predict"),
                    help="train fits/saves models; predict loads saved models and reads latest parquet")
    ap.add_argument("--model-dir", default="output_models/latest")
    ap.add_argument("--asset-workers", type=int, default=4,
                    help="Number of symbols to process in parallel during train "
                         "mode (4-5 is the sweet spot on a 16GB machine; use 1 "
                         "to disable parallelism)")
    ap.add_argument("--checkpoint-dir", default="output_checkpoints",
                    help="Per-symbol checkpoint directory for train mode")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse completed per-symbol checkpoints from --checkpoint-dir")
    ap.add_argument("--feature-store-dir", default="",
                    help=("Partitioned parquet feature-store root. If omitted with "
                          "--universe core25 train mode, defaults to output_feature_store/core25."))
    ap.add_argument("--embargo-bars", type=int, default=78)
    ap.add_argument("--allow-short-embargo", action="store_true",
                    help=("Debug/legacy only: do not lift --embargo-bars to cover the "
                          "maximum active label horizon"))
    ap.add_argument("--gate-q", type=float, default=0.70)
    ap.add_argument("--gate-t-today", type=float, default=0.50, dest="gate_t_today")
    ap.add_argument("--gate-min-distance-atr", type=float, default=0.5,
                    dest="gate_min_distance_atr")
    ap.add_argument("--gate-max-distance-atr", type=float, default=12.0,
                    dest="gate_max_distance_atr")
    ap.add_argument("--gate-min-bucket-n", type=int, default=30, dest="gate_min_bucket_n")
    ap.add_argument("--gate-min-post-touch-strict", type=float, default=0.45,
                    dest="gate_min_post_touch_strict",
                    help="Minimum historical post-touch strict reaction rate for live trades")
    ap.add_argument("--execution-mode", default="reclaim_confirmed",
                    choices=("blind_limit", "touch_confirmed", "reclaim_confirmed",
                             "displacement_confirmed"),
                    help="Live execution style. Default waits for reclaim/confirmation after touch.")
    ap.add_argument("--cost-product", default="intraday",
                    choices=("intraday", "delivery"),
                    help="Zerodha equity cost profile used for net expectancy")
    ap.add_argument("--slippage-bps", type=float, default=1.0,
                    help="Assumed slippage in bps per side")
    ap.add_argument("--cost-quantity", type=int, default=1,
                    help="Legacy v1 quantity used when notional sizing is disabled")
    ap.add_argument("--execution-notional-inr", type=float, default=100_000.0,
                    help="Per-trade notional used by V2 execution/policy labels. Set 0 to use --cost-quantity.")
    ap.add_argument("--skip-execution-backtest", action="store_true",
                    help="Skip Phase 2B OOS execution-mode backtest artifacts")
    ap.add_argument("--execution-backtest-split", default="oos", choices=("oos", "train"),
                    help="Historical split used for execution-mode profitability reports")
    ap.add_argument("--run-execution-backtest-v2", action="store_true",
                    help="Run Phase 4 execution simulator v2 artifacts")
    ap.add_argument("--fill-policy", default="neutral",
                    choices=("generous", "neutral", "conservative"),
                    help="Phase 4 v2 fill policy for adverse-selection-aware fills")
    ap.add_argument("--use-1m-resolution", action="store_true",
                    help="Use raw 1-minute bars to resolve v2 stop/target path")
    ap.add_argument("--raw-1m-dir", default="",
                    help=("Directory containing per-symbol raw 1m parquet. If omitted with "
                          "--data-dir .../resampled, uses sibling raw_1m when present."))
    ap.add_argument("--slippage-model", default="state_dependent",
                    choices=("state_dependent", "flat"),
                    help="Phase 4 v2 slippage model")
    ap.add_argument("--v2-base-slippage-bps", type=float, default=2.0,
                    help="Base bps/side used by execution simulator v2")
    ap.add_argument("--execution-exchange", default="NSE", choices=("NSE", "BSE"),
                    help="Exchange transaction charge profile for v2 itemized costs")
    ap.add_argument("--skip-leakage-audit", action="store_true",
                    help="Skip Phase 3C leakage audit artifacts")
    ap.add_argument("--allow-leakage-errors", action="store_true",
                    help="Write leakage artifacts but do not fail the run on ERROR rows")
    ap.add_argument("--replay-audit-samples", type=int, default=3,
                    help="Sampled detector replay checks per symbol for Phase 3C leakage audit")
    ap.add_argument("--replay-audit-severity", default="warn", choices=("warn", "error"),
                    help="Severity assigned to sampled replay misses")
    ap.add_argument("--skip-policy-labels", action="store_true",
                    help="Skip Phase 3C execution-policy label artifacts")
    ap.add_argument("--policy-label-split", default="oos", choices=("oos", "train", "final"),
                    help="Pool split used for execution-policy triple-barrier labels")
    ap.add_argument("--policy-execution-version", default="v2", choices=("v1", "v2"),
                    help="Execution simulator used for policy labels. v2 matches MIS/notional/slippage audits.")
    ap.add_argument("--skip-policy-model", action="store_true",
                    help="Skip Phase 3D policy outcome model diagnostics")
    ap.add_argument("--policy-model-min-trades", type=int, default=80,
                    help="Minimum generated trades per execution mode to train a policy outcome model")
    ap.add_argument("--skip-policy-return-model", action="store_true",
                    help="Skip the R1 policy RETURN model (Huber regression on realized net R)")
    ap.add_argument("--policy-return-min-trades", type=int, default=80,
                    help="Minimum generated trades per mode to train a policy return regressor")
    ap.add_argument("--r-policy-threshold", type=float, default=None,
                    help="If set, in predict mode demote candidates with policy-return-model "
                         "predicted R below this threshold from TRADEABLE to WATCH_ONLY. "
                         "Default (None) = report only, no gating.")
    ap.add_argument("--reaction-feature-bars", type=int, default=6,
                    help="Bars after first touch used by Phase 3 reaction confirmation features")
    ap.add_argument("--reaction-alert-lookback-bars", type=int, default=78,
                    help="Only score pools touched within this many base bars")
    ap.add_argument("--reaction-confirm-threshold", type=float, default=0.60,
                    help="Minimum post-touch model probability for confirmation alerts")
    ap.add_argument("--reaction-break-risk-threshold", type=float, default=0.60,
                    help="Break-continuation probability that blocks confirmation alerts")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    if args.data_source == "parquet" and not args.data_dir:
        raise SystemExit("--data-source parquet requires --data-dir")
    if args.mode == "predict" and args.data_source != "parquet":
        raise SystemExit("--mode predict currently requires --data-source parquet")

    if args.feature_store_dir == "" and args.mode == "train" and args.universe == "core25":
        args.feature_store_dir = "output_feature_store/core25"

    if args.universe != "custom":
        universe_symbols = symbols_for_universe(args.universe)
        if args.symbols:
            print(f"[universe] --symbols overrides --universe {args.universe}")
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = universe_symbols
    else:
        default_symbols = ("HDFCBANK.NS,ICICIBANK.NS,SBIN.NS,AXISBANK.NS,"
                           "TCS.NS,INFY.NS,HCLTECH.NS,"
                           "MARUTI.NS,TATAMOTORS.NS,"
                           "HINDUNILVR.NS")
        symbols = [s.strip() for s in (args.symbols or default_symbols).split(",") if s.strip()]
    if not symbols:
        raise SystemExit("--symbols cannot be empty")

    args.requested_embargo_bars = int(args.embargo_bars)
    provisional_cfg = Config(
        symbol=symbols[0],
        base_interval=args.base,
        period=args.period,
        higher_tfs=args.tfs.split(","),
        test_horizon_bars=args.horizon,
        embargo_bars=args.requested_embargo_bars,
        regularization_preset=args.regularization_preset,
    )
    args.max_active_label_horizon = max_active_label_horizon(provisional_cfg)
    effective_embargo_bars = args.requested_embargo_bars
    if not args.allow_short_embargo:
        effective_embargo_bars = max(
            args.requested_embargo_bars,
            int(args.max_active_label_horizon),
        )
    args.effective_embargo_bars = int(effective_embargo_bars)

    cfg = Config(
        symbol=symbols[0],          # cfg.symbol is just informational here
        base_interval=args.base,
        period=args.period,
        higher_tfs=args.tfs.split(","),
        test_horizon_bars=args.horizon,
        opt_iterations=args.iters,
        opt_explore_frac=0.35,
        opt_seed=11,
        min_pool_score=args.min_score,
        embargo_bars=args.effective_embargo_bars,
        regularization_preset=args.regularization_preset,
    )

    print(f"=== MULTI-ASSET RUN ({len(symbols)} assets) ===")
    print(f"  symbols:    {symbols}")
    print(f"  base/tfs:   {args.base} + {cfg.higher_tfs}")
    print(f"  folds:      {args.folds}  iters/fold: {args.iters}  final iters: {args.final_iters}")
    print(f"  data source:{args.data_source}"
          f"{' @ ' + args.data_dir if args.data_source == 'parquet' else ''}")
    print(f"  universe:   {args.universe}"
          f"{'  feature-store: ' + args.feature_store_dir if args.feature_store_dir else ''}")
    print(f"  embargo:    requested={args.requested_embargo_bars} "
          f"effective={cfg.embargo_bars} "
          f"max_horizon={args.max_active_label_horizon}"
          f"{' (short override)' if args.allow_short_embargo else ''}")
    if args.mode == "train":
        print(f"  workers:    {args.asset_workers}  checkpoints: {args.checkpoint_dir}"
              f"{' (resume)' if args.resume else ''}")

    def prog(symbol, step):
        print(f"  [{symbol}]  {step}")

    data_provider = ParquetProvider(args.data_dir) if args.data_source == "parquet" else None
    if args.mode == "predict":
        report = _load_predict_report(args.model_dir, data_provider, cfg)
    else:
        report = run_multi_asset(symbols, cfg,
                                  n_folds=args.folds, train_frac=args.train_frac,
                                  iters_per_fold=args.iters, final_iters=args.final_iters,
                                  min_train_days=args.min_train_days,
                                  data_provider=data_provider,
                                  asset_workers=args.asset_workers,
                                  checkpoint_dir=args.checkpoint_dir,
                                  resume=args.resume,
                                  feature_store_dir=args.feature_store_dir or None,
                                  progress=prog)
        _save_model_bundle(report, args.model_dir, args, cfg)

    if report.skipped_symbols:
        print("\n⚠  WARNING — assets dropped from run:")
        for sym in report.skipped_symbols:
            print(f"   - {sym}: {report.skip_reasons.get(sym, 'unknown')}")
        print(f"   (running with {len(report.assets)} of {len(symbols)} requested assets)")

    print_multi_asset_summary(report)

    # ---- Cross-asset prediction at "now" ----
    print("\n================ TODAY'S CROSS-ASSET PLAN ================")
    if report.unified_ml is None or not report.unified_proximity:
        print("  (unified models unavailable — see warnings above)")
        return

    PROX_HORIZONS = sorted(report.unified_proximity.keys())
    primary_h = PROX_HORIZONS[0]
    DIR_ALIGN_MARGIN = 0.10
    TRADEABLE_T_TODAY = 0.05
    TRADEABLE_Q = 0.55
    GATE_Q = args.gate_q
    GATE_T_TODAY = args.gate_t_today
    PRACTICAL_MIN_ATR = args.gate_min_distance_atr
    PRACTICAL_MAX_ATR = args.gate_max_distance_atr
    MIN_BUCKET_N = args.gate_min_bucket_n
    MIN_POST_TOUCH_STRICT = args.gate_min_post_touch_strict
    cost_cfg = ZerodhaEquityCostConfig(
        product=args.cost_product,
        slippage_bps_per_side=args.slippage_bps,
    )
    model_health_warning = _model_health_warning(report)

    asset_dfs_for_sectors = {sym: ad.base_df for sym, ad in report.assets.items()}
    sec_metrics = compute_sector_metrics(asset_dfs_for_sectors)
    rotation = detect_rotation(sec_metrics)
    blended_bucket_n = {}
    if report.unified_oos_audit is not None:
        for row in report.unified_oos_audit.distance_bucket_metrics.get("blended", []):
            blended_bucket_n[row["bucket"]] = int(row["n"])

    candidates = []        # cross-asset list
    per_asset_summary = {} # symbol -> {current, atr, direction, top_q, top_t_today}

    for symbol, ad in report.assets.items():
        base = ad.base_df
        pools = ad.final_pools
        results = ad.final_results
        if not pools:
            continue
        atr_proxy = float((base["high"] - base["low"]).rolling(14).mean().iloc[-1])
        if atr_proxy <= 0:
            atr_proxy = 1.0
        current = float(base["close"].iloc[-1])
        now_ts = base.index[-1]

        # Per-asset current state (live pools = available + not-broken + not-touched)
        live = [p for p, r in zip(pools, results)
                if p.available_at <= now_ts
                and not r.is_break
                and (r.touched_at is None or r.touched_at > now_ts)]
        state_feat = StateFeaturizer(base)
        current_state = state_feat.features_at(len(base) - 1, live)

        dir_p_up = None
        if report.unified_direction is not None:
            dir_p_up = report.unified_direction.predict_state(current_state)

        def direction_tag(side_str: str, p_up=dir_p_up) -> str:
            if p_up is None:
                return "DIR_NA"
            if p_up >= 0.5 + DIR_ALIGN_MARGIN:
                return "DIR_ALIGN" if side_str == "above" else "DIR_FIGHT"
            if p_up <= 0.5 - DIR_ALIGN_MARGIN:
                return "DIR_ALIGN" if side_str == "below" else "DIR_FIGHT"
            return "DIR_NEUTRAL"

        # Iterate top-quality untouched pools both sides
        active = [(p, r) for p, r in zip(pools, results)
                   if not r.is_break and r.outcome != "horizon_insufficient"
                   and (r.touched_at is None or r.touched_at > now_ts)]
        # Take a compact but wider set per side. The original live gate still
        # prints only top rows, but the Track A pre-touch research panel needs
        # enough active pools to avoid missing a high-proximity journey setup.
        live_pool_side_limit = 20
        above = sorted([(p, r) for p, r in active if p.price_low > current],
                        key=lambda x: -x[0].score)[:live_pool_side_limit]
        below = sorted([(p, r) for p, r in active if p.price_high < current],
                        key=lambda x: -x[0].score)[:live_pool_side_limit]

        asset_candidates = []   # candidates for THIS asset, used to populate per_asset_summary

        for side_str, pool_list in (("above", above), ("below", below)):
            for p, r in pool_list:
                dist = (p.mid - current) if side_str == "above" else (current - p.mid)
                dist_atr = dist / max(atr_proxy, 1e-9)
                X = report.unified_featurizer.transform_batch([p])
                q_components = {}
                if hasattr(report.unified_ml, "predict_components"):
                    comp = report.unified_ml.predict_components(X, [p]).iloc[0].to_dict()
                    q = float(comp["blended_q"])
                    q_components = comp
                    # Phase 3D — if a Phase 3C learned dynamic gate was fit on
                    # the OOS audit, override the static per-sector blend with
                    # the per-pool learned blend. Falls through silently when
                    # the gate isn't loaded (Phase 3E safety rail).
                    gate = getattr(report, "unified_learned_gate", None)
                    if gate is not None:
                        sec_q_raw = comp.get("sector_q")
                        sec_q = float(sec_q_raw) if sec_q_raw is not None and not pd.isna(sec_q_raw) else None
                        learned_w, learned_q = gate.blend_one(
                            global_q=float(comp.get("global_q", q)),
                            sector_q=sec_q,
                            sector=str(comp.get("sector", sector_of(symbol))),
                            asset=str(comp.get("asset", symbol)),
                            pool_score=float(p.score),
                            pool_width=float(p.price_high - p.price_low),
                            n_tfs=int(len(set(p.tfs))),
                            distance_atr=float(dist_atr),
                        )
                        q = learned_q
                        q_components["learned_gate_weight"] = learned_w
                        q_components["learned_blended_q"] = learned_q
                else:
                    q = float(report.unified_ml.predict(X, pools=[p])[0])
                t_by_h = {}
                for h in PROX_HORIZONS:
                    pm = report.unified_proximity[h]
                    t_by_h[h] = pm.predict_one(p, dist_atr, side_str, current_state, q)
                p_direction_to_pool = None
                if dir_p_up is not None:
                    p_direction_to_pool = (
                        float(dir_p_up) if side_str == "above" else 1.0 - float(dir_p_up)
                    )
                tag = direction_tag(side_str)
                bonus = 1.15 if tag == "DIR_ALIGN" else (0.85 if tag == "DIR_FIGHT" else 1.0)
                t_today = t_by_h.get(primary_h, 0.0)
                sec = sector_of(symbol)
                trade_side_word = "buy" if side_str == "below" else "sell"
                sec_decision = sector_execution_filter(
                    sec_metrics, sec, trade_side_word, q,
                    min_override_q=max(TRADEABLE_Q + 0.08, 0.65),
                )
                dist_bucket = distance_bucket(float(dist_atr))
                headline_factor = _headline_factor(p)
                tf_count = len(set(p.tfs))
                levels = trade_levels(p.price_low, p.price_high, atr_proxy, side_str)
                skeleton = {
                    "sector": sec,
                    "headline_factor": headline_factor,
                    "tf_count": tf_count,
                }
                reaction_prior = _reaction_prior_for_candidate(
                    skeleton, report.post_touch_reaction_metrics, MIN_BUCKET_N,
                )
                p_reaction = min(float(q), float(reaction_prior["shrunk_strict_rate"]))
                if tag == "DIR_FIGHT":
                    p_reaction *= 0.85
                p_reaction *= max(0.0, min(1.0, float(sec_decision["multiplier"])))
                p_reaction = max(0.0, min(1.0, p_reaction))
                trade_ev = expected_trade_value(
                    side=side_str,
                    entry=levels["entry"],
                    stop=levels["stop"],
                    target=levels["target"],
                    p_touch=t_today,
                    p_reaction=p_reaction,
                    quantity=args.cost_quantity,
                    cfg=cost_cfg,
                )
                ev = trade_ev["net_expectancy_r"]
                cand = {
                    "symbol": symbol, "pool": p, "result": r, "side": side_str,
                    "dist_atr": dist_atr, "q": q, "p_respect": q, "t_by_h": t_by_h,
                    "p_touch": t_today, "p_reaction": p_reaction,
                    "p_trade": trade_ev["p_trade"],
                    "dir_tag": tag, "ev": ev,
                    "gross_ev_legacy": q * t_today * bonus * sec_decision["multiplier"],
                    "direction": trade_ev["direction"],
                    "direction_sign": trade_ev["direction_sign"],
                    "net_expectancy_per_share": trade_ev["net_expectancy_per_share"],
                    "net_expectancy_r": trade_ev["net_expectancy_r"],
                    "directional_net_expectancy_r": trade_ev["directional_net_expectancy_r"],
                    "conditional_net_expectancy_per_share": (
                        trade_ev["conditional_net_expectancy_per_share"]
                    ),
                    "expected_cost_per_share": trade_ev["expected_cost_per_share"],
                    "entry": levels["entry"],
                    "stop": levels["stop"],
                    "target": levels["target"],
                    "current": current, "atr_proxy": atr_proxy,
                    "dir_p_up": dir_p_up,
                    "p_direction_to_pool": p_direction_to_pool,
                    "sector": sec,
                    "sector_mult": sec_decision["multiplier"],
                    "sector_alignment": sec_decision["alignment"],
                    "sector_regime": sec_decision["regime"],
                    "sector_block_reason": sec_decision["block_reason"],
                    "sector_allow_trade": sec_decision["allow_trade"],
                    "distance_bucket": dist_bucket,
                    "historical_bucket_n": blended_bucket_n.get(dist_bucket, 0),
                    "headline_factor": headline_factor,
                    "tf_count": tf_count,
                    "post_touch_bucket_group": reaction_prior["bucket_group"],
                    "post_touch_bucket": reaction_prior["bucket"],
                    "post_touch_bucket_n": reaction_prior["n"],
                    "post_touch_strict_rate": reaction_prior["shrunk_strict_rate"],
                    "post_touch_raw_strict_rate": reaction_prior["raw_strict_rate"],
                    "global_q": q_components.get("global_q"),
                    "sector_q": q_components.get("sector_q"),
                    "sector_weight": q_components.get("gate_weight", 0.0),
                    "moe_fallback_reason": q_components.get("fallback_reason", ""),
                }
                candidates.append(cand)
                asset_candidates.append(cand)

        # Summarise THIS asset's situation (used in the per-asset dashboard table).
        top_q = max((c["q"] for c in asset_candidates), default=0.0)
        top_t = max((c["t_by_h"].get(primary_h, 0.0) for c in asset_candidates), default=0.0)
        top_ev = max((c["ev"] for c in asset_candidates), default=0.0)
        per_asset_summary[symbol] = {
            "current": current, "atr_proxy": atr_proxy,
            "dir_p_up": dir_p_up,
            "n_candidates": len(asset_candidates),
            "top_q": top_q, "top_t_today": top_t, "top_ev": top_ev,
        }

    candidates.sort(key=lambda c: -c["ev"])
    track_a_pretouch = _track_a_pretouch_setups(
        candidates,
        cost_cfg=cost_cfg,
        quantity=args.cost_quantity,
    )

    def live_gate_reasons(c):
        reasons = []
        if c["q"] < GATE_Q:
            reasons.append(f"Q {c['q']:.0%} < {GATE_Q:.0%}")
        t_today = c["t_by_h"].get(primary_h, 0.0)
        if t_today < GATE_T_TODAY:
            reasons.append(f"T_today {t_today:.0%} < {GATE_T_TODAY:.0%}")
        if c["dir_tag"] != "DIR_ALIGN":
            reasons.append(f"direction {c['dir_tag']} not DIR_ALIGN")
        if not c.get("sector_allow_trade", True) or c.get("sector_mult", 1.0) <= 0:
            reasons.append(c.get("sector_block_reason") or "sector flow negative")
        if not (PRACTICAL_MIN_ATR <= c["dist_atr"] <= PRACTICAL_MAX_ATR):
            reasons.append(f"distance {c['dist_atr']:.1f}ATR outside "
                           f"{PRACTICAL_MIN_ATR:.1f}-{PRACTICAL_MAX_ATR:.1f}")
        if c.get("historical_bucket_n", 0) < MIN_BUCKET_N:
            reasons.append(f"bucket n={c.get('historical_bucket_n', 0)} < {MIN_BUCKET_N}")
        if c.get("post_touch_bucket_n", 0) < MIN_BUCKET_N:
            reasons.append(f"post-touch bucket n={c.get('post_touch_bucket_n', 0)} < "
                           f"{MIN_BUCKET_N}")
        if c.get("post_touch_strict_rate", 0.0) < MIN_POST_TOUCH_STRICT:
            reasons.append(f"post-touch strict {c.get('post_touch_strict_rate', 0.0):.0%} "
                           f"< {MIN_POST_TOUCH_STRICT:.0%}")
        if c.get("headline_factor") == "REJ" and c.get("post_touch_strict_rate", 0.0) < 0.50:
            reasons.append("REJ factor not validated above 50% post-touch strict")
        if c.get("net_expectancy_per_share", 0.0) <= 0.0:
            reasons.append(f"net expectancy ₹{c.get('net_expectancy_per_share', 0.0):.2f} <= 0 "
                           "after costs")
        return reasons

    # R1: score every live candidate with the policy_return_model regressor (if
    # present in the bundle). Always populated for display/CSV even when no
    # threshold is set; only applied as a gate reason when --r-policy-threshold
    # is explicitly enabled.
    r_policy_suite = getattr(report, "policy_return_model", None)
    r_policy_preds = _candidate_predicted_r(
        candidates, r_policy_suite, args.execution_mode,
    )
    r_policy_threshold = args.r_policy_threshold
    r_policy_active = r_policy_threshold is not None and bool(r_policy_preds)
    for c in candidates:
        pred = r_policy_preds.get(id(c))
        c["predicted_r"] = pred  # may be None — caller treats as "no signal"

    gate_decisions = []
    tradeable = []
    watch_only = []
    rejected = []
    for c in candidates:
        reasons = live_gate_reasons(c)
        if r_policy_active:
            pred_r = c.get("predicted_r")
            if pred_r is not None and pred_r < r_policy_threshold:
                reasons.append(
                    f"predicted R {pred_r:+.2f} < r-policy threshold "
                    f"{r_policy_threshold:+.2f}"
                )
        if not reasons:
            decision_name = "TRADEABLE"
        elif c["q"] >= TRADEABLE_Q or c["t_by_h"].get(primary_h, 0.0) >= 0.20:
            decision_name = "WATCH_ONLY"
        else:
            decision_name = "REJECTED_WITH_REASON"
        decision = {
            "symbol": c["symbol"],
            "sector": c["sector"],
            "side": c["side"],
            "pool_low": c["pool"].price_low,
            "pool_high": c["pool"].price_high,
            "q": c["q"],
            "p_respect": c["p_respect"],
            "t_today": c["t_by_h"].get(primary_h, 0.0),
            "p_touch": c["p_touch"],
            "p_reaction": c["p_reaction"],
            "p_trade": c["p_trade"],
            "dir_tag": c["dir_tag"],
            "direction": c["direction"],
            "direction_sign": c["direction_sign"],
            "distance_atr": c["dist_atr"],
            "distance_bucket": c["distance_bucket"],
            "historical_bucket_n": c.get("historical_bucket_n", 0),
            "headline_factor": c.get("headline_factor"),
            "tf_count": c.get("tf_count"),
            "post_touch_bucket_group": c.get("post_touch_bucket_group"),
            "post_touch_bucket": c.get("post_touch_bucket"),
            "post_touch_bucket_n": c.get("post_touch_bucket_n", 0),
            "post_touch_strict_rate": c.get("post_touch_strict_rate", 0.0),
            "sector_weight": c.get("sector_weight", 0.0),
            "ev": c["ev"],
            "net_expectancy_r": c["net_expectancy_r"],
            "directional_net_expectancy_r": c["directional_net_expectancy_r"],
            "net_expectancy_per_share": c["net_expectancy_per_share"],
            "expected_cost_per_share": c["expected_cost_per_share"],
            "execution_mode": args.execution_mode,
            "entry": c["entry"],
            "stop": c["stop"],
            "target": c["target"],
            "decision": decision_name,
            "reasons": reasons,
            "policy_return_model_predicted_r": c.get("predicted_r"),
        }
        gate_decisions.append(decision)
        c["gate_reasons"] = reasons
        if decision_name == "TRADEABLE":
            tradeable.append(c)
        elif decision_name == "WATCH_ONLY":
            watch_only.append(c)
        else:
            rejected.append(c)
    watchlist = watch_only + rejected
    reaction_alerts, reaction_alert_note = _build_reaction_alerts(
        report,
        feature_bars=args.reaction_feature_bars,
        lookback_bars=args.reaction_alert_lookback_bars,
        confirm_threshold=args.reaction_confirm_threshold,
        break_risk_threshold=args.reaction_break_risk_threshold,
    )

    # Day verdict — explicitly track the "best observed T" pool too so the user can see
    # a high-T pool that failed the Q threshold (the Q/T anti-correlation case).
    best_ev = tradeable[0]["ev"] if tradeable else 0.0
    max_t_today_cand = max(candidates, key=lambda c: c["t_by_h"].get(primary_h, 0.0)) \
                       if candidates else None
    max_t_today = (max_t_today_cand["t_by_h"].get(primary_h, 0.0)
                   if max_t_today_cand else 0.0)
    max_t_2d = (max((c["t_by_h"].get(PROX_HORIZONS[1], 0.0) for c in candidates), default=0.0)
                if len(PROX_HORIZONS) > 1 else 0.0)

    if tradeable and best_ev >= 0.20:
        verdict = "TRADE_HIGH_CONFIDENCE"
        verdict_why = f"top setup net expectancy={best_ev:+.2f}R after costs"
    elif tradeable:
        verdict = "TRADE_CAUTIOUS"
        verdict_why = f"top setup net expectancy={best_ev:+.2f}R after costs"
    elif max_t_today >= 0.30 and max_t_today_cand is not None:
        verdict = "WATCH"
        verdict_why = (f"touch likely, reaction quality weak — watch only "
                       f"(P_touch={max_t_today:.0%} on {max_t_today_cand['symbol']}, "
                       f"P_reaction={max_t_today_cand['p_reaction']:.0%})")
    elif max_t_today >= 0.02 or max_t_2d >= 0.20:
        verdict = "WATCH"
        verdict_why = (f"no setup today; max T_today={max_t_today:.0%}, "
                       f"max T_2d={max_t_2d:.0%} (1-2 day setup possible)")
    else:
        verdict = "NO_TRADE"
        verdict_why = "no actionable pool across the basket"

    print(f"  Assets in basket:           {', '.join(report.assets.keys())}")
    print(f"  Best gated net EV today:    {best_ev:+.2f}R  "
          f"({'qualified' if tradeable else 'none qualified'})")
    if max_t_today_cand is not None:
        sym_t = max_t_today_cand["symbol"]
        q_t = max_t_today_cand["q"]
        print(f"  Best observed T_today:      {max_t_today:.1%}  "
              f"on [{sym_t}] @ ₹{max_t_today_cand['pool'].mid:.2f}  "
              f"(Q={q_t:.0%}, dir={max_t_today_cand['dir_tag']}, "
              f"sector={max_t_today_cand.get('sector_alignment', 'NEUTRAL')})")
    print(f"  Max T_2d (any pool):        {max_t_2d:.1%}")
    print(f"  VERDICT:                    {verdict}")
    print(f"  Why:                        {verdict_why}")
    if model_health_warning:
        print(f"  Model health warning:       {model_health_warning}")
    print(f"  Live gate:                  Q≥{GATE_Q:.0%}, T_today≥{GATE_T_TODAY:.0%}, "
          f"DIR_ALIGN, {PRACTICAL_MIN_ATR:.1f}-{PRACTICAL_MAX_ATR:.1f}ATR, "
          f"bucket n≥{MIN_BUCKET_N}, post-touch strict≥{MIN_POST_TOUCH_STRICT:.0%}, "
          f"net EV>0")
    if r_policy_suite is not None:
        scored = sum(1 for c in candidates if c.get("predicted_r") is not None)
        if scored > 0:
            pred_vals = [c["predicted_r"] for c in candidates
                         if c.get("predicted_r") is not None]
            mean_pred = sum(pred_vals) / max(1, len(pred_vals))
            line = (f"  R policy (mode={args.execution_mode}): "
                    f"scored {scored}/{len(candidates)} candidates, "
                    f"mean predicted R={mean_pred:+.2f}")
            if r_policy_active:
                cut = sum(1 for c in candidates
                          if (c.get("predicted_r") is not None
                              and c["predicted_r"] < r_policy_threshold))
                line += (f"  |  gate ON @ threshold={r_policy_threshold:+.2f}; "
                         f"would demote {cut}/{scored}")
            else:
                line += "  |  gate OFF (set --r-policy-threshold to enable)"
            print(line)
        else:
            print(f"  R policy: no model for mode={args.execution_mode} "
                  f"(suite has: {sorted(getattr(r_policy_suite, 'models', {}).keys())})")
    print("----------------------------------------------------------")

    # Per-asset compact dashboard
    print("\n[per-asset dashboard]")
    print(f"  {'symbol':<14} {'price':>10} {'P(up)':>7} {'bias':>10} "
          f"{'top_Q':>7} {'top_T_today':>13} {'top_EV_R':>9}")
    for sym, s in per_asset_summary.items():
        p_up = s["dir_p_up"]
        if p_up is None:
            p_up_str = "n/a"
            bias_str = "n/a"
        else:
            p_up_str = f"{p_up:.0%}"
            if p_up >= 0.5 + DIR_ALIGN_MARGIN:
                bias_str = "STRONG UP"
            elif p_up <= 0.5 - DIR_ALIGN_MARGIN:
                bias_str = "STRONG DOWN"
            elif p_up >= 0.5:
                bias_str = "weak up"
            else:
                bias_str = "weak down"
        print(f"  {sym:<14} ₹{s['current']:>8.2f}  {p_up_str:>6}  {bias_str:>10} "
              f"{s['top_q']:>6.0%} {s['top_t_today']:>12.0%} {s['top_ev']:>+8.2f}")

    if tradeable:
        top = tradeable[0]
        p = top["pool"]
        srcs = sorted({c.source.split('@')[0] for c in p.contributors})
        side_label = "BELOW (buy)" if top["side"] == "below" else "ABOVE (sell)"
        print(f">>> BEST SETUP TODAY  ({top['symbol']}) <<<")
        print(f"  Pool ₹{p.price_low:.2f}-{p.price_high:.2f}  mid ₹{p.mid:.2f}  [{side_label}]")
        print(f"  Distance: {top['dist_atr']:.2f} ATRs from current ₹{top['current']:.2f}  "
              f"({top['dir_tag']})")
        print(f"  P_touch={top['p_touch']:.1%}   P_respect={top['p_respect']:.1%}   "
              f"P_reaction={top['p_reaction']:.1%}   P_trade={top['p_trade']:.1%}")
        print(f"  Net EV={top['net_expectancy_r']:+.2f}R  "
              f"(₹{top['net_expectancy_per_share']:+.2f}/share after costs)")
        print(f"  Direction: {top['direction']}  "
              f"direction-coded EV={top['directional_net_expectancy_r']:+.2f}R")
        print(f"  Drivers: {', '.join(srcs)}   |  TFs: {'+'.join(p.tfs)}")
        print(f"  Execution: {args.execution_mode}; arm alert, wait for touch + confirmation")
        side_word = "BUY" if top["side"] == "below" else "SELL"
        print(f"  Plan: {side_word} trigger near ₹{top['entry']:.2f}, "
              f"stop ₹{top['stop']:.2f}, target ₹{top['target']:.2f}")
    else:
        print(">>> NO TRADEABLE SETUP IN BASKET TODAY <<<")
    print("==========================================================")

    print(f"\n--- TRADEABLE  (post-touch + net-expectancy gate passed) ---")
    if not tradeable:
        print("  (none — all pools across basket too far or low-quality)")
    else:
        for rank, c in enumerate(tradeable[:10], 1):
            p = c["pool"]
            t_strs = "  ".join(
                f"T_h{h}={c['t_by_h'].get(h, 0.0):.1%}" for h in PROX_HORIZONS
            )
            side_lbl = "BELOW" if c["side"] == "below" else "ABOVE"
            print(f"  #{rank} [{c['symbol']:<14}] ₹{p.price_low:.2f}-{p.price_high:.2f}  "
                  f"[{side_lbl}, {c['dist_atr']:.1f}ATR, {c['dir_tag']}, "
                  f"{c.get('sector_alignment', 'NEUTRAL')}]   "
                  f"P_respect={c['p_respect']:.1%} P_reaction={c['p_reaction']:.1%} "
                  f"P_trade={c['p_trade']:.1%}  {t_strs}  "
                  f"netEV={c['net_expectancy_r']:+.2f}R "
                  f"dirEV={c['directional_net_expectancy_r']:+.2f}R")

    print(f"\n--- WATCH_ONLY  (interesting, but failed at least one strict gate) ---")
    if not watch_only:
        print("  (none)")
    else:
        for c in watch_only[:10]:
            p = c["pool"]
            side_lbl = "BELOW" if c["side"] == "below" else "ABOVE"
            print(f"  [{c['symbol']:<14}] ₹{p.price_low:.2f}-{p.price_high:.2f} "
                  f"[{side_lbl}, {c['dist_atr']:.1f}ATR, {c['dir_tag']}] "
                  f"P_touch={c['p_touch']:.1%} P_respect={c['p_respect']:.1%} "
                  f"P_reaction={c['p_reaction']:.1%} "
                  f"watch: {'; '.join(c['gate_reasons'])}")

    print(f"\n--- REJECTED_WITH_REASON  (top 10 by EV) ---")
    if not rejected:
        print("  (none)")
    else:
        for c in rejected[:10]:
            p = c["pool"]
            print(f"  [{c['symbol']:<14}] ₹{p.price_low:.2f}-{p.price_high:.2f} "
                  f"P_touch={c['p_touch']:.1%} P_reaction={c['p_reaction']:.1%} "
                  f"netEV={c['net_expectancy_r']:+.2f}R "
                  f"dirEV={c['directional_net_expectancy_r']:+.2f}R rejected: "
                  f"{'; '.join(c['gate_reasons'])}")

    # IMMINENT TOUCH list: pools likely to be touched today regardless of quality. Surfaces
    # the high-T-but-low-Q pools that the Q-sorted watch list hides. These are warnings rather
    # than trades — "price is coming here, decide what to do when it arrives".
    IMMINENT_THRESHOLD = 0.20
    imminent = sorted(
        [c for c in candidates if c["t_by_h"].get(primary_h, 0.0) >= IMMINENT_THRESHOLD],
        key=lambda x: -x["t_by_h"].get(primary_h, 0.0),
    )
    print(f"\n--- IMMINENT TOUCH  (T_today ≥ {IMMINENT_THRESHOLD:.0%}, any Q) ---")
    if not imminent:
        print(f"  (no pool likely to be touched today across the basket)")
    else:
        for c in imminent[:10]:
            p = c["pool"]
            side_lbl = "BELOW" if c["side"] == "below" else "ABOVE"
            t_strs = "  ".join(f"T_h{h}={c['t_by_h'].get(h, 0.0):.1%}" for h in PROX_HORIZONS)
            q_tag = "Q-OK" if c["q"] >= TRADEABLE_Q else "Q-LOW"
            print(f"  [{c['symbol']:<14}] ₹{p.price_low:.2f}-{p.price_high:.2f}  "
                  f"[{side_lbl}, {c['dist_atr']:.1f}ATR, {c['dir_tag']}, {q_tag}]   "
                  f"P_respect={c['p_respect']:.1%} P_reaction={c['p_reaction']:.1%}  "
                  f"{t_strs}")

    print("\n--- TRACK A PRE-TOUCH POCKET  "
          "(research-only: long AUTO/PHARMA/FMCG, T≥75%, D≥65%, 3-8ATR) ---")
    if not track_a_pretouch:
        print("  (none in the current active-pool scan)")
    else:
        for row in track_a_pretouch[:10]:
            print(f"  [{row['symbol']:<14}] {row['sector']:<6} "
                  f"entry≈₹{row['entry_reference']:.2f} → target ₹{row['target']:.2f} "
                  f"stop ₹{row['stop']:.2f}  "
                  f"dist={row['distance_atr']:.1f}ATR "
                  f"T={row['p_touch']:.1%} D={row['p_direction_to_pool']:.1%} "
                  f"Q={row['q']:.1%} netTarget={row['net_target_r']:+.2f}R "
                  f"score={row['pocket_score']:+.2f}")
        print("  Note: not live-approved; this pocket still needs CPCV/null baselines.")

    # Display block — show the highest-confidence subset, but make the selection
    # explicit so a reader cannot mistake the top-10 truncation for "every
    # alert is ~98%". The list is sorted descending by reaction probability
    # upstream; printing only [:10] without context implied saturation.
    print("\n--- POST-TOUCH REACTION CONFIRMATIONS  "
          f"(last {args.reaction_alert_lookback_bars} bars) ---")
    if not reaction_alerts:
        print(f"  ({reaction_alert_note or 'none'})")
    else:
        confirm_thr = float(args.reaction_confirm_threshold)
        strict_p = [a.get("p_strict_reaction") for a in reaction_alerts
                    if a.get("p_strict_reaction") is not None]
        n_total = len(reaction_alerts)
        n_below = sum(1 for a in reaction_alerts
                      if (a.get("p_strict_reaction") or 0.0) < confirm_thr)
        shown = reaction_alerts[:10]
        n_more = max(0, n_total - len(shown))
        if strict_p:
            arr = np.asarray(strict_p, dtype=float)
            print(f"  full pool: {n_total} alerts  strict p median={arr.mean():.2f} "
                  f"p25={float(np.quantile(arr, 0.25)):.2f} "
                  f"p75={float(np.quantile(arr, 0.75)):.2f}  "
                  f"{n_below} below confirm threshold {confirm_thr:.0%}")
        print(f"  showing top {len(shown)} by reaction probability "
              f"(highest-confidence subset — sort is descending by design):")
        for a in shown:
            side_lbl = "BELOW" if a["side"] == "below" else "ABOVE"
            price_zone = f"{_fmt_price(a['pool_low'])}-{_fmt_price(a['pool_high'])}"
            print(f"  [{a['symbol']:<14}] {price_zone} "
                  f"[{side_lbl}, {a['direction']}, touched {a['bars_since_touch']}b ago] "
                  f"strict={_fmt_pct(a.get('p_strict_reaction'))} "
                  f"reclaim={_fmt_pct(a.get('p_reclaim_success'))} "
                  f"break={_fmt_pct(a.get('p_break_continuation'))} "
                  f"→ {a['action']}: {a['reason']}")
        if n_more:
            print(f"  ... {n_more} additional alerts not shown "
                  f"(see reaction_alerts.csv for the full set)")
        print("  Note: these are post-touch confirmation alerts, not automatic entries.")

    print(f"\n--- QUALITY WATCH  (top 10 by Q, T_today below tradeable threshold) ---")
    for c in sorted([c for c in watchlist if c["t_by_h"].get(primary_h, 0.0) < TRADEABLE_T_TODAY],
                     key=lambda x: -x["q"])[:10]:
        p = c["pool"]
        side_lbl = "BELOW" if c["side"] == "below" else "ABOVE"
        t_strs = "  ".join(f"T_h{h}={c['t_by_h'].get(h, 0.0):.1%}" for h in PROX_HORIZONS)
        print(f"  [{c['symbol']:<14}] ₹{p.price_low:.2f}-{p.price_high:.2f}  "
              f"[{side_lbl}, {c['dist_atr']:.1f}ATR, {c['dir_tag']}]   "
              f"P_respect={c['p_respect']:.1%} P_reaction={c['p_reaction']:.1%}  "
              f"{t_strs}")

    blocked_by_sector = [c for c in candidates if c.get("sector_block_reason")]
    if blocked_by_sector:
        print(f"\n--- SECTOR-GATED WATCHLIST  (top 8 blocked by strong sector regime) ---")
        for c in sorted(blocked_by_sector, key=lambda x: -x["ev"])[:8]:
            p = c["pool"]
            print(f"  [{c['symbol']:<14}] {c['sector']:<9} ₹{p.price_low:.2f}-{p.price_high:.2f}  "
                  f"Q={c['q']:.1%} T_today={c['t_by_h'].get(primary_h, 0.0):.1%}  "
                  f"{c['sector_block_reason']}")

    # ---- Sector intelligence (Track 4.5 — basket-aware, money-flow rotation) ----
    print("\n================ SECTOR INTELLIGENCE ================")

    sec_oos = per_sector_oos(report.assets and
                              [p for ad in report.assets.values() for p in ad.walkforward.oos_pools],
                              [r for ad in report.assets.values() for r in ad.walkforward.oos_results])
    corr_df = sector_correlation_matrix(asset_dfs_for_sectors)

    # Per-sector pooled OOS + recent momentum
    if sec_metrics or sec_oos:
        print("\n[per-sector basket]")
        print(f"  {'sector':<10} {'symbols':<28} {'oos_resp':>9} {'strict':>8} "
              f"{'regime':>8} {'conv':>5} {'ret_5d':>8} {'ret_20d':>9} "
              f"{'ret_60d':>9} {'vol_20d':>9}")
        all_secs = sorted(set(sec_metrics.keys()) | set(sec_oos.keys()))
        for sec in all_secs:
            m = sec_metrics.get(sec, {})
            o = sec_oos.get(sec, {})
            syms = ",".join([s.replace(".NS", "") for s in m.get("symbols", o.get("symbols", []))])
            if len(syms) > 26:
                syms = syms[:24] + ".."
            resp = o.get("respect")
            strict = o.get("strict_respect")
            resp_s = f"{resp:>8.1%}" if resp is not None else "    n/a"
            strict_s = f"{strict:>7.1%}" if strict is not None else "    n/a"
            ret5 = m.get("ret_5d", 0.0)
            ret20 = m.get("ret_20d", 0.0)
            ret60 = m.get("ret_60d", 0.0)
            vol = m.get("vol_20d_annualised", 0.0)
            regime = sector_regime_signal(sec_metrics, sec)
            regime_dir = regime.get("direction", "neutral") if regime else "neutral"
            conviction = regime.get("conviction", 0.0) if regime else 0.0
            print(f"  {sec:<10} {syms:<28} {resp_s} {strict_s} "
                  f"{regime_dir[:8]:>8} {conviction:>4.2f} "
                  f"{ret5:>+7.1%} {ret20:>+8.1%} {ret60:>+8.1%} {vol:>8.1%}")

    # Rotation signal narrative
    if rotation and rotation.get("narrative"):
        print(f"\n[money-flow rotation]")
        print(f"  {rotation['narrative']}")
        if rotation.get("rotation_in"):
            print(f"  → bias TOWARD pools in: {', '.join(rotation['rotation_in'])}")
        if rotation.get("rotation_out"):
            print(f"  → bias AGAINST pools in: {', '.join(rotation['rotation_out'])}")

    # Correlation matrix (only if >=2 sectors)
    if corr_df is not None and not corr_df.empty and len(corr_df) >= 2:
        print(f"\n[sector correlation matrix — daily returns over period]")
        # Compact print
        col_w = 9
        secs = list(corr_df.columns)
        header = " " * 12 + "".join(f"{s[:7]:>{col_w}}" for s in secs)
        print("  " + header)
        for s in secs:
            row = f"  {s[:10]:<10}" + " " * 2
            for s2 in secs:
                row += f"{corr_df.loc[s, s2]:>{col_w}.2f}"
            print(row)

    # ---- Phase 2B: execution-mode profitability backtest ----
    execution_trades = pd.DataFrame()
    execution_summary = pd.DataFrame()
    execution_by_direction = pd.DataFrame()
    execution_v2_trades = pd.DataFrame()
    execution_v2_summary = pd.DataFrame()
    execution_v2_delta = pd.DataFrame()
    if not args.skip_execution_backtest:
        print("\n================ EXECUTION BACKTEST ================")
        try:
            execution_trades, execution_summary = build_execution_backtest_for_report(
                report,
                cfg,
                cost_cfg,
                quantity=args.cost_quantity,
                split=args.execution_backtest_split,
            )
            execution_by_direction = summarise_execution_by_direction(execution_trades)
            if execution_summary.empty:
                print("  (no historical execution trades generated)")
            else:
                print(f"  Split: {args.execution_backtest_split.upper()}  "
                      f"Costs: Zerodha {args.cost_product}, "
                      f"{args.slippage_bps:.1f}bps/side slippage")
                print(f"  {'mode':<18} {'trades':>8} {'win':>7} "
                      f"{'net_exp':>10} {'net_R':>8} {'up/dn':>11} "
                      f"{'up_R':>8} {'dn_R':>8} {'PF':>7} {'maxDD':>10}")
                for _, row in execution_summary.iterrows():
                    pf = row.get("profit_factor", 0.0)
                    pf_s = "inf" if not np.isfinite(pf) else f"{pf:.2f}"
                    print(f"  {row['mode']:<18} {int(row['trades']):>8} "
                          f"{row['win_rate']:>6.1%} "
                          f"₹{row['net_expectancy']:>8.2f} "
                          f"{row['net_expectancy_r']:>+7.2f} "
                          f"{int(row.get('up_trades', 0)):>5}/"
                          f"{int(row.get('down_trades', 0)):<5} "
                          f"{row.get('up_net_expectancy_r', 0.0):>+7.2f} "
                          f"{row.get('down_net_expectancy_r', 0.0):>+7.2f} "
                          f"{pf_s:>7} ₹{row['max_drawdown']:>9.0f}")
                print("  Note: net_R is profit/loss. Direction columns split UP longs "
                      "from DOWN shorts.")
        except Exception as e:
            print(f"  [execution_backtest] skipped: {e}")

    resolved_raw_1m_dir = args.raw_1m_dir
    if not resolved_raw_1m_dir and args.data_dir:
        data_dir = Path(args.data_dir).expanduser()
        sibling_raw = data_dir.parent / "raw_1m"
        if sibling_raw.exists():
            resolved_raw_1m_dir = str(sibling_raw)
    execution_notional_inr = (
        float(args.execution_notional_inr)
        if args.execution_notional_inr and args.execution_notional_inr > 0
        else None
    )

    # ---- Phase 4: execution simulator v2 ----
    if args.run_execution_backtest_v2 and not args.skip_execution_backtest:
        print("\n================ EXECUTION BACKTEST V2 ================")
        try:
            v2_cfg = ExecutionV2Config(
                fill_policy=args.fill_policy,
                use_1m_resolution=args.use_1m_resolution,
                slippage_model=args.slippage_model,
                base_slippage_bps=args.v2_base_slippage_bps,
                exchange=args.execution_exchange,
                quantity=args.cost_quantity,
                notional_inr=execution_notional_inr,
            )
            execution_v2_trades, execution_v2_summary = build_execution_backtest_v2_for_report(
                report,
                cfg,
                v2_cfg,
                raw_1m_dir=resolved_raw_1m_dir,
                split=args.execution_backtest_split,
            )
            execution_v2_delta = compare_execution_summaries_v1_v2(
                execution_summary, execution_v2_summary,
            )
            if execution_v2_summary.empty:
                print("  (no v2 historical execution trades generated)")
            else:
                resolution = "1m" if args.use_1m_resolution else "5m fallback"
                print(f"  Split: {args.execution_backtest_split.upper()}  "
                      f"fill={args.fill_policy} slippage={args.slippage_model} "
                      f"resolution={resolution}")
                print(f"  {'mode':<18} {'trades':>8} {'win':>7} "
                      f"{'net_R':>8} {'PF':>7} {'maxDD':>10}")
                for _, row in execution_v2_summary.iterrows():
                    pf = row.get("profit_factor", 0.0)
                    pf_s = "inf" if not np.isfinite(pf) else f"{pf:.2f}"
                    print(f"  {row['mode']:<18} {int(row['trades']):>8} "
                          f"{row['win_rate']:>6.1%} "
                          f"{row['net_expectancy_r']:>+7.2f} "
                          f"{pf_s:>7} ₹{row['max_drawdown']:>9.0f}")
                if args.use_1m_resolution and not resolved_raw_1m_dir:
                    print("  Warning: --use-1m-resolution requested, but no raw_1m directory "
                          "was found; v2 used 5m fallback where needed.")
        except Exception as e:
            print(f"  [execution_backtest_v2] skipped: {e}")

    # ---- Phase 3C: leakage probes and execution-policy labels ----
    leakage_summary = {}
    leakage_issues = pd.DataFrame()
    if not args.skip_leakage_audit:
        print("\n================ PHASE 3C CONSISTENCY ================")
        try:
            leakage_summary, leakage_issues = build_leakage_audit_for_report(
                report,
                cfg,
                requested_embargo_bars=args.requested_embargo_bars,
                allow_short_embargo=args.allow_short_embargo,
                replay_audit_samples=args.replay_audit_samples,
                replay_severity=args.replay_audit_severity.upper(),
            )
            counts = leakage_summary.get("severity_counts", {})
            embargo = leakage_summary.get("embargo", {})
            replay = leakage_summary.get("replay_audit", {})
            print(f"  Status: {leakage_summary.get('status', 'UNKNOWN')}  "
                  f"errors={counts.get('ERROR', 0)} warnings={counts.get('WARN', 0)}")
            print(f"  Embargo: requested={embargo.get('requested_embargo_bars', args.requested_embargo_bars)} "
                  f"effective={embargo.get('effective_embargo_bars', cfg.embargo_bars)} bars; "
                  f"max active horizon={embargo.get('max_active_horizon', 'n/a')} bars; "
                  f"covers={embargo.get('embargo_covers_max_horizon', False)}")
            print(f"  Replay audit: checked={replay.get('checked', 0)} "
                  f"misses={replay.get('misses', 0)} "
                  f"skipped={replay.get('skipped', 0)} "
                  f"severity={replay.get('severity', args.replay_audit_severity.upper())}")
            if not leakage_issues.empty:
                # Phase 3C produces hundreds of warnings on healthy bundles
                # because boundary bars near session end legitimately have
                # less-than-full-horizon label windows. Printing 5 raw rows
                # + "475 more in artifact" is signal-poor and crowds the
                # real summary. Group by (severity, check, message) and
                # print one count-per-group line instead — the full row
                # detail still goes to leakage_issues.csv.
                grouped = (
                    leakage_issues
                    .groupby(["severity", "check", "message"])
                    .size().reset_index(name="count")
                    .sort_values(["severity", "count"],
                                  ascending=[True, False])
                )
                print(f"  Issue summary ({len(leakage_issues)} rows total, "
                       f"{len(grouped)} unique issue classes):")
                for _, row in grouped.head(10).iterrows():
                    print(f"  [{row['severity']}] {row['check']}: "
                           f"{row['count']} × {row['message']}")
                if len(grouped) > 10:
                    print(f"  ... {len(grouped) - 10} more issue classes "
                           f"in leakage_issues.csv")
        except Exception as e:
            leakage_summary = {"status": "ERROR", "error": str(e)}
            print(f"  [leakage_audit] skipped: {e}")

    policy_labels = pd.DataFrame()
    policy_label_summary = pd.DataFrame()
    policy_feature_store_rows = 0
    policy_feature_store_root = None
    policy_v2_cfg = ExecutionV2Config(
        fill_policy=args.fill_policy,
        use_1m_resolution=args.use_1m_resolution,
        slippage_model=args.slippage_model,
        base_slippage_bps=args.v2_base_slippage_bps,
        exchange=args.execution_exchange,
        quantity=args.cost_quantity,
        notional_inr=execution_notional_inr,
    )
    if not args.skip_policy_labels:
        print("\n================ POLICY LABELS ================")
        try:
            policy_labels = build_policy_labels_for_report(
                report,
                cfg,
                cost_cfg,
                quantity=args.cost_quantity,
                execution_version=args.policy_execution_version,
                v2_cfg=policy_v2_cfg,
                raw_1m_dir=resolved_raw_1m_dir,
                split=args.policy_label_split,
            )
            policy_label_summary = summarise_policy_labels(policy_labels)
            if policy_label_summary.empty:
                print("  (no policy labels generated)")
            else:
                sizing_s = (
                    f"notional ₹{execution_notional_inr:,.0f}"
                    if execution_notional_inr is not None else f"quantity {args.cost_quantity}"
                )
                print(f"  Split: {args.policy_label_split.upper()}  "
                      f"exec={args.policy_execution_version.upper()} {sizing_s}  "
                      "target = net return under explicit execution policy")
                print(f"  {'mode':<22} {'rows':>8} {'trades':>8} {'trade%':>7} "
                      f"{'win':>7} {'mean_R':>8} {'PF':>7} {'no_trade':>9}")
                for _, row in policy_label_summary.iterrows():
                    pf = row.get("profit_factor", 0.0)
                    pf_s = "inf" if not np.isfinite(pf) else f"{pf:.2f}"
                    print(f"  {row['mode']:<22} {int(row['pool_policy_rows']):>8} "
                          f"{int(row['generated_trades']):>8} "
                          f"{row['trade_rate']:>6.1%} {row['win_rate']:>6.1%} "
                          f"{row['mean_return_r']:>+7.2f} {pf_s:>7} "
                          f"{row['no_trade_rate']:>8.1%}")
                best = policy_label_summary.iloc[0]
                worst = policy_label_summary.iloc[-1]
                print(f"  Best/Worst mean_R: {best['mode']}={best['mean_return_r']:+.2f} / "
                      f"{worst['mode']}={worst['mean_return_r']:+.2f}")
                print("  Note: no-trade rows are kept so gates cannot hide selection bias.")
                if args.mode == "train" and not policy_labels.empty:
                    root = (
                        (report.feature_store_stats or {}).get("root")
                        if report.feature_store_stats else None
                    ) or (args.feature_store_dir or None)
                    if root:
                        fs = FeatureStore(root)
                        policy_feature_store_rows = fs.write_policy_labels(policy_labels)
                        policy_feature_store_root = str(fs.root)
                        print(f"  Feature store: wrote {policy_feature_store_rows} "
                              f"policy label rows -> {policy_feature_store_root}/policy_labels")
        except Exception as e:
            print(f"  [policy_labels] skipped: {e}")

    # Shared between the binary policy model and the R1 return regressor so we
    # don't rebuild policy labels twice.
    train_policy_labels = None
    oos_policy_labels = None

    def _ensure_policy_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
        nonlocal train_policy_labels, oos_policy_labels
        if train_policy_labels is None:
            train_policy_labels = (
                policy_labels if args.policy_label_split == "train" else
                build_policy_labels_for_report(
                    report, cfg, cost_cfg,
                    quantity=args.cost_quantity,
                    execution_version=args.policy_execution_version,
                    v2_cfg=policy_v2_cfg,
                    raw_1m_dir=resolved_raw_1m_dir,
                    split="train",
                )
            )
        if oos_policy_labels is None:
            oos_policy_labels = (
                policy_labels if args.policy_label_split == "oos" else
                build_policy_labels_for_report(
                    report, cfg, cost_cfg,
                    quantity=args.cost_quantity,
                    execution_version=args.policy_execution_version,
                    v2_cfg=policy_v2_cfg,
                    raw_1m_dir=resolved_raw_1m_dir,
                    split="oos",
                )
            )
        return train_policy_labels, oos_policy_labels

    policy_model_report = pd.DataFrame()
    policy_model_calibration = pd.DataFrame()
    policy_model_importance = pd.DataFrame()
    policy_model_predictions = pd.DataFrame()
    if (
        args.mode == "train"
        and not args.skip_policy_labels
        and not args.skip_policy_model
    ):
        print("\n================ POLICY OUTCOME MODEL ================")
        try:
            train_policy_labels, oos_policy_labels = _ensure_policy_labels()
            suite = PolicyOutcomeModelSuite().fit(
                train_policy_labels,
                oos_policy_labels,
                seed=cfg.opt_seed + 900,
                min_trades=args.policy_model_min_trades,
            )
            report.policy_model = suite
            policy_model_report = suite.report_frame()
            policy_model_calibration = suite.calibration_table
            policy_model_importance = suite.feature_importance_frame(top_k=40)
            policy_model_predictions = suite.prediction_table
            report.policy_model_report = (
                policy_model_report.to_dict(orient="records")
                if not policy_model_report.empty else []
            )
            report.policy_model_calibration = (
                policy_model_calibration.to_dict(orient="records")
                if not policy_model_calibration.empty else []
            )
            report.policy_model_feature_importance = (
                policy_model_importance.to_dict(orient="records")
                if not policy_model_importance.empty else []
            )
            if policy_model_report.empty:
                print("  (no policy outcome models trained)")
            else:
                print(f"  Target: policy_target_win on generated trades only; "
                      f"min trades/mode={args.policy_model_min_trades}")
                print(f"  {'mode':<22} {'status':<8} {'tr':>6} {'oos':>6} "
                      f"{'base':>7} {'auc':>7} {'top10_R':>9} {'mean_R':>8}")
                for _, row in policy_model_report.iterrows():
                    auc = row.get("oos_auc")
                    auc_s = "n/a" if pd.isna(auc) else f"{auc:.3f}"
                    print(f"  {row['mode']:<22} {row['status']:<8} "
                          f"{int(row.get('train_n', 0)):>6} "
                          f"{int(row.get('oos_n', 0)):>6} "
                          f"{row.get('oos_base_win', 0.0):>6.1%} "
                          f"{auc_s:>7} "
                          f"{row.get('oos_top_decile_return_r', 0.0):>+8.2f} "
                          f"{row.get('oos_mean_return_r', 0.0):>+7.2f}")
                print("  Note: this is research-only; live gates do not use it yet.")
        except Exception as e:
            print(f"  [policy_model] skipped: {e}")

    # R1 policy RETURN model: Huber regression on realized net R per generated trade.
    # Sits alongside the binary classifier above; both predict on the same population
    # so their outputs are directly comparable. Live gates do not consume this yet —
    # surfaced via --r-policy-threshold (opt-in).
    policy_return_model_report = pd.DataFrame()
    policy_return_model_calibration = pd.DataFrame()
    policy_return_model_importance = pd.DataFrame()
    policy_return_model_predictions = pd.DataFrame()
    if (
        args.mode == "train"
        and not args.skip_policy_labels
        and not args.skip_policy_return_model
    ):
        print("\n================ POLICY RETURN MODEL (R1) ================")
        try:
            train_policy_labels_r, oos_policy_labels_r = _ensure_policy_labels()
            r_suite = PolicyReturnModelSuite().fit(
                train_policy_labels_r,
                oos_policy_labels_r,
                seed=cfg.opt_seed + 950,
                min_trades=args.policy_return_min_trades,
            )
            report.policy_return_model = r_suite
            policy_return_model_report = r_suite.report_frame()
            policy_return_model_calibration = r_suite.calibration_table
            policy_return_model_importance = r_suite.feature_importance_frame(top_k=40)
            policy_return_model_predictions = r_suite.prediction_table
            report.policy_return_model_report = (
                policy_return_model_report.to_dict(orient="records")
                if not policy_return_model_report.empty else []
            )
            report.policy_return_model_calibration = (
                policy_return_model_calibration.to_dict(orient="records")
                if not policy_return_model_calibration.empty else []
            )
            report.policy_return_model_feature_importance = (
                policy_return_model_importance.to_dict(orient="records")
                if not policy_return_model_importance.empty else []
            )
            if policy_return_model_report.empty:
                print("  (no policy return regressors trained)")
            else:
                print(f"  Target: policy_target_return_r (winsorized for fit); "
                      f"min trades/mode={args.policy_return_min_trades}")
                print(f"  {'mode':<22} {'status':<8} {'tr':>6} {'oos':>6} "
                      f"{'mean_real':>10} {'top10_R':>9} {'top25_R':>9} {'mae':>6} {'spearman':>8}")
                for _, row in policy_return_model_report.iterrows():
                    sp = row.get("oos_spearman")
                    sp_s = "n/a" if pd.isna(sp) else f"{sp:+.3f}"
                    print(f"  {row['mode']:<22} {row['status']:<8} "
                          f"{int(row.get('train_n', 0)):>6} "
                          f"{int(row.get('oos_n', 0)):>6} "
                          f"{row.get('oos_mean_realized_r', 0.0):>+9.2f} "
                          f"{row.get('oos_top_decile_realized_r', 0.0):>+8.2f} "
                          f"{row.get('oos_top_quartile_realized_r', 0.0):>+8.2f} "
                          f"{row.get('oos_mae', 0.0):>6.2f} {sp_s:>8}")
                print("  Edge gate: at least one mode should show top10_R > 0 and spearman > 0.10.")
                print("  Note: research-only; use --r-policy-threshold to gate live candidates.")
        except Exception as e:
            print(f"  [policy_return_model] skipped: {e}")

    # The early model save happened before policy diagnostics; refresh the bundle once
    # both classifier + return-regressor are attached so predict-mode can use them.
    if args.mode == "train" and not args.skip_policy_labels:
        try:
            _save_model_bundle(report, args.model_dir, args, cfg)
        except Exception as e:
            print(f"  [bundle refresh] skipped: {e}")

    leakage_counts = leakage_summary.get("severity_counts", {}) if leakage_summary else {}
    leakage_error_count = int(leakage_counts.get("ERROR", 0))
    if leakage_summary.get("status") == "ERROR" and not leakage_counts:
        leakage_error_count = 1
    embargo_summary = leakage_summary.get("embargo", {}) if leakage_summary else {}
    consistency_status = "SKIPPED" if args.skip_leakage_audit else (
        "FAIL" if leakage_error_count else
        "WARN" if int(leakage_counts.get("WARN", 0)) else
        "PASS"
    )
    consistency_details = {
        "status": consistency_status,
        "leakage_errors_block_run": bool(leakage_error_count and not args.allow_leakage_errors),
        "leakage_error_count": leakage_error_count,
        "leakage_warning_count": int(leakage_counts.get("WARN", 0)),
        "requested_embargo_bars": args.requested_embargo_bars,
        "effective_embargo_bars": int(cfg.embargo_bars),
        "max_active_label_horizon": int(args.max_active_label_horizon),
        "allow_short_embargo": bool(args.allow_short_embargo),
        "embargo_covers_max_horizon": bool(
            embargo_summary.get("embargo_covers_max_horizon", False)
        ) if leakage_summary else None,
        "replay_audit": leakage_summary.get("replay_audit", {}) if leakage_summary else {},
        "policy_feature_store_rows": int(policy_feature_store_rows),
        "policy_feature_store_root": policy_feature_store_root,
    }

    # ---- Artifacts ----
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sector_intel = serialise_sector_intel(sec_metrics, sec_oos, rotation, corr_df)
    sector_intel_path = out / "sector_data.json"
    sector_intel_path.write_text(json.dumps(sector_intel, indent=2, default=str))
    audit_csv_path = None
    audit_calibration_path = None
    if report.unified_oos_audit is not None and not report.unified_oos_audit.table.empty:
        audit_csv_path = out / "phase3_oos_prediction_audit.csv"
        report.unified_oos_audit.table.to_csv(audit_csv_path, index=False)
        if not report.unified_oos_audit.sector_calibration.empty:
            audit_calibration_path = out / "phase3_sector_calibration.csv"
            report.unified_oos_audit.sector_calibration.to_csv(audit_calibration_path,
                                                               index=False)

    audit_metrics = {}
    distance_bucket_metrics = {}
    if report.unified_oos_audit is not None:
        for name, mt in report.unified_oos_audit.metrics.items():
            audit_metrics[name] = {
                "n": mt.n,
                "brier": mt.brier,
                "logloss": mt.logloss,
                "auc": mt.auc,
                "top_decile_hit_rate": mt.top_decile_hit_rate,
                "base_rate": mt.base_rate,
                "mean_prediction": mt.mean_prediction,
            }
        distance_bucket_metrics = report.unified_oos_audit.distance_bucket_metrics

    sector_shrinkage_report = {}
    if report.unified_ml is not None and hasattr(report.unified_ml, "sector_stats"):
        sector_shrinkage_report = report.unified_ml.sector_stats

    summary = {
        "symbols": symbols,
        "validation_method": getattr(report.unified_ml, "validation_method", None)
                             if report.unified_ml else cfg.validation_method,
        "regularization_preset": cfg.regularization_preset,
        "embargo_bars": cfg.embargo_bars,
        "requested_embargo_bars": args.requested_embargo_bars,
        "effective_embargo_bars": int(cfg.embargo_bars),
        "max_active_label_horizon": int(args.max_active_label_horizon),
        "allow_short_embargo": bool(args.allow_short_embargo),
        "consistency_status": consistency_details,
        "model_hyperparameters": getattr(report.unified_ml, "hyperparameters", {})
                                  if report.unified_ml else {},
        "validation_fold_stats": [
            {
                "fold": st.fold,
                "validation_start": st.validation_start,
                "validation_end": st.validation_end,
                "original_train_size": st.original_train_size,
                "purged_train_size": st.purged_train_size,
                "purged_rows_removed": st.purged_rows_removed,
                "embargoed_rows_removed": st.embargoed_rows_removed,
                "validation_size": st.validation_size,
            }
            for st in (getattr(report.unified_ml, "validation_fold_stats", [])
                       if report.unified_ml else [])
        ],
        "total_oos_tested": report.total_oos_tested,
        "pooled_oos_respect": report.pooled_oos_respect,
        "pooled_oos_strict": report.pooled_oos_strict,
        "pooled_oos_ci_wilson": list(report.pooled_oos_ci_wilson),
        "verdict": verdict,
        "best_ev": best_ev,
        "best_net_expectancy_r": best_ev,
        "n_tradeable": len(tradeable),
        "n_watchlist": len(watchlist),
        "model_health_warning": model_health_warning,
        "execution_mode": args.execution_mode,
        "cost_model": cost_cfg.to_dict(),
        "execution_backtest_split": args.execution_backtest_split,
        "execution_backtest_summary": (
            execution_summary.to_dict(orient="records") if not execution_summary.empty else []
        ),
        "execution_backtest_by_direction": (
            execution_by_direction.to_dict(orient="records")
            if not execution_by_direction.empty else []
        ),
        "execution_backtest_v2_config": {
            "enabled": bool(args.run_execution_backtest_v2),
            "fill_policy": args.fill_policy,
            "use_1m_resolution": bool(args.use_1m_resolution),
            "slippage_model": args.slippage_model,
            "base_slippage_bps": args.v2_base_slippage_bps,
            "exchange": args.execution_exchange,
        },
        "execution_backtest_v2_summary": (
            execution_v2_summary.to_dict(orient="records")
            if not execution_v2_summary.empty else []
        ),
        "execution_backtest_v2_vs_v1_delta": (
            execution_v2_delta.to_dict(orient="records")
            if not execution_v2_delta.empty else []
        ),
        "leakage_audit": leakage_summary,
        "policy_label_split": args.policy_label_split,
        "policy_label_summary": (
            policy_label_summary.to_dict(orient="records")
            if not policy_label_summary.empty else []
        ),
        "policy_feature_store_rows": int(policy_feature_store_rows),
        "policy_feature_store_root": policy_feature_store_root,
        "policy_model_report": (
            policy_model_report.to_dict(orient="records")
            if not policy_model_report.empty else getattr(report, "policy_model_report", [])
        ),
        "policy_model_calibration": (
            policy_model_calibration.to_dict(orient="records")
            if not policy_model_calibration.empty
            else getattr(report, "policy_model_calibration", [])
        ),
        "gate_post_touch_min_strict": MIN_POST_TOUCH_STRICT,
        "unified_ml_val_auc": report.unified_ml.val_auc if report.unified_ml else None,
        "unified_direction_auc": (report.unified_timing_report.direction_auc
                                   if report.unified_timing_report else None),
        "phase3_oos_audit": audit_metrics,
        "distance_bucket_metrics": {
            "quality": distance_bucket_metrics,
            "proximity": {
                str(s.horizon): s.distance_bucket_metrics
                for s in (report.unified_timing_report.proximity_per_horizon
                          if report.unified_timing_report else [])
            },
        },
        "sector_shrinkage_report": sector_shrinkage_report,
        "post_touch_reaction_metrics": report.post_touch_reaction_metrics,
        "reaction_model_report": getattr(report, "reaction_model_report", []),
        "reaction_model_calibration": getattr(report, "reaction_model_calibration", []),
        "reaction_alert_config": {
            "feature_bars": args.reaction_feature_bars,
            "lookback_bars": args.reaction_alert_lookback_bars,
            "confirm_threshold": args.reaction_confirm_threshold,
            "break_risk_threshold": args.reaction_break_risk_threshold,
        },
        "reaction_alert_note": reaction_alert_note,
        "reaction_alerts": reaction_alerts,
        "feature_store": report.feature_store_stats,
        "gate_decisions": gate_decisions,
        "track_a_pretouch": {
            "research_only": True,
            "source": "Phase 4 constrained validation pocket",
            "enabled": True,
            "filters": {
                "direction": "UP only",
                "allowed_sectors": sorted(TRACK_A_ALLOWED_SECTORS),
                "min_p_touch": TRACK_A_MIN_TOUCH,
                "min_p_direction_to_pool": TRACK_A_MIN_DIRECTION,
                "min_distance_atr": TRACK_A_MIN_DISTANCE_ATR,
                "max_distance_atr": TRACK_A_MAX_DISTANCE_ATR,
                "target_fraction": TRACK_A_TARGET_FRACTION,
                "stop_atr_mult": TRACK_A_STOP_ATR_MULT,
                "max_hold_bars": TRACK_A_MAX_HOLD_BARS,
            },
            "setups": track_a_pretouch[:50],
        },
    }
    out_json = out / "multi_asset_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    research_json = out / "research_summary.json"
    research_json.write_text(json.dumps(summary, indent=2, default=str))
    live_plan = {
        "verdict": verdict,
        "why": verdict_why,
        "best_ev": best_ev,
        "best_net_expectancy_r": best_ev,
        "model_health_warning": model_health_warning,
        "execution_mode": args.execution_mode,
        "cost_model": cost_cfg.to_dict(),
        "execution_backtest_summary": (
            execution_summary.to_dict(orient="records") if not execution_summary.empty else []
        ),
        "execution_backtest_by_direction": (
            execution_by_direction.to_dict(orient="records")
            if not execution_by_direction.empty else []
        ),
        "execution_backtest_v2_config": {
            "enabled": bool(args.run_execution_backtest_v2),
            "fill_policy": args.fill_policy,
            "use_1m_resolution": bool(args.use_1m_resolution),
            "slippage_model": args.slippage_model,
            "base_slippage_bps": args.v2_base_slippage_bps,
            "exchange": args.execution_exchange,
        },
        "execution_backtest_v2_summary": (
            execution_v2_summary.to_dict(orient="records")
            if not execution_v2_summary.empty else []
        ),
        "execution_backtest_v2_vs_v1_delta": (
            execution_v2_delta.to_dict(orient="records")
            if not execution_v2_delta.empty else []
        ),
        "consistency_status": consistency_details,
        "leakage_audit": leakage_summary,
        "policy_label_summary": (
            policy_label_summary.to_dict(orient="records")
            if not policy_label_summary.empty else []
        ),
        "policy_model_report": (
            policy_model_report.to_dict(orient="records")
            if not policy_model_report.empty else getattr(report, "policy_model_report", [])
        ),
        "reaction_model_report": getattr(report, "reaction_model_report", []),
        "reaction_confirmation": {
            "note": reaction_alert_note,
            "feature_bars": args.reaction_feature_bars,
            "lookback_bars": args.reaction_alert_lookback_bars,
            "confirm_threshold": args.reaction_confirm_threshold,
            "break_risk_threshold": args.reaction_break_risk_threshold,
            "alerts": reaction_alerts[:20],
        },
        "gate": {
            "min_q": GATE_Q,
            "min_p_touch_today": GATE_T_TODAY,
            "required_direction": "DIR_ALIGN",
            "min_distance_atr": PRACTICAL_MIN_ATR,
            "max_distance_atr": PRACTICAL_MAX_ATR,
            "min_bucket_n": MIN_BUCKET_N,
            "min_post_touch_strict": MIN_POST_TOUCH_STRICT,
            "requires_positive_net_expectancy_after_costs": True,
        },
        "best_observed_touch": {
            "symbol": max_t_today_cand["symbol"] if max_t_today_cand else None,
            "p_touch_today": max_t_today,
            "p_respect": max_t_today_cand["q"] if max_t_today_cand else None,
            "p_reaction": max_t_today_cand["p_reaction"] if max_t_today_cand else None,
            "p_trade": max_t_today_cand["p_trade"] if max_t_today_cand else None,
            "decision": "WATCH_ONLY" if max_t_today_cand and not tradeable else None,
        },
        "tradeable_setups": [
            {
                "symbol": c["symbol"],
                "side": c["side"],
                "pool_low": c["pool"].price_low,
                "pool_high": c["pool"].price_high,
                "entry": c["entry"],
                "stop": c["stop"],
                "target": c["target"],
                "p_touch": c["p_touch"],
                "p_respect": c["p_respect"],
                "p_reaction": c["p_reaction"],
                "p_trade": c["p_trade"],
                "direction": c["direction"],
                "direction_sign": c["direction_sign"],
                "net_expectancy_r": c["net_expectancy_r"],
                "directional_net_expectancy_r": c["directional_net_expectancy_r"],
                "net_expectancy_per_share": c["net_expectancy_per_share"],
            }
            for c in tradeable[:20]
        ],
        "track_a_pretouch": {
            "research_only": True,
            "source": "Phase 4 constrained validation pocket",
            "filters": {
                "direction": "UP only",
                "allowed_sectors": sorted(TRACK_A_ALLOWED_SECTORS),
                "min_p_touch": TRACK_A_MIN_TOUCH,
                "min_p_direction_to_pool": TRACK_A_MIN_DIRECTION,
                "min_distance_atr": TRACK_A_MIN_DISTANCE_ATR,
                "max_distance_atr": TRACK_A_MAX_DISTANCE_ATR,
                "target_fraction": TRACK_A_TARGET_FRACTION,
                "stop_atr_mult": TRACK_A_STOP_ATR_MULT,
                "max_hold_bars": TRACK_A_MAX_HOLD_BARS,
            },
            "setups": track_a_pretouch[:20],
        },
    }
    live_plan_path = out / "live_plan.json"
    live_plan_path.write_text(json.dumps(live_plan, indent=2, default=str))
    brief_writer = OutcomeLogWriter(root=str(out / "outcome_log"))
    brief_target_date = _next_trading_date_after_latest_bar(report)
    daily_brief = generate_brief(
        report,
        trading_date_ist=brief_target_date,
        indexes_covered=[],
        model_bundle_version=_brief_bundle_label(args.model_dir),
        outcome_log_writer=brief_writer,
    )
    daily_brief_json_path = out / "daily_brief.json"
    daily_brief_text_path = out / "daily_brief.txt"
    daily_brief_json_path.write_text(
        json.dumps(daily_brief.to_dict(), indent=2, default=str)
    )
    daily_brief_text_path.write_text(render_email(daily_brief))
    track_a_pretouch_path = out / "track_a_pretouch_setups.csv"
    pd.DataFrame(track_a_pretouch, columns=TRACK_A_OUTPUT_COLUMNS).to_csv(
        track_a_pretouch_path, index=False,
    )
    gate_df = pd.DataFrame(gate_decisions)
    if not gate_df.empty:
        gate_df.to_csv(out / "live_gate_decisions.csv", index=False)
        gate_df[gate_df["decision"] == "TRADEABLE"].to_csv(
            out / "tradeable_setups.csv", index=False,
        )
        gate_df[gate_df["decision"] == "WATCH_ONLY"].to_csv(
            out / "watchlist.csv", index=False,
        )
        gate_df[gate_df["decision"] == "REJECTED_WITH_REASON"].to_csv(
            out / "rejected_setups.csv", index=False,
        )
    reaction_alerts_path = None
    if reaction_alerts:
        reaction_alerts_path = out / "reaction_alerts.csv"
        pd.DataFrame(reaction_alerts).to_csv(reaction_alerts_path, index=False)
    leakage_audit_path = None
    leakage_issues_path = None
    if leakage_summary:
        leakage_audit_path = out / "leakage_audit.json"
        leakage_audit_path.write_text(json.dumps(leakage_summary, indent=2, default=str))
    if not leakage_issues.empty:
        leakage_issues_path = out / "leakage_issues.csv"
        leakage_issues.to_csv(leakage_issues_path, index=False)
    policy_label_summary_path = None
    policy_labels_path = None
    if not policy_label_summary.empty:
        policy_label_summary_path = out / "policy_label_summary.csv"
        policy_label_summary.to_csv(policy_label_summary_path, index=False)
    if not policy_labels.empty:
        policy_labels_path = out / "execution_policy_outcomes.parquet"
        policy_labels.to_parquet(policy_labels_path, index=False)
    policy_model_report_path = None
    policy_model_calibration_path = None
    policy_model_importance_path = None
    policy_model_predictions_path = None
    if not policy_model_report.empty:
        policy_model_report_path = out / "policy_model_report.csv"
        policy_model_report.to_csv(policy_model_report_path, index=False)
    if not policy_model_calibration.empty:
        policy_model_calibration_path = out / "policy_model_calibration.csv"
        policy_model_calibration.to_csv(policy_model_calibration_path, index=False)
    if not policy_model_importance.empty:
        policy_model_importance_path = out / "policy_model_feature_importance.csv"
        policy_model_importance.to_csv(policy_model_importance_path, index=False)
    if not policy_model_predictions.empty:
        policy_model_predictions_path = out / "policy_model_oos_predictions.csv"
        policy_model_predictions.to_csv(policy_model_predictions_path, index=False)
    # R1 policy return regressor outputs.
    policy_return_report_path = None
    policy_return_calibration_path = None
    policy_return_importance_path = None
    policy_return_predictions_path = None
    if not policy_return_model_report.empty:
        policy_return_report_path = out / "policy_return_model_report.csv"
        policy_return_model_report.to_csv(policy_return_report_path, index=False)
    if not policy_return_model_calibration.empty:
        policy_return_calibration_path = out / "policy_return_model_calibration.csv"
        policy_return_model_calibration.to_csv(policy_return_calibration_path, index=False)
    if not policy_return_model_importance.empty:
        policy_return_importance_path = out / "policy_return_model_feature_importance.csv"
        policy_return_model_importance.to_csv(policy_return_importance_path, index=False)
    if not policy_return_model_predictions.empty:
        policy_return_predictions_path = out / "policy_return_model_oos_predictions.csv"
        policy_return_model_predictions.to_csv(policy_return_predictions_path, index=False)
    execution_summary_path = None
    execution_trades_path = None
    execution_by_direction_path = None
    execution_v2_summary_path = None
    execution_v2_trades_path = None
    execution_v2_delta_path = None
    if not execution_summary.empty:
        execution_summary_path = out / "execution_backtest_summary.csv"
        execution_summary.to_csv(execution_summary_path, index=False)
    if not execution_by_direction.empty:
        execution_by_direction_path = out / "execution_backtest_by_direction.csv"
        execution_by_direction.to_csv(execution_by_direction_path, index=False)
    if not execution_trades.empty:
        execution_trades_path = out / "execution_backtest_trades.csv"
        execution_trades.to_csv(execution_trades_path, index=False)
    if not execution_v2_summary.empty:
        execution_v2_summary_path = out / "execution_backtest_v2_summary.csv"
        execution_v2_summary.to_csv(execution_v2_summary_path, index=False)
    if not execution_v2_trades.empty:
        execution_v2_trades_path = out / "execution_backtest_v2_trades.csv"
        execution_v2_trades.to_csv(execution_v2_trades_path, index=False)
    if not execution_v2_delta.empty:
        execution_v2_delta_path = out / "execution_backtest_v2_vs_v1_delta.csv"
        execution_v2_delta.to_csv(execution_v2_delta_path, index=False)
    validation_rows = summary["validation_fold_stats"]
    if validation_rows:
        pd.DataFrame(validation_rows).to_csv(out / "validation_report.csv", index=False)
    if report.unified_ml is not None and hasattr(report.unified_ml, "feature_importance"):
        pd.DataFrame(
            report.unified_ml.feature_importance(top_k=200),
            columns=["feature", "importance_gain"],
        ).to_csv(out / "feature_importance.csv", index=False)
    if report.feature_store_stats:
        try:
            fs = FeatureStore(report.feature_store_stats["root"])
            events = fs.scan("reaction_events/*/*.parquet")
            if not events.empty:
                events.to_parquet(out / "post_touch_events.parquet", index=False)
                events.groupby(["sector", "reaction_label"]).size().reset_index(
                    name="n"
                ).to_csv(out / "post_touch_report.csv", index=False)
        except Exception as e:
            print(f"[feature_store] could not export post_touch_events.parquet: {e}")
    reaction_report_path = None
    reaction_calibration_path = None
    reaction_importance_path = None
    reaction_model_report = getattr(report, "reaction_model_report", [])
    reaction_model_calibration = getattr(report, "reaction_model_calibration", [])
    reaction_feature_importance = getattr(report, "reaction_feature_importance", [])
    if reaction_model_report:
        reaction_report_path = out / "reaction_model_report.csv"
        pd.DataFrame(reaction_model_report).to_csv(reaction_report_path, index=False)
    if reaction_model_calibration:
        reaction_calibration_path = out / "reaction_model_calibration.csv"
        pd.DataFrame(reaction_model_calibration).to_csv(
            reaction_calibration_path, index=False,
        )
    if reaction_feature_importance:
        reaction_importance_path = out / "reaction_feature_importance.csv"
        pd.DataFrame(reaction_feature_importance).to_csv(
            reaction_importance_path, index=False,
        )
    print(f"\nartifacts: {out_json}")
    print(f"           {research_json}")
    print(f"           {live_plan_path}")
    print(f"           {daily_brief_json_path}  ← Daily Research Brief JSON")
    print(f"           {daily_brief_text_path}  ← Daily Research Brief email text")
    print(f"           {track_a_pretouch_path}  ← Phase 4 Track A pre-touch research setups")
    print(f"           {sector_intel_path}  ← Track 5 will ingest this for live decisions")
    if audit_csv_path is not None:
        print(f"           {audit_csv_path}  ← Phase 3A global/sector/blended OOS rows")
    if audit_calibration_path is not None:
        print(f"           {audit_calibration_path}  ← Phase 3A per-sector calibration")
    if execution_summary_path is not None:
        print(f"           {execution_summary_path}  ← Phase 2B execution-mode PnL")
    if execution_by_direction_path is not None:
        print(f"           {execution_by_direction_path}  ← Phase 2C UP/DOWN split")
    if execution_trades_path is not None:
        print(f"           {execution_trades_path}  ← Phase 2B trade-level fills")
    if execution_v2_summary_path is not None:
        print(f"           {execution_v2_summary_path}  ← Phase 4 execution simulator v2 PnL")
    if execution_v2_trades_path is not None:
        print(f"           {execution_v2_trades_path}  ← Phase 4 v2 itemized trade fills")
    if execution_v2_delta_path is not None:
        print(f"           {execution_v2_delta_path}  ← Phase 4 v2 vs v1 delta")
    if leakage_audit_path is not None:
        print(f"           {leakage_audit_path}  ← Phase 3C leakage audit")
    if leakage_issues_path is not None:
        print(f"           {leakage_issues_path}  ← Phase 3C leakage issue rows")
    if policy_label_summary_path is not None:
        print(f"           {policy_label_summary_path}  ← Phase 3C policy label summary")
    if policy_labels_path is not None:
        print(f"           {policy_labels_path}  ← Phase 3C policy outcome labels")
    if policy_feature_store_rows:
        print(f"           {policy_feature_store_root}/policy_labels  ← Phase 3C feature-store policy labels")
    if policy_model_report_path is not None:
        print(f"           {policy_model_report_path}  ← Phase 3D policy outcome model metrics")
    if policy_model_calibration_path is not None:
        print(f"           {policy_model_calibration_path}  ← Phase 3D policy model calibration")
    if policy_model_importance_path is not None:
        print(f"           {policy_model_importance_path}  ← Phase 3D policy model features")
    if policy_model_predictions_path is not None:
        print(f"           {policy_model_predictions_path}  ← Phase 3D top OOS policy predictions")
    if policy_return_report_path is not None:
        print(f"           {policy_return_report_path}  ← R1 policy return regressor metrics")
    if policy_return_calibration_path is not None:
        print(f"           {policy_return_calibration_path}  ← R1 policy return calibration")
    if policy_return_importance_path is not None:
        print(f"           {policy_return_importance_path}  ← R1 policy return features")
    if policy_return_predictions_path is not None:
        print(f"           {policy_return_predictions_path}  ← R1 top OOS predicted-R candidates")
    if reaction_report_path is not None:
        print(f"           {reaction_report_path}  ← Phase 3 post-touch model metrics")
    if reaction_calibration_path is not None:
        print(f"           {reaction_calibration_path}  ← Phase 3 reaction calibration")
    if reaction_importance_path is not None:
        print(f"           {reaction_importance_path}  ← Phase 3 reaction features")
    if reaction_alerts_path is not None:
        print(f"           {reaction_alerts_path}  ← Phase 3 post-touch alerts")

    if leakage_error_count and not args.allow_leakage_errors:
        raise SystemExit(
            "leakage audit found "
            f"{leakage_error_count} ERROR row(s); artifacts were written. "
            "Use --allow-leakage-errors only for debugging/legacy comparisons."
        )

    # Per-asset chart
    for symbol, ad in report.assets.items():
        safe = symbol.replace(".", "_")
        html = out / f"{safe}_multi.html"
        plot_chart(ad.base_df, ad.final_pools, results=ad.final_results,
                   title=f"{symbol} — multi-asset run", out_path=str(html), top_n=20)


if __name__ == "__main__":
    main()
