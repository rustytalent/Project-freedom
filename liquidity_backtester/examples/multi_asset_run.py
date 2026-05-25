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
    expected_trade_value,
    trade_levels,
)
from liqpool.data import ParquetProvider
from liqpool.featurize import MultiAssetFeaturizer
from liqpool.feature_store import FeatureStore
from liqpool.multi_asset import (AssetData, run_multi_asset, print_multi_asset_summary,
                                 distance_bucket)
from liqpool.pools import build_pools, project_to_base
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
    ap.add_argument("--asset-workers", type=int, default=1,
                    help="Number of symbols to process in parallel during train mode")
    ap.add_argument("--checkpoint-dir", default="output_checkpoints",
                    help="Per-symbol checkpoint directory for train mode")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse completed per-symbol checkpoints from --checkpoint-dir")
    ap.add_argument("--feature-store-dir", default="",
                    help=("Partitioned parquet feature-store root. If omitted with "
                          "--universe core25 train mode, defaults to output_feature_store/core25."))
    ap.add_argument("--embargo-bars", type=int, default=78)
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
                    choices=("blind_limit", "touch_confirmed", "reclaim_confirmed"),
                    help="Live execution style. Default waits for reclaim/confirmation after touch.")
    ap.add_argument("--cost-product", default="intraday",
                    choices=("intraday", "delivery"),
                    help="Zerodha equity cost profile used for net expectancy")
    ap.add_argument("--slippage-bps", type=float, default=1.0,
                    help="Assumed slippage in bps per side")
    ap.add_argument("--cost-quantity", type=int, default=1,
                    help="Quantity used for flat-charge cost estimates in reports")
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
        embargo_bars=args.embargo_bars,
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
        # Take top-5 by score on each side
        above = sorted([(p, r) for p, r in active if p.price_low > current],
                        key=lambda x: -x[0].score)[:5]
        below = sorted([(p, r) for p, r in active if p.price_high < current],
                        key=lambda x: -x[0].score)[:5]

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
                else:
                    q = float(report.unified_ml.predict(X, pools=[p])[0])
                t_by_h = {}
                for h in PROX_HORIZONS:
                    pm = report.unified_proximity[h]
                    t_by_h[h] = pm.predict_one(p, dist_atr, side_str, current_state, q)
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
                    "net_expectancy_per_share": trade_ev["net_expectancy_per_share"],
                    "net_expectancy_r": trade_ev["net_expectancy_r"],
                    "conditional_net_expectancy_per_share": (
                        trade_ev["conditional_net_expectancy_per_share"]
                    ),
                    "expected_cost_per_share": trade_ev["expected_cost_per_share"],
                    "entry": levels["entry"],
                    "stop": levels["stop"],
                    "target": levels["target"],
                    "current": current, "atr_proxy": atr_proxy,
                    "dir_p_up": dir_p_up,
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

    gate_decisions = []
    tradeable = []
    watch_only = []
    rejected = []
    for c in candidates:
        reasons = live_gate_reasons(c)
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
            "net_expectancy_per_share": c["net_expectancy_per_share"],
            "expected_cost_per_share": c["expected_cost_per_share"],
            "execution_mode": args.execution_mode,
            "entry": c["entry"],
            "stop": c["stop"],
            "target": c["target"],
            "decision": decision_name,
            "reasons": reasons,
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
                  f"netEV={c['net_expectancy_r']:+.2f}R")

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
                  f"netEV={c['net_expectancy_r']:+.2f}R rejected: "
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
        "feature_store": report.feature_store_stats,
        "gate_decisions": gate_decisions,
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
                "net_expectancy_r": c["net_expectancy_r"],
                "net_expectancy_per_share": c["net_expectancy_per_share"],
            }
            for c in tradeable[:20]
        ],
    }
    live_plan_path = out / "live_plan.json"
    live_plan_path.write_text(json.dumps(live_plan, indent=2, default=str))
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
    print(f"\nartifacts: {out_json}")
    print(f"           {research_json}")
    print(f"           {live_plan_path}")
    print(f"           {sector_intel_path}  ← Track 5 will ingest this for live decisions")
    if audit_csv_path is not None:
        print(f"           {audit_csv_path}  ← Phase 3A global/sector/blended OOS rows")
    if audit_calibration_path is not None:
        print(f"           {audit_calibration_path}  ← Phase 3A per-sector calibration")

    # Per-asset chart
    for symbol, ad in report.assets.items():
        safe = symbol.replace(".", "_")
        html = out / f"{safe}_multi.html"
        plot_chart(ad.base_df, ad.final_pools, results=ad.final_results,
                   title=f"{symbol} — multi-asset run", out_path=str(html), top_n=20)


if __name__ == "__main__":
    main()
