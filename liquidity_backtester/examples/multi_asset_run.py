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
from pathlib import Path

import numpy as np

from liqpool import Config, plot_chart
from liqpool.featurize import MultiAssetFeaturizer
from liqpool.multi_asset import run_multi_asset, print_multi_asset_summary
from liqpool.timing import StateFeaturizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols",
                    default="HDFCBANK.NS,ICICIBANK.NS,KOTAKBANK.NS,SBIN.NS,AXISBANK.NS",
                    help="comma-separated NSE symbols")
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
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
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
    )

    print(f"=== MULTI-ASSET RUN ({len(symbols)} assets) ===")
    print(f"  symbols:    {symbols}")
    print(f"  base/tfs:   {args.base} + {cfg.higher_tfs}")
    print(f"  folds:      {args.folds}  iters/fold: {args.iters}  final iters: {args.final_iters}")

    def prog(symbol, step):
        print(f"  [{symbol}]  {step}")

    report = run_multi_asset(symbols, cfg,
                              n_folds=args.folds, train_frac=args.train_frac,
                              iters_per_fold=args.iters, final_iters=args.final_iters,
                              min_train_days=args.min_train_days,
                              progress=prog)

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

    candidates = []        # cross-asset list

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

        for side_str, pool_list in (("above", above), ("below", below)):
            for p, r in pool_list:
                dist = (p.mid - current) if side_str == "above" else (current - p.mid)
                dist_atr = dist / max(atr_proxy, 1e-9)
                X = report.unified_featurizer.transform_batch([p])
                q = float(report.unified_ml.predict(X, pools=[p])[0])
                t_by_h = {}
                for h in PROX_HORIZONS:
                    pm = report.unified_proximity[h]
                    t_by_h[h] = pm.predict_one(p, dist_atr, side_str, current_state, q)
                tag = direction_tag(side_str)
                bonus = 1.15 if tag == "DIR_ALIGN" else (0.85 if tag == "DIR_FIGHT" else 1.0)
                t_today = t_by_h.get(primary_h, 0.0)
                ev = q * t_today * bonus
                candidates.append({
                    "symbol": symbol, "pool": p, "result": r, "side": side_str,
                    "dist_atr": dist_atr, "q": q, "t_by_h": t_by_h,
                    "dir_tag": tag, "ev": ev,
                    "current": current, "atr_proxy": atr_proxy,
                    "dir_p_up": dir_p_up,
                })

    candidates.sort(key=lambda c: -c["ev"])

    tradeable = [c for c in candidates
                  if c["q"] >= TRADEABLE_Q and c["t_by_h"].get(primary_h, 0.0) >= TRADEABLE_T_TODAY]
    watchlist = [c for c in candidates if c not in tradeable]

    # Day verdict
    best_ev = tradeable[0]["ev"] if tradeable else 0.0
    max_t_today = max((c["t_by_h"].get(primary_h, 0.0) for c in candidates), default=0.0)
    max_t_2d = (max((c["t_by_h"].get(PROX_HORIZONS[1], 0.0) for c in candidates), default=0.0)
                if len(PROX_HORIZONS) > 1 else 0.0)
    if tradeable and best_ev >= 0.20:
        verdict = "TRADE_HIGH_CONFIDENCE"
    elif tradeable:
        verdict = "TRADE_CAUTIOUS"
    elif max_t_today >= 0.02 or max_t_2d >= 0.20:
        verdict = "WATCH (no setup today; pool likely tradeable in 1-2 days)"
    else:
        verdict = "NO_TRADE (no actionable pool across the basket)"

    print(f"  Assets in basket:   {', '.join(report.assets.keys())}")
    print(f"  Cross-asset best EV today: {best_ev:.1%}")
    print(f"  Max T_today across all:    {max_t_today:.1%}    "
          f"Max T_2d: {max_t_2d:.1%}")
    print(f"  VERDICT:            {verdict}")
    print("----------------------------------------------------------")

    if tradeable:
        top = tradeable[0]
        p = top["pool"]
        srcs = sorted({c.source.split('@')[0] for c in p.contributors})
        side_label = "BELOW (buy)" if top["side"] == "below" else "ABOVE (sell)"
        print(f">>> BEST SETUP TODAY  ({top['symbol']}) <<<")
        print(f"  Pool ₹{p.price_low:.2f}-{p.price_high:.2f}  mid ₹{p.mid:.2f}  [{side_label}]")
        print(f"  Distance: {top['dist_atr']:.2f} ATRs from current ₹{top['current']:.2f}  "
              f"({top['dir_tag']})")
        print(f"  Q={top['q']:.1%}   T_today={top['t_by_h'][primary_h]:.1%}   "
              f"EV={top['ev']:.1%}")
        print(f"  Drivers: {', '.join(srcs)}   |  TFs: {'+'.join(p.tfs)}")
        a = top["atr_proxy"]
        if top["side"] == "below":
            print(f"  Action: LIMIT BUY at ₹{p.price_high:.2f}, "
                  f"stop ₹{p.price_low - 0.5 * a:.2f}, "
                  f"target ₹{p.price_high + 2 * a:.2f}+")
        else:
            print(f"  Action: LIMIT SELL at ₹{p.price_low:.2f}, "
                  f"stop ₹{p.price_high + 0.5 * a:.2f}, "
                  f"target ₹{p.price_low - 2 * a:.2f}-")
    else:
        print(">>> NO TRADEABLE SETUP IN BASKET TODAY <<<")
    print("==========================================================")

    print(f"\n--- TRADEABLE TODAY  (Q ≥ {TRADEABLE_Q:.0%} AND T_today ≥ {TRADEABLE_T_TODAY:.0%}) ---")
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
                  f"[{side_lbl}, {c['dist_atr']:.1f}ATR, {c['dir_tag']}]   "
                  f"Q={c['q']:.1%}  {t_strs}  EV={c['ev']:.1%}")

    print(f"\n--- WATCH LIST  (top 10 by Q, not tradeable today) ---")
    for c in sorted(watchlist, key=lambda x: -x["q"])[:10]:
        p = c["pool"]
        side_lbl = "BELOW" if c["side"] == "below" else "ABOVE"
        t_strs = "  ".join(f"T_h{h}={c['t_by_h'].get(h, 0.0):.1%}" for h in PROX_HORIZONS)
        print(f"  [{c['symbol']:<14}] ₹{p.price_low:.2f}-{p.price_high:.2f}  "
              f"[{side_lbl}, {c['dist_atr']:.1f}ATR, {c['dir_tag']}]   "
              f"Q={c['q']:.1%}  {t_strs}")

    # ---- Artifacts ----
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = {
        "symbols": symbols,
        "total_oos_tested": report.total_oos_tested,
        "pooled_oos_respect": report.pooled_oos_respect,
        "pooled_oos_strict": report.pooled_oos_strict,
        "pooled_oos_ci_wilson": list(report.pooled_oos_ci_wilson),
        "verdict": verdict,
        "best_ev": best_ev,
        "n_tradeable": len(tradeable),
        "n_watchlist": len(watchlist),
        "unified_ml_val_auc": report.unified_ml.val_auc if report.unified_ml else None,
        "unified_direction_auc": (report.unified_timing_report.direction_auc
                                   if report.unified_timing_report else None),
    }
    out_json = out / "multi_asset_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nartifacts: {out_json}")

    # Per-asset chart
    for symbol, ad in report.assets.items():
        safe = symbol.replace(".", "_")
        html = out / f"{safe}_multi.html"
        plot_chart(ad.base_df, ad.final_pools, results=ad.final_results,
                   title=f"{symbol} — multi-asset run", out_path=str(html), top_n=20)


if __name__ == "__main__":
    main()
