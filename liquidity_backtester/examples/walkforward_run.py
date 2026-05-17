"""Walk-forward validation of the liquidity-pool backtester.

Workflow:
  1. Walk-forward: split history into expanding-train, held-out-test folds. Optimize per fold
     on TRAIN-ONLY data (with horizon-aware no-leak rule); evaluate on OOS. Aggregate.
  2. Print per-fold table, overfit gap, and pooled OOS respect rate with 90% CIs (Wilson +
     bootstrap).
  3. Fit a FINAL config on the full history (this is what a "live" trader would use). Compare
     its in-sample respect to the OOS estimate so we know how much salt to apply.
  4. Identify next pool above/below current price, annotated with the OOS CI as the realistic
     expected respect probability.

Run:
    python examples/walkforward_run.py --symbol HDFCBANK.NS --period 60d --folds 5 --iters 80
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from liqpool import Config, fetch, multi_timeframe, build_pools, test_pools, plot_chart, summarise
from liqpool.pools import project_to_base
from liqpool.optimizer import optimize
from liqpool.walkforward import walk_forward, print_report as print_wf_report
from liqpool.stratified import print_model as print_stratified_model
from liqpool.featurize import Featurizer


def nearest_untouched(pools, results, current_price, side, k=3):
    by_idx = {r.pool_idx: r for r in results}
    cand = []
    for i, p in enumerate(pools):
        r = by_idx.get(i)
        if r is None or r.is_break or r.outcome == "horizon_insufficient":
            continue
        if side == "above" and p.price_low > current_price:
            cand.append((p, r))
        if side == "below" and p.price_high < current_price:
            cand.append((p, r))
    cand.sort(key=lambda x: (-x[0].score, abs(x[0].mid - current_price)))
    return cand[:k]


def _print_summary(label, s):
    print(f"{label:<30} n={s['n']:4d}  eligible={s.get('eligible_n', s['n']):4d}  "
          f"tested={s['tested_n']:4d}  respect={s['respect_rate']:.2%}  "
          f"break={s['break_rate']:.2%}  untouched={s['untouched_rate']:.2%}  "
          f"horizon_insuff={s.get('horizon_insufficient', 0):3d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="HDFCBANK.NS")
    ap.add_argument("--base", default="5m")
    ap.add_argument("--period", default="60d")
    ap.add_argument("--tfs", default="15min,60min,180min,1D,1W")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.6, dest="train_frac",
                    help="fraction of total history reserved for the first fold's training")
    ap.add_argument("--iters", type=int, default=80,
                    help="optimizer iterations per fold")
    ap.add_argument("--final-iters", type=int, default=200, dest="final_iters",
                    help="iterations for the final fit on full history")
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--min-score", type=float, default=2.0, dest="min_score")
    ap.add_argument("--min-train-days", type=int, default=10, dest="min_train_days")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    cfg = Config(
        symbol=args.symbol, base_interval=args.base, period=args.period,
        higher_tfs=args.tfs.split(","),
        test_horizon_bars=args.horizon,
        opt_iterations=args.iters, opt_explore_frac=0.35, opt_seed=11,
        min_pool_score=args.min_score,
    )

    print(f"=== {cfg.symbol} {cfg.base_interval}  higher TFs: {cfg.higher_tfs} ===")
    base = fetch(cfg.symbol, cfg.base_interval, cfg.period)
    tf = multi_timeframe(base, cfg.higher_tfs)
    print(f"bars: base={len(base)}  range: {base.index[0]} → {base.index[-1]}")
    for k, v in tf.items():
        if k != "base":
            print(f"      {k}: {len(v)}")

    # ---- 1. Walk-forward ----
    print(f"\n--- WALK-FORWARD  ({args.folds} folds, train_frac={args.train_frac}, "
          f"{args.iters} iters/fold) ---")

    def prog(i, total, label):
        print(f"  fold {i+1}/{total}  {label}")

    wf = walk_forward(tf, cfg, n_folds=args.folds, train_frac=args.train_frac,
                       iters_per_fold=args.iters, min_train_days=args.min_train_days,
                       progress=prog)
    print_wf_report(wf)

    # Stratified per-pool probability model fit on OOS data (used below for next-pool predictions)
    if wf.stratified_model is not None:
        print_stratified_model(wf.stratified_model)

    # ---- 2. Final fit on FULL history (this is the "deployment" config) ----
    print(f"\n--- FINAL FIT on full history  ({args.final_iters} iters) ---")
    final_cfg = copy.deepcopy(cfg)
    final_cfg.opt_iterations = args.final_iters

    best_so_far = -1e18
    def fprog(t, rec):
        nonlocal best_so_far
        tag = ""
        if rec["objective"] > best_so_far:
            best_so_far = rec["objective"]
            tag = "  <-- new best"
        if t % max(1, args.final_iters // 20) == 0 or tag:
            print(f"  t={t:4d} [{rec['phase']:7s}] J={rec['objective']:+.3f}  "
                  f"respect={rec.get('respect_rate', 0):.2%}  "
                  f"tested={rec.get('tested_n', 0):4d}  n={rec.get('n', 0):4d}{tag}")

    best, _ = optimize(tf, final_cfg, progress=fprog)
    pools = project_to_base(build_pools(tf, best), tf["base"].index)
    results = test_pools(tf["base"], pools, best)
    in_sample = summarise(results)
    _print_summary("FINAL in-sample stats:", in_sample)

    # Two honest numbers from walk-forward:
    #   broad rate  = weak+strong respects vs all breaks
    #   strict rate = strong respects vs strong breaks ONLY (decisive outcomes)
    print(f"\n  in-sample respect:       {in_sample['respect_rate']:.1%}  (optimistic)")
    print(f"  OOS respect (broad):     {wf.oos_respect_pooled:.1%}  "
          f"[{wf.oos_respect_ci_wilson[0]:.1%} – {wf.oos_respect_ci_wilson[1]:.1%}]")
    print(f"  OOS respect (strict):    {wf.oos_respect_strict:.1%}  "
          f"(decisive outcomes only — usually the meaningful signal)")

    # ---- 3. Next pool ----
    current = float(base["close"].iloc[-1])
    atr_proxy = float((base["high"] - base["low"]).rolling(14).mean().iloc[-1])
    print(f"\nlast close: {current:.2f}   (14-bar avg range ≈ {atr_proxy:.2f})")
    print(f"each pool's 'expected respect' is looked up in the STRATIFIED OOS model — "
          f"different pools get different probabilities by feature profile.")

    model = wf.stratified_model
    ml = wf.ml_model
    # Pre-build feature matrix for the final-config pools so we can ML-predict on demand.
    feat = Featurizer(tf["base"]) if ml is not None else None

    def ml_predict_one(pool):
        if ml is None or feat is None:
            return None
        X = feat.transform_batch([pool])
        return float(ml.predict(X, pools=[pool])[0])

    for tag, side in (("ABOVE (sell-side, upside target)", "above"),
                      ("BELOW (buy-side, downside target)", "below")):
        print(f"\n--- NEXT POOL {tag} ---")
        nearest = nearest_untouched(pools, results, current, side, k=3)
        if not nearest:
            print(f"  (no qualifying pool {side} current price)")
            continue
        for rank, (p, r) in enumerate(nearest, 1):
            srcs = sorted({c.source.split('@')[0] for c in p.contributors})
            tf_list = "+".join(p.tfs)
            dist = (p.mid - current) if side == "above" else (current - p.mid)
            dist_atr = dist / max(atr_proxy, 1e-9)
            print(f"  #{rank} zone {p.price_low:.2f}-{p.price_high:.2f}  mid {p.mid:.2f}  "
                  f"score {p.score:.2f}  outcome {r.outcome}")
            print(f"      distance: {'+' if side == 'above' else '-'}{dist_atr:.2f} ATRs")
            print(f"      formed:   {p.formed_at}")
            print(f"      known_at: {p.available_at}")
            print(f"      TFs:      {tf_list}")
            print(f"      drivers:  {', '.join(srcs)}")
            if model is not None:
                bucket, src = model.predict(p)
                print(f"      [stratified] P(respect):  {bucket.rate_broad:.0%}  "
                      f"[{bucket.ci_low:.0%}-{bucket.ci_high:.0%}]   "
                      f"strict={bucket.rate_strict:.0%}   bucket={src}  (n={bucket.tested})")
            ml_p = ml_predict_one(p)
            if ml_p is not None:
                print(f"      [ML model ]  P(respect):  {ml_p:.0%}    "
                      f"(LightGBM + isotonic + bucket-shrinkage, calibrated on OOS)")

    # ---- 4. Artifacts ----
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    safe = cfg.symbol.replace(".", "_")
    html = out / f"{safe}_walkforward.html"
    plot_chart(tf["base"], pools, results=results,
               title=f"{cfg.symbol} — final config (walk-forward OOS respect "
                     f"{wf.oos_respect_pooled:.0%} [{wf.oos_respect_ci_wilson[0]:.0%}–"
                     f"{wf.oos_respect_ci_wilson[1]:.0%}])",
               out_path=str(html), top_n=30)
    (out / f"{safe}_walkforward_report.json").write_text(json.dumps({
        "folds": [{
            "fold": fr.fold + 1,
            "train": [str(fr.train_start), str(fr.train_end)],
            "test": [str(fr.test_start), str(fr.test_end)],
            "train_tested": fr.train_n_tested, "train_respect": fr.train_respect,
            "test_tested": fr.test_n_tested, "test_respect": fr.test_respect,
            "test_respect_ci": [fr.test_respect_ci_low, fr.test_respect_ci_high],
            "overfit_gap": fr.overfit_gap,
            "best_weights": fr.best_weights,
        } for fr in wf.folds],
        "n_total_oos_tested": wf.n_total_oos_tested,
        "oos_respect_pooled": wf.oos_respect_pooled,
        "oos_respect_ci_wilson": list(wf.oos_respect_ci_wilson),
        "oos_respect_ci_bootstrap": list(wf.oos_respect_ci_bootstrap),
        "mean_overfit_gap": wf.mean_overfit_gap,
        "final_in_sample": in_sample,
        "final_best_weights": best.weights.as_dict(),
    }, indent=2, default=str))
    print(f"\nartifacts: {html}")
    print(f"           {out / (safe + '_walkforward_report.json')}")


if __name__ == "__main__":
    main()
