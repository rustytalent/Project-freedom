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
from liqpool.timing import StateFeaturizer


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
    dir_m = wf.direction_model
    prox_models = wf.proximity_models if wf.proximity_models else {}
    # Pre-build feature matrix for the final-config pools so we can ML-predict on demand.
    feat = Featurizer(tf["base"]) if ml is not None else None
    state_feat = StateFeaturizer(tf["base"]) if dir_m is not None else None

    # Compute the current state vector ONCE (most recent bar).
    current_state = None
    direction_p_up = None
    if state_feat is not None:
        j_now = len(tf["base"]) - 1
        live = [p for p, r in zip(pools, results)
                if p.available_at <= tf["base"].index[j_now] and not r.is_break]
        current_state = state_feat.features_at(j_now, live)
        if dir_m is not None:
            direction_p_up = dir_m.predict_state(current_state)

    # Direction interpretation thresholds. With AUC ~0.70 and top-quartile-confidence ~89%,
    # |p-0.5| >= 0.10 is meaningful; we use 0.10 as the alignment threshold.
    DIR_ALIGN_MARGIN = 0.10

    def direction_tag(side_str: str) -> str:
        if direction_p_up is None:
            return ""
        if direction_p_up >= 0.5 + DIR_ALIGN_MARGIN:
            return "DIR_ALIGN" if side_str == "above" else "DIR_FIGHT"
        if direction_p_up <= 0.5 - DIR_ALIGN_MARGIN:
            return "DIR_ALIGN" if side_str == "below" else "DIR_FIGHT"
        return "DIR_NEUTRAL"

    if direction_p_up is not None:
        dir_word = "UP" if direction_p_up >= 0.5 else "DOWN"
        conf_word = "STRONG" if abs(direction_p_up - 0.5) >= DIR_ALIGN_MARGIN else "WEAK"
        auc_str = (f"OOS AUC {wf.timing_report.direction_auc:.2f}, "
                   f"top-quartile-confidence acc "
                   f"{wf.timing_report.direction_top_quartile_acc:.0%}"
                   if wf.timing_report else "")
        print(f"\n[direction prediction at now]  P(up | next "
              f"{dir_m.horizon} bars) = {direction_p_up:.1%}   "
              f"[{conf_word} {dir_word}]   ({auc_str})")

    def ml_predict_one(pool):
        if ml is None or feat is None:
            return None
        X = feat.transform_batch([pool])
        return float(ml.predict(X, pools=[pool])[0])

    def prox_predict(pool, dist_atr, side_str, horizon):
        pm = prox_models.get(horizon)
        if pm is None or current_state is None:
            return None
        q = ml_predict_one(pool) or 0.5
        return pm.predict_one(pool, dist_atr, side_str, current_state, q)

    # Tradeable filter thresholds — tunable.
    TRADEABLE_T_TODAY = 0.05      # 5% minimum touch probability today to be tradeable today
    TRADEABLE_Q = 0.55            # 55% minimum quality

    sorted_horizons = sorted(prox_models.keys())     # e.g. [78, 156, 312]
    primary_h = sorted_horizons[0] if sorted_horizons else None    # "today"

    # Gather candidates from both sides.
    candidates = []
    for side_str in ("above", "below"):
        nearest = nearest_untouched(pools, results, current, side_str, k=5)
        for p, r in nearest:
            dist = (p.mid - current) if side_str == "above" else (current - p.mid)
            dist_atr = dist / max(atr_proxy, 1e-9)
            q = ml_predict_one(p)
            t_by_h = {}
            for h in sorted_horizons:
                t_by_h[h] = prox_predict(p, dist_atr, side_str, h)
            candidates.append({
                "pool": p, "result": r, "side": side_str,
                "dist_atr": dist_atr, "q": q, "t_by_h": t_by_h,
                "dir_tag": direction_tag(side_str),
            })

    # Best Setup Today: tradeable + highest expected value (Q × T_today × direction-alignment bonus)
    def ev_score(c):
        q = c.get("q") or 0.0
        t = c["t_by_h"].get(primary_h) if primary_h else None
        if t is None:
            return 0.0
        bonus = 1.0
        if c["dir_tag"] == "DIR_ALIGN":
            bonus = 1.15
        elif c["dir_tag"] == "DIR_FIGHT":
            bonus = 0.85
        return q * t * bonus

    tradeable = [c for c in candidates
                  if c.get("q") is not None
                  and primary_h is not None
                  and c["t_by_h"].get(primary_h) is not None
                  and c["t_by_h"][primary_h] >= TRADEABLE_T_TODAY
                  and c["q"] >= TRADEABLE_Q]
    watchlist = [c for c in candidates if c not in tradeable]

    tradeable.sort(key=ev_score, reverse=True)
    watchlist.sort(key=lambda c: -(c.get("q") or 0.0))

    # ===== BEST SETUP TODAY =====
    print("\n================ TODAY'S TRADING PLAN ================")
    if tradeable:
        best = tradeable[0]
        p = best["pool"]; r = best["result"]
        srcs = sorted({c.source.split('@')[0] for c in p.contributors})
        side_label = "BELOW (buy)" if best["side"] == "below" else "ABOVE (sell)"
        t_today = best["t_by_h"].get(primary_h)
        q = best["q"]
        print(f">>> BEST SETUP TODAY <<<")
        print(f"  Pool ₹{p.price_low:.2f}-{p.price_high:.2f}  mid ₹{p.mid:.2f}  "
              f"[{side_label}]")
        print(f"  Distance: {best['dist_atr']:.2f} ATRs from current ₹{current:.2f}  "
              f"({best['dir_tag']})")
        print(f"  Quality Q = {q:.1%}   Touch T_today = {t_today:.1%}   "
              f"EV(today) ≈ {q * t_today:.1%}")
        print(f"  Drivers: {', '.join(srcs)}   |  TFs: {'+'.join(p.tfs)}")
        if best["side"] == "below":
            print(f"  Action: LIMIT BUY at ₹{p.price_high:.2f}, stop "
                  f"₹{p.price_low - 0.5 * atr_proxy:.2f}, "
                  f"target ₹{p.price_high + 2 * atr_proxy:.2f}+")
        else:
            print(f"  Action: LIMIT SELL at ₹{p.price_low:.2f}, stop "
                  f"₹{p.price_high + 0.5 * atr_proxy:.2f}, "
                  f"target ₹{p.price_low - 2 * atr_proxy:.2f}-")
    else:
        print(">>> NO TRADEABLE SETUP TODAY <<<")
        print(f"  No pool has both T_today >= {TRADEABLE_T_TODAY:.0%} "
              f"AND Q >= {TRADEABLE_Q:.0%}.")
        print(f"  Watch the levels below for tomorrow / later in the week.")
    print("======================================================")

    # ===== TRADEABLE TODAY =====
    print(f"\n--- TRADEABLE TODAY  (T_today >= {TRADEABLE_T_TODAY:.0%} AND "
          f"Q >= {TRADEABLE_Q:.0%}) ---")
    if not tradeable:
        print("  (none — all pools are either too far or low-quality)")
    else:
        for rank, c in enumerate(tradeable, 1):
            p = c["pool"]; r = c["result"]
            srcs = sorted({c2.source.split('@')[0] for c2 in p.contributors})
            side_str_long = "BELOW" if c["side"] == "below" else "ABOVE"
            t_today = c["t_by_h"].get(primary_h)
            q = c["q"]
            t_strs = "  ".join(
                f"T_h{h}={c['t_by_h'][h]:.1%}" for h in sorted_horizons
                if c['t_by_h'].get(h) is not None
            )
            print(f"  #{rank} ₹{p.price_low:.2f}-{p.price_high:.2f} "
                  f"[{side_str_long}, {c['dist_atr']:.1f} ATR, {c['dir_tag']}]")
            print(f"       Q={q:.1%}   {t_strs}   "
                  f"EV(today)={q * t_today:.1%}   outcome_was={r.outcome}")
            print(f"       drivers: {', '.join(srcs)}   TFs: {'+'.join(p.tfs)}")

    # ===== WATCH LIST =====
    print(f"\n--- WATCH LIST  (T_today < {TRADEABLE_T_TODAY:.0%} or Q < {TRADEABLE_Q:.0%}) ---")
    if not watchlist:
        print("  (none)")
    else:
        for c in watchlist[:6]:
            p = c["pool"]
            side_str_long = "BELOW" if c["side"] == "below" else "ABOVE"
            q = c.get("q") or 0.0
            t_strs = "  ".join(
                f"T_h{h}={c['t_by_h'][h]:.1%}" if c['t_by_h'].get(h) is not None else f"T_h{h}=n/a"
                for h in sorted_horizons
            )
            print(f"  ₹{p.price_low:.2f}-{p.price_high:.2f} "
                  f"[{side_str_long}, {c['dist_atr']:.1f} ATR, {c['dir_tag']}]   "
                  f"Q={q:.1%}   {t_strs}")

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
