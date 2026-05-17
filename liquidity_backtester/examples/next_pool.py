"""Run multi-resolution liquidity-pool backtest with forward-bias-free testing, then evolve
factor weights, analyse which factors/TFs actually drive respect, re-test with the best combo
as a filter, and report the "next pool" above and below current price.

Run:
    python examples/next_pool.py --symbol HDFCBANK.NS --iters 200
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from liqpool import (
    Config, fetch, multi_timeframe, build_pools, test_pools,
    plot_chart, summarise, correlation,
)
from liqpool.pools import project_to_base
from liqpool.optimizer import optimize


def nearest_untouched(pools, results, current_price: float, side: str, k: int = 3):
    """Top-k highest-score pools on `side` (above/below current price) that the tester classified
    as 'untouched' or 'respected' (i.e. not broken and not horizon-insufficient)."""
    by_idx = {r.pool_idx: r for r in results}
    cand = []
    for i, p in enumerate(pools):
        r = by_idx.get(i)
        if r is None or r.outcome in ("broken", "horizon_insufficient"):
            continue
        if side == "above" and p.price_low > current_price:
            cand.append((p, r))
        if side == "below" and p.price_high < current_price:
            cand.append((p, r))
    cand.sort(key=lambda x: (-x[0].score, abs(x[0].mid - current_price)))
    return cand[:k]


def _print_summary(label: str, s: dict):
    print(f"{label:<26} n={s['n']:4d}  eligible={s.get('eligible_n', s['n']):4d}  "
          f"tested={s['tested_n']:4d}  respect={s['respect_rate']:.2%}  "
          f"break={s['break_rate']:.2%}  untouched={s['untouched_rate']:.2%}  "
          f"horizon_insuff={s.get('horizon_insufficient', 0):3d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="HDFCBANK.NS")
    ap.add_argument("--base", default="5m")
    ap.add_argument("--period", default="60d")
    ap.add_argument("--tfs", default="15min,60min,180min,1D,1W",
                    help="comma-separated higher TFs (default includes 3H)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--min-score", type=float, default=2.0, dest="min_score")
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--min-sample", type=int, default=15, dest="min_sample",
                    help="min #tested pools required for a factor combo to enter the analysis")
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

    # ---- 1. Baseline run (forward-bias-free by construction now) ----
    pools0 = project_to_base(build_pools(tf, cfg), tf["base"].index)
    res0 = test_pools(tf["base"], pools0, cfg)
    _print_summary("baseline (default w):", summarise(res0))

    # ---- 2. Evolve weights + detection params ----
    best_so_far = -1e18

    def progress(t, rec):
        nonlocal best_so_far
        tag = ""
        if rec["objective"] > best_so_far:
            best_so_far = rec["objective"]
            tag = "  <-- new best"
        if t % max(1, cfg.opt_iterations // 25) == 0 or tag:
            print(f"  t={t:4d} [{rec['phase']:7s}] J={rec['objective']:+.3f}  "
                  f"respect={rec.get('respect_rate', 0):.2%}  "
                  f"tested={rec.get('tested_n', 0):4d}  n={rec.get('n', 0):4d}{tag}")

    print("\n--- evolving weights + detection params ---")
    best, log = optimize(tf, cfg, progress=progress)
    pools = project_to_base(build_pools(tf, best), tf["base"].index)
    results = test_pools(tf["base"], pools, best)
    _print_summary("evolved best config :", summarise(results))

    print("\nbest factor weights:")
    for k, v in sorted(best.weights.as_dict().items(), key=lambda x: -x[1]):
        print(f"  {k:24s} {v:.3f}")

    # ---- 3. Factor & TF correlation analysis on the evolved-config pool set ----
    report = correlation.analyze(pools, results, min_sample=args.min_sample, top_k=12)
    correlation.print_report(report)

    # ---- 4. Re-test using the most informative *combination* as a filter ----
    # Strategy: take the top-1 triplet (or pair if no triplet has enough samples) and re-run the
    # respect/break stats restricted to pools that contain it. This proves whether the combo
    # actually adds value beyond the optimiser's all-pools view.
    chosen, chosen_label = None, None
    if report.get("triplets"):
        chosen = set(report["triplets"][0]["factors"])
        chosen_label = "+".join(sorted(chosen))
    elif report.get("pairs"):
        chosen = set(report["pairs"][0]["factors"])
        chosen_label = "+".join(sorted(chosen))

    if chosen:
        fp, fr = correlation.filter_by_required(pools, results, required_factors=chosen,
                                                 min_distinct_tfs=2)
        print(f"\n--- re-test: subset of pools containing [{chosen_label}] with >=2 TFs ---")
        _print_summary(f"filtered ({chosen_label})", summarise(fr))

    # Also report the multi-TF confluence filter on its own (>= 3 distinct TFs)
    fp3, fr3 = correlation.filter_by_required(pools, results, min_distinct_tfs=3)
    print(f"\n--- re-test: subset of pools spanning >= 3 distinct TFs ---")
    _print_summary("filtered (>=3 TFs)", summarise(fr3))

    # ---- 5. Next pool above / below current price ----
    current_price = float(base["close"].iloc[-1])
    atr_proxy = float((base["high"] - base["low"]).rolling(14).mean().iloc[-1])
    print(f"\nlast close: {current_price:.2f}   (14-bar avg range ≈ {atr_proxy:.2f})")

    for tag, side in (("ABOVE (sell-side liquidity, upside target)", "above"),
                      ("BELOW (buy-side liquidity, downside target)", "below")):
        print(f"\n--- NEXT POOL {tag} ---")
        nearest = nearest_untouched(pools, results, current_price, side, k=3)
        if not nearest:
            print(f"  (no qualifying pool {side} current price)")
            continue
        for rank, (p, r) in enumerate(nearest, 1):
            srcs = sorted({c.source.split('@')[0] for c in p.contributors})
            tf_list = "+".join(p.tfs)
            dist = (p.mid - current_price) if side == "above" else (current_price - p.mid)
            dist_atr = dist / max(atr_proxy, 1e-9)
            sign = "+" if side == "above" else "-"
            print(f"  #{rank} zone {p.price_low:.2f}-{p.price_high:.2f}  mid {p.mid:.2f}  "
                  f"score {p.score:.2f}  outcome {r.outcome}")
            print(f"      distance: {sign}{dist_atr:.2f} ATRs from last close")
            print(f"      formed:   {p.formed_at}")
            print(f"      known_at: {p.available_at}   (this is when a trader could first see it)")
            print(f"      TFs:      {tf_list}")
            print(f"      drivers:  {', '.join(srcs)}")

    # ---- 6. Save artifacts ----
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    safe = cfg.symbol.replace(".", "_")
    html = out / f"{safe}_evolved.html"
    plot_chart(tf["base"], pools, results=results,
               title=f"{cfg.symbol} evolved pools (forward-bias-free, {cfg.opt_iterations} trials)",
               out_path=str(html), top_n=30)
    (out / f"{safe}_best_weights.json").write_text(json.dumps(best.weights.as_dict(), indent=2))
    (out / f"{safe}_correlation.json").write_text(json.dumps(report, indent=2, default=str))
    (out / f"{safe}_trials.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"\nartifacts: {html}")


if __name__ == "__main__":
    main()
