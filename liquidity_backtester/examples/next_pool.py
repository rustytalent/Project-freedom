"""Run long evolutionary optimisation and report the 'next pool' above and below current price.

Run:  python -m examples.next_pool --symbol HDFCBANK.NS --iters 200
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from liqpool import Config, fetch, multi_timeframe, build_pools, test_pools, plot_chart
from liqpool.pools import project_to_base
from liqpool.tester import summarise
from liqpool.optimizer import optimize


def nearest_untouched(pools, results, current_price: float, side: str, k: int = 3):
    """Return up to k highest-score pools on `side` (above/below current price) that the tester
    classified as untouched or respected (i.e. not broken). Sorted by score then by proximity."""
    by_idx = {r.pool_idx: r for r in results}
    cand = []
    for i, p in enumerate(pools):
        r = by_idx.get(i)
        if r is None or r.outcome == "broken":
            continue
        if side == "above" and p.price_low > current_price:
            cand.append((p, r))
        if side == "below" and p.price_high < current_price:
            cand.append((p, r))
    cand.sort(key=lambda x: (-x[0].score, abs(x[0].mid - current_price)))
    return cand[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="HDFCBANK.NS")
    ap.add_argument("--base", default="5m")
    ap.add_argument("--period", default="60d")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--min-score", type=float, default=2.0)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    cfg = Config(
        symbol=args.symbol, base_interval=args.base, period=args.period,
        higher_tfs=["15min", "60min", "240min", "1D", "1W"],
        test_horizon_bars=args.horizon,
        opt_iterations=args.iters, opt_explore_frac=0.35, opt_seed=11,
        min_pool_score=args.min_score,
    )

    base = fetch(cfg.symbol, cfg.base_interval, cfg.period)
    tf = multi_timeframe(base, cfg.higher_tfs)
    print(f"\n=== {cfg.symbol} {cfg.base_interval}  bars: {len(base)}  "
          f"range: {base.index[0]} → {base.index[-1]}")

    # Baseline (default weights) for comparison
    pools0 = project_to_base(build_pools(tf, cfg), tf["base"].index)
    res0 = test_pools(tf["base"], pools0, cfg)
    s0 = summarise(res0)
    print(f"baseline   : n={s0['n']:4d}  tested={s0['tested_n']:4d}  "
          f"respect={s0['respect_rate']:.2%}  break={s0['break_rate']:.2%}  "
          f"untouched={s0['untouched_rate']:.2%}")

    # Evolve
    best_so_far = -1e18
    trail = []

    def progress(t, rec):
        nonlocal best_so_far
        improved = ""
        if rec["objective"] > best_so_far:
            best_so_far = rec["objective"]
            improved = "  <-- new best"
        trail.append((t, rec["phase"], rec["objective"], rec.get("respect_rate", 0),
                      rec.get("tested_n", 0), rec.get("n", 0)))
        if t % max(1, cfg.opt_iterations // 25) == 0 or improved:
            print(f"  t={t:4d} [{rec['phase']:7s}] J={rec['objective']:+.3f}  "
                  f"respect={rec.get('respect_rate', 0):.2%}  "
                  f"tested={rec.get('tested_n', 0):4d}  n={rec.get('n', 0):4d}{improved}")

    best, log = optimize(tf, cfg, progress=progress)
    pools = project_to_base(build_pools(tf, best), tf["base"].index)
    results = test_pools(tf["base"], pools, best)
    sB = summarise(results)
    print(f"\nbest-config: n={sB['n']:4d}  tested={sB['tested_n']:4d}  "
          f"respect={sB['respect_rate']:.2%}  break={sB['break_rate']:.2%}  "
          f"untouched={sB['untouched_rate']:.2%}")

    print("\nbest factor weights (the evolution converged here):")
    for k, v in sorted(best.weights.as_dict().items(), key=lambda x: -x[1]):
        print(f"  {k:24s} {v:.3f}")

    current_price = float(base["close"].iloc[-1])
    print(f"\nlast close: {current_price:.2f}")

    print("\n--- NEXT POOL ABOVE (sell-side liquidity, upside target) ---")
    above = nearest_untouched(pools, results, current_price, "above", k=3)
    if not above:
        print("  (no qualifying pool above current price)")
    for rank, (p, r) in enumerate(above, 1):
        srcs = sorted({c.source.split('@')[0] for c in p.contributors})
        tfs_str = "+".join(p.tfs)
        dist_atr = (p.mid - current_price) / max((base['high'] - base['low']).rolling(14).mean().iloc[-1], 1e-9)
        print(f"  #{rank} zone {p.price_low:.2f}-{p.price_high:.2f}  mid {p.mid:.2f}  "
              f"score {p.score:.2f}  outcome {r.outcome}")
        print(f"      distance: {dist_atr:+.2f} ATRs above last close")
        print(f"      formed:   {p.formed_at}")
        print(f"      TFs:      {tfs_str}")
        print(f"      drivers:  {', '.join(srcs)}")

    print("\n--- NEXT POOL BELOW (buy-side liquidity, downside target) ---")
    below = nearest_untouched(pools, results, current_price, "below", k=3)
    if not below:
        print("  (no qualifying pool below current price)")
    for rank, (p, r) in enumerate(below, 1):
        srcs = sorted({c.source.split('@')[0] for c in p.contributors})
        tfs_str = "+".join(p.tfs)
        dist_atr = (current_price - p.mid) / max((base['high'] - base['low']).rolling(14).mean().iloc[-1], 1e-9)
        print(f"  #{rank} zone {p.price_low:.2f}-{p.price_high:.2f}  mid {p.mid:.2f}  "
              f"score {p.score:.2f}  outcome {r.outcome}")
        print(f"      distance: -{dist_atr:.2f} ATRs below last close")
        print(f"      formed:   {p.formed_at}")
        print(f"      TFs:      {tfs_str}")
        print(f"      drivers:  {', '.join(srcs)}")

    # Save artifacts
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    html = out / f"{cfg.symbol.replace('.', '_')}_evolved.html"
    plot_chart(tf["base"], pools, results=results,
               title=f"{cfg.symbol} evolved pools (after {cfg.opt_iterations} trials)",
               out_path=str(html), top_n=30)
    (out / f"{cfg.symbol.replace('.', '_')}_best_weights.json").write_text(
        json.dumps(best.weights.as_dict(), indent=2))
    (out / f"{cfg.symbol.replace('.', '_')}_trials.json").write_text(
        json.dumps(log, indent=2, default=str))

    # Convergence curve in text
    print("\n--- evolution trace (best-so-far respect rate every ~5% of trials) ---")
    best_j, best_r = -1e18, 0.0
    step = max(1, cfg.opt_iterations // 20)
    for t, phase, j, rr, tested, n in trail:
        if j > best_j:
            best_j, best_r = j, rr
        if t % step == 0:
            print(f"  t={t:4d} best_J={best_j:+.3f} best_respect={best_r:.2%}")

    print(f"\nartifacts: {html}")


if __name__ == "__main__":
    main()
