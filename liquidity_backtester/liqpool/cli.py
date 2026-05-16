"""Command-line entry point.

Usage:
    python -m liqpool.cli run     --symbol AAPL --period 60d --base 5m
    python -m liqpool.cli optimize --symbol AAPL --period 60d --iters 100
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

from .config import Config
from .data import fetch, multi_timeframe
from .pools import build_pools, project_to_base
from .tester import test_pools, summarise
from .optimizer import optimize
from .plotting import plot_chart


def _make_config(args) -> Config:
    cfg = Config()
    cfg.symbol = args.symbol
    cfg.base_interval = args.base
    cfg.period = args.period
    cfg.start = args.start
    cfg.end = args.end
    if args.tfs:
        cfg.higher_tfs = args.tfs.split(",")
    if args.horizon is not None:
        cfg.test_horizon_bars = args.horizon
    if args.min_score is not None:
        cfg.min_pool_score = args.min_score
    if args.iters is not None:
        cfg.opt_iterations = args.iters
    return cfg


def cmd_run(args):
    cfg = _make_config(args)
    print(f"[run] {cfg.symbol} {cfg.base_interval} period={cfg.period} tfs={cfg.higher_tfs}")
    base = fetch(cfg.symbol, interval=cfg.base_interval, period=cfg.period,
                 start=cfg.start, end=cfg.end)
    print(f"[run] base bars: {len(base)}  range: {base.index[0]} → {base.index[-1]}")
    tf_data = multi_timeframe(base, cfg.higher_tfs)
    for tf, df in tf_data.items():
        print(f"  {tf}: {len(df)} bars")

    pools = build_pools(tf_data, cfg)
    pools = project_to_base(pools, tf_data["base"].index)
    print(f"[pools] {len(pools)} pools (score >= {cfg.min_pool_score})")
    results = test_pools(tf_data["base"], pools, cfg)
    stats = summarise(results)
    print(f"[test]  {json.dumps(stats, default=str)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"{cfg.symbol}_{cfg.base_interval}_pools.html"
    plot_chart(tf_data["base"], pools, results=results,
               title=f"{cfg.symbol} {cfg.base_interval} liquidity pools",
               out_path=str(html), top_n=args.top_n)
    print(f"[plot]  wrote {html}")

    pools_path = out_dir / f"{cfg.symbol}_pools.json"
    pools_path.write_text(json.dumps([p.summary() for p in pools], indent=2, default=str))
    results_path = out_dir / f"{cfg.symbol}_results.json"
    results_path.write_text(json.dumps([r.as_dict() for r in results], indent=2, default=str))
    print(f"[save]  {pools_path}  {results_path}")


def cmd_optimize(args):
    cfg = _make_config(args)
    base = fetch(cfg.symbol, interval=cfg.base_interval, period=cfg.period,
                 start=cfg.start, end=cfg.end)
    tf_data = multi_timeframe(base, cfg.higher_tfs)

    def progress(t, rec):
        if t % max(1, cfg.opt_iterations // 20) == 0 or t == cfg.opt_iterations - 1:
            print(f"  trial {t:4d} [{rec['phase']}] obj={rec['objective']:+.3f} "
                  f"respect={rec.get('respect_rate', 0):.2%} tested={rec.get('tested_n', 0)} "
                  f"n={rec.get('n', 0)}")

    best_cfg, log = optimize(tf_data, cfg, progress=progress)
    print("[opt]  best objective:", max(r["objective"] for r in log))
    print("[opt]  best weights:", best_cfg.weights.as_dict())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{cfg.symbol}_opt_trials.json"
    log_path.write_text(json.dumps(log, indent=2, default=str))

    # Final run with best cfg
    pools = build_pools(tf_data, best_cfg)
    pools = project_to_base(pools, tf_data["base"].index)
    results = test_pools(tf_data["base"], pools, best_cfg)
    stats = summarise(results)
    print(f"[opt:final] {json.dumps(stats, default=str)}")

    html = out_dir / f"{cfg.symbol}_best_pools.html"
    plot_chart(tf_data["base"], pools, results=results,
               title=f"{cfg.symbol} best-config liquidity pools",
               out_path=str(html), top_n=args.top_n)
    print(f"[opt:plot] {html}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="liqpool")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--symbol", required=True)
        sp.add_argument("--base", default="5m")
        sp.add_argument("--period", default="60d")
        sp.add_argument("--start", default=None)
        sp.add_argument("--end", default=None)
        sp.add_argument("--tfs", default=None, help="comma-separated higher timeframes")
        sp.add_argument("--horizon", type=int, default=None, help="test horizon in base bars")
        sp.add_argument("--min-score", type=float, default=None, dest="min_score")
        sp.add_argument("--iters", type=int, default=None, dest="iters")
        sp.add_argument("--top-n", type=int, default=25, dest="top_n")
        sp.add_argument("--out", default="output")

    sr = sub.add_parser("run"); add_common(sr); sr.set_defaults(func=cmd_run)
    so = sub.add_parser("optimize"); add_common(so); so.set_defaults(func=cmd_optimize)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
