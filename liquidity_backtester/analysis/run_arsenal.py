"""Run the alpha arsenal against a saved bundle.

Loads ``output_models/<dir>/multi_asset_report.pkl``, runs every
registered alpha through the shared MIS + Zerodha-cost pipeline, and
writes a comprehensive report:

  * trades.parquet           — one row per signal that survived to a trade
  * per_alpha_summary.csv    — n, win, mean_R, CI95, PF per alpha
  * per_regime_summary.csv   — alpha × session × side stratification with CI95
  * pairwise_combinations.csv— overlap analysis (do alpha pairs agree?)
  * null_tests.csv           — permutation null tests (time-shuffle + sign-flip)

Console: a clear verdict with per-alpha + per-regime summary tables and
the bottom-line "which alphas have statistically significant edge under
null tests."

Usage::

    PYTHONPATH=. .venv/bin/python analysis/run_arsenal.py \\
        --model-dir output_models/core25_latest \\
        --alphas pool_reach,mean_reversion,momentum \\
        --null-trials 100 \\
        --out output_audit/arsenal

The script is heavy on I/O but doesn't retrain any upstream model; just
walks bars + applies alpha logic + triple-barrier labels. ~5-10 min for
a 25-asset bundle.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from liqpool.arsenal import (
    ArsenalEvaluator,
    EvaluatorConfig,
    NullResult,
    default_registry,
    sign_flip_null,
    time_shuffle_null,
)
from liqpool.arsenal.evaluator import _execute_signal
from liqpool.indicators import atr
from liqpool.sectors import sector_of


_PARALLEL_REPORT = None
_PARALLEL_ALPHAS = None
_PARALLEL_CONFIG = None


def _load_bundle(model_dir: Path):
    p = model_dir / "multi_asset_report.pkl"
    if not p.exists():
        raise SystemExit(f"missing bundle: {p}")
    with p.open("rb") as f:
        return pickle.load(f)


def _parallel_asset_worker(symbol: str) -> pd.DataFrame:
    report = _PARALLEL_REPORT
    alphas = _PARALLEL_ALPHAS
    config = _PARALLEL_CONFIG
    if report is None or alphas is None or config is None:
        raise RuntimeError("parallel arsenal worker was not initialised")
    ad = report.assets[symbol]
    df_base = ad.base_df
    if df_base is None or df_base.empty:
        return pd.DataFrame()
    atr_series = atr(df_base, config.atr_period).bfill()
    sec = sector_of(symbol)
    asset_extras = {
        "asset_data": ad,
        "sector": sec,
        "report": report,
    }
    rows = []
    for alpha in alphas:
        signals = alpha.candidates(
            symbol=symbol,
            df_base=df_base,
            atr_series=atr_series,
            extra=asset_extras,
        )
        for sig in signals:
            row = _execute_signal(sig, df_base, atr_series, sec, config)
            if row is None:
                continue
            tags = alpha.regime_tags(sig, df_base, atr_series)
            for k, v in tags.items():
                row[f"regime_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def _run_parallel_arsenal(report, alphas, config: EvaluatorConfig,
                          workers: int) -> pd.DataFrame:
    global _PARALLEL_REPORT, _PARALLEL_ALPHAS, _PARALLEL_CONFIG
    _PARALLEL_REPORT = report
    _PARALLEL_ALPHAS = alphas
    _PARALLEL_CONFIG = config
    symbols = list(report.assets.keys())
    ctx = mp.get_context("fork")
    frames = []
    with ctx.Pool(processes=int(workers)) as pool:
        for frame in pool.imap_unordered(_parallel_asset_worker, symbols):
            if frame is not None and not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _print_per_alpha(summary: pd.DataFrame) -> None:
    print("\n================ PER-ALPHA SUMMARY ================")
    if summary.empty:
        print("  (no trades emitted by any alpha)")
        return
    print(f"  {'alpha':<22} {'n':>6} {'win':>6} {'mean_R':>7} "
          f"{'95% CI mean_R':>16} {'PF':>5} {'avg_cost':>9}")
    for _, r in summary.iterrows():
        pf_s = "inf" if not np.isfinite(r["PF"]) else f"{r['PF']:.2f}"
        ci = f"[{r['ci95_lo']:+.2f},{r['ci95_hi']:+.2f}]"
        print(f"  {r['alpha_name']:<22} {int(r['trades']):>6} "
              f"{r['win']:>5.1%} {r['mean_R']:>+6.2f} {ci:>16} "
              f"{pf_s:>5} ₹{r['avg_cost_inr']:>7.2f}")


def _print_per_regime(per_regime: pd.DataFrame, dim_label: str) -> None:
    print(f"\n================ PER-{dim_label.upper()} STRATIFICATION ================")
    if per_regime.empty:
        print("  (none)")
        return
    print(f"  {'alpha':<22} {'regime':<22} {'n':>6} {'win':>6} "
          f"{'mean_R':>7} {'95% CI':>16}")
    for _, r in per_regime.iterrows():
        ci = f"[{r['ci95_lo']:+.2f},{r['ci95_hi']:+.2f}]"
        print(f"  {r['alpha_name']:<22} {str(r['regime']):<22} {int(r['trades']):>6} "
              f"{r['win']:>5.1%} {r['mean_R']:>+6.2f} {ci:>16}")


def _print_pairwise(pw: pd.DataFrame) -> None:
    print("\n================ PAIRWISE COMBINATIONS ================")
    if pw.empty:
        print("  (no overlapping signals across alpha pairs)")
        return
    print(f"  {'alpha_a':<22} {'alpha_b':<22} {'overlap_n':>9} "
          f"{'combined_mean_R':>16} {'95% CI':>16}")
    for _, r in pw.iterrows():
        ci = f"[{r['ci95_lo']:+.2f},{r['ci95_hi']:+.2f}]"
        print(f"  {r['alpha_a']:<22} {r['alpha_b']:<22} {int(r['overlap_n']):>9} "
              f"{r['combined_mean_R']:>+15.2f} {ci:>16}")


def _print_correlations(corr: pd.DataFrame) -> None:
    print("\n================ DAILY ALPHA CORRELATION ================")
    if corr.empty:
        print("  (not enough daily alpha streams)")
        return
    print(corr.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))


def _print_null_tests(null_results: List[NullResult]) -> None:
    print("\n================ NULL TESTS ================")
    if not null_results:
        print("  (no null tests run)")
        return
    print(f"  {'alpha':<22} {'test':<16} {'actual_R':>9} "
          f"{'null_mean':>10} {'null_std':>9} {'p_value':>8} {'n_trials':>9}")
    for nr in null_results:
        pv = "n/a" if math.isnan(nr.null_p_value) else f"{nr.null_p_value:.3f}"
        print(f"  {nr.alpha_name:<22} {nr.test_name:<16} "
              f"{nr.actual_mean_R:>+8.2f} {nr.null_mean_R_mean:>+9.2f} "
              f"{nr.null_mean_R_std:>9.3f} {pv:>8} {nr.n_trials:>9}")


def _final_verdict(summary: pd.DataFrame,
                   null_results: List[NullResult]) -> List[str]:
    out: List[str] = []
    if summary.empty:
        out.append("No trades produced. Either the bundle has no OOS data "
                   "or every alpha was filtered out by MIS constraints.")
        return out

    positive = summary[summary["ci95_lo"] > 0]
    borderline = summary[(summary["mean_R"] > 0) & (summary["ci95_lo"] <= 0)
                          & (summary["ci95_hi"] > 0)]
    negative = summary[summary["mean_R"] <= 0]

    out.append(f"Alphas evaluated: {len(summary)}")
    out.append(f"  Statistically positive (ci95_lo > 0): {len(positive)}")
    for _, r in positive.iterrows():
        out.append(f"    + {r['alpha_name']:<22} n={int(r['trades']):>5} "
                   f"mean_R={r['mean_R']:+.2f} CI=[{r['ci95_lo']:+.2f},{r['ci95_hi']:+.2f}]")
    out.append(f"  Borderline (positive mean, CI crosses zero): {len(borderline)}")
    out.append(f"  Negative or neutral: {len(negative)}")

    # Null-test verdict per alpha.
    if null_results:
        out.append("")
        out.append("Null-test summary:")
        by_alpha: dict = {}
        for nr in null_results:
            by_alpha.setdefault(nr.alpha_name, []).append(nr)
        for alpha_name, nrs in by_alpha.items():
            time_p = next((nr.null_p_value for nr in nrs
                           if nr.test_name == "time_shuffle"), float("nan"))
            sign_p = next((nr.null_p_value for nr in nrs
                           if nr.test_name == "sign_flip"), float("nan"))
            time_s = "n/a" if math.isnan(time_p) else f"{time_p:.3f}"
            sign_s = "n/a" if math.isnan(sign_p) else f"{sign_p:.3f}"
            # Edge survives both nulls if p < 0.05.
            both_pass = (time_p < 0.05) and (sign_p < 0.05)
            tag = "PASS" if both_pass else ("BORDERLINE" if (time_p < 0.10 or sign_p < 0.10)
                                             else "FAIL")
            out.append(f"  {alpha_name:<22} time-shuffle p={time_s}  "
                       f"sign-flip p={sign_s}  -> {tag}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="output_models/core25_latest")
    ap.add_argument("--alphas", default="",
                    help="comma-separated alpha names; empty = run all registered")
    ap.add_argument("--notional-inr", type=float, default=50_000.0)
    ap.add_argument("--null-trials", type=int, default=100)
    ap.add_argument("--skip-null-tests", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes by asset; Linux/VPS path uses fork")
    ap.add_argument("--out", default="output_audit/arsenal")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[arsenal] loading bundle: {model_dir/'multi_asset_report.pkl'}")
    report = _load_bundle(model_dir)

    registry = default_registry()
    if args.alphas.strip():
        names = [n.strip() for n in args.alphas.split(",") if n.strip()]
        alphas = registry.subset(names)
    else:
        alphas = registry.all()
    print(f"[arsenal] alphas: {[a.name for a in alphas]}")
    for a in alphas:
        print(f"  - {a.name}: {a.description}")

    config = EvaluatorConfig(notional_inr=args.notional_inr)
    evaluator = ArsenalEvaluator(alphas, config=config)

    print(f"\n[arsenal] running evaluator across {len(report.assets)} assets "
          f"(workers={max(1, int(args.workers))})...")
    if int(args.workers) > 1:
        trades = _run_parallel_arsenal(report, alphas, config, int(args.workers))
    else:
        trades = evaluator.run(report)
    print(f"[arsenal] {len(trades):,} trades produced")

    trades_path = out_dir / "trades.parquet"
    if not trades.empty:
        # Convert tz-aware/timestamp columns to str for parquet stability.
        ts_cols = [c for c in trades.columns
                    if trades[c].dtype.kind == "M" or "_at" in c]
        for c in ts_cols:
            trades[c] = trades[c].astype(str)
        trades.to_parquet(trades_path, index=False)

    summary = evaluator.per_alpha_summary(trades)
    per_session = evaluator.per_regime_summary(trades, "regime_session")
    per_side = evaluator.per_regime_summary(trades, "regime_side")
    pw = evaluator.pairwise_combinations(trades)
    daily_returns = evaluator.daily_alpha_returns(trades)
    alpha_corr = evaluator.alpha_correlation(daily_returns)

    summary.to_csv(out_dir / "per_alpha_summary.csv", index=False)
    per_session.to_csv(out_dir / "per_regime_session.csv", index=False)
    per_side.to_csv(out_dir / "per_regime_side.csv", index=False)
    pw.to_csv(out_dir / "pairwise_combinations.csv", index=False)
    daily_returns.to_csv(out_dir / "daily_alpha_returns.csv", index=False)
    alpha_corr.to_csv(out_dir / "alpha_correlation.csv", index=False)

    extra_regime_dims = [
        "regime_factor",
        "regime_tf_bucket",
        "regime_p_touch_bucket",
        "regime_dist_bucket",
        "regime_score_bucket",
        "regime_sector_rotation",
        "regime_vol_regime",
    ]
    for dim in extra_regime_dims:
        reg = evaluator.per_regime_summary(trades, dim)
        if not reg.empty:
            reg.to_csv(out_dir / f"per_{dim}.csv", index=False)

    _print_per_alpha(summary)
    _print_per_regime(per_session, "session")
    _print_per_regime(per_side, "side")
    _print_pairwise(pw)
    _print_correlations(alpha_corr)

    null_results: List[NullResult] = []
    if not args.skip_null_tests:
        print(f"\n[arsenal] running null tests ({args.null_trials} trials each)...")
        for alpha in alphas:
            t = time_shuffle_null(alpha, report, config,
                                  n_trials=args.null_trials, seed=17)
            s = sign_flip_null(alpha, report, config,
                                n_trials=max(50, args.null_trials // 2),
                                seed=23)
            null_results.append(t)
            null_results.append(s)
        null_df = pd.DataFrame([nr.__dict__ for nr in null_results])
        null_df.to_csv(out_dir / "null_tests.csv", index=False)
    _print_null_tests(null_results)

    print("\n================ VERDICT ================")
    for line in _final_verdict(summary, null_results):
        print(line)
    print(f"\n[arsenal] artifacts written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
