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
from typing import Dict, List, Optional, Tuple

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


def _resolve_model_dir(bundle: str, model_dir: str) -> Path:
    if bundle:
        p = Path(bundle).expanduser()
        if (p / "multi_asset_report.pkl").exists():
            return p
        if p.name == str(p):
            return Path("output_models") / str(p)
        return p
    return Path(model_dir).expanduser()


def _load_bundle(model_dir: Path):
    p = model_dir / "multi_asset_report.pkl"
    if not p.exists():
        raise SystemExit(f"missing bundle: {p}")
    with p.open("rb") as f:
        return pickle.load(f)


def _normal_p_gt_zero(mean_r: float, se_r: float) -> float:
    if not np.isfinite(mean_r) or not np.isfinite(se_r) or se_r <= 0:
        return 1.0 if mean_r <= 0 else 0.0
    z = mean_r / se_r
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def _summary_with_pvalues(trades: pd.DataFrame,
                          p_threshold: float) -> pd.DataFrame:
    summary = ArsenalEvaluator.per_alpha_summary(trades)
    if summary.empty:
        return summary
    pvals = []
    for _, row in summary.iterrows():
        p = _normal_p_gt_zero(float(row["mean_R"]), float(row["se_R"]))
        pvals.append(p)
    summary["p_mean_R_gt_0"] = pvals
    summary["p_threshold"] = float(p_threshold)
    summary["passes_p_threshold"] = summary["p_mean_R_gt_0"] < float(p_threshold)
    return summary


def _split_holdout(trades: pd.DataFrame, holdout_frac: float
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    meta = {
        "requested_holdout_frac": float(holdout_frac),
        "method": "bonferroni",
        "used_holdout": False,
        "reason": "holdout_frac<=0 or insufficient dated trades",
        "cutoff": None,
        "span_days": 0.0,
    }
    if trades.empty or holdout_frac <= 0 or "entry_at" not in trades.columns:
        return trades.copy(), trades.iloc[0:0].copy(), meta
    ts = pd.to_datetime(trades["entry_at"], errors="coerce")
    valid = ts.notna()
    if not valid.any():
        meta["reason"] = "entry_at could not be parsed"
        return trades.copy(), trades.iloc[0:0].copy(), meta
    span_days = float((ts[valid].max() - ts[valid].min()).total_seconds() / 86400.0)
    meta["span_days"] = span_days
    if span_days < 180:
        meta["reason"] = "OOS span < 180 days, using Bonferroni instead of holdout"
        return trades.copy(), trades.iloc[0:0].copy(), meta

    valid_ts = ts[valid].sort_values()
    q = max(0.01, min(0.99, 1.0 - holdout_frac))
    cutoff_idx = min(len(valid_ts) - 1, max(0, int(math.floor(len(valid_ts) * q)) - 1))
    cutoff = valid_ts.iloc[cutoff_idx]
    selection = trades[ts <= cutoff].copy()
    holdout = trades[ts > cutoff].copy()
    if selection.empty or holdout.empty:
        meta["reason"] = "chronological split produced an empty side"
        return trades.copy(), trades.iloc[0:0].copy(), meta
    meta.update({
        "method": "chronological_holdout",
        "used_holdout": True,
        "reason": "last OOS slice reserved before alpha selection",
        "cutoff": str(cutoff),
    })
    return selection, holdout, meta


def _per_alpha_turnover(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    ts = pd.to_datetime(trades["entry_at"], errors="coerce")
    frame = trades.copy()
    frame["_entry_ts"] = ts
    for alpha_name, g in frame.groupby("alpha_name"):
        n = int(len(g))
        symbols = max(1, int(g["symbol"].nunique()))
        dated = g["_entry_ts"].dropna()
        if len(dated) >= 2:
            months = max(1.0 / 21.0, float((dated.max() - dated.min()).days) / 30.4375)
        else:
            months = 1.0 / 21.0
        rows.append({
            "alpha_name": alpha_name,
            "trades": n,
            "unique_symbols": symbols,
            "mean_holding_bars": float(pd.to_numeric(g["bars_held"], errors="coerce").mean()),
            "entries_per_asset_month": float(n / max(symbols * months, 1e-9)),
            "too_sparse_n_lt_200": bool(n < 200),
        })
    return pd.DataFrame(rows).sort_values("alpha_name").reset_index(drop=True)


def _cost_multiplier_summary(trades: pd.DataFrame,
                             multipliers: List[float],
                             p_threshold: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for alpha_name, g in trades.groupby("alpha_name"):
        base_r = pd.to_numeric(g["net_r"], errors="coerce")
        cost = pd.to_numeric(g["cost_inr"], errors="coerce")
        risk = pd.to_numeric(g["risk_inr"], errors="coerce").replace(0, np.nan)
        for m in multipliers:
            stressed = base_r - (cost * (float(m) - 1.0) / risk)
            stressed = stressed.replace([np.inf, -np.inf], np.nan).dropna()
            n = int(len(stressed))
            mean_r = float(stressed.mean()) if n else 0.0
            se = float(stressed.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            p = _normal_p_gt_zero(mean_r, se)
            wins = stressed > 0
            rows.append({
                "alpha_name": alpha_name,
                "cost_multiplier": float(m),
                "trades": n,
                "win": float(wins.mean()) if n else 0.0,
                "mean_R": mean_r,
                "se_R": se,
                "ci95_lo": mean_r - 1.96 * se,
                "ci95_hi": mean_r + 1.96 * se,
                "p_mean_R_gt_0": p,
                "p_threshold": float(p_threshold),
                "passes_cost_wall": bool(mean_r > 0 and p < p_threshold),
            })
    return pd.DataFrame(rows).sort_values(
        ["cost_multiplier", "mean_R"], ascending=[True, False]
    ).reset_index(drop=True)


def _pool_volume_effect(trades: pd.DataFrame,
                        p_threshold: float,
                        alpha_name: str = "pool_reach") -> pd.DataFrame:
    required = {
        "alpha_name", "pool_volume_confirmed_at_touch", "pool_q_pred",
        "confidence", "net_r",
    }
    if trades.empty or not required.issubset(set(trades.columns)):
        return pd.DataFrame()
    frame = trades[trades["alpha_name"].astype(str) == alpha_name].copy()
    if frame.empty:
        return pd.DataFrame()
    frame = frame[frame["pool_volume_confirmed_at_touch"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()

    def _to_bool(value) -> Optional[bool]:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "1", "yes"}:
                return True
            if low in {"false", "0", "no"}:
                return False
        if pd.isna(value):
            return None
        return bool(value)

    frame["_pool_volume_confirmed_bool"] = frame[
        "pool_volume_confirmed_at_touch"
    ].map(_to_bool)
    frame = frame[frame["_pool_volume_confirmed_bool"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for confirmed, g in frame.groupby("_pool_volume_confirmed_bool"):
        r = pd.to_numeric(g["net_r"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        n = int(len(r))
        mean_r = float(r.mean()) if n else 0.0
        se = float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        p = _normal_p_gt_zero(mean_r, se)
        rows.append({
            "population": alpha_name,
            "pool_volume_confirmed_at_touch": bool(confirmed),
            "trades": n,
            "mean_pool_q_pred": float(pd.to_numeric(g["pool_q_pred"], errors="coerce").mean()),
            "mean_confidence": float(pd.to_numeric(g["confidence"], errors="coerce").mean()),
            "mean_R": mean_r,
            "se_R": se,
            "ci95_lo": mean_r - 1.96 * se,
            "ci95_hi": mean_r + 1.96 * se,
            "hit_rate": float((r > 0).mean()) if n else 0.0,
            "p_mean_R_gt_0": p,
            "p_threshold": float(p_threshold),
            "passes_p_threshold": bool(mean_r > 0 and p < p_threshold),
        })
    df = pd.DataFrame(rows).sort_values(
        "pool_volume_confirmed_at_touch", ascending=False
    ).reset_index(drop=True)
    present = set(df["pool_volume_confirmed_at_touch"].map(_to_bool).dropna())
    if {True, False}.issubset(present):
        confirmed_mask = df["pool_volume_confirmed_at_touch"].astype(bool)
        yes = df[confirmed_mask].iloc[0]
        no = df[df["pool_volume_confirmed_at_touch"].map(_to_bool) == False].iloc[0]
        delta = {
            "population": alpha_name,
            "pool_volume_confirmed_at_touch": "delta_true_minus_false",
            "trades": int(yes["trades"]) - int(no["trades"]),
            "mean_pool_q_pred": float(yes["mean_pool_q_pred"]) - float(no["mean_pool_q_pred"]),
            "mean_confidence": float(yes["mean_confidence"]) - float(no["mean_confidence"]),
            "mean_R": float(yes["mean_R"]) - float(no["mean_R"]),
            "se_R": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
            "hit_rate": float(yes["hit_rate"]) - float(no["hit_rate"]),
            "p_mean_R_gt_0": float("nan"),
            "p_threshold": float(p_threshold),
            "passes_p_threshold": False,
        }
        df = pd.concat([df, pd.DataFrame([delta])], ignore_index=True)
    return df


def _high_correlation_pairs(alpha_corr: pd.DataFrame,
                            threshold: float = 0.80) -> pd.DataFrame:
    if alpha_corr.empty or "alpha" not in alpha_corr.columns:
        return pd.DataFrame()
    rows = []
    matrix = alpha_corr.set_index("alpha")
    names = [str(x) for x in matrix.index]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if b not in matrix.columns:
                continue
            corr = matrix.loc[a, b]
            if pd.isna(corr):
                continue
            corr_f = float(corr)
            if abs(corr_f) >= float(threshold):
                rows.append({
                    "alpha_a": a,
                    "alpha_b": b,
                    "corr": corr_f,
                    "duplicate_candidate": bool(corr_f > 0),
                    "threshold": float(threshold),
                })
    if not rows:
        return pd.DataFrame(columns=[
            "alpha_a", "alpha_b", "corr", "duplicate_candidate", "threshold",
        ])
    return pd.DataFrame(rows).sort_values(
        "corr", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def _confidence_decile_lift(g: pd.DataFrame) -> float:
    if g.empty or "confidence" not in g.columns or len(g) < 20:
        return float("nan")
    ordered = g.sort_values("confidence")
    n = max(1, len(ordered) // 10)
    bottom = float(pd.to_numeric(ordered.head(n)["net_r"], errors="coerce").mean())
    top = float(pd.to_numeric(ordered.tail(n)["net_r"], errors="coerce").mean())
    if abs(bottom) < 1e-9:
        return float("inf") if top > 0 else float("nan")
    return float(top / bottom)


def _baseline_delta(trades: pd.DataFrame,
                    baseline: str = "proximity_journey_baseline",
                    upgraded: str = "proximity_journey") -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for name in (baseline, upgraded):
        g = trades[trades["alpha_name"] == name]
        if g.empty:
            rows.append({"alpha_name": name, "missing": True})
            continue
        r = pd.to_numeric(g["net_r"], errors="coerce").dropna()
        rows.append({
            "alpha_name": name,
            "missing": False,
            "trades": int(len(r)),
            "mean_R": float(r.mean()) if len(r) else 0.0,
            "hit_rate": float((r > 0).mean()) if len(r) else 0.0,
            "sharpe": float(r.mean() / r.std(ddof=1)) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0,
            "confidence_decile_lift": _confidence_decile_lift(g),
        })
    df = pd.DataFrame(rows)
    if len(df) == 2 and not bool(df["missing"].any()):
        b = df[df["alpha_name"] == baseline].iloc[0]
        u = df[df["alpha_name"] == upgraded].iloc[0]
        df = pd.concat([df, pd.DataFrame([{
            "alpha_name": "delta_upgraded_minus_baseline",
            "missing": False,
            "trades": int(u["trades"]) - int(b["trades"]),
            "mean_R": float(u["mean_R"]) - float(b["mean_R"]),
            "hit_rate": float(u["hit_rate"]) - float(b["hit_rate"]),
            "sharpe": float(u["sharpe"]) - float(b["sharpe"]),
            "confidence_decile_lift": (
                float(u["confidence_decile_lift"]) - float(b["confidence_decile_lift"])
                if np.isfinite(float(u["confidence_decile_lift"]))
                and np.isfinite(float(b["confidence_decile_lift"]))
                else float("nan")
            ),
        }])], ignore_index=True)
    return df


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
    try:
        from liqpool.timing import StateFeaturizer
        state_featurizer = StateFeaturizer(df_base)
    except Exception:
        state_featurizer = None
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
            row = _execute_signal(
                sig, df_base, atr_series, sec, config,
                state_featurizer=state_featurizer,
            )
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
                   null_results: List[NullResult],
                   p_threshold: float = 0.05,
                   method_meta: Optional[Dict] = None) -> List[str]:
    out: List[str] = []
    if summary.empty:
        out.append("No trades produced. Either the bundle has no OOS data "
                   "or every alpha was filtered out by MIS constraints.")
        return out

    positive = summary[summary["ci95_lo"] > 0]
    borderline = summary[(summary["mean_R"] > 0) & (summary["ci95_lo"] <= 0)
                          & (summary["ci95_hi"] > 0)]
    negative = summary[summary["mean_R"] <= 0]

    method_meta = method_meta or {}
    out.append(f"Alphas evaluated: {len(summary)}")
    out.append(f"Methodology: {method_meta.get('method', 'bonferroni')} "
               f"(p_threshold={p_threshold:.5f})")
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
            # Edge survives both nulls if p clears the selected threshold.
            both_pass = (time_p < p_threshold) and (sign_p < p_threshold)
            tag = "PASS" if both_pass else ("BORDERLINE" if (time_p < 0.10 or sign_p < 0.10)
                                             else "FAIL")
            out.append(f"  {alpha_name:<22} time-shuffle p={time_s}  "
                       f"sign-flip p={sign_s}  -> {tag}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="",
                    help="bundle name under output_models/ or explicit model dir")
    ap.add_argument("--model-dir", default="output_models/core25_latest")
    ap.add_argument("--alphas", default="",
                    help="comma-separated alpha names; empty = run all registered")
    ap.add_argument("--notional-inr", type=float, default=50_000.0)
    ap.add_argument("--null-trials", type=int, default=100)
    ap.add_argument("--skip-null-tests", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes by asset; Linux/VPS path uses fork")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="reserve the last chronological fraction for final alpha reporting when OOS span >= 180 days")
    ap.add_argument("--cost-multipliers", default="1.0,1.25,1.5,2.0",
                    help="comma-separated cost stress multipliers for alpha summaries")
    ap.add_argument("--emit-alpha-correlation-matrix", action="store_true",
                    help="accepted for runbook clarity; matrix is always emitted")
    ap.add_argument("--emit-per-alpha-turnover", action="store_true",
                    help="accepted for runbook clarity; turnover is always emitted")
    ap.add_argument("--out", default="output_audit/arsenal")
    args = ap.parse_args()

    model_dir = _resolve_model_dir(args.bundle, args.model_dir)
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

    n_alphas = max(1, len(alphas))
    selection_trades, holdout_trades, method_meta = _split_holdout(
        trades, float(args.holdout_frac)
    )
    p_threshold = 0.05 if method_meta["used_holdout"] else 0.05 / n_alphas
    method_meta.update({
        "n_alphas": n_alphas,
        "p_threshold": p_threshold,
        "bonferroni_applied": not bool(method_meta["used_holdout"]),
    })

    trades_path = out_dir / "trades.parquet"
    if not trades.empty:
        # Convert tz-aware/timestamp columns to str for parquet stability.
        ts_cols = [c for c in trades.columns
                    if trades[c].dtype.kind == "M" or "_at" in c]
        for c in ts_cols:
            trades[c] = trades[c].astype(str)
        trades.to_parquet(trades_path, index=False)

    summary = _summary_with_pvalues(trades, p_threshold)
    selection_summary = _summary_with_pvalues(selection_trades, p_threshold)
    holdout_summary = _summary_with_pvalues(holdout_trades, p_threshold)
    selected_names: List[str] = []
    if not selection_summary.empty:
        selected_names = list(selection_summary[
            (selection_summary["mean_R"] > 0)
            & (selection_summary["p_mean_R_gt_0"] < p_threshold)
            & (selection_summary["trades"] >= 200)
        ]["alpha_name"].astype(str))
    holdout_selected = (
        holdout_summary[holdout_summary["alpha_name"].isin(selected_names)].copy()
        if not holdout_summary.empty else pd.DataFrame()
    )
    per_session = evaluator.per_regime_summary(trades, "regime_session")
    per_side = evaluator.per_regime_summary(trades, "regime_side")
    pw = evaluator.pairwise_combinations(trades)
    daily_returns = evaluator.daily_alpha_returns(trades)
    alpha_corr_full = evaluator.alpha_correlation(daily_returns)
    selection_daily_returns = evaluator.daily_alpha_returns(selection_trades)
    holdout_daily_returns = evaluator.daily_alpha_returns(holdout_trades)
    alpha_corr_selection = evaluator.alpha_correlation(selection_daily_returns)
    alpha_corr_holdout = evaluator.alpha_correlation(holdout_daily_returns)
    duplicate_corr_base = (
        alpha_corr_selection if method_meta["used_holdout"] else alpha_corr_full
    )
    high_corr = _high_correlation_pairs(duplicate_corr_base)
    high_corr_holdout = _high_correlation_pairs(alpha_corr_holdout)
    turnover = _per_alpha_turnover(trades)
    multipliers = [float(x) for x in str(args.cost_multipliers).split(",") if x.strip()]
    cost_summary_all = _cost_multiplier_summary(trades, multipliers, p_threshold)
    cost_summary_holdout = _cost_multiplier_summary(
        holdout_trades if method_meta["used_holdout"] else trades,
        multipliers,
        p_threshold,
    )
    baseline_delta_all = _baseline_delta(trades)
    baseline_delta_holdout = _baseline_delta(
        holdout_trades if method_meta["used_holdout"] else trades
    )
    pool_volume_effect_all = _pool_volume_effect(trades, p_threshold)
    pool_volume_effect_primary = _pool_volume_effect(
        holdout_trades if method_meta["used_holdout"] else trades,
        p_threshold,
    )

    summary.to_csv(out_dir / "per_alpha_summary.csv", index=False)
    selection_summary.to_csv(out_dir / "selection_per_alpha_summary.csv", index=False)
    holdout_summary.to_csv(out_dir / "holdout_per_alpha_summary.csv", index=False)
    holdout_selected.to_csv(out_dir / "holdout_selected_alpha_summary.csv", index=False)
    per_session.to_csv(out_dir / "per_regime_session.csv", index=False)
    per_side.to_csv(out_dir / "per_regime_side.csv", index=False)
    pw.to_csv(out_dir / "pairwise_combinations.csv", index=False)
    daily_returns.to_csv(out_dir / "daily_alpha_returns.csv", index=False)
    selection_daily_returns.to_csv(out_dir / "selection_daily_alpha_returns.csv", index=False)
    holdout_daily_returns.to_csv(out_dir / "holdout_daily_alpha_returns.csv", index=False)
    duplicate_corr_base.to_csv(out_dir / "alpha_correlation.csv", index=False)
    alpha_corr_full.to_csv(out_dir / "alpha_correlation_full_oos.csv", index=False)
    alpha_corr_selection.to_csv(out_dir / "alpha_correlation_selection.csv", index=False)
    alpha_corr_holdout.to_csv(out_dir / "alpha_correlation_holdout.csv", index=False)
    high_corr.to_csv(out_dir / "high_alpha_correlations.csv", index=False)
    high_corr_holdout.to_csv(out_dir / "high_alpha_correlations_holdout.csv", index=False)
    turnover.to_csv(out_dir / "per_alpha_turnover.csv", index=False)
    cost_summary_all.to_csv(out_dir / "per_alpha_cost_multipliers.csv", index=False)
    cost_summary_holdout.to_csv(
        out_dir / "holdout_or_bonferroni_cost_multipliers.csv", index=False)
    baseline_delta_all.to_csv(out_dir / "proximity_journey_delta.csv", index=False)
    baseline_delta_holdout.to_csv(
        out_dir / "holdout_or_bonferroni_proximity_journey_delta.csv", index=False)
    pool_volume_effect_all.to_csv(out_dir / "pool_volume_confirmed_effect_all.csv", index=False)
    pool_volume_effect_primary.to_csv(
        out_dir / "pool_volume_confirmed_effect.csv", index=False)
    (out_dir / "methodology_summary.json").write_text(
        json.dumps(method_meta, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "requested_dimension_status.json").write_text(
        json.dumps({
            "pool_volume_confirmed": (
                "computed for pool-based signals using pool_mid_at_touch and "
                "the nearest of POC/VAH/VAL from the signal decision/touch bar; "
                "post-touch pool_reach decision_idx is the touch bar, while "
                "pre-touch journey alphas use decision-time volume context."
            )
        }, indent=2, sort_keys=True), encoding="utf-8")

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
    print("\n[arsenal] alpha_correlation.csv uses "
          f"{'selection slice' if method_meta['used_holdout'] else 'full OOS with Bonferroni'} "
          "for duplicate detection.")
    _print_correlations(duplicate_corr_base)
    if not high_corr.empty:
        print("\n================ HIGH ALPHA CORRELATIONS (|rho| >= 0.80) ================")
        print(high_corr.to_string(index=False))
    if method_meta["used_holdout"] and not high_corr_holdout.empty:
        print("\n================ HOLDOUT HIGH ALPHA CORRELATIONS (sanity) ================")
        print(high_corr_holdout.to_string(index=False))
    if not pool_volume_effect_primary.empty:
        print("\n================ POOL VOLUME CONFIRMED EFFECT ================")
        print(pool_volume_effect_primary.to_string(index=False))
    if not turnover.empty:
        print("\n================ PER-ALPHA TURNOVER ================")
        print(turnover.to_string(index=False))
    if not baseline_delta_holdout.empty:
        print("\n================ PROXIMITY JOURNEY ATTRIBUTION ================")
        print(baseline_delta_holdout.to_string(index=False))
    if not cost_summary_holdout.empty:
        print("\n================ COST WALL SUMMARY ================")
        view = cost_summary_holdout[
            cost_summary_holdout["cost_multiplier"].isin([1.0, 1.5])
        ].copy()
        print(view.to_string(index=False))
    if method_meta["used_holdout"]:
        print("\n================ HOLDOUT ALPHA SELECTION ================")
        print(f"  cutoff: {method_meta['cutoff']}")
        print(f"  selected on first {(1.0 - args.holdout_frac):.0%}: {selected_names or '(none)'}")
        if holdout_selected.empty:
            print("  no selected alpha cleared the selection gate for holdout reporting")
        else:
            print(holdout_selected.to_string(index=False))
    else:
        print("\n================ MULTIPLE-TESTING CONTROL ================")
        print(f"  using Bonferroni p < {p_threshold:.5f} across {n_alphas} alphas")
        print(f"  reason: {method_meta.get('reason')}")

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
    deliverable_summary = holdout_selected if method_meta["used_holdout"] else summary
    for line in _final_verdict(deliverable_summary, null_results, p_threshold, method_meta):
        print(line)
    print(f"\n[arsenal] artifacts written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
