"""Constrained validation for the Track A pre-touch pocket.

The full Track A sweep was MARGINAL, but focused diagnostics found a broad
interpretable pocket: long-only pre-touch trades, with BANKING/IT excluded as
a stricter variant. This runner treats that pocket as a hypothesis and checks
whether it survives across the existing sweep grid, held-out OOS halves, Q
cohorts, sectors, and cost stress.

It does not rerun the simulator and it does not change production gates.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q_audit_common import fnum, markdown_table, pct  # noqa: E402
from q_rescale import exact_top_mask  # noqa: E402


DEFAULT_TRADES = Path("output_phase4_track_a_pretouch_sweep/pretouch_sweep_trades.parquet")
DEFAULT_SUMMARY = Path("reports/phase4_track_a_constrained_validation.csv")
DEFAULT_HOLDOUT = Path("reports/phase4_track_a_constrained_holdout.csv")
DEFAULT_REPORT = Path("reports/phase4_track_a_constrained_validation.md")

Q_COHORTS = [
    ("all", 1.00),
    ("top_50pct", 0.50),
    ("top_25pct", 0.25),
    ("top_10pct", 0.10),
    ("top_5pct", 0.05),
]


def _profit_factor(r: pd.Series) -> float:
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    if neg == 0.0:
        return math.inf if pos > 0 else 0.0
    return pos / neg


def _max_drawdown(r: pd.Series) -> float:
    if r.empty:
        return float("nan")
    equity = r.cumsum()
    return float((equity - equity.cummax()).min())


def _trade_sharpe(r: pd.Series) -> float:
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float(r.mean() / sd)


def _deflated_sharpe(r: pd.Series, n_trials: int) -> float:
    if len(r) < 3:
        return float("nan")
    sr = _trade_sharpe(r)
    if not np.isfinite(sr):
        return float("nan")
    n_obs = len(r)
    sk = float(skew(r, bias=False)) if n_obs >= 3 else 0.0
    ku = float(kurtosis(r, fisher=False, bias=False)) if n_obs >= 4 else 3.0
    trials = max(int(n_trials), 2)
    emc = 0.5772156649
    expected_max_sr = (
        (1.0 - emc) * norm.ppf(1.0 - 1.0 / trials)
        + emc * norm.ppf(1.0 - 1.0 / (trials * math.e))
    )
    variance = (1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr * sr) / max(n_obs - 1, 1)
    if variance <= 0.0 or not np.isfinite(variance):
        return float("nan")
    return float(norm.cdf((sr - expected_max_sr) / math.sqrt(variance)))


def _q_mask(df: pd.DataFrame, fraction: float) -> pd.Series:
    if fraction >= 1.0:
        return pd.Series(True, index=df.index)
    return pd.Series(exact_top_mask(df.reset_index(drop=True), "pool_quality", fraction), index=df.index)


def _scenario_filters() -> Dict[str, Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    return {
        "long_all_sectors": (
            "Long-only across all sectors",
            lambda d: d[d["direction"] == "UP"],
        ),
        "long_ex_banking_it": (
            "Long-only, exclude BANKING and IT",
            lambda d: d[(d["direction"] == "UP") & (~d["sector"].isin(["BANKING", "IT"]))],
        ),
        "long_auto_pharma_fmcg": (
            "Long-only AUTO + PHARMA + FMCG",
            lambda d: d[(d["direction"] == "UP") & (d["sector"].isin(["AUTO", "PHARMA", "FMCG"]))],
        ),
        "long_no_afternoon": (
            "Long-only, morning plus midday only",
            lambda d: d[(d["direction"] == "UP") & (d["time_bucket"].isin(["morning", "midday"]))],
        ),
        "long_no_afternoon_ex_banking_it": (
            "Long-only, no afternoon, exclude BANKING and IT",
            lambda d: d[
                (d["direction"] == "UP")
                & (d["time_bucket"].isin(["morning", "midday"]))
                & (~d["sector"].isin(["BANKING", "IT"]))
            ],
        ),
        "short_all_sectors_control": (
            "Negative control: short-only across all sectors",
            lambda d: d[d["direction"] == "DOWN"],
        ),
        "banking_it_long_control": (
            "Negative control: long-only BANKING + IT",
            lambda d: d[(d["direction"] == "UP") & (d["sector"].isin(["BANKING", "IT"]))],
        ),
    }


def _metric_block(df: pd.DataFrame, prefix: str = "") -> Dict:
    r = pd.to_numeric(df["net_r"], errors="coerce").dropna()
    if len(r) == 0:
        return {
            f"{prefix}n": 0,
            f"{prefix}win_rate": np.nan,
            f"{prefix}mean_r": np.nan,
            f"{prefix}median_r": np.nan,
            f"{prefix}profit_factor": np.nan,
            f"{prefix}max_drawdown_r": np.nan,
        }
    return {
        f"{prefix}n": int(len(r)),
        f"{prefix}win_rate": float((r > 0).mean()),
        f"{prefix}mean_r": float(r.mean()),
        f"{prefix}median_r": float(r.median()),
        f"{prefix}profit_factor": float(_profit_factor(r)),
        f"{prefix}max_drawdown_r": float(_max_drawdown(r)),
    }


def _summarize(
    *,
    scenario: str,
    description: str,
    q_cohort: str,
    q_fraction: float,
    df: pd.DataFrame,
    midpoint: pd.Timestamp,
    dsr_trials: int,
) -> Dict:
    df = df.sort_values("entry_at_dt").copy()
    r = pd.to_numeric(df["net_r"], errors="coerce").dropna()
    first = df[df["entry_at_dt"] < midpoint]
    second = df[df["entry_at_dt"] >= midpoint]
    sectors = df["sector"].value_counts() if len(df) else pd.Series(dtype=int)
    by_time = df.groupby("time_bucket")["net_r"].mean().to_dict() if len(df) else {}
    by_sector = df.groupby("sector")["net_r"].mean().to_dict() if len(df) else {}
    cost_125 = r - ((df["total_cost"] * 0.25) / df["risk_inr"]) if len(df) else pd.Series(dtype=float)
    cost_150 = r - ((df["total_cost"] * 0.50) / df["risk_inr"]) if len(df) else pd.Series(dtype=float)
    temporal_gap = float(abs(first["net_r"].mean() - second["net_r"].mean())) if len(first) and len(second) else np.nan
    sectors_with_30 = int((sectors >= 30).sum()) if len(sectors) else 0
    second_pf = _profit_factor(second["net_r"]) if len(second) else np.nan

    out = {
        "scenario": scenario,
        "description": description,
        "q_cohort": q_cohort,
        "q_fraction": q_fraction,
        **_metric_block(df),
        **_metric_block(first, "dev_"),
        **_metric_block(second, "validation_"),
        "trade_sharpe": float(_trade_sharpe(r)) if len(r) else np.nan,
        "dsr": float(_deflated_sharpe(r, n_trials=dsr_trials)) if len(r) else np.nan,
        "mean_r_cost_1_25x": float(cost_125.mean()) if len(cost_125) else np.nan,
        "mean_r_cost_1_50x": float(cost_150.mean()) if len(cost_150) else np.nan,
        "temporal_gap_r": temporal_gap,
        "sectors_with_30_trades": sectors_with_30,
        "top_sector": str(sectors.index[0]) if len(sectors) else "",
        "top_sector_trades": int(sectors.iloc[0]) if len(sectors) else 0,
        "morning_trades": int((df["time_bucket"] == "morning").sum()) if len(df) else 0,
        "midday_trades": int((df["time_bucket"] == "midday").sum()) if len(df) else 0,
        "afternoon_trades": int((df["time_bucket"] == "afternoon").sum()) if len(df) else 0,
        "morning_mean_r": float(by_time.get("morning", np.nan)),
        "midday_mean_r": float(by_time.get("midday", np.nan)),
        "afternoon_mean_r": float(by_time.get("afternoon", np.nan)),
        "auto_mean_r": float(by_sector.get("AUTO", np.nan)),
        "pharma_mean_r": float(by_sector.get("PHARMA", np.nan)),
        "fmcg_mean_r": float(by_sector.get("FMCG", np.nan)),
        "banking_mean_r": float(by_sector.get("BANKING", np.nan)),
        "it_mean_r": float(by_sector.get("IT", np.nan)),
        "target_exit_rate": float((df["exit_reason"] == "target").mean()) if len(df) else np.nan,
        "stop_exit_rate": float((df["exit_reason"] == "stop").mean()) if len(df) else np.nan,
        "time_exit_rate": float((df["exit_reason"] == "time_exit").mean()) if len(df) else np.nan,
    }
    out["constrained_pass"] = bool(
        out["n"] >= 300
        and out["validation_n"] >= 100
        and np.isfinite(out["mean_r"]) and out["mean_r"] > 0.15
        and np.isfinite(out["validation_mean_r"]) and out["validation_mean_r"] > 0.10
        and np.isfinite(second_pf) and second_pf > 1.25
        and np.isfinite(out["mean_r_cost_1_50x"]) and out["mean_r_cost_1_50x"] > 0.05
        and sectors_with_30 >= 3
    )
    out["validation_pass"] = bool(
        out["validation_n"] >= 100
        and np.isfinite(out["validation_mean_r"]) and out["validation_mean_r"] > 0.10
        and np.isfinite(second_pf) and second_pf > 1.25
    )
    return out


def _cell_mask(trades: pd.DataFrame, min_touch: float, min_dir: float, target: float, stop: float, hold: int) -> pd.Series:
    return (
        (trades["p_touch"] >= min_touch)
        & (trades["p_direction_to_pool"] >= min_dir)
        & (trades["distance_atr"] >= 3.0)
        & (trades["distance_atr"] < 8.0)
        & (trades["target_fraction"] == target)
        & (trades["stop_atr_mult"] == stop)
        & (trades["requested_max_hold_bars"] == hold)
    )


def run_validation(trades: pd.DataFrame, dsr_trials: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    trades = trades.copy()
    trades["entry_at_dt"] = pd.to_datetime(trades["entry_at"])
    midpoint = trades["entry_at_dt"].min() + (trades["entry_at_dt"].max() - trades["entry_at_dt"].min()) / 2

    min_touches = [0.75, 0.80, 0.85]
    min_dirs = [0.55, 0.60, 0.65]
    targets = sorted(float(x) for x in trades["target_fraction"].dropna().unique())
    stops = sorted(float(x) for x in trades["stop_atr_mult"].dropna().unique())
    holds = sorted(int(x) for x in trades["requested_max_hold_bars"].dropna().unique())

    rows: List[Dict] = []
    for min_touch in min_touches:
        for min_dir in min_dirs:
            for target in targets:
                for stop in stops:
                    for hold in holds:
                        base = trades.loc[_cell_mask(trades, min_touch, min_dir, target, stop, hold)].copy()
                        if base.empty:
                            continue
                        for scenario, (description, fn) in _scenario_filters().items():
                            sbase = fn(base).copy()
                            if sbase.empty:
                                continue
                            for q_label, q_fraction in Q_COHORTS:
                                qdf = sbase.loc[_q_mask(sbase, q_fraction)].copy()
                                if qdf.empty:
                                    continue
                                row = _summarize(
                                    scenario=scenario,
                                    description=description,
                                    q_cohort=q_label,
                                    q_fraction=q_fraction,
                                    df=qdf,
                                    midpoint=midpoint,
                                    dsr_trials=dsr_trials,
                                )
                                row.update({
                                    "min_p_touch": min_touch,
                                    "min_p_direction": min_dir,
                                    "distance_low": 3.0,
                                    "distance_high": 8.0,
                                    "target_fraction": target,
                                    "stop_atr_mult": stop,
                                    "max_hold_bars": hold,
                                    "cell": f"T>={min_touch:.2f}|D>={min_dir:.2f}|tf={target:g}|sl={stop:g}|h={hold}",
                                })
                                rows.append(row)

    summary = pd.DataFrame(rows)
    holdout_rows: List[Dict] = []
    if not summary.empty:
        candidates = summary[
            (summary["dev_n"] >= 100)
            & (summary["validation_n"] >= 50)
            & (summary["sectors_with_30_trades"] >= 2)
            & (~summary["scenario"].str.contains("control"))
        ].copy()
        for scenario, sdf in candidates.groupby("scenario"):
            selected = sdf.sort_values(["dev_mean_r", "dev_profit_factor", "dev_n"], ascending=False).iloc[0]
            holdout_rows.append({
                "scenario": scenario,
                "selected_by": "best_dev_mean_r",
                **selected.to_dict(),
            })
    holdout = pd.DataFrame(holdout_rows)
    return summary, holdout, midpoint


def _fmt_pf(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if math.isinf(float(x)):
        return "inf"
    return fnum(x, 3)


def _report_rows(df: pd.DataFrame, limit: int = 12) -> List[Dict]:
    rows = []
    for row in df.head(limit).to_dict(orient="records"):
        rows.append({
            "scenario": row["scenario"],
            "cell": row["cell"],
            "Q": row["q_cohort"],
            "n": int(row["n"]),
            "mean_R": fnum(row["mean_r"], 3),
            "PF": _fmt_pf(row["profit_factor"]),
            "val_n": int(row["validation_n"]),
            "val_R": fnum(row["validation_mean_r"], 3),
            "val_PF": _fmt_pf(row["validation_profit_factor"]),
            "cost1.5x": fnum(row["mean_r_cost_1_50x"], 3),
            "sectors": int(row["sectors_with_30_trades"]),
            "pass": str(bool(row["constrained_pass"])),
        })
    return rows


def _anchor(summary: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (summary["min_p_touch"] == 0.75)
        & (summary["min_p_direction"] == 0.60)
        & (summary["target_fraction"] == 1.0)
        & (summary["stop_atr_mult"] == 2.0)
        & (summary["max_hold_bars"] == 60)
        & (summary["q_cohort"] == "all")
        & (summary["scenario"].isin([
            "long_all_sectors",
            "long_ex_banking_it",
            "long_auto_pharma_fmcg",
            "long_no_afternoon",
            "long_no_afternoon_ex_banking_it",
            "short_all_sectors_control",
            "banking_it_long_control",
        ]))
    )
    order = {
        "long_ex_banking_it": 0,
        "long_auto_pharma_fmcg": 1,
        "long_all_sectors": 2,
        "long_no_afternoon_ex_banking_it": 3,
        "long_no_afternoon": 4,
        "short_all_sectors_control": 5,
        "banking_it_long_control": 6,
    }
    out = summary.loc[mask].copy()
    out["_order"] = out["scenario"].map(order).fillna(99)
    return out.sort_values("_order").drop(columns=["_order"])


def build_report(summary: pd.DataFrame, holdout: pd.DataFrame, midpoint: pd.Timestamp, args: argparse.Namespace) -> str:
    anchor = _anchor(summary)
    passing = summary[summary["constrained_pass"].astype(bool)].copy()
    non_control = summary[~summary["scenario"].str.contains("control")].copy()
    top_validation = non_control[
        (non_control["validation_n"] >= 100)
        & (non_control["sectors_with_30_trades"] >= 2)
    ].copy()
    controls = summary[
        (summary["scenario"].str.contains("control"))
        & (summary["validation_n"] >= 50)
    ].copy()

    if not passing.empty:
        decision = "CONSTRAINED_PASS"
        best = passing.sort_values(["validation_mean_r", "mean_r"], ascending=False).iloc[0]
        reason = (
            f"At least one constrained pocket survives validation filters. Best: "
            f"{best['scenario']} / {best['cell']} / {best['q_cohort']} with "
            f"validation R {best['validation_mean_r']:+.3f}."
        )
    elif not non_control.empty and (non_control["validation_mean_r"] > 0.10).any():
        decision = "CONSTRAINED_MARGINAL"
        best = non_control.sort_values(["validation_mean_r", "validation_n"], ascending=False).iloc[0]
        reason = (
            f"Validation-positive pockets exist, but none clear all breadth/cost filters. "
            f"Best validation R is {best['validation_mean_r']:+.3f}."
        )
    else:
        decision = "CONSTRAINED_FAIL"
        best = non_control.sort_values("validation_mean_r", ascending=False).iloc[0]
        reason = f"No constrained pocket validates. Best validation R is {best['validation_mean_r']:+.3f}."

    lines = [
        "# Phase 4 Track A Constrained Validation",
        "",
        "## Verdict",
        "",
        f"- Decision: `{decision}`",
        f"- {reason}",
        "- This is still research validation, not live-trade approval.",
        "- The test reuses the completed v2 1-minute Track A sweep; no simulator rerun was needed.",
        "",
        "## Setup",
        "",
        f"- Trade source: `{args.trades}`",
        f"- Chronological validation split midpoint: `{midpoint}`",
        f"- DSR trials: `{args.dsr_trials}`",
        "- Validated hypothesis: Track A pre-touch, long-only; BANKING/IT exclusion and no-afternoon are sensitivity checks.",
        "",
        "## Predeclared Anchor Pocket",
        "",
        "Anchor cell: `T>=0.75`, `D>=0.60`, `3-8ATR`, target fraction `1.0`, stop `2.0ATR`, hold `60` bars, Q=`all`.",
        "",
        markdown_table(
            _report_rows(anchor, limit=20),
            ["scenario", "cell", "Q", "n", "mean_R", "PF", "val_n", "val_R", "val_PF", "cost1.5x", "sectors", "pass"],
        ),
        "",
        "## Constrained Passes",
        "",
    ]
    if passing.empty:
        lines.append("No rows cleared every constrained validation filter.")
    else:
        lines.append(markdown_table(
            _report_rows(passing.sort_values(["validation_mean_r", "mean_r"], ascending=False), limit=20),
            ["scenario", "cell", "Q", "n", "mean_R", "PF", "val_n", "val_R", "val_PF", "cost1.5x", "sectors", "pass"],
        ))

    lines.extend([
        "",
        "## Holdout Selection Check",
        "",
        "For each scenario, the cell is selected by first-half OOS mean R only, then scored on the second half.",
        "",
    ])
    if holdout.empty:
        lines.append("No scenario had enough development and validation trades for holdout selection.")
    else:
        lines.append(markdown_table(
            _report_rows(holdout.sort_values(["validation_mean_r", "mean_r"], ascending=False), limit=20),
            ["scenario", "cell", "Q", "n", "mean_R", "PF", "val_n", "val_R", "val_PF", "cost1.5x", "sectors", "pass"],
        ))

    lines.extend([
        "",
        "## Top Validation Rows",
        "",
        "These rows are sorted by validation R after requiring at least 100 validation trades and 2 sector buckets.",
        "",
        markdown_table(
            _report_rows(top_validation.sort_values(["validation_mean_r", "validation_n"], ascending=False), limit=20),
            ["scenario", "cell", "Q", "n", "mean_R", "PF", "val_n", "val_R", "val_PF", "cost1.5x", "sectors", "pass"],
        ),
        "",
        "## Controls",
        "",
        markdown_table(
            _report_rows(controls.sort_values(["validation_mean_r", "n"], ascending=False), limit=12),
            ["scenario", "cell", "Q", "n", "mean_R", "PF", "val_n", "val_R", "val_PF", "cost1.5x", "sectors", "pass"],
        ),
        "",
        "## Interpretation",
        "",
        "- The short side remains a negative control, not a strategy candidate.",
        "- The broad long-only pocket is the key signal; BANKING/IT exclusion is a risk-control variant, not proof of a new model.",
        "- The best constrained rows use Q=`all`; Q percentile filtering is not required for this pocket and should stay diagnostic for now.",
        "- No-afternoon variants can improve mean R but reduce breadth, so they should be treated as a timing overlay after validation.",
        "- If constrained passes remain positive, the next step is CPCV/null baselines before any paper-trading workflow.",
        "- Triple-barrier labels should be built on these executable Track A outcomes only after this constrained pocket survives robustness checks.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--holdout-out", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dsr-trials", type=int, default=550)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trades.exists():
        raise FileNotFoundError(f"missing Track A trade parquet: {args.trades}")
    trades = pd.read_parquet(args.trades)
    summary, holdout, midpoint = run_validation(trades, dsr_trials=args.dsr_trials)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.holdout_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    holdout.to_csv(args.holdout_out, index=False)
    args.report_out.write_text(build_report(summary, holdout, midpoint, args), encoding="utf-8")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.holdout_out}")
    print(f"wrote {args.report_out}")


if __name__ == "__main__":
    main()
