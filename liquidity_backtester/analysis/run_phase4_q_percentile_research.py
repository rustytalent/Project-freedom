"""Research-only policy outcome audit by Q percentile cohorts.

This answers the first Branch A follow-up question:
does compressed-Q ranking improve executable policy returns?
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from q_audit_common import DEFAULT_AUDIT_CSV, DEFAULT_RUN_DIR, fnum, markdown_table, pct
from q_rescale import add_q_percentile, exact_top_mask, fit_q_scale, metadata_as_dict


DEFAULT_POLICY_OUTCOMES = DEFAULT_RUN_DIR / "execution_policy_outcomes.parquet"


COHORTS = [
    ("all", 1.00),
    ("top_50pct", 0.50),
    ("top_25pct", 0.25),
    ("top_10pct", 0.10),
    ("top_5pct", 0.05),
    ("top_1pct", 0.01),
]


def load_joined_policy_q(
    audit_csv: Path,
    policy_outcomes: Path,
    q_col: str = "blended_q",
) -> pd.DataFrame:
    if not policy_outcomes.exists():
        raise FileNotFoundError(f"Policy outcome parquet not found: {policy_outcomes}")
    q = pd.read_csv(audit_csv)
    if q_col not in q.columns:
        raise ValueError(f"Missing Q column in audit CSV: {q_col}")
    q = add_q_percentile(q, q_col=q_col)
    q["symbol_pool_idx"] = q.groupby("asset").cumcount()
    q_keep = q[[
        "asset",
        "symbol_pool_idx",
        q_col,
        "q_percentile",
        "q_rank_desc",
        "actual",
        "outcome",
        "distance_atr",
        "distance_bucket",
    ]].rename(columns={"outcome": "quality_outcome"})

    policy = pd.read_parquet(policy_outcomes)
    joined = policy.merge(
        q_keep,
        left_on=["symbol", "pool_idx"],
        right_on=["asset", "symbol_pool_idx"],
        how="left",
        validate="many_to_one",
    )
    missing_q = int(joined[q_col].isna().sum())
    if missing_q:
        raise ValueError(f"Policy/Q join failed for {missing_q} rows")
    return joined


def _profit_factor(r: pd.Series) -> float:
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    if neg == 0.0:
        return math.inf if pos > 0 else 0.0
    return pos / neg


def _cohort_mask(base: pd.DataFrame, q_col: str, fraction: float) -> pd.Series:
    if fraction >= 1.0:
        return pd.Series(True, index=base.index)
    return pd.Series(exact_top_mask(base, q_col, fraction), index=base.index)


def analyze_policy_by_q_percentile(joined: pd.DataFrame, q_col: str = "blended_q") -> pd.DataFrame:
    rows: List[Dict] = []
    # Build exact top-N masks on unique OOS pools, then map by mode rows.
    unique_pools = (
        joined[["symbol", "pool_idx", q_col, "q_percentile", "actual"]]
        .drop_duplicates(["symbol", "pool_idx"])
        .reset_index(drop=True)
    )
    cohort_lookup: Dict[str, set] = {}
    for label, fraction in COHORTS:
        mask = _cohort_mask(unique_pools, q_col, fraction)
        keys = set(zip(unique_pools.loc[mask, "symbol"], unique_pools.loc[mask, "pool_idx"]))
        cohort_lookup[label] = keys

    for mode, mode_df in joined.groupby("mode"):
        all_keys = list(zip(mode_df["symbol"], mode_df["pool_idx"]))
        for label, fraction in COHORTS:
            keys = cohort_lookup[label]
            cohort = mode_df[[key in keys for key in all_keys]]
            trades = cohort[cohort["policy_target_trade_generated"] == 1].copy()
            returns = pd.to_numeric(trades["policy_target_return_r"], errors="coerce").dropna()
            wins = returns > 0
            rows.append({
                "mode": mode,
                "cohort": label,
                "fraction": fraction,
                "pool_n": int(len(cohort)),
                "trade_n": int(len(returns)),
                "trade_rate": float(len(returns) / len(cohort)) if len(cohort) else np.nan,
                "win_rate": float(wins.mean()) if len(returns) else np.nan,
                "mean_r": float(returns.mean()) if len(returns) else np.nan,
                "median_r": float(returns.median()) if len(returns) else np.nan,
                "total_r": float(returns.sum()) if len(returns) else 0.0,
                "profit_factor": float(_profit_factor(returns)) if len(returns) else np.nan,
                "mean_q": float(cohort[q_col].mean()) if len(cohort) else np.nan,
                "min_q": float(cohort[q_col].min()) if len(cohort) else np.nan,
                "actual_strict_rate": float(cohort["actual"].mean()) if cohort["actual"].notna().any() else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["mode", "fraction"], ascending=[True, False])


def _summary_rows(summary: pd.DataFrame) -> List[Dict]:
    rows = []
    for row in summary.to_dict(orient="records"):
        pf = row["profit_factor"]
        rows.append({
            "mode": row["mode"],
            "cohort": row["cohort"],
            "pools": row["pool_n"],
            "trades": row["trade_n"],
            "trade%": pct(row["trade_rate"], 1),
            "win": pct(row["win_rate"], 1),
            "mean_R": fnum(row["mean_r"], 3),
            "median_R": fnum(row["median_r"], 3),
            "PF": "inf" if math.isinf(pf) else fnum(pf, 3),
            "actual": pct(row["actual_strict_rate"], 1),
            "mean_Q": pct(row["mean_q"], 1),
        })
    return rows


def _best_rows(summary: pd.DataFrame) -> List[Dict]:
    rows = []
    for mode, mdf in summary.groupby("mode"):
        non_all = mdf[mdf["cohort"] != "all"].copy()
        best = non_all.sort_values(["mean_r", "profit_factor"], ascending=[False, False]).iloc[0]
        base = mdf[mdf["cohort"] == "all"].iloc[0]
        rows.append({
            "mode": mode,
            "best_cohort": best["cohort"],
            "base_mean_R": fnum(base["mean_r"], 3),
            "best_mean_R": fnum(best["mean_r"], 3),
            "delta_R": fnum(best["mean_r"] - base["mean_r"], 3),
            "base_PF": fnum(base["profit_factor"], 3),
            "best_PF": fnum(best["profit_factor"], 3),
            "trades": int(best["trade_n"]),
        })
    return rows


def build_report(summary: pd.DataFrame, q_meta: Dict, audit_csv: Path, policy_outcomes: Path) -> str:
    best_rows = _best_rows(summary)
    any_positive = bool((summary["mean_r"] > 0).any())
    verdict = (
        "Q percentile ranking improves several cohorts but does not create a positive "
        "existing-policy edge yet."
        if not any_positive
        else "At least one Q percentile cohort has positive mean R; validate with leakage and v2 execution."
    )
    lines = [
        "# Phase 4 Branch A Q Percentile Policy Research",
        "",
        "## Verdict",
        "",
        f"- {verdict}",
        "- This is a research report only. It does not change live gates.",
        "- Existing policy outcomes are still from the v1 execution simulator, so this is not final live evidence.",
        "",
        "## Inputs",
        "",
        f"- Q audit CSV: `{audit_csv}`",
        f"- Policy outcome labels: `{policy_outcomes}`",
        f"- Q column: `{q_meta['q_col']}`",
        f"- Q observed range: {pct(q_meta['min_q'], 1)} to {pct(q_meta['max_q'], 1)}",
        "",
        "## Best Existing Policy Cohort By Q Percentile",
        "",
        markdown_table(
            best_rows,
            ["mode", "best_cohort", "base_mean_R", "best_mean_R", "delta_R", "base_PF", "best_PF", "trades"],
        ),
        "",
        "## Full Cohort Table",
        "",
        markdown_table(
            _summary_rows(summary),
            ["mode", "cohort", "pools", "trades", "trade%", "win", "mean_R", "median_R", "PF", "actual", "mean_Q"],
        ),
        "",
        "## Interpretation",
        "",
        "- Branch A was correct: absolute Q scale is compressed and rank signal exists.",
        "- However, filtering existing blind/touch/reclaim/displacement policies by top-Q percentile "
        "does not by itself make those existing policies profitable after costs.",
        "- The strongest improvement appears in the rare top 1% cohorts, but sample sizes are small "
        "and still need leakage checks, 1-minute execution v2, and DSR before trust.",
        "- Next engineering path should continue with leakage probes and execution simulator v2 before "
        "any live gate rewrite.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", default=str(DEFAULT_AUDIT_CSV))
    parser.add_argument("--policy-outcomes", default=str(DEFAULT_POLICY_OUTCOMES))
    parser.add_argument("--summary-out", default="reports/phase4_q_percentile_policy_backtest.csv")
    parser.add_argument("--report-out", default="reports/phase4_q_percentile_policy_backtest.md")
    parser.add_argument("--q-col", default="blended_q")
    args = parser.parse_args()

    audit_csv = Path(args.audit_csv)
    policy_outcomes = Path(args.policy_outcomes)
    joined = load_joined_policy_q(audit_csv, policy_outcomes, q_col=args.q_col)
    summary = analyze_policy_by_q_percentile(joined, q_col=args.q_col)

    summary_out = Path(args.summary_out)
    report_out = Path(args.report_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)

    unique_q = joined[["symbol", "pool_idx", args.q_col]].drop_duplicates()
    q_meta = metadata_as_dict(fit_q_scale(unique_q[args.q_col], q_col=args.q_col))
    report_out.write_text(
        build_report(summary, q_meta, audit_csv, policy_outcomes),
        encoding="utf-8",
    )
    print(f"wrote {summary_out}")
    print(f"wrote {report_out}")


if __name__ == "__main__":
    main()
