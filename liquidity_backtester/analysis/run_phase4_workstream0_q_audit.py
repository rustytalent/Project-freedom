"""Generate the Phase 4 Workstream 0 Q-scale audit report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from q_audit_common import (
    DEFAULT_AUDIT_CSV,
    DEFAULT_LIVE_PLAN_JSON,
    fnum,
    markdown_table,
    pct,
    pp,
)
from q_decile_actual_rate import analyze_q_decile_performance
from q_distribution_audit import analyze_q_distribution
from q_gate_reachability import analyze_gate_reachability
from q_top_percentile_lift import analyze_top_percentile_lift


def _overall_distribution_rows(distribution: Dict) -> List[Dict]:
    rows = []
    for q_col, stats in distribution["overall"].items():
        rows.append({
            "q_col": q_col,
            "n": stats["n"],
            "min": fnum(stats["min"]),
            "p50": fnum(stats["percentiles"]["50"]),
            "p90": fnum(stats["percentiles"]["90"]),
            "p95": fnum(stats["percentiles"]["95"]),
            "p99": fnum(stats["percentiles"]["99"]),
            "max": fnum(stats["max"]),
            "mean": fnum(stats["mean"]),
            "std": fnum(stats["std"]),
            "range": fnum(stats["range"]),
        })
    return rows


def _asset_rows(distribution: Dict) -> List[Dict]:
    df = distribution["per_asset"].sort_values("std", ascending=False)
    rows = []
    for row in df.to_dict(orient="records"):
        rows.append({
            "asset": row["asset"],
            "min": fnum(row["min"]),
            "max": fnum(row["max"]),
            "mean": fnum(row["mean"]),
            "std": fnum(row["std"]),
            "range": fnum(row["range"]),
        })
    return rows


def _sector_rows(distribution: Dict) -> List[Dict]:
    df = distribution["per_sector"].sort_values("std", ascending=False)
    rows = []
    for row in df.to_dict(orient="records"):
        rows.append({
            "sector": row["sector"],
            "min": fnum(row["min"]),
            "max": fnum(row["max"]),
            "mean": fnum(row["mean"]),
            "std": fnum(row["std"]),
            "range": fnum(row["range"]),
        })
    return rows


def _decile_rows(deciles) -> List[Dict]:
    rows = []
    for row in deciles.to_dict(orient="records"):
        rows.append({
            "decile": int(row["q_decile"]),
            "n": int(row["n"]),
            "mean_q": fnum(row["mean_q_predicted"]),
            "min_q": fnum(row["min_q"]),
            "max_q": fnum(row["max_q"]),
            "actual": pct(row["actual_strict_respect"], 1),
            "lift": pp(row["lift_over_base"], 2),
            "rel_lift": f"{row['lift_pct_relative']:+.1f}%",
        })
    return rows


def _top_lift_rows(top_lift: Dict) -> List[Dict]:
    labels = ["top_50pct", "top_25pct", "top_10pct", "top_5pct", "top_1pct", "top_0_1pct"]
    rows = []
    for label in labels:
        stats = top_lift["slices"][label]
        rows.append({
            "slice": label.replace("_", " "),
            "n": stats["n"],
            "min_q": fnum(stats["min_q"]),
            "mean_q": fnum(stats["mean_q"]),
            "max_q": fnum(stats["max_q"]),
            "actual": pct(stats["actual_respect"], 1),
            "lift": pp(stats["lift_over_base"], 2),
            "rel_lift": f"{stats['lift_pct_relative']:+.1f}%",
        })
    return rows


def _gate_rows(gate: Dict) -> List[Dict]:
    thresholds = gate["suggested_percentile_thresholds"]
    return [
        {
            "check": "current absolute gate",
            "value": pct(gate["current_absolute_gate"], 1),
            "result": gate["gate_status"],
        },
        {
            "check": "max observed blended_q",
            "value": pct(gate["max_q_observed"], 1),
            "result": "below current gate",
        },
        {
            "check": "setups passing current gate",
            "value": f"{gate['pct_setups_passing_gate']:.4f}%",
            "result": "none",
        },
        {
            "check": "top 25% threshold",
            "value": pct(thresholds["top_25pct_threshold"], 1),
            "result": "candidate watch threshold",
        },
        {
            "check": "top 10% threshold",
            "value": pct(thresholds["top_10pct_threshold"], 1),
            "result": "candidate strict threshold",
        },
        {
            "check": "top 5% threshold",
            "value": pct(thresholds["top_5pct_threshold"], 1),
            "result": "candidate high-confidence research threshold",
        },
        {
            "check": "top 1% threshold",
            "value": pct(thresholds["top_1pct_threshold"], 1),
            "result": "rare tail threshold",
        },
    ]


def _classify_branch(distribution: Dict, top_lift: Dict, gate: Dict) -> Dict:
    blended = distribution["overall"]["blended_q"]
    max_q = float(blended["max"])
    std_q = float(blended["std"])
    top5 = float(top_lift["slices"]["top_5pct"]["lift_over_base"])
    top1 = float(top_lift["slices"]["top_1pct"]["lift_over_base"])

    compressed = max_q < 0.60 and std_q < 0.05
    gate_unreachable = gate["gate_status"] == "UNREACHABLE"
    has_strong_tail_lift = top5 >= 0.05 or top1 >= 0.08

    if compressed and has_strong_tail_lift:
        branch = "Branch A - Q is compressed and has edge"
        decision = (
            "The 70% absolute Q gate is broken for this model scale. Replace production "
            "research gates with percentile-based Q gates, then re-run execution backtests "
            "before enabling any live trade confidence."
        )
    elif not compressed and top5 < 0.02:
        branch = "Branch B - Q is genuinely weak, not compressed"
        decision = (
            "The current weak-Q diagnosis stands. Continue with post-touch/execution work."
        )
    elif compressed and top5 < 0.02:
        branch = "Branch C - Q is compressed and underlying lift is weak"
        decision = (
            "The gate scale is broken, but percentile gates are unlikely to reveal strong "
            "pre-touch quality edge. Fix reporting/gates, but keep post-touch/execution focus."
        )
    else:
        branch = "Mixed - compressed with moderate lift"
        decision = (
            "Gate scale is broken, but tail lift does not meet the strongest Branch A rule. "
            "Use percentile-gated research backtests before any live decision changes."
        )

    return {
        "branch": branch,
        "compressed": compressed,
        "gate_unreachable": gate_unreachable,
        "top5_lift": top5,
        "top1_lift": top1,
        "decision": decision,
    }


def build_report(audit_csv: Path, live_plan_json: Path) -> str:
    distribution = analyze_q_distribution(audit_csv)
    deciles = analyze_q_decile_performance(audit_csv)
    top_lift = analyze_top_percentile_lift(audit_csv)
    gate = analyze_gate_reachability(audit_csv, live_plan_json)
    branch = _classify_branch(distribution, top_lift, gate)

    blended = distribution["overall"]["blended_q"]
    base_rate = top_lift["base_rate"]

    lines = [
        "# Phase 4 Workstream 0 Q Scale Audit",
        "",
        "## Verdict",
        "",
        f"- **Decision:** {branch['branch']}.",
        f"- **Current Q gate reachable?** {'No' if branch['gate_unreachable'] else 'Yes'}.",
        f"- **Actual blended Q range:** {pct(blended['min'], 1)} to {pct(blended['max'], 1)}.",
        f"- **Blended Q standard deviation:** {pct(blended['std'], 2)}.",
        f"- **Base strict-respect rate:** {pct(base_rate, 1)}.",
        f"- **Exact top 5% lift:** {pp(branch['top5_lift'], 2)}.",
        f"- **Exact top 1% lift:** {pp(branch['top1_lift'], 2)}.",
        "",
        branch["decision"],
        "",
        "Plain English: the quality model is not expressing confidence on a 0-100% live-gate "
        "scale. Its useful ranking signal lives inside a very narrow band around the base "
        "rate. A `Q >= 70%` live gate is therefore mathematically unreachable for the current "
        "conservative model outputs.",
        "",
        "## Inputs",
        "",
        f"- Audit CSV: `{audit_csv}`",
        f"- Live plan JSON: `{live_plan_json}`",
        "- No retraining was performed.",
        "- Only decisive OOS rows (`actual` not null) are used for realized strict-respect lift.",
        "",
        "## Q Distribution",
        "",
        markdown_table(
            _overall_distribution_rows(distribution),
            ["q_col", "n", "min", "p50", "p90", "p95", "p99", "max", "mean", "std", "range"],
        ),
        "",
        "## Gate Reachability",
        "",
        markdown_table(_gate_rows(gate), ["check", "value", "result"]),
        "",
        "## Q Decile Actual Strict-Respect Rates",
        "",
        markdown_table(
            _decile_rows(deciles),
            ["decile", "n", "mean_q", "min_q", "max_q", "actual", "lift", "rel_lift"],
        ),
        "",
        "## Exact Top-Percentile Lift",
        "",
        markdown_table(
            _top_lift_rows(top_lift),
            ["slice", "n", "min_q", "mean_q", "max_q", "actual", "lift", "rel_lift"],
        ),
        "",
        "## Per-Sector Compression",
        "",
        markdown_table(_sector_rows(distribution), ["sector", "min", "max", "mean", "std", "range"]),
        "",
        "## Per-Asset Compression",
        "",
        markdown_table(_asset_rows(distribution), ["asset", "min", "max", "mean", "std", "range"]),
        "",
        "## Percentile-Gate Replacement Spec",
        "",
        "This report does not implement gate changes. If the user approves Branch A follow-up, "
        "the next code change should replace absolute Q thresholds with percentile-aware "
        "thresholds learned from OOS calibration artifacts.",
        "",
        "- Persist Q distribution metadata with the model bundle: `q_col`, OOS quantiles, base "
        "strict rate, top-slice lifts, and source artifact hash if available.",
        "- Convert each live candidate's `blended_q` into `q_percentile` using the saved OOS "
        "distribution.",
        "- Replace the current hard `Q >= 70%` gate with a research gate such as "
        "`q_percentile >= 0.90` or `q_percentile >= 0.95`, then backtest before promoting it.",
        "- Keep `P_touch`, `P_reaction`, direction alignment, distance, bucket sample size, "
        "sector regime, and net expectancy gates. Percentile Q is not allowed to create a "
        "trade by itself.",
        "- Re-run execution backtests on top-percentile-Q cohorts and report PF, mean R, max "
        "drawdown, and DSR before changing live behavior.",
        "",
        "## Next Step",
        "",
        "Stop here for review. Do not start Track A, Track B, simulator v2, or leakage CI until "
        "this audit is accepted. The immediate likely follow-up is a percentile-gate research "
        "backtest, not live trading.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", default=str(DEFAULT_AUDIT_CSV))
    parser.add_argument("--live-plan-json", default=str(DEFAULT_LIVE_PLAN_JSON))
    parser.add_argument("--out", default="reports/phase4_workstream0_q_audit.md")
    args = parser.parse_args()

    audit_csv = Path(args.audit_csv)
    live_plan_json = Path(args.live_plan_json)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(audit_csv, live_plan_json)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
