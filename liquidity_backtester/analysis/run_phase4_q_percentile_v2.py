"""Task 1: Q percentile filtering under the v2 execution simulator.

This is analysis-only. It reads completed Phase 4 v2 artifacts and asks whether
compressed-Q ranking rescues any existing post-touch execution policy once the
honest 1-minute v2 simulator and costs are applied.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q_audit_common import fnum, markdown_table, pct
from q_rescale import exact_top_mask


DEFAULT_RUN_DIRS = [
    Path("output_core25_phase4_v2_neutral"),
    Path("output_core25_phase4_v2_generous"),
    Path("output_core25_phase4_v2_conservative"),
]

COHORTS = [
    ("all", 1.00),
    ("top_25pct", 0.25),
    ("top_10pct", 0.10),
    ("top_5pct", 0.05),
    ("top_1pct", 0.01),
]

MODES = [
    "blind_limit",
    "displacement_confirmed",
    "reclaim_confirmed",
    "touch_confirmed",
]

POOL_MATCH_KEYS = [
    "symbol",
    "pool_idx",
    "direction",
    "outcome",
    "score_key",
    "tf_count",
    "factor",
    "mae_key",
    "mfe_key",
]


def _profit_factor(r: pd.Series) -> float:
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    if neg == 0.0:
        return math.inf if pos > 0 else 0.0
    return pos / neg


def _max_drawdown(r: pd.Series) -> float:
    if r.empty:
        return np.nan
    equity = r.cumsum()
    dd = equity - equity.cummax()
    return float(dd.min())


def _load_run(run_dir: Path, q_col: str) -> pd.DataFrame:
    trades_path = run_dir / "execution_backtest_v2_trades.csv"
    audit_path = run_dir / "phase3_oos_prediction_audit.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"missing v2 trades file: {trades_path}")
    if not audit_path.exists():
        raise FileNotFoundError(f"missing Q audit file: {audit_path}")

    trades = pd.read_csv(trades_path)
    audit = pd.read_csv(audit_path)
    if q_col not in audit.columns:
        raise ValueError(f"{audit_path} is missing Q column {q_col!r}")
    required = {"asset", "pool_index", q_col, "actual", "outcome", "distance_atr", "distance_bucket"}
    missing = sorted(required - set(audit.columns))
    if missing:
        raise ValueError(f"{audit_path} missing columns: {missing}")

    q = audit.copy()
    q["symbol_pool_pos"] = q.groupby("asset").cumcount()
    q = q[[
        "asset",
        "pool_index",
        "symbol_pool_pos",
        q_col,
        "actual",
        "outcome",
        "distance_atr",
        "distance_bucket",
    ]].rename(columns={
        "asset": "q_asset",
        "pool_index": "q_pool_index",
        "outcome": "quality_outcome",
    })
    pool_map = _load_pool_idx_map(run_dir)
    if pool_map is not None:
        trades = _add_pool_match_keys(trades)
        trades = trades.merge(
            pool_map,
            on=POOL_MATCH_KEYS,
            how="left",
            validate="many_to_one",
        )
        missing_pool_pos = int(trades["symbol_pool_pos"].isna().sum())
        if missing_pool_pos:
            raise ValueError(f"{run_dir}: pool index map failed for {missing_pool_pos:,} v2 trade rows")
        left_keys = ["symbol", "symbol_pool_pos"]
        right_keys = ["q_asset", "symbol_pool_pos"]
    else:
        # Legacy fallback for artifacts that wrote local OOS positions directly.
        trades["symbol_pool_pos"] = pd.to_numeric(trades["pool_idx"], errors="coerce")
        left_keys = ["symbol", "symbol_pool_pos"]
        right_keys = ["q_asset", "symbol_pool_pos"]

    merged = trades.merge(
        q,
        left_on=left_keys,
        right_on=right_keys,
        how="left",
        validate="many_to_one",
    )
    missing_q = int(merged[q_col].isna().sum())
    if missing_q:
        raise ValueError(f"{run_dir}: Q join failed for {missing_q:,} v2 trade rows")
    merged["run_dir"] = str(run_dir)
    if "fill_policy" not in merged.columns:
        merged["fill_policy"] = run_dir.name.rsplit("_", 1)[-1]
    return merged


def _add_pool_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score_key"] = pd.to_numeric(out["score"], errors="coerce").round(6)
    out["mae_key"] = pd.to_numeric(out["mae_atr"], errors="coerce").round(6)
    out["mfe_key"] = pd.to_numeric(out["mfe_atr"], errors="coerce").round(6)
    out["pool_idx"] = pd.to_numeric(out["pool_idx"], errors="coerce").astype("int64")
    out["tf_count"] = pd.to_numeric(out["tf_count"], errors="coerce").astype("int64")
    return out


def _infer_model_bundle(run_dir: Path) -> Path:
    name = run_dir.name
    if name.startswith("output_"):
        return Path("output_models") / name.removeprefix("output_") / "multi_asset_report.pkl"
    return Path("output_models") / name / "multi_asset_report.pkl"


def _load_pool_idx_map(run_dir: Path) -> Optional[pd.DataFrame]:
    """Map saved v2 trade pool indexes back to OOS audit row positions.

    V2 trade rows intentionally preserve ``PoolResult.pool_idx`` from the detector's
    per-symbol pool list. The Phase 3 OOS audit CSV stores a global concatenated
    ``pool_index`` instead, so joining directly on ``pool_idx`` silently misaligns
    every symbol after the first. The persisted report bundle keeps the original
    OOS result ordering, which gives us a deterministic map:
    ``(symbol, result.pool_idx) -> symbol-local OOS position``.
    """
    bundle = _infer_model_bundle(run_dir)
    if not bundle.exists():
        return None
    from liqpool.execution_backtest import _headline_factor

    with bundle.open("rb") as fh:
        report = pickle.load(fh)
    rows = []
    for symbol, asset_data in getattr(report, "assets", {}).items():
        results = getattr(getattr(asset_data, "walkforward", None), "oos_results", [])
        pools = getattr(getattr(asset_data, "walkforward", None), "oos_pools", [])
        for symbol_pool_pos, (pool, result) in enumerate(zip(pools, results)):
            rows.append({
                "symbol": symbol,
                "pool_idx": int(result.pool_idx),
                "direction": "UP" if pool.side == "low" else "DOWN",
                "outcome": result.outcome,
                "score_key": round(float(pool.score), 6),
                "tf_count": int(len(set(pool.tfs))),
                "factor": _headline_factor(pool),
                "mae_key": round(float(result.max_excursion_through), 6),
                "mfe_key": round(float(result.reaction_atr), 6),
                "symbol_pool_pos": int(symbol_pool_pos),
            })
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        return None
    duplicates = int(mapping.duplicated(POOL_MATCH_KEYS).sum())
    if duplicates:
        raise ValueError(f"{bundle}: duplicate pool match keys: {duplicates}")
    return mapping


def _cohort_keys(unique_pools: pd.DataFrame, q_col: str) -> Dict[str, set]:
    keys: Dict[str, set] = {}
    for label, fraction in COHORTS:
        if fraction >= 1.0:
            mask = pd.Series(True, index=unique_pools.index)
        else:
            mask = pd.Series(exact_top_mask(unique_pools, q_col, fraction), index=unique_pools.index)
        keys[label] = set(zip(
            unique_pools.loc[mask, "symbol"],
            unique_pools.loc[mask, "pool_idx"],
        ))
    return keys


def summarize_by_q(trades: pd.DataFrame, q_col: str = "blended_q") -> pd.DataFrame:
    rows: List[Dict] = []
    for policy, pdf in trades.groupby("fill_policy", sort=True):
        unique_pools = (
            pdf[["symbol", "pool_idx", q_col, "actual"]]
            .drop_duplicates(["symbol", "pool_idx"])
            .reset_index(drop=True)
        )
        lookup = _cohort_keys(unique_pools, q_col)

        for mode, mdf in pdf.groupby("mode", sort=True):
            mode_keys = list(zip(mdf["symbol"], mdf["pool_idx"]))
            for cohort, fraction in COHORTS:
                keys = lookup[cohort]
                cdf = mdf[[key in keys for key in mode_keys]].copy()
                r = pd.to_numeric(cdf["net_r"], errors="coerce").dropna()
                net = pd.to_numeric(cdf["net_pnl"], errors="coerce").dropna()
                rows.append({
                    "fill_policy": policy,
                    "mode": mode,
                    "cohort": cohort,
                    "fraction": fraction,
                    "trade_n": int(len(r)),
                    "win_rate": float((r > 0).mean()) if len(r) else np.nan,
                    "mean_r": float(r.mean()) if len(r) else np.nan,
                    "median_r": float(r.median()) if len(r) else np.nan,
                    "total_r": float(r.sum()) if len(r) else 0.0,
                    "profit_factor": float(_profit_factor(r)) if len(r) else np.nan,
                    "max_drawdown_r": float(_max_drawdown(r)) if len(r) else np.nan,
                    "mean_net_pnl": float(net.mean()) if len(net) else np.nan,
                    "total_net_pnl": float(net.sum()) if len(net) else 0.0,
                    "mean_q": float(cdf[q_col].mean()) if len(cdf) else np.nan,
                    "min_q": float(cdf[q_col].min()) if len(cdf) else np.nan,
                    "actual_strict_rate": float(cdf["actual"].mean()) if cdf["actual"].notna().any() else np.nan,
                })
    return pd.DataFrame(rows).sort_values(["fill_policy", "mode", "fraction"], ascending=[True, True, False])


def _format_pf(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if math.isinf(float(x)):
        return "inf"
    return fnum(x, 3)


def _decision_for_neutral(summary: pd.DataFrame) -> tuple[str, str]:
    neutral = summary[(summary["fill_policy"] == "neutral") & (summary["cohort"] == "top_1pct")]
    if neutral.empty:
        return "UNKNOWN", "Neutral top-1% cohort was not available."
    best = float(neutral["mean_r"].max())
    if best > -0.30:
        return (
            "TRACK_B_RESCUE_POTENTIAL",
            f"Best neutral top-1% Q cohort is {best:+.2f}R, better than -0.30R.",
        )
    if best < -0.50:
        return (
            "TRACK_B_DEAD",
            f"Best neutral top-1% Q cohort is {best:+.2f}R, worse than -0.50R.",
        )
    return (
        "TRACK_B_MARGINAL",
        f"Best neutral top-1% Q cohort is {best:+.2f}R, between -0.50R and -0.30R.",
    )


def _matrix_rows(summary: pd.DataFrame, policy: str) -> List[Dict]:
    rows = []
    sdf = summary[summary["fill_policy"] == policy]
    for mode in MODES:
        mdf = sdf[sdf["mode"] == mode].set_index("cohort")
        if mdf.empty:
            continue
        row = {"mode": mode}
        for cohort in ["all", "top_25pct", "top_10pct", "top_5pct", "top_1pct"]:
            if cohort in mdf.index:
                rr = mdf.loc[cohort, "mean_r"]
                n = int(mdf.loc[cohort, "trade_n"])
                row[cohort] = f"{float(rr):+.3f}R ({n})"
            else:
                row[cohort] = "n/a"
        rows.append(row)
    return rows


def _full_rows(summary: pd.DataFrame) -> List[Dict]:
    rows = []
    for row in summary.to_dict(orient="records"):
        rows.append({
            "fill": row["fill_policy"],
            "mode": row["mode"],
            "cohort": row["cohort"],
            "trades": int(row["trade_n"]),
            "win": pct(row["win_rate"], 1),
            "mean_R": fnum(row["mean_r"], 3),
            "median_R": fnum(row["median_r"], 3),
            "PF": _format_pf(row["profit_factor"]),
            "maxDD_R": fnum(row["max_drawdown_r"], 1),
            "actual": pct(row["actual_strict_rate"], 1),
            "mean_Q": pct(row["mean_q"], 1),
        })
    return rows


def build_report(summary: pd.DataFrame, run_dirs: Iterable[Path]) -> str:
    decision, reason = _decision_for_neutral(summary)
    lines = [
        "# Q Percentile Filtering Does Not Rescue Track B Under V2 Simulator"
        if decision == "TRACK_B_DEAD"
        else "# Phase 4 Task 1: Q Percentile Filtering Under V2 Simulator",
        "",
        "## Verdict",
        "",
        f"- Decision: `{decision}`",
        f"- {reason}",
        "- Official decision uses the neutral fill policy; generous/conservative are sensitivity checks.",
        "- This report does not change live gates and does not prove any tradeable edge.",
        "",
        "## Inputs",
        "",
    ]
    for rd in run_dirs:
        lines.append(f"- `{rd}`")
    lines.extend([
        "",
        "## Required Matrix: Neutral Fill Policy",
        "",
        "Each cell is `mean R (trade count)`.",
        "",
        markdown_table(
            _matrix_rows(summary, "neutral"),
            ["mode", "all", "top_25pct", "top_10pct", "top_5pct", "top_1pct"],
        ),
        "",
        "## Sensitivity: Generous Fill Policy",
        "",
        markdown_table(
            _matrix_rows(summary, "generous"),
            ["mode", "all", "top_25pct", "top_10pct", "top_5pct", "top_1pct"],
        ),
        "",
        "## Sensitivity: Conservative Fill Policy",
        "",
        markdown_table(
            _matrix_rows(summary, "conservative"),
            ["mode", "all", "top_25pct", "top_10pct", "top_5pct", "top_1pct"],
        ),
        "",
        "## Full Cohort Table",
        "",
        markdown_table(
            _full_rows(summary),
            ["fill", "mode", "cohort", "trades", "win", "mean_R", "median_R", "PF", "maxDD_R", "actual", "mean_Q"],
        ),
        "",
        "## Interpretation",
        "",
        "- Q ranking still improves some cohorts, but not enough to overcome v2 execution friction.",
        "- The old post-touch policies remain negative even after filtering to the highest-Q pools.",
        "- Track B should stay deprioritized unless a future execution policy changes the payoff geometry.",
        "- The next primary research path remains Track A pre-touch directional sweep under v2.",
        "",
    ])
    return "\n".join(lines)


def parse_run_dirs(raw: str) -> List[Path]:
    if not raw:
        return [p for p in DEFAULT_RUN_DIRS if p.exists()]
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dirs",
        default=",".join(str(p) for p in DEFAULT_RUN_DIRS),
        help="Comma-separated v2 output directories",
    )
    parser.add_argument("--q-col", default="blended_q")
    parser.add_argument("--summary-out", default="reports/phase4_q_percentile_v2.csv")
    parser.add_argument("--report-out", default="reports/phase4_q_percentile_v2.md")
    args = parser.parse_args()

    run_dirs = parse_run_dirs(args.run_dirs)
    if not run_dirs:
        raise SystemExit("no v2 run directories found")
    frames = [_load_run(rd, args.q_col) for rd in run_dirs]
    trades = pd.concat(frames, ignore_index=True)
    summary = summarize_by_q(trades, q_col=args.q_col)

    summary_out = Path(args.summary_out)
    report_out = Path(args.report_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    report_out.write_text(build_report(summary, run_dirs), encoding="utf-8")
    print(f"wrote {summary_out}")
    print(f"wrote {report_out}")


if __name__ == "__main__":
    main()
