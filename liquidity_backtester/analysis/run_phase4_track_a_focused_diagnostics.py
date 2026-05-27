"""Focused diagnostics for the marginal Track A pre-touch sweep.

This reads the completed Track A trade parquet and asks whether the marginal
positive cell contains a durable pocket after simple, interpretable filters:
long-only, time-of-day, sector groups, and combinations. It does not rerun the
simulator and it does not change any production gate.
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
DEFAULT_CELLS = Path("reports/phase4_track_a_sweep_cells.csv")
DEFAULT_SUMMARY = Path("reports/phase4_track_a_focused_diagnostics.csv")
DEFAULT_REPORT = Path("reports/phase4_track_a_focused_diagnostics.md")

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


def _select_base_cell(cells: pd.DataFrame) -> pd.Series:
    passing = cells[cells["passes_min_filters"].astype(bool)].copy()
    if passing.empty:
        raise ValueError("no passing Track A sweep cells found")
    # Start from the best unfiltered cell. Q cohorts are analyzed post-hoc below.
    all_q = passing[passing["q_cohort"] == "all"]
    if all_q.empty:
        all_q = passing
    return all_q.sort_values("mean_r", ascending=False).iloc[0]


def _base_trades(trades: pd.DataFrame, cell: pd.Series) -> pd.DataFrame:
    return trades[
        (trades["p_touch"] >= float(cell["min_p_touch"]))
        & (trades["p_direction_to_pool"] >= float(cell["min_p_direction"]))
        & (trades["distance_atr"] >= float(cell["distance_low"]))
        & (trades["distance_atr"] < float(cell["distance_high"]))
        & (trades["target_fraction"] == float(cell["target_fraction"]))
        & (trades["stop_atr_mult"] == float(cell["stop_atr_mult"]))
        & (trades["requested_max_hold_bars"] == int(cell["max_hold_bars"]))
    ].copy()


def _scenario_filters() -> Dict[str, Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    return {
        "all": ("Baseline best marginal cell", lambda d: d),
        "long_only": ("Only pool-side UP/long trades", lambda d: d[d["direction"] == "UP"]),
        "short_only": ("Only pool-side DOWN/short trades", lambda d: d[d["direction"] == "DOWN"]),
        "morning_only": ("Entries in morning window only", lambda d: d[d["time_bucket"] == "morning"]),
        "midday_only": ("Entries in midday window only", lambda d: d[d["time_bucket"] == "midday"]),
        "afternoon_only": ("Entries in afternoon window only", lambda d: d[d["time_bucket"] == "afternoon"]),
        "morning_midday": (
            "Block afternoon; keep morning plus midday",
            lambda d: d[d["time_bucket"].isin(["morning", "midday"])],
        ),
        "auto_only": ("AUTO sector only", lambda d: d[d["sector"] == "AUTO"]),
        "pharma_only": ("PHARMA sector only", lambda d: d[d["sector"] == "PHARMA"]),
        "fmcg_only": ("FMCG sector only", lambda d: d[d["sector"] == "FMCG"]),
        "banking_only": ("BANKING sector only", lambda d: d[d["sector"] == "BANKING"]),
        "it_only": ("IT sector only", lambda d: d[d["sector"] == "IT"]),
        "ex_banking_it": (
            "Exclude BANKING and IT",
            lambda d: d[~d["sector"].isin(["BANKING", "IT"])],
        ),
        "auto_pharma_fmcg": (
            "AUTO + PHARMA + FMCG",
            lambda d: d[d["sector"].isin(["AUTO", "PHARMA", "FMCG"])],
        ),
        "long_morning": (
            "Long-only morning",
            lambda d: d[(d["direction"] == "UP") & (d["time_bucket"] == "morning")],
        ),
        "long_morning_midday": (
            "Long-only, block afternoon",
            lambda d: d[(d["direction"] == "UP") & (d["time_bucket"].isin(["morning", "midday"]))],
        ),
        "long_ex_banking_it": (
            "Long-only, exclude BANKING and IT",
            lambda d: d[(d["direction"] == "UP") & (~d["sector"].isin(["BANKING", "IT"]))],
        ),
        "long_auto_pharma_fmcg": (
            "Long-only AUTO + PHARMA + FMCG",
            lambda d: d[(d["direction"] == "UP") & (d["sector"].isin(["AUTO", "PHARMA", "FMCG"]))],
        ),
        "long_morning_ex_banking_it": (
            "Long-only morning, exclude BANKING and IT",
            lambda d: d[
                (d["direction"] == "UP")
                & (d["time_bucket"] == "morning")
                & (~d["sector"].isin(["BANKING", "IT"]))
            ],
        ),
        "long_morning_midday_ex_banking_it": (
            "Long-only, block afternoon, exclude BANKING and IT",
            lambda d: d[
                (d["direction"] == "UP")
                & (d["time_bucket"].isin(["morning", "midday"]))
                & (~d["sector"].isin(["BANKING", "IT"]))
            ],
        ),
    }


def _summarize(name: str, description: str, df: pd.DataFrame, q_label: str,
               q_fraction: float, dsr_trials: int) -> Dict:
    r = pd.to_numeric(df["net_r"], errors="coerce").dropna()
    ts = pd.to_datetime(df["entry_at"]) if len(df) else pd.Series(dtype="datetime64[ns]")
    if len(df):
        midpoint = ts.min() + (ts.max() - ts.min()) / 2
        first = df.loc[ts < midpoint, "net_r"]
        second = df.loc[ts >= midpoint, "net_r"]
    else:
        first = second = pd.Series(dtype=float)
    sectors = df["sector"].value_counts() if len(df) else pd.Series(dtype=int)
    long_n = int((df["direction"] == "UP").sum()) if len(df) else 0
    short_n = int((df["direction"] == "DOWN").sum()) if len(df) else 0
    cost_125 = r - ((df["total_cost"] * 0.25) / df["risk_inr"]) if len(df) else pd.Series(dtype=float)
    cost_150 = r - ((df["total_cost"] * 0.50) / df["risk_inr"]) if len(df) else pd.Series(dtype=float)
    temporal_gap = float(abs(first.mean() - second.mean())) if len(first) and len(second) else np.nan
    mean_r = float(r.mean()) if len(r) else np.nan
    second_mean = float(second.mean()) if len(second) else np.nan
    cost_150_mean = float(cost_150.mean()) if len(cost_150) else np.nan
    sectors_with_30 = int((sectors >= 30).sum()) if len(sectors) else 0

    # This is a diagnostic flag, not live approval. It asks whether a pocket is
    # worth re-validating with a constrained sweep / full-universe run.
    focused_survives = bool(
        len(r) >= 150
        and np.isfinite(mean_r) and mean_r > 0.15
        and len(second) >= 50
        and np.isfinite(second_mean) and second_mean > 0.10
        and np.isfinite(cost_150_mean) and cost_150_mean > 0.05
        and sectors_with_30 >= 2
    )
    strict_ready = bool(
        focused_survives
        and len(r) >= 300
        and sectors_with_30 >= 3
        and np.isfinite(temporal_gap) and temporal_gap < 0.15
        and _profit_factor(r) > 1.25
    )

    return {
        "scenario": name,
        "description": description,
        "q_cohort": q_label,
        "q_fraction": q_fraction,
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "mean_r": mean_r,
        "median_r": float(r.median()) if len(r) else np.nan,
        "profit_factor": float(_profit_factor(r)) if len(r) else np.nan,
        "max_drawdown_r": float(_max_drawdown(r)) if len(r) else np.nan,
        "trade_sharpe": float(_trade_sharpe(r)) if len(r) else np.nan,
        "dsr": float(_deflated_sharpe(r, n_trials=dsr_trials)) if len(r) else np.nan,
        "mean_r_cost_1_25x": float(cost_125.mean()) if len(cost_125) else np.nan,
        "mean_r_cost_1_50x": cost_150_mean,
        "first_half_n": int(len(first)),
        "second_half_n": int(len(second)),
        "first_half_mean_r": float(first.mean()) if len(first) else np.nan,
        "second_half_mean_r": second_mean,
        "temporal_gap_r": temporal_gap,
        "long_n": long_n,
        "short_n": short_n,
        "sectors_with_30_trades": sectors_with_30,
        "top_sector": str(sectors.index[0]) if len(sectors) else "",
        "top_sector_trades": int(sectors.iloc[0]) if len(sectors) else 0,
        "morning_trades": int((df["time_bucket"] == "morning").sum()) if len(df) else 0,
        "midday_trades": int((df["time_bucket"] == "midday").sum()) if len(df) else 0,
        "afternoon_trades": int((df["time_bucket"] == "afternoon").sum()) if len(df) else 0,
        "target_exit_rate": float((df["exit_reason"] == "target").mean()) if len(df) else np.nan,
        "stop_exit_rate": float((df["exit_reason"] == "stop").mean()) if len(df) else np.nan,
        "time_exit_rate": float((df["exit_reason"] == "time_exit").mean()) if len(df) else np.nan,
        "focused_survives": focused_survives,
        "strict_ready": strict_ready,
    }


def run_diagnostics(base: pd.DataFrame, dsr_trials: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for scenario, (description, fn) in _scenario_filters().items():
        sdf = fn(base).copy()
        if sdf.empty:
            continue
        for q_label, q_fraction in Q_COHORTS:
            qdf = sdf.loc[_q_mask(sdf, q_fraction)].copy()
            rows.append(_summarize(scenario, description, qdf, q_label, q_fraction, dsr_trials))
    return pd.DataFrame(rows).sort_values(
        ["focused_survives", "strict_ready", "mean_r", "trades"],
        ascending=[False, False, False, False],
    )


def _fmt_pf(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if math.isinf(float(x)):
        return "inf"
    return fnum(x, 3)


def _report_rows(df: pd.DataFrame, limit: int = 20) -> List[Dict]:
    rows = []
    for row in df.head(limit).to_dict(orient="records"):
        rows.append({
            "scenario": row["scenario"],
            "Q": row["q_cohort"],
            "n": int(row["trades"]),
            "win": pct(row["win_rate"], 1),
            "mean_R": fnum(row["mean_r"], 3),
            "PF": _fmt_pf(row["profit_factor"]),
            "cost1.5x": fnum(row["mean_r_cost_1_50x"], 3),
            "1st/2nd": f"{fnum(row['first_half_mean_r'], 3)}/{fnum(row['second_half_mean_r'], 3)}",
            "L/S": f"{int(row['long_n'])}/{int(row['short_n'])}",
            "sectors": int(row["sectors_with_30_trades"]),
            "survives": str(bool(row["focused_survives"])),
            "strict": str(bool(row["strict_ready"])),
        })
    return rows


def _detail_rows(base: pd.DataFrame, col: str) -> List[Dict]:
    rows = []
    for key, df in base.groupby(col):
        r = pd.to_numeric(df["net_r"], errors="coerce")
        rows.append({
            col: key,
            "n": int(len(df)),
            "win": pct(float((r > 0).mean()), 1),
            "mean_R": fnum(float(r.mean()), 3),
            "PF": _fmt_pf(_profit_factor(r)),
        })
    return sorted(rows, key=lambda row: float(row["mean_R"]) if row["mean_R"] != "n/a" else -999, reverse=True)


def build_report(summary: pd.DataFrame, base: pd.DataFrame, cell: pd.Series) -> str:
    survivors = summary[summary["focused_survives"]].sort_values("mean_r", ascending=False)
    strict = summary[summary["strict_ready"]].sort_values("mean_r", ascending=False)
    if not strict.empty:
        decision = "FOCUSED_POCKET_READY_FOR_CONSTRAINED_VALIDATION"
        reason = "At least one filtered pocket is broad enough for the next constrained validation run."
    elif not survivors.empty:
        decision = "FOCUSED_POCKET_PROMISING_BUT_NARROW"
        reason = "Filtered pockets survive second-half and cost stress, but remain post-hoc/narrow."
    else:
        decision = "NO_FOCUSED_POCKET"
        reason = "No simple focused filter survives the diagnostic thresholds."

    best = (
        strict.iloc[0]
        if not strict.empty
        else survivors.iloc[0]
        if not survivors.empty
        else summary.sort_values("mean_r", ascending=False).iloc[0]
    )
    lines = [
        "# Phase 4 Track A Focused Diagnostics",
        "",
        "## Verdict",
        "",
        f"- Decision: `{decision}`",
        f"- {reason}",
        f"- Best broad diagnostic pocket: `{best['scenario']}` / `{best['q_cohort']}` "
        f"with `{int(best['trades'])}` trades, mean R `{best['mean_r']:+.3f}`, "
        f"PF `{best['profit_factor']:.2f}`, second-half R `{best['second_half_mean_r']:+.3f}`, "
        f"and 1.5x-cost R `{best['mean_r_cost_1_50x']:+.3f}`.",
        "- This is still not live-trade approval; these filters were discovered after the sweep.",
        "",
        "## Base Cell",
        "",
        f"- Gate: `{cell['gate']}`",
        f"- Geometry: target_fraction `{cell['target_fraction']}`, stop `{cell['stop_atr_mult']} ATR`, hold `{int(cell['max_hold_bars'])}` bars",
        f"- Base trades: `{len(base)}`",
        "",
        "## Strict-Ready Filtered Pockets",
        "",
        markdown_table(
            _report_rows(strict, 10),
            ["scenario", "Q", "n", "win", "mean_R", "PF", "cost1.5x", "1st/2nd", "L/S", "sectors", "survives", "strict"],
        ),
        "",
        "## Focused Survivors",
        "",
        markdown_table(
            _report_rows(survivors, 20),
            ["scenario", "Q", "n", "win", "mean_R", "PF", "cost1.5x", "1st/2nd", "L/S", "sectors", "survives", "strict"],
        ),
        "",
        "## Top Diagnostics By Mean R",
        "",
        markdown_table(
            _report_rows(summary.sort_values("mean_r", ascending=False), 25),
            ["scenario", "Q", "n", "win", "mean_R", "PF", "cost1.5x", "1st/2nd", "L/S", "sectors", "survives", "strict"],
        ),
        "",
        "## Base Breakdown By Sector",
        "",
        markdown_table(_detail_rows(base, "sector"), ["sector", "n", "win", "mean_R", "PF"]),
        "",
        "## Base Breakdown By Direction",
        "",
        markdown_table(_detail_rows(base, "direction"), ["direction", "n", "win", "mean_R", "PF"]),
        "",
        "## Base Breakdown By Time Bucket",
        "",
        markdown_table(_detail_rows(base, "time_bucket"), ["time_bucket", "n", "win", "mean_R", "PF"]),
        "",
        "## Interpretation",
        "",
        "- The short side is actively harmful in the base cell.",
        "- BANKING and IT are negative in the base cell; AUTO/PHARMA/FMCG carry the useful signal.",
        "- Blocking afternoon entries improves the result, but can reduce sector breadth.",
        "- The cleanest next experiment is a constrained validation run, not live trading: long-only, AUTO/PHARMA/FMCG, same Track A gate family, and explicit no-afternoon sensitivity.",
        "- Triple-barrier/final-stage trade modeling remains the right model upgrade after this pocket is validated out-of-sample or on the full universe.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", default=str(DEFAULT_TRADES))
    parser.add_argument("--cells", default=str(DEFAULT_CELLS))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--dsr-trials", type=int, default=700)
    args = parser.parse_args()

    trades_path = Path(args.trades)
    cells_path = Path(args.cells)
    if not trades_path.exists():
        raise FileNotFoundError(f"missing Track A trade parquet: {trades_path}")
    if not cells_path.exists():
        raise FileNotFoundError(f"missing Track A cell CSV: {cells_path}")

    trades = pd.read_parquet(trades_path)
    cells = pd.read_csv(cells_path)
    cell = _select_base_cell(cells)
    base = _base_trades(trades, cell)
    if base.empty:
        raise ValueError("selected base cell produced no trade rows")
    summary = run_diagnostics(base, dsr_trials=args.dsr_trials)

    summary_out = Path(args.summary_out)
    report_out = Path(args.report_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    report_out.write_text(build_report(summary, base, cell), encoding="utf-8")
    print(f"wrote {summary_out}")
    print(f"wrote {report_out}")


if __name__ == "__main__":
    main()
