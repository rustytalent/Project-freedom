"""Phase 4 Track A CPCV-style path audit and first component nulls.

This is the first hardening layer after the constrained Track A pocket passed.
It intentionally does not create synthetic random pools or ATR-offset pools yet;
those require a simulator rerun against generated pool candidates. This script
uses the already-computed Track A v2 trade parquet to answer faster questions:

1. Does the fixed pocket survive many chronological validation paths?
2. Does excluding BANKING/IT help consistently, or only in one OOS slice?
3. Is the short side still a negative control?
4. Does shuffled direction score degrade the pocket?

The output is research-only and should not be used as live approval.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q_audit_common import fnum, markdown_table, pct  # noqa: E402


DEFAULT_TRADES = Path("output_phase4_track_a_pretouch_sweep/pretouch_sweep_trades.parquet")
DEFAULT_PATHS = Path("reports/phase4_track_a_cpcv_paths.csv")
DEFAULT_NULLS = Path("reports/phase4_track_a_component_nulls.csv")
DEFAULT_REPORT = Path("reports/phase4_track_a_cpcv_nulls.md")

POCKET_MIN_TOUCH = 0.75
POCKET_MIN_DIRECTION = 0.65
POCKET_DISTANCE_LOW = 3.0
POCKET_DISTANCE_HIGH = 8.0
POCKET_TARGET_FRACTION = 1.0
POCKET_STOP_ATR_MULT = 2.0
POCKET_HOLD_BARS = 60
POCKET_ALLOWED_SECTORS = {"AUTO", "PHARMA", "FMCG"}


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


def _metric_row(df: pd.DataFrame, *, label: str, dsr_trials: int) -> Dict:
    r = pd.to_numeric(df["net_r"], errors="coerce").dropna()
    return {
        "label": label,
        "n": int(len(r)),
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "mean_r": float(r.mean()) if len(r) else np.nan,
        "median_r": float(r.median()) if len(r) else np.nan,
        "profit_factor": float(_profit_factor(r)) if len(r) else np.nan,
        "max_drawdown_r": float(_max_drawdown(r)) if len(r) else np.nan,
        "trade_sharpe": float(_trade_sharpe(r)) if len(r) else np.nan,
        "dsr": float(_deflated_sharpe(r, dsr_trials)) if len(r) else np.nan,
    }


def _base_geometry(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[
        (trades["p_touch"] >= POCKET_MIN_TOUCH)
        & (trades["distance_atr"] >= POCKET_DISTANCE_LOW)
        & (trades["distance_atr"] < POCKET_DISTANCE_HIGH)
        & (trades["target_fraction"] == POCKET_TARGET_FRACTION)
        & (trades["stop_atr_mult"] == POCKET_STOP_ATR_MULT)
        & (trades["requested_max_hold_bars"] == POCKET_HOLD_BARS)
    ].copy()


def _scenario_frame(base: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == "long_ex_banking_it":
        return base[
            (base["direction"] == "UP")
            & (base["p_direction_to_pool"] >= POCKET_MIN_DIRECTION)
            & (base["sector"].isin(POCKET_ALLOWED_SECTORS))
        ].copy()
    if scenario == "long_all_sectors":
        return base[
            (base["direction"] == "UP")
            & (base["p_direction_to_pool"] >= POCKET_MIN_DIRECTION)
        ].copy()
    if scenario == "banking_it_long_control":
        return base[
            (base["direction"] == "UP")
            & (base["p_direction_to_pool"] >= POCKET_MIN_DIRECTION)
            & (base["sector"].isin(["BANKING", "IT"]))
        ].copy()
    if scenario == "short_all_sectors_control":
        return base[
            (base["direction"] == "DOWN")
            & (base["p_direction_to_pool"] >= POCKET_MIN_DIRECTION)
        ].copy()
    raise ValueError(f"unknown scenario: {scenario}")


def _assign_blocks(trades: pd.DataFrame, n_blocks: int) -> pd.DataFrame:
    out = trades.sort_values("entry_at_dt").copy()
    if out.empty:
        out["cpcv_block"] = []
        return out
    # Rank rows rather than dates so dense event days do not create empty folds.
    ranks = np.arange(len(out))
    out["cpcv_block"] = np.floor(ranks * n_blocks / max(len(out), 1)).astype(int)
    out["cpcv_block"] = out["cpcv_block"].clip(0, n_blocks - 1)
    return out


def _path_combinations(n_blocks: int, validation_blocks_per_path: int,
                       max_paths: int | None) -> List[Tuple[int, ...]]:
    combos = list(itertools.combinations(range(n_blocks), validation_blocks_per_path))
    if max_paths is not None and len(combos) > max_paths:
        # Deterministic spread over all combinations.
        idx = np.linspace(0, len(combos) - 1, max_paths).round().astype(int)
        combos = [combos[int(i)] for i in idx]
    return combos


def run_cpcv_paths(
    trades: pd.DataFrame,
    *,
    n_blocks: int,
    validation_blocks_per_path: int,
    max_paths: int | None,
    dsr_trials: int,
    min_path_trades: int,
) -> pd.DataFrame:
    base = _base_geometry(trades)
    scenarios = [
        "long_ex_banking_it",
        "long_all_sectors",
        "banking_it_long_control",
        "short_all_sectors_control",
    ]
    rows: List[Dict] = []
    for scenario in scenarios:
        sdf = _assign_blocks(_scenario_frame(base, scenario), n_blocks)
        for path_id, blocks in enumerate(
            _path_combinations(n_blocks, validation_blocks_per_path, max_paths),
            start=1,
        ):
            vdf = sdf[sdf["cpcv_block"].isin(blocks)].copy()
            row = _metric_row(vdf, label=scenario, dsr_trials=dsr_trials)
            row.update({
                "scenario": scenario,
                "path_id": path_id,
                "validation_blocks": "+".join(str(x) for x in blocks),
                "n_blocks": n_blocks,
                "validation_blocks_per_path": validation_blocks_per_path,
                "path_pass": bool(
                    row["n"] >= min_path_trades
                    and np.isfinite(row["mean_r"])
                    and row["mean_r"] > 0.0
                    and np.isfinite(row["profit_factor"])
                    and row["profit_factor"] > 1.0
                ),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def run_component_nulls(
    trades: pd.DataFrame,
    *,
    shuffles: int,
    seed: int,
    dsr_trials: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = _base_geometry(trades)
    actual = _scenario_frame(base, "long_ex_banking_it")
    rows = [_metric_row(actual, label="actual_long_ex_banking_it", dsr_trials=dsr_trials)]
    rows[-1]["kind"] = "actual"

    all_sectors = _scenario_frame(base, "long_all_sectors")
    rows.append(_metric_row(all_sectors, label="actual_long_all_sectors", dsr_trials=dsr_trials))
    rows[-1]["kind"] = "sector_control"

    banking_it = _scenario_frame(base, "banking_it_long_control")
    rows.append(_metric_row(banking_it, label="actual_banking_it_long_control", dsr_trials=dsr_trials))
    rows[-1]["kind"] = "negative_control"

    short_side = _scenario_frame(base, "short_all_sectors_control")
    rows.append(_metric_row(short_side, label="actual_short_all_sectors_control", dsr_trials=dsr_trials))
    rows[-1]["kind"] = "negative_control"

    shuffle_universe = base[
        (base["direction"] == "UP")
        & (base["sector"].isin(POCKET_ALLOWED_SECTORS))
    ].copy()
    p_values = shuffle_universe["p_direction_to_pool"].to_numpy(dtype=float)
    actual_mean = float(actual["net_r"].mean()) if len(actual) else np.nan
    shuffle_means = []
    for i in range(shuffles):
        if len(shuffle_universe) == 0:
            sdf = shuffle_universe
        else:
            shuffled = rng.permutation(p_values)
            sdf = shuffle_universe[shuffled >= POCKET_MIN_DIRECTION].copy()
        row = _metric_row(sdf, label=f"shuffled_direction_{i + 1:03d}", dsr_trials=dsr_trials)
        row["kind"] = "shuffled_direction_null"
        rows.append(row)
        if np.isfinite(row["mean_r"]):
            shuffle_means.append(row["mean_r"])

    out = pd.DataFrame(rows)
    if shuffle_means and np.isfinite(actual_mean):
        p_value = float(np.mean(np.asarray(shuffle_means) >= actual_mean))
        out["actual_vs_shuffle_p_value"] = p_value
    else:
        out["actual_vs_shuffle_p_value"] = np.nan
    return out


def summarize_paths(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, sdf in paths.groupby("scenario"):
        valid = sdf[sdf["n"] > 0]
        rows.append({
            "scenario": scenario,
            "paths": int(len(sdf)),
            "paths_with_trades": int(len(valid)),
            "median_n": float(valid["n"].median()) if len(valid) else np.nan,
            "mean_path_r": float(valid["mean_r"].mean()) if len(valid) else np.nan,
            "median_path_r": float(valid["mean_r"].median()) if len(valid) else np.nan,
            "worst_path_r": float(valid["mean_r"].min()) if len(valid) else np.nan,
            "best_path_r": float(valid["mean_r"].max()) if len(valid) else np.nan,
            "positive_path_rate": float((valid["mean_r"] > 0.0).mean()) if len(valid) else np.nan,
            "pf_gt_1_rate": float((valid["profit_factor"] > 1.0).mean()) if len(valid) else np.nan,
            "path_pass_rate": float(valid["path_pass"].mean()) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows)


def _fmt_pf(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if math.isinf(float(value)):
        return "inf"
    return fnum(value, 3)


def _path_rows(df: pd.DataFrame) -> List[Dict]:
    rows = []
    for row in df.sort_values("path_pass_rate", ascending=False).to_dict(orient="records"):
        rows.append({
            "scenario": row["scenario"],
            "paths": int(row["paths"]),
            "median_n": fnum(row["median_n"], 1),
            "mean_R": fnum(row["mean_path_r"], 3),
            "median_R": fnum(row["median_path_r"], 3),
            "worst_R": fnum(row["worst_path_r"], 3),
            "positive": pct(row["positive_path_rate"], 1),
            "PF>1": pct(row["pf_gt_1_rate"], 1),
            "pass": pct(row["path_pass_rate"], 1),
        })
    return rows


def _null_rows(nulls: pd.DataFrame) -> List[Dict]:
    if nulls.empty:
        return []
    # Actual/control rows, plus aggregate shuffled null row.
    rows = []
    controls = nulls[nulls["kind"] != "shuffled_direction_null"].copy()
    for row in controls.to_dict(orient="records"):
        rows.append({
            "kind": row["kind"],
            "label": row["label"],
            "n": int(row["n"]),
            "win": pct(row["win_rate"], 1),
            "mean_R": fnum(row["mean_r"], 3),
            "PF": _fmt_pf(row["profit_factor"]),
            "DSR": pct(row["dsr"], 1),
        })
    shuf = nulls[nulls["kind"] == "shuffled_direction_null"].copy()
    if not shuf.empty:
        rows.append({
            "kind": "shuffled_direction_null",
            "label": f"{len(shuf)} shuffles; p(null>=actual)="
                     f"{float(shuf['actual_vs_shuffle_p_value'].iloc[0]):.3f}",
            "n": f"{int(shuf['n'].median())} median",
            "win": pct(float(shuf["win_rate"].median()), 1),
            "mean_R": f"{float(shuf['mean_r'].median()):.3f} median",
            "PF": f"{float(shuf['profit_factor'].median()):.3f} median",
            "DSR": pct(float(shuf["dsr"].median()), 1),
        })
    return rows


def decide(path_summary: pd.DataFrame, nulls: pd.DataFrame) -> Tuple[str, str]:
    pocket = path_summary[path_summary["scenario"] == "long_ex_banking_it"]
    if pocket.empty:
        return "FAIL", "No Track A pocket paths were produced."
    row = pocket.iloc[0]
    pass_rate = float(row["path_pass_rate"])
    positive_rate = float(row["positive_path_rate"])
    worst = float(row["worst_path_r"])
    p_value = float(nulls["actual_vs_shuffle_p_value"].dropna().iloc[0]) \
        if "actual_vs_shuffle_p_value" in nulls and nulls["actual_vs_shuffle_p_value"].notna().any() else np.nan
    if pass_rate >= 0.70 and positive_rate >= 0.80 and worst > -0.15 and (pd.isna(p_value) or p_value <= 0.10):
        return "CPCV_COMPONENT_PASS", (
            "The fixed pocket survives most chronological paths and beats the shuffled-direction null."
        )
    if pass_rate >= 0.50 and positive_rate >= 0.65:
        return "CPCV_COMPONENT_MARGINAL", (
            "The fixed pocket is still positive across many paths, but robustness is not decisive."
        )
    return "CPCV_COMPONENT_FAIL", (
        "The fixed pocket does not survive enough chronological validation paths."
    )


def build_report(paths: pd.DataFrame, nulls: pd.DataFrame, args: argparse.Namespace) -> str:
    path_summary = summarize_paths(paths)
    decision, reason = decide(path_summary, nulls)
    pocket_paths = paths[paths["scenario"] == "long_ex_banking_it"].copy()
    weak_paths = pocket_paths.sort_values("mean_r").head(8)
    lines = [
        "# Phase 4 Track A CPCV / Component Nulls",
        "",
        "## Verdict",
        "",
        f"- Decision: `{decision}`",
        f"- {reason}",
        "- Scope note: this is a fixed-pocket CPCV-style path audit using existing v2 Track A trades.",
        "- It includes shuffled-direction and sector/short controls, but not synthetic random-pool or ATR-offset nulls yet.",
        "",
        "## Fixed Pocket",
        "",
        f"- T >= `{POCKET_MIN_TOUCH}`",
        f"- Direction-to-pool >= `{POCKET_MIN_DIRECTION}`",
        f"- Distance `{POCKET_DISTANCE_LOW}-{POCKET_DISTANCE_HIGH} ATR`",
        f"- Geometry target fraction `{POCKET_TARGET_FRACTION}`, stop `{POCKET_STOP_ATR_MULT} ATR`, hold `{POCKET_HOLD_BARS}` bars",
        f"- Primary pocket: long-only, sectors `{', '.join(sorted(POCKET_ALLOWED_SECTORS))}`",
        "",
        "## Path Summary",
        "",
        markdown_table(
            _path_rows(path_summary),
            ["scenario", "paths", "median_n", "mean_R", "median_R", "worst_R", "positive", "PF>1", "pass"],
        ),
        "",
        "## Component Nulls",
        "",
        markdown_table(
            _null_rows(nulls),
            ["kind", "label", "n", "win", "mean_R", "PF", "DSR"],
        ),
        "",
        "## Weakest Pocket Paths",
        "",
        markdown_table(
            [
                {
                    "path": int(row["path_id"]),
                    "blocks": row["validation_blocks"],
                    "n": int(row["n"]),
                    "mean_R": fnum(row["mean_r"], 3),
                    "PF": _fmt_pf(row["profit_factor"]),
                    "pass": str(bool(row["path_pass"])),
                }
                for row in weak_paths.to_dict(orient="records")
            ],
            ["path", "blocks", "n", "mean_R", "PF", "pass"],
        ),
        "",
        "## Interpretation",
        "",
        "- If the pocket passes here, the next missing test is the expensive synthetic-null suite: random pools and ATR-offset pools.",
        "- If all-sectors is close to ex-BANKING/IT, the sector exclusion should become a soft risk weight rather than a hard rule.",
        "- If shuffled direction is close to actual, the direction model is not contributing enough and Track A should simplify toward proximity-only.",
        "- This report still does not authorize live trading; it only decides whether to invest compute in full nulls/final-stage labels.",
        "",
        "## Outputs",
        "",
        f"- Path CSV: `{args.paths_out}`",
        f"- Null CSV: `{args.nulls_out}`",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--paths-out", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--nulls-out", type=Path, default=DEFAULT_NULLS)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--n-blocks", type=int, default=10)
    parser.add_argument("--validation-blocks-per-path", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=None)
    parser.add_argument("--min-path-trades", type=int, default=30)
    parser.add_argument("--direction-shuffles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--dsr-trials", type=int, default=650)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trades.exists():
        raise FileNotFoundError(f"missing Track A trade parquet: {args.trades}")
    trades = pd.read_parquet(args.trades).copy()
    trades["entry_at_dt"] = pd.to_datetime(trades["entry_at"])
    paths = run_cpcv_paths(
        trades,
        n_blocks=args.n_blocks,
        validation_blocks_per_path=args.validation_blocks_per_path,
        max_paths=args.max_paths,
        dsr_trials=args.dsr_trials,
        min_path_trades=args.min_path_trades,
    )
    nulls = run_component_nulls(
        trades,
        shuffles=args.direction_shuffles,
        seed=args.seed,
        dsr_trials=args.dsr_trials,
    )
    args.paths_out.parent.mkdir(parents=True, exist_ok=True)
    args.nulls_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    paths.to_csv(args.paths_out, index=False)
    nulls.to_csv(args.nulls_out, index=False)
    args.report_out.write_text(build_report(paths, nulls, args), encoding="utf-8")
    print(f"wrote {args.paths_out}")
    print(f"wrote {args.nulls_out}")
    print(f"wrote {args.report_out}")


if __name__ == "__main__":
    main()
