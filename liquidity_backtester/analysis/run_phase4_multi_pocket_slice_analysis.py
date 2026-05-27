"""Phase 4 Track A multi-pocket slice discovery.

This script reads the already-computed Track A v2 pre-touch sweep trades and
looks for additional research pockets. It does not retrain models, rerun the
1-minute simulator, or change production gates.

The output is deliberately hypothesis-generating: every candidate reported here
must still be validated with a focused mini-sweep, CPCV, and synthetic nulls.
"""

from __future__ import annotations

import argparse
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
DEFAULT_CANDIDATES = Path("output_phase4_track_a_pretouch_sweep/pretouch_candidates.parquet")
DEFAULT_SLICES = Path("reports/phase4_multi_pocket_slices.csv")
DEFAULT_CANDIDATES_OUT = Path("reports/phase4_multi_pocket_candidates.csv")
DEFAULT_CORR = Path("reports/phase4_multi_pocket_correlations.csv")
DEFAULT_REPORT = Path("reports/phase4_multi_pocket_slice_analysis.md")

MERGE_KEYS = ["symbol", "fold", "pool_idx", "bar_idx"]
ENRICH_COLUMNS = [
    "headline_factor",
    "n_contributors",
    "pool_n_tfs",
    "vol_ratio",
    "adx_14",
    "range_6_atr",
    "ret_6",
    "ret_24",
    "minutes_since_session_open",
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


def _fmt_pf(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if math.isinf(float(x)):
        return "inf"
    return fnum(x, 3)


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing Track A sweep trades: {path}")
    trades = pd.read_parquet(path)
    required = {
        "symbol",
        "sector",
        "fold",
        "pool_idx",
        "bar_idx",
        "entry_at",
        "time_bucket",
        "direction",
        "factor",
        "tf_bucket",
        "distance_atr",
        "p_touch",
        "p_direction_to_pool",
        "pool_quality",
        "net_r",
        "total_cost",
        "risk_inr",
    }
    missing = sorted(required - set(trades.columns))
    if missing:
        raise ValueError(f"Track A trade parquet missing required columns: {missing}")
    return trades


def _enrich_trades(trades: pd.DataFrame, candidates_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    out = trades.copy()
    unavailable: List[str] = []
    if not candidates_path.exists():
        return out, [f"candidate enrichment file missing: {candidates_path}"]

    candidates = pd.read_parquet(candidates_path)
    missing_keys = [col for col in MERGE_KEYS if col not in candidates.columns]
    if missing_keys:
        return out, [f"candidate enrichment missing merge keys: {missing_keys}"]

    present_cols = [col for col in ENRICH_COLUMNS if col in candidates.columns]
    unavailable.extend([col for col in ENRICH_COLUMNS if col not in candidates.columns])
    if not present_cols:
        return out, unavailable

    enrich = candidates[MERGE_KEYS + present_cols].drop_duplicates(MERGE_KEYS)
    out = out.merge(enrich, on=MERGE_KEYS, how="left", suffixes=("", "_candidate"))
    if "headline_factor" in out.columns:
        out["factor_enriched"] = out["headline_factor"].fillna(out["factor"])
    else:
        out["factor_enriched"] = out["factor"]
    return out, unavailable


def _bucketize(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    unavailable: List[str] = []
    out["entry_at_dt"] = pd.to_datetime(out["entry_at"], errors="coerce")
    out["entry_date"] = out["entry_at_dt"].dt.date.astype(str)
    out["day_of_week"] = out["entry_at_dt"].dt.day_name()
    out["distance_bucket"] = pd.cut(
        out["distance_atr"],
        bins=[0, 1, 3, 5, 8, 12, 20, np.inf],
        labels=["0-1", "1-3", "3-5", "5-8", "8-12", "12-20", "20+"],
        right=False,
    ).astype("string").fillna("unknown")
    out["p_touch_tier"] = pd.cut(
        out["p_touch"],
        bins=[0, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01],
        labels=["<75", "75-80", "80-85", "85-90", "90-95", "95+"],
        right=False,
    ).astype("string").fillna("unknown")
    out["direction_conf_tier"] = pd.cut(
        out["p_direction_to_pool"],
        bins=[0, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01],
        labels=["<55", "55-60", "60-65", "65-70", "70-80", "80+"],
        right=False,
    ).astype("string").fillna("unknown")
    out["q_percentile"] = out["pool_quality"].rank(method="average", pct=True)
    out["q_percentile_tier"] = pd.cut(
        out["q_percentile"],
        bins=[0, 0.50, 0.75, 0.90, 0.95, 1.01],
        labels=["bottom50", "top50-25", "top25-10", "top10-5", "top5"],
        include_lowest=True,
        right=False,
    ).astype("string").fillna("unknown")

    if "vol_ratio" in out.columns and out["vol_ratio"].notna().any():
        out["vol_regime"] = pd.cut(
            out["vol_ratio"],
            bins=[-np.inf, 0.80, 1.20, np.inf],
            labels=["low_vol", "normal_vol", "high_vol"],
        ).astype("string").fillna("unknown")
    else:
        unavailable.append("vol_regime")

    if "n_contributors" in out.columns and out["n_contributors"].notna().any():
        out["confluence_bucket"] = pd.cut(
            out["n_contributors"],
            bins=[0, 1, 2, 3, np.inf],
            labels=["1", "2", "3", "4+"],
            right=True,
        ).astype("string").fillna("unknown")
    else:
        unavailable.append("confluence_bucket")

    if "pool_n_tfs" in out.columns and out["pool_n_tfs"].notna().any():
        out["tf_count_bucket"] = pd.cut(
            out["pool_n_tfs"],
            bins=[0, 1, 2, 3, np.inf],
            labels=["1", "2", "3", "4+"],
            right=True,
        ).astype("string").fillna("unknown")
    else:
        unavailable.append("tf_count_bucket")

    return out, unavailable


def _slice_columns(df: pd.DataFrame) -> Tuple[List[Tuple[str, str]], List[str]]:
    requested = [
        ("time_bucket", "time_bucket"),
        ("distance_bucket", "distance_bucket"),
        ("sector", "sector"),
        ("symbol", "symbol"),
        ("factor", "factor_enriched"),
        ("tf_bucket", "tf_bucket"),
        ("tf_count_bucket", "tf_count_bucket"),
        ("direction", "direction"),
        ("p_touch_tier", "p_touch_tier"),
        ("direction_conf_tier", "direction_conf_tier"),
        ("q_percentile_tier", "q_percentile_tier"),
        ("day_of_week", "day_of_week"),
        ("vol_regime", "vol_regime"),
        ("confluence_bucket", "confluence_bucket"),
    ]
    available: List[Tuple[str, str]] = []
    unavailable: List[str] = []
    for family, col in requested:
        if col in df.columns and df[col].notna().any():
            available.append((family, col))
        else:
            unavailable.append(family)
    return available, unavailable


def _cross_columns(df: pd.DataFrame) -> Tuple[List[Tuple[str, Sequence[str]]], List[str]]:
    requested = [
        ("time_x_distance", ["time_bucket", "distance_bucket"]),
        ("sector_x_direction", ["sector", "direction"]),
        ("factor_x_distance", ["factor_enriched", "distance_bucket"]),
        ("volatility_x_distance", ["vol_regime", "distance_bucket"]),
        ("time_x_direction", ["time_bucket", "direction"]),
    ]
    available: List[Tuple[str, Sequence[str]]] = []
    unavailable: List[str] = []
    for family, cols in requested:
        missing = [col for col in cols if col not in df.columns or not df[col].notna().any()]
        if missing:
            unavailable.append(f"{family} missing {missing}")
        else:
            available.append((family, cols))
    return available, unavailable


def _cost_stress_r(df: pd.DataFrame, multiplier_add: float) -> pd.Series:
    risk = pd.to_numeric(df["risk_inr"], errors="coerce").replace(0, np.nan)
    extra_cost_r = (pd.to_numeric(df["total_cost"], errors="coerce") * multiplier_add) / risk
    return pd.to_numeric(df["net_r"], errors="coerce") - extra_cost_r.fillna(0.0)


def _candidate_decision(row: Dict) -> Tuple[bool, bool, str]:
    reasons: List[str] = []
    if row["trades"] < 100:
        reasons.append("n<100")
    if not np.isfinite(row["mean_r"]) or row["mean_r"] <= 0.10:
        reasons.append("mean_R<=0.10")
    if not np.isfinite(row["profit_factor"]) or row["profit_factor"] <= 1.30:
        reasons.append("PF<=1.30")
    if not np.isfinite(row["win_rate"]) or row["win_rate"] <= 0.50:
        reasons.append("win<=50%")
    if not np.isfinite(row["second_half_mean_r"]) or row["second_half_mean_r"] <= 0.0:
        reasons.append("second_half<=0")
    if not np.isfinite(row["mean_r_cost_1_50x"]) or row["mean_r_cost_1_50x"] <= 0.0:
        reasons.append("cost1.5x<=0")

    diagnostic_only = False
    if row["trades"] < 100:
        diagnostic_only = True
    if row["unique_symbols"] < 2:
        diagnostic_only = True
        reasons.append("one_symbol")
    if row["unique_sectors"] < 2:
        diagnostic_only = True
        reasons.append("one_sector")
    if (
        np.isfinite(row["first_half_mean_r"])
        and row["first_half_mean_r"] > 0.0
        and (not np.isfinite(row["second_half_mean_r"]) or row["second_half_mean_r"] <= 0.0)
    ):
        diagnostic_only = True
        reasons.append("first_half_only")

    candidate = not reasons and not diagnostic_only
    if candidate:
        return True, False, "candidate"
    if diagnostic_only:
        return False, True, "; ".join(dict.fromkeys(reasons)) or "diagnostic_only"
    return False, False, "; ".join(dict.fromkeys(reasons))


def _metric_row(df: pd.DataFrame, *, family: str, value: str, filter_expr: str,
                dsr_trials: int) -> Dict:
    r = pd.to_numeric(df["net_r"], errors="coerce").dropna()
    ts = pd.to_datetime(df["entry_at_dt"], errors="coerce")
    if len(df) and ts.notna().any():
        midpoint = ts.min() + (ts.max() - ts.min()) / 2
        first = df.loc[ts < midpoint, "net_r"]
        second = df.loc[ts >= midpoint, "net_r"]
    else:
        first = second = pd.Series(dtype=float)
    sectors = df["sector"].value_counts() if len(df) else pd.Series(dtype=int)
    symbols = df["symbol"].value_counts() if len(df) else pd.Series(dtype=int)
    cost_125 = _cost_stress_r(df, 0.25) if len(df) else pd.Series(dtype=float)
    cost_150 = _cost_stress_r(df, 0.50) if len(df) else pd.Series(dtype=float)
    row: Dict = {
        "slice_id": f"{family}:{value}",
        "family": family,
        "slice_value": value,
        "filter_expr": filter_expr,
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "mean_r": float(r.mean()) if len(r) else np.nan,
        "median_r": float(r.median()) if len(r) else np.nan,
        "profit_factor": float(_profit_factor(r)) if len(r) else np.nan,
        "max_drawdown_r": float(_max_drawdown(r)) if len(r) else np.nan,
        "trade_sharpe": float(_trade_sharpe(r)) if len(r) else np.nan,
        "dsr": float(_deflated_sharpe(r, dsr_trials)) if len(r) else np.nan,
        "mean_r_cost_1_25x": float(cost_125.mean()) if len(cost_125) else np.nan,
        "mean_r_cost_1_50x": float(cost_150.mean()) if len(cost_150) else np.nan,
        "first_half_n": int(len(first)),
        "second_half_n": int(len(second)),
        "first_half_mean_r": float(first.mean()) if len(first) else np.nan,
        "second_half_mean_r": float(second.mean()) if len(second) else np.nan,
        "temporal_gap_r": float(abs(first.mean() - second.mean())) if len(first) and len(second) else np.nan,
        "unique_sectors": int(df["sector"].nunique()) if len(df) else 0,
        "unique_symbols": int(df["symbol"].nunique()) if len(df) else 0,
        "top_sector": str(sectors.index[0]) if len(sectors) else "",
        "top_sector_trades": int(sectors.iloc[0]) if len(sectors) else 0,
        "top_symbol": str(symbols.index[0]) if len(symbols) else "",
        "top_symbol_trades": int(symbols.iloc[0]) if len(symbols) else 0,
        "long_n": int((df["direction"] == "UP").sum()) if len(df) else 0,
        "short_n": int((df["direction"] == "DOWN").sum()) if len(df) else 0,
        "target_exit_rate": float((df["exit_reason"] == "target").mean()) if len(df) else np.nan,
        "stop_exit_rate": float((df["exit_reason"] == "stop").mean()) if len(df) else np.nan,
        "time_exit_rate": float((df["exit_reason"] == "time_exit").mean()) if len(df) else np.nan,
        "avg_bars_held": float(df["bars_held"].mean()) if len(df) and "bars_held" in df else np.nan,
        "mean_p_touch": float(df["p_touch"].mean()) if len(df) else np.nan,
        "mean_p_direction": float(df["p_direction_to_pool"].mean()) if len(df) else np.nan,
        "mean_q": float(df["pool_quality"].mean()) if len(df) else np.nan,
    }
    candidate, diagnostic_only, reason = _candidate_decision(row)
    row["candidate"] = bool(candidate)
    row["diagnostic_only"] = bool(diagnostic_only)
    row["reject_reason"] = reason
    return row


def _value_label(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    return str(value)


def build_slices(df: pd.DataFrame, *, dsr_trials: int) -> Tuple[pd.DataFrame, Dict[str, pd.Series], List[str]]:
    rows: List[Dict] = []
    masks: Dict[str, pd.Series] = {}
    unavailable: List[str] = []

    single_cols, missing_singles = _slice_columns(df)
    unavailable.extend(missing_singles)
    for family, col in single_cols:
        for value, sdf in df.groupby(col, dropna=False, observed=False):
            label = _value_label(value)
            mask = df[col].eq(value)
            row = _metric_row(sdf, family=family, value=label, filter_expr=f"{col} == {label!r}",
                              dsr_trials=dsr_trials)
            rows.append(row)
            masks[row["slice_id"]] = mask

    cross_cols, missing_crosses = _cross_columns(df)
    unavailable.extend(missing_crosses)
    for family, cols in cross_cols:
        cross_value = df[list(cols)].astype("string").fillna("unknown").agg(" / ".join, axis=1)
        tmp = df.assign(_cross_value=cross_value)
        for value, sdf in tmp.groupby("_cross_value", dropna=False, observed=False):
            label = _value_label(value)
            mask = cross_value.eq(value)
            filter_expr = " & ".join(f"{col} == {part!r}" for col, part in zip(cols, label.split(" / ")))
            row = _metric_row(sdf.drop(columns=["_cross_value"]), family=family, value=label,
                              filter_expr=filter_expr, dsr_trials=dsr_trials)
            rows.append(row)
            masks[row["slice_id"]] = mask

    out = pd.DataFrame(rows)
    if out.empty:
        return out, masks, unavailable
    out = out.sort_values(
        ["candidate", "diagnostic_only", "mean_r", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return out, masks, sorted(set(unavailable))


def _daily_r(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return df.groupby("entry_date")["net_r"].sum()


def build_correlation(
    df: pd.DataFrame,
    slices: pd.DataFrame,
    masks: Dict[str, pd.Series],
    *,
    max_candidates: int,
) -> pd.DataFrame:
    candidates = slices[slices["candidate"]].sort_values(["mean_r", "trades"], ascending=False)
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.head(max_candidates)
    daily: Dict[str, pd.Series] = {}
    for row in candidates.to_dict(orient="records"):
        sid = row["slice_id"]
        mask = masks.get(sid)
        if mask is None:
            continue
        daily[sid] = _daily_r(df.loc[mask])
    if not daily:
        return pd.DataFrame()
    all_dates = sorted(set().union(*(series.index for series in daily.values())))
    matrix = pd.DataFrame({
        sid: series.reindex(all_dates, fill_value=0.0)
        for sid, series in daily.items()
    })
    corr = matrix.corr().reset_index().rename(columns={"index": "candidate"})
    return corr


def _top_pairs(corr: pd.DataFrame, limit: int = 10) -> List[Dict]:
    if corr.empty or "candidate" not in corr.columns:
        return []
    long = corr.melt(id_vars="candidate", var_name="other", value_name="corr")
    long = long[long["candidate"] < long["other"]].copy()
    long = long[np.isfinite(long["corr"])]
    long = long.sort_values("corr", ascending=True).head(limit)
    rows = []
    for row in long.to_dict(orient="records"):
        rows.append({
            "candidate_a": row["candidate"],
            "candidate_b": row["other"],
            "corr": fnum(row["corr"], 3),
            "diversifying": "yes" if row["corr"] <= 0.50 else "no",
        })
    return rows


def _report_rows(df: pd.DataFrame, limit: int = 20) -> List[Dict]:
    rows = []
    for row in df.head(limit).to_dict(orient="records"):
        rows.append({
            "family": row["family"],
            "slice": row["slice_value"],
            "n": int(row["trades"]),
            "win": pct(row["win_rate"], 1),
            "mean_R": fnum(row["mean_r"], 3),
            "PF": _fmt_pf(row["profit_factor"]),
            "cost1.5x": fnum(row["mean_r_cost_1_50x"], 3),
            "1st/2nd": f"{fnum(row['first_half_mean_r'], 3)}/{fnum(row['second_half_mean_r'], 3)}",
            "symbols": int(row["unique_symbols"]),
            "sectors": int(row["unique_sectors"]),
            "top": f"{row['top_symbol']}:{int(row['top_symbol_trades'])}",
            "reject": row["reject_reason"],
        })
    return rows


def _mini_sweep_rows(candidates: pd.DataFrame, limit: int = 5) -> List[Dict]:
    rows = []
    for row in candidates.sort_values(["mean_r", "trades"], ascending=False).head(limit).to_dict(orient="records"):
        rows.append({
            "candidate": row["slice_id"],
            "filter": row["filter_expr"],
            "why": f"n={int(row['trades'])}, R={row['mean_r']:+.3f}, PF={row['profit_factor']:.2f}",
            "mini_sweep": "T [0.70,0.75,0.80], D [0.60,0.65,0.70], target [0.6,0.8,1.0], stop [1.5,2.0,2.5], hold [24,36,60]",
        })
    return rows


def build_report(
    slices: pd.DataFrame,
    candidates: pd.DataFrame,
    corr: pd.DataFrame,
    unavailable: Sequence[str],
    *,
    source_trades: Path,
    source_candidates: Path,
) -> str:
    n_trials = int(len(slices))
    if candidates.empty:
        decision = "NO_ADDITIONAL_CANDIDATE_POCKETS"
        reason = "No slice passed the pre-registered candidate filters."
    elif len(candidates) >= 3:
        decision = "MULTI_POCKET_CANDIDATES_FOUND"
        reason = f"{len(candidates)} candidate slices passed the anti-overfit filters."
    else:
        decision = "LIMITED_CANDIDATE_POCKETS_FOUND"
        reason = f"{len(candidates)} candidate slice(s) passed; still needs mini-sweep/CPCV/null validation."

    diagnostic = slices[slices["diagnostic_only"] & ~slices["candidate"]].sort_values(
        ["mean_r", "trades"], ascending=False,
    )
    rejected = slices[~slices["candidate"] & ~slices["diagnostic_only"]].sort_values(
        ["mean_r", "trades"], ascending=False,
    )
    lines = [
        "# Phase 4 Multi-Pocket Slice Analysis",
        "",
        "## Verdict",
        "",
        f"- Decision: `{decision}`",
        f"- {reason}",
        f"- Slice hypotheses tested: `{n_trials}`",
        "- This is discovery only. It does not authorize live trading or paper trading.",
        "- Distance `1-3 ATR` is only partially covered because the existing Track A sweep mostly spans `2-10 ATR`.",
        "- Slice rows can include multiple geometries for the same market event; candidates are hypotheses for mini-sweeps.",
        "",
        "## Sources",
        "",
        f"- Trades: `{source_trades}`",
        f"- Candidate enrichment: `{source_candidates}`",
        "",
        "## Candidate Pockets",
        "",
        markdown_table(
            _report_rows(candidates.sort_values(["mean_r", "trades"], ascending=False), 25),
            ["family", "slice", "n", "win", "mean_R", "PF", "cost1.5x", "1st/2nd", "symbols", "sectors", "top", "reject"],
        ),
        "",
        "## Diagnostic-Only Pockets",
        "",
        markdown_table(
            _report_rows(diagnostic, 20),
            ["family", "slice", "n", "win", "mean_R", "PF", "cost1.5x", "1st/2nd", "symbols", "sectors", "top", "reject"],
        ),
        "",
        "## Rejected High-R Slices",
        "",
        markdown_table(
            _report_rows(rejected, 15),
            ["family", "slice", "n", "win", "mean_R", "PF", "cost1.5x", "1st/2nd", "symbols", "sectors", "top", "reject"],
        ),
        "",
        "## Diversification View",
        "",
        markdown_table(
            _top_pairs(corr, 10),
            ["candidate_a", "candidate_b", "corr", "diversifying"],
        ),
        "",
        "## Recommended Mini-Sweeps",
        "",
        markdown_table(
            _mini_sweep_rows(candidates, 5),
            ["candidate", "filter", "why", "mini_sweep"],
        ),
        "",
        "## Unavailable Or Skipped Dimensions",
        "",
        markdown_table(
            [{"dimension": item} for item in unavailable] if unavailable else [{"dimension": "none"}],
            ["dimension"],
        ),
        "",
        "## Interpretation",
        "",
        "- Treat candidate pockets as pre-registered hypotheses for the next mini-sweep, not as results.",
        "- A useful multi-pocket portfolio needs candidates that survive validation and are not highly correlated.",
        "- If candidates mostly repeat the existing long AUTO/FMCG/PHARMA theme, the project still has one broad pocket, not many independent alphas.",
        "- Full synthetic nulls remain required before final-stage trade modeling or paper trading.",
        "",
        "## Outputs",
        "",
        f"- All slices: `{DEFAULT_SLICES}`",
        f"- Candidate pockets: `{DEFAULT_CANDIDATES_OUT}`",
        f"- Correlation matrix: `{DEFAULT_CORR}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", default=str(DEFAULT_TRADES))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--slices-out", default=str(DEFAULT_SLICES))
    parser.add_argument("--candidates-out", default=str(DEFAULT_CANDIDATES_OUT))
    parser.add_argument("--correlations-out", default=str(DEFAULT_CORR))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--dsr-trials", type=int, default=1000)
    parser.add_argument("--max-corr-candidates", type=int, default=30)
    args = parser.parse_args()

    trades_path = Path(args.trades)
    candidate_path = Path(args.candidates)
    trades = _load_trades(trades_path)
    enriched, enrich_unavailable = _enrich_trades(trades, candidate_path)
    enriched, bucket_unavailable = _bucketize(enriched)
    slices, masks, slice_unavailable = build_slices(enriched, dsr_trials=args.dsr_trials)
    if slices.empty:
        raise ValueError("no slices were generated")
    candidates = slices[slices["candidate"]].copy()
    corr = build_correlation(enriched, slices, masks, max_candidates=args.max_corr_candidates)

    slices_out = Path(args.slices_out)
    candidates_out = Path(args.candidates_out)
    correlations_out = Path(args.correlations_out)
    report_out = Path(args.report_out)
    for path in [slices_out, candidates_out, correlations_out, report_out]:
        path.parent.mkdir(parents=True, exist_ok=True)

    slices.to_csv(slices_out, index=False)
    candidates.sort_values(["mean_r", "trades"], ascending=False).to_csv(candidates_out, index=False)
    corr.to_csv(correlations_out, index=False)
    report = build_report(
        slices,
        candidates,
        corr,
        sorted(set(enrich_unavailable + bucket_unavailable + slice_unavailable)),
        source_trades=trades_path,
        source_candidates=candidate_path,
    )
    report_out.write_text(report, encoding="utf-8")
    print(f"wrote {slices_out}")
    print(f"wrote {candidates_out}")
    print(f"wrote {correlations_out}")
    print(f"wrote {report_out}")


if __name__ == "__main__":
    main()
