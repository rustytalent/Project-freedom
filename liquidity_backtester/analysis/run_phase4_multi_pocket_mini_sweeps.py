"""Phase 4 multi-pocket mini-sweeps from existing Track A trade rows.

This is Option B from the Opus review: reuse the existing v2 pre-touch sweep
trade parquet instead of rerunning the 1-minute simulator. It de-confounds the
5-8 ATR discovery by testing pre-registered pocket families over the already
available gate/geometry grid.

The output is validation staging, not live approval. Any survivor here still
needs CPCV and synthetic random/ATR nulls.
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
DEFAULT_CANDIDATES = Path("output_phase4_track_a_pretouch_sweep/pretouch_candidates.parquet")
DEFAULT_CELLS = Path("reports/phase4_multi_pocket_mini_sweep_cells.csv")
DEFAULT_SURVIVORS = Path("reports/phase4_multi_pocket_mini_sweep_survivors.csv")
DEFAULT_REPORT = Path("reports/phase4_multi_pocket_mini_sweeps.md")

MERGE_KEYS = ["symbol", "fold", "pool_idx", "bar_idx"]

TOUCH_GRID = [0.75, 0.80, 0.85]
DIRECTION_GRID = [0.55, 0.60, 0.65, 0.70]
TARGET_GRID = [0.6, 0.8, 1.0]
STOP_GRID = [1.0, 1.5, 2.0]
HOLD_GRID = [12, 24, 36, 60]


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


def _event_ids(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    cols = ["symbol", "fold", "pool_idx", "bar_idx", "entry_at"]
    return set(df[cols].astype(str).agg("|".join, axis=1))


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing Track A sweep trades: {path}")
    df = pd.read_parquet(path)
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
        "target_fraction",
        "stop_atr_mult",
        "requested_max_hold_bars",
        "distance_atr",
        "p_touch",
        "p_direction_to_pool",
        "pool_quality",
        "net_r",
        "total_cost",
        "risk_inr",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"trade parquet missing required columns: {missing}")
    out = df.copy()
    out["entry_at_dt"] = pd.to_datetime(out["entry_at"], errors="coerce")
    out["entry_date"] = out["entry_at_dt"].dt.date.astype(str)
    out["event_id"] = out[["symbol", "fold", "pool_idx", "bar_idx", "entry_at"]].astype(str).agg("|".join, axis=1)
    return out


def _enrich(trades: pd.DataFrame, candidates_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    out = trades.copy()
    unavailable: List[str] = []
    if not candidates_path.exists():
        out["factor_enriched"] = out["factor"]
        return out, [f"candidate enrichment file missing: {candidates_path}"]
    candidates = pd.read_parquet(candidates_path)
    missing_keys = [col for col in MERGE_KEYS if col not in candidates.columns]
    if missing_keys:
        out["factor_enriched"] = out["factor"]
        return out, [f"candidate enrichment missing merge keys: {missing_keys}"]
    keep = [col for col in ["headline_factor", "vol_ratio"] if col in candidates.columns]
    if not keep:
        out["factor_enriched"] = out["factor"]
        return out, ["headline_factor", "vol_ratio"]
    enrich = candidates[MERGE_KEYS + keep].drop_duplicates(MERGE_KEYS)
    out = out.merge(enrich, on=MERGE_KEYS, how="left")
    out["factor_enriched"] = out.get("headline_factor", pd.Series(index=out.index, dtype=object)).fillna(out["factor"])
    if "vol_ratio" in out.columns and out["vol_ratio"].notna().any():
        out["vol_regime"] = pd.cut(
            out["vol_ratio"],
            bins=[-np.inf, 0.80, 1.20, np.inf],
            labels=["low_vol", "normal_vol", "high_vol"],
        ).astype("string").fillna("unknown")
    else:
        out["vol_regime"] = "unknown"
        unavailable.append("vol_ratio/vol_regime")
    return out, unavailable


def _pocket_masks(df: pd.DataFrame) -> Dict[str, Tuple[str, pd.Series]]:
    long_5_8 = (
        (df["direction"] == "UP")
        & (df["distance_atr"] >= 5.0)
        & (df["distance_atr"] < 8.0)
    )
    return {
        "base_5_8_long": (
            "Long-only, distance 5-8 ATR",
            long_5_8,
        ),
        "morning_5_8_long": (
            "Long-only, morning, distance 5-8 ATR",
            long_5_8 & (df["time_bucket"] == "morning"),
        ),
        "midday_5_8_long": (
            "Long-only, midday, distance 5-8 ATR",
            long_5_8 & (df["time_bucket"] == "midday"),
        ),
        "eqhl_5_8_long": (
            "Long-only, EQHL, distance 5-8 ATR",
            long_5_8 & (df["factor_enriched"] == "EQHL"),
        ),
        "normal_vol_5_8_long": (
            "Long-only, normal volatility, distance 5-8 ATR",
            long_5_8 & (df["vol_regime"] == "normal_vol"),
        ),
    }


def _cost_stress_r(df: pd.DataFrame, multiplier_add: float) -> pd.Series:
    risk = pd.to_numeric(df["risk_inr"], errors="coerce").replace(0, np.nan)
    extra_cost_r = (pd.to_numeric(df["total_cost"], errors="coerce") * multiplier_add) / risk
    return pd.to_numeric(df["net_r"], errors="coerce") - extra_cost_r.fillna(0.0)


def _sector_stats(df: pd.DataFrame) -> Tuple[str, str, int]:
    if df.empty:
        return "", "", 0
    parts_n: List[str] = []
    parts_r: List[str] = []
    positive = 0
    for sector, sdf in df.groupby("sector"):
        r = pd.to_numeric(sdf["net_r"], errors="coerce")
        mean_r = float(r.mean())
        parts_n.append(f"{sector}:{len(sdf)}")
        parts_r.append(f"{sector}:{mean_r:+.3f}")
        if len(sdf) >= 10 and mean_r > 0.0:
            positive += 1
    return ";".join(parts_n), ";".join(parts_r), positive


def _temporal_halves(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, float]:
    if df.empty or df["entry_at_dt"].isna().all():
        empty = pd.Series(dtype=float)
        return empty, empty, float("nan")
    ts = df["entry_at_dt"]
    midpoint = ts.min() + (ts.max() - ts.min()) / 2
    first = pd.to_numeric(df.loc[ts < midpoint, "net_r"], errors="coerce")
    second = pd.to_numeric(df.loc[ts >= midpoint, "net_r"], errors="coerce")
    if not len(first) or not len(second):
        return first, second, float("nan")
    mean_r = float(pd.to_numeric(df["net_r"], errors="coerce").mean())
    gap = abs(float(first.mean()) - float(second.mean()))
    score = max(0.0, 1.0 - gap / (abs(mean_r) + 1e-9))
    return first, second, float(score)


def _mean_ci95_lower(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.mean() - 1.96 * r.std(ddof=1) / math.sqrt(len(r)))


def _summarize_cell(
    df: pd.DataFrame,
    *,
    pocket_family: str,
    pocket_description: str,
    min_p_touch: float,
    min_p_direction: float,
    target_fraction: float,
    stop_atr_mult: float,
    max_hold_bars: int,
    dsr_trials: int,
    bonferroni_threshold: float,
    min_trades: int,
) -> Tuple[Dict, set[str]]:
    r = pd.to_numeric(df["net_r"], errors="coerce").dropna()
    first, second, temporal_score = _temporal_halves(df)
    sectors_n, sectors_r, sectors_positive = _sector_stats(df)
    cost_125 = _cost_stress_r(df, 0.25) if len(df) else pd.Series(dtype=float)
    cost_150 = _cost_stress_r(df, 0.50) if len(df) else pd.Series(dtype=float)
    event_ids = _event_ids(df)
    dsr = float(_deflated_sharpe(r, dsr_trials)) if len(r) else np.nan
    row: Dict = {
        "cell_id": (
            f"{pocket_family}|T{min_p_touch:.2f}|D{min_p_direction:.2f}|"
            f"tf{target_fraction:.1f}|sl{stop_atr_mult:.1f}|h{max_hold_bars}"
        ),
        "pocket_family": pocket_family,
        "pocket_description": pocket_description,
        "min_p_touch": min_p_touch,
        "min_p_direction": min_p_direction,
        "distance_low": 5.0,
        "distance_high": 8.0,
        "target_fraction": target_fraction,
        "stop_atr_mult": stop_atr_mult,
        "max_hold_bars": max_hold_bars,
        "n_trades": int(len(r)),
        "n_events": int(len(event_ids)),
        "n_long": int((df["direction"] == "UP").sum()) if len(df) else 0,
        "n_short": int((df["direction"] == "DOWN").sum()) if len(df) else 0,
        "mean_r": float(r.mean()) if len(r) else np.nan,
        "median_r": float(r.median()) if len(r) else np.nan,
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "profit_factor": float(_profit_factor(r)) if len(r) else np.nan,
        "max_drawdown_r": float(_max_drawdown(r)) if len(r) else np.nan,
        "trade_sharpe": float(_trade_sharpe(r)) if len(r) else np.nan,
        "dsr": dsr,
        "dsr_pass": bool(np.isfinite(dsr) and dsr >= 0.95),
        "bonferroni_dsr_pass": bool(np.isfinite(dsr) and dsr >= bonferroni_threshold),
        "mean_r_cost_1_25x": float(cost_125.mean()) if len(cost_125) else np.nan,
        "mean_r_cost_1_50x": float(cost_150.mean()) if len(cost_150) else np.nan,
        "cost_robust": bool(len(cost_150) and float(cost_150.mean()) > 0.0),
        "first_half_n": int(len(first)),
        "second_half_n": int(len(second)),
        "first_half_mean_r": float(first.mean()) if len(first) else np.nan,
        "second_half_mean_r": float(second.mean()) if len(second) else np.nan,
        "temporal_stability_score": temporal_score,
        "temporal_stable": bool(np.isfinite(temporal_score) and temporal_score > 0.70),
        "unique_symbols": int(df["symbol"].nunique()) if len(df) else 0,
        "unique_sectors": int(df["sector"].nunique()) if len(df) else 0,
        "sectors_with_positive_r": sectors_positive,
        "per_sector_n": sectors_n,
        "per_sector_mean_r": sectors_r,
        "mean_r_ci95_lower": float(_mean_ci95_lower(r)) if len(r) else np.nan,
        "target_exit_rate": float((df["exit_reason"] == "target").mean()) if len(df) else np.nan,
        "stop_exit_rate": float((df["exit_reason"] == "stop").mean()) if len(df) else np.nan,
        "time_exit_rate": float((df["exit_reason"] == "time_exit").mean()) if len(df) else np.nan,
        "avg_bars_held": float(df["bars_held"].mean()) if len(df) and "bars_held" in df else np.nan,
        "mean_p_touch": float(df["p_touch"].mean()) if len(df) else np.nan,
        "mean_p_direction": float(df["p_direction_to_pool"].mean()) if len(df) else np.nan,
        "mean_q": float(df["pool_quality"].mean()) if len(df) else np.nan,
    }
    reasons: List[str] = []
    if row["n_trades"] < min_trades:
        reasons.append(f"n<{min_trades}")
    if not np.isfinite(row["mean_r"]) or row["mean_r"] <= 0.10:
        reasons.append("mean_R<=0.10")
    if not np.isfinite(row["profit_factor"]) or row["profit_factor"] <= 1.40:
        reasons.append("PF<=1.40")
    if not row["temporal_stable"]:
        reasons.append("temporal_score<=0.70")
    if row["sectors_with_positive_r"] < 3:
        reasons.append("positive_sectors<3")
    if not row["cost_robust"]:
        reasons.append("cost1.5x<=0")
    if not np.isfinite(row["mean_r_ci95_lower"]) or row["mean_r_ci95_lower"] <= 0.0:
        reasons.append("mean_R_ci95_lower<=0")

    economic_survivor = not reasons
    strict_survivor = bool(economic_survivor and row["dsr_pass"] and row["bonferroni_dsr_pass"])
    row["economic_survivor"] = bool(economic_survivor)
    row["strict_survivor"] = strict_survivor
    row["reject_reason"] = "economic_survivor" if economic_survivor else "; ".join(reasons)
    return row, event_ids


def run_mini_sweeps(
    trades: pd.DataFrame,
    *,
    dsr_trials: int,
    min_trades: int,
) -> Tuple[pd.DataFrame, Dict[str, set[str]]]:
    rows: List[Dict] = []
    event_sets: Dict[str, set[str]] = {}
    pocket_defs = _pocket_masks(trades)
    bonferroni_threshold = 1.0 - 0.05 / max(len(pocket_defs), 1)
    geometries = list(itertools.product(TARGET_GRID, STOP_GRID, HOLD_GRID))
    for pocket_family, (description, base_mask) in pocket_defs.items():
        base = trades.loc[base_mask].copy()
        if base.empty:
            continue
        for min_p_touch, min_p_direction in itertools.product(TOUCH_GRID, DIRECTION_GRID):
            gate = base[
                (base["p_touch"] >= min_p_touch)
                & (base["p_direction_to_pool"] >= min_p_direction)
            ]
            if gate.empty:
                continue
            for target_fraction, stop_atr_mult, max_hold_bars in geometries:
                cell = gate[
                    (gate["target_fraction"] == target_fraction)
                    & (gate["stop_atr_mult"] == stop_atr_mult)
                    & (gate["requested_max_hold_bars"] == max_hold_bars)
                ].copy()
                if cell.empty:
                    continue
                row, events = _summarize_cell(
                    cell.sort_values("entry_at_dt"),
                    pocket_family=pocket_family,
                    pocket_description=description,
                    min_p_touch=min_p_touch,
                    min_p_direction=min_p_direction,
                    target_fraction=target_fraction,
                    stop_atr_mult=stop_atr_mult,
                    max_hold_bars=max_hold_bars,
                    dsr_trials=dsr_trials,
                    bonferroni_threshold=bonferroni_threshold,
                    min_trades=min_trades,
                )
                rows.append(row)
                event_sets[row["cell_id"]] = events
    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells, event_sets
    cells = cells.sort_values(
        ["strict_survivor", "economic_survivor", "mean_r", "n_trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return cells, event_sets


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    union = len(a | b)
    if union == 0:
        return float("nan")
    return float(len(a & b) / union)


def attach_overlap(cells: pd.DataFrame, event_sets: Dict[str, set[str]]) -> pd.DataFrame:
    out = cells.copy()
    out["max_overlap_other"] = np.nan
    out["max_overlap_cell"] = ""
    out["max_overlap_dedupe"] = np.nan
    out["max_overlap_dedupe_cell"] = ""
    out["dedupe_survivor"] = False
    out["redundant_with"] = ""
    survivor_mask = out["economic_survivor"].astype(bool)
    survivor_ids = out.loc[survivor_mask, "cell_id"].tolist()
    for idx, row in out.iterrows():
        sid = row["cell_id"]
        overlaps = []
        for other in survivor_ids:
            if other == sid:
                continue
            overlaps.append((other, _jaccard(event_sets.get(sid, set()), event_sets.get(other, set()))))
        overlaps = [(other, val) for other, val in overlaps if np.isfinite(val)]
        if overlaps:
            other, val = max(overlaps, key=lambda item: item[1])
            out.at[idx, "max_overlap_cell"] = other
            out.at[idx, "max_overlap_other"] = val

    selected: List[str] = []
    selected_by_family: Dict[str, str] = {}
    sorted_survivors = out[survivor_mask].sort_values(
        ["strict_survivor", "mean_r_cost_1_50x", "mean_r", "n_trades"],
        ascending=[False, False, False, False],
    )
    for idx, row in sorted_survivors.iterrows():
        sid = row["cell_id"]
        redundant_with = ""
        family = str(row["pocket_family"])
        if family in selected_by_family:
            redundant_with = selected_by_family[family]
        for chosen in selected:
            if redundant_with:
                break
            overlap = _jaccard(event_sets.get(sid, set()), event_sets.get(chosen, set()))
            if np.isfinite(overlap) and overlap > 0.70:
                redundant_with = chosen
                break
        if redundant_with:
            out.at[idx, "redundant_with"] = redundant_with
        else:
            selected.append(sid)
            selected_by_family[family] = sid
            out.at[idx, "dedupe_survivor"] = True

    dedupe_ids = out.loc[out["dedupe_survivor"].astype(bool), "cell_id"].tolist()
    for idx, row in out.iterrows():
        sid = row["cell_id"]
        overlaps = []
        for other in dedupe_ids:
            if other == sid:
                continue
            overlaps.append((other, _jaccard(event_sets.get(sid, set()), event_sets.get(other, set()))))
        overlaps = [(other, val) for other, val in overlaps if np.isfinite(val)]
        if overlaps:
            other, val = max(overlaps, key=lambda item: item[1])
            out.at[idx, "max_overlap_dedupe_cell"] = other
            out.at[idx, "max_overlap_dedupe"] = val
    return out


def family_verdicts(cells: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for family, df in cells.groupby("pocket_family"):
        econ = df[df["economic_survivor"].astype(bool)]
        strict = df[df["strict_survivor"].astype(bool)]
        dedup = df[df["dedupe_survivor"].astype(bool)]
        if not strict.empty and not dedup.empty:
            verdict = "ROBUST"
        elif not econ.empty and not dedup.empty:
            verdict = "FRAGILE_NEEDS_CPCV"
        elif not econ.empty:
            verdict = "REDUNDANT"
        else:
            verdict = "DEAD"
        best = (
            df.sort_values(["economic_survivor", "mean_r", "n_trades"], ascending=[False, False, False]).iloc[0]
            if len(df)
            else None
        )
        rows.append({
            "pocket_family": family,
            "verdict": verdict,
            "cells": int(len(df)),
            "economic_survivors": int(len(econ)),
            "strict_survivors": int(len(strict)),
            "dedupe_survivors": int(len(dedup)),
            "best_cell": "" if best is None else best["cell_id"],
            "best_n": 0 if best is None else int(best["n_trades"]),
            "best_mean_r": np.nan if best is None else float(best["mean_r"]),
            "best_pf": np.nan if best is None else float(best["profit_factor"]),
        })
    return pd.DataFrame(rows).sort_values(
        ["dedupe_survivors", "economic_survivors", "best_mean_r"],
        ascending=[False, False, False],
    )


def _report_rows(df: pd.DataFrame, limit: int = 20) -> List[Dict]:
    rows = []
    for row in df.head(limit).to_dict(orient="records"):
        rows.append({
            "family": row["pocket_family"],
            "T": fnum(row["min_p_touch"], 2),
            "D": fnum(row["min_p_direction"], 2),
            "tf": fnum(row["target_fraction"], 1),
            "sl": fnum(row["stop_atr_mult"], 1),
            "h": int(row["max_hold_bars"]),
            "n": int(row["n_trades"]),
            "win": pct(row["win_rate"], 1),
            "mean_R": fnum(row["mean_r"], 3),
            "PF": _fmt_pf(row["profit_factor"]),
            "cost1.5x": fnum(row["mean_r_cost_1_50x"], 3),
            "1st/2nd": f"{fnum(row['first_half_mean_r'], 3)}/{fnum(row['second_half_mean_r'], 3)}",
            "temp": fnum(row["temporal_stability_score"], 2),
            "pos_sec": int(row["sectors_with_positive_r"]),
            "ci_low": fnum(row["mean_r_ci95_lower"], 3),
            "DSR": pct(row["dsr"], 1),
            "sel_ov": fnum(row["max_overlap_dedupe"], 3),
            "dedupe": str(bool(row["dedupe_survivor"])),
        })
    return rows


def _family_rows(verdicts: pd.DataFrame) -> List[Dict]:
    rows = []
    for row in verdicts.to_dict(orient="records"):
        rows.append({
            "family": row["pocket_family"],
            "verdict": row["verdict"],
            "cells": int(row["cells"]),
            "econ": int(row["economic_survivors"]),
            "strict": int(row["strict_survivors"]),
            "dedupe": int(row["dedupe_survivors"]),
            "best_n": int(row["best_n"]),
            "best_R": fnum(row["best_mean_r"], 3),
            "best_PF": _fmt_pf(row["best_pf"]),
        })
    return rows


def _overlap_rows(cells: pd.DataFrame, event_sets: Dict[str, set[str]], limit: int = 12) -> List[Dict]:
    survivors = cells[cells["economic_survivor"].astype(bool)].head(40)
    pairs: List[Dict] = []
    ids = survivors["cell_id"].tolist()
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            val = _jaccard(event_sets.get(left, set()), event_sets.get(right, set()))
            if not np.isfinite(val):
                continue
            pairs.append({
                "cell_a": left,
                "cell_b": right,
                "overlap": val,
                "independent": "yes" if val < 0.30 else "partial" if val <= 0.70 else "no",
            })
    pairs = sorted(pairs, key=lambda row: row["overlap"])
    return [
        {
            "cell_a": row["cell_a"],
            "cell_b": row["cell_b"],
            "overlap": fnum(row["overlap"], 3),
            "independent": row["independent"],
        }
        for row in pairs[:limit]
    ]


def build_report(
    cells: pd.DataFrame,
    survivors: pd.DataFrame,
    event_sets: Dict[str, set[str]],
    unavailable: Sequence[str],
    *,
    dsr_trials: int,
    min_trades: int,
) -> str:
    verdicts = family_verdicts(cells)
    economic = cells[cells["economic_survivor"].astype(bool)].sort_values(
        ["dedupe_survivor", "mean_r_cost_1_50x", "mean_r"],
        ascending=[False, False, False],
    )
    strict = economic[economic["strict_survivor"].astype(bool)]
    dedupe = economic[economic["dedupe_survivor"].astype(bool)]
    if not strict.empty:
        decision = "DSR_CONFIRMED_SURVIVORS_FOUND"
        reason = f"{len(strict)} cells pass economic criteria plus DSR/Bonferroni."
    elif not dedupe.empty:
        decision = "ECONOMIC_SURVIVORS_NEED_CPCV"
        reason = f"{len(dedupe)} deduplicated economic survivor cells found, but none clear DSR."
    elif not economic.empty:
        decision = "ONLY_REDUNDANT_ECONOMIC_SURVIVORS"
        reason = "Economic cells exist but are redundant with stronger cells."
    else:
        decision = "NO_MINI_SWEEP_SURVIVORS"
        reason = "No cell passed the seven economic survivor criteria."

    lines = [
        "# Phase 4 Multi-Pocket Mini-Sweeps",
        "",
        "## Verdict",
        "",
        f"- Decision: `{decision}`",
        f"- {reason}",
        f"- Cells tested: `{len(cells)}`",
        f"- DSR trials used: `{dsr_trials}`",
        f"- Minimum trades per cell: `{min_trades}`",
        "- This is Option B: reuse existing v2 sweep trades; no new 1-minute simulation was run.",
        "- Because the source sweep admitted candidates around `T>=0.75` and `D>=0.55`, this run cannot test looser entry gates such as `T=0.70` from scratch.",
        "- Survivors here still need CPCV and synthetic random/ATR nulls before paper trading.",
        "",
        "## Pocket Family Verdicts",
        "",
        markdown_table(
            _family_rows(verdicts),
            ["family", "verdict", "cells", "econ", "strict", "dedupe", "best_n", "best_R", "best_PF"],
        ),
        "",
        "## Deduplicated Economic Survivors",
        "",
        markdown_table(
            _report_rows(dedupe, 25),
            ["family", "T", "D", "tf", "sl", "h", "n", "win", "mean_R", "PF", "cost1.5x", "1st/2nd", "temp", "pos_sec", "ci_low", "DSR", "sel_ov", "dedupe"],
        ),
        "",
        "## Top Economic Survivors Before Deduplication",
        "",
        markdown_table(
            _report_rows(economic.sort_values(["mean_r", "n_trades"], ascending=False), 25),
            ["family", "T", "D", "tf", "sl", "h", "n", "win", "mean_R", "PF", "cost1.5x", "1st/2nd", "temp", "pos_sec", "ci_low", "DSR", "sel_ov", "dedupe"],
        ),
        "",
        "## Top Cells By Mean R",
        "",
        markdown_table(
            _report_rows(cells.sort_values(["mean_r", "n_trades"], ascending=False), 20),
            ["family", "T", "D", "tf", "sl", "h", "n", "win", "mean_R", "PF", "cost1.5x", "1st/2nd", "temp", "pos_sec", "ci_low", "DSR", "sel_ov", "dedupe"],
        ),
        "",
        "## Cross-Pocket Overlap",
        "",
        markdown_table(
            _overlap_rows(cells, event_sets, 12),
            ["cell_a", "cell_b", "overlap", "independent"],
        ),
        "",
        "## Survivor Criteria",
        "",
        "- `n_trades >= min_trades`",
        "- `mean_R > +0.10`",
        "- `PF > 1.40`",
        "- `temporal_stability_score > 0.70`",
        "- at least `3` sectors with positive mean R",
        "- mean R remains positive under `1.5x` costs",
        "- approximate 95% lower confidence bound of mean R is above zero",
        "- strict survivor additionally requires `DSR >= 95%` and Bonferroni-adjusted DSR pass.",
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
        "- If only one deduplicated survivor remains, the likely result is a refined single Track A pocket, not a multi-pocket portfolio.",
        "- If multiple deduplicated survivors remain with low overlap, they become candidates for CPCV and synthetic nulls.",
        "- DSR is expected to be hard to clear at this stage because this project has already tested many hypotheses. Treat economic survivors as candidates, not validated edges.",
        "",
        "## Outputs",
        "",
        f"- Cells CSV: `{DEFAULT_CELLS}`",
        f"- Survivors CSV: `{DEFAULT_SURVIVORS}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", default=str(DEFAULT_TRADES))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--cells-out", default=str(DEFAULT_CELLS))
    parser.add_argument("--survivors-out", default=str(DEFAULT_SURVIVORS))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--dsr-trials", type=int, default=2000)
    parser.add_argument("--min-trades", type=int, default=100)
    args = parser.parse_args()

    trades_path = Path(args.trades)
    candidates_path = Path(args.candidates)
    trades = _load_trades(trades_path)
    trades, unavailable = _enrich(trades, candidates_path)
    cells, event_sets = run_mini_sweeps(trades, dsr_trials=args.dsr_trials, min_trades=args.min_trades)
    if cells.empty:
        raise ValueError("no mini-sweep cells generated")
    cells = attach_overlap(cells, event_sets)
    survivors = cells[cells["economic_survivor"].astype(bool)].sort_values(
        ["dedupe_survivor", "strict_survivor", "mean_r_cost_1_50x", "mean_r"],
        ascending=[False, False, False, False],
    )

    cells_out = Path(args.cells_out)
    survivors_out = Path(args.survivors_out)
    report_out = Path(args.report_out)
    for path in [cells_out, survivors_out, report_out]:
        path.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(cells_out, index=False)
    survivors.to_csv(survivors_out, index=False)
    report_out.write_text(
        build_report(
            cells,
            survivors,
            event_sets,
            unavailable,
            dsr_trials=args.dsr_trials,
            min_trades=args.min_trades,
        ),
        encoding="utf-8",
    )
    print(f"wrote {cells_out}")
    print(f"wrote {survivors_out}")
    print(f"wrote {report_out}")


if __name__ == "__main__":
    main()
