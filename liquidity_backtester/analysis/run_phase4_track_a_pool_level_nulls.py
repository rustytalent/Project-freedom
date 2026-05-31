"""Phase 4 Track A pool-level synthetic null replay.

This is the heavier validation step after
``run_phase4_track_a_synthetic_nulls.py``.  The preflight used existing trade
rows and shuffled labels.  This script mutates the pool levels themselves and
replays them through the same Track A v2 execution path.

Nulls implemented:

* random_pool_matched: random synthetic pool levels with matched distance
  distribution and matched side distribution.
* atr_offset_k: synthetic pools at spot +/- k*ATR for k in {1,2,3,5}.
* sector_neutral_random: shuffle sector gate labels, then randomize pool levels.
* shuffled_direction_replay: shuffle the direction/side gate, then replay the
  selected real pools.  This is run only for the direction-hard pocket.

Scope note: the script uses the existing candidate artifact, so model scoring is
not recomputed for synthetic pools.  It answers whether the discovered candidate
events beat fake pool locations under the same execution mechanics.  It is more
honest than label-shuffling, but still not a retrained synthetic-feature model.
"""
from __future__ import annotations

import argparse
import itertools
import math
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from liqpool.indicators import atr  # noqa: E402
try:  # noqa: E402
    from q_audit_common import fnum, markdown_table, pct
except ModuleNotFoundError:  # pragma: no cover - test import path
    from analysis.q_audit_common import fnum, markdown_table, pct

try:  # noqa: E402
    from run_phase4_track_a_pretouch_sweep import (
        Geometry,
        _load_symbol_1m,
        _simulate_one,
    )
except ModuleNotFoundError:  # pragma: no cover - test import path
    from analysis.run_phase4_track_a_pretouch_sweep import (
        Geometry,
        _load_symbol_1m,
        _simulate_one,
    )


DEFAULT_MODEL_REPORT = Path("output_models/core25_latest/multi_asset_report.pkl")
DEFAULT_CANDIDATES = Path("output_phase4_track_a_pretouch_sweep/pretouch_candidates.parquet")
DEFAULT_OUT_DIR = Path("output_audit/track_a_pool_level_nulls")
DEFAULT_REPORT = Path("reports/phase4_track_a_pool_level_nulls.md")

FAVORABLE_SECTORS = ("AUTO", "FMCG", "PHARMA")
TIME_GATE_COLUMNS = ("st_session_morning", "st_session_midday")
ATR_OFFSETS = (1.0, 2.0, 3.0, 5.0)

_WORKER_REPORT_PATH: Optional[str] = None
_WORKER_REPORT = None


def _load_report(path: Path):
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def _worker_report(path: str):
    global _WORKER_REPORT_PATH, _WORKER_REPORT
    if _WORKER_REPORT is None or _WORKER_REPORT_PATH != path:
        _WORKER_REPORT = _load_report(Path(path))
        _WORKER_REPORT_PATH = path
    return _WORKER_REPORT


def parse_csv_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing Track A candidates: {path}")
    df = pd.read_parquet(path).copy()
    required = {
        "symbol",
        "sector",
        "fold",
        "pool_idx",
        "bar_idx",
        "ts",
        "pool_side",
        "price_low",
        "price_high",
        "distance_atr",
        "pool_width_atr",
        "p_touch",
        "p_up",
        "p_direction_to_pool",
        "pool_quality",
        "touch_label",
        "direction_target_to_pool",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"candidate parquet missing required columns: {missing}")
    return df


def enrich_execution_refs(candidates: pd.DataFrame, report) -> pd.DataFrame:
    """Add entry reference and ATR at decision for synthetic price generation."""
    frames: List[pd.DataFrame] = []
    for symbol, sdf in candidates.groupby("symbol", sort=True):
        asset = getattr(report, "assets", {}).get(symbol)
        if asset is None:
            continue
        base = getattr(asset, "base_df", None)
        if base is None or base.empty:
            continue
        cfg = getattr(asset, "final_cfg", None)
        atr_period = getattr(getattr(cfg, "detect", None), "atr_period", 14)
        atr_values = atr(base, atr_period).bfill()
        idx = pd.DatetimeIndex(base.index)
        out = sdf.copy()
        decision_idx = pd.to_numeric(out["bar_idx"], errors="coerce").astype("Int64")
        entry_idx = decision_idx + 1
        valid = entry_idx.notna() & (entry_idx >= 0) & (entry_idx < len(base))
        out = out.loc[valid].copy()
        entry_idx = entry_idx.loc[valid].astype(int)
        decision_idx = decision_idx.loc[valid].astype(int)
        out["entry_ref_for_null"] = base["open"].iloc[entry_idx.to_numpy()].to_numpy(dtype=float)
        out["atr_at_decision_for_null"] = atr_values.iloc[decision_idx.to_numpy()].to_numpy(dtype=float)
        width_abs = pd.to_numeric(out["price_high"], errors="coerce") - pd.to_numeric(out["price_low"], errors="coerce")
        width_atr = pd.to_numeric(out["pool_width_atr"], errors="coerce")
        fallback = width_abs / out["atr_at_decision_for_null"].replace(0.0, np.nan)
        out["pool_width_atr_for_null"] = width_atr.where(width_atr > 0, fallback).fillna(0.15).clip(lower=0.02)
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=list(candidates.columns) + [
            "entry_ref_for_null",
            "atr_at_decision_for_null",
            "pool_width_atr_for_null",
        ])
    return pd.concat(frames, ignore_index=True)


def _time_gate_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in TIME_GATE_COLUMNS:
        if col in df.columns:
            mask |= pd.to_numeric(df[col], errors="coerce").fillna(0.0) >= 0.5
    return mask


def pocket_candidates(candidates: pd.DataFrame, pocket: str) -> pd.DataFrame:
    mask = _time_gate_mask(candidates)
    mask &= candidates["sector"].isin(FAVORABLE_SECTORS)
    if pocket == "direction_hard":
        mask &= candidates["pool_side"].astype(str).eq("above")
    elif pocket == "direction_soft":
        pass
    else:
        raise ValueError(f"unknown pocket: {pocket}")
    return candidates.loc[mask].copy()


def pocket_base_for_sector_neutral(candidates: pd.DataFrame, pocket: str) -> pd.DataFrame:
    mask = _time_gate_mask(candidates)
    if pocket == "direction_hard":
        mask &= candidates["pool_side"].astype(str).eq("above")
    elif pocket == "direction_soft":
        pass
    else:
        raise ValueError(f"unknown pocket: {pocket}")
    return candidates.loc[mask].copy()


def pocket_base_for_direction_shuffle(candidates: pd.DataFrame) -> pd.DataFrame:
    mask = _time_gate_mask(candidates)
    mask &= candidates["sector"].isin(FAVORABLE_SECTORS)
    return candidates.loc[mask].copy()


def make_geometries(
    target_fractions: Sequence[float],
    stop_atr_mults: Sequence[float],
    max_hold_bars: Sequence[int],
) -> List[Geometry]:
    return [
        Geometry(target_fraction=tf, stop_atr_mult=sl, max_hold_bars=hold)
        for tf, sl, hold in itertools.product(target_fractions, stop_atr_mults, max_hold_bars)
    ]


def _apply_synthetic_prices(df: pd.DataFrame, distances: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    atr_values = pd.to_numeric(out["atr_at_decision_for_null"], errors="coerce").to_numpy(dtype=float)
    entry_values = pd.to_numeric(out["entry_ref_for_null"], errors="coerce").to_numpy(dtype=float)
    widths = pd.to_numeric(out["pool_width_atr_for_null"], errors="coerce").fillna(0.15).clip(lower=0.02).to_numpy(dtype=float)
    dist = np.asarray(distances, dtype=float)
    width_abs = widths * atr_values
    side = out["pool_side"].astype(str).to_numpy()
    above = side == "above"

    price_low = np.empty(len(out), dtype=float)
    price_high = np.empty(len(out), dtype=float)
    price_low[above] = entry_values[above] + dist[above] * atr_values[above]
    price_high[above] = price_low[above] + width_abs[above]
    price_high[~above] = entry_values[~above] - dist[~above] * atr_values[~above]
    price_low[~above] = price_high[~above] - width_abs[~above]

    out["price_low"] = price_low
    out["price_high"] = price_high
    out["mid"] = (price_low + price_high) / 2.0
    out["distance_atr"] = dist
    out["side_above"] = above.astype(float)
    out["headline_factor"] = "SYNTH"
    return out


def make_random_pool_candidates(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    source = pd.to_numeric(df["distance_atr"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(source) == 0:
        source = np.full(len(df), 5.0)
    distances = rng.choice(source, size=len(df), replace=True)
    distances = np.clip(distances, 0.25, 20.0)
    return _apply_synthetic_prices(df, distances)


def make_atr_offset_candidates(df: pd.DataFrame, k: float) -> pd.DataFrame:
    distances = np.full(len(df), float(k), dtype=float)
    return _apply_synthetic_prices(df, distances)


def make_sector_neutral_candidates(
    candidates: pd.DataFrame,
    pocket: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    base = pocket_base_for_sector_neutral(candidates, pocket)
    if base.empty:
        return base
    labels = rng.permutation(base["sector"].astype(str).to_numpy())
    selected = base.loc[np.isin(labels, FAVORABLE_SECTORS)].copy()
    return make_random_pool_candidates(selected, rng)


def make_direction_shuffle_candidates(
    candidates: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    base = pocket_base_for_direction_shuffle(candidates)
    if base.empty:
        return base
    labels = rng.permutation(base["pool_side"].astype(str).to_numpy())
    return base.loc[labels == "above"].copy()


def _profit_factor(r: pd.Series) -> float:
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    if neg == 0.0:
        return math.inf if pos > 0 else 0.0
    return pos / neg


def _ci95(r: pd.Series) -> Tuple[float, float]:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if r.empty:
        return float("nan"), float("nan")
    if len(r) == 1:
        val = float(r.iloc[0])
        return val, val
    mean = float(r.mean())
    se = float(r.std(ddof=1) / math.sqrt(len(r)))
    return mean - 1.96 * se, mean + 1.96 * se


def summarize_trades(trades: pd.DataFrame) -> Dict:
    r = pd.to_numeric(trades.get("net_r", pd.Series(dtype=float)), errors="coerce").dropna()
    ci_lo, ci_hi = _ci95(r)
    if trades.empty:
        cost_150 = pd.Series(dtype=float)
    else:
        risk = pd.to_numeric(trades["risk_inr"], errors="coerce").replace(0.0, np.nan)
        cost_150 = r - (pd.to_numeric(trades["total_cost"], errors="coerce") * 0.50 / risk).fillna(0.0)
    return {
        "n_trades": int(len(r)),
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "mean_r": float(r.mean()) if len(r) else np.nan,
        "median_r": float(r.median()) if len(r) else np.nan,
        "profit_factor": float(_profit_factor(r)) if len(r) else np.nan,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "mean_r_cost_1_50x": float(cost_150.mean()) if len(cost_150) else np.nan,
    }


def simulate_candidate_frame(
    candidates: pd.DataFrame,
    *,
    report,
    raw_1m_dir: Optional[Path],
    geometries: Sequence[Geometry],
    data_timestamps_utc: bool,
    session_exit_time: str,
    use_1m_resolution: bool,
    slippage_model: str,
    base_slippage_bps: float,
    exchange: str,
    quantity: int,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for symbol, sdf in candidates.groupby("symbol", sort=True):
        asset = getattr(report, "assets", {}).get(symbol)
        if asset is None:
            continue
        base = getattr(asset, "base_df", None)
        if base is None or base.empty:
            continue
        cfg = getattr(asset, "final_cfg", None)
        atr_period = getattr(getattr(cfg, "detect", None), "atr_period", 14)
        atr_values = atr(base, atr_period).bfill()
        intrabar_1m = _load_symbol_1m(symbol, raw_1m_dir) if use_1m_resolution else None
        for geometry in geometries:
            rows = []
            for _, row in sdf.iterrows():
                trade = _simulate_one(
                    row,
                    base_df=base,
                    atr_values=atr_values,
                    intrabar_1m=intrabar_1m,
                    geometry=geometry,
                    data_timestamps_utc=data_timestamps_utc,
                    session_exit_time=session_exit_time,
                    use_1m_resolution=use_1m_resolution,
                    slippage_model=slippage_model,
                    base_slippage_bps=base_slippage_bps,
                    exchange=exchange,
                    quantity=quantity,
                )
                if trade is not None:
                    rows.append(trade)
            if rows:
                frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _simulate_and_summarize(command: Dict, candidates: pd.DataFrame, config: Mapping) -> Dict:
    report = _worker_report(str(config["model_report"]))
    raw_dir = Path(str(config["raw_1m_dir"])).expanduser() if config.get("raw_1m_dir") else None
    rng = np.random.default_rng(int(command.get("seed", 0)))
    pocket = str(command["pocket"])
    null_test = str(command["null_test"])

    if null_test == "actual_replay":
        frame = pocket_candidates(candidates, pocket)
    elif null_test.startswith("atr_offset_"):
        frame = make_atr_offset_candidates(
            pocket_candidates(candidates, pocket),
            float(command["atr_k"]),
        )
    elif null_test == "random_pool_matched":
        frame = make_random_pool_candidates(pocket_candidates(candidates, pocket), rng)
    elif null_test == "sector_neutral_random":
        frame = make_sector_neutral_candidates(candidates, pocket, rng)
    elif null_test == "shuffled_direction_replay":
        frame = make_direction_shuffle_candidates(candidates, rng)
    else:
        raise ValueError(f"unknown null command: {null_test}")

    max_candidates = config.get("max_candidates")
    if max_candidates is not None:
        frame = frame.head(int(max_candidates)).copy()

    trades = simulate_candidate_frame(
        frame,
        report=report,
        raw_1m_dir=raw_dir,
        geometries=config["geometries"],
        data_timestamps_utc=bool(config["data_timestamps_utc"]),
        session_exit_time=str(config["session_exit_time"]),
        use_1m_resolution=bool(config["use_1m_resolution"]),
        slippage_model=str(config["slippage_model"]),
        base_slippage_bps=float(config["base_slippage_bps"]),
        exchange=str(config["exchange"]),
        quantity=int(config["quantity"]),
    )
    row = summarize_trades(trades)
    row.update({
        "pocket": pocket,
        "null_test": null_test,
        "trial": int(command["trial"]),
        "candidate_rows": int(len(frame)),
    })
    return row


def _run_batch(payload: Dict) -> List[Dict]:
    candidates = payload["candidates"]
    config = payload["config"]
    return [
        _simulate_and_summarize(command, candidates, config)
        for command in payload["commands"]
    ]


def _split_evenly(items: Sequence[Dict], chunks: int) -> List[List[Dict]]:
    chunks = max(1, min(int(chunks), len(items)))
    out = [[] for _ in range(chunks)]
    for i, item in enumerate(items):
        out[i % chunks].append(item)
    return [chunk for chunk in out if chunk]


def _run_batches(payloads: Sequence[Dict], workers: int) -> List[Dict]:
    if not payloads:
        return []
    workers = max(1, min(int(workers), len(payloads)))
    if workers == 1:
        rows: List[Dict] = []
        for payload in payloads:
            rows.extend(_run_batch(payload))
        return rows
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_run_batch, payloads))
    except (OSError, PermissionError):
        parts = [_run_batch(payload) for payload in payloads]
    rows = []
    for part in parts:
        rows.extend(part)
    return rows


def _config_from_args(args: argparse.Namespace) -> Dict:
    return {
        "model_report": str(args.model_report),
        "raw_1m_dir": str(args.raw_1m_dir) if args.raw_1m_dir else "",
        "geometries": args._geometries,
        "data_timestamps_utc": not bool(args.data_timestamps_local_ist),
        "session_exit_time": args.session_exit_time,
        "use_1m_resolution": bool(args.use_1m_resolution),
        "slippage_model": args.slippage_model,
        "base_slippage_bps": float(args.base_slippage_bps),
        "exchange": args.exchange,
        "quantity": int(args.quantity),
        "max_candidates": args.max_candidates,
    }


def build_commands(args: argparse.Namespace) -> List[Dict]:
    rng = np.random.default_rng(args.seed)
    commands: List[Dict] = []
    pockets = [p.strip() for p in args.pockets.split(",") if p.strip()]
    for pocket in pockets:
        commands.append({
            "pocket": pocket,
            "null_test": "actual_replay",
            "trial": 0,
        })
        for k in ATR_OFFSETS:
            commands.append({
                "pocket": pocket,
                "null_test": f"atr_offset_{k:g}",
                "trial": 0,
                "atr_k": float(k),
            })
        for trial in range(1, args.trials + 1):
            trial_seed = int(rng.integers(0, 2**31 - 1))
            commands.append({
                "pocket": pocket,
                "null_test": "random_pool_matched",
                "trial": trial,
                "seed": trial_seed,
            })
            commands.append({
                "pocket": pocket,
                "null_test": "sector_neutral_random",
                "trial": trial,
                "seed": trial_seed + 17,
            })
            if pocket == "direction_hard":
                commands.append({
                    "pocket": pocket,
                    "null_test": "shuffled_direction_replay",
                    "trial": trial,
                    "seed": trial_seed + 31,
                })
    return commands


def build_batch_payloads(candidates: pd.DataFrame, args: argparse.Namespace) -> List[Dict]:
    commands = build_commands(args)
    chunks = _split_evenly(commands, args.workers)
    config = _config_from_args(args)
    return [
        {
            "candidates": candidates,
            "config": config,
            "commands": chunk,
        }
        for chunk in chunks
    ]


def aggregate_results(rows: pd.DataFrame) -> pd.DataFrame:
    actual = rows[rows["null_test"] == "actual_replay"].copy()
    actual_map = {
        row["pocket"]: row
        for row in actual.to_dict(orient="records")
    }
    out: List[Dict] = []
    for (pocket, null_test), df in rows.groupby(["pocket", "null_test"], sort=True):
        actual_row = actual_map.get(pocket, {})
        actual_mean = float(actual_row.get("mean_r", np.nan))
        means = pd.to_numeric(df["mean_r"], errors="coerce").dropna().to_numpy(dtype=float)
        if null_test == "actual_replay":
            p_value = np.nan
        elif len(means) and np.isfinite(actual_mean):
            p_value = float((np.sum(means >= actual_mean) + 1) / (len(means) + 1))
        else:
            p_value = np.nan
        first = df.iloc[0]
        out.append({
            "pocket": pocket,
            "null_test": null_test,
            "trials": int(len(df)),
            "actual_mean_r": actual_mean,
            "actual_n_trades": int(actual_row.get("n_trades", 0) or 0),
            "actual_ci95_lo": float(actual_row.get("ci95_lo", np.nan)),
            "actual_ci95_hi": float(actual_row.get("ci95_hi", np.nan)),
            "actual_pf": float(actual_row.get("profit_factor", np.nan)),
            "actual_cost_1_50x": float(actual_row.get("mean_r_cost_1_50x", np.nan)),
            "null_mean_r_mean": float(np.nanmean(means)) if len(means) else np.nan,
            "null_mean_r_p50": float(np.nanquantile(means, 0.50)) if len(means) else np.nan,
            "null_mean_r_p95": float(np.nanquantile(means, 0.95)) if len(means) else np.nan,
            "null_median_trades": float(pd.to_numeric(df["n_trades"], errors="coerce").median()),
            "null_median_candidates": float(pd.to_numeric(df["candidate_rows"], errors="coerce").median()),
            "p_value": p_value,
            "example_candidate_rows": int(first.get("candidate_rows", 0) or 0),
        })
    return pd.DataFrame(out).sort_values(["pocket", "null_test"]).reset_index(drop=True)


def decide(summary: pd.DataFrame) -> Tuple[str, str]:
    hard = summary[summary["pocket"] == "direction_hard"]
    if hard.empty:
        return "POOL_NULL_FAIL", "No direction-hard rows were produced."
    core_tests = hard[hard["null_test"].isin(["random_pool_matched", "sector_neutral_random"])]
    core_pass = bool(len(core_tests) == 2 and (core_tests["p_value"] <= 0.05).all())
    atr_tests = hard[hard["null_test"].str.startswith("atr_offset_")]
    atr_pass = bool(not atr_tests.empty and (atr_tests["actual_mean_r"] > atr_tests["null_mean_r_p95"]).all())
    direction = hard[hard["null_test"] == "shuffled_direction_replay"]
    direction_pass = bool(not direction.empty and float(direction["p_value"].iloc[0]) <= 0.05)
    if core_pass and atr_pass and direction_pass:
        return "POOL_LEVEL_PASS", "Direction-hard Track A beats random, ATR-offset, sector-neutral, and direction-shuffle replays."
    if core_pass and atr_pass and not direction_pass:
        return "POOL_LEVEL_CORE_PASS_DIRECTION_WEAK", (
            "Track A beats generated-pool core nulls, but direction-hard gating remains weak."
        )
    if core_pass:
        return "POOL_LEVEL_PARTIAL_PASS", "Track A beats random/sector generated-pool nulls but not all ATR/direction tests."
    return "POOL_LEVEL_FAIL", "Track A did not beat the generated-pool core nulls."


def _fmt_pf(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if math.isinf(float(value)):
        return "inf"
    return fnum(value, 3)


def _summary_rows(df: pd.DataFrame, pocket: str) -> List[Dict]:
    rows = []
    for row in df[df["pocket"] == pocket].to_dict(orient="records"):
        rows.append({
            "null": row["null_test"],
            "trials": int(row["trials"]),
            "actual_R": fnum(row["actual_mean_r"], 3),
            "actual_CI": f"{fnum(row['actual_ci95_lo'], 3)}/{fnum(row['actual_ci95_hi'], 3)}",
            "actual_PF": _fmt_pf(row["actual_pf"]),
            "null_R_med": fnum(row["null_mean_r_p50"], 3),
            "null_R_95": fnum(row["null_mean_r_p95"], 3),
            "p": fnum(row["p_value"], 4),
            "n_med": fnum(row["null_median_trades"], 0),
        })
    return rows


def build_report(summary: pd.DataFrame, args: argparse.Namespace) -> str:
    decision, reason = decide(summary)
    resolution = "1m" if args.use_1m_resolution else "5m"
    lines = [
        "# Phase 4 Track A Pool-Level Synthetic Nulls",
        "",
        "## Verdict",
        "",
        f"- Decision: `{decision}`",
        f"- {reason}",
        f"- Trials per stochastic null: `{args.trials}`",
        f"- Workers requested: `{args.workers}`",
        f"- Replay resolution: `{resolution}`",
        f"- Raw 1m dir: `{args.raw_1m_dir or ''}`",
        "- Scope: generated synthetic pool levels from the existing Track A candidate artifact; model scores are not recomputed for fake pools.",
        "",
        "## Direction-Hard Pocket",
        "",
        markdown_table(
            _summary_rows(summary, "direction_hard"),
            ["null", "trials", "actual_R", "actual_CI", "actual_PF", "null_R_med", "null_R_95", "p", "n_med"],
        ),
        "",
        "## Direction-Soft Pocket",
        "",
        markdown_table(
            _summary_rows(summary, "direction_soft"),
            ["null", "trials", "actual_R", "actual_CI", "actual_PF", "null_R_med", "null_R_95", "p", "n_med"],
        ),
        "",
        "## Interpretation",
        "",
        "- `random_pool_matched` tests whether fake levels at the same distance distribution work as well as detected pools.",
        "- `atr_offset_*` tests whether simple spot +/- k*ATR levels explain the edge.",
        "- `sector_neutral_random` tests whether favorable-sector gating survives when sector labels are randomized before fake-pool replay.",
        "- `shuffled_direction_replay` tests whether the direction-hard UP gate adds edge after replaying randomly selected real-pool sides.",
        "- Passing this report still does not authorize live trading; it decides whether to proceed to CPCV/final-stage modeling.",
        "",
        "## Outputs",
        "",
        f"- Summary CSV: `{args.out_dir / 'track_a_pool_level_nulls.csv'}`",
        f"- Trial CSV: `{args.out_dir / 'track_a_pool_level_null_trials.csv'}`",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", type=Path, default=DEFAULT_MODEL_REPORT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--raw-1m-dir", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pockets", default="direction_hard,direction_soft")
    parser.add_argument("--target-fractions", default="0.6,0.8,1.0")
    parser.add_argument("--stop-atr-mults", default="1.0,1.5,2.0")
    parser.add_argument("--max-hold-bars", default="12,24,36,60")
    parser.add_argument("--session-exit-time", default="15:10")
    parser.add_argument("--data-timestamps-local-ist", action="store_true")
    parser.add_argument("--use-1m-resolution", action="store_true")
    parser.add_argument("--slippage-model", choices=["flat", "state_dependent"], default="state_dependent")
    parser.add_argument("--base-slippage-bps", type=float, default=2.0)
    parser.add_argument("--exchange", default="NSE")
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--max-candidates", type=int, default=None, help="Smoke helper: cap candidate rows per task.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    args._geometries = make_geometries(
        parse_csv_floats(args.target_fractions),
        parse_csv_floats(args.stop_atr_mults),
        parse_csv_ints(args.max_hold_bars),
    )
    return args


def main() -> int:
    global _WORKER_REPORT, _WORKER_REPORT_PATH
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    print(f"[pool-nulls] loading model report: {args.model_report}", flush=True)
    report = _load_report(args.model_report)
    _WORKER_REPORT = report
    _WORKER_REPORT_PATH = str(args.model_report)
    candidates = enrich_execution_refs(load_candidates(args.candidates), report)
    if candidates.empty:
        raise SystemExit("no executable candidates after enrichment")
    print(
        f"[pool-nulls] candidates={len(candidates):,} trials={args.trials} "
        f"workers={args.workers} geometries={len(args._geometries)}",
        flush=True,
    )
    commands = build_commands(args)
    batches = build_batch_payloads(candidates, args)
    print(
        f"[pool-nulls] replay commands={len(commands):,} batches={len(batches):,} "
        f"batch_size~={math.ceil(len(commands) / max(len(batches), 1))}",
        flush=True,
    )
    rows = pd.DataFrame(_run_batches(batches, args.workers))
    print(f"[pool-nulls] completed replay rows={len(rows):,}", flush=True)
    summary = aggregate_results(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    trial_csv = args.out_dir / "track_a_pool_level_null_trials.csv"
    summary_csv = args.out_dir / "track_a_pool_level_nulls.csv"
    rows.to_csv(trial_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    args.report_out.write_text(build_report(summary, args), encoding="utf-8")
    decision, reason = decide(summary)
    print(f"wrote {summary_csv}")
    print(f"wrote {trial_csv}")
    print(f"wrote {args.report_out}")
    print(f"decision: {decision} - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
