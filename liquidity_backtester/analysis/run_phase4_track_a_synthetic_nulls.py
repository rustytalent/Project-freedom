"""Phase 4 Track A synthetic-null preflight from existing pre-touch trades.

This script is the fast null layer before the expensive generated-pool replay.
It consumes ``pretouch_sweep_trades.parquet`` from the v2 Track A sweep and
tests whether the discovered pre-touch pockets beat simple artifact-backed
nulls:

* matched-random rows: same geometry/distance-bucket counts, random trades
* time-label shuffle: preserve trades, randomize time buckets
* sector-label shuffle: preserve trades, randomize sectors
* direction-label shuffle: preserve trades, randomize UP/DOWN labels

These are not a replacement for full random-pool or ATR-offset re-simulation.
They are a cheap preflight that decides whether full null replay is worth
spending compute on.
"""
from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # noqa: E402
    from q_audit_common import fnum, markdown_table, pct
except ModuleNotFoundError:  # pragma: no cover - exercised by test imports
    from analysis.q_audit_common import fnum, markdown_table, pct


DEFAULT_TRADES = Path("output_phase4_track_a_pretouch_sweep/pretouch_sweep_trades.parquet")
DEFAULT_OUT_DIR = Path("output_audit/track_a_synthetic_nulls")
DEFAULT_REPORT = Path("reports/phase4_track_a_synthetic_nulls.md")

POCKETS: Dict[str, Dict[str, Sequence[str]]] = {
    "winning_combo": {
        "time_buckets": ("morning", "midday"),
        "sectors": ("AUTO", "FMCG", "PHARMA"),
        "directions": ("UP",),
    },
    "morning": {"time_buckets": ("morning",)},
    "midday": {"time_buckets": ("midday",)},
    "morning_midday": {"time_buckets": ("morning", "midday")},
    "auto_only": {"sectors": ("AUTO",)},
}

NULL_TESTS = (
    "matched_random_rows",
    "time_bucket_shuffle",
    "sector_shuffle",
    "direction_shuffle",
)

CORE_NULL_TESTS = (
    "matched_random_rows",
    "time_bucket_shuffle",
    "sector_shuffle",
)

MATCH_COLUMNS = (
    "target_fraction",
    "stop_atr_mult",
    "requested_max_hold_bars",
    "distance_bucket",
)


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
    se = float(r.std(ddof=1) / math.sqrt(len(r)))
    mean = float(r.mean())
    return mean - 1.96 * se, mean + 1.96 * se


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing Track A trade parquet: {path}")
    df = pd.read_parquet(path).copy()
    required = {
        "net_r",
        "risk_inr",
        "total_cost",
        "time_bucket",
        "sector",
        "direction",
        "target_fraction",
        "stop_atr_mult",
        "requested_max_hold_bars",
        "distance_atr",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"trade parquet missing required columns: {missing}")
    df["distance_bucket"] = pd.cut(
        pd.to_numeric(df["distance_atr"], errors="coerce"),
        bins=[-np.inf, 3.0, 5.0, 8.0, np.inf],
        labels=["lt3", "3-5", "5-8", "8+"],
    ).astype("string").fillna("unknown")
    return df


def filter_trades(trades: pd.DataFrame, pocket: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    mask = pd.Series(True, index=trades.index)
    if "time_buckets" in pocket:
        mask &= trades["time_bucket"].isin(pocket["time_buckets"])
    if "sectors" in pocket:
        mask &= trades["sector"].isin(pocket["sectors"])
    if "directions" in pocket:
        mask &= trades["direction"].isin(pocket["directions"])
    return trades.loc[mask].copy()


def summarize_frame(df: pd.DataFrame) -> Dict[str, float]:
    r = pd.to_numeric(df["net_r"], errors="coerce").dropna()
    ci_lo, ci_hi = _ci95(r)
    cost_150 = _cost_stress_r(df, 0.50)
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


def _cost_stress_r(df: pd.DataFrame, multiplier_add: float) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    risk = pd.to_numeric(df["risk_inr"], errors="coerce").replace(0.0, np.nan)
    extra_cost_r = pd.to_numeric(df["total_cost"], errors="coerce") * multiplier_add / risk
    return pd.to_numeric(df["net_r"], errors="coerce") - extra_cost_r.fillna(0.0)


def _allowed_mask(values: np.ndarray, allowed: Sequence[str] | None) -> np.ndarray:
    if not allowed:
        return np.ones(len(values), dtype=bool)
    return np.isin(values, list(allowed))


def _pocket_mask_arrays(
    pocket: Mapping[str, Sequence[str]],
    time_values: np.ndarray,
    sector_values: np.ndarray,
    direction_values: np.ndarray,
) -> np.ndarray:
    return (
        _allowed_mask(time_values, pocket.get("time_buckets"))
        & _allowed_mask(sector_values, pocket.get("sectors"))
        & _allowed_mask(direction_values, pocket.get("directions"))
    )


def _matched_random_means(
    trades: pd.DataFrame,
    pocket: Mapping[str, Sequence[str]],
    *,
    trials: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    actual = filter_trades(trades, pocket)
    if actual.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    group_counts = actual.groupby(list(MATCH_COLUMNS), observed=True).size()
    grouped_values = {
        key: pd.to_numeric(group["net_r"], errors="coerce").dropna().to_numpy(dtype=float)
        for key, group in trades.groupby(list(MATCH_COLUMNS), observed=True)
    }

    means: List[float] = []
    ns: List[int] = []
    for _ in range(trials):
        sampled_parts: List[np.ndarray] = []
        for key, n in group_counts.items():
            values = grouped_values.get(key)
            if values is None or len(values) == 0:
                continue
            sampled_parts.append(rng.choice(values, size=int(n), replace=True))
        if not sampled_parts:
            means.append(np.nan)
            ns.append(0)
            continue
        sampled = np.concatenate(sampled_parts)
        means.append(float(np.mean(sampled)))
        ns.append(int(len(sampled)))
    return np.asarray(means, dtype=float), np.asarray(ns, dtype=float)


def _label_shuffle_means(
    trades: pd.DataFrame,
    pocket: Mapping[str, Sequence[str]],
    *,
    null_name: str,
    trials: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    net_r = pd.to_numeric(trades["net_r"], errors="coerce").to_numpy(dtype=float)
    time_values = trades["time_bucket"].astype(str).to_numpy()
    sector_values = trades["sector"].astype(str).to_numpy()
    direction_values = trades["direction"].astype(str).to_numpy()
    means: List[float] = []
    ns: List[int] = []
    for _ in range(trials):
        if null_name == "time_bucket_shuffle":
            mask = _pocket_mask_arrays(
                pocket,
                rng.permutation(time_values),
                sector_values,
                direction_values,
            )
        elif null_name == "sector_shuffle":
            mask = _pocket_mask_arrays(
                pocket,
                time_values,
                rng.permutation(sector_values),
                direction_values,
            )
        elif null_name == "direction_shuffle":
            mask = _pocket_mask_arrays(
                pocket,
                time_values,
                sector_values,
                rng.permutation(direction_values),
            )
        else:
            raise ValueError(f"unknown label-shuffle null: {null_name}")
        values = net_r[mask & np.isfinite(net_r)]
        means.append(float(values.mean()) if len(values) else np.nan)
        ns.append(int(len(values)))
    return np.asarray(means, dtype=float), np.asarray(ns, dtype=float)


def _trial_worker(payload: Tuple[pd.DataFrame, Dict[str, Sequence[str]], str, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    trades, pocket, null_name, trials, seed = payload
    if null_name == "matched_random_rows":
        return _matched_random_means(trades, pocket, trials=trials, seed=seed)
    return _label_shuffle_means(trades, pocket, null_name=null_name, trials=trials, seed=seed)


def _split_trials(total: int, workers: int) -> List[int]:
    workers = max(1, min(workers, total))
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers) if base + (1 if i < rem else 0) > 0]


def run_null_distribution(
    trades: pd.DataFrame,
    pocket: Mapping[str, Sequence[str]],
    *,
    null_name: str,
    trials: int,
    seed: int,
    workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    chunks = _split_trials(trials, workers)
    payloads = [
        (trades, dict(pocket), null_name, chunk, seed + 1009 * i)
        for i, chunk in enumerate(chunks)
    ]
    if len(payloads) == 1:
        return _trial_worker(payloads[0])
    try:
        with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
            parts = list(pool.map(_trial_worker, payloads))
    except (OSError, PermissionError):
        # Some local sandboxes disallow multiprocessing semaphores. The VPS
        # path still uses process workers; this fallback keeps smoke tests
        # and Codex-local validation usable.
        parts = [_trial_worker(payload) for payload in payloads]
    means = np.concatenate([p[0] for p in parts]) if parts else np.asarray([], dtype=float)
    ns = np.concatenate([p[1] for p in parts]) if parts else np.asarray([], dtype=float)
    return means, ns


def _p_value(null_means: np.ndarray, actual_mean: float) -> float:
    clean = null_means[np.isfinite(null_means)]
    if clean.size == 0 or not np.isfinite(actual_mean):
        return float("nan")
    # Plus-one correction avoids reporting impossible p=0 on finite trials.
    return float((np.sum(clean >= actual_mean) + 1) / (clean.size + 1))


def run_null_suite(
    trades: pd.DataFrame,
    *,
    pockets: Mapping[str, Mapping[str, Sequence[str]]] = POCKETS,
    null_tests: Sequence[str] = NULL_TESTS,
    trials: int = 1000,
    seed: int = 20260601,
    workers: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    trial_rows: List[Dict] = []
    for pocket_idx, (pocket_name, pocket) in enumerate(pockets.items()):
        actual = filter_trades(trades, pocket)
        actual_summary = summarize_frame(actual)
        actual_mean = float(actual_summary["mean_r"])
        applicable_nulls = [
            null_name
            for null_name in null_tests
            if (
                null_name == "matched_random_rows"
                or (null_name == "time_bucket_shuffle" and "time_buckets" in pocket)
                or (null_name == "sector_shuffle" and "sectors" in pocket)
                or (null_name == "direction_shuffle" and "directions" in pocket)
            )
        ]
        for null_idx, null_name in enumerate(applicable_nulls):
            null_seed = seed + pocket_idx * 100_000 + null_idx * 10_000
            means, ns = run_null_distribution(
                trades,
                pocket,
                null_name=null_name,
                trials=trials,
                seed=null_seed,
                workers=workers,
            )
            clean = means[np.isfinite(means)]
            row = {
                "pocket": pocket_name,
                "null_test": null_name,
                **actual_summary,
                "null_trials": int(len(clean)),
                "null_mean_r_mean": float(clean.mean()) if len(clean) else np.nan,
                "null_mean_r_std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
                "null_mean_r_p05": float(np.quantile(clean, 0.05)) if len(clean) else np.nan,
                "null_mean_r_p50": float(np.quantile(clean, 0.50)) if len(clean) else np.nan,
                "null_mean_r_p95": float(np.quantile(clean, 0.95)) if len(clean) else np.nan,
                "null_median_n": float(np.nanmedian(ns)) if len(ns) else np.nan,
                "p_value": _p_value(means, actual_mean),
            }
            rows.append(row)
            for i, (mean_r, n_trades) in enumerate(zip(means, ns), start=1):
                trial_rows.append({
                    "pocket": pocket_name,
                    "null_test": null_name,
                    "trial": i,
                    "mean_r": mean_r,
                    "n_trades": n_trades,
                })
    return pd.DataFrame(rows), pd.DataFrame(trial_rows)


def _fmt_pf(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if math.isinf(float(value)):
        return "inf"
    return fnum(value, 3)


def _result_rows(results: pd.DataFrame, pocket_name: str) -> List[Dict]:
    rows = []
    for row in results[results["pocket"] == pocket_name].to_dict(orient="records"):
        rows.append({
            "null": row["null_test"],
            "n": int(row["n_trades"]),
            "actual_R": fnum(row["mean_r"], 3),
            "actual_CI": f"{fnum(row['ci95_lo'], 3)}/{fnum(row['ci95_hi'], 3)}",
            "PF": _fmt_pf(row["profit_factor"]),
            "null_R_med": fnum(row["null_mean_r_p50"], 3),
            "null_R_95": fnum(row["null_mean_r_p95"], 3),
            "p": fnum(row["p_value"], 4),
            "pass": "yes" if float(row["p_value"]) <= 0.05 else "no",
        })
    return rows


def _named_nulls_pass(
    results: pd.DataFrame,
    pocket_name: str,
    null_names: Sequence[str],
    threshold: float,
) -> bool:
    subset = results[results["pocket"] == pocket_name]
    subset = subset[subset["null_test"].isin(null_names)]
    if subset.empty or set(subset["null_test"]) != set(null_names):
        return False
    actual_ok = bool((subset["ci95_lo"] > 0.0).all())
    p_ok = bool((subset["p_value"] <= threshold).all())
    return actual_ok and p_ok


def decide(results: pd.DataFrame, primary_pocket: str) -> Tuple[str, str]:
    primary = results[results["pocket"] == primary_pocket]
    if primary.empty:
        return "FAIL", f"No rows were produced for primary pocket `{primary_pocket}`."
    core_pass = _named_nulls_pass(results, primary_pocket, CORE_NULL_TESTS, 0.05)
    direction_pass = _named_nulls_pass(results, primary_pocket, ("direction_shuffle",), 0.05)
    if core_pass and direction_pass:
        return "SYNTHETIC_PREFLIGHT_PASS", (
            f"`{primary_pocket}` beats all artifact-backed nulls at p<=0.05 and has positive CI."
        )
    if core_pass and not direction_pass:
        return "SYNTHETIC_PREFLIGHT_CORE_PASS_DIRECTION_WEAK", (
            f"`{primary_pocket}` beats matched-random/time/sector nulls, but the direction hard gate "
            "does not beat direction-label shuffle."
        )
    if _named_nulls_pass(results, primary_pocket, CORE_NULL_TESTS, 0.10):
        return "SYNTHETIC_PREFLIGHT_MARGINAL", (
            f"`{primary_pocket}` beats core artifact-backed nulls only at p<=0.10."
        )
    failed = primary[(primary["null_test"].isin(CORE_NULL_TESTS)) & (primary["p_value"] > 0.05)]["null_test"].tolist()
    return "SYNTHETIC_PREFLIGHT_FAIL", (
        f"`{primary_pocket}` failed at least one core artifact-backed null at p<=0.05: {', '.join(failed)}."
    )


def build_report(results: pd.DataFrame, args: argparse.Namespace) -> str:
    decision, reason = decide(results, args.primary_pocket)
    lines = [
        "# Phase 4 Track A Synthetic Null Preflight",
        "",
        "## Verdict",
        "",
        f"- Decision: `{decision}`",
        f"- {reason}",
        f"- Trials per pocket/null: `{args.trials}`",
        f"- Workers requested: `{args.workers}`",
        "- Scope: artifact-backed preflight using existing v2 pre-touch trade rows.",
        "- This does **not** replace full generated random-pool or ATR-offset replay through the simulator.",
        "",
        f"## Primary Pocket: `{args.primary_pocket}`",
        "",
        markdown_table(
            _result_rows(results, args.primary_pocket),
            ["null", "n", "actual_R", "actual_CI", "PF", "null_R_med", "null_R_95", "p", "pass"],
        ),
        "",
        "## Other Pockets",
        "",
    ]
    for pocket in [p for p in POCKETS if p != args.primary_pocket]:
        lines.extend([
            f"### `{pocket}`",
            "",
            markdown_table(
                _result_rows(results, pocket),
                ["null", "n", "actual_R", "actual_CI", "PF", "null_R_med", "null_R_95", "p", "pass"],
            ),
            "",
        ])
    lines.extend([
        "## Interpretation",
        "",
        "- Passing here means the pocket beats cheap label/matched-row nulls and is worth full synthetic replay.",
        "- A direction-shuffle failure is a component warning: direction should become a soft feature, not necessarily a hard gate.",
        "- Failing a core null means the apparent edge is likely explained by time, sector, or geometry/distance sampling effects.",
        "- If this passes, the next step is the heavier pool-level null suite: random pools, ATR-offset pools, sector-neutral random pools, and full V2 shuffled direction.",
        "",
        "## Outputs",
        "",
        f"- Summary CSV: `{args.out_dir / 'track_a_synthetic_nulls.csv'}`",
        f"- Trial CSV: `{args.out_dir / 'track_a_synthetic_null_trials.csv'}`",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--primary-pocket", default="winning_combo", choices=sorted(POCKETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trials < 10:
        raise SystemExit("--trials must be at least 10")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    trades = load_trades(args.trades)
    workers = max(1, min(args.workers, args.trials))
    print(
        f"[track-a-nulls] loaded {len(trades):,} trades from {args.trades}; "
        f"trials={args.trials} workers={workers}"
    )
    results, trial_results = run_null_suite(
        trades,
        trials=args.trials,
        seed=args.seed,
        workers=workers,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "track_a_synthetic_nulls.csv"
    trials_csv = args.out_dir / "track_a_synthetic_null_trials.csv"
    results.to_csv(summary_csv, index=False)
    trial_results.to_csv(trials_csv, index=False)
    args.report_out.write_text(build_report(results, args), encoding="utf-8")
    decision, reason = decide(results, args.primary_pocket)
    print(f"wrote {summary_csv}")
    print(f"wrote {trials_csv}")
    print(f"wrote {args.report_out}")
    print(f"decision: {decision} - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
