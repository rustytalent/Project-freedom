"""Audit persisted proximity distance features for decision-time causality.

This is a Phase 4 Workstream 2 slow/artifact-backed probe. It reads the saved
Core25 feature store and recomputes every persisted proximity ``distance_atr``
from:

* the 5m decision bar close,
* the trailing ATR(14) available at that bar, and
* the pool zone referenced by the proximity row.

It intentionally does not retrain any model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from liqpool.indicators import atr


DEFAULT_FEATURE_STORE = Path("output_feature_store/core25_fresh_may25")
DEFAULT_REPORT = Path("reports/phase4_proximity_distance_leakage_audit.md")
DEFAULT_ISSUES = Path("reports/phase4_proximity_distance_leakage_issues.csv")
DEFAULT_HORIZONS = (78, 156, 312)
DEFAULT_SPLIT = "oos"
ISSUE_COLUMNS = [
    "symbol",
    "split",
    "horizon",
    "file",
    "row",
    "ts",
    "bar_idx",
    "pool_idx",
    "stored_distance_atr",
    "abs_error",
    "stored_side_above",
    "distance_mismatch",
    "side_mismatch",
    "inside_zone",
]


@dataclass
class FileAudit:
    symbol: str
    split: str
    horizon: int
    file: str
    rows: int
    rows_checked: int
    distance_mismatches: int
    side_mismatches: int
    inside_zone_rows: int
    invalid_bar_idx: int
    invalid_pool_idx: int
    max_abs_error: float
    mean_abs_error: float


def _parse_csv_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _discover_symbols(root: Path) -> List[str]:
    paths = sorted((root / "pools").glob("symbol=*"))
    return [p.name.split("=", 1)[1] for p in paths if p.is_dir()]


def _load_bars(root: Path, symbol: str) -> pd.DataFrame:
    paths = sorted((root / "bars").glob(f"timeframe=5m/symbol={symbol}/year=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(f"missing 5m bars for {symbol} under {root}")
    bars = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    missing = {"ts", "close", "high", "low"}.difference(bars.columns)
    if missing:
        raise ValueError(f"5m bars for {symbol} missing columns: {sorted(missing)}")
    bars["ts"] = pd.to_datetime(bars["ts"])
    bars = bars.sort_values("ts").reset_index(drop=True)
    bars["_atr14"] = atr(bars, 14).bfill()
    return bars


def _load_combined_pools(root: Path, symbol: str) -> pd.DataFrame:
    """Load pools in the same index order used by proximity rows.

    The feature-store writer builds proximity rows against ``train_pools +
    oos_pools``. The persisted pool shards keep local split pool_idx values, so
    the audit must recreate the concatenated index explicitly.
    """
    frames = []
    for split in ("train", "oos"):
        path = root / f"pools/symbol={symbol}/{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df = df.copy()
            df["source_split"] = split
            df["source_pool_idx"] = df.get("pool_idx", pd.Series(np.arange(len(df))))
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"missing pool shards for {symbol} under {root}")
    pools = pd.concat(frames, ignore_index=True)
    missing = {"price_low", "price_high", "mid"}.difference(pools.columns)
    if missing:
        raise ValueError(f"pool shards for {symbol} missing columns: {sorted(missing)}")
    pools["proximity_pool_idx"] = np.arange(len(pools), dtype=int)
    return pools


def _proximity_files(root: Path, symbol: str, split: str, horizon: int) -> List[Path]:
    return sorted(
        (root / "proximity").glob(
            f"horizon={int(horizon)}/symbol={symbol}/split={split}/fold=*.parquet"
        )
    )


def _audit_file(
    path: Path,
    *,
    root: Path,
    symbol: str,
    split: str,
    horizon: int,
    bars: pd.DataFrame,
    pools: pd.DataFrame,
    tolerance: float,
    issue_rows: List[Dict],
    max_issue_rows: int,
    max_rows_per_file: int | None,
) -> FileAudit:
    columns = ["symbol", "split", "fold", "ts", "bar_idx", "pool_idx", "distance_atr", "side_above"]
    prox = pd.read_parquet(path, columns=columns)
    if max_rows_per_file is not None and len(prox) > max_rows_per_file:
        prox = prox.head(max_rows_per_file).copy()

    rows = int(len(prox))
    if prox.empty:
        return FileAudit(symbol, split, horizon, str(path.relative_to(root)), 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)

    bar_idx = pd.to_numeric(prox["bar_idx"], errors="coerce").to_numpy(dtype=float)
    pool_idx = pd.to_numeric(prox["pool_idx"], errors="coerce").to_numpy(dtype=float)
    stored = pd.to_numeric(prox["distance_atr"], errors="coerce").to_numpy(dtype=float)
    side_above = pd.to_numeric(prox["side_above"], errors="coerce").to_numpy(dtype=float)

    bar_valid = np.isfinite(bar_idx) & (bar_idx >= 0) & (bar_idx < len(bars))
    pool_valid = np.isfinite(pool_idx) & (pool_idx >= 0) & (pool_idx < len(pools))
    finite_valid = np.isfinite(stored) & np.isfinite(side_above)
    valid = bar_valid & pool_valid & finite_valid

    invalid_bar_idx = int((~bar_valid).sum())
    invalid_pool_idx = int((~pool_valid).sum())
    rows_checked = int(valid.sum())

    abs_error = np.full(rows, np.nan, dtype=float)
    distance_mismatch = np.zeros(rows, dtype=bool)
    side_mismatch = np.zeros(rows, dtype=bool)
    inside_zone = np.zeros(rows, dtype=bool)

    if rows_checked:
        bidx = bar_idx[valid].astype(int)
        pidx = pool_idx[valid].astype(int)
        closes = bars["close"].to_numpy(dtype=float)[bidx]
        atr_vals = np.maximum(bars["_atr14"].to_numpy(dtype=float)[bidx], 1e-9)
        mids = pools["mid"].to_numpy(dtype=float)[pidx]
        lows = pools["price_low"].to_numpy(dtype=float)[pidx]
        highs = pools["price_high"].to_numpy(dtype=float)[pidx]
        stored_side = side_above[valid] >= 0.5

        expected = np.where(stored_side, (mids - closes) / atr_vals, (closes - mids) / atr_vals)
        err = np.abs(stored[valid] - expected)
        valid_positions = np.flatnonzero(valid)
        abs_error[valid_positions] = err
        distance_mismatch[valid_positions] = err > tolerance

        expected_above = lows > closes
        expected_below = highs < closes
        inside = ~(expected_above | expected_below)
        inside_zone[valid_positions] = inside
        side_mismatch[valid_positions] = (~inside) & (stored_side != expected_above)

        issue_mask = distance_mismatch | side_mismatch | inside_zone
        if issue_mask.any() and len(issue_rows) < max_issue_rows:
            idxs = np.flatnonzero(issue_mask)[: max_issue_rows - len(issue_rows)]
            for i in idxs:
                issue_rows.append({
                    "symbol": symbol,
                    "split": split,
                    "horizon": int(horizon),
                    "file": str(path.relative_to(root)),
                    "row": int(i),
                    "ts": str(prox.iloc[i]["ts"]),
                    "bar_idx": int(bar_idx[i]) if np.isfinite(bar_idx[i]) else None,
                    "pool_idx": int(pool_idx[i]) if np.isfinite(pool_idx[i]) else None,
                    "stored_distance_atr": float(stored[i]) if np.isfinite(stored[i]) else None,
                    "abs_error": float(abs_error[i]) if np.isfinite(abs_error[i]) else None,
                    "stored_side_above": float(side_above[i]) if np.isfinite(side_above[i]) else None,
                    "distance_mismatch": bool(distance_mismatch[i]),
                    "side_mismatch": bool(side_mismatch[i]),
                    "inside_zone": bool(inside_zone[i]),
                })

    finite_err = abs_error[np.isfinite(abs_error)]
    return FileAudit(
        symbol=symbol,
        split=split,
        horizon=int(horizon),
        file=str(path.relative_to(root)),
        rows=rows,
        rows_checked=rows_checked,
        distance_mismatches=int(distance_mismatch.sum()),
        side_mismatches=int(side_mismatch.sum()),
        inside_zone_rows=int(inside_zone.sum()),
        invalid_bar_idx=invalid_bar_idx,
        invalid_pool_idx=invalid_pool_idx,
        max_abs_error=float(finite_err.max()) if len(finite_err) else 0.0,
        mean_abs_error=float(finite_err.mean()) if len(finite_err) else 0.0,
    )


def run_audit(
    *,
    root: Path,
    symbols: Sequence[str] | None,
    split: str,
    horizons: Sequence[int],
    tolerance: float,
    max_issue_rows: int,
    max_rows_per_file: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not root.exists():
        raise FileNotFoundError(f"feature-store root not found: {root}")

    selected_symbols = list(symbols) if symbols else _discover_symbols(root)
    if not selected_symbols:
        raise ValueError(f"no symbols found under feature-store root: {root}")

    file_audits: List[FileAudit] = []
    issue_rows: List[Dict] = []
    for symbol in selected_symbols:
        bars = _load_bars(root, symbol)
        pools = _load_combined_pools(root, symbol)
        for horizon in horizons:
            paths = _proximity_files(root, symbol, split, int(horizon))
            for path in paths:
                file_audits.append(
                    _audit_file(
                        path,
                        root=root,
                        symbol=symbol,
                        split=split,
                        horizon=int(horizon),
                        bars=bars,
                        pools=pools,
                        tolerance=tolerance,
                        issue_rows=issue_rows,
                        max_issue_rows=max_issue_rows,
                        max_rows_per_file=max_rows_per_file,
                    )
                )

    return pd.DataFrame([asdict(x) for x in file_audits]), pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS)


def _fmt_int(x: int | float) -> str:
    return f"{int(x):,}"


def _write_report(
    *,
    report_path: Path,
    issues_path: Path,
    root: Path,
    split: str,
    horizons: Sequence[int],
    tolerance: float,
    audits: pd.DataFrame,
    issues: pd.DataFrame,
    max_rows_per_file: int | None,
) -> None:
    total_rows = int(audits["rows"].sum()) if not audits.empty else 0
    total_checked = int(audits["rows_checked"].sum()) if not audits.empty else 0
    total_distance = int(audits["distance_mismatches"].sum()) if not audits.empty else 0
    total_side = int(audits["side_mismatches"].sum()) if not audits.empty else 0
    total_inside = int(audits["inside_zone_rows"].sum()) if not audits.empty else 0
    total_bad_idx = int((audits["invalid_bar_idx"] + audits["invalid_pool_idx"]).sum()) if not audits.empty else 0
    max_abs = float(audits["max_abs_error"].max()) if not audits.empty else 0.0
    status = "PASS" if (total_distance + total_side + total_inside + total_bad_idx) == 0 else "FAIL"

    by_horizon = []
    if not audits.empty:
        grouped = audits.groupby("horizon", dropna=False).agg(
            files=("file", "count"),
            rows_checked=("rows_checked", "sum"),
            distance_mismatches=("distance_mismatches", "sum"),
            side_mismatches=("side_mismatches", "sum"),
            inside_zone_rows=("inside_zone_rows", "sum"),
            max_abs_error=("max_abs_error", "max"),
        ).reset_index()
        for row in grouped.to_dict("records"):
            by_horizon.append(
                "| {horizon} | {files} | {rows_checked} | {distance_mismatches} | "
                "{side_mismatches} | {inside_zone_rows} | {max_abs_error:.3e} |".format(
                    horizon=int(row["horizon"]),
                    files=_fmt_int(row["files"]),
                    rows_checked=_fmt_int(row["rows_checked"]),
                    distance_mismatches=_fmt_int(row["distance_mismatches"]),
                    side_mismatches=_fmt_int(row["side_mismatches"]),
                    inside_zone_rows=_fmt_int(row["inside_zone_rows"]),
                    max_abs_error=float(row["max_abs_error"]),
                )
            )

    issue_note = (
        f"Issue samples were written to `{issues_path}`."
        if not issues.empty else
        f"No issue rows were found; `{issues_path}` contains headers only."
    )
    sample_note = (
        f"Sampled first {max_rows_per_file:,} rows per file for a smoke audit."
        if max_rows_per_file else
        "Audited every row in the selected proximity shards."
    )

    lines = [
        "# Phase 4 Proximity Distance Leakage Audit",
        "",
        f"**Status:** {status}",
        "",
        "This Workstream 2 audit checks whether persisted proximity `distance_atr` rows can be",
        "recomputed from the decision-time 5m close, trailing ATR(14), and the referenced pool",
        "zone. This is the critical causality probe for Track A because `distance_atr` is the",
        "dominant proximity feature.",
        "",
        "## Scope",
        "",
        f"- Feature store: `{root}`",
        f"- Split: `{split}`",
        f"- Horizons: `{', '.join(str(h) for h in horizons)}`",
        f"- Tolerance: `{tolerance:g}`",
        f"- Files checked: {_fmt_int(len(audits))}",
        f"- Rows checked: {_fmt_int(total_checked)} / {_fmt_int(total_rows)}",
        f"- Sampling: {sample_note}",
        "",
        "## Result",
        "",
        f"- Distance mismatches: {_fmt_int(total_distance)}",
        f"- Side mismatches: {_fmt_int(total_side)}",
        f"- Rows already inside pool zone: {_fmt_int(total_inside)}",
        f"- Invalid bar/pool indexes: {_fmt_int(total_bad_idx)}",
        f"- Max absolute distance error: `{max_abs:.3e}`",
        "",
        "## By Horizon",
        "",
        "| Horizon | Files | Rows Checked | Distance Mismatches | Side Mismatches | Inside-Zone Rows | Max Abs Error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *by_horizon,
        "",
        "## Pool Index Semantics",
        "",
        "Proximity shards store `pool_idx` against the in-memory `train_pools + oos_pools` list,",
        "not the local pool index inside `pools/symbol=<SYMBOL>/oos.parquet`. The audit recreates",
        "that combined table before recomputing distances. This is now documented because using",
        "the local OOS pool index creates false mismatch alarms.",
        "",
        "## Interpretation",
        "",
    ]
    if status == "PASS":
        lines.extend([
            "The persisted proximity `distance_atr` feature is reproducible from decision-time",
            "inputs for the audited artifacts. This does **not** prove the full proximity model",
            "is leakage-free; it specifically clears the highest-priority distance feature probe.",
        ])
    else:
        lines.extend([
            "At least one persisted proximity row could not be reproduced from decision-time",
            "inputs. Track A should stay blocked until these rows are explained and fixed.",
        ])
    lines.extend([
        "",
        "## Issue Rows",
        "",
        issue_note,
        "",
        "## Next Step",
        "",
        "Continue Workstream 2 with broader artifact-backed probes: MTF closed-bar replay and",
        "pool availability replay. Track A should still wait for those checks plus the distance",
        "bucket AUC gate.",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store-dir", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    parser.add_argument("--symbols", default="", help="Comma-separated subset; default discovers all symbols.")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--max-issue-rows", type=int, default=1000)
    parser.add_argument("--max-rows-per-file", type=int, default=0, help="0 means audit all rows.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--issues", type=Path, default=DEFAULT_ISSUES)
    args = parser.parse_args(list(argv) if argv is not None else None)

    horizons = _parse_csv_ints(args.horizons)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
    max_rows_per_file = args.max_rows_per_file if args.max_rows_per_file > 0 else None

    audits, issues = run_audit(
        root=args.feature_store_dir,
        symbols=symbols,
        split=args.split,
        horizons=horizons,
        tolerance=args.tolerance,
        max_issue_rows=args.max_issue_rows,
        max_rows_per_file=max_rows_per_file,
    )
    args.issues.parent.mkdir(parents=True, exist_ok=True)
    issues.to_csv(args.issues, index=False)
    _write_report(
        report_path=args.report,
        issues_path=args.issues,
        root=args.feature_store_dir,
        split=args.split,
        horizons=horizons,
        tolerance=args.tolerance,
        audits=audits,
        issues=issues,
        max_rows_per_file=max_rows_per_file,
    )
    total_errors = 0
    if not audits.empty:
        total_errors = int(
            audits["distance_mismatches"].sum()
            + audits["side_mismatches"].sum()
            + audits["inside_zone_rows"].sum()
            + audits["invalid_bar_idx"].sum()
            + audits["invalid_pool_idx"].sum()
        )
    print(f"distance leakage audit status: {'PASS' if total_errors == 0 else 'FAIL'}")
    print(f"rows checked: {int(audits['rows_checked'].sum()) if not audits.empty else 0:,}")
    print(f"report: {args.report}")
    print(f"issues: {args.issues}")
    if total_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
