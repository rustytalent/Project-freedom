"""Audit pool availability and sampled detector replay determinism.

This Phase 4 Workstream 2 probe checks two separate things:

1. Every saved pool is consumed no earlier than its contributors' ``known_at``
   timestamps. This full metadata pass is cheap and covers final, train, and
   OOS pool sets from the saved model report.
2. A sampled replay re-runs pool detection on history truncated at
   ``available_at`` and checks that the saved pool is reproduced near the same
   side/price zone. Full replay for every Core25 pool is intentionally not the
   default because it is expensive on the local Mac.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from liqpool.leakage_audit import (
    LeakageIssue,
    _bar_period_seconds,
    _replay_audit_for_symbol,
    _tf_period,
    _ts,
)


DEFAULT_MODEL_REPORT = Path("output_models/core25_latest/multi_asset_report.pkl")
DEFAULT_REPORT = Path("reports/phase4_pool_availability_replay_audit.md")
DEFAULT_ISSUES = Path("reports/phase4_pool_availability_replay_issues.csv")
DEFAULT_POOL_SETS = ("final", "train", "oos")
ISSUE_COLUMNS = [
    "severity",
    "check",
    "symbol",
    "pool_idx",
    "message",
    "pool_set",
    "contributor_index",
    "source",
    "tf",
    "ts",
    "source_close",
    "known_at",
    "formed_at",
    "available_at",
    "contributor_known_at",
    "pool_available_at",
    "touched_at",
    "broken_at",
    "error",
    "side",
    "pool_low",
    "pool_high",
    "tolerance",
    "replayed_pool_count",
]


@dataclass
class AvailabilitySummary:
    symbol: str
    pool_set: str
    pools: int
    contributors: int
    missing_available_at: int
    available_before_formed: int
    missing_contributor_known_at: int
    contributor_known_before_ts: int
    contributor_known_before_source_close: int
    available_before_contributor_known: int
    available_after_final_base_bar: int
    touched_before_available: int
    broken_before_available: int


def _parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _issue(severity: str, check: str, symbol: str, pool_idx: Optional[int],
           message: str, **detail) -> LeakageIssue:
    return LeakageIssue(
        severity=severity,
        check=check,
        symbol=symbol,
        pool_idx=pool_idx,
        message=message,
        detail=detail,
    )


def _load_report(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"model report not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _pool_set(asset_data, name: str) -> Tuple[Sequence, Sequence]:
    wf = getattr(asset_data, "walkforward", None)
    if name == "final":
        return list(getattr(asset_data, "final_pools", []) or []), list(getattr(asset_data, "final_results", []) or [])
    if name == "train":
        return list(getattr(wf, "train_pools_for_ml", []) or []), list(getattr(wf, "train_results_for_ml", []) or [])
    if name == "oos":
        return list(getattr(wf, "oos_pools", []) or []), list(getattr(wf, "oos_results", []) or [])
    raise ValueError(f"unknown pool set: {name}")


def _empty_summary(symbol: str, pool_set: str) -> AvailabilitySummary:
    return AvailabilitySummary(
        symbol=symbol,
        pool_set=pool_set,
        pools=0,
        contributors=0,
        missing_available_at=0,
        available_before_formed=0,
        missing_contributor_known_at=0,
        contributor_known_before_ts=0,
        contributor_known_before_source_close=0,
        available_before_contributor_known=0,
        available_after_final_base_bar=0,
        touched_before_available=0,
        broken_before_available=0,
    )


def _append_issue(issues: List[LeakageIssue], issue: LeakageIssue, max_issues: int) -> None:
    if len(issues) < max_issues:
        issues.append(issue)


def _audit_pool_set(
    *,
    symbol: str,
    pool_set: str,
    pools: Sequence,
    results: Sequence,
    base_index: pd.Index,
    base_period_seconds: float,
    max_issues: int,
    issues: List[LeakageIssue],
) -> AvailabilitySummary:
    summary = _empty_summary(symbol, pool_set)
    summary.pools = int(len(pools))
    result_list = list(results)
    for pool_idx, pool in enumerate(pools):
        result = result_list[pool_idx] if pool_idx < len(result_list) else None
        available_at = _ts(getattr(pool, "available_at", None))
        formed_at = _ts(getattr(pool, "formed_at", None))

        if available_at is None:
            summary.missing_available_at += 1
            _append_issue(issues, _issue(
                "ERROR", "pool_availability", symbol, pool_idx,
                "pool.available_at is missing or invalid",
                pool_set=pool_set,
            ), max_issues)
            continue
        if formed_at is not None and available_at < formed_at:
            summary.available_before_formed += 1
            _append_issue(issues, _issue(
                "ERROR", "pool_availability", symbol, pool_idx,
                "pool is available before its formation timestamp",
                pool_set=pool_set,
                formed_at=str(formed_at),
                available_at=str(available_at),
            ), max_issues)

        if len(base_index):
            start_idx = int(base_index.searchsorted(available_at, side="left"))
            if start_idx >= len(base_index):
                summary.available_after_final_base_bar += 1
                _append_issue(issues, _issue(
                    "WARN", "pool_availability", symbol, pool_idx,
                    "pool becomes available after the final base bar",
                    pool_set=pool_set,
                    available_at=str(available_at),
                ), max_issues)

        for c_idx, contributor in enumerate(getattr(pool, "contributors", []) or []):
            summary.contributors += 1
            known_at = _ts(getattr(contributor, "known_at", None))
            ts = _ts(getattr(contributor, "ts", None))
            source = getattr(contributor, "source", "")
            tf = getattr(contributor, "tf", "base")
            if known_at is None:
                summary.missing_contributor_known_at += 1
                _append_issue(issues, _issue(
                    "ERROR", "contributor_availability", symbol, pool_idx,
                    "contributor known_at is missing",
                    pool_set=pool_set,
                    contributor_index=c_idx,
                    source=source,
                ), max_issues)
                continue
            if ts is not None and known_at < ts:
                summary.contributor_known_before_ts += 1
                _append_issue(issues, _issue(
                    "ERROR", "contributor_availability", symbol, pool_idx,
                    "contributor is known before its source timestamp",
                    pool_set=pool_set,
                    contributor_index=c_idx,
                    source=source,
                    ts=str(ts),
                    known_at=str(known_at),
                ), max_issues)
            period = _tf_period(str(tf), base_period_seconds)
            if ts is not None and period is not None:
                source_close = ts + period
                if known_at < source_close:
                    summary.contributor_known_before_source_close += 1
                    _append_issue(issues, _issue(
                        "ERROR", "mtf_closed_bar", symbol, pool_idx,
                        "contributor is known before its source bar is fully closed",
                        pool_set=pool_set,
                        contributor_index=c_idx,
                        source=source,
                        tf=str(tf),
                        ts=str(ts),
                        source_close=str(source_close),
                        known_at=str(known_at),
                    ), max_issues)
            if available_at < known_at:
                summary.available_before_contributor_known += 1
                _append_issue(issues, _issue(
                    "ERROR", "pool_availability", symbol, pool_idx,
                    "pool available_at is earlier than a contributor known_at",
                    pool_set=pool_set,
                    contributor_index=c_idx,
                    source=source,
                    contributor_known_at=str(known_at),
                    pool_available_at=str(available_at),
                ), max_issues)

        if result is not None:
            touched_at = _ts(getattr(result, "touched_at", None))
            broken_at = _ts(getattr(result, "broken_at", None))
            if touched_at is not None and touched_at < available_at:
                summary.touched_before_available += 1
                _append_issue(issues, _issue(
                    "ERROR", "label_window", symbol, pool_idx,
                    "pool touched before it became available",
                    pool_set=pool_set,
                    touched_at=str(touched_at),
                    available_at=str(available_at),
                ), max_issues)
            if broken_at is not None and broken_at < available_at:
                summary.broken_before_available += 1
                _append_issue(issues, _issue(
                    "ERROR", "label_window", symbol, pool_idx,
                    "pool broke before it became available",
                    pool_set=pool_set,
                    broken_at=str(broken_at),
                    available_at=str(available_at),
                ), max_issues)
    return summary


def _issue_frame(issues: Sequence[LeakageIssue]) -> pd.DataFrame:
    rows = []
    for issue in issues:
        row = issue.to_dict()
        detail = row.pop("detail", {}) or {}
        for key, value in detail.items():
            row[str(key)] = value
        rows.append(row)
    return pd.DataFrame(rows, columns=ISSUE_COLUMNS)


def run_audit(
    *,
    model_report: Path,
    symbols: Sequence[str] | None,
    pool_sets: Sequence[str],
    samples_per_symbol: int,
    max_issues: int,
    replay_severity: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    report = _load_report(model_report)
    assets = getattr(report, "assets", {}) or {}
    selected_symbols = list(symbols) if symbols else list(assets.keys())
    summaries: List[AvailabilitySummary] = []
    issues: List[LeakageIssue] = []
    replay_summary = {
        "samples_per_symbol": int(max(0, samples_per_symbol)),
        "checked": 0,
        "misses": 0,
        "skipped": 0,
        "severity": replay_severity.upper(),
        "per_symbol": {},
    }

    for symbol in selected_symbols:
        ad = assets.get(symbol)
        if ad is None:
            _append_issue(issues, _issue(
                "ERROR", "asset_lookup", symbol, None,
                "symbol was requested but is missing from model report",
            ), max_issues)
            continue
        base = getattr(ad, "base_df", None)
        base_index = getattr(base, "index", pd.Index([]))
        base_period_seconds = _bar_period_seconds(base_index)
        for pool_set in pool_sets:
            pools, results = _pool_set(ad, pool_set)
            summaries.append(_audit_pool_set(
                symbol=symbol,
                pool_set=pool_set,
                pools=pools,
                results=results,
                base_index=base_index,
                base_period_seconds=base_period_seconds,
                max_issues=max_issues,
                issues=issues,
            ))

        replay_issues, replay = _replay_audit_for_symbol(
            symbol,
            ad,
            samples=int(max(0, samples_per_symbol)),
            severity=replay_severity.upper(),
        )
        for issue in replay_issues:
            _append_issue(issues, issue, max_issues)
        replay_summary["checked"] += int(replay.get("checked", 0))
        replay_summary["misses"] += int(replay.get("misses", 0))
        replay_summary["skipped"] += int(replay.get("skipped", 0))
        replay_summary["per_symbol"][symbol] = replay

    return (
        pd.DataFrame([asdict(summary) for summary in summaries]),
        _issue_frame(issues),
        replay_summary,
    )


def _fmt_int(x: int | float) -> str:
    return f"{int(x):,}"


def _write_report(
    *,
    report_path: Path,
    issues_path: Path,
    model_report: Path,
    summaries: pd.DataFrame,
    issues: pd.DataFrame,
    replay_summary: Dict,
) -> None:
    severity_col = issues["severity"] if "severity" in issues.columns else pd.Series([], dtype=object)
    error_count = int((severity_col == "ERROR").sum())
    warn_count = int((severity_col == "WARN").sum())
    status = "FAIL" if error_count else "WARN" if warn_count else "PASS"

    total_pools = int(summaries["pools"].sum()) if not summaries.empty else 0
    total_contributors = int(summaries["contributors"].sum()) if not summaries.empty else 0
    replay_checked = int(replay_summary.get("checked", 0))
    replay_misses = int(replay_summary.get("misses", 0))
    replay_skipped = int(replay_summary.get("skipped", 0))

    set_rows = []
    if not summaries.empty:
        grouped = summaries.groupby("pool_set", dropna=False).sum(numeric_only=True).reset_index()
        for row in grouped.to_dict("records"):
            set_rows.append(
                "| {pool_set} | {pools} | {contributors} | {missing_available} | "
                "{known_before_close} | {available_before_known} | {after_final} | "
                "{touch_before} | {break_before} |".format(
                    pool_set=row["pool_set"],
                    pools=_fmt_int(row["pools"]),
                    contributors=_fmt_int(row["contributors"]),
                    missing_available=_fmt_int(row["missing_available_at"]),
                    known_before_close=_fmt_int(row["contributor_known_before_source_close"]),
                    available_before_known=_fmt_int(row["available_before_contributor_known"]),
                    after_final=_fmt_int(row["available_after_final_base_bar"]),
                    touch_before=_fmt_int(row["touched_before_available"]),
                    break_before=_fmt_int(row["broken_before_available"]),
                )
            )

    replay_rows = []
    for symbol, replay in sorted((replay_summary.get("per_symbol") or {}).items()):
        replay_rows.append(
            "| {symbol} | {checked} | {misses} | {skipped} |".format(
                symbol=symbol,
                checked=_fmt_int(replay.get("checked", 0)),
                misses=_fmt_int(replay.get("misses", 0)),
                skipped=_fmt_int(replay.get("skipped", 0)),
            )
        )

    lines = [
        "# Phase 4 Pool Availability Replay Audit",
        "",
        f"**Status:** {status}",
        "",
        "This Workstream 2 audit checks whether saved liquidity pools are point-in-time",
        "available before labels/trades consume them. It combines a full contributor",
        "timestamp pass with a sampled detector replay truncated at `available_at`.",
        "",
        "## Scope",
        "",
        f"- Model report: `{model_report}`",
        f"- Pool rows audited: {_fmt_int(total_pools)}",
        f"- Contributor rows audited: {_fmt_int(total_contributors)}",
        f"- Replay samples per symbol: `{replay_summary.get('samples_per_symbol', 0)}`",
        f"- Replay checked/misses/skipped: {_fmt_int(replay_checked)} / {_fmt_int(replay_misses)} / {_fmt_int(replay_skipped)}",
        f"- Issues: ERROR={error_count}, WARN={warn_count}",
        "",
        "## Availability Metadata By Pool Set",
        "",
        "| Pool Set | Pools | Contributors | Missing Available | Known Before Source Close | Available Before Known | Available After Final Bar | Touch Before Available | Break Before Available |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *set_rows,
        "",
        "## Detector Replay By Symbol",
        "",
        "| Symbol | Checked | Misses | Skipped |",
        "| --- | ---: | ---: | ---: |",
        *replay_rows,
        "",
        "## Interpretation",
        "",
    ]
    if error_count == 0 and replay_misses == 0:
        lines.extend([
            "No hard pool-availability leakage was found in the audited model report.",
            "The sampled replay reproduced every checked pool from data truncated at",
            "`available_at`.",
        ])
    else:
        lines.extend([
            "At least one pool-availability or replay issue was found. Track A/B strategy",
            "work should remain blocked until the issue rows are explained or fixed.",
        ])
    lines.extend([
        "",
        "Full replay of every Core25 pool remains intentionally disabled by default because",
        "each replay re-runs the detector stack over truncated history. Increase",
        "`--samples-per-symbol` for slower local or cloud runs.",
        "",
        "## Issue Rows",
        "",
        f"Detailed rows were written to `{issues_path}`.",
        "",
        "## Next Step",
        "",
        "With Q scale, fast probes, distance causality, MTF closed-bar semantics, and pool",
        "availability replay covered, the next Phase 4 gate is the Track A proximity",
        "distance-bucket AUC audit.",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", type=Path, default=DEFAULT_MODEL_REPORT)
    parser.add_argument("--symbols", default="", help="Comma-separated subset; default uses all report assets.")
    parser.add_argument("--pool-sets", default=",".join(DEFAULT_POOL_SETS))
    parser.add_argument("--samples-per-symbol", type=int, default=1)
    parser.add_argument("--max-issues", type=int, default=5000)
    parser.add_argument("--replay-severity", default="ERROR", choices=["ERROR", "WARN", "INFO"])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--issues", type=Path, default=DEFAULT_ISSUES)
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbols = _parse_csv(args.symbols) or None
    pool_sets = _parse_csv(args.pool_sets)
    summaries, issues, replay_summary = run_audit(
        model_report=args.model_report,
        symbols=symbols,
        pool_sets=pool_sets,
        samples_per_symbol=args.samples_per_symbol,
        max_issues=args.max_issues,
        replay_severity=args.replay_severity,
    )
    args.issues.parent.mkdir(parents=True, exist_ok=True)
    issues.to_csv(args.issues, index=False)
    _write_report(
        report_path=args.report,
        issues_path=args.issues,
        model_report=args.model_report,
        summaries=summaries,
        issues=issues,
        replay_summary=replay_summary,
    )
    severity_col = issues["severity"] if "severity" in issues.columns else pd.Series([], dtype=object)
    error_count = int((severity_col == "ERROR").sum())
    warn_count = int((severity_col == "WARN").sum())
    status = "FAIL" if error_count else "WARN" if warn_count else "PASS"
    print(f"pool availability replay audit status: {status}")
    print(f"issues: ERROR={error_count} WARN={warn_count}")
    print(f"replay checked/misses/skipped: {replay_summary['checked']:,}/{replay_summary['misses']:,}/{replay_summary['skipped']:,}")
    print(f"report: {args.report}")
    print(f"issue rows: {args.issues}")
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
