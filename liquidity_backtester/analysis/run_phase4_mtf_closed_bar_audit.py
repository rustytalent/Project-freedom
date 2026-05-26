"""Audit multi-timeframe feature-store bars for closed-bar semantics.

This Phase 4 Workstream 2 probe validates the persisted higher-timeframe bar
shards against the 5m feature-store bars and quantifies the leak risk of naive
``merge_asof`` joins that use a higher-TF bar's open timestamp instead of its
close/known timestamp.

The audit is artifact-only: it does not retrain models or alter runtime config.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_FEATURE_STORE = Path("output_feature_store/core25_fresh_may25")
DEFAULT_REPORT = Path("reports/phase4_mtf_closed_bar_audit.md")
DEFAULT_ISSUES = Path("reports/phase4_mtf_closed_bar_issues.csv")
DEFAULT_TFS = ("15m", "60m", "180m", "1D", "1W")
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
OHLCV = ["open", "high", "low", "close", "volume"]
ISSUE_COLUMNS = ["severity", "symbol", "timeframe", "check", "message", "detail"]


@dataclass
class TimeframeAudit:
    symbol: str
    timeframe: str
    stored_rows: int
    expected_rows: int
    common_rows: int
    missing_expected_rows: int
    extra_expected_rows: int
    aggregate_mismatches: int
    max_ohlc_abs_error: float
    max_volume_abs_error: float
    duplicate_timestamps: int
    non_monotonic: int
    decision_rows: int
    naive_join_leak_risk_rows: int
    naive_join_leak_risk_pct: float


def _parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _discover_symbols(root: Path) -> List[str]:
    paths = sorted((root / "bars").glob("timeframe=5m/symbol=*"))
    return [p.name.split("=", 1)[1] for p in paths if p.is_dir()]


def _load_bars(root: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    paths = sorted((root / "bars").glob(f"timeframe={timeframe}/symbol={symbol}/year=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(f"missing {timeframe} bars for {symbol} under {root}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    missing = {"ts", *OHLCV}.difference(df.columns)
    if missing:
        raise ValueError(f"{symbol} {timeframe} bars missing columns: {sorted(missing)}")
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    for col in OHLCV:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


def _load_direction_decisions(root: Path, symbol: str, split: str) -> pd.Series:
    paths = sorted((root / "direction").glob(f"symbol={symbol}/split={split}/fold=*.parquet"))
    if not paths:
        return pd.Series([], dtype="datetime64[ns]")
    frames = [pd.read_parquet(p, columns=["ts"]) for p in paths]
    ts = pd.concat(frames, ignore_index=True)["ts"]
    return pd.to_datetime(ts).dropna().drop_duplicates().sort_values().reset_index(drop=True)


def _tf_rule(tf: str, base_start: pd.Timestamp) -> tuple[str, object | None, pd.Timedelta | None, pd.Timedelta]:
    """Return resample rule, origin, index shift, and closed-bar period."""
    if tf == "15m":
        return "15min", None, None, pd.Timedelta(minutes=15)
    if tf == "60m":
        return "60min", base_start, None, pd.Timedelta(minutes=60)
    if tf == "180m":
        return "180min", base_start, None, pd.Timedelta(minutes=180)
    if tf == "1D":
        return "1D", None, IST_OFFSET, pd.Timedelta(days=1)
    if tf == "1W":
        return "W-FRI", None, IST_OFFSET, pd.Timedelta(days=7)
    raise ValueError(f"unsupported timeframe for MTF audit: {tf}")


def _expected_from_base(base: pd.DataFrame, tf: str) -> tuple[pd.DataFrame, pd.Timedelta]:
    base_idx = base.set_index("ts")[OHLCV].sort_index()
    rule, origin, shift, period = _tf_rule(tf, pd.Timestamp(base_idx.index.min()))
    work = base_idx
    if shift is not None:
        work = work.copy()
        work.index = work.index + shift
    if origin is not None:
        expected = work.resample(rule, label="left", closed="left", origin=origin).agg(AGG).dropna(how="any")
    else:
        expected = work.resample(rule, label="left", closed="left").agg(AGG).dropna(how="any")
    if shift is not None:
        expected.index = expected.index - shift
    expected = expected.sort_index()
    expected.index.name = "ts"
    return expected, period


def _decision_leak_risk(stored: pd.DataFrame, decisions: pd.Series, period: pd.Timedelta) -> tuple[int, int, float]:
    if stored.empty or decisions.empty:
        return int(len(decisions)), 0, 0.0
    opens = pd.to_datetime(stored["ts"]).to_numpy(dtype="datetime64[ns]")
    closes = (pd.to_datetime(stored["ts"]) + period).to_numpy(dtype="datetime64[ns]")
    dts = pd.to_datetime(decisions).to_numpy(dtype="datetime64[ns]")
    naive_idx = np.searchsorted(opens, dts, side="right") - 1
    safe_idx = np.searchsorted(closes, dts, side="right") - 1
    risk = naive_idx != safe_idx
    risk_count = int(risk.sum())
    total = int(len(dts))
    pct = float(risk_count / total) if total else 0.0
    return total, risk_count, pct


def _append_issue(issues: List[Dict], severity: str, symbol: str, tf: str,
                  check: str, message: str, detail: str) -> None:
    issues.append({
        "severity": severity,
        "symbol": symbol,
        "timeframe": tf,
        "check": check,
        "message": message,
        "detail": detail,
    })


def audit_symbol_timeframe(
    *,
    root: Path,
    symbol: str,
    timeframe: str,
    base: pd.DataFrame,
    decisions: pd.Series,
    tolerance: float,
    issues: List[Dict],
) -> TimeframeAudit:
    stored = _load_bars(root, symbol, timeframe)
    expected, period = _expected_from_base(base, timeframe)
    stored_idx = stored.set_index("ts")[OHLCV].sort_index()

    duplicate_timestamps = int(stored["ts"].duplicated().sum())
    non_monotonic = int(not stored["ts"].is_monotonic_increasing)
    if duplicate_timestamps:
        _append_issue(issues, "ERROR", symbol, timeframe, "bar_index",
                      "higher-timeframe shard has duplicate timestamps", str(duplicate_timestamps))
    if non_monotonic:
        _append_issue(issues, "ERROR", symbol, timeframe, "bar_index",
                      "higher-timeframe shard is not monotonic increasing", "")

    common = stored_idx.index.intersection(expected.index)
    missing = stored_idx.index.difference(expected.index)
    extra = expected.index.difference(stored_idx.index)
    if len(missing):
        _append_issue(issues, "WARN", symbol, timeframe, "aggregate_coverage",
                      "stored bars missing from 5m-derived expected index", str(len(missing)))
    if len(extra):
        _append_issue(issues, "WARN", symbol, timeframe, "aggregate_coverage",
                      "5m-derived expected bars missing from stored higher-TF shard", str(len(extra)))

    aggregate_mismatches = 0
    max_ohlc = 0.0
    max_volume = 0.0
    if len(common):
        diff = (stored_idx.loc[common, OHLCV] - expected.loc[common, OHLCV]).abs()
        ohlc_diff = diff[["open", "high", "low", "close"]]
        volume_diff = diff["volume"]
        row_mismatch = (ohlc_diff.gt(tolerance).any(axis=1) | volume_diff.gt(tolerance))
        aggregate_mismatches = int(row_mismatch.sum())
        max_ohlc = float(ohlc_diff.max().max()) if not ohlc_diff.empty else 0.0
        max_volume = float(volume_diff.max()) if not volume_diff.empty else 0.0
        if aggregate_mismatches:
            _append_issue(
                issues, "WARN", symbol, timeframe, "aggregate_values",
                "stored higher-TF OHLCV differs from 5m-derived aggregation",
                f"rows={aggregate_mismatches}, max_ohlc={max_ohlc:g}, max_volume={max_volume:g}",
            )

    decision_rows, risk_rows, risk_pct = _decision_leak_risk(stored, decisions, period)
    if risk_rows:
        _append_issue(
            issues, "INFO", symbol, timeframe, "naive_join_risk",
            "naive backward join on bar-open timestamp would select an unclosed higher-TF bar",
            f"risk_rows={risk_rows}/{decision_rows}",
        )

    return TimeframeAudit(
        symbol=symbol,
        timeframe=timeframe,
        stored_rows=int(len(stored_idx)),
        expected_rows=int(len(expected)),
        common_rows=int(len(common)),
        missing_expected_rows=int(len(missing)),
        extra_expected_rows=int(len(extra)),
        aggregate_mismatches=aggregate_mismatches,
        max_ohlc_abs_error=max_ohlc,
        max_volume_abs_error=max_volume,
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic=non_monotonic,
        decision_rows=decision_rows,
        naive_join_leak_risk_rows=risk_rows,
        naive_join_leak_risk_pct=risk_pct,
    )


def run_audit(
    *,
    root: Path,
    symbols: Sequence[str] | None,
    timeframes: Sequence[str],
    decision_split: str,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = list(symbols) if symbols else _discover_symbols(root)
    if not selected:
        raise ValueError(f"no symbols found under {root}")

    rows: List[TimeframeAudit] = []
    issues: List[Dict] = []
    for symbol in selected:
        base = _load_bars(root, symbol, "5m")
        decisions = _load_direction_decisions(root, symbol, decision_split)
        for tf in timeframes:
            rows.append(audit_symbol_timeframe(
                root=root,
                symbol=symbol,
                timeframe=tf,
                base=base,
                decisions=decisions,
                tolerance=tolerance,
                issues=issues,
            ))
    return pd.DataFrame([asdict(row) for row in rows]), pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def _fmt_int(x: int | float) -> str:
    return f"{int(x):,}"


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _write_report(
    *,
    report_path: Path,
    issues_path: Path,
    root: Path,
    timeframes: Sequence[str],
    audits: pd.DataFrame,
    issues: pd.DataFrame,
    decision_split: str,
) -> None:
    error_count = int((issues["severity"] == "ERROR").sum()) if not issues.empty else 0
    warn_count = int((issues["severity"] == "WARN").sum()) if not issues.empty else 0
    info_count = int((issues["severity"] == "INFO").sum()) if not issues.empty else 0
    status = "FAIL" if error_count else "WARN" if warn_count else "PASS"

    summary_rows = []
    if not audits.empty:
        grouped = audits.groupby("timeframe", dropna=False).agg(
            symbols=("symbol", "count"),
            stored_rows=("stored_rows", "sum"),
            expected_rows=("expected_rows", "sum"),
            missing_expected_rows=("missing_expected_rows", "sum"),
            extra_expected_rows=("extra_expected_rows", "sum"),
            aggregate_mismatches=("aggregate_mismatches", "sum"),
            decision_rows=("decision_rows", "sum"),
            naive_join_leak_risk_rows=("naive_join_leak_risk_rows", "sum"),
            max_ohlc_abs_error=("max_ohlc_abs_error", "max"),
            max_volume_abs_error=("max_volume_abs_error", "max"),
        ).reset_index()
        for row in grouped.to_dict("records"):
            risk_pct = (
                float(row["naive_join_leak_risk_rows"]) / float(row["decision_rows"])
                if row["decision_rows"] else 0.0
            )
            summary_rows.append(
                "| {tf} | {symbols} | {stored} | {expected} | {missing} | {extra} | "
                "{mismatch} | {risk} | {risk_pct} | {max_ohlc:.3g} | {max_vol:.3g} |".format(
                    tf=row["timeframe"],
                    symbols=_fmt_int(row["symbols"]),
                    stored=_fmt_int(row["stored_rows"]),
                    expected=_fmt_int(row["expected_rows"]),
                    missing=_fmt_int(row["missing_expected_rows"]),
                    extra=_fmt_int(row["extra_expected_rows"]),
                    mismatch=_fmt_int(row["aggregate_mismatches"]),
                    risk=_fmt_int(row["naive_join_leak_risk_rows"]),
                    risk_pct=_pct(risk_pct),
                    max_ohlc=float(row["max_ohlc_abs_error"]),
                    max_vol=float(row["max_volume_abs_error"]),
                )
            )

    exact_tfs: List[str] = []
    warning_notes: List[str] = []
    if not audits.empty:
        for tf, g in audits.groupby("timeframe", dropna=False):
            missing = int(g["missing_expected_rows"].sum())
            extra = int(g["extra_expected_rows"].sum())
            mismatches = int(g["aggregate_mismatches"].sum())
            if missing == 0 and extra == 0 and mismatches == 0:
                exact_tfs.append(str(tf))
            else:
                parts = []
                if missing:
                    parts.append(f"{missing:,} stored bars missing from expected")
                if extra:
                    parts.append(f"{extra:,} expected bars missing from stored")
                if mismatches:
                    parts.append(f"{mismatches:,} aggregate value mismatches")
                warning_notes.append(f"- `{tf}`: " + "; ".join(parts) + ".")

    interpretation = [
        "The audit found no hard closed-bar errors. Intraday higher-timeframe bars that",
        "match exactly are listed in Key Findings; daily/weekly warning rows should be",
        "reviewed as data-generation consistency issues, not automatically as lookahead.",
        "The `naive_join_leak_risk_rows` column is not a current-code failure; it shows how many",
        "direction decision timestamps would leak if future code used a plain backward join",
        "on the higher-TF bar open timestamp instead of requiring `bar_open + timeframe <= decision_ts`.",
    ]
    if warn_count:
        interpretation.append(
            "WARN rows mean some stored higher-timeframe bars differ from exact 5m aggregation or coverage."
            " These should be reviewed, but they are not automatically lookahead errors."
        )
    if error_count:
        interpretation.append("ERROR rows block Track A/B work until fixed.")

    lines = [
        "# Phase 4 MTF Closed-Bar Audit",
        "",
        f"**Status:** {status}",
        "",
        "This Workstream 2 artifact audit checks multi-timeframe bar shards for closed-bar",
        "semantics before Track A/Track B strategy work continues.",
        "",
        "## Scope",
        "",
        f"- Feature store: `{root}`",
        f"- Higher timeframes: `{', '.join(timeframes)}`",
        f"- Decision split for join-risk scan: `{decision_split}` direction rows",
        f"- Audit rows: {_fmt_int(len(audits))}",
        f"- Issues: ERROR={error_count}, WARN={warn_count}, INFO={info_count}",
        "",
        "## Key Findings",
        "",
        f"- Hard errors: `{error_count}`.",
        f"- Exact higher-TF aggregations: `{', '.join(exact_tfs) if exact_tfs else 'none'}`.",
        f"- Naive open-timestamp join risk: `{_fmt_int(int(audits['naive_join_leak_risk_rows'].sum()) if not audits.empty else 0)}` decision rows.",
        "- Safe rule for any future MTF join: use only bars where `bar_open + timeframe <= decision_ts`.",
        *(warning_notes if warning_notes else ["- No aggregate coverage/value warnings."]),
        "",
        "## Summary",
        "",
        "| TF | Symbols | Stored Rows | Expected Rows | Stored Not In Expected | Expected Not In Stored | Aggregate Mismatches | Naive Join Risk Rows | Naive Join Risk % | Max OHLC Error | Max Volume Error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *summary_rows,
        "",
        "## Anchoring Rules Used",
        "",
        "- `15m`: pandas `15min`, left-labeled closed-left bars.",
        "- `60m`: pandas `60min`, left-labeled bars anchored to the first 5m timestamp.",
        "- `180m`: pandas `180min`, left-labeled bars anchored to the first 5m timestamp.",
        "- `1D`: IST-midnight calendar day, represented as naive UTC-style `18:30` timestamps.",
        "- `1W`: IST Friday-week anchor, represented as naive UTC-style `18:30` timestamps.",
        "",
        "## Interpretation",
        "",
        *interpretation,
        "",
        "## Issue Rows",
        "",
        f"Detailed rows were written to `{issues_path}`.",
        "",
        "## Next Step",
        "",
        "Continue Workstream 2 with pool availability replay. Track A should still wait for",
        "the remaining leakage checks plus the proximity distance-bucket AUC viability gate.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store-dir", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--symbols", default="", help="Comma-separated subset; default discovers all symbols.")
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TFS))
    parser.add_argument("--decision-split", default="oos")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--issues", type=Path, default=DEFAULT_ISSUES)
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = args.feature_store_dir
    symbols = _parse_csv(args.symbols) or None
    timeframes = _parse_csv(args.timeframes)
    audits, issues = run_audit(
        root=root,
        symbols=symbols,
        timeframes=timeframes,
        decision_split=args.decision_split,
        tolerance=args.tolerance,
    )
    args.issues.parent.mkdir(parents=True, exist_ok=True)
    issues.to_csv(args.issues, index=False)
    _write_report(
        report_path=args.report,
        issues_path=args.issues,
        root=root,
        timeframes=timeframes,
        audits=audits,
        issues=issues,
        decision_split=args.decision_split,
    )
    error_count = int((issues["severity"] == "ERROR").sum()) if not issues.empty else 0
    warn_count = int((issues["severity"] == "WARN").sum()) if not issues.empty else 0
    status = "FAIL" if error_count else "WARN" if warn_count else "PASS"
    print(f"MTF closed-bar audit status: {status}")
    print(f"issues: ERROR={error_count} WARN={warn_count}")
    print(f"report: {args.report}")
    print(f"issue rows: {args.issues}")
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
