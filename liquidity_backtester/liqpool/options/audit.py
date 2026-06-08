"""Options executor audit — Table A, Table B, and SKIP counter-factual.

Aggregates per-trade outcome rows into the three calibration tables
described in ``docs/options_executor_layered_conviction.md`` §7-§8
and the methodology doc §7 + §10 (D.6).

The three tables answer three distinct questions:

  Table A — How does the executor perform on trades it took and
            exited normally? Standard hit-rate + mean realized R per
            (side × tenor × ToD) bucket.

  Table B — How does the conviction-hold discipline perform? Trades
            where the conviction model held through a drawdown,
            then exited. Bucketed by ``lcs_at_hold`` to expose
            whether high-LCS holds recover more often than low-LCS
            ones. If Table B doesn't beat a naive percent-stop
            baseline on real data, the conviction-hold logic gets
            retired (the methodology doc's Gate-2 commitment).

  SKIP counter-factual — What would the trades the executor REFUSED
            have returned if taken? If this number is consistently
            large positive, the SKIP rules are too conservative; if
            small/negative, the SKIP discipline is paying.

The audit reads outcome-log rows already written by the live runner.
Schema additions in this commit are ADDITIVE — old partitions
without the new fields read back as None, treated as missing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

OPTIONS_EXECUTOR_PREDICTION_TYPE: str = "options_executor"
OPTIONS_SKIP_PREDICTION_TYPE: str = "options_executor_skipped"


# Buckets for LCS_at_hold reporting (Table B sub-aggregation).
LCS_AT_HOLD_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("high",     0.50, 1.01),
    ("moderate", 0.30, 0.50),
    ("low",      0.10, 0.30),
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TableARow:
    bucket: str                    # "buy/weekly/open" etc.
    n: int
    mean_realized_r_atr: float
    win_rate: float
    by_exit_action: Dict[str, int]


@dataclass
class TableBLcsBucketRow:
    lcs_at_hold_bucket: str        # "high" | "moderate" | "low"
    n: int
    pct_recovered_to_profit: float
    mean_realized_r_atr: float
    pct_lost_more: float           # fraction of holds that ended worse
                                    # than the drawdown low


@dataclass
class TableB:
    n_total: int
    mean_realized_r_atr: float
    pct_recovered_to_profit: float
    by_lcs_at_hold: List[TableBLcsBucketRow] = field(default_factory=list)


@dataclass
class SkipCounterfactual:
    n: int
    mean_counterfactual_r_atr: float
    pct_would_have_been_profitable: float


@dataclass
class ExecutorAuditReport:
    trading_date_ist: str
    n_total: int
    table_a: List[TableARow] = field(default_factory=list)
    table_b: TableB = field(default_factory=lambda: TableB(0, 0.0, 0.0))
    skip_counterfactual: SkipCounterfactual = field(
        default_factory=lambda: SkipCounterfactual(0, 0.0, 0.0))
    gate2_naive_baseline_mean_r: Optional[float] = None
    gate2_conviction_minus_naive: Optional[float] = None
    gate2_pass: Optional[bool] = None
    is_retrospective_calibration: bool = False
    retrospective_share: float = 0.0


# ---------------------------------------------------------------------------
# Row helpers — embed options-executor state in PredictionRecord JSON
# ---------------------------------------------------------------------------

def options_executor_meta(*,
                           bucket_key: str,
                           lcs_at_entry: float,
                           lcs_at_exit: float,
                           agreeing_layers_at_entry: int,
                           agreeing_layers_at_exit: int,
                           min_lcs_during_trade: float,
                           min_realized_r_during_trade: float,
                           realized_r_at_exit_atr: float,
                           naive_exit_r_atr: Optional[float],
                           was_conviction_hold: bool,
                           exit_action: str,
                           predicted_target_r_atr: float,
                           ) -> Dict[str, Any]:
    """Build the dict that goes into ``PredictionRecord.regime_tags_at_prediction``
    (or ``ResolutionRecord.resolution_details``).

    Carries every piece of state the audit needs to compute Table A,
    Table B, and the naive-baseline comparison. ``naive_exit_r_atr``
    is the realized R the trade WOULD have produced at the naive
    percent-stop baseline — see the methodology doc Gate 2.
    """
    return {
        "bucket_key": bucket_key,
        "lcs_at_entry": float(lcs_at_entry),
        "lcs_at_exit": float(lcs_at_exit),
        "agreeing_layers_at_entry": int(agreeing_layers_at_entry),
        "agreeing_layers_at_exit": int(agreeing_layers_at_exit),
        "min_lcs_during_trade": float(min_lcs_during_trade),
        "min_realized_r_during_trade": float(min_realized_r_during_trade),
        "realized_r_at_exit_atr": float(realized_r_at_exit_atr),
        "naive_exit_r_atr": (
            float(naive_exit_r_atr) if naive_exit_r_atr is not None else None
        ),
        "was_conviction_hold": bool(was_conviction_hold),
        "exit_action": str(exit_action),
        "predicted_target_r_atr": float(predicted_target_r_atr),
    }


def options_skip_meta(*,
                       bucket_key: str,
                       lcs_at_decision: float,
                       counterfactual_r_atr: float,
                       skip_reason: str,
                       ) -> Dict[str, Any]:
    """Build the dict for a SKIP decision row.

    ``counterfactual_r_atr`` is the realized R the trade WOULD have
    produced if the executor had not skipped it — measured from the
    same bar data that produced the OOS labels.
    """
    return {
        "bucket_key": bucket_key,
        "lcs_at_decision": float(lcs_at_decision),
        "counterfactual_r_atr": float(counterfactual_r_atr),
        "skip_reason": str(skip_reason),
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def _bucket_for_lcs(lcs: float) -> Optional[str]:
    a = abs(lcs)
    for label, lo, hi in LCS_AT_HOLD_BUCKETS:
        if lo <= a < hi:
            return label
    return None


def _parse_meta(blob: Any) -> Dict[str, Any]:
    """Decode meta from either dict or JSON-string form."""
    if isinstance(blob, dict):
        return blob
    if isinstance(blob, str) and blob:
        try:
            d = json.loads(blob)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            return {}
    return {}


def _gather_meta(row: pd.Series) -> Dict[str, Any]:
    """Merge regime_tags + resolution_details + any other JSON columns."""
    out: Dict[str, Any] = {}
    for col in ("regime_tags_json", "resolution_details_json"):
        if col in row.index:
            out.update(_parse_meta(row.get(col)))
    for col in ("regime_tags_at_prediction", "resolution_details"):
        if col in row.index:
            v = row.get(col)
            if isinstance(v, dict):
                out.update(v)
    return out


def compute_executor_audit(joined: pd.DataFrame,
                            trading_date_ist: str,
                            ) -> ExecutorAuditReport:
    """Aggregate joined predictions+resolutions for one IST trading
    date into the three audit tables + the Gate-2 comparison.

    ``joined`` is the output of
    ``OutcomeLogWriter.read_joined(trading_date_ist)``: one row per
    prediction with resolution columns merged in. Rows whose
    ``prediction_type`` is not ``options_executor`` or
    ``options_executor_skipped`` are ignored — this audit only
    speaks about the options executor.
    """
    report = ExecutorAuditReport(
        trading_date_ist=trading_date_ist, n_total=0)
    if joined is None or joined.empty:
        return report
    if "prediction_type" not in joined.columns:
        return report

    taken = joined[
        joined["prediction_type"] == OPTIONS_EXECUTOR_PREDICTION_TYPE
    ].copy()
    skipped = joined[
        joined["prediction_type"] == OPTIONS_SKIP_PREDICTION_TYPE
    ].copy()

    report.n_total = int(len(taken) + len(skipped))
    report.table_a = _table_a(taken)
    report.table_b = _table_b(taken)
    report.skip_counterfactual = _skip_counterfactual(skipped)

    # Gate-2 comparison: mean realized R of conviction-holds vs naive
    # baseline. Naive baseline is recorded per-trade via
    # ``naive_exit_r_atr`` so we don't have to re-simulate.
    convictions: List[float] = []
    naives: List[float] = []
    for _, row in taken.iterrows():
        meta = _gather_meta(row)
        if meta.get("was_conviction_hold") and meta.get(
                "realized_r_at_exit_atr") is not None:
            convictions.append(float(meta["realized_r_at_exit_atr"]))
        if meta.get("naive_exit_r_atr") is not None:
            naives.append(float(meta["naive_exit_r_atr"]))
    if naives:
        report.gate2_naive_baseline_mean_r = float(
            sum(naives) / len(naives))
    if convictions and report.gate2_naive_baseline_mean_r is not None:
        mean_conv = sum(convictions) / len(convictions)
        delta = mean_conv - report.gate2_naive_baseline_mean_r
        report.gate2_conviction_minus_naive = float(delta)
        report.gate2_pass = bool(delta >= 0.20)
    elif convictions:
        # Have conviction outcomes but no naive baseline → cannot pass.
        report.gate2_pass = False

    # Stream G retrospective share — when the prediction frame carries
    # is_retrospective per row.
    if "is_retrospective" in joined.columns:
        share = joined["is_retrospective"].fillna(False).astype(bool).mean()
        report.retrospective_share = float(share)
        report.is_retrospective_calibration = bool(share > 0.5)

    return report


def _table_a(taken: pd.DataFrame) -> List[TableARow]:
    """Standard performance per bucket: mean R + win rate + exit-
    action histogram, restricted to NON-conviction-hold trades."""
    if taken.empty:
        return []
    rows: List[TableARow] = []
    bucket_to_trades: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in taken.iterrows():
        meta = _gather_meta(row)
        if meta.get("was_conviction_hold"):
            continue
        key = meta.get("bucket_key", "unknown")
        bucket_to_trades.setdefault(key, []).append(meta)
    for bucket, trades in bucket_to_trades.items():
        rs = [t.get("realized_r_at_exit_atr") for t in trades
              if t.get("realized_r_at_exit_atr") is not None]
        if not rs:
            continue
        by_action: Dict[str, int] = {}
        for t in trades:
            a = t.get("exit_action", "UNKNOWN")
            by_action[a] = by_action.get(a, 0) + 1
        rows.append(TableARow(
            bucket=bucket,
            n=len(rs),
            mean_realized_r_atr=float(sum(rs) / len(rs)),
            win_rate=float(sum(1 for r in rs if r > 0) / len(rs)),
            by_exit_action=by_action,
        ))
    rows.sort(key=lambda r: r.bucket)
    return rows


def _table_b(taken: pd.DataFrame) -> TableB:
    """Conviction-hold performance: bucketed by LCS at the hold
    moment so we can see whether high-LCS holds recover more than
    low-LCS ones."""
    if taken.empty:
        return TableB(0, 0.0, 0.0)
    holds: List[Dict[str, Any]] = []
    for _, row in taken.iterrows():
        meta = _gather_meta(row)
        if meta.get("was_conviction_hold"):
            holds.append(meta)
    if not holds:
        return TableB(0, 0.0, 0.0)
    rs = [h.get("realized_r_at_exit_atr", 0.0) for h in holds]
    min_rs = [h.get("min_realized_r_during_trade", 0.0) for h in holds]
    n = len(holds)
    mean_r = float(sum(rs) / n)
    recovered = sum(1 for r in rs if r > 0)
    pct_recovered = float(recovered / n)
    lost_more = sum(
        1 for r, mr in zip(rs, min_rs) if r < mr
    )
    # Sub-bucket by lcs_at_hold (we use min_lcs_during_trade as the
    # hold moment).
    sub_rows: List[TableBLcsBucketRow] = []
    by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    for h in holds:
        lcs_at_hold = float(h.get("min_lcs_during_trade", 0.0))
        bucket = _bucket_for_lcs(lcs_at_hold)
        if bucket is None:
            continue
        by_bucket.setdefault(bucket, []).append(h)
    for label, lo, hi in LCS_AT_HOLD_BUCKETS:
        bucket_holds = by_bucket.get(label, [])
        if not bucket_holds:
            continue
        brs = [b.get("realized_r_at_exit_atr", 0.0) for b in bucket_holds]
        bmin = [b.get("min_realized_r_during_trade", 0.0)
                for b in bucket_holds]
        sub_rows.append(TableBLcsBucketRow(
            lcs_at_hold_bucket=label,
            n=len(brs),
            mean_realized_r_atr=float(sum(brs) / len(brs)),
            pct_recovered_to_profit=float(
                sum(1 for r in brs if r > 0) / len(brs)),
            pct_lost_more=float(
                sum(1 for r, m in zip(brs, bmin) if r < m) / len(brs)),
        ))
    return TableB(
        n_total=n,
        mean_realized_r_atr=mean_r,
        pct_recovered_to_profit=pct_recovered,
        by_lcs_at_hold=sub_rows,
    )


def _skip_counterfactual(skipped: pd.DataFrame) -> SkipCounterfactual:
    if skipped.empty:
        return SkipCounterfactual(0, 0.0, 0.0)
    cfs: List[float] = []
    for _, row in skipped.iterrows():
        meta = _gather_meta(row)
        v = meta.get("counterfactual_r_atr")
        if v is not None:
            cfs.append(float(v))
    if not cfs:
        return SkipCounterfactual(0, 0.0, 0.0)
    return SkipCounterfactual(
        n=len(cfs),
        mean_counterfactual_r_atr=float(sum(cfs) / len(cfs)),
        pct_would_have_been_profitable=float(
            sum(1 for r in cfs if r > 0) / len(cfs)),
    )


# ---------------------------------------------------------------------------
# Render the audit as a Yesterday-Audit payload chunk
# ---------------------------------------------------------------------------

def audit_to_yesterday_audit_payload(report: ExecutorAuditReport,
                                      ) -> Dict[str, Any]:
    """Convert the audit report to the dict shape the brief's
    ``yesterday_audit`` section accepts as a sub-block.

    The brief renderer can either show the entire structure or a
    compressed summary — both forms are legal because the dict is
    keyed by descriptive names and never reorders silently.
    """
    return {
        "options_executor": {
            "n_total": report.n_total,
            "trading_date_ist": report.trading_date_ist,
            "table_a": [
                {
                    "bucket": r.bucket,
                    "n": r.n,
                    "mean_realized_r_atr": r.mean_realized_r_atr,
                    "win_rate": r.win_rate,
                    "by_exit_action": r.by_exit_action,
                }
                for r in report.table_a
            ],
            "table_b": {
                "n_total": report.table_b.n_total,
                "mean_realized_r_atr": report.table_b.mean_realized_r_atr,
                "pct_recovered_to_profit":
                    report.table_b.pct_recovered_to_profit,
                "by_lcs_at_hold": [
                    {
                        "lcs_at_hold_bucket": r.lcs_at_hold_bucket,
                        "n": r.n,
                        "pct_recovered_to_profit":
                            r.pct_recovered_to_profit,
                        "mean_realized_r_atr": r.mean_realized_r_atr,
                        "pct_lost_more": r.pct_lost_more,
                    }
                    for r in report.table_b.by_lcs_at_hold
                ],
            },
            "skip_counterfactual": {
                "n": report.skip_counterfactual.n,
                "mean_counterfactual_r_atr":
                    report.skip_counterfactual.mean_counterfactual_r_atr,
                "pct_would_have_been_profitable":
                    report.skip_counterfactual.pct_would_have_been_profitable,
            },
            "gate2": {
                "naive_baseline_mean_r": report.gate2_naive_baseline_mean_r,
                "conviction_minus_naive": report.gate2_conviction_minus_naive,
                "pass": report.gate2_pass,
            },
            "is_retrospective_calibration":
                report.is_retrospective_calibration,
            "retrospective_share": report.retrospective_share,
        }
    }
