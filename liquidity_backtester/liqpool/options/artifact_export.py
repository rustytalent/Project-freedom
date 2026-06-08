"""Options artifact exporters (Stream D.7).

Renders the live brief's options data into CSV artifacts that the
artifact pusher uploads to the website portal. Two kinds at v1:

  ``options_executor_calls_csv``
    One row per (index, strike, side) that the brief flagged in
    options_suitability. Carries the executor decision (ENTER /
    WAIT / SKIP) and the predicted_R from the
    OptionsExpectedReturnModel.

  ``options_executor_audit_csv``
    A flat dump of the D.6 audit: Table A per bucket, Table B per
    LCS-at-hold sub-bucket, the SKIP counter-factual, and the
    Gate-2 verdict, in a single CSV (one "section" column tags the
    row type so the consumer can split client-side).

Both functions are PURE — they emit CSV strings without touching
the filesystem or network. The convenience ``push_*`` wrappers
write to a temporary path and call the artifact pusher with the
correct kind + tier.
"""
from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..products.artifact_pusher import push_artifact
from .audit import ExecutorAuditReport


# ---------------------------------------------------------------------------
# Per-strike executor calls
# ---------------------------------------------------------------------------

_CALLS_COLUMNS: tuple[str, ...] = (
    "trading_date_ist",
    "index_name",
    "strike",
    "key_level_type",
    "side_from_open",
    "underlying_level",
    "p_test_today",
    "p_test_within_60min",
    "predicted_net_return_buy_atr",
    "predicted_net_return_sell_atr",
    "executor_decision_buy",
    "executor_decision_sell",
)


def options_strikes_csv(strikes_by_index: Dict[str, List[Dict[str, Any]]],
                         trading_date_ist: str) -> str:
    """Render the per-strike rows from one Daily Brief as CSV.

    ``strikes_by_index`` maps each index name to the
    ``strike_levels_in_play`` list produced by ``generate_brief``.
    Missing values become empty cells.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_CALLS_COLUMNS)
    for idx_name, rows in (strikes_by_index or {}).items():
        for row in rows or []:
            writer.writerow([
                trading_date_ist,
                idx_name,
                row.get("strike"),
                row.get("key_level_type"),
                row.get("side_from_open"),
                row.get("underlying_level"),
                row.get("p_test_today"),
                row.get("p_test_within_60min"),
                row.get("predicted_net_return_buy_atr"),
                row.get("predicted_net_return_sell_atr"),
                row.get("executor_decision_buy"),
                row.get("executor_decision_sell"),
            ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Executor-audit CSV (Table A + Table B + SKIP + Gate-2)
# ---------------------------------------------------------------------------

_AUDIT_HEADER: tuple[str, ...] = (
    "trading_date_ist",
    "section",                 # table_a | table_b | table_b_bucket |
                               # skip_counterfactual | gate2
    "bucket_key",              # bucket name for table_a / table_b_bucket
    "lcs_at_hold_bucket",      # only for table_b_bucket
    "n",
    "mean_realized_r_atr",
    "win_rate",
    "pct_recovered_to_profit",
    "pct_lost_more",
    "mean_counterfactual_r_atr",
    "pct_would_have_been_profitable",
    "naive_baseline_mean_r",
    "conviction_minus_naive",
    "gate2_pass",
    "is_retrospective_calibration",
    "retrospective_share",
)


def executor_audit_csv(report: ExecutorAuditReport) -> str:
    """Render the D.6 audit report as a single CSV file.

    Row sections (in order):
      * one ``table_a`` row per bucket
      * one summary ``table_b`` row + one ``table_b_bucket`` row per
        LCS-at-hold sub-bucket
      * one ``skip_counterfactual`` row
      * one ``gate2`` row carrying the Gate-2 verdict

    All numeric columns are emitted as strings (CSV does not have a
    numeric type); empty when the corresponding field doesn't apply.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_AUDIT_HEADER)

    common: List[Any] = [
        report.trading_date_ist, "", "", "", "", "", "", "", "", "", "",
        "", "", "",
        int(report.is_retrospective_calibration),
        f"{report.retrospective_share:.4f}",
    ]

    # Table A
    for r in report.table_a:
        row = list(common)
        row[1] = "table_a"
        row[2] = r.bucket
        row[4] = r.n
        row[5] = f"{r.mean_realized_r_atr:.4f}"
        row[6] = f"{r.win_rate:.4f}"
        writer.writerow(row)

    # Table B summary + per-sub-bucket
    tb = report.table_b
    if tb.n_total > 0:
        row = list(common)
        row[1] = "table_b"
        row[4] = tb.n_total
        row[5] = f"{tb.mean_realized_r_atr:.4f}"
        row[7] = f"{tb.pct_recovered_to_profit:.4f}"
        writer.writerow(row)
        for r in tb.by_lcs_at_hold:
            row = list(common)
            row[1] = "table_b_bucket"
            row[3] = r.lcs_at_hold_bucket
            row[4] = r.n
            row[5] = f"{r.mean_realized_r_atr:.4f}"
            row[7] = f"{r.pct_recovered_to_profit:.4f}"
            row[8] = f"{r.pct_lost_more:.4f}"
            writer.writerow(row)

    # SKIP counter-factual
    sk = report.skip_counterfactual
    if sk.n > 0:
        row = list(common)
        row[1] = "skip_counterfactual"
        row[4] = sk.n
        row[9] = f"{sk.mean_counterfactual_r_atr:.4f}"
        row[10] = f"{sk.pct_would_have_been_profitable:.4f}"
        writer.writerow(row)

    # Gate-2 verdict
    row = list(common)
    row[1] = "gate2"
    if report.gate2_naive_baseline_mean_r is not None:
        row[11] = f"{report.gate2_naive_baseline_mean_r:.4f}"
    if report.gate2_conviction_minus_naive is not None:
        row[12] = f"{report.gate2_conviction_minus_naive:.4f}"
    if report.gate2_pass is not None:
        row[13] = "1" if report.gate2_pass else "0"
    writer.writerow(row)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Push convenience wrappers
# ---------------------------------------------------------------------------

def push_options_strikes(strikes_by_index: Dict[str, List[Dict[str, Any]]],
                          trading_date_ist: str,
                          *,
                          tier: str = "paid_intraday",
                          description: Optional[str] = None,
                          ) -> Dict[str, Any]:
    """Render + push the per-strike calls CSV.

    Returns the website's acceptance dict. Caller decides whether to
    swallow ``ArtifactPushError`` — the production brief generator
    wraps this in a try/except so a website outage doesn't kill the
    daily run.
    """
    body = options_strikes_csv(strikes_by_index, trading_date_ist)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"options-strikes-{trading_date_ist}.csv"
        p.write_text(body, encoding="utf-8")
        return push_artifact(
            path=p,
            kind="options_executor_calls_csv",
            tier=tier,
            trading_date_ist=trading_date_ist,
            description=description or (
                f"Per-strike executor decisions for IST trading "
                f"session {trading_date_ist}."
            ),
        )


def push_options_executor_audit(report: ExecutorAuditReport,
                                 *,
                                 tier: str = "free_signup",
                                 description: Optional[str] = None,
                                 ) -> Dict[str, Any]:
    """Render + push the executor audit CSV (Table A/B/SKIP/Gate-2).

    Default tier is ``free_signup`` because aggregate calibration is
    customer-facing trust material; the per-trade details that
    would expose levels stay on the paid tier via the
    options_executor_calls_csv path.
    """
    body = executor_audit_csv(report)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / (
            f"options-executor-audit-{report.trading_date_ist}.csv")
        p.write_text(body, encoding="utf-8")
        return push_artifact(
            path=p,
            kind="options_executor_audit_csv",
            tier=tier,
            trading_date_ist=report.trading_date_ist,
            description=description or (
                "Executor calibration tables (Table A non-conviction, "
                "Table B conviction-holds bucketed by LCS-at-hold, "
                "SKIP counter-factual, Gate-2 verdict) for IST "
                f"trading session {report.trading_date_ist}."
            ),
        )
