"""Tests for the options artifact exporters (Stream D.7).

Pinned contracts:

  1. options_strikes_csv has the documented header and emits one row
     per (index, strike) — order preserved, missing values blank.
  2. executor_audit_csv emits Table A / Table B summary /
     Table B sub-bucket / SKIP / Gate-2 sections in order. The
     section column tags row type; numeric cells are 4dp strings.
  3. push_options_strikes forwards the correct kind + tier and
     surfaces the website's acceptance dict.
  4. push_options_executor_audit forwards the correct kind + tier.
  5. The new artifact kinds are present in the pusher's VALID_KINDS.
"""
from __future__ import annotations

import csv
import io
import os
import unittest
from typing import Any, Dict, List
from unittest import mock

from liqpool.options.artifact_export import (
    executor_audit_csv,
    options_strikes_csv,
    push_options_executor_audit,
    push_options_strikes,
)
from liqpool.options.audit import (
    ExecutorAuditReport,
    SkipCounterfactual,
    TableARow,
    TableB,
    TableBLcsBucketRow,
)
from liqpool.products.artifact_pusher import VALID_KINDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _strikes_payload() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "NIFTY50": [
            {
                "strike": 24500.0,
                "p_test_today": 0.78,
                "p_test_within_60min": 0.34,
                "side_from_open": "below",
                "key_level_type": "demand_pool",
                "underlying_level": 24512.0,
                "predicted_net_return_buy_atr": -0.45,
                "predicted_net_return_sell_atr": +0.30,
                "executor_decision_buy": "WAIT",
                "executor_decision_sell": "SKIP",
            },
            {
                "strike": 24700.0,
                "p_test_today": 0.45,
                "p_test_within_60min": 0.18,
                "side_from_open": "above",
                "key_level_type": "supply_pool",
                "underlying_level": 24512.0,
                "predicted_net_return_buy_atr": None,
                "predicted_net_return_sell_atr": None,
                "executor_decision_buy": None,
                "executor_decision_sell": None,
            },
        ],
    }


def _audit_report() -> ExecutorAuditReport:
    return ExecutorAuditReport(
        trading_date_ist="2026-06-09",
        n_total=5,
        table_a=[
            TableARow(
                bucket="buy/weekly/open",
                n=3,
                mean_realized_r_atr=+0.33,
                win_rate=0.667,
                by_exit_action={"EXIT_TARGET": 2, "EXIT_INVALIDATION": 1},
            ),
        ],
        table_b=TableB(
            n_total=2,
            mean_realized_r_atr=+0.20,
            pct_recovered_to_profit=0.50,
            by_lcs_at_hold=[
                TableBLcsBucketRow(
                    lcs_at_hold_bucket="high",
                    n=1,
                    pct_recovered_to_profit=1.0,
                    mean_realized_r_atr=+0.60,
                    pct_lost_more=0.0,
                ),
            ],
        ),
        skip_counterfactual=SkipCounterfactual(
            n=4, mean_counterfactual_r_atr=-0.10,
            pct_would_have_been_profitable=0.25,
        ),
        gate2_naive_baseline_mean_r=0.05,
        gate2_conviction_minus_naive=0.15,
        gate2_pass=False,
        is_retrospective_calibration=True,
        retrospective_share=0.6,
    )


def _parse(text: str) -> List[List[str]]:
    return [row for row in csv.reader(io.StringIO(text)) if row]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class StrikesCsvTests(unittest.TestCase):

    def test_header_and_row_shape(self):
        text = options_strikes_csv(_strikes_payload(), "2026-06-09")
        rows = _parse(text)
        self.assertEqual(rows[0][0], "trading_date_ist")
        self.assertEqual(rows[0][1], "index_name")
        self.assertEqual(rows[0][2], "strike")
        # Two data rows (one strike each).
        data_rows = rows[1:]
        self.assertEqual(len(data_rows), 2)
        # First row's index name + strike.
        self.assertEqual(data_rows[0][0], "2026-06-09")
        self.assertEqual(data_rows[0][1], "NIFTY50")
        self.assertEqual(data_rows[0][2], "24500.0")
        # Second row has empty predicted_R and executor columns.
        self.assertEqual(data_rows[1][8], "")  # predicted_net_return_buy_atr
        self.assertEqual(data_rows[1][9], "")  # predicted_net_return_sell_atr
        self.assertEqual(data_rows[1][10], "")  # executor_decision_buy
        self.assertEqual(data_rows[1][11], "")  # executor_decision_sell


class ExecutorAuditCsvTests(unittest.TestCase):

    def test_sections_emitted_in_order(self):
        text = executor_audit_csv(_audit_report())
        rows = _parse(text)
        # Section column is at index 1.
        sections = [r[1] for r in rows[1:]]
        # Expected order: table_a × N + table_b summary + table_b_bucket × M
        # + skip_counterfactual + gate2.
        self.assertEqual(sections[0], "table_a")
        self.assertEqual(sections[1], "table_b")
        self.assertEqual(sections[2], "table_b_bucket")
        self.assertEqual(sections[3], "skip_counterfactual")
        self.assertEqual(sections[4], "gate2")

    def test_gate2_row_carries_verdict(self):
        text = executor_audit_csv(_audit_report())
        rows = _parse(text)
        gate2 = [r for r in rows if r and r[1] == "gate2"][0]
        # naive_baseline_mean_r at index 11
        self.assertEqual(gate2[11], "0.0500")
        # conviction_minus_naive at index 12
        self.assertEqual(gate2[12], "0.1500")
        # gate2_pass at index 13 → "0" since the report has pass=False
        self.assertEqual(gate2[13], "0")

    def test_retrospective_share_in_every_row(self):
        text = executor_audit_csv(_audit_report())
        rows = _parse(text)
        for r in rows[1:]:
            # is_retrospective_calibration at index 14, share at 15
            self.assertEqual(r[14], "1")
            self.assertEqual(r[15], "0.6000")


class PushWrapperTests(unittest.TestCase):

    @mock.patch.dict(os.environ, {
        "WEBSITE_BASE_URL": "https://example.test",
        "ENGINE_INGEST_TOKEN": "secret",
    })
    @mock.patch("liqpool.options.artifact_export.push_artifact")
    def test_push_strikes_forwards_kind_tier_date(self, fake_push):
        fake_push.return_value = {"accepted": True, "id": "x"}
        result = push_options_strikes(
            _strikes_payload(), trading_date_ist="2026-06-09")
        self.assertEqual(result["accepted"], True)
        kwargs = fake_push.call_args.kwargs
        self.assertEqual(kwargs["kind"], "options_executor_calls_csv")
        self.assertEqual(kwargs["tier"], "paid_intraday")
        self.assertEqual(kwargs["trading_date_ist"], "2026-06-09")

    @mock.patch.dict(os.environ, {
        "WEBSITE_BASE_URL": "https://example.test",
        "ENGINE_INGEST_TOKEN": "secret",
    })
    @mock.patch("liqpool.options.artifact_export.push_artifact")
    def test_push_audit_forwards_kind_tier_date(self, fake_push):
        fake_push.return_value = {"accepted": True, "id": "y"}
        result = push_options_executor_audit(_audit_report())
        self.assertEqual(result["accepted"], True)
        kwargs = fake_push.call_args.kwargs
        self.assertEqual(kwargs["kind"], "options_executor_audit_csv")
        # Audit ships at free_signup tier per the calibration trust
        # framing (aggregate, not actionable).
        self.assertEqual(kwargs["tier"], "free_signup")
        self.assertEqual(kwargs["trading_date_ist"], "2026-06-09")


class ValidKindsTests(unittest.TestCase):

    def test_new_kinds_present(self):
        self.assertIn("options_executor_audit_csv", VALID_KINDS)
        self.assertIn("options_executor_calls_csv", VALID_KINDS)


if __name__ == "__main__":
    unittest.main()
