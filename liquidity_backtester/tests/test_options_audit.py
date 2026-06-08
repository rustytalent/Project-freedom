"""Tests for the options executor audit (Stream D.6).

Pinned contracts:

  1. Empty input → empty report (no crash).
  2. Table A aggregates non-conviction-hold trades by bucket: n,
     mean R, win rate, exit-action histogram.
  3. Table B aggregates conviction-hold trades and computes
     pct_recovered_to_profit + sub-buckets by LCS-at-hold strength.
  4. SKIP counter-factual: n, mean R the skipped trades would have
     returned, pct that would have been profitable.
  5. Gate-2 verdict: passes iff (mean_conviction_r − mean_naive_r) ≥
     +0.20. Set to False when conviction outcomes exist but naive
     baseline is missing.
  6. Retrospective share carries through (Stream G) when the joined
     frame has ``is_retrospective`` column.
  7. ``audit_to_yesterday_audit_payload`` returns a JSON-shaped dict
     suitable for embedding in the brief's Yesterday Audit section.
"""
from __future__ import annotations

import json
import unittest
from typing import Dict, List

import pandas as pd

from liqpool.options.audit import (
    OPTIONS_EXECUTOR_PREDICTION_TYPE,
    OPTIONS_SKIP_PREDICTION_TYPE,
    audit_to_yesterday_audit_payload,
    compute_executor_audit,
    options_executor_meta,
    options_skip_meta,
)


# ---------------------------------------------------------------------------
# Helpers — build a synthetic joined frame
# ---------------------------------------------------------------------------

def _exec_row(*, bucket_key: str, realized_r: float,
              naive_exit_r: float, was_conviction_hold: bool = False,
              exit_action: str = "EXIT_TARGET",
              lcs_at_entry: float = 0.5, lcs_at_exit: float = 0.5,
              min_lcs: float = 0.3, min_realized_r: float = 0.0,
              predicted_target_r: float = 1.5,
              agreeing_at_entry: int = 4, agreeing_at_exit: int = 4,
              is_retrospective: bool = False) -> Dict:
    meta = options_executor_meta(
        bucket_key=bucket_key,
        lcs_at_entry=lcs_at_entry, lcs_at_exit=lcs_at_exit,
        agreeing_layers_at_entry=agreeing_at_entry,
        agreeing_layers_at_exit=agreeing_at_exit,
        min_lcs_during_trade=min_lcs,
        min_realized_r_during_trade=min_realized_r,
        realized_r_at_exit_atr=realized_r,
        naive_exit_r_atr=naive_exit_r,
        was_conviction_hold=was_conviction_hold,
        exit_action=exit_action,
        predicted_target_r_atr=predicted_target_r,
    )
    return {
        "prediction_type": OPTIONS_EXECUTOR_PREDICTION_TYPE,
        "regime_tags_json": json.dumps(meta),
        "is_retrospective": is_retrospective,
    }


def _skip_row(*, bucket_key: str, counterfactual_r: float,
              lcs_at_decision: float = 0.05,
              skip_reason: str = "macro_gate_closed",
              is_retrospective: bool = False) -> Dict:
    meta = options_skip_meta(
        bucket_key=bucket_key,
        lcs_at_decision=lcs_at_decision,
        counterfactual_r_atr=counterfactual_r,
        skip_reason=skip_reason,
    )
    return {
        "prediction_type": OPTIONS_SKIP_PREDICTION_TYPE,
        "regime_tags_json": json.dumps(meta),
        "is_retrospective": is_retrospective,
    }


def _frame(rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class EmptyInputTests(unittest.TestCase):

    def test_none_input(self):
        report = compute_executor_audit(None, "2026-06-09")
        self.assertEqual(report.n_total, 0)
        self.assertEqual(report.table_a, [])

    def test_empty_frame(self):
        report = compute_executor_audit(pd.DataFrame(), "2026-06-09")
        self.assertEqual(report.n_total, 0)

    def test_frame_without_prediction_type_column(self):
        report = compute_executor_audit(
            pd.DataFrame([{"foo": 1}]), "2026-06-09")
        self.assertEqual(report.n_total, 0)


class TableATests(unittest.TestCase):

    def test_aggregates_per_bucket_with_win_rate(self):
        # Three non-conviction trades in one bucket: +1.0, +0.8, -0.5.
        # win rate = 2/3 ≈ 0.667; mean R = 0.433.
        rows = [
            _exec_row(bucket_key="buy/weekly/open",
                       realized_r=+1.0, naive_exit_r=+0.7,
                       exit_action="EXIT_TARGET"),
            _exec_row(bucket_key="buy/weekly/open",
                       realized_r=+0.8, naive_exit_r=+0.6,
                       exit_action="EXIT_TARGET"),
            _exec_row(bucket_key="buy/weekly/open",
                       realized_r=-0.5, naive_exit_r=-0.3,
                       exit_action="EXIT_INVALIDATION"),
        ]
        report = compute_executor_audit(_frame(rows), "2026-06-09")
        self.assertEqual(len(report.table_a), 1)
        row = report.table_a[0]
        self.assertEqual(row.bucket, "buy/weekly/open")
        self.assertEqual(row.n, 3)
        self.assertAlmostEqual(row.mean_realized_r_atr, 0.4333, places=3)
        self.assertAlmostEqual(row.win_rate, 2 / 3, places=3)
        self.assertEqual(row.by_exit_action,
                          {"EXIT_TARGET": 2, "EXIT_INVALIDATION": 1})

    def test_conviction_holds_are_excluded_from_table_a(self):
        rows = [
            _exec_row(bucket_key="b1", realized_r=+0.5, naive_exit_r=+0.1),
            _exec_row(bucket_key="b1", realized_r=+0.5, naive_exit_r=+0.1,
                       was_conviction_hold=True),
        ]
        report = compute_executor_audit(_frame(rows), "2026-06-09")
        # Only one trade should land in Table A.
        self.assertEqual(len(report.table_a), 1)
        self.assertEqual(report.table_a[0].n, 1)


class TableBTests(unittest.TestCase):

    def test_conviction_hold_recovery_stats(self):
        # Three conviction-holds: two recovered, one didn't.
        rows = [
            _exec_row(bucket_key="buy/weekly/open",
                       realized_r=+0.6, naive_exit_r=-0.3,
                       was_conviction_hold=True,
                       min_realized_r=-0.8, min_lcs=0.45),  # moderate
            _exec_row(bucket_key="buy/weekly/open",
                       realized_r=+0.4, naive_exit_r=-0.4,
                       was_conviction_hold=True,
                       min_realized_r=-0.6, min_lcs=0.55),  # high
            _exec_row(bucket_key="buy/weekly/open",
                       realized_r=-0.9, naive_exit_r=-0.4,
                       was_conviction_hold=True,
                       min_realized_r=-1.2, min_lcs=0.40),  # moderate
        ]
        report = compute_executor_audit(_frame(rows), "2026-06-09")
        tb = report.table_b
        self.assertEqual(tb.n_total, 3)
        self.assertAlmostEqual(tb.mean_realized_r_atr,
                                (0.6 + 0.4 - 0.9) / 3, places=4)
        # Two of three recovered to positive.
        self.assertAlmostEqual(tb.pct_recovered_to_profit, 2 / 3, places=3)
        # Each sub-bucket has the expected count.
        bucket_to_n = {r.lcs_at_hold_bucket: r.n for r in tb.by_lcs_at_hold}
        self.assertEqual(bucket_to_n.get("high"), 1)
        self.assertEqual(bucket_to_n.get("moderate"), 2)


class SkipCounterfactualTests(unittest.TestCase):

    def test_aggregates_skipped_counterfactuals(self):
        rows = [
            _skip_row(bucket_key="buy/weekly/open",
                       counterfactual_r=-0.4),
            _skip_row(bucket_key="buy/weekly/open",
                       counterfactual_r=+0.1),
            _skip_row(bucket_key="buy/weekly/open",
                       counterfactual_r=-0.7),
        ]
        report = compute_executor_audit(_frame(rows), "2026-06-09")
        sk = report.skip_counterfactual
        self.assertEqual(sk.n, 3)
        self.assertAlmostEqual(sk.mean_counterfactual_r_atr,
                                (-0.4 + 0.1 - 0.7) / 3, places=4)
        self.assertAlmostEqual(sk.pct_would_have_been_profitable,
                                1 / 3, places=3)


class Gate2Tests(unittest.TestCase):

    def test_passes_when_conviction_beats_naive_by_plus_0_20(self):
        # Conviction hold mean = +0.50; naive mean = +0.20; delta +0.30.
        rows = [
            _exec_row(bucket_key="b1", realized_r=+0.5, naive_exit_r=+0.2,
                       was_conviction_hold=True,
                       min_realized_r=-0.4, min_lcs=0.55),
            _exec_row(bucket_key="b1", realized_r=+0.5, naive_exit_r=+0.2,
                       was_conviction_hold=True,
                       min_realized_r=-0.3, min_lcs=0.55),
        ]
        report = compute_executor_audit(_frame(rows), "2026-06-09")
        self.assertIsNotNone(report.gate2_pass)
        self.assertTrue(report.gate2_pass)
        self.assertAlmostEqual(
            report.gate2_naive_baseline_mean_r, 0.20, places=3)
        self.assertAlmostEqual(
            report.gate2_conviction_minus_naive, 0.30, places=3)

    def test_fails_when_conviction_below_naive_plus_0_20(self):
        # Conviction +0.10, naive +0.05 → delta +0.05 < 0.20.
        rows = [
            _exec_row(bucket_key="b1", realized_r=+0.1, naive_exit_r=+0.05,
                       was_conviction_hold=True,
                       min_realized_r=-0.2, min_lcs=0.50),
        ]
        report = compute_executor_audit(_frame(rows), "2026-06-09")
        self.assertFalse(report.gate2_pass)


class RetrospectiveShareTests(unittest.TestCase):

    def test_share_carries_through(self):
        # 2 retrospective, 1 live.
        rows = [
            _exec_row(bucket_key="b1", realized_r=+0.5,
                       naive_exit_r=+0.1, is_retrospective=True),
            _exec_row(bucket_key="b1", realized_r=+0.3,
                       naive_exit_r=+0.1, is_retrospective=True),
            _exec_row(bucket_key="b1", realized_r=+0.4,
                       naive_exit_r=+0.1, is_retrospective=False),
        ]
        report = compute_executor_audit(_frame(rows), "2026-06-09")
        self.assertAlmostEqual(report.retrospective_share, 2 / 3, places=3)
        self.assertTrue(report.is_retrospective_calibration)


class PayloadSerializationTests(unittest.TestCase):

    def test_round_trip_to_dict(self):
        rows = [
            _exec_row(bucket_key="buy/weekly/open",
                       realized_r=+0.5, naive_exit_r=+0.3),
            _exec_row(bucket_key="buy/weekly/open",
                       realized_r=+0.4, naive_exit_r=+0.2,
                       was_conviction_hold=True,
                       min_realized_r=-0.3, min_lcs=0.55),
            _skip_row(bucket_key="buy/weekly/open",
                       counterfactual_r=-0.2),
        ]
        report = compute_executor_audit(_frame(rows), "2026-06-09")
        payload = audit_to_yesterday_audit_payload(report)
        self.assertIn("options_executor", payload)
        block = payload["options_executor"]
        self.assertEqual(block["trading_date_ist"], "2026-06-09")
        self.assertEqual(block["n_total"], 3)
        # JSON-serialisable.
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
