"""Smoke tests for the artifact packer.

Pinned contracts:
  * Missing source files don't crash — they show as None in the bundle.
  * JSON sources load into JSON bundles; CSV sources convert to lists.
  * MAX_ROWS_PER_FILE caps row counts on the bloat-prone files.
  * Pass-through files (daily_brief.*) are copied verbatim.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.pack_artifacts import (
    MAX_ROWS_PER_FILE,
    PASS_THROUGH,
    pack,
    pack_customer_delivery,
)
from liqpool.products.outcome_log import (
    OutcomeLogWriter,
    PredictionRecord,
    ResolutionRecord,
)


def _brief_fixture() -> dict:
    return {
        "schema_version": "1.0",
        "brief_metadata": {
            "brief_id": "BRIEF_2026_06_04",
            "trading_date_ist": "2026-06-04",
            "generated_at_utc": "2026-06-04T03:00:00Z",
            "model_bundle_version": "test_bundle@abc123",
            "feature_version": "42_features_v3",
            "session_status": "pre_open",
            "reading_time_minutes": 6,
            "indexes_covered": ["NIFTY50"],
        },
        "index_regime": {"_status": "pending", "_reason": "fixture"},
        "sector_regime": {
            "as_of_ist": "2026-06-04T09:15:00+05:30",
            "trending_up": ["BANKING"],
            "trending_down": [],
            "chopping": [],
            "neutral": ["IT"],
            "leadership_change_vs_yesterday": [],
        },
        "top_watchlist": [{
            "symbol": "HDFCBANK",
            "sector": "BANKING",
            "side": "up",
            "reason_tag": "proximity_high_p_touch_journey",
            "key_level": 1742.5,
            "key_level_type": "demand_pool",
            "p_touch_today": 0.74,
            "p_touch_within_60min": 0.41,
            "model_confidence_bucket": "high",
            "regime_tags": ["banking_trending_up"],
            "avoidance_note": None,
        }],
        "options_suitability": {"_status": "pending", "_reason": "fixture"},
        "avoid_list": [],
        "key_zones": [{
            "symbol": "BANKNIFTY",
            "level": 52000.0,
            "level_type": "supply_pool",
            "side_from_open": "above",
            "p_test_today": 0.66,
            "p_test_within_60min": 0.22,
            "horizons_evaluated": [12, 36, 60],
            "p_per_horizon": {"12": 0.22, "36": 0.51, "60": 0.66},
            "confluence_with_poc": False,
            "confluence_with_vah_val": False,
            "model_confidence_bucket": "moderate",
        }],
        "confidence_notes": {
            "calibrated_today": ["proximity_h60"],
            "drifting_today": [],
            "drift_reason": {},
            "overall_brief_confidence": "moderate",
            "operator_note": None,
        },
        "yesterday_audit": {
            "yesterday_brief_id": "BRIEF_2026_06_03",
            "predictions_made": 1,
            "predictions_resolved": 1,
            "hit_rate_by_confidence_bucket": [],
        },
    }


class PackArtifactsTests(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_missing_inputs_produce_none_entries(self):
        # Empty input dir -> all bundles exist but have all-None entries.
        out_dir = self.input_dir / "packed"
        written = pack(self.input_dir, out_dir)
        self.assertIn("consolidated_summary", written)
        summary = json.loads(written["consolidated_summary"].read_text())
        # Every key should map to None for an empty input.
        for k, v in summary.items():
            self.assertIsNone(v, f"key {k} should be None on empty input")

    def test_json_source_round_trips(self):
        (self.input_dir / "multi_asset_summary.json").write_text(
            json.dumps({"pooled_oos_broad": 0.504, "n_pools": 10954}))
        out_dir = self.input_dir / "packed"
        written = pack(self.input_dir, out_dir)
        summary = json.loads(written["consolidated_summary"].read_text())
        self.assertEqual(summary["multi_asset_summary"]["pooled_oos_broad"],
                          0.504)
        self.assertEqual(summary["multi_asset_summary"]["n_pools"], 10954)

    def test_csv_source_round_trips(self):
        df = pd.DataFrame({
            "model": ["global", "blended"],
            "auc": [0.538, 0.540],
        })
        df.to_csv(self.input_dir / "phase3_oos_prediction_audit.csv",
                   index=False)
        out_dir = self.input_dir / "packed"
        written = pack(self.input_dir, out_dir)
        models = json.loads(written["consolidated_models"].read_text())
        rows = models["phase3_oos_prediction_audit"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["model"], "global")
        self.assertAlmostEqual(rows[0]["auc"], 0.538)

    def test_row_cap_applies_to_bloat_files(self):
        # reaction_alerts has MAX_ROWS_PER_FILE = 300.
        cap = MAX_ROWS_PER_FILE["reaction_alerts"]
        df = pd.DataFrame({"i": range(cap * 3)})
        df.to_csv(self.input_dir / "reaction_alerts.csv", index=False)
        out_dir = self.input_dir / "packed"
        written = pack(self.input_dir, out_dir)
        bt = json.loads(written["consolidated_backtest"].read_text())
        self.assertEqual(len(bt["reaction_alerts"]), cap)

    def test_pass_through_daily_brief_files_copied(self):
        (self.input_dir / "daily_brief.json").write_text('{"x": 1}')
        (self.input_dir / "daily_brief.txt").write_text("Daily Research Brief")
        out_dir = self.input_dir / "packed"
        written = pack(self.input_dir, out_dir)
        for name in PASS_THROUGH:
            self.assertIn(name, written)
            self.assertTrue((out_dir / name).exists())

    def test_output_defaults_to_input_dir(self):
        # When --output is omitted, files land in --input.
        written = pack(self.input_dir, output_dir=None)
        for name in ("consolidated_summary", "consolidated_models",
                      "consolidated_backtest"):
            self.assertTrue(
                written[name].parent.resolve() == self.input_dir.resolve()
            )

    def test_three_consolidated_bundles_always_written(self):
        # Even with zero input files, the three consolidated bundles exist.
        out_dir = self.input_dir / "packed"
        written = pack(self.input_dir, out_dir)
        for name in ("consolidated_summary", "consolidated_models",
                      "consolidated_backtest"):
            self.assertIn(name, written)
            self.assertTrue(written[name].exists())

    def test_customer_delivery_licensed_keeps_exact_brief_levels(self):
        (self.input_dir / "daily_brief.json").write_text(
            json.dumps(_brief_fixture()))
        out_dir = self.input_dir / "licensed_pack"
        written = pack_customer_delivery(
            self.input_dir,
            out_dir,
            customer_id="cust-paid",
            customer_tier="licensed",
        )
        self.assertIn("manifest", written)
        brief = json.loads((out_dir / "daily_brief.json").read_text())
        self.assertEqual(brief["top_watchlist"][0]["key_level"], 1742.5)
        manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["reference_zone_policy"], "exact")
        self.assertNotIn("manifest.json", manifest["files"])

    def test_customer_delivery_demo_abstracts_brief_levels_and_text(self):
        (self.input_dir / "daily_brief.json").write_text(
            json.dumps(_brief_fixture()))
        out_dir = self.input_dir / "demo_pack"
        pack_customer_delivery(
            self.input_dir,
            out_dir,
            customer_id="cust-demo",
            customer_tier="demo",
        )
        brief = json.loads((out_dir / "daily_brief.json").read_text())
        self.assertNotEqual(brief["top_watchlist"][0]["key_level"], 1742.5)
        self.assertNotEqual(brief["key_zones"][0]["level"], 52000.0)
        txt = (out_dir / "daily_brief.txt").read_text()
        self.assertNotIn("1742.50", txt)
        manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["reference_zone_policy"], "abstracted")

    def test_customer_delivery_summarizes_outcome_log(self):
        (self.input_dir / "daily_brief.json").write_text(
            json.dumps(_brief_fixture()))
        log_root = self.input_dir / "outcome_log"
        writer = OutcomeLogWriter(root=str(log_root))
        pred = PredictionRecord(
            prediction_id="p1",
            brief_id="BRIEF_2026_06_03",
            generated_at_utc="2026-06-03T03:00:00Z",
            trading_date_ist="2026-06-03",
            model_bundle_version="bundle@test",
            feature_version="42_features_v3",
            prediction_type="proximity",
            instrument_kind="equity",
            symbol="HDFCBANK",
            target_description={"kind": "level_touch"},
            predicted_value=0.70,
            predicted_value_kind="calibrated_probability",
            confidence_bucket="high",
            raw_score_diagnostic=0.71,
            regime_tags_at_prediction={},
            is_retrospective=True,
        )
        res = ResolutionRecord(
            prediction_id="p1",
            resolved_at_utc="2026-06-03T10:00:00Z",
            resolution_method="level_touched",
            outcome_boolean=True,
            outcome_continuous=None,
            resolution_details={},
            had_data_gap=False,
            resolution_quality="clean",
            trading_date_ist="2026-06-03",
        )
        writer.write_prediction(pred)
        writer.write_resolution(res)
        writer.commit()

        out_dir = self.input_dir / "pack_with_log"
        pack_customer_delivery(
            self.input_dir,
            out_dir,
            outcome_log_root=log_root,
            customer_id="cust-paid",
            customer_tier="licensed",
        )
        summary = json.loads((out_dir / "outcome_log_summary.json").read_text())
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["predictions"], 1)
        self.assertEqual(summary["resolved"], 1)
        self.assertEqual(summary["retrospective_share"], 1.0)
        self.assertTrue(summary["is_retrospective_calibration"])


if __name__ == "__main__":
    unittest.main()
