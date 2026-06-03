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
)


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


if __name__ == "__main__":
    unittest.main()
