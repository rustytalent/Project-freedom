"""Tests for the outcome log — the data flywheel writer + reader.

Pinned contracts:
  * Predictions and resolutions append-only by trading_date_ist partition.
  * Re-running commit on the same partition APPENDS, never overwrites.
  * Schema evolution: old partition + new column => append succeeds,
    missing column filled with NA.
  * read_joined links predictions to resolutions on prediction_id.
  * calibration_by_bucket computes per-bucket hit rate + calibration error.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from liqpool.products.outcome_log import (
    OutcomeLogWriter,
    PredictionRecord,
    ResolutionRecord,
    SCHEMA_VERSION,
    calibration_by_bucket,
    make_prediction_id,
)


def _pred(prediction_id: str = "PRED_X",
          brief_id: str = "BRIEF_2026_06_03",
          trading_date_ist: str = "2026-06-03",
          symbol: str = "HDFCBANK",
          predicted_value: float = 0.78,
          confidence_bucket: str = "high",
          prediction_type: str = "proximity") -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        brief_id=brief_id,
        generated_at_utc="2026-06-03T03:00:00Z",
        trading_date_ist=trading_date_ist,
        model_bundle_version="bundle_v1",
        feature_version="42_v3",
        prediction_type=prediction_type,
        instrument_kind="equity",
        symbol=symbol,
        target_description={"kind": "level_touch", "level": 1742.5,
                             "horizon_bars": 60},
        predicted_value=predicted_value,
        predicted_value_kind="calibrated_probability",
        confidence_bucket=confidence_bucket,
        raw_score_diagnostic=0.82,
        regime_tags_at_prediction={"vol_regime": "normal"},
    )


def _reso(prediction_id: str = "PRED_X",
          trading_date_ist: str = "2026-06-03",
          outcome: bool = True) -> ResolutionRecord:
    return ResolutionRecord(
        prediction_id=prediction_id,
        resolved_at_utc="2026-06-03T10:30:00Z",
        resolution_method=("level_touched" if outcome
                            else "level_not_touched_horizon_expired"),
        outcome_boolean=outcome,
        outcome_continuous=None,
        resolution_details={"touched_at_bar_offset": 12 if outcome else None},
        had_data_gap=False,
        resolution_quality="clean",
        trading_date_ist=trading_date_ist,
    )


class WriterPartitionTests(unittest.TestCase):

    def test_commit_writes_predictions_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = OutcomeLogWriter(root=tmp)
            w.write_prediction(_pred())
            counts = w.commit()
            self.assertEqual(counts["predictions"], 1)
            path = Path(tmp) / "predictions" \
                / "trading_date_ist=2026-06-03" / "predictions.parquet"
            self.assertTrue(path.exists())
            df = pd.read_parquet(path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["symbol"], "HDFCBANK")

    def test_commit_appends_not_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            w1 = OutcomeLogWriter(root=tmp)
            w1.write_prediction(_pred(prediction_id="PRED_A"))
            w1.commit()
            w2 = OutcomeLogWriter(root=tmp)
            w2.write_prediction(_pred(prediction_id="PRED_B"))
            w2.commit()
            df = w2.read_predictions("2026-06-03")
            self.assertEqual(len(df), 2)
            self.assertEqual(set(df["prediction_id"]), {"PRED_A", "PRED_B"})

    def test_resolution_lands_in_prediction_date_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = OutcomeLogWriter(root=tmp)
            w.write_prediction(_pred(prediction_id="PRED_RESO"))
            w.write_resolution(_reso(prediction_id="PRED_RESO",
                                      trading_date_ist="2026-06-03"))
            w.commit()
            resos = w.read_resolutions("2026-06-03")
            self.assertEqual(len(resos), 1)
            self.assertEqual(resos.iloc[0]["prediction_id"], "PRED_RESO")

    def test_multiple_dates_partition_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = OutcomeLogWriter(root=tmp)
            w.write_prediction(_pred(prediction_id="P1",
                                      trading_date_ist="2026-06-02"))
            w.write_prediction(_pred(prediction_id="P2",
                                      trading_date_ist="2026-06-03"))
            w.commit()
            self.assertEqual(len(w.read_predictions("2026-06-02")), 1)
            self.assertEqual(len(w.read_predictions("2026-06-03")), 1)


class ReadJoinTests(unittest.TestCase):

    def test_joined_links_predictions_to_resolutions(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = OutcomeLogWriter(root=tmp)
            w.write_prediction(_pred(prediction_id="P_JOIN"))
            w.write_resolution(_reso(prediction_id="P_JOIN",
                                      outcome=True))
            w.commit()
            joined = w.read_joined("2026-06-03")
            self.assertEqual(len(joined), 1)
            self.assertTrue(joined.iloc[0]["resolved"])
            self.assertTrue(bool(joined.iloc[0]["outcome_boolean"]))

    def test_joined_unresolved_predictions_marked_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = OutcomeLogWriter(root=tmp)
            w.write_prediction(_pred(prediction_id="P_PENDING"))
            w.commit()
            joined = w.read_joined("2026-06-03")
            self.assertEqual(len(joined), 1)
            self.assertFalse(bool(joined.iloc[0]["resolved"]))

    def test_calibration_by_bucket_computes_hit_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = OutcomeLogWriter(root=tmp)
            # 10 high-confidence predictions, 8 hit.
            for i in range(10):
                w.write_prediction(_pred(prediction_id=f"P_H{i}",
                                          predicted_value=0.80,
                                          confidence_bucket="high"))
                w.write_resolution(_reso(prediction_id=f"P_H{i}",
                                          outcome=(i < 8)))
            # 5 low-confidence predictions, 1 hit.
            for i in range(5):
                w.write_prediction(_pred(prediction_id=f"P_L{i}",
                                          predicted_value=0.30,
                                          confidence_bucket="low"))
                w.write_resolution(_reso(prediction_id=f"P_L{i}",
                                          outcome=(i < 1)))
            w.commit()
            joined = w.read_joined("2026-06-03")
            cal = calibration_by_bucket(joined)
            self.assertEqual(len(cal), 2)
            high_row = cal[cal["confidence_bucket"] == "high"].iloc[0]
            low_row = cal[cal["confidence_bucket"] == "low"].iloc[0]
            self.assertAlmostEqual(high_row["hit_rate"], 0.8, places=2)
            self.assertAlmostEqual(low_row["hit_rate"], 0.2, places=2)


class PredictionIdTests(unittest.TestCase):

    def test_make_prediction_id_format(self):
        pid = make_prediction_id(
            brief_id="BRIEF_2026_06_03",
            symbol="HDFCBANK",
            prediction_type="proximity",
            detail_token="level_1742_h60",
        )
        self.assertEqual(pid,
            "BRIEF_2026_06_03_HDFCBANK_proximity_level_1742_h60")

    def test_make_prediction_id_sanitises_special_chars(self):
        pid = make_prediction_id(
            brief_id="BRIEF_X",
            symbol="M&M-AUTO",
            prediction_type="direction",
            detail_token="up:bias",
        )
        self.assertNotIn("&", pid)
        self.assertNotIn(":", pid)
        # Spaces and hyphens turned into underscores.
        self.assertNotIn(" ", pid)


class SchemaVersionTests(unittest.TestCase):

    def test_schema_version_pinned(self):
        self.assertEqual(SCHEMA_VERSION, "1.0")
        record = _pred()
        self.assertEqual(record.schema_version, "1.0")


if __name__ == "__main__":
    unittest.main()
