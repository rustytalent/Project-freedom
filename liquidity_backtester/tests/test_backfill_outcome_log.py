"""Smoke tests for the backfill script.

Full end-to-end testing against a real saved bundle happens on Codex's
side with the 710362b bundle. These tests pin the structural contract:

  * _trading_dates_in_bundle correctly walks asset frames.
  * run_backfill respects dry-run.
  * Resolution functions return ResolutionRecord-shaped objects with
    the documented resolution_method values.
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from analysis.backfill_outcome_log import (
    _resolve_avoidance,
    _resolve_proximity,
    _trading_dates_in_bundle,
    run_backfill,
)


def _bars(n: int = 80, base: float = 100.0,
          start_utc: str = "2026-05-26 03:45") -> pd.DataFrame:
    idx = pd.date_range(start_utc, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = base + np.cumsum(rng.normal(0.0, 0.05, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": np.full(n, 1000.0),
    }, index=idx)


@dataclass
class _AD:
    base_df: pd.DataFrame
    walkforward: Any = None


@dataclass
class _Report:
    assets: Dict[str, _AD] = field(default_factory=dict)


class TradingDatesTests(unittest.TestCase):

    def test_extracts_weekdays_from_bundle(self) -> None:
        # 2026-05-22 (Fri) -> 2026-05-26 (Tue): weekdays only
        df_a = _bars(start_utc="2026-05-22 03:45")
        df_b = _bars(start_utc="2026-05-26 03:45")
        report = _Report(assets={"X": _AD(base_df=df_a),
                                  "Y": _AD(base_df=df_b)})
        dates = _trading_dates_in_bundle(report)
        # Both fixtures span 8 days each (~80 bars 5min = 400min = 6.67h).
        # Returned dates should include weekdays only.
        for d in dates:
            wd = pd.Timestamp(d).weekday()
            self.assertLess(wd, 5,
                f"weekend date {d} (weekday {wd}) leaked into bundle dates")


class ProximityResolverTests(unittest.TestCase):

    def test_level_touched_returns_true(self) -> None:
        df = _bars(start_utc="2026-05-26 03:45")
        # Level just above session open -> definitely touched.
        max_high = float(df["high"].max())
        row = pd.Series({
            "prediction_id": "PID_X",
            "trading_date_ist": "2026-05-26",
            "target_description_json":
                ("{\"level\": " + str(max_high - 0.1)
                  + ", \"side_from_open\": \"above\"}"),
        })
        out = _resolve_proximity(row, df)
        self.assertIsNotNone(out)
        self.assertEqual(out.resolution_method, "level_touched")
        self.assertTrue(out.outcome_boolean)

    def test_level_not_touched_returns_false(self) -> None:
        df = _bars(start_utc="2026-05-26 03:45")
        # Level far above the session high -> not touched.
        row = pd.Series({
            "prediction_id": "PID_X",
            "trading_date_ist": "2026-05-26",
            "target_description_json":
                "{\"level\": 99999.0, \"side_from_open\": \"above\"}",
        })
        out = _resolve_proximity(row, df)
        self.assertEqual(out.resolution_method,
                          "level_not_touched_horizon_expired")
        self.assertFalse(out.outcome_boolean)

    def test_data_gap_when_no_bars_for_date(self) -> None:
        df = _bars(start_utc="2026-05-26 03:45")
        row = pd.Series({
            "prediction_id": "PID_X",
            "trading_date_ist": "2026-06-15",       # no bars here
            "target_description_json":
                "{\"level\": 100.0, \"side_from_open\": \"above\"}",
        })
        out = _resolve_proximity(row, df)
        self.assertEqual(out.resolution_method, "data_gap")
        self.assertTrue(out.had_data_gap)


class AvoidanceResolverTests(unittest.TestCase):

    def test_validated_when_at_least_one_asset_loses(self) -> None:
        # Construct one losing asset.
        n = 80
        idx = pd.date_range("2026-05-26 03:45", periods=n, freq="5min")
        # Strong down trend.
        close = 100.0 - np.arange(n) * 0.5
        df_loser = pd.DataFrame({
            "open": close, "high": close + 0.1, "low": close - 0.1,
            "close": close, "volume": np.full(n, 1000.0),
        }, index=idx)
        report = _Report(assets={"LOSER": _AD(base_df=df_loser)})
        out = _resolve_avoidance("ALL_BASKET", "2026-05-26", report)
        self.assertEqual(out.resolution_method, "avoidance_validated")

    def test_unnecessary_when_assets_are_fine(self) -> None:
        # Strongly trending up asset.
        n = 80
        idx = pd.date_range("2026-05-26 03:45", periods=n, freq="5min")
        close = 100.0 + np.arange(n) * 0.5
        df_winner = pd.DataFrame({
            "open": close, "high": close + 0.1, "low": close - 0.1,
            "close": close, "volume": np.full(n, 1000.0),
        }, index=idx)
        report = _Report(assets={"WINNER": _AD(base_df=df_winner)})
        out = _resolve_avoidance("ALL_BASKET", "2026-05-26", report)
        self.assertEqual(out.resolution_method, "avoidance_unnecessary")


class DryRunTests(unittest.TestCase):

    def test_dry_run_does_not_write(self) -> None:
        # Fake bundle file (we patch _load_bundle by writing a pickle
        # that produces the right shape).
        import pickle
        df = _bars(start_utc="2026-05-26 03:45")
        report = _Report(assets={"X": _AD(base_df=df)})
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "bundle.pkl"
            with open(bundle_path, "wb") as f:
                pickle.dump(report, f)
            out_root = Path(tmp) / "log"
            result = run_backfill(bundle_path, out_root,
                                    days=5, dry_run=True)
            self.assertIn("dates_planned", result)
            # No files written.
            self.assertFalse((out_root / "predictions").exists()
                              and any((out_root / "predictions").iterdir()))


if __name__ == "__main__":
    unittest.main()
