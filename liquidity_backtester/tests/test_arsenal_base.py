"""Tests for liqpool.arsenal.base — AlphaSignal + Alpha contract."""
from __future__ import annotations

import unittest

import pandas as pd
import pytest

from liqpool.arsenal.base import (
    Alpha,
    AlphaSignal,
    _bucket_minute,
    signals_to_frame,
)


class AlphaSignalValidationTests(unittest.TestCase):
    def _kwargs(self, **overrides):
        defaults = dict(
            alpha_name="x", symbol="HDFCBANK",
            decision_at=pd.Timestamp("2026-05-26 04:30"),
            decision_idx=5, side="long",
            entry_reference=100.0, stop_atr=0.5, target_atr=2.0,
            horizon_bars=24, confidence=0.5,
        )
        defaults.update(overrides)
        return defaults

    def test_valid_signal_constructs(self):
        sig = AlphaSignal(**self._kwargs())
        self.assertEqual(sig.side, "long")
        self.assertEqual(sig.stop_atr, 0.5)

    def test_invalid_side_raises(self):
        with self.assertRaises(ValueError):
            AlphaSignal(**self._kwargs(side="up"))

    def test_non_positive_barriers_raise(self):
        with self.assertRaises(ValueError):
            AlphaSignal(**self._kwargs(stop_atr=0.0))
        with self.assertRaises(ValueError):
            AlphaSignal(**self._kwargs(target_atr=-1.0))
        with self.assertRaises(ValueError):
            AlphaSignal(**self._kwargs(horizon_bars=0))

    def test_confidence_outside_unit_interval_raises(self):
        with self.assertRaises(ValueError):
            AlphaSignal(**self._kwargs(confidence=1.5))
        with self.assertRaises(ValueError):
            AlphaSignal(**self._kwargs(confidence=-0.1))


class BucketMinuteTests(unittest.TestCase):
    """The session bucketing helper used in default regime tags."""

    def test_pre_open(self):
        self.assertEqual(_bucket_minute(8 * 60), "pre_open")

    def test_open_hour(self):
        self.assertEqual(_bucket_minute(9 * 60 + 30), "open_hour")
        self.assertEqual(_bucket_minute(10 * 60 + 14), "open_hour")

    def test_morning(self):
        self.assertEqual(_bucket_minute(10 * 60 + 15), "morning")
        self.assertEqual(_bucket_minute(11 * 60 + 59), "morning")

    def test_midday(self):
        self.assertEqual(_bucket_minute(12 * 60), "midday")
        self.assertEqual(_bucket_minute(13 * 60 + 29), "midday")

    def test_afternoon(self):
        self.assertEqual(_bucket_minute(13 * 60 + 30), "afternoon")
        self.assertEqual(_bucket_minute(15 * 60 + 30), "afternoon")

    def test_post_close(self):
        self.assertEqual(_bucket_minute(15 * 60 + 31), "post_close")
        self.assertEqual(_bucket_minute(20 * 60), "post_close")


class DefaultRegimeTagsTests(unittest.TestCase):
    """Verify the default Alpha.regime_tags wiring without instantiating
    a concrete alpha (use a minimal subclass for the test)."""

    class _Dummy(Alpha):
        @property
        def name(self):
            return "dummy"
        def candidates(self, *, symbol, df_base, atr_series, extra=None):
            return []

    def test_default_tags_include_session_and_side(self):
        # 04:30 UTC = 10:00 IST = "opening_hour" via nse_session (10:15 cutoff).
        # ist_minute_of_day_bucket boundaries here: <10:15 = open_hour.
        sig = AlphaSignal(
            alpha_name="dummy", symbol="X", side="short",
            decision_at=pd.Timestamp("2026-05-26 04:30"),
            decision_idx=0, entry_reference=100.0,
            stop_atr=0.5, target_atr=2.0, horizon_bars=12,
        )
        tags = self._Dummy().regime_tags(sig, pd.DataFrame(), pd.Series())
        self.assertEqual(tags["side"], "short")
        self.assertEqual(tags["session"], "opening_hour")
        self.assertEqual(tags["ist_minute_of_day_bucket"], "open_hour")


class SignalsToFrameTests(unittest.TestCase):
    def test_empty_input_returns_empty_frame(self):
        df = signals_to_frame([])
        self.assertTrue(df.empty)

    def test_columns_are_stable(self):
        sigs = [AlphaSignal(
            alpha_name="x", symbol="A",
            decision_at=pd.Timestamp("2026-05-26 04:30"),
            decision_idx=1, side="long",
            entry_reference=100.0, stop_atr=0.5, target_atr=2.0,
            horizon_bars=12,
        )]
        df = signals_to_frame(sigs)
        for col in ("alpha_name", "symbol", "decision_at", "decision_idx",
                    "side", "entry_reference", "stop_atr", "target_atr",
                    "horizon_bars", "confidence"):
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
