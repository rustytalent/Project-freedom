"""Tests for the Kite warehouse reader.

Builds a synthetic mini-warehouse in a temp dir matching the EXACT
layout the user documented, then verifies each loader validates schema,
sorts by time, bridges timezone, keeps daily/intraday separate, and the
lookahead-safety helpers behave.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from liqpool.warehouse import (
    WarehouseReader,
    WarehouseFileMissing,
    WarehouseSchemaError,
    WarehouseError,
    align_daily_to_intraday,
    assert_sorted_by_time,
    causal_rolling,
)


# ---------------------------------------------------------------------------
# Synthetic warehouse builder
# ---------------------------------------------------------------------------

def _bars_ist(n: int, start="2024-01-01 09:15", freq="1min",
              base=100.0, with_oi=False) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq, tz="Asia/Kolkata")
    rng = np.random.default_rng(0)
    close = base + np.cumsum(rng.normal(0, 0.1, n))
    df = pd.DataFrame({
        "timestamp": idx,
        "open": close, "high": close + 0.2, "low": close - 0.2,
        "close": close, "volume": rng.integers(100, 1000, n),
        "symbol": "TESTSYM", "instrument_token": 12345,
    })
    if with_oi:
        df["open_interest"] = 0
    return df


def _build_warehouse(root: Path) -> None:
    # Layer 1: raw_1m
    (root / "raw_1m").mkdir(parents=True)
    _bars_ist(120).assign(symbol="HDFCBANK").to_parquet(
        root / "raw_1m" / "HDFCBANK_1m.parquet")

    # Layer 2: resampled/5m
    (root / "resampled" / "5m").mkdir(parents=True)
    _bars_ist(30, freq="5min").assign(symbol="HDFCBANK").to_parquet(
        root / "resampled" / "5m" / "HDFCBANK_5m.parquet")

    # Layer 3: spot
    (root / "spot" / "NIFTY50").mkdir(parents=True)
    _bars_ist(30, freq="5min", base=24000, with_oi=True).assign(
        symbol="NIFTY50").to_parquet(root / "spot" / "NIFTY50" / "5min.parquet")
    _bars_ist(10, freq="1D", base=24000, with_oi=True).assign(
        symbol="NIFTY50").to_parquet(root / "spot" / "NIFTY50" / "day.parquet")

    # Layer 4: equity_daily
    (root / "equity_daily" / "HDFCBANK").mkdir(parents=True)
    _bars_ist(20, freq="1D", with_oi=True).assign(symbol="HDFCBANK").to_parquet(
        root / "equity_daily" / "HDFCBANK" / "day.parquet")

    # Layer 9: model_options_with_greeks (with some NaN IV rows)
    (root / "model_options_with_greeks").mkdir(parents=True)
    n = 50
    iv = np.full(n, 0.15)
    iv[:5] = np.nan                          # deep ITM/OTM inversion fails
    opt = pd.DataFrame({
        "trading_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "symbol": "NIFTY", "expiry_date": pd.Timestamp("2024-02-29"),
        "strike": 24000, "side": "CE",
        "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0,
        "settlement": 105.0, "volume": 5000, "open_interest": 20000,
        "change_in_oi": 100, "days_to_expiry": 30, "atm_strike_est": 24000,
        "strike_distance_pct": 0.0, "spot_close": 24010.0,
        "risk_free_rate": 0.07, "t": 0.082, "flag": "c",
        "moneyness": 1.0, "log_moneyness": 0.0, "distance_from_spot_pct": 0.04,
        "option_price_for_iv": 105.0, "implied_volatility": iv,
        "delta": 0.55, "gamma": 0.002, "theta": -5.0, "vega": 12.0,
    })
    opt.to_parquet(root / "model_options_with_greeks"
                   / "NIFTY_model_options_with_greeks.parquet")

    # Layer 10: macro
    (root / "macro").mkdir(parents=True)
    pd.DataFrame({
        "trading_date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "risk_free_rate": 0.07,
    }).to_parquet(root / "macro" / "risk_free_rate_daily.parquet")


class WarehouseReaderTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _build_warehouse(self.root)
        self.wh = WarehouseReader(root=str(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    # --- existence + schema ---------------------------------------------

    def test_missing_file_raises_named_error(self):
        with self.assertRaises(WarehouseFileMissing):
            self.wh.load_equity_1m("NONEXISTENT")

    def test_schema_validation_catches_bad_columns(self):
        bad = self.root / "raw_1m" / "BADSYM_1m.parquet"
        pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")],
                      "close": [100.0]}).to_parquet(bad)
        with self.assertRaises(WarehouseSchemaError):
            self.wh.load_equity_1m("BADSYM")

    # --- equity 1m ------------------------------------------------------

    def test_load_equity_1m_returns_sorted_utc(self):
        df, rep = self.wh.load_equity_1m("HDFCBANK")
        self.assertEqual(rep.timeframe_class, "intraday")
        self.assertEqual(rep.n_rows, 120)
        # normalize_utc=True default -> tz-naive timestamps.
        self.assertIsNone(df["timestamp"].dt.tz)
        # sorted ascending.
        ts = df["timestamp"].values
        self.assertTrue(np.all(ts[1:] >= ts[:-1]))
        # IST 09:15 -> UTC 03:45.
        self.assertEqual(df["timestamp"].iloc[0].hour, 3)
        self.assertEqual(df["timestamp"].iloc[0].minute, 45)

    def test_normalize_utc_false_keeps_tz(self):
        df, _ = self.wh.load_equity_1m("HDFCBANK", normalize_utc=False)
        self.assertIsNotNone(df["timestamp"].dt.tz)

    # --- spot -----------------------------------------------------------

    def test_load_spot_5min_and_day(self):
        df5, rep5 = self.wh.load_spot("NIFTY50", "5min")
        dfd, repd = self.wh.load_spot("NIFTY50", "day")
        self.assertEqual(rep5.timeframe_class, "intraday")
        self.assertEqual(repd.timeframe_class, "daily")
        self.assertIn("open_interest", df5.columns)

    def test_spot_bad_tf_raises(self):
        with self.assertRaises(WarehouseError):
            self.wh.spot_path("NIFTY50", "17min")

    # --- equity daily ---------------------------------------------------

    def test_load_equity_daily_is_daily_class(self):
        _, rep = self.wh.load_equity_daily("HDFCBANK")
        self.assertEqual(rep.timeframe_class, "daily")

    # --- options + greeks -----------------------------------------------

    def test_greeks_keeps_missing_iv_by_default(self):
        df, rep = self.wh.load_model_options_with_greeks("NIFTY")
        self.assertEqual(rep.timeframe_class, "eod_options")
        # 5 NaN IV rows preserved, surfaced in the report.
        self.assertEqual(rep.missing_summary.get("implied_volatility"), 5)
        self.assertEqual(len(df), 50)

    def test_greeks_drop_missing_iv(self):
        df, rep = self.wh.load_model_options_with_greeks(
            "NIFTY", drop_missing_iv=True)
        self.assertEqual(len(df), 45)
        self.assertEqual(rep.missing_summary.get("dropped_missing_iv"), 5)
        self.assertFalse(df["implied_volatility"].isna().any())

    # --- macro ----------------------------------------------------------

    def test_load_macro(self):
        df, rep = self.wh.load_macro()
        self.assertEqual(list(df.columns)[:2], ["trading_date", "risk_free_rate"])
        self.assertEqual(rep.timeframe_class, "daily")

    # --- discovery ------------------------------------------------------

    def test_available_symbols_and_indexes(self):
        self.assertIn("HDFCBANK", self.wh.available_equity_symbols())
        self.assertIn("NIFTY50", self.wh.available_spot_indexes())

    def test_summary_does_not_crash_on_partial_warehouse(self):
        reports = self.wh.summary(print_report=False)
        self.assertGreater(len(reports), 0)


class LookaheadHelperTests(unittest.TestCase):

    def test_assert_sorted_passes_on_sorted(self):
        df = _bars_ist(10)
        assert_sorted_by_time(df)               # no raise

    def test_assert_sorted_raises_on_unsorted(self):
        df = _bars_ist(10).iloc[::-1].reset_index(drop=True)
        with self.assertRaises(WarehouseError):
            assert_sorted_by_time(df)

    def test_causal_rolling_never_backfills(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = causal_rolling(s, window=3, fn="mean")
        # First value is just itself (expanding), never a future-smoothed
        # value. mean of [1] = 1.0, not mean of [1,2,3].
        self.assertEqual(out.iloc[0], 1.0)
        self.assertEqual(out.iloc[1], 1.5)
        self.assertAlmostEqual(out.iloc[2], 2.0)
        self.assertAlmostEqual(out.iloc[4], 4.0)

    def test_align_daily_to_intraday_no_lookahead(self):
        # Intraday bars on 2024-01-02 must see the DAILY feature from
        # 2024-01-01, never same-day.
        intraday = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-02 09:15", "2024-01-02 09:20",
                "2024-01-03 09:15"]),
            "close": [100, 101, 102],
        })
        daily = pd.DataFrame({
            "trading_date": pd.to_datetime(["2024-01-01", "2024-01-02",
                                             "2024-01-03"]),
            "regime_value": [10.0, 20.0, 30.0],
        })
        merged = align_daily_to_intraday(intraday, daily,
                                          feature_cols=["regime_value"])
        # 2024-01-02 bars get 2024-01-01's value (10.0), shifted by 1.
        jan2 = merged[merged["timestamp"].dt.day == 2]
        self.assertTrue((jan2["regime_value"] == 10.0).all())
        # 2024-01-03 bar gets 2024-01-02's value (20.0).
        jan3 = merged[merged["timestamp"].dt.day == 3]
        self.assertTrue((jan3["regime_value"] == 20.0).all())


if __name__ == "__main__":
    unittest.main()
