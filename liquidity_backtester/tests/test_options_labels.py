"""Tests for the options label generator (Stream D.2).

Pinned contracts (one test class per):

  1. Round-trip math: entry 100 → exit 120 (buy side, zero slippage,
     zero STT) gives realized_pct = +0.20.
  2. Side sign: same forward move yields +R for buy, −R for sell
     (sell side also incurs STT, captured in the magnitude check).
  3. Slippage applied at the right magnitude (4 ticks × tick_inr /
     entry, default).
  4. Zero-volume bar → label NaN + label_valid=False.
  5. Causality: per-strike ATR uses only days strictly before the
     label window opens (1-day lag enforced via the warehouse helper).
  6. No-lookahead under truncation: same label at a given bar
     whether the option_bars frame is truncated to bar+horizon or
     run in full.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liqpool.options.labels import (
    LABEL_OUTPUT_COLUMNS,
    OptionsLabelParams,
    add_options_labels,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _intraday_index(n_bars: int = 30, start: str = "2026-06-09 09:15"):
    return pd.date_range(start, periods=n_bars, freq="5min")


def _features_frame(strike: float, side: str = "buy",
                    n_bars: int = 30) -> pd.DataFrame:
    """Minimal feature frame matching what the D.1 featurizer would emit
    for one strike × side over n_bars."""
    idx = _intraday_index(n_bars)
    return pd.DataFrame({
        "trading_date_ist": idx.normalize(),
        "bar_ts": idx,
        "underlying": "NIFTY50",
        "strike": float(strike),
        "side": side,
    })


def _option_bars(strike: float,
                 premium_series: list[float],
                 start: str = "2026-06-09 09:15",
                 volume: int = 1_000) -> pd.DataFrame:
    """Build an option-bars frame with a custom premium path.

    Volume is constant unless overridden. Includes a 60-day daily
    history of premium closes BEFORE the intraday session so the
    rolling-ATR denominator has the warmup it needs.
    """
    n = len(premium_series)
    intraday_idx = pd.date_range(start, periods=n, freq="5min")
    # 70 daily historical bars ending the day before the session.
    session_date = pd.Timestamp(start).normalize()
    daily_dates = pd.date_range(
        end=session_date - pd.Timedelta(days=1), periods=70, freq="B")
    # Stable, non-zero, mildly varying daily premium history so ATR
    # warms up cleanly.
    daily_history = [(100.0 + (i % 5) * 0.5) for i in range(70)]
    daily_rows = pd.DataFrame({
        "timestamp": daily_dates + pd.Timedelta(hours=15, minutes=15),
        "underlying": "NIFTY50",
        "strike": float(strike),
        "premium_close": daily_history,
        "premium_volume": volume,
    })
    intraday_rows = pd.DataFrame({
        "timestamp": intraday_idx,
        "underlying": "NIFTY50",
        "strike": float(strike),
        "premium_close": premium_series,
        "premium_volume": volume,
    })
    return pd.concat([daily_rows, intraday_rows], ignore_index=True)


def _flat_premium(value: float, n: int) -> list[float]:
    return [value] * n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class RoundTripMathTests(unittest.TestCase):
    """Entry 100 → exit 120 (buy side, no slippage) gives +0.20."""

    def test_buy_side_no_friction(self):
        features = _features_frame(strike=24500.0, side="buy", n_bars=30)
        # Premium climbs from 100 to 120 by bar 12.
        premium = [100.0] * 12 + [120.0] * (30 - 12)
        bars = _option_bars(24500.0, premium)
        params = OptionsLabelParams(
            tick_inr=0.0, slippage_ticks_per_leg=0,
            stt_sell_pct=0.0, brokerage_pct_per_side=0.0,
        )
        out = add_options_labels(features, bars, params)
        first = out.iloc[0]
        self.assertTrue(first["label_valid"])
        self.assertAlmostEqual(
            first["realized_premium_pct_60min"], 0.20, places=6)
        # Schema sanity: every documented column is present.
        for col in LABEL_OUTPUT_COLUMNS:
            self.assertIn(col, out.columns,
                f"label output schema missing column {col!r}")


class SideSignTests(unittest.TestCase):
    """Same forward move yields +R for buy, −R for sell (modulo STT)."""

    def test_buy_vs_sell_opposite_signs(self):
        # Same path: premium drops from 100 to 80.
        premium = [100.0] * 12 + [80.0] * (30 - 12)
        bars = _option_bars(24500.0, premium)
        params = OptionsLabelParams(
            tick_inr=0.0, slippage_ticks_per_leg=0,
            stt_sell_pct=0.0, brokerage_pct_per_side=0.0,
        )

        buy = add_options_labels(
            _features_frame(24500.0, "buy"), bars, params)
        sell = add_options_labels(
            _features_frame(24500.0, "sell"), bars, params)

        b = buy.iloc[0]["realized_premium_pct_60min"]
        s = sell.iloc[0]["realized_premium_pct_60min"]
        self.assertAlmostEqual(b, -0.20, places=6)
        self.assertAlmostEqual(s, +0.20, places=6)


class SlippageMagnitudeTests(unittest.TestCase):
    """4 ticks round-trip at ₹0.05/tick on entry 100 → 0.20% friction."""

    def test_slippage_subtraction(self):
        # Premium flat → realized_pct should equal -slippage_pct.
        premium = [100.0] * 30
        bars = _option_bars(24500.0, premium)
        params = OptionsLabelParams(
            tick_inr=0.05, slippage_ticks_per_leg=2,
            stt_sell_pct=0.0, brokerage_pct_per_side=0.0,
        )
        out = add_options_labels(
            _features_frame(24500.0, "buy"), bars, params)
        first = out.iloc[0]
        # round-trip cost = 2 legs × 2 ticks × 0.05 = ₹0.20 / 100 = 0.20%.
        self.assertAlmostEqual(first["slippage_pct"], 0.20 / 100, places=6)
        self.assertAlmostEqual(
            first["realized_premium_pct_60min"], -0.20 / 100, places=6)

    def test_stt_applied_on_sell_only(self):
        premium = [100.0] * 30
        bars = _option_bars(24500.0, premium)
        params = OptionsLabelParams(
            tick_inr=0.0, slippage_ticks_per_leg=0,
            stt_sell_pct=0.001, brokerage_pct_per_side=0.0,
        )
        buy = add_options_labels(
            _features_frame(24500.0, "buy"), bars, params)
        sell = add_options_labels(
            _features_frame(24500.0, "sell"), bars, params)
        # Buy STT = 0 → realized = 0.
        self.assertAlmostEqual(
            buy.iloc[0]["realized_premium_pct_60min"], 0.0, places=8)
        # Sell STT = 0.1% → realized = -0.1%.
        self.assertAlmostEqual(
            sell.iloc[0]["realized_premium_pct_60min"], -0.001, places=8)


class ZeroVolumeTests(unittest.TestCase):
    """Zero-volume bar at entry OR exit → label NaN + label_valid=False."""

    def test_zero_volume_entry_invalidates(self):
        premium = [100.0] * 12 + [120.0] * (30 - 12)
        bars = _option_bars(24500.0, premium, volume=1_000)
        # Zero the volume on the entry bar of row 0 (the 09:15 bar).
        entry_ts = pd.Timestamp("2026-06-09 09:15")
        mask = (bars["timestamp"] == entry_ts) & (bars["strike"] == 24500.0)
        bars.loc[mask, "premium_volume"] = 0
        params = OptionsLabelParams(
            tick_inr=0.0, slippage_ticks_per_leg=0, stt_sell_pct=0.0,
            brokerage_pct_per_side=0.0,
        )
        out = add_options_labels(
            _features_frame(24500.0, "buy"), bars, params)
        first = out.iloc[0]
        self.assertFalse(first["label_valid"])
        self.assertTrue(pd.isna(first["realized_premium_pct_60min"]))
        self.assertTrue(pd.isna(first["realized_premium_atr_units_60min"]))

    def test_zero_volume_exit_invalidates(self):
        premium = [100.0] * 12 + [120.0] * (30 - 12)
        bars = _option_bars(24500.0, premium, volume=1_000)
        # Zero the volume on the EXIT bar for row 0 — exit = 09:15 + 60min
        # = 10:15.
        exit_ts = pd.Timestamp("2026-06-09 10:15")
        mask = (bars["timestamp"] == exit_ts) & (bars["strike"] == 24500.0)
        bars.loc[mask, "premium_volume"] = 0
        params = OptionsLabelParams(
            tick_inr=0.0, slippage_ticks_per_leg=0, stt_sell_pct=0.0,
            brokerage_pct_per_side=0.0,
        )
        out = add_options_labels(
            _features_frame(24500.0, "buy"), bars, params)
        first = out.iloc[0]
        self.assertFalse(first["label_valid"])


class CausalAtrTests(unittest.TestCase):
    """The per-strike ATR denominator uses ONLY trading days strictly
    before the label window's start date."""

    def test_atr_is_lagged_by_one_trading_day(self):
        # Construct a strike whose daily premium history has a STRICTLY
        # increasing volatility schedule — daily |return| grows. If
        # the rolling-ATR was leaking same-day data, the bar on date D
        # would see a different ATR value than what the lagged join
        # produces.
        n_intraday = 30
        intraday_idx = pd.date_range(
            "2026-06-09 09:15", periods=n_intraday, freq="5min")
        session_date = pd.Timestamp("2026-06-09").normalize()
        daily_dates = pd.date_range(
            end=session_date - pd.Timedelta(days=1), periods=80, freq="B")
        # Daily premium pattern: alternating ±2 to ensure non-zero
        # |return| throughout warmup.
        base = 100.0
        daily_history = []
        for i in range(80):
            base = base + (2.0 if i % 2 == 0 else -2.0)
            daily_history.append(base)
        # Also include "today's" daily close = 200 (huge return) — IF
        # the ATR used same-day data, we'd see a spike in atr_premium_pct_60d
        # at the intraday rows.
        daily_history_with_today = daily_history + [200.0]
        daily_dates_with_today = daily_dates.tolist() + [session_date]

        daily_rows = pd.DataFrame({
            "timestamp": [d + pd.Timedelta(hours=15, minutes=15)
                          for d in daily_dates_with_today],
            "underlying": "NIFTY50",
            "strike": 24500.0,
            "premium_close": daily_history_with_today,
            "premium_volume": 1_000,
        })
        intraday_rows = pd.DataFrame({
            "timestamp": intraday_idx,
            "underlying": "NIFTY50",
            "strike": 24500.0,
            "premium_close": [200.0] * n_intraday,
            "premium_volume": 1_000,
        })
        bars = pd.concat([daily_rows, intraday_rows], ignore_index=True)
        features = _features_frame(24500.0, "buy", n_intraday)
        out = add_options_labels(features, bars, OptionsLabelParams())

        # The ATR observed at any intraday row on 2026-06-09 must NOT
        # incorporate the 80→200 jump (which happened on 2026-06-09).
        # The lagged ATR uses data up to 2026-06-08 — all daily moves
        # before then were small (±2 on base ~100) giving |return| ~ 2%.
        atr = out["atr_premium_pct_60d"].iloc[0]
        self.assertTrue(np.isfinite(atr),
            "ATR must be populated after the 30-day warmup")
        self.assertLess(atr, 0.10,
            f"ATR ({atr}) should reflect 2% daily moves, NOT the "
            f"~50% jump on the bar date. Same-day leak detected.")


class NoLookaheadUnderTruncationTests(unittest.TestCase):
    """Truncating option_bars to bar+horizon must produce the same
    label at that bar as running on the full frame."""

    def test_label_invariant_under_truncation(self):
        # 30 intraday bars; we read the label at bar 5.
        n = 30
        # Make premium oscillate so the bar-5 label is non-trivial.
        rng = np.random.default_rng(0)
        premium = 100.0 + 5 * np.cumsum(rng.normal(0, 0.3, n))
        bars_full = _option_bars(24500.0, list(premium))
        features = _features_frame(24500.0, "buy", n)

        # Truncate the intraday portion to bar 5 + horizon=12 bars + 1
        # bar of safety = bar 17. Keep the daily history intact.
        truncate_at = 5 + 12 + 1
        intraday_mask = bars_full["timestamp"] >= pd.Timestamp(
            "2026-06-09 09:15")
        truncated_intraday_ts = pd.date_range(
            "2026-06-09 09:15", periods=truncate_at, freq="5min")
        keep_intraday = bars_full[intraday_mask & bars_full[
            "timestamp"].isin(truncated_intraday_ts)]
        keep_daily = bars_full[~intraday_mask]
        bars_short = pd.concat(
            [keep_daily, keep_intraday], ignore_index=True)

        full_out = add_options_labels(features, bars_full, OptionsLabelParams())
        short_out = add_options_labels(
            features.iloc[: truncate_at], bars_short, OptionsLabelParams())
        # Bar 5's label must be identical.
        full_row = full_out.iloc[5]
        short_row = short_out.iloc[5]
        for col in ("premium_entry", "premium_exit",
                    "slippage_pct", "stt_pct",
                    "realized_premium_pct_60min",
                    "atr_premium_pct_60d",
                    "realized_premium_atr_units_60min",
                    "label_valid"):
            f, s = full_row[col], short_row[col]
            if isinstance(f, float) and np.isnan(f):
                self.assertTrue(np.isnan(s),
                    f"{col} mismatch under truncation: full=NaN short={s}")
            else:
                self.assertEqual(f, s,
                    f"{col} mismatch under truncation: full={f} short={s}")


if __name__ == "__main__":
    unittest.main()
