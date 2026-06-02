"""Tests for the three seed alphas — pool_reach / mean_reversion / momentum.

Each alpha is tested on synthetic data designed to either trigger or
suppress its specific thesis. Goal: validate that the alpha's
candidates() method honors its declared triggers and barrier preferences.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from liqpool.arsenal.alphas import (
    LiquidityPoolReachAlpha,
    MeanReversionAlpha,
    MomentumAlpha,
)
from liqpool.arsenal.alphas.pool_reach import _headline_for_pool, _tf_bucket
from liqpool.indicators import atr


def _bars_ist(n: int, start_ist: str = "10:00",
              base_price: float = 100.0, vol: float = 0.5):
    h, m = start_ist.split(":")
    ist_min = int(h) * 60 + int(m)
    utc_min = ist_min - (5 * 60 + 30)
    if utc_min < 0:
        utc_min += 24 * 60
    utc_h, utc_m = divmod(utc_min, 60)
    start_utc = f"2026-05-26 {utc_h:02d}:{utc_m:02d}"
    idx = pd.date_range(start_utc, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = base_price + np.cumsum(rng.normal(0, vol, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": rng.uniform(800, 1200, n),
    }, index=idx)


# ---------------------------------------------------------------------------
# Pool-reach alpha
# ---------------------------------------------------------------------------

@dataclass
class _StubResult:
    pool_idx: int
    touched_at: object
    pool_quality: float = 0.5


class _StubContributor:
    def __init__(self, source: str):
        self.source = source


class _StubPool:
    def __init__(self, side, price_low, price_high, tfs, contributors=None,
                 score=1.0):
        self.side = side
        self.price_low = price_low
        self.price_high = price_high
        self.tfs = list(tfs)
        self.contributors = contributors or [_StubContributor("EQH")]
        self.score = score

    @property
    def mid(self) -> float:
        return (self.price_low + self.price_high) / 2.0


@dataclass
class _StubWalkforward:
    oos_pools: List
    oos_results: List


@dataclass
class _StubAssetData:
    walkforward: _StubWalkforward


class PoolReachAlphaTests(unittest.TestCase):

    def test_emits_one_signal_per_touched_pool(self):
        df = _bars_ist(50, "10:00")
        atr_s = atr(df, 14).bfill()
        # 3 touched pools, 1 untouched.
        pools = [
            _StubPool("low", 95.0, 96.0, ["base", "15min"]),
            _StubPool("high", 110.0, 111.0, ["base"]),
            _StubPool("low", 90.0, 91.0, ["base", "60min"]),
            _StubPool("low", 80.0, 81.0, ["base"]),     # untouched
        ]
        results = [
            _StubResult(0, df.index[10]),
            _StubResult(1, df.index[20]),
            _StubResult(2, df.index[30]),
            _StubResult(3, None),                          # untouched
        ]
        ad = _StubAssetData(_StubWalkforward(pools, results))
        signals = LiquidityPoolReachAlpha().candidates(
            symbol="HDFCBANK", df_base=df, atr_series=atr_s,
            extra={"asset_data": ad},
        )
        self.assertEqual(len(signals), 3)
        # Side mapping: low -> long; high -> short.
        self.assertEqual([s.side for s in signals], ["long", "short", "long"])
        # Default barriers.
        for s in signals:
            self.assertEqual(s.stop_atr, 0.5)
            self.assertEqual(s.target_atr, 2.0)
            self.assertEqual(s.horizon_bars, 40)

    def test_skips_when_no_asset_data(self):
        df = _bars_ist(20)
        atr_s = atr(df, 14).bfill()
        sigs = LiquidityPoolReachAlpha().candidates(
            symbol="X", df_base=df, atr_series=atr_s, extra={},
        )
        self.assertEqual(sigs, [])

    def test_headline_factor_priority(self):
        # OB priority > FVG > EQHL.
        pool_ob = _StubPool("low", 0, 1, ["base"], [
            _StubContributor("OB_bull"), _StubContributor("EQH"),
        ])
        pool_fvg = _StubPool("low", 0, 1, ["base"], [
            _StubContributor("FVG_bull"), _StubContributor("EQH"),
        ])
        pool_eqhl = _StubPool("low", 0, 1, ["base"], [
            _StubContributor("EQH"),
        ])
        self.assertEqual(_headline_for_pool(pool_ob), "OB")
        self.assertEqual(_headline_for_pool(pool_fvg), "FVG")
        self.assertEqual(_headline_for_pool(pool_eqhl), "EQHL")

    def test_tf_bucket_categories(self):
        self.assertEqual(_tf_bucket(0), "1tf")
        self.assertEqual(_tf_bucket(1), "1tf")
        self.assertEqual(_tf_bucket(2), "2tf")
        self.assertEqual(_tf_bucket(3), "3tf")
        self.assertEqual(_tf_bucket(5), "4plus_tf")


# ---------------------------------------------------------------------------
# Mean-reversion alpha
# ---------------------------------------------------------------------------

class MeanReversionAlphaTests(unittest.TestCase):

    def test_no_signals_on_quiet_data(self):
        # Random noise around 100 with small std -> no extreme z-scores.
        rng = np.random.default_rng(42)
        n = 80
        close = 100.0 + rng.normal(0, 0.1, n)
        idx = pd.date_range("2026-05-26 04:30", periods=n, freq="5min")
        df = pd.DataFrame({
            "open": close, "high": close + 0.05, "low": close - 0.05,
            "close": close, "volume": np.ones(n) * 1000,
        }, index=idx)
        atr_s = atr(df, 14).bfill()
        sigs = MeanReversionAlpha().candidates(
            symbol="X", df_base=df, atr_series=atr_s,
        )
        # Should be very few signals — quiet noisy data rarely sustains z>2.
        # We allow up to 5/80 bars to account for occasional random spikes.
        self.assertLessEqual(len(sigs), 5)

    def test_signal_on_engineered_spike(self):
        # Stable 100, then a sudden jump to 110 at bar 30 -> high z-score.
        n = 60
        close = np.full(n, 100.0)
        close[30:] = 110.0
        idx = pd.date_range("2026-05-26 04:30", periods=n, freq="5min")
        df = pd.DataFrame({
            "open": close, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": np.ones(n) * 1000,
        }, index=idx)
        atr_s = atr(df, 14).bfill()
        sigs = MeanReversionAlpha().candidates(
            symbol="X", df_base=df, atr_series=atr_s,
        )
        # The spike should trigger at least one SHORT signal (z > 0 -> short).
        self.assertGreater(len(sigs), 0)
        shorts = [s for s in sigs if s.side == "short"]
        self.assertGreater(len(shorts), 0)
        for s in shorts:
            self.assertEqual(s.stop_atr, 0.5)
            self.assertEqual(s.target_atr, 1.5)
            self.assertEqual(s.horizon_bars, 12)
            self.assertGreater(s.state["zscore"], 2.0)

    def test_respects_min_bar_gap(self):
        # Persistent extreme spike -> first signal fires, subsequent within
        # MIN_BAR_GAP=6 must be suppressed.
        n = 60
        close = np.full(n, 100.0)
        close[25:] = 120.0
        idx = pd.date_range("2026-05-26 04:30", periods=n, freq="5min")
        df = pd.DataFrame({
            "open": close, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": np.ones(n) * 1000,
        }, index=idx)
        atr_s = atr(df, 14).bfill()
        sigs = MeanReversionAlpha().candidates(
            symbol="X", df_base=df, atr_series=atr_s,
        )
        if len(sigs) >= 2:
            gaps = [sigs[i + 1].decision_idx - sigs[i].decision_idx
                    for i in range(len(sigs) - 1)]
            for g in gaps:
                self.assertGreaterEqual(g, MeanReversionAlpha.MIN_BAR_GAP)


# ---------------------------------------------------------------------------
# Momentum alpha
# ---------------------------------------------------------------------------

class MomentumAlphaTests(unittest.TestCase):

    def test_signal_on_engineered_trend(self):
        # Linear up-trend over 100 bars at 0.5 per bar -> 12-bar return of 6,
        # ATR ~ 1.something -> exceeds 1.5 threshold.
        n = 100
        close = 100.0 + np.arange(n, dtype=float) * 0.5
        # Add small noise so ATR is positive.
        rng = np.random.default_rng(7)
        close += rng.normal(0, 0.05, n)
        idx = pd.date_range("2026-05-26 04:30", periods=n, freq="5min")
        df = pd.DataFrame({
            "open": close, "high": close + 0.3, "low": close - 0.3,
            "close": close, "volume": np.ones(n) * 1000,
        }, index=idx)
        atr_s = atr(df, 14).bfill()
        sigs = MomentumAlpha().candidates(
            symbol="X", df_base=df, atr_series=atr_s,
        )
        # Should produce at least one LONG signal.
        self.assertGreater(len(sigs), 0)
        longs = [s for s in sigs if s.side == "long"]
        self.assertGreater(len(longs), 0)
        for s in longs:
            self.assertEqual(s.stop_atr, 0.75)
            self.assertEqual(s.target_atr, 2.0)
            self.assertEqual(s.horizon_bars, 24)

    def test_no_signal_on_flat_data(self):
        n = 100
        close = np.full(n, 100.0) + np.random.default_rng(0).normal(0, 0.01, n)
        idx = pd.date_range("2026-05-26 04:30", periods=n, freq="5min")
        df = pd.DataFrame({
            "open": close, "high": close + 0.01, "low": close - 0.01,
            "close": close, "volume": np.ones(n) * 1000,
        }, index=idx)
        atr_s = atr(df, 14).bfill()
        sigs = MomentumAlpha().candidates(
            symbol="X", df_base=df, atr_series=atr_s,
        )
        self.assertEqual(len(sigs), 0)

    def test_regime_tag_includes_vol_bucket(self):
        n = 100
        close = 100.0 + np.arange(n, dtype=float) * 0.5
        rng = np.random.default_rng(7)
        close += rng.normal(0, 0.05, n)
        idx = pd.date_range("2026-05-26 04:30", periods=n, freq="5min")
        df = pd.DataFrame({
            "open": close, "high": close + 0.3, "low": close - 0.3,
            "close": close, "volume": np.ones(n) * 1000,
        }, index=idx)
        atr_s = atr(df, 14).bfill()
        alpha = MomentumAlpha()
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=atr_s)
        if sigs:
            tags = alpha.regime_tags(sigs[0], df, atr_s)
            self.assertIn(tags.get("vol_regime_bucket"),
                           {"low_vol", "normal_vol", "high_vol"})


if __name__ == "__main__":
    unittest.main()
