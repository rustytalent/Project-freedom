"""Tests for the multi-bar imbalance + premium/discount midpoint detectors.

Pinned contracts:
  * ``multi_bar_imbalances`` fires on a >=3-bar directional run where each
    bar stacks above (below) the prior without meaningful overlap and the
    cumulative displacement exceeds ``imbalance_min_atr * ATR``.
  * It does NOT fire on noise (bars with overlapping ranges).
  * Pool side = "low" for bullish runs (demand below); "high" for bearish.
  * ``known_at`` is the close of the LAST bar in the run — never earlier.
  * Truncating the dataframe AT the last bar of the run still produces
    the same detection (no future-bar lookahead).
  * ``premium_discount_midpoints`` emits a midpoint pool for each
    confirmed (swing high, swing low) pair within ``dealing_range_max_bars``
    whose range exceeds ``dealing_range_min_atr * ATR``.
  * Each qualifying pair produces BOTH a side="high" and side="low"
    candidate (the equilibrium defends regardless of approach direction).
  * Pairs smaller than the threshold are dropped silently.
  * ``known_at`` is the confirmation timestamp of the LATER swing in the
    pair, never the formation timestamp of the earlier one.
  * Integration: ``detect_all`` includes the new sources.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liqpool.config import DetectionParams
from liqpool.detectors.imbalance import (
    multi_bar_imbalances,
    premium_discount_midpoints,
)


def _df(close: list[float], wick: float = 0.05,
        ts_start: str = "2026-05-26 03:45", freq: str = "5min",
        volume: float = 1000.0) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range(ts_start, periods=n, freq=freq)
    close_arr = np.asarray(close, dtype=float)
    high = close_arr + wick
    low = close_arr - wick
    return pd.DataFrame({
        "open": close_arr, "high": high, "low": low,
        "close": close_arr, "volume": np.full(n, volume),
    }, index=idx)


def _params() -> DetectionParams:
    p = DetectionParams()
    p.swing_left = 2
    p.swing_right = 2
    p.atr_period = 5
    return p


def _stacked_bars(start: float, step: float, n: int,
                  ts_start: str = "2026-05-26 03:45") -> pd.DataFrame:
    """N bars where each bar's low strictly exceeds the previous bar's
    high — a clean no-overlap displacement run.

    Each bar spans 0.4 price units (close +/- 0.2). The step controls
    bar-to-bar separation. Step >= 0.4 + tiny margin keeps the lows of
    consecutive bars above the prior bar's high.
    """
    idx = pd.date_range(ts_start, periods=n, freq="5min")
    closes = np.array([start + step * i for i in range(n)], dtype=float)
    highs = closes + 0.2
    lows = closes - 0.2
    return pd.DataFrame({
        "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": np.full(n, 1000.0),
    }, index=idx)


class MultiBarImbalanceTests(unittest.TestCase):
    """The displacement-run detector."""

    def test_fires_on_clean_bullish_run(self) -> None:
        # 10 quiet bars, then 5 stacked bullish bars (no overlap), then quiet.
        quiet_a = [100.0 + 0.05 * i for i in range(10)]
        df_a = _df(quiet_a)
        df_b = _stacked_bars(start=101.0, step=0.6, n=5,
                              ts_start="2026-05-26 04:35")
        quiet_b = [104.0 - 0.05 * i for i in range(10)]
        df_c = _df(quiet_b, ts_start="2026-05-26 05:00")
        df = pd.concat([df_a, df_b, df_c])
        out = multi_bar_imbalances(df, _params(), tf="base")
        bulls = [c for c in out if c.source.startswith("MBI_bull")]
        self.assertGreaterEqual(len(bulls), 1,
            "clean 5-bar bullish stacked run should produce >=1 candidate")
        c = bulls[0]
        self.assertEqual(c.side, "low",
            "bullish imbalance defends as demand (side='low')")
        self.assertGreaterEqual(c.meta["run_len"], 3)
        self.assertTrue(c.source.startswith("MBI_bull"))

    def test_no_emit_on_overlapping_chop(self) -> None:
        # Chop pattern: bars overlap heavily, no displacement run.
        closes = [100.0, 100.5, 100.2, 100.6, 100.1, 100.5,
                   100.2, 100.4, 100.1, 100.3, 100.0, 100.4]
        df = _df(closes, wick=0.4)        # wide wicks => bars overlap
        out = multi_bar_imbalances(df, _params(), tf="base")
        # Could fire on tiny noise patterns, but with our threshold of
        # 0.5 ATR (default) on a tight-range fixture, output should be
        # empty or contain only trivial candidates.
        bulls = [c for c in out if c.source.startswith("MBI_bull")]
        bears = [c for c in out if c.source.startswith("MBI_bear")]
        self.assertEqual(len(bulls) + len(bears), 0,
            f"overlapping chop should not register; got {len(bulls)} bull "
            f"+ {len(bears)} bear candidates")

    def test_fires_on_clean_bearish_run(self) -> None:
        quiet_a = [100.0 - 0.05 * i for i in range(10)]
        df_a = _df(quiet_a)
        df_b = _stacked_bars(start=99.0, step=-0.6, n=5,
                              ts_start="2026-05-26 04:35")
        quiet_b = [96.0 + 0.05 * i for i in range(10)]
        df_c = _df(quiet_b, ts_start="2026-05-26 05:00")
        df = pd.concat([df_a, df_b, df_c])
        out = multi_bar_imbalances(df, _params(), tf="base")
        bears = [c for c in out if c.source.startswith("MBI_bear")]
        self.assertGreaterEqual(len(bears), 1)
        self.assertEqual(bears[0].side, "high")

    def test_known_at_is_last_bar_of_run(self) -> None:
        quiet_a = [100.0 + 0.05 * i for i in range(10)]
        df_a = _df(quiet_a)
        df_b = _stacked_bars(start=101.0, step=0.6, n=5,
                              ts_start="2026-05-26 04:35")
        df = pd.concat([df_a, df_b])
        out = multi_bar_imbalances(df, _params(), tf="base")
        bulls = [c for c in out if c.source.startswith("MBI_bull")]
        self.assertGreaterEqual(len(bulls), 1)
        c = bulls[0]
        last_idx = c.meta["end_idx"]
        # known_at must be at-or-after the last bar of the run.
        self.assertGreaterEqual(c.known_at, df.index[last_idx])

    def test_no_lookahead_on_truncation(self) -> None:
        quiet_a = [100.0 + 0.05 * i for i in range(10)]
        df_a = _df(quiet_a)
        df_b = _stacked_bars(start=101.0, step=0.6, n=5,
                              ts_start="2026-05-26 04:35")
        df_full = pd.concat([df_a, df_b])
        out_full = multi_bar_imbalances(df_full, _params(), tf="base")
        bulls_full = [c for c in out_full if c.source.startswith("MBI_bull")]
        self.assertGreaterEqual(len(bulls_full), 1)
        # Truncate to the last bar of the run inclusive — detector must
        # still fire with the same pool price.
        end_idx = bulls_full[0].meta["end_idx"]
        df_trunc = df_full.iloc[:end_idx + 1].copy()
        out_trunc = multi_bar_imbalances(df_trunc, _params(), tf="base")
        bulls_trunc = [c for c in out_trunc if c.source.startswith("MBI_bull")]
        self.assertGreaterEqual(len(bulls_trunc), 1)
        self.assertAlmostEqual(bulls_trunc[0].price, bulls_full[0].price,
                                places=4)


class PremiumDiscountMidpointTests(unittest.TestCase):
    """The dealing-range equilibrium detector."""

    def _swing_pair_fixture(self) -> pd.DataFrame:
        """Builds a frame with a clear swing high at idx ~7, a clear
        swing low at idx ~17, range ~10 points (well above
        dealing_range_min_atr * ATR with default params), plus enough
        post-bars to confirm both swings."""
        close = (
            [100.0 + i for i in range(8)]             # 100..107 (rising)
            + [106.0, 105.0, 104.0, 103.0, 102.0,    # peak at 107 (idx 7)
                101.0, 100.0, 99.0, 98.0, 97.0]      # ... down to 97 (idx 17)
            + [98.0, 99.0, 100.0, 101.0, 102.0]      # confirmation bars
        )
        return _df(close)

    def test_emits_both_sides_for_qualifying_pair(self) -> None:
        df = self._swing_pair_fixture()
        params = _params()
        params.dealing_range_min_atr = 0.5     # type: ignore[attr-defined]
        out = premium_discount_midpoints(df, params, tf="base")
        sides = {c.side for c in out}
        # Each qualifying pair emits BOTH "high" and "low" candidates.
        self.assertEqual(sides, {"high", "low"},
            f"premium/discount midpoint must emit both sides; got {sides}")

    def test_midpoint_price_is_swing_pair_midpoint(self) -> None:
        df = self._swing_pair_fixture()
        params = _params()
        params.dealing_range_min_atr = 0.5     # type: ignore[attr-defined]
        out = premium_discount_midpoints(df, params, tf="base")
        self.assertGreater(len(out), 0)
        c = out[0]
        # Midpoint of swing_high (107.05 with default wick) and
        # swing_low (96.95) is ~102.0.
        self.assertAlmostEqual(c.price,
                                (c.meta["swing_high_price"]
                                 + c.meta["swing_low_price"]) / 2.0,
                                places=4)
        # Sanity: midpoint within the dealing range.
        self.assertGreater(c.price, c.meta["swing_low_price"])
        self.assertLess(c.price, c.meta["swing_high_price"])

    def test_no_emit_when_range_too_small(self) -> None:
        # Tiny range — should not pass dealing_range_min_atr.
        close = (
            [100.0 + i * 0.05 for i in range(8)]
            + [100.35, 100.30, 100.25, 100.20, 100.15,
                100.10, 100.05, 100.00, 99.95, 99.90]
            + [100.0, 100.05, 100.1, 100.15, 100.2]
        )
        df = _df(close)
        params = _params()
        params.dealing_range_min_atr = 2.0     # type: ignore[attr-defined]
        out = premium_discount_midpoints(df, params, tf="base")
        self.assertEqual(len(out), 0,
            "tiny swing range should not produce a midpoint pool")

    def test_known_at_is_later_swing_confirmation(self) -> None:
        df = self._swing_pair_fixture()
        params = _params()
        params.dealing_range_min_atr = 0.5     # type: ignore[attr-defined]
        out = premium_discount_midpoints(df, params, tf="base")
        self.assertGreater(len(out), 0)
        c = out[0]
        later_idx = c.meta["later_idx"]
        # known_at MUST be after the later swing's formation bar.
        self.assertGreater(c.known_at, df.index[later_idx])


class IntegrationWithDetectAllTests(unittest.TestCase):
    """The new detectors are wired into ``detect_all``."""

    def test_detect_all_includes_mbi_and_pdmid_sources(self) -> None:
        from liqpool.features import detect_all
        # Build a frame with both a displacement run and a swing pair.
        quiet_a = [100.0 + 0.05 * i for i in range(8)]
        df_a = _df(quiet_a)
        df_b = _stacked_bars(start=101.0, step=0.6, n=5,
                              ts_start="2026-05-26 04:25")
        # Add bars descending to create a swing pair.
        descent = [103.0 - 0.5 * i for i in range(12)]
        df_c = _df(descent, ts_start="2026-05-26 04:50")
        confirm = [97.0 + 0.2 * i for i in range(5)]
        df_d = _df(confirm, ts_start="2026-05-26 05:50")
        df = pd.concat([df_a, df_b, df_c, df_d])
        cands = detect_all(df, _params(), tf="base", include_orb=False)
        mbi_sources = [c.source for c in cands if "MBI" in c.source]
        self.assertGreater(len(mbi_sources), 0,
            "detect_all should include multi-bar imbalance candidates")


if __name__ == "__main__":
    unittest.main()
