"""Tests for the volume-aware detectors:
``volume_weighted_swings`` and ``cumulative_delta_divergences``.

Pinned contracts:

  * ``volume_weighted_swings`` fires ONLY at swings where the swing bar's
    volume exceeded ``vw_swing_multiplier`` x the rolling-mean volume
    of the prior ``vw_swing_window`` bars. It does NOT fire on
    equal-volume swings.
  * The detector silently no-ops when the dataframe has no ``volume``
    column (defensive — some intraday parquets land without volume).
  * ``known_at`` matches the plain swing detector's confirmation
    rule: at-or-after bar (swing_idx + swing_right).
  * ``cumulative_delta_divergences`` fires at a new-high swing where
    the rolling cumulative delta did NOT confirm the new high (lower
    cumulative delta high). Mirror for new-low swings.
  * ``known_at`` matches the plain swing rule (same as above).
  * Integration: ``detect_all`` includes the new sources.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liqpool.config import DetectionParams
from liqpool.detectors.volume import (
    cumulative_delta_divergences,
    volume_weighted_swings,
)


def _params() -> DetectionParams:
    p = DetectionParams()
    p.swing_left = 2
    p.swing_right = 2
    p.atr_period = 5
    return p


def _df(close: list[float], volume: list[float] | None = None,
        wick: float = 0.05,
        ts_start: str = "2026-05-26 03:45",
        open_offset: list[float] | None = None) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range(ts_start, periods=n, freq="5min")
    close_arr = np.asarray(close, dtype=float)
    if volume is None:
        vol_arr = np.full(n, 1000.0)
    else:
        vol_arr = np.asarray(volume, dtype=float)
    if open_offset is None:
        open_arr = close_arr - 0.01     # mild bullish bars (close > open)
    else:
        open_arr = close_arr - np.asarray(open_offset, dtype=float)
    return pd.DataFrame({
        "open": open_arr,
        "high": close_arr + wick,
        "low": close_arr - wick,
        "close": close_arr,
        "volume": vol_arr,
    }, index=idx)


class VolumeWeightedSwingTests(unittest.TestCase):
    """Only high-volume swings fire."""

    def test_fires_on_high_volume_swing(self) -> None:
        # Swing high at idx 5 = 105. Make that bar's volume 10x normal.
        close = [100, 101, 102, 103, 104, 105,
                  104.5, 104, 103.5, 103,
                  102, 102.5, 103, 103.5, 104, 105, 104, 103, 102, 101]
        volume = [1000.0] * len(close)
        volume[5] = 10000.0           # institutional swing
        df = _df(close, volume=volume)
        out = volume_weighted_swings(df, _params(), tf="base")
        highs = [c for c in out if c.source.startswith("VW_SWING_H")]
        self.assertGreaterEqual(len(highs), 1,
            "high-volume swing should produce VW_SWING_H candidate")
        c = highs[0]
        self.assertEqual(c.side, "high")
        self.assertGreater(c.meta["volume_ratio"], 1.8)
        self.assertGreater(c.strength, 1.0)

    def test_no_emit_on_equal_volume_swing(self) -> None:
        # Same close pattern but flat volume.
        close = [100, 101, 102, 103, 104, 105,
                  104.5, 104, 103.5, 103,
                  102, 102.5, 103, 103.5, 104, 105, 104, 103, 102, 101]
        df = _df(close, volume=[1000.0] * len(close))
        out = volume_weighted_swings(df, _params(), tf="base")
        # Flat-volume swing must NOT pass the multiplier threshold.
        self.assertEqual(len(out), 0,
            f"flat-volume swing should not register; got {len(out)} cands")

    def test_silently_noop_when_no_volume_column(self) -> None:
        # Frame without volume column.
        close = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101]
        idx = pd.date_range("2026-05-26 03:45", periods=len(close),
                              freq="5min")
        df = pd.DataFrame({
            "open": close, "high": [c + 0.05 for c in close],
            "low": [c - 0.05 for c in close], "close": close,
        }, index=idx)
        out = volume_weighted_swings(df, _params(), tf="base")
        self.assertEqual(out, [],
            "missing volume column must produce empty output, not raise")

    def test_known_at_after_swing_confirmation(self) -> None:
        close = [100, 101, 102, 103, 104, 105,
                  104.5, 104, 103.5, 103,
                  102, 102.5, 103, 103.5, 104, 105, 104, 103, 102, 101]
        volume = [1000.0] * len(close)
        volume[5] = 10000.0
        df = _df(close, volume=volume)
        out = volume_weighted_swings(df, _params(), tf="base")
        self.assertGreaterEqual(len(out), 1)
        c = out[0]
        # known_at must be after the swing bar (swing_right=2 confirmation).
        params = _params()
        self.assertGreater(c.known_at, df.index[5 + params.swing_right - 1])


class CumulativeDeltaDivergenceTests(unittest.TestCase):
    """Divergence between price and cumulative signed-volume proxy."""

    def test_fires_on_bearish_divergence_at_swing_high(self) -> None:
        # Setup:
        #   Phase 1: rally to swing high at idx 5 (close=105), strong
        #     bullish bars with HIGH volume → cum delta strongly positive.
        #   Phase 2: pullback.
        #   Phase 3: rally to strict swing high at idx 16 (close=107),
        #     but bars are bearish (close < open) with high volume
        #     → cum delta DROPS during phase 3, so cd[16] < cd[5].
        # Both swing highs are confirmed because the post-bars taper down.
        n = 25
        close = ([100, 101, 102, 103, 104, 105,        # idx 5: swing high
                   104.5, 104, 103.5, 103,             # pullback
                   103.5, 104, 105, 105.5, 106,        # rally
                   106.5, 107,                          # idx 16: swing high
                   106.5, 106, 105.5, 105, 104.5,
                   104, 103.5, 103])
        # Phase 1 (bars 0-5): close > open by 0.5 → bullish, sv > 0.
        # Phase 2 (bars 6-11): mild bullish closes.
        # Phase 3 (bars 12-24): close < open → bearish, sv < 0.
        open_offset = [0.5] * 12 + [-0.5] * 13
        # Heavy volume during phase 1 AND phase 3 → cum delta swings
        # strongly positive in phase 1, strongly negative in phase 3.
        volume = ([5000.0] * 6 + [500.0] * 6 + [5000.0] * 13)
        df = _df(close, volume=volume, open_offset=open_offset)
        out = cumulative_delta_divergences(df, _params(), tf="base")
        bear_div = [c for c in out if c.source.startswith("CD_DIV_H")]
        self.assertGreaterEqual(len(bear_div), 1,
            "higher swing high with weaker cum delta should produce "
            "a CD_DIV_H candidate")
        c = bear_div[0]
        self.assertEqual(c.side, "high")
        self.assertGreater(c.strength, 1.0)
        self.assertIn("prior_swing_idx", c.meta)
        self.assertIn("cum_delta_current", c.meta)
        # Confirm the divergence shape: current cum_delta strictly less
        # than the prior swing's cum_delta.
        self.assertLess(c.meta["cum_delta_current"],
                         c.meta["cum_delta_prior"])

    def test_no_emit_when_cum_delta_confirms(self) -> None:
        # Higher swing high AND higher cum delta = trend confirmation,
        # NOT divergence.
        n = 25
        close = (
            [100, 101, 102, 103, 104, 105,
              104.5, 104, 103.5, 103,
              103.5, 104, 105, 105.5, 106,
              106.5, 107, 107,
              106.5, 106, 105.5, 105, 104.5, 104, 103.5]
        )
        # Strong bullish closes throughout phases 1 AND 3 -> cum delta
        # keeps rising. No divergence.
        open_offset = [0.5] * n
        volume = [1000.0] * n
        df = _df(close, volume=volume, open_offset=open_offset)
        out = cumulative_delta_divergences(df, _params(), tf="base")
        bear_div = [c for c in out if c.source.startswith("CD_DIV_H")]
        self.assertEqual(len(bear_div), 0,
            f"trend confirmation (higher delta) should NOT fire as "
            f"divergence; got {len(bear_div)} candidates")

    def test_silently_noop_when_no_volume_column(self) -> None:
        close = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101]
        idx = pd.date_range("2026-05-26 03:45", periods=len(close),
                              freq="5min")
        df = pd.DataFrame({
            "open": close, "high": [c + 0.05 for c in close],
            "low": [c - 0.05 for c in close], "close": close,
        }, index=idx)
        out = cumulative_delta_divergences(df, _params(), tf="base")
        self.assertEqual(out, [])


class IntegrationWithDetectAllTests(unittest.TestCase):
    """The new detectors are wired into ``detect_all``."""

    def test_detect_all_includes_volume_sources_when_volume_present(self) -> None:
        from liqpool.features import detect_all
        close = [100, 101, 102, 103, 104, 105,
                  104.5, 104, 103.5, 103,
                  102, 102.5, 103, 103.5, 104, 105, 104, 103, 102, 101]
        volume = [1000.0] * len(close)
        volume[5] = 10000.0
        df = _df(close, volume=volume)
        cands = detect_all(df, _params(), tf="base", include_orb=False)
        vw_sources = [c.source for c in cands if "VW_SWING" in c.source]
        self.assertGreater(len(vw_sources), 0,
            "detect_all should include VW_SWING candidates when "
            "a high-volume swing exists")


if __name__ == "__main__":
    unittest.main()
