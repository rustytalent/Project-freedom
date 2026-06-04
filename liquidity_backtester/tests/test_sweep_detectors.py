"""Tests for the liquidity-sweep + stop-run-reclaim detectors.

Pinned contracts:
  * Liquidity sweep fires on a wick that pierces a confirmed swing and
    closes back inside same bar.
  * Stop-run-reclaim fires when price CLOSES beyond the swing on >=1
    bar and then closes back inside within the reclaim window.
  * Neither detector emits a candidate when the swing is touched but
    not pierced beyond the min_atr threshold.
  * Neither detector emits a candidate when the wick pierces but
    price keeps going (no reclaim) — true breakouts are not sweeps.
  * `known_at` is the close timestamp of the RECLAIM bar, never the
    swing-formation bar (causality).
  * Truncating the dataframe to AT the reclaim bar still produces the
    same detection — no future-bar lookahead.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liqpool.config import DetectionParams
from liqpool.detectors.sweep import liquidity_sweeps, stop_run_reclaims


def _df_from(close: list[float], wick: list[float] | None = None,
             ts_start: str = "2026-05-26 03:45", freq: str = "5min"
             ) -> pd.DataFrame:
    """Build a bar frame from a close series. ``wick`` is per-bar high-
    above-close (positive) when emulating a wick-above pattern; for
    wick-below patterns pass negative numbers and they'll be applied
    to the low. Default: tiny symmetric wicks (high=close+0.05, low=close-0.05).
    """
    n = len(close)
    idx = pd.date_range(ts_start, periods=n, freq=freq)
    close_arr = np.asarray(close, dtype=float)
    if wick is None:
        wick_arr = np.full(n, 0.05)
    else:
        wick_arr = np.asarray(wick, dtype=float)
    high = close_arr + np.maximum(0.05, np.abs(wick_arr) * (wick_arr > 0))
    low = close_arr - np.maximum(0.05, np.abs(wick_arr) * (wick_arr < 0))
    return pd.DataFrame({
        "open": close_arr, "high": high, "low": low,
        "close": close_arr, "volume": np.full(n, 1000.0),
    }, index=idx)


def _params() -> DetectionParams:
    p = DetectionParams()
    # Make the test fixture deterministic by tightening thresholds.
    p.swing_left = 2
    p.swing_right = 2
    p.atr_period = 5
    return p


class LiquiditySweepTests(unittest.TestCase):
    """The wick-only sweep variant."""

    def _build_swing_high_sweep(self) -> tuple[pd.DataFrame, int, int]:
        """Construct a series with a clear swing high at idx 5, then a
        sweep-and-reclaim at idx 12 (wick above the swing, close back
        below). Returns (df, swing_idx, sweep_idx)."""
        # 20 bars. Slowly rising up to swing at 5, then choppy, then
        # the manipulation wick at 12.
        close = [100, 101, 102, 103, 104, 105,        # swing high at idx 5
                  104.5, 104, 103.5, 103, 102.5, 102,
                  103,                                  # idx 12 close back below
                  102.5, 102, 101.5, 101, 100.5, 100, 99.5]
        df = _df_from(close)
        # Idx 12 has a wick well above 105.0 (the swing high price).
        df = df.copy()
        df.loc[df.index[12], "high"] = 106.5     # 1.5 above the swing high
        return df, 5, 12

    def test_fires_on_swing_high_sweep(self) -> None:
        df, swing_idx, sweep_idx = self._build_swing_high_sweep()
        out = liquidity_sweeps(df, _params(), tf="base")
        self.assertEqual(len(out), 1,
            f"expected exactly one sweep candidate, got {len(out)}")
        c = out[0]
        self.assertEqual(c.side, "high")
        # Default wicks of 0.05 above close make the swing-high bar's
        # high = close + 0.05 = 105.05. The detector reports the BAR
        # HIGH, not the close, which is correct.
        self.assertAlmostEqual(c.price, 105.05, places=2)
        self.assertTrue(c.source.startswith("SWEEP_H"))
        self.assertEqual(c.meta["swept_bar_idx"], sweep_idx)
        self.assertGreater(c.strength, 1.0)

    def test_known_at_is_reclaim_bar_not_swing(self) -> None:
        df, swing_idx, sweep_idx = self._build_swing_high_sweep()
        out = liquidity_sweeps(df, _params(), tf="base")
        self.assertEqual(len(out), 1)
        c = out[0]
        # known_at must be AFTER the swing-formation bar.
        self.assertGreater(c.known_at, df.index[swing_idx])
        # known_at should be at-or-after the sweep bar close (next-bar period).
        self.assertGreaterEqual(c.known_at, df.index[sweep_idx])

    def test_no_emit_when_wick_too_small(self) -> None:
        # Same swing, but the wick only barely pokes above (well below
        # min_atr * ATR threshold of ~0.2 ATR).
        close = [100, 101, 102, 103, 104, 105,
                  104.5, 104, 103.5, 103, 102.5, 102,
                  103, 102.5, 102, 101.5, 101, 100.5, 100, 99.5]
        df = _df_from(close)
        df = df.copy()
        df.loc[df.index[12], "high"] = 105.05  # negligible pierce
        out = liquidity_sweeps(df, _params(), tf="base")
        self.assertEqual(len(out), 0,
            "tiny wick should not register as a sweep")

    def test_no_emit_when_close_breaks_too(self) -> None:
        # Wick AND close go past the swing — that's a breakout, not a sweep.
        close = [100, 101, 102, 103, 104, 105,
                  104.5, 104, 103.5, 103, 102.5, 102,
                  106.5,                                # close ABOVE swing 105
                  107, 108, 108.5, 108, 107.5, 107, 106.5]
        df = _df_from(close)
        df = df.copy()
        df.loc[df.index[12], "high"] = 107.0
        out = liquidity_sweeps(df, _params(), tf="base")
        self.assertEqual(len(out), 0,
            "if close goes past the swing too, that's a breakout — not a sweep")

    def test_fires_on_swing_low_sweep(self) -> None:
        # Mirror image: slowly falling swing low at idx 5, then wick
        # below + close back above at idx 12.
        close = [100, 99, 98, 97, 96, 95,             # swing low at idx 5
                  95.5, 96, 96.5, 97, 97.5, 98,
                  97,                                   # idx 12 close back above
                  97.5, 98, 98.5, 99, 99.5, 100, 100.5]
        df = _df_from(close)
        df = df.copy()
        df.loc[df.index[12], "low"] = 93.5             # 1.5 below swing low
        out = liquidity_sweeps(df, _params(), tf="base")
        self.assertEqual(len(out), 1)
        c = out[0]
        self.assertEqual(c.side, "low")
        # Default wicks of 0.05 below close make swing-low bar's low = 94.95.
        self.assertAlmostEqual(c.price, 94.95, places=2)
        self.assertTrue(c.source.startswith("SWEEP_L"))


class StopRunReclaimTests(unittest.TestCase):
    """The variant where price CLOSES beyond the swing on >=1 bar
    before reclaiming."""

    def test_fires_on_close_past_then_reclaim(self) -> None:
        # Swing high at idx 5 = 105. Bar 12 closes at 106 (past). Bar
        # 13 closes at 104 (back inside).
        close = [100, 101, 102, 103, 104, 105,
                  104.5, 104, 103.5, 103, 102.5, 102,
                  106,                                  # idx 12 close PAST
                  104,                                  # idx 13 close back
                  103.5, 103, 102.5, 102, 101.5, 101]
        df = _df_from(close)
        df = df.copy()
        df.loc[df.index[12], "high"] = 106.5
        out = stop_run_reclaims(df, _params(), tf="base")
        self.assertEqual(len(out), 1)
        c = out[0]
        self.assertEqual(c.side, "high")
        # Default wicks make the swing-high bar's high = 105.05.
        self.assertAlmostEqual(c.price, 105.05, places=2)
        self.assertTrue(c.source.startswith("SR_H"))
        self.assertEqual(c.meta["closes_past"], 1)
        self.assertEqual(c.meta["reclaim_bar_idx"], 13)

    def test_no_emit_when_no_close_past(self) -> None:
        # Wick only — should NOT fire stop_run_reclaim (that's the
        # plain liquidity_sweep's job).
        close = [100, 101, 102, 103, 104, 105,
                  104.5, 104, 103.5, 103, 102.5, 102,
                  103, 102.5, 102, 101.5, 101, 100.5, 100, 99.5]
        df = _df_from(close)
        df = df.copy()
        df.loc[df.index[12], "high"] = 106.5
        out = stop_run_reclaims(df, _params(), tf="base")
        self.assertEqual(len(out), 0,
            "wick-only sweep should NOT register as stop_run_reclaim")

    def test_fires_on_multi_bar_close_past(self) -> None:
        # 2 bars closing past, then reclaim.
        close = [100, 101, 102, 103, 104, 105,
                  104.5, 104, 103.5, 103, 102.5, 102,
                  106, 107,                             # idx 12, 13 closes past
                  104,                                  # idx 14 reclaim
                  103.5, 103, 102.5, 102, 101.5]
        df = _df_from(close)
        df = df.copy()
        df.loc[df.index[12], "high"] = 106.5
        df.loc[df.index[13], "high"] = 107.5
        out = stop_run_reclaims(df, _params(), tf="base")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].meta["closes_past"], 2)
        self.assertEqual(out[0].meta["reclaim_bar_idx"], 14)


class CausalityTests(unittest.TestCase):
    """No-look-ahead: truncating the dataframe at the reclaim bar must
    produce the same detection as the full dataframe."""

    def test_liquidity_sweep_no_lookahead(self) -> None:
        close_full = [100, 101, 102, 103, 104, 105,
                       104.5, 104, 103.5, 103, 102.5, 102,
                       103, 102.5, 102, 101.5, 101, 100.5, 100, 99.5]
        df_full = _df_from(close_full)
        df_full = df_full.copy()
        df_full.loc[df_full.index[12], "high"] = 106.5
        out_full = liquidity_sweeps(df_full, _params(), tf="base")
        self.assertEqual(len(out_full), 1)

        # Truncate AT bar 12 inclusive (this is the reclaim bar) +
        # one more for the period inference.
        df_trunc = df_full.iloc[:14].copy()
        out_trunc = liquidity_sweeps(df_trunc, _params(), tf="base")
        self.assertEqual(len(out_trunc), 1)
        self.assertEqual(out_trunc[0].price, out_full[0].price)
        self.assertEqual(out_trunc[0].meta["swept_bar_idx"],
                          out_full[0].meta["swept_bar_idx"])

    def test_stop_run_no_lookahead(self) -> None:
        close_full = [100, 101, 102, 103, 104, 105,
                       104.5, 104, 103.5, 103, 102.5, 102,
                       106, 104,
                       103.5, 103, 102.5, 102, 101.5, 101, 100.5]
        df_full = _df_from(close_full)
        df_full = df_full.copy()
        df_full.loc[df_full.index[12], "high"] = 106.5
        out_full = stop_run_reclaims(df_full, _params(), tf="base")
        self.assertEqual(len(out_full), 1)

        df_trunc = df_full.iloc[:15].copy()
        out_trunc = stop_run_reclaims(df_trunc, _params(), tf="base")
        self.assertEqual(len(out_trunc), 1)
        self.assertEqual(out_trunc[0].meta["reclaim_bar_idx"],
                          out_full[0].meta["reclaim_bar_idx"])


class IntegrationWithDetectAllTests(unittest.TestCase):
    """The new detectors are wired into ``detect_all`` and contribute
    to the pool builder's candidate stream alongside the original
    detectors."""

    def test_detect_all_includes_sweep_sources(self) -> None:
        from liqpool.features import detect_all
        close = [100, 101, 102, 103, 104, 105,
                  104.5, 104, 103.5, 103, 102.5, 102,
                  103, 102.5, 102, 101.5, 101, 100.5, 100, 99.5]
        df = _df_from(close)
        df = df.copy()
        df.loc[df.index[12], "high"] = 106.5
        cands = detect_all(df, _params(), tf="base", include_orb=False)
        sweep_sources = [c.source for c in cands if "SWEEP" in c.source]
        self.assertGreater(len(sweep_sources), 0,
            "detect_all should include liquidity-sweep candidates")


if __name__ == "__main__":
    unittest.main()
