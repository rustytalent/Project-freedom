"""Multi-timeframe today-relative state features.

The ``StateFeaturizer`` previously only emitted bar-relative features (returns,
z-score, momentum) plus pool geometry. The direction model therefore lacked
"where are we within today's session?" context — the same z-score has very
different implications at 09:30 IST (open volatility) vs 14:30 IST (last hour
before EOD cap).

These tests pin the 5 new today-relative features:

  * ``htf_today_range_atr``        — session range so far / ATR
  * ``htf_today_pos_in_range``     — close's position within today's range
  * ``htf_today_open_to_now_atr``  — net move from today's first 5-min open
  * ``htf_session_volume_ratio``   — cum vol so far / 20-day avg TOTAL vol
  * ``htf_overnight_gap_atr``      — today's open vs yesterday's last close

All five are derivable from the 5-min base bars without a separate higher-TF
data feed, and all five honour the IST session boundary (they reset every day
exactly at the 09:15 IST open).
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liqpool.timing import StateFeaturizer, STATE_FEATURE_NAMES


def _bars_three_sessions(n_per_session: int = 75,
                         session_bases: tuple = (100.0, 200.0, 150.0),
                         volumes: tuple = (1000.0, 2000.0, 500.0)) -> pd.DataFrame:
    """Three back-to-back NSE sessions of n_per_session bars each.

    Each session's bars open at 09:15 IST and step every 5 min. The price
    discontinuities across sessions (100 -> 200 -> 150) make overnight-gap
    detection trivial, and the differing per-bar volumes make the volume
    ratio's day-to-day comparison observable.
    """
    sess1_idx = pd.date_range("2026-05-22 03:45", periods=n_per_session, freq="5min")
    sess2_idx = pd.date_range("2026-05-25 03:45", periods=n_per_session, freq="5min")
    sess3_idx = pd.date_range("2026-05-26 03:45", periods=n_per_session, freq="5min")

    def _ohlc(idx, base, vol, seed):
        rng = np.random.default_rng(seed)
        close = base + np.cumsum(rng.normal(0.0, 0.05, len(idx)))
        return pd.DataFrame({
            "open":   close, "high":   close + 0.5, "low":    close - 0.5,
            "close":  close, "volume": np.full(len(idx), vol),
        }, index=idx)

    return pd.concat([
        _ohlc(sess1_idx, session_bases[0], volumes[0], 0),
        _ohlc(sess2_idx, session_bases[1], volumes[1], 1),
        _ohlc(sess3_idx, session_bases[2], volumes[2], 2),
    ])


class MTFStateFeatureNamesTests(unittest.TestCase):

    def test_new_feature_names_present_in_canonical_list(self) -> None:
        expected = {
            "htf_today_range_atr",
            "htf_today_pos_in_range",
            "htf_today_open_to_now_atr",
            "htf_session_volume_ratio",
            "htf_overnight_gap_atr",
        }
        self.assertTrue(expected.issubset(set(STATE_FEATURE_NAMES)),
                        f"missing one of {expected} in STATE_FEATURE_NAMES")


class MTFStateFeatureValueTests(unittest.TestCase):

    def setUp(self) -> None:
        self.df = _bars_three_sessions(n_per_session=75)
        self.featurizer = StateFeaturizer(self.df)

    def test_first_bar_of_first_session_has_no_gap_and_neutral_volume_ratio(self) -> None:
        # j=1 is still inside session 1, but it's not the FIRST bar of the
        # very first day, so the running stats only see j=0,1. The 20-day
        # avg has 0 prior days here -> neutral 1.0.
        f = self.featurizer.features_at(j=1, active_pools=[])
        # No prior session -> gap defaults to 0.
        self.assertEqual(f["htf_overnight_gap_atr"], 0.0)
        # No prior history of full sessions -> ratio defaults to 1.0.
        self.assertEqual(f["htf_session_volume_ratio"], 1.0)

    def test_today_range_grows_within_session(self) -> None:
        # In the same session, running range_atr should be monotonically
        # non-decreasing as j advances (cum max of high, cum min of low).
        ranges = [self.featurizer.features_at(j=k, active_pools=[])["htf_today_range_atr"]
                  for k in range(2, 60)]
        for prev, cur in zip(ranges, ranges[1:]):
            self.assertGreaterEqual(cur + 1e-9, prev,
                "today_range_atr must be monotone within a session")

    def test_today_range_resets_at_new_session(self) -> None:
        # Last bar of session 1 (j=74) has a fully developed range.
        # First bar of session 2 (j=75) must reset to a fresh single-bar range.
        last_s1 = self.featurizer.features_at(j=74, active_pools=[])
        first_s2 = self.featurizer.features_at(j=75, active_pools=[])
        self.assertGreater(last_s1["htf_today_range_atr"], 0.5,
            "session 1 should have accumulated range by its last bar")
        # Day 2's first bar: range is exactly bar 75's (high - low) / atr.
        # Just check it's smaller than the cumulative range of all of day 1.
        self.assertLess(first_s2["htf_today_range_atr"],
                        last_s1["htf_today_range_atr"])

    def test_overnight_gap_matches_session_base_jump(self) -> None:
        # Sessions jump from base 100 to base 200, so the open of session 2
        # is ~200 vs last close of session 1 ~100. Gap_atr ≈ 100 / ATR.
        atr_at_s2_open = max(float(self.featurizer.atr_14.iloc[75]), 1e-9)
        f = self.featurizer.features_at(j=75, active_pools=[])
        # Use loose bounds because the fixture adds tiny per-bar walks.
        expected_gap_atr = (self.df["open"].iloc[75]
                            - self.df["close"].iloc[74]) / atr_at_s2_open
        self.assertAlmostEqual(f["htf_overnight_gap_atr"], expected_gap_atr,
                               places=4)
        # And the gap should be large in absolute value (~100 / atr).
        self.assertGreater(abs(f["htf_overnight_gap_atr"]), 10.0)

    def test_overnight_gap_uses_same_prices_throughout_day(self) -> None:
        # The gap is a per-DAY price value: today_open - prev_day_last_close.
        # Different bars in the same session divide it by a different ATR(j),
        # so gap_atr values differ — but reconstructing the price gap
        # (gap_atr * atr_14[j]) must be identical across bars within the day.
        first_s2 = self.featurizer.features_at(j=75, active_pools=[])
        mid_s2 = self.featurizer.features_at(j=100, active_pools=[])
        atr_first = float(self.featurizer.atr_14.iloc[75])
        atr_mid = float(self.featurizer.atr_14.iloc[100])
        gap_price_at_first = first_s2["htf_overnight_gap_atr"] * atr_first
        gap_price_at_mid = mid_s2["htf_overnight_gap_atr"] * atr_mid
        self.assertAlmostEqual(gap_price_at_first, gap_price_at_mid, places=4)

    def test_today_pos_in_range_within_unit_interval(self) -> None:
        for j in range(2, len(self.df) - 5):
            v = self.featurizer.features_at(j=j, active_pools=[])["htf_today_pos_in_range"]
            self.assertGreaterEqual(v, 0.0 - 1e-9,
                f"pos_in_range below 0 at j={j}: {v}")
            self.assertLessEqual(v, 1.0 + 1e-9,
                f"pos_in_range above 1 at j={j}: {v}")

    def test_session_volume_ratio_uses_prior_day_total(self) -> None:
        # Session 2 has per-bar volume = 2000. After K bars, cum_vol = 2000*K.
        # Prior 20-day avg total = session 1's total = 1000 * 75 = 75000.
        # So at bar 75 + 10 (10 bars into session 2): ratio = 2000*10 / 75000.
        bar = 75 + 9      # j=84 is the 10th bar of session 2
        f = self.featurizer.features_at(j=bar, active_pools=[])
        # Volume per bar = 2000, 10 bars in -> 20000 cumulative.
        # Average prior daily total = 1000 * 75 = 75000.
        expected = 20000.0 / 75000.0
        self.assertAlmostEqual(f["htf_session_volume_ratio"], expected, places=4)

    def test_session_volume_ratio_resets_at_new_session(self) -> None:
        # At the start of session 3, only the cum_vol of bar 0 of session 3
        # is in the ratio numerator (vol=500), not the cumulative volumes of
        # sessions 1+2.
        f = self.featurizer.features_at(j=150, active_pools=[])
        # Bar j=150 is the FIRST bar of session 3 -> cum_v = 500.
        # Prior 2 days have totals 75000 (session 1) and 150000 (session 2),
        # so avg = (75000+150000)/2 = 112500.
        expected = 500.0 / 112500.0
        self.assertAlmostEqual(f["htf_session_volume_ratio"], expected, places=4)

    def test_today_open_to_now_zero_at_first_bar_of_session(self) -> None:
        # At the very first bar of any session, today's open price == close
        # (the fixture's open == close), so open-to-now = 0.
        for sess_start in (0, 75, 150):
            # Skip j=0 because features_at returns zeros for j < 1.
            if sess_start == 0:
                continue
            f = self.featurizer.features_at(j=sess_start, active_pools=[])
            self.assertAlmostEqual(f["htf_today_open_to_now_atr"], 0.0,
                                   places=6,
                                   msg=f"open_to_now != 0 at first bar j={sess_start}")


class MTFStateFeatureDeterminismTests(unittest.TestCase):

    def test_same_bars_same_features(self) -> None:
        df = _bars_three_sessions(n_per_session=75)
        a = StateFeaturizer(df).features_at(j=100, active_pools=[])
        b = StateFeaturizer(df).features_at(j=100, active_pools=[])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
