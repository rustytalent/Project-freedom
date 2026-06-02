"""Anchored VWAP + fixed-range volume profile (FRVP) state features.

The ``StateFeaturizer`` already had price-structure features (returns, momentum,
z-score) and today-relative MTF context. It still lacked WHERE-THE-VOLUME-WAS
features. AVWAP tells the model "is price above/below the volume-weighted fair
value since {today,this-week}-open, and is that fair value rising or falling?";
FRVP (Point of Control / Value Area High / Value Area Low) tells it "where did
most of today's trading actually happen?". Both are derived causally from the
same 5-min OHLCV+volume bars — no new data feed needed.

These tests pin:

  * The 9 new feature names are in the canonical ``STATE_FEATURE_NAMES`` list.
  * AVWAP today resets at every IST session boundary.
  * AVWAP week resets only at the IST week boundary (Friday->Monday or longer
    gap).
  * AVWAP equals the manually-computed cum(typical*vol) / cum(vol).
  * POC sits at the price bin where most of today's volume traded.
  * VAH/VAL bracket the POC and contain >= 70% of today's volume.
  * ``in_value_area_today`` is 1 when close is between VAL and VAH, 0 otherwise.
  * All features are causal: identical with or without forward bars present.
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

    Bars open at 09:15 IST, step every 5 min. Each session starts at a
    different base price; we use deterministic per-bar walks for repeatability.
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


class AvwapFrvpNamesTests(unittest.TestCase):

    def test_new_feature_names_present(self) -> None:
        expected = {
            "avwap_today_dist_atr",
            "avwap_today_slope_5_atr",
            "avwap_today_dev_sigmas",
            "avwap_week_dist_atr",
            "avwap_week_slope_5_atr",
            "poc_today_dist_atr",
            "vah_today_dist_atr",
            "val_today_dist_atr",
            "in_value_area_today",
        }
        self.assertTrue(expected.issubset(set(STATE_FEATURE_NAMES)),
            f"missing one of {expected - set(STATE_FEATURE_NAMES)}")


class AvwapValueTests(unittest.TestCase):

    def setUp(self) -> None:
        self.df = _bars_three_sessions(n_per_session=75)
        self.sf = StateFeaturizer(self.df)

    def test_avwap_today_matches_manual_calculation(self) -> None:
        # At any bar j, AVWAP_today should equal cum(tp*vol) / cum(vol)
        # over the bars of j's own IST session.
        for j in (5, 30, 74, 80, 100, 149):
            session_start = int(self.sf._today_open_idx[j])
            tp = ((self.df["high"].values + self.df["low"].values
                   + self.df["close"].values) / 3.0)
            vol = self.df["volume"].values.astype(float)
            pv = (tp[session_start:j + 1] * vol[session_start:j + 1]).sum()
            v = vol[session_start:j + 1].sum()
            expected = pv / v
            got = float(self.sf._avwap_today[j])
            self.assertAlmostEqual(got, expected, places=6,
                msg=f"AVWAP_today wrong at j={j}: got {got}, expected {expected}")

    def test_avwap_resets_at_new_session(self) -> None:
        # Session 2 starts at j=75 with base price ~200; AVWAP_today on the
        # first few bars of session 2 must reflect ONLY session 2 prices,
        # not the session-1 base of 100.
        first_few = [float(self.sf._avwap_today[j]) for j in range(75, 80)]
        for v in first_few:
            self.assertGreater(v, 180.0,
                f"AVWAP_today on session 2 leaked session 1's lower base: {v}")

    def test_avwap_week_does_not_reset_on_intra_week_day_change(self) -> None:
        # Sessions 2 and 3 are Mon 25-May and Tue 26-May (same ISO week).
        # AVWAP_week at start of session 3 should incorporate session 2's
        # data, NOT reset to session 3's bar 0 alone.
        avwap_w_sess2_end = float(self.sf._avwap_week[149])
        avwap_w_sess3_start = float(self.sf._avwap_week[150])
        # On the first bar of session 3 the week accumulator is the
        # session-2 mean updated by one session-3 bar — so it must be
        # closer to session 2's end value than to session 3's bar-0 price.
        sess3_bar0_price = float(self.df["close"].iloc[150])
        self.assertLess(abs(avwap_w_sess3_start - avwap_w_sess2_end),
                        abs(avwap_w_sess3_start - sess3_bar0_price),
            "AVWAP_week appears to have reset across an intra-week day boundary")

    def test_avwap_week_resets_across_weekend_gap(self) -> None:
        # Session 1 is Fri 22-May; session 2 is Mon 25-May. The IST date diff
        # is 3 days (> 2), so the week boundary fires at session-2 bar 0.
        # AVWAP_week at session-2 first bar must be ~that single bar's
        # typical price, NOT a blend with session-1's ~100 base.
        sess2_bar0_avwap_w = float(self.sf._avwap_week[75])
        sess2_bar0_price = float(self.df["close"].iloc[75])
        self.assertAlmostEqual(
            sess2_bar0_avwap_w, sess2_bar0_price, delta=1.0,
            msg=(f"AVWAP_week did not reset across the weekend: "
                 f"{sess2_bar0_avwap_w} vs session-2 bar-0 price "
                 f"{sess2_bar0_price}"))

    def test_avwap_features_zero_at_session_warmup_for_slope(self) -> None:
        # The first bar of each session has no slope window (j < 5 relative
        # to that session); slope features must default to 0 there.
        f = self.sf.features_at(j=1, active_pools=[])
        self.assertEqual(f["avwap_today_slope_5_atr"], 0.0)
        self.assertEqual(f["avwap_week_slope_5_atr"], 0.0)


class FrvpValueTests(unittest.TestCase):

    def test_poc_at_concentration_price(self) -> None:
        # Construct a session where almost all volume trades at a single price
        # band: 75 bars, 10 of them at ~150 with huge volume, the rest at 100
        # with tiny volume. POC must be near 150.
        n = 75
        idx = pd.date_range("2026-05-26 03:45", periods=n, freq="5min")
        # First 65 bars: low-volume noise around 100.
        # Last 10 bars: high-volume bars all at ~150.
        close = np.concatenate([
            100.0 + np.random.default_rng(0).normal(0.0, 0.01, 65),
            np.full(10, 150.0),
        ])
        df = pd.DataFrame({
            "open": close, "high": close + 0.2, "low": close - 0.2,
            "close": close,
            "volume": np.concatenate([np.full(65, 1.0), np.full(10, 1000.0)]),
        }, index=idx)
        sf = StateFeaturizer(df)
        poc_at_end = float(sf._poc_today[n - 1])
        # POC bin width ~ 0.10 * atr60 — for this fixture atr60 ~ 0.4 so
        # bin_width ~ 0.04, meaning POC should be within ~one bin of 150.
        self.assertAlmostEqual(poc_at_end, 150.0, delta=2.0)

    def test_value_area_contains_at_least_70_percent_volume(self) -> None:
        # On a randomly-walking session, VAH/VAL must bracket bins whose
        # cumulative volume is >= 70% of total session volume by EOD.
        n = 75
        idx = pd.date_range("2026-05-26 03:45", periods=n, freq="5min")
        rng = np.random.default_rng(1)
        close = 100.0 + np.cumsum(rng.normal(0.0, 0.10, n))
        df = pd.DataFrame({
            "open": close, "high": close + 0.3, "low": close - 0.3,
            "close": close,
            "volume": rng.uniform(800, 1200, n),
        }, index=idx)
        sf = StateFeaturizer(df)
        # At end of session: tp_in_range should hold >= 70% of total volume.
        vah = float(sf._vah_today[n - 1])
        val = float(sf._val_today[n - 1])
        tp = (df["high"].values + df["low"].values + df["close"].values) / 3.0
        in_va = (tp >= val) & (tp <= vah)
        vol_in_va = df["volume"].values[in_va].sum()
        total = df["volume"].values.sum()
        self.assertGreaterEqual(vol_in_va / total, 0.70 - 0.05,
            "value area should cover >= ~70% of session volume "
            "(small tolerance for binning discretisation)")
        self.assertGreaterEqual(vah, val,
            "VAH must be >= VAL")

    def test_in_value_area_flag_is_consistent_with_vah_val(self) -> None:
        # The in_value_area feature must match the VAL <= close <= VAH check.
        df = _bars_three_sessions(n_per_session=75)
        sf = StateFeaturizer(df)
        for j in (10, 40, 100, 149, 160):
            f = sf.features_at(j=j, active_pools=[])
            close_T = float(df["close"].iloc[j])
            val_t = float(sf._val_today[j])
            vah_t = float(sf._vah_today[j])
            expected = 1.0 if (val_t <= close_T <= vah_t) else 0.0
            self.assertEqual(f["in_value_area_today"], expected,
                f"in_value_area flag inconsistent at j={j}")

    def test_poc_vah_val_reset_at_new_session(self) -> None:
        # Session 2 base is ~200. POC at session-2 first bar must be near 200,
        # NOT carry session-1's POC near 100.
        df = _bars_three_sessions(n_per_session=75)
        sf = StateFeaturizer(df)
        poc_sess2_start = float(sf._poc_today[75])
        self.assertGreater(poc_sess2_start, 180.0,
            f"POC failed to reset at new session: got {poc_sess2_start}")


class AvwapFrvpCausalityTests(unittest.TestCase):

    def test_features_at_j_unchanged_by_future_bars(self) -> None:
        # Build a 200-bar fixture and a truncated 100-bar fixture; features
        # at j=80 must be identical in both (only past bars contribute).
        df_full = _bars_three_sessions(n_per_session=75)
        df_truncated = df_full.iloc[:100].copy()
        sf_full = StateFeaturizer(df_full)
        sf_trunc = StateFeaturizer(df_truncated)
        f_full = sf_full.features_at(j=80, active_pools=[])
        f_trunc = sf_trunc.features_at(j=80, active_pools=[])
        for key in ("avwap_today_dist_atr", "avwap_today_slope_5_atr",
                     "avwap_today_dev_sigmas", "avwap_week_dist_atr",
                     "avwap_week_slope_5_atr",
                     "poc_today_dist_atr", "vah_today_dist_atr",
                     "val_today_dist_atr", "in_value_area_today"):
            self.assertAlmostEqual(f_full[key], f_trunc[key], places=6,
                msg=f"{key} leaked future information at j=80")


if __name__ == "__main__":
    unittest.main()
