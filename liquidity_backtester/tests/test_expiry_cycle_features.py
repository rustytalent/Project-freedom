"""NSE F&O expiry-cycle event features for intraday equity strategies.

Indian equity volatility is heavily driven by the weekly (Thursday) and
monthly (last-Thursday-of-month) options-expiry rhythm even for traders
who never touch options. These features expose the cycle as pure calendar
math off the bar's own IST date — no extra data feed needed.

Pinned contracts:
  * is_weekly_expiry_day == 1 on Thursdays, 0 otherwise.
  * is_morning_after_expiry == 1 on the first bar of the IST trading day
    immediately AFTER a Thursday.
  * days_to_monthly_expiry counts down to the LAST Thursday of the bar's
    IST month, clamped at 0 after that day (post-expiry leftover days).
  * is_monthly_expiry_week == 1 when days_to_monthly_expiry <= 7.

Limitations documented in the implementation: when Thursday is an NSE
holiday, true expiry shifts to the preceding Wednesday. This featurizer
does NOT carry an NSE holiday calendar, so the days_to_monthly_expiry
value is off by one in those <5% of months. These tests deliberately
avoid that edge case.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liqpool.timing import StateFeaturizer, STATE_FEATURE_NAMES


def _bars_on_ist_date(date_str: str, n_bars: int = 12,
                      base_price: float = 100.0) -> pd.DataFrame:
    """One IST trading day worth of 5-min bars starting at 09:15 IST."""
    h_utc, m_utc = 3, 45
    start = pd.Timestamp(f"{date_str} {h_utc:02d}:{m_utc:02d}")
    idx = pd.date_range(start, periods=n_bars, freq="5min")
    rng = np.random.default_rng(0)
    close = base_price + np.cumsum(rng.normal(0.0, 0.05, n_bars))
    return pd.DataFrame({
        "open":   close, "high":   close + 0.3, "low":    close - 0.3,
        "close":  close, "volume": np.full(n_bars, 1000.0),
    }, index=idx)


def _multi_day(dates: list, n_per_day: int = 12) -> pd.DataFrame:
    """Stitch together a sequence of single-day fixtures."""
    return pd.concat([_bars_on_ist_date(d, n_per_day) for d in dates])


class ExpiryFeatureNamesTests(unittest.TestCase):

    def test_new_feature_names_present(self) -> None:
        expected = {
            "days_to_monthly_expiry",
            "is_weekly_expiry_day",
            "is_morning_after_expiry",
            "is_monthly_expiry_week",
        }
        self.assertTrue(expected.issubset(set(STATE_FEATURE_NAMES)),
            f"missing one of {expected - set(STATE_FEATURE_NAMES)}")


class WeeklyExpiryTests(unittest.TestCase):

    def test_thursday_flagged_as_weekly_expiry_day(self) -> None:
        # 2026-05-28 IST is a Thursday.
        # 2026-05-27 IST is Wednesday.
        df = _multi_day(["2026-05-27", "2026-05-28"], n_per_day=12)
        sf = StateFeaturizer(df)
        # First 12 bars = Wednesday -> 0; next 12 = Thursday -> 1.
        for j in range(1, 12):
            f = sf.features_at(j=j, active_pools=[])
            self.assertEqual(f["is_weekly_expiry_day"], 0.0,
                f"Wednesday flagged as expiry at j={j}")
        for j in range(12, 24):
            f = sf.features_at(j=j, active_pools=[])
            self.assertEqual(f["is_weekly_expiry_day"], 1.0,
                f"Thursday not flagged as expiry at j={j}")

    def test_friday_morning_flagged_as_morning_after_expiry(self) -> None:
        # 2026-05-28 IST = Thursday; 2026-05-29 IST = Friday.
        df = _multi_day(["2026-05-28", "2026-05-29"], n_per_day=12)
        sf = StateFeaturizer(df)
        # Thursday bars: morning-after-expiry should be 0.
        for j in range(1, 12):
            f = sf.features_at(j=j, active_pools=[])
            self.assertEqual(f["is_morning_after_expiry"], 0.0)
        # Friday's bars: morning-after-expiry should be 1.
        for j in range(12, 24):
            f = sf.features_at(j=j, active_pools=[])
            self.assertEqual(f["is_morning_after_expiry"], 1.0,
                f"Friday morning not flagged morning_after_expiry at j={j}")


class MonthlyExpiryTests(unittest.TestCase):

    def test_days_to_monthly_expiry_decreasing_within_month(self) -> None:
        # Last Thursday of May 2026 is 2026-05-28.
        # 2026-05-25 (Mon) -> 3 days, 2026-05-26 (Tue) -> 2, 2026-05-27 -> 1,
        # 2026-05-28 (the expiry) -> 0, 2026-05-29 (Fri, post-expiry) -> 0
        # (clamped).
        df = _multi_day(["2026-05-25", "2026-05-26", "2026-05-27",
                          "2026-05-28", "2026-05-29"], n_per_day=4)
        sf = StateFeaturizer(df)
        days_seen = [
            sf.features_at(j=k, active_pools=[])["days_to_monthly_expiry"]
            for k in (1, 5, 9, 13, 17)        # one bar per day
        ]
        self.assertEqual(days_seen, [3.0, 2.0, 1.0, 0.0, 0.0],
            f"days_to_monthly_expiry sequence wrong: {days_seen}")

    def test_monthly_expiry_week_flag(self) -> None:
        # Within 7 calendar days of the last Thursday -> flag = 1.
        # 2026-05-21 (Thu) is 7 days before 2026-05-28 (last Thu of May) ->
        # should be 1. 2026-05-20 (Wed) is 8 days before -> should be 0.
        df = _multi_day(["2026-05-20", "2026-05-21"], n_per_day=4)
        sf = StateFeaturizer(df)
        f_pre = sf.features_at(j=1, active_pools=[])
        f_post = sf.features_at(j=5, active_pools=[])
        self.assertEqual(f_pre["is_monthly_expiry_week"], 0.0,
            f"day 8 before expiry was flagged as expiry week: "
            f"days_to={f_pre['days_to_monthly_expiry']}")
        self.assertEqual(f_post["is_monthly_expiry_week"], 1.0,
            f"day 7 before expiry was NOT flagged as expiry week: "
            f"days_to={f_post['days_to_monthly_expiry']}")


class MonthBoundaryTests(unittest.TestCase):

    def test_days_to_expiry_jumps_at_month_change(self) -> None:
        # 2026-05-29 (Fri, post-May-expiry) -> 0 days to MAY expiry.
        # 2026-06-01 (Mon) -> ?? days to JUNE expiry. June 2026's last
        # Thursday is 2026-06-25.  From 2026-06-01 -> 24 days.
        df = _multi_day(["2026-05-29", "2026-06-01"], n_per_day=4)
        sf = StateFeaturizer(df)
        post_may = sf.features_at(j=1, active_pools=[])
        june_start = sf.features_at(j=5, active_pools=[])
        self.assertEqual(post_may["days_to_monthly_expiry"], 0.0)
        self.assertEqual(june_start["days_to_monthly_expiry"], 24.0,
            f"June 1 -> June 25 expiry should be 24 days, got "
            f"{june_start['days_to_monthly_expiry']}")


class ExpiryCausalityTests(unittest.TestCase):

    def test_features_at_j_unchanged_by_future_bars(self) -> None:
        # Identical features whether the dataframe extends beyond j or not.
        df_full = _multi_day(["2026-05-25", "2026-05-26", "2026-05-27",
                                "2026-05-28", "2026-05-29"], n_per_day=12)
        df_truncated = df_full.iloc[:24].copy()
        sf_full = StateFeaturizer(df_full)
        sf_trunc = StateFeaturizer(df_truncated)
        f_full = sf_full.features_at(j=20, active_pools=[])
        f_trunc = sf_trunc.features_at(j=20, active_pools=[])
        for key in ("days_to_monthly_expiry", "is_weekly_expiry_day",
                     "is_morning_after_expiry", "is_monthly_expiry_week"):
            self.assertEqual(f_full[key], f_trunc[key],
                f"{key} leaked future information at j=20")


if __name__ == "__main__":
    unittest.main()
