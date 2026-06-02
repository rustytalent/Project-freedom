"""Path / context state features.

Snapshot features (z-score, momentum, distance-to-pool) can't distinguish a
bull-trap from a genuine breakdown — both can have identical state at the
trap-fade moment but opposite futures. The path features encode the SHAPE
of how price got here, not just the level: opening-range character,
trendiness-vs-choppiness, the gap-then-fade pattern an institutional
liquidity grab leaves on the tape, and the current vol regime.

Pinned contracts:
  * first_15min_range_atr / first_15min_direction_atr are 0.0 before the 3rd
    bar of an IST session (warmup).
  * path_efficiency_30 is in [0, 1]; close to 1 = trending, close to 0 = chop.
  * direction_changes_30 is in [0, 29]; counts close-to-close sign flips.
  * is_gap_up_trap_fade fires only when overnight gap > 0.5 ATR AND opening
    15-min direction < -0.3 ATR.
  * is_gap_down_reversal fires only when overnight gap < -0.5 ATR AND
    opening 15-min direction > +0.3 ATR.
  * vol_regime_zscore_20d uses prior 20 IST days only (not today).
  * All features are causal under df-truncation.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liqpool.timing import StateFeaturizer, STATE_FEATURE_NAMES


def _bars_with_open(idx, opens, closes, highs=None, lows=None, vols=None):
    """Build a synthetic OHLCV frame with explicit open/close arrays."""
    n = len(idx)
    if highs is None:
        highs = np.maximum(opens, closes) + 0.1
    if lows is None:
        lows = np.minimum(opens, closes) - 0.1
    if vols is None:
        vols = np.full(n, 1000.0)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": vols,
    }, index=idx)


def _trending_session(date_str: str, n_bars: int = 30,
                      base: float = 100.0, drift: float = 0.5) -> pd.DataFrame:
    """A monotone-uptrending session."""
    idx = pd.date_range(f"{date_str} 03:45", periods=n_bars, freq="5min")
    closes = base + np.arange(n_bars) * drift
    opens = np.concatenate(([base], closes[:-1]))
    return _bars_with_open(idx, opens, closes)


def _choppy_session(date_str: str, n_bars: int = 30,
                    base: float = 100.0) -> pd.DataFrame:
    """A zig-zag session that returns near base."""
    idx = pd.date_range(f"{date_str} 03:45", periods=n_bars, freq="5min")
    # alternating +0.5, -0.5 -> net move ~ 0, sum |moves| = n*0.5
    moves = np.tile([0.5, -0.5], n_bars)[:n_bars]
    closes = base + np.cumsum(moves)
    opens = np.concatenate(([base], closes[:-1]))
    return _bars_with_open(idx, opens, closes)


class PathFeatureNamesTests(unittest.TestCase):

    def test_new_path_feature_names_present(self) -> None:
        expected = {
            "first_15min_range_atr",
            "first_15min_direction_atr",
            "path_efficiency_30",
            "direction_changes_30",
            "is_gap_up_trap_fade",
            "is_gap_down_reversal",
            "vol_regime_zscore_20d",
        }
        self.assertTrue(expected.issubset(set(STATE_FEATURE_NAMES)),
            f"missing one of {expected - set(STATE_FEATURE_NAMES)}")


class First15MinFeaturesTests(unittest.TestCase):

    def test_zero_before_third_bar_of_session(self) -> None:
        df = _trending_session("2026-05-26", n_bars=10)
        sf = StateFeaturizer(df)
        # Bar 1 (second bar in session) — only 1 future bar in, still under warmup.
        f = sf.features_at(j=1, active_pools=[])
        self.assertEqual(f["first_15min_range_atr"], 0.0)
        self.assertEqual(f["first_15min_direction_atr"], 0.0)

    def test_nonzero_after_third_bar(self) -> None:
        df = _trending_session("2026-05-26", n_bars=10)
        sf = StateFeaturizer(df)
        # Bar 5 — well past 3rd bar of session, features should be live.
        f = sf.features_at(j=5, active_pools=[])
        self.assertGreater(f["first_15min_range_atr"], 0.0)
        # Trending up -> first_15min_direction must be positive.
        self.assertGreater(f["first_15min_direction_atr"], 0.0)


class PathEfficiencyTests(unittest.TestCase):

    def test_trending_session_has_high_efficiency(self) -> None:
        df = _trending_session("2026-05-26", n_bars=40, drift=0.5)
        sf = StateFeaturizer(df)
        f = sf.features_at(j=35, active_pools=[])
        # Pure linear trend -> efficiency should be very close to 1.0.
        self.assertGreater(f["path_efficiency_30"], 0.95,
            f"trending session should have efficiency near 1.0, got "
            f"{f['path_efficiency_30']}")

    def test_choppy_session_has_low_efficiency(self) -> None:
        df = _choppy_session("2026-05-26", n_bars=40)
        sf = StateFeaturizer(df)
        f = sf.features_at(j=35, active_pools=[])
        # Net move ≈ 0, sum |moves| = large -> efficiency near 0.
        self.assertLess(f["path_efficiency_30"], 0.20,
            f"choppy session should have efficiency near 0.0, got "
            f"{f['path_efficiency_30']}")

    def test_choppy_session_has_high_direction_changes(self) -> None:
        df = _choppy_session("2026-05-26", n_bars=40)
        sf = StateFeaturizer(df)
        f = sf.features_at(j=35, active_pools=[])
        # 30-bar window of alternating +/-: expect ~28 direction changes.
        self.assertGreater(f["direction_changes_30"], 20.0,
            f"choppy session should have many direction changes, got "
            f"{f['direction_changes_30']}")


class TrapPatternTests(unittest.TestCase):

    def _gap_up_then_fade(self) -> pd.DataFrame:
        # Day 1: closes at 100. Day 2: opens at 102 (gap up), then fades
        # down to 99 over the first 3 bars (the trap). Then irrelevant.
        idx1 = pd.date_range("2026-05-25 03:45", periods=10, freq="5min")
        d1 = _bars_with_open(idx1, opens=np.full(10, 100.0),
                              closes=np.full(10, 100.0))
        idx2 = pd.date_range("2026-05-26 03:45", periods=20, freq="5min")
        opens2 = np.concatenate(([102.0],            # gap up
                                  np.linspace(101.5, 99.0, 19)))
        closes2 = np.concatenate(([101.5, 100.5, 99.0],   # 3-bar fade
                                    np.linspace(99.0, 99.5, 17)))
        # Highs need to reflect the gap up.
        highs2 = np.maximum(opens2, closes2) + 0.1
        lows2 = np.minimum(opens2, closes2) - 0.1
        d2 = _bars_with_open(idx2, opens=opens2, closes=closes2,
                              highs=highs2, lows=lows2)
        return pd.concat([d1, d2])

    def test_gap_up_trap_fade_fires_on_pattern(self) -> None:
        df = self._gap_up_then_fade()
        sf = StateFeaturizer(df)
        # Day 2 bar 5 (well past first 3 bars) — features should fire on the
        # trap-fade pattern.
        f = sf.features_at(j=15, active_pools=[])      # 5 bars into day 2
        self.assertEqual(f["is_gap_up_trap_fade"], 1.0,
            "gap-up then 15-min fade-down should flag is_gap_up_trap_fade")
        self.assertEqual(f["is_gap_down_reversal"], 0.0)

    def test_continuation_does_not_fire_trap_flag(self) -> None:
        # Day 2 opens gap-up and continues UP (no trap).
        idx1 = pd.date_range("2026-05-25 03:45", periods=10, freq="5min")
        d1 = _bars_with_open(idx1, opens=np.full(10, 100.0),
                              closes=np.full(10, 100.0))
        idx2 = pd.date_range("2026-05-26 03:45", periods=20, freq="5min")
        # Gap up + continues up.
        closes2 = np.linspace(102.0, 106.0, 20)
        opens2 = np.concatenate(([102.0], closes2[:-1]))
        d2 = _bars_with_open(idx2, opens=opens2, closes=closes2)
        df = pd.concat([d1, d2])
        sf = StateFeaturizer(df)
        f = sf.features_at(j=15, active_pools=[])
        self.assertEqual(f["is_gap_up_trap_fade"], 0.0,
            "gap-up continuation should NOT flag is_gap_up_trap_fade")


class VolRegimeTests(unittest.TestCase):

    def test_zscore_zero_with_insufficient_history(self) -> None:
        # 5 days is below the 5-day-minimum threshold for the z-score.
        dfs = [_trending_session(d, n_bars=20)
               for d in ("2026-05-22", "2026-05-25", "2026-05-26",
                          "2026-05-27", "2026-05-28")]
        df = pd.concat(dfs)
        sf = StateFeaturizer(df)
        # Bar 100 lands in day 5 — but day 5's z-score uses only days 0..3
        # (4 prior days) — less than the 5-day minimum -> z=0.
        f = sf.features_at(j=100, active_pools=[])
        self.assertEqual(f["vol_regime_zscore_20d"], 0.0)


class PathCausalityTests(unittest.TestCase):

    def test_features_at_j_unchanged_by_future_bars(self) -> None:
        df_full = _trending_session("2026-05-26", n_bars=40)
        df_trunc = df_full.iloc[:25].copy()
        sf_full = StateFeaturizer(df_full)
        sf_trunc = StateFeaturizer(df_trunc)
        f_full = sf_full.features_at(j=20, active_pools=[])
        f_trunc = sf_trunc.features_at(j=20, active_pools=[])
        for key in ("first_15min_range_atr", "first_15min_direction_atr",
                     "path_efficiency_30", "direction_changes_30",
                     "is_gap_up_trap_fade", "is_gap_down_reversal",
                     "vol_regime_zscore_20d"):
            self.assertAlmostEqual(f_full[key], f_trunc[key], places=6,
                msg=f"{key} leaked future bars at j=20")


if __name__ == "__main__":
    unittest.main()
