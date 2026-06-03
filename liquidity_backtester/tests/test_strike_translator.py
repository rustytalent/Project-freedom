"""Tests for the level-to-strike translator.

Pinned contracts:
  * All 5 indexes have their strike step and lot size pinned.
  * nearest_strike snaps to the correct grid.
  * translate_proximity filters by min p_test_today threshold.
  * theta_danger_score is in [0, 1] across the input domain.
  * premium regime mapping is monotone in theta_score.
"""
from __future__ import annotations

import unittest

from liqpool.products.strike_translator import (
    INDEX_CONFIGS,
    StrikeLevelInPlay,
    is_known_index,
    nearest_strike,
    nearest_strikes,
    premium_regime_for_buyers_and_sellers,
    theta_danger_score,
    translate_proximity,
)


class IndexConfigTests(unittest.TestCase):

    def test_all_five_indexes_present(self) -> None:
        expected = {"NIFTY50", "BANKNIFTY", "FINNIFTY",
                     "NIFTYMIDCAPSELECT", "SENSEX"}
        self.assertEqual(set(INDEX_CONFIGS), expected)

    def test_strike_steps_pinned(self) -> None:
        self.assertEqual(INDEX_CONFIGS["NIFTY50"].strike_step, 50.0)
        self.assertEqual(INDEX_CONFIGS["BANKNIFTY"].strike_step, 100.0)
        self.assertEqual(INDEX_CONFIGS["FINNIFTY"].strike_step, 50.0)
        self.assertEqual(INDEX_CONFIGS["NIFTYMIDCAPSELECT"].strike_step, 25.0)
        self.assertEqual(INDEX_CONFIGS["SENSEX"].strike_step, 100.0)

    def test_lot_sizes_pinned(self) -> None:
        # Pin current values so a regression here surfaces immediately.
        self.assertEqual(INDEX_CONFIGS["NIFTY50"].lot_size, 75)
        self.assertEqual(INDEX_CONFIGS["BANKNIFTY"].lot_size, 30)
        self.assertEqual(INDEX_CONFIGS["FINNIFTY"].lot_size, 65)


class NearestStrikeTests(unittest.TestCase):

    def test_snaps_nifty_to_nearest_50(self) -> None:
        self.assertEqual(nearest_strike(24512.0, "NIFTY50"), 24500.0)
        self.assertEqual(nearest_strike(24530.0, "NIFTY50"), 24550.0)
        self.assertEqual(nearest_strike(24500.0, "NIFTY50"), 24500.0)

    def test_snaps_banknifty_to_nearest_100(self) -> None:
        self.assertEqual(nearest_strike(52042.0, "BANKNIFTY"), 52000.0)
        self.assertEqual(nearest_strike(52050.0, "BANKNIFTY"), 52000.0)
        self.assertEqual(nearest_strike(52051.0, "BANKNIFTY"), 52100.0)

    def test_snaps_midcap_to_nearest_25(self) -> None:
        self.assertEqual(nearest_strike(12463.0, "NIFTYMIDCAPSELECT"), 12475.0)

    def test_unknown_index_raises(self) -> None:
        with self.assertRaises(KeyError):
            nearest_strike(100.0, "DOWJONES")

    def test_nearest_strikes_k_returns_surrounding(self) -> None:
        strikes = nearest_strikes(24512.0, "NIFTY50", k=4)
        self.assertIn(24500.0, strikes)
        self.assertIn(24450.0, strikes)
        self.assertIn(24550.0, strikes)


class TranslateProximityTests(unittest.TestCase):

    def test_emits_strike_per_qualifying_level(self) -> None:
        preds = [
            {"level": 24512.0, "p_test_today": 0.80,
             "p_test_within_60min": 0.40, "side_from_open": "below",
             "key_level_type": "demand_pool"},
            {"level": 24735.0, "p_test_today": 0.35,
             "p_test_within_60min": 0.10, "side_from_open": "above",
             "key_level_type": "supply_pool"},
            {"level": 24800.0, "p_test_today": 0.05,    # below threshold
             "p_test_within_60min": 0.01, "side_from_open": "above",
             "key_level_type": "supply_pool"},
        ]
        out = translate_proximity(preds, "NIFTY50")
        self.assertEqual(len(out), 2)            # third filtered by threshold
        # Strikes round to the grid:
        self.assertEqual(out[0].strike, 24500.0)
        self.assertEqual(out[1].strike, 24750.0)

    def test_sorted_by_p_test_today_descending(self) -> None:
        preds = [
            {"level": 24735.0, "p_test_today": 0.35,
             "p_test_within_60min": 0.10, "side_from_open": "above",
             "key_level_type": "supply_pool"},
            {"level": 24512.0, "p_test_today": 0.80,
             "p_test_within_60min": 0.40, "side_from_open": "below",
             "key_level_type": "demand_pool"},
        ]
        out = translate_proximity(preds, "NIFTY50")
        self.assertGreaterEqual(out[0].p_test_today, out[1].p_test_today)
        self.assertEqual(out[0].strike, 24500.0)

    def test_threshold_filtering(self) -> None:
        preds = [
            {"level": 24500.0, "p_test_today": 0.20,
             "p_test_within_60min": 0.05, "side_from_open": "below",
             "key_level_type": "demand_pool"},
        ]
        # Default threshold 0.15 lets 0.20 through.
        self.assertEqual(len(translate_proximity(preds, "NIFTY50")), 1)
        # Raised threshold filters it.
        self.assertEqual(
            len(translate_proximity(preds, "NIFTY50",
                                     min_p_test_today=0.50)),
            0)


class ThetaDangerTests(unittest.TestCase):

    def test_low_efficiency_high_change_high_vol_returns_high(self) -> None:
        s = theta_danger_score(path_efficiency_30=0.10,
                                direction_changes_30=25.0,
                                vol_regime_zscore_20d=1.5)
        self.assertGreater(s, 0.65,
            f"chop + change + vol should produce high theta danger; got {s}")

    def test_high_efficiency_low_change_low_vol_returns_low(self) -> None:
        s = theta_danger_score(path_efficiency_30=0.95,
                                direction_changes_30=2.0,
                                vol_regime_zscore_20d=-0.5)
        self.assertLess(s, 0.30,
            f"trending + steady + low-vol should produce low theta danger; got {s}")

    def test_score_in_unit_interval(self) -> None:
        # Sweep extremes — must stay clamped to [0, 1].
        for ef in (-1.0, 0.0, 0.5, 1.0, 5.0):
            for dc in (0.0, 10.0, 50.0, 100.0):
                for vol in (-3.0, 0.0, 3.0):
                    s = theta_danger_score(ef, dc, vol)
                    self.assertGreaterEqual(s, 0.0)
                    self.assertLessEqual(s, 1.0)


class PremiumRegimeTests(unittest.TestCase):

    def test_high_theta_favours_sellers(self) -> None:
        r = premium_regime_for_buyers_and_sellers(0.80)
        self.assertEqual(r["regime_for_premium_buyers"], "unfavourable")
        self.assertEqual(r["regime_for_premium_sellers"], "favourable")

    def test_low_theta_favours_buyers(self) -> None:
        r = premium_regime_for_buyers_and_sellers(0.20)
        self.assertEqual(r["regime_for_premium_buyers"], "favourable")
        self.assertEqual(r["regime_for_premium_sellers"], "unfavourable")

    def test_mid_theta_is_moderate_both_sides(self) -> None:
        r = premium_regime_for_buyers_and_sellers(0.50)
        self.assertEqual(r["regime_for_premium_buyers"], "moderate")
        self.assertEqual(r["regime_for_premium_sellers"], "moderate")


class IsKnownIndexTests(unittest.TestCase):

    def test_known(self) -> None:
        self.assertTrue(is_known_index("NIFTY50"))
        self.assertTrue(is_known_index("BANKNIFTY"))

    def test_unknown(self) -> None:
        self.assertFalse(is_known_index("DAX"))


if __name__ == "__main__":
    unittest.main()
