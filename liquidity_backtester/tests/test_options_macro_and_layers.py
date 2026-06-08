"""Tests for the macro scrape + layer-score modules (Stream D.5a).

Pinned contracts:

Macro scrape:
  1. ``_last_dod_pct_from_chart`` correctly extracts DoD percent from
     a Yahoo-shaped payload.
  2. ``_closes_from_chart`` drops trailing ``None`` cleanly.
  3. ``YahooMacroAdapter`` returns a snapshot whose source map tags
     errors per ticker when ``urllib.request.urlopen`` raises.
  4. ``empty_snapshot()`` has every numeric field NaN.

Layer scores:
  5. macro_score returns 0 on a fully-empty snapshot (cascade gate).
  6. macro_score is positive when SPX/DJI/NDX up and negative when
     VIX up.
  7. macro_score skips NaN components without zeroing the average.
  8. regime_score prefers (p_up, path_efficiency) over the proxy.
  9. pool_score sign matches the side ("below" → +, "above" → -).
  10. options_score is negative when IV percentile is high.
  11. micro_score combines direction with cleanness.
  12. manipulation_score: sweep-high → negative, sweep-low → positive.
  13. compute_layer_scores_from_inputs returns the 6-key dict with
      every value in [-1, +1].
"""
from __future__ import annotations

import json
import math
import unittest
import urllib.error
from unittest import mock

from liqpool.options.layer_scores import (
    compute_layer_scores_from_inputs,
    macro_score,
    manipulation_score,
    micro_score,
    options_score,
    pool_score,
    regime_score,
)
from liqpool.options.macro_scrape import (
    MacroOvernightSnapshot,
    YahooMacroAdapter,
    _closes_from_chart,
    _last_dod_pct_from_chart,
    empty_snapshot,
)


# ---------------------------------------------------------------------------
# Yahoo payload fixtures
# ---------------------------------------------------------------------------

def _chart(closes: list) -> dict:
    """Build a minimal Yahoo chart payload around a closes list."""
    return {
        "chart": {
            "result": [{
                "indicators": {
                    "quote": [{"close": closes}],
                },
                "timestamp": list(range(len(closes))),
            }],
        },
    }


# ---------------------------------------------------------------------------
# Macro scrape — pure helpers
# ---------------------------------------------------------------------------

class ChartHelpersTests(unittest.TestCase):

    def test_closes_drops_trailing_none(self):
        payload = _chart([5500.0, 5510.0, 5530.5, None])
        closes = _closes_from_chart(payload)
        self.assertEqual(closes, [5500.0, 5510.0, 5530.5])

    def test_last_dod_pct_is_signed_percent(self):
        payload = _chart([5500.0, 5500.0, 5527.5])
        # last 5527.5, prev 5500 → +0.5%
        self.assertAlmostEqual(
            _last_dod_pct_from_chart(payload), 0.5, places=4)

    def test_last_dod_raises_when_too_few(self):
        with self.assertRaises(ValueError):
            _last_dod_pct_from_chart(_chart([5500.0]))


class YahooAdapterTests(unittest.TestCase):

    def test_sources_tag_errors_per_ticker(self):
        # Always raise on every fetch.
        with mock.patch(
            "liqpool.options.macro_scrape.urllib.request.urlopen",
            side_effect=urllib.error.URLError("boom"),
        ):
            snap = YahooMacroAdapter().fetch()
        for k, status in snap.sources.items():
            self.assertEqual(status, "URLError",
                f"{k} expected URLError tag, got {status}")
        # All fields NaN.
        self.assertTrue(math.isnan(snap.spx_dod_pct))
        self.assertTrue(math.isnan(snap.vix_us_dod_points))


class EmptySnapshotTests(unittest.TestCase):

    def test_every_numeric_field_is_nan(self):
        snap = empty_snapshot()
        for v in (snap.spx_dod_pct, snap.dji_dod_pct, snap.ndx_dod_pct,
                  snap.vix_us_dod_points, snap.nifty_yday_close_pct):
            self.assertTrue(math.isnan(v))
        self.assertFalse(snap.has_any_data)


# ---------------------------------------------------------------------------
# Layer scores — macro
# ---------------------------------------------------------------------------

class MacroScoreTests(unittest.TestCase):

    def test_empty_snapshot_returns_zero(self):
        self.assertEqual(macro_score(snapshot=empty_snapshot()), 0.0)

    def test_us_indexes_up_gives_positive_score(self):
        snap = MacroOvernightSnapshot(
            as_of_utc="t", spx_dod_pct=+0.6,
            dji_dod_pct=+0.4, ndx_dod_pct=+0.8,
            vix_us_dod_points=-0.5,
            nifty_yday_close_pct=+0.4, sources={},
        )
        self.assertGreater(macro_score(snapshot=snap), 0.20)

    def test_vix_up_gives_negative_score(self):
        snap = MacroOvernightSnapshot(
            as_of_utc="t",
            spx_dod_pct=float("nan"),
            dji_dod_pct=float("nan"),
            ndx_dod_pct=float("nan"),
            vix_us_dod_points=+3.0,
            nifty_yday_close_pct=float("nan"),
            sources={},
        )
        self.assertLess(macro_score(snapshot=snap), 0.0)

    def test_nan_components_skipped(self):
        # Two real signals (both +) + NaNs → score should still be +.
        score = macro_score(
            spx_dod_pct=+0.5, ndx_dod_pct=+0.5,
            dji_dod_pct=None, vix_us_dod_points=None,
            usdinr_dod_pct=None, nifty_yday_close_pct=None,
        )
        self.assertGreater(score, 0.0)


# ---------------------------------------------------------------------------
# Layer scores — regime
# ---------------------------------------------------------------------------

class RegimeScoreTests(unittest.TestCase):

    def test_p_up_above_half_is_positive(self):
        score = regime_score(p_up=0.7, path_efficiency=0.8)
        # (2*0.7 - 1) * 0.8 = +0.32
        self.assertAlmostEqual(score, 0.32, places=4)

    def test_proxy_path_when_p_up_missing(self):
        # underlying_30m_return=+0.4% → tanh(0.4 / 0.5) ≈ 0.66
        score = regime_score(underlying_30m_return=0.004)
        self.assertGreater(score, 0.0)

    def test_vol_regime_damps_magnitude(self):
        base = regime_score(p_up=0.7, path_efficiency=0.8)
        damped = regime_score(p_up=0.7, path_efficiency=0.8,
                              vol_regime_zscore_20d=2.0)
        self.assertLess(abs(damped), abs(base))


# ---------------------------------------------------------------------------
# Layer scores — pool
# ---------------------------------------------------------------------------

class PoolScoreTests(unittest.TestCase):

    def test_pool_below_is_positive(self):
        self.assertAlmostEqual(
            pool_score(proximity_p_60min=0.6,
                       pool_side_from_spot="below"),
            +0.6, places=4)

    def test_pool_above_is_negative(self):
        self.assertAlmostEqual(
            pool_score(proximity_p_60min=0.6,
                       pool_side_from_spot="above"),
            -0.6, places=4)

    def test_missing_inputs_return_zero(self):
        self.assertEqual(pool_score(), 0.0)
        self.assertEqual(
            pool_score(proximity_p_60min=0.5,
                       pool_side_from_spot=None), 0.0)


# ---------------------------------------------------------------------------
# Layer scores — options
# ---------------------------------------------------------------------------

class OptionsScoreTests(unittest.TestCase):

    def test_high_iv_is_negative(self):
        score = options_score(iv_percentile_60d=0.90)
        self.assertLess(score, -0.5)

    def test_low_iv_is_positive(self):
        score = options_score(iv_percentile_60d=0.10)
        self.assertGreater(score, +0.5)

    def test_heavy_theta_negative(self):
        score = options_score(theta_per_day_pct=0.10)
        self.assertLess(score, 0.0)


# ---------------------------------------------------------------------------
# Layer scores — micro
# ---------------------------------------------------------------------------

class MicroScoreTests(unittest.TestCase):

    def test_clean_uptrend_gives_positive(self):
        # 5m return + small chop + high path efficiency.
        score = micro_score(
            path_efficiency_30=0.7,
            direction_changes_30=8,
            underlying_5m_return=+0.002,
        )
        self.assertGreater(score, 0.0)

    def test_choppy_neutral_return_near_zero(self):
        score = micro_score(
            path_efficiency_30=0.3,
            direction_changes_30=25,
            underlying_5m_return=0.0,
        )
        self.assertAlmostEqual(score, 0.0, places=4)


# ---------------------------------------------------------------------------
# Layer scores — manipulation
# ---------------------------------------------------------------------------

class ManipulationScoreTests(unittest.TestCase):

    def test_sweep_hi_is_negative(self):
        score = manipulation_score(sweep_hi_detected=True)
        self.assertLess(score, 0.0)

    def test_sweep_lo_is_positive(self):
        score = manipulation_score(sweep_lo_detected=True)
        self.assertGreater(score, 0.0)

    def test_zero_when_no_detectors_fire(self):
        self.assertEqual(manipulation_score(), 0.0)


# ---------------------------------------------------------------------------
# One-call wrapper
# ---------------------------------------------------------------------------

class WrapperTests(unittest.TestCase):

    def test_returns_six_keys_all_in_range(self):
        out = compute_layer_scores_from_inputs(
            macro_snapshot=MacroOvernightSnapshot(
                as_of_utc="t", spx_dod_pct=+0.4, dji_dod_pct=+0.4,
                ndx_dod_pct=+0.4, vix_us_dod_points=-0.5,
                nifty_yday_close_pct=+0.3, sources={}),
            p_up=0.65, path_efficiency_30=0.7,
            direction_changes_30=10, underlying_30m_return=+0.003,
            underlying_5m_return=+0.001,
            proximity_p_60min=0.55, pool_side_from_spot="below",
            iv_percentile_60d=0.55, theta_per_day_pct=0.05,
            dte_trading_days=5, vega_per_volpoint_pct=0.18,
            sweep_lo_detected=True,
        )
        self.assertEqual(set(out.keys()), {
            "macro", "regime", "pool", "options",
            "micro", "manipulation"})
        for k, v in out.items():
            self.assertGreaterEqual(v, -1.0,
                f"{k}={v} below -1 bound")
            self.assertLessEqual(v, 1.0,
                f"{k}={v} above +1 bound")


if __name__ == "__main__":
    unittest.main()
