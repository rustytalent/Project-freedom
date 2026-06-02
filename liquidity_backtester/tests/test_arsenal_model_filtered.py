"""Tests for the model-wrapped alphas (QualityFilteredPool /
DirectionConfirmedPool / PolicyReturnAlpha).

These tests use stub model objects to avoid needing a real saved bundle.
The contract we pin:

  * Each wrapped alpha graceful-falls-back (returns []) if its required
    bundle attribute is missing — never crashes.
  * Each correctly filters pool-reach signals by its model's prediction.
  * Thresholds and naming are configurable so threshold sweeps work.
  * Regime tags include the alpha-specific dimension (q_bucket /
    direction_alignment_bucket / predicted_r_bucket).
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from liqpool.arsenal.alphas import (
    DistanceBandJourneyAlpha,
    DirectionConfirmedPoolAlpha,
    OpeningRangeToPoolAlpha,
    PolicyReturnAlpha,
    ProximityFilteredPoolAlpha,
    ProximityDirectionSoftAlpha,
    QualityFilteredPoolAlpha,
    SectorRotationJourneyAlpha,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic bundle + stub models
# ---------------------------------------------------------------------------

def _bars_ist(n, start_ist="10:00", base_price=100.0):
    h, m = start_ist.split(":")
    ist_min = int(h) * 60 + int(m)
    utc_min = ist_min - (5 * 60 + 30)
    if utc_min < 0:
        utc_min += 24 * 60
    utc_h, utc_m = divmod(utc_min, 60)
    start_utc = f"2026-05-26 {utc_h:02d}:{utc_m:02d}"
    idx = pd.date_range(start_utc, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = base_price + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": rng.uniform(800, 1200, n),
    }, index=idx)


@dataclass
class _Contributor:
    source: str = "EQH"


@dataclass
class _Pool:
    side: str
    price_low: float
    price_high: float
    tfs: list
    score: float = 1.0
    formed_at: object = None
    available_at: object = None
    contributors: list = None
    width: float = 1.0

    def __post_init__(self):
        if self.contributors is None:
            self.contributors = [_Contributor("EQH")]
        if self.available_at is None:
            self.available_at = pd.Timestamp("2026-05-26 04:00")
        if self.formed_at is None:
            self.formed_at = self.available_at

    @property
    def mid(self):
        return (self.price_low + self.price_high) / 2.0


@dataclass
class _Result:
    pool_idx: int
    touched_at: object
    side: str = "low"
    pool_quality: float = 0.5
    formed_at: object = pd.Timestamp("2026-05-26 04:00")
    price_low: float = 100.0
    price_high: float = 101.0
    score: float = 1.0
    outcome: str = "respected_strong"


@dataclass
class _WF:
    oos_pools: list
    oos_results: list


@dataclass
class _AD:
    base_df: pd.DataFrame
    walkforward: _WF


class _StubQModel:
    """Returns a configurable Q prediction per pool (indexed in order)."""
    def __init__(self, predictions):
        self._preds = list(predictions)
    def predict(self, X, pools=None):
        return np.asarray(self._preds[:len(X)], dtype=float)


class _StubFeaturizer:
    def transform_batch(self, pools):
        # Return a dummy DataFrame with len(pools) rows
        return pd.DataFrame({"x": np.zeros(len(pools))})


class _StubDirectionModel:
    """Returns a constant P(up) for the requested state."""
    def __init__(self, p_up: float):
        self.p_up = p_up
    def predict_state(self, state):
        return self.p_up


class _StubReport:
    def __init__(self, unified_ml=None, unified_featurizer=None,
                 unified_direction=None, policy_return_model=None,
                 assets=None):
        self.unified_ml = unified_ml
        self.unified_featurizer = unified_featurizer
        self.unified_direction = unified_direction
        self.policy_return_model = policy_return_model
        self.assets = assets or {}


def _two_touched_pools(df):
    pools = [
        _Pool(side="low", price_low=95.0, price_high=96.0, tfs=["base"]),
        _Pool(side="high", price_low=105.0, price_high=106.0, tfs=["base"]),
    ]
    results = [
        _Result(pool_idx=0, touched_at=df.index[10]),
        _Result(pool_idx=1, touched_at=df.index[25], side="high"),
    ]
    return _AD(base_df=df, walkforward=_WF(oos_pools=pools, oos_results=results))


# ---------------------------------------------------------------------------
# QualityFilteredPoolAlpha
# ---------------------------------------------------------------------------

class QualityFilteredPoolAlphaTests(unittest.TestCase):

    def test_emits_empty_when_required_bundle_attrs_missing(self):
        df = _bars_ist(50, "10:00")
        ad = _two_touched_pools(df)
        # No unified_ml + no unified_featurizer in report -> no crash, empty.
        report = _StubReport()
        alpha = QualityFilteredPoolAlpha(min_q=0.50)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertEqual(sigs, [])

    def test_filters_by_threshold(self):
        df = _bars_ist(50, "10:00")
        ad = _two_touched_pools(df)
        # Pool 0 gets Q=0.65 (above 0.55), pool 1 gets Q=0.40 (below).
        report = _StubReport(
            unified_ml=_StubQModel([0.65, 0.40]),
            unified_featurizer=_StubFeaturizer(),
        )
        alpha = QualityFilteredPoolAlpha(min_q=0.55)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].state["q_pred"], 0.65)
        self.assertEqual(sigs[0].side, "long")            # pool 0 was side=low

    def test_low_threshold_keeps_both(self):
        df = _bars_ist(50, "10:00")
        ad = _two_touched_pools(df)
        report = _StubReport(
            unified_ml=_StubQModel([0.65, 0.40]),
            unified_featurizer=_StubFeaturizer(),
        )
        alpha = QualityFilteredPoolAlpha(min_q=0.30)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertEqual(len(sigs), 2)

    def test_high_threshold_emits_nothing(self):
        df = _bars_ist(50, "10:00")
        ad = _two_touched_pools(df)
        report = _StubReport(
            unified_ml=_StubQModel([0.65, 0.40]),
            unified_featurizer=_StubFeaturizer(),
        )
        alpha = QualityFilteredPoolAlpha(min_q=0.95)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertEqual(sigs, [])

    def test_custom_name_is_respected(self):
        alpha = QualityFilteredPoolAlpha(min_q=0.55, name="q_top_45pct")
        self.assertEqual(alpha.name, "q_top_45pct")

    def test_regime_tag_includes_q_bucket(self):
        df = _bars_ist(50)
        ad = _two_touched_pools(df)
        report = _StubReport(
            unified_ml=_StubQModel([0.60, 0.20]),
            unified_featurizer=_StubFeaturizer(),
        )
        alpha = QualityFilteredPoolAlpha(min_q=0.10)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        tags_high = alpha.regime_tags(sigs[0], df, None)
        tags_low = alpha.regime_tags(sigs[1], df, None)
        # 0.60 is q_top_25 (>= 0.55), 0.20 is q_below_50
        self.assertEqual(tags_high["q_bucket"], "q_top_25")
        self.assertEqual(tags_low["q_bucket"], "q_below_50")


# ---------------------------------------------------------------------------
# DirectionConfirmedPoolAlpha
# ---------------------------------------------------------------------------

class DirectionConfirmedPoolAlphaTests(unittest.TestCase):

    def test_init_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            DirectionConfirmedPoolAlpha(min_direction=0.40)        # < 0.50
        with self.assertRaises(ValueError):
            DirectionConfirmedPoolAlpha(min_direction=0.99)        # > 0.95

    def test_returns_empty_without_direction_model(self):
        df = _bars_ist(50)
        ad = _two_touched_pools(df)
        report = _StubReport()
        sigs = DirectionConfirmedPoolAlpha().candidates(
            symbol="X", df_base=df, atr_series=None,
            extra={"asset_data": ad, "report": report},
        )
        self.assertEqual(sigs, [])

    def test_p_up_high_keeps_long_drops_short(self):
        # P(up) = 0.70 -> long allowed (>= 0.55), short requires (1-p) >= 0.55
        # so 1 - 0.70 = 0.30 < 0.55 -> short rejected.
        df = _bars_ist(50)
        ad = _two_touched_pools(df)
        report = _StubReport(unified_direction=_StubDirectionModel(p_up=0.70))
        alpha = DirectionConfirmedPoolAlpha(min_direction=0.55)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        # Pool 0 is side=low -> long: kept. Pool 1 is side=high -> short: dropped.
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].side, "long")

    def test_p_up_low_keeps_short_drops_long(self):
        df = _bars_ist(50)
        ad = _two_touched_pools(df)
        report = _StubReport(unified_direction=_StubDirectionModel(p_up=0.30))
        alpha = DirectionConfirmedPoolAlpha(min_direction=0.55)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].side, "short")

    def test_neutral_p_up_drops_both_sides(self):
        df = _bars_ist(50)
        ad = _two_touched_pools(df)
        report = _StubReport(unified_direction=_StubDirectionModel(p_up=0.50))
        alpha = DirectionConfirmedPoolAlpha(min_direction=0.55)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertEqual(sigs, [])


# ---------------------------------------------------------------------------
# PolicyReturnAlpha
# ---------------------------------------------------------------------------

class _StubPolicyReturnModel:
    """Predicts a per-row R from a frame; index-aligned with input order."""
    def __init__(self, predictions):
        self._preds = list(predictions)
    def predict_frame(self, frame):
        return np.asarray(self._preds[:len(frame)], dtype=float)


class _StubPolicyReturnSuite:
    def __init__(self, models_by_mode):
        self.models = dict(models_by_mode)


class PolicyReturnAlphaTests(unittest.TestCase):

    def test_returns_empty_without_policy_return_model(self):
        df = _bars_ist(50)
        ad = _two_touched_pools(df)
        report = _StubReport()
        sigs = PolicyReturnAlpha(execution_mode="touch_confirmed").candidates(
            symbol="X", df_base=df, atr_series=None,
            extra={"asset_data": ad, "report": report},
        )
        self.assertEqual(sigs, [])

    def test_returns_empty_if_mode_not_in_suite(self):
        df = _bars_ist(50)
        ad = _two_touched_pools(df)
        report = _StubReport(policy_return_model=_StubPolicyReturnSuite({}))
        sigs = PolicyReturnAlpha(execution_mode="touch_confirmed").candidates(
            symbol="X", df_base=df, atr_series=None,
            extra={"asset_data": ad, "report": report},
        )
        self.assertEqual(sigs, [])

    def test_emits_only_above_predicted_r_threshold(self):
        df = _bars_ist(50)
        ad = _two_touched_pools(df)
        # First pool predicted +0.35R (above 0), second -0.10R (below).
        suite = _StubPolicyReturnSuite({
            "touch_confirmed": _StubPolicyReturnModel([+0.35, -0.10]),
        })
        report = _StubReport(policy_return_model=suite)
        sigs = PolicyReturnAlpha(
            execution_mode="touch_confirmed", min_predicted_r=0.0,
        ).candidates(symbol="X", df_base=df, atr_series=None,
                     extra={"asset_data": ad, "report": report})
        self.assertEqual(len(sigs), 1)
        self.assertAlmostEqual(sigs[0].state["predicted_r"], 0.35, places=6)


# ---------------------------------------------------------------------------
# ProximityFilteredPoolAlpha
# ---------------------------------------------------------------------------

class _StubProximityModel:
    """Returns a configurable P(touch within horizon) per call.

    Records each call so tests can assert what was queried.
    """
    def __init__(self, return_value: float = 0.75):
        self.return_value = float(return_value)
        self.calls: list = []

    def predict_one(self, pool, dist_atr, side, state, quality_pred,
                    atr_val=1.0):
        self.calls.append({
            "pool_idx": getattr(pool, "_idx", None),
            "dist_atr": float(dist_atr),
            "side": side,
            "quality_pred": float(quality_pred),
            "atr_val": float(atr_val),
        })
        return self.return_value


def _far_pools_for_journey(df):
    """One pool well above current price, one well below.

    Picked so both are out-of-zone for the journey alpha and produce
    distinct trade sides (long toward the above-pool, short toward below).
    """
    p_above = _Pool(side="high", price_low=110.0, price_high=111.0,
                    tfs=["base"])
    p_above._idx = 0
    p_below = _Pool(side="low", price_low=90.0, price_high=91.0,
                    tfs=["base"])
    p_below._idx = 1
    # Touch-times AFTER the candidate-emission window so the alpha sees them
    # as still-untouched at decision time.
    r_above = _Result(pool_idx=0, touched_at=df.index[-1], side="high")
    r_below = _Result(pool_idx=1, touched_at=df.index[-1], side="low")
    return _AD(base_df=df, walkforward=_WF(oos_pools=[p_above, p_below],
                                             oos_results=[r_above, r_below]))


class _ProximityReport(_StubReport):
    def __init__(self, prox_model, *, primary_h: int = 12,
                 unified_ml=None, unified_featurizer=None, assets=None):
        super().__init__(unified_ml=unified_ml,
                         unified_featurizer=unified_featurizer, assets=assets)
        self.unified_proximity = {primary_h: prox_model}


class ProximityFilteredPoolAlphaTests(unittest.TestCase):

    def _bars(self, n=120):
        # Long enough that with sample_every=12 and warm-up=80, several
        # decision bars fit before the last_decision = n - 12 - 1 cap.
        return _bars_ist(n, "10:00", base_price=100.0)

    def test_init_rejects_invalid_args(self):
        with self.assertRaises(ValueError):
            ProximityFilteredPoolAlpha(min_p_touch=0.0)
        with self.assertRaises(ValueError):
            ProximityFilteredPoolAlpha(min_p_touch=1.0)
        with self.assertRaises(ValueError):
            ProximityFilteredPoolAlpha(max_dist_atr=0.0)
        with self.assertRaises(ValueError):
            ProximityFilteredPoolAlpha(sample_every=0)

    def test_returns_empty_without_proximity_model(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        report = _StubReport()
        sigs = ProximityFilteredPoolAlpha().candidates(
            symbol="X", df_base=df, atr_series=None,
            extra={"asset_data": ad, "report": report},
        )
        self.assertEqual(sigs, [])

    def test_returns_empty_without_quality_model(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        report = _ProximityReport(_StubProximityModel(0.80))
        # No unified_ml in report -> can't compute Q -> empty.
        sigs = ProximityFilteredPoolAlpha().candidates(
            symbol="X", df_base=df, atr_series=None,
            extra={"asset_data": ad, "report": report},
        )
        self.assertEqual(sigs, [])

    def test_emits_journey_signals_above_threshold(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        prox = _StubProximityModel(return_value=0.80)         # above 0.60
        report = _ProximityReport(prox,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        # Wide max_dist_atr so the synthetic 10-ATR-away pools qualify.
        alpha = ProximityFilteredPoolAlpha(min_p_touch=0.60,
                                            max_dist_atr=50.0,
                                            sample_every=12)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertEqual(len(sigs), 2,
            "should emit one signal per pool (dedup keeps the earliest)")
        sides = {s.side for s in sigs}
        self.assertEqual(sides, {"long", "short"})
        # Confidence == p_touch.
        for s in sigs:
            self.assertAlmostEqual(s.confidence, 0.80, places=6)
            self.assertAlmostEqual(s.state["p_touch"], 0.80, places=6)

    def test_below_threshold_emits_nothing(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        prox = _StubProximityModel(return_value=0.50)         # < 0.60
        report = _ProximityReport(prox,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        alpha = ProximityFilteredPoolAlpha(min_p_touch=0.60,
                                            max_dist_atr=50.0)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertEqual(sigs, [])

    def test_dedup_one_signal_per_pool(self):
        df = self._bars(n=200)
        ad = _far_pools_for_journey(df)
        prox = _StubProximityModel(return_value=0.80)
        report = _ProximityReport(prox,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        alpha = ProximityFilteredPoolAlpha(min_p_touch=0.60,
                                            max_dist_atr=50.0,
                                            sample_every=6)    # many samples
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        pool_idxs = [s.state["pool_idx"] for s in sigs]
        self.assertEqual(sorted(pool_idxs), sorted(set(pool_idxs)),
            "each pool should appear at most once")

    def test_skips_pools_already_touched(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        # Move pool 0's touched_at to BEFORE the warm-up so the alpha treats
        # it as already-touched at every decision bar.
        ad.walkforward.oos_results[0].touched_at = df.index[10]
        prox = _StubProximityModel(return_value=0.80)
        report = _ProximityReport(prox,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        alpha = ProximityFilteredPoolAlpha(min_p_touch=0.60,
                                            max_dist_atr=50.0)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        # Only pool 1 (untouched until last bar) qualifies.
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].state["pool_idx"], 1)

    def test_target_atr_capped_by_max_target(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        prox = _StubProximityModel(return_value=0.80)
        report = _ProximityReport(prox,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        # Tiny cap forces target_atr = 2.0 regardless of how far the pool is.
        alpha = ProximityFilteredPoolAlpha(min_p_touch=0.60,
                                            max_dist_atr=50.0,
                                            max_target_atr=2.0)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        for s in sigs:
            self.assertLessEqual(s.target_atr, 2.0 + 1e-9)

    def test_max_dist_atr_excludes_far_pools(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        prox = _StubProximityModel(return_value=0.80)
        report = _ProximityReport(prox,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        # Pools sit ~10 ATR away; max_dist_atr=1.0 excludes them all.
        alpha = ProximityFilteredPoolAlpha(min_p_touch=0.60, max_dist_atr=1.0)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertEqual(sigs, [])

    def test_signal_carries_horizon_from_proximity_model(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        prox = _StubProximityModel(return_value=0.80)
        # Two horizons in the bundle — primary is the smallest (12).
        report = _ProximityReport(prox, primary_h=12,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        report.unified_proximity[36] = prox                   # second horizon
        alpha = ProximityFilteredPoolAlpha(min_p_touch=0.60,
                                            max_dist_atr=50.0)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        for s in sigs:
            self.assertEqual(s.horizon_bars, 12)
            self.assertEqual(s.state["proximity_horizon"], 12)

    def test_regime_tags_include_p_touch_and_dist_buckets(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        prox = _StubProximityModel(return_value=0.85)
        report = _ProximityReport(prox,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        alpha = ProximityFilteredPoolAlpha(min_p_touch=0.60,
                                            max_dist_atr=50.0)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        tags = alpha.regime_tags(sigs[0], df, None)
        self.assertEqual(tags["p_touch_bucket"], "p_high")
        self.assertIn(tags["dist_bucket"], {"near_0_1_atr", "mid_1_3_atr",
                                               "far_3_plus_atr"})
        self.assertIn("score_bucket", tags)

    def test_soft_direction_variant_uses_direction_without_hard_gate(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        prox = _StubProximityModel(return_value=0.80)
        report = _ProximityReport(prox,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        report.unified_direction = _StubDirectionModel(p_up=0.70)
        alpha = ProximityDirectionSoftAlpha(max_dist_atr=50.0,
                                            min_score=0.10)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertGreaterEqual(len(sigs), 1)
        self.assertTrue(any("p_direction_to_pool" in s.state for s in sigs))
        self.assertTrue(all("score_direction" in s.state for s in sigs))

    def test_distance_band_variant_filters_to_configured_band(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        prox = _StubProximityModel(return_value=0.80)
        report = _ProximityReport(prox,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        # Synthetic pools sit far from price, so a 0-1 ATR band should reject.
        alpha = DistanceBandJourneyAlpha(distance_band=(0.0, 1.0),
                                         max_dist_atr=50.0,
                                         min_score=0.10)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertEqual(sigs, [])

    def test_sector_rotation_variant_emits_and_scores_components(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        prox = _StubProximityModel(return_value=0.90)
        report = _ProximityReport(prox,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        report.unified_direction = _StubDirectionModel(p_up=0.70)
        report.assets = {"X": ad}
        alpha = SectorRotationJourneyAlpha(max_dist_atr=50.0,
                                           min_score=0.10)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report,
                                       "sector": "AUTO"})
        self.assertGreaterEqual(len(sigs), 1)
        self.assertTrue(all("score_sector" in s.state for s in sigs))
        self.assertTrue(all("score_vol" in s.state for s in sigs))

    def test_opening_range_variant_is_constructible_and_graceful(self):
        df = self._bars()
        ad = _far_pools_for_journey(df)
        prox = _StubProximityModel(return_value=0.90)
        report = _ProximityReport(prox,
                                  unified_ml=_StubQModel([0.50, 0.50]),
                                  unified_featurizer=_StubFeaturizer())
        report.unified_direction = _StubDirectionModel(p_up=0.70)
        alpha = OpeningRangeToPoolAlpha(max_dist_atr=50.0,
                                        min_score=0.10)
        sigs = alpha.candidates(symbol="X", df_base=df, atr_series=None,
                                extra={"asset_data": ad, "report": report})
        self.assertIsInstance(sigs, list)


if __name__ == "__main__":
    unittest.main()
