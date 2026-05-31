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
    DirectionConfirmedPoolAlpha,
    PolicyReturnAlpha,
    QualityFilteredPoolAlpha,
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


if __name__ == "__main__":
    unittest.main()
