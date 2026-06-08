"""Tests for the D.4 brief integration with the
OptionsExpectedReturnModel suite.

Pinned contracts:

  1. Brief generation accepts optional ``options_er_suite`` +
     ``options_feature_rows`` and does NOT change behaviour when
     either is omitted (backward compatibility).
  2. When both are supplied, each strike entry in
     ``options_suitability[<idx>]["strike_levels_in_play"]`` gains
     ``predicted_net_return_buy_atr`` and
     ``predicted_net_return_sell_atr`` fields.
  3. The renderer surfaces a populated OPTIONS SUITABILITY block
     instead of the pending stub when at least one index block is
     populated.
  4. The renderer text never contains tipster vocabulary
     ("buy ", "sell ", "go long", "go short", "target at", etc.).
     We use "buying-side" / "selling-side" instead.
  5. When the model suite raises mid-prediction, generate_brief
     swallows the error and returns predicted_R = None — the brief
     still ships.
  6. predict_strikes routes correctly: each input dict carries
     ``side``, ``dte_trading_days``, ``tod_bucket`` for the bucket
     lookup; missing-bucket rows are returned with
     ``predict_status="no_head"`` and predicted_R = None.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from liqpool.products.brief_renderer import (
    TIPSTER_VOCABULARY,
    render_email,
)
from liqpool.products.daily_brief import generate_brief
from liqpool.options.expected_return_model import (
    OptionsExpectedReturnModelSuite,
)


# ---------------------------------------------------------------------------
# Re-used minimal report stub from test_daily_brief.py
# ---------------------------------------------------------------------------

@dataclass
class _StubContributor:
    source: str = "EQH"
    ts: Any = field(default_factory=lambda: pd.Timestamp("2026-05-01"))


@dataclass
class _StubPool:
    side: str
    price_low: float
    price_high: float
    tfs: List[str]
    score: float = 1.0
    available_at: Any = None
    formed_at: Any = None
    width: float = 1.0

    def __post_init__(self):
        if self.available_at is None:
            self.available_at = pd.Timestamp("2026-05-01")
        if self.formed_at is None:
            self.formed_at = pd.Timestamp("2026-05-01")
        self.contributors = [_StubContributor()]

    @property
    def mid(self) -> float:
        return (self.price_low + self.price_high) / 2.0


@dataclass
class _StubResult:
    pool_idx: int = 0
    touched_at: Any = None
    broken_at: Any = None


@dataclass
class _StubWalkforward:
    oos_pools: List[Any]
    oos_results: List[Any]


@dataclass
class _StubAssetData:
    base_df: pd.DataFrame
    walkforward: Any


@dataclass
class _StubReport:
    assets: Dict[str, Any]
    unified_direction: Any = None
    unified_proximity: Dict[int, Any] = None
    unified_oos_audit: Any = None

    def __post_init__(self):
        if self.unified_proximity is None:
            self.unified_proximity = {}


class _StubProximityModel:
    def __init__(self, returned_p: float = 0.40) -> None:
        self.returned_p = float(returned_p)
        self.val_auc = 0.85

    def predict_one(self, pool, dist_atr, side, state, quality_pred,
                    atr_val=1.0):
        return self.returned_p


def _index_data_with_strikes() -> Dict[str, Any]:
    """Index data carrying two strike levels in play for NIFTY50."""
    return {
        "NIFTY50": {
            "previous_close": 24521.30,
            "vol_regime": "normal",
            "vol_regime_zscore_20d": 0.34,
            "directional_bias": "mild_up",
            "expected_range_today_atr": 1.6,
            "path_efficiency_30": 0.40,
            "direction_changes_30": 12.0,
            "proximity_predictions": [
                {"level": 24500, "p_test_today": 0.78,
                 "p_test_within_60min": 0.34,
                 "side_from_open": "below",
                 "key_level_type": "demand_pool"},
                {"level": 24700, "p_test_today": 0.45,
                 "p_test_within_60min": 0.18,
                 "side_from_open": "above",
                 "key_level_type": "supply_pool"},
            ],
        },
    }


def _bars(n: int = 80, base: float = 100.0,
          start: str = "2026-05-22 03:45") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = base + np.cumsum(rng.normal(0.0, 0.05, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.2, "low": close - 0.2,
        "close": close, "volume": np.full(n, 1000.0),
    }, index=idx)


def _make_report() -> _StubReport:
    return _StubReport(
        assets={
            "HDFCBANK": _StubAssetData(
                base_df=_bars(),
                walkforward=_StubWalkforward(
                    oos_pools=[
                        _StubPool("low", 90.0, 91.0, tfs=["base", "15min"]),
                    ],
                    oos_results=[_StubResult(pool_idx=0)],
                ),
            ),
        },
        unified_direction=None,
        unified_proximity={12: _StubProximityModel(0.40)},
    )


class _StubSuite:
    """Minimal stub that mimics the predict_strikes interface."""
    def __init__(self, buy_r: float = -0.45, sell_r: float = +0.30):
        self.buy_r, self.sell_r = buy_r, sell_r

    def predict_strikes(self, inputs: List[Dict[str, Any]]
                        ) -> List[Dict[str, Any]]:
        out = []
        for row in inputs:
            r = self.buy_r if row.get("side") == "buy" else self.sell_r
            aug = dict(row)
            aug["predicted_net_return_atr"] = r
            aug["bucket_key"] = (
                f"{row['side']}/weekly/{row.get('tod_bucket', 'open')}")
            aug["predict_status"] = "ok"
            out.append(aug)
        return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class BackwardCompatibilityTests(unittest.TestCase):

    def test_no_suite_no_features_existing_behaviour(self):
        report = _make_report()
        brief = generate_brief(
            report,
            trading_date_ist="2026-06-03",
            indexes_covered=["NIFTY50"],
            index_data=_index_data_with_strikes(),
            # No options_er_suite, no options_feature_rows.
        )
        # The strike entries exist; the new predicted_R fields are None.
        idx_block = brief.options_suitability["NIFTY50"]
        strikes = idx_block["strike_levels_in_play"]
        self.assertGreater(len(strikes), 0)
        for s in strikes:
            self.assertIn("predicted_net_return_buy_atr", s)
            self.assertIn("predicted_net_return_sell_atr", s)
            self.assertIsNone(s["predicted_net_return_buy_atr"])
            self.assertIsNone(s["predicted_net_return_sell_atr"])


class PredictedReturnPopulatedTests(unittest.TestCase):

    def test_suite_provided_populates_predicted_r(self):
        report = _make_report()
        suite = _StubSuite(buy_r=-0.45, sell_r=+0.30)
        feature_rows = {
            ("NIFTY50", 24500): {
                "dte_trading_days": 4, "tod_bucket": "open",
                "underlying_30m_return": 0.0,
                "iv_percentile_60d": 0.55,
            },
            ("NIFTY50", 24700): {
                "dte_trading_days": 4, "tod_bucket": "open",
                "underlying_30m_return": 0.0,
                "iv_percentile_60d": 0.42,
            },
        }
        brief = generate_brief(
            report,
            trading_date_ist="2026-06-03",
            indexes_covered=["NIFTY50"],
            index_data=_index_data_with_strikes(),
            options_er_suite=suite,
            options_feature_rows=feature_rows,
        )
        strikes = brief.options_suitability["NIFTY50"][
            "strike_levels_in_play"]
        self.assertEqual(len(strikes), 2)
        for s in strikes:
            self.assertAlmostEqual(
                s["predicted_net_return_buy_atr"], -0.45, places=4)
            self.assertAlmostEqual(
                s["predicted_net_return_sell_atr"], +0.30, places=4)


class RendererTests(unittest.TestCase):

    def test_options_block_surfaces_when_populated(self):
        report = _make_report()
        suite = _StubSuite()
        feature_rows = {
            ("NIFTY50", 24500): {
                "dte_trading_days": 4, "tod_bucket": "open",
            },
            ("NIFTY50", 24700): {
                "dte_trading_days": 4, "tod_bucket": "open",
            },
        }
        brief = generate_brief(
            report,
            trading_date_ist="2026-06-03",
            indexes_covered=["NIFTY50"],
            index_data=_index_data_with_strikes(),
            options_er_suite=suite,
            options_feature_rows=feature_rows,
        )
        text = render_email(brief)
        # The section header changed from pending stub to real block.
        self.assertIn("OPTIONS SUITABILITY", text)
        self.assertNotIn(
            "OPTIONS SUITABILITY — pending: no_indexes_covered", text)
        # Per-strike prose includes the predicted_R numbers.
        self.assertIn("buying-side net expected return", text)
        self.assertIn("selling-side net expected return", text)
        self.assertIn("-0.45 ATR", text)
        self.assertIn("+0.30 ATR", text)

    def test_no_tipster_vocabulary_in_options_prose(self):
        report = _make_report()
        suite = _StubSuite()
        feature_rows = {
            ("NIFTY50", 24500): {
                "dte_trading_days": 4, "tod_bucket": "open",
            },
            ("NIFTY50", 24700): {
                "dte_trading_days": 4, "tod_bucket": "open",
            },
        }
        brief = generate_brief(
            report,
            trading_date_ist="2026-06-03",
            indexes_covered=["NIFTY50"],
            index_data=_index_data_with_strikes(),
            options_er_suite=suite,
            options_feature_rows=feature_rows,
        )
        text = render_email(brief).lower()
        for phrase in TIPSTER_VOCABULARY:
            self.assertNotIn(phrase, text,
                f"tipster phrase {phrase!r} appeared in rendered brief")


class FailSafeTests(unittest.TestCase):

    def test_suite_exception_does_not_break_brief(self):
        class _Boom:
            def predict_strikes(self, inputs):
                raise RuntimeError("model exploded")
        report = _make_report()
        feature_rows = {
            ("NIFTY50", 24500): {
                "dte_trading_days": 4, "tod_bucket": "open",
            },
        }
        # Must not raise.
        brief = generate_brief(
            report,
            trading_date_ist="2026-06-03",
            indexes_covered=["NIFTY50"],
            index_data=_index_data_with_strikes(),
            options_er_suite=_Boom(),
            options_feature_rows=feature_rows,
        )
        strikes = brief.options_suitability["NIFTY50"][
            "strike_levels_in_play"]
        # Predicted_R fields exist but are None — graceful fall-back.
        for s in strikes:
            self.assertIsNone(s["predicted_net_return_buy_atr"])
            self.assertIsNone(s["predicted_net_return_sell_atr"])


class SuitePredictStrikesTests(unittest.TestCase):

    def test_no_head_returns_status(self):
        # Empty suite (no heads fit).
        suite = OptionsExpectedReturnModelSuite(min_trades=200)
        scored = suite.predict_strikes([
            {"side": "buy", "dte_trading_days": 5, "tod_bucket": "open"},
        ])
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]["predict_status"], "no_head")
        self.assertIsNone(scored[0]["predicted_net_return_atr"])

    def test_out_of_scope_tenor_returns_status(self):
        suite = OptionsExpectedReturnModelSuite(min_trades=200)
        scored = suite.predict_strikes([
            {"side": "buy", "dte_trading_days": 25, "tod_bucket": "open"},
        ])
        self.assertEqual(scored[0]["predict_status"], "out_of_scope")
        self.assertIsNone(scored[0]["predicted_net_return_atr"])


if __name__ == "__main__":
    unittest.main()
