"""Tests for liqpool.arsenal.evaluator — shared MIS + cost pipeline.

We construct a small synthetic bundle by hand (no parquet, no LightGBM)
and verify the evaluator's contract:

  * Signals outside the session or past 14:30 IST are dropped.
  * Surviving signals produce trades with finite net_r and bounded cost.
  * per_alpha_summary groups by alpha_name and reports CI95.
  * per_regime_summary stratifies by an arbitrary regime column.
  * pairwise_combinations only emits rows when both alphas hit the same
    (symbol, decision_idx).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import unittest

import numpy as np
import pandas as pd

from liqpool.arsenal.base import Alpha, AlphaSignal
from liqpool.arsenal.evaluator import (
    ArsenalEvaluator,
    EvaluatorConfig,
    _execute_signal,
)


# ---------------------------------------------------------------------------
# Synthetic bundle fixtures
# ---------------------------------------------------------------------------

def _bars_ist(n: int, start_ist: str = "10:00", base_price: float = 100.0,
              vol: float = 0.5):
    """5-min OHLCV frame whose IST clock starts at start_ist."""
    h, m = start_ist.split(":")
    ist_min = int(h) * 60 + int(m)
    utc_min = ist_min - (5 * 60 + 30)
    if utc_min < 0:
        utc_min += 24 * 60
    utc_h, utc_m = divmod(utc_min, 60)
    start_utc = f"2026-05-26 {utc_h:02d}:{utc_m:02d}"
    idx = pd.date_range(start_utc, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = base_price + np.cumsum(rng.normal(0, vol, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": rng.uniform(800, 1200, n),
    }, index=idx)


@dataclass
class _Asset:
    base_df: pd.DataFrame
    walkforward: object = None


@dataclass
class _Bundle:
    assets: Dict[str, _Asset]


class _DummyAlpha(Alpha):
    """Emits a fixed list of signals at known bar indices."""
    def __init__(self, name: str, signals_per_symbol: Dict[str, List[Dict]]):
        self._n = name
        self._sigs = signals_per_symbol
    @property
    def name(self) -> str:
        return self._n
    def candidates(self, *, symbol, df_base, atr_series, extra=None):
        out: List[AlphaSignal] = []
        for s in self._sigs.get(symbol, []):
            idx = s["decision_idx"]
            if idx >= len(df_base):
                continue
            out.append(AlphaSignal(
                alpha_name=self._n, symbol=symbol,
                decision_at=df_base.index[idx], decision_idx=idx,
                side=s["side"], entry_reference=float(df_base["close"].iloc[idx]),
                stop_atr=s.get("stop_atr", 0.5),
                target_atr=s.get("target_atr", 2.0),
                horizon_bars=s.get("horizon_bars", 12),
                confidence=s.get("confidence", 0.5),
                state=s.get("state", {}),
            ))
        return out


# ---------------------------------------------------------------------------
# Evaluator: single-trade execution
# ---------------------------------------------------------------------------

class ExecuteSignalTests(unittest.TestCase):

    def setUp(self):
        self.df = _bars_ist(60, start_ist="10:00")
        from liqpool.indicators import atr
        self.atr = atr(self.df, 14).bfill()
        self.cfg = EvaluatorConfig(notional_inr=50_000)

    def _sig(self, idx, side="long", stop=0.5, target=2.0, horizon=12):
        return AlphaSignal(
            alpha_name="t", symbol="HDFCBANK",
            decision_at=self.df.index[idx], decision_idx=idx,
            side=side, entry_reference=float(self.df["close"].iloc[idx]),
            stop_atr=stop, target_atr=target, horizon_bars=horizon,
        )

    def test_normal_signal_produces_finite_trade(self):
        sig = self._sig(idx=20)
        row = _execute_signal(sig, self.df, self.atr, "BANKING", self.cfg)
        self.assertIsNotNone(row)
        self.assertTrue(np.isfinite(row["net_r"]))
        self.assertTrue(np.isfinite(row["net_pnl"]))
        # bars_held should be > 0 and <= horizon.
        self.assertGreaterEqual(row["bars_held"], 1)
        self.assertLessEqual(row["bars_held"], 12)

    def test_volume_profile_fields_are_emitted_when_pool_mid_available(self):
        from liqpool.timing import StateFeaturizer
        sig = AlphaSignal(
            alpha_name="pool_reach", symbol="HDFCBANK",
            decision_at=self.df.index[20], decision_idx=20,
            side="long", entry_reference=float(self.df["close"].iloc[20]),
            stop_atr=0.5, target_atr=2.0, horizon_bars=12,
            confidence=0.55,
            state={
                "pool_mid": float(self.df["close"].iloc[20]),
                "q_pred": 0.55,
            },
        )
        row = _execute_signal(
            sig, self.df, self.atr, "BANKING", self.cfg,
            state_featurizer=StateFeaturizer(self.df),
        )
        self.assertIsNotNone(row)
        self.assertTrue(np.isfinite(row["pool_mid_at_touch"]))
        self.assertTrue(np.isfinite(row["poc_today_at_touch"]))
        self.assertTrue(np.isfinite(row["vah_today_at_touch"]))
        self.assertTrue(np.isfinite(row["val_today_at_touch"]))
        self.assertIn("pool_volume_confirmed_at_touch", row)
        self.assertEqual(row["pool_q_pred"], 0.55)

    def test_signal_with_late_entry_returns_none(self):
        # Build bars starting at 14:00 IST; signal at bar 6 -> entry at 14:35 IST
        # which is past the 14:30 NO_NEW_ENTRY cutoff.
        df = _bars_ist(20, start_ist="14:00")
        from liqpool.indicators import atr
        a = atr(df, 14).bfill()
        sig = AlphaSignal(
            alpha_name="t", symbol="X",
            decision_at=df.index[6], decision_idx=6,
            side="long", entry_reference=float(df["close"].iloc[6]),
            stop_atr=0.5, target_atr=2.0, horizon_bars=12,
        )
        row = _execute_signal(sig, df, a, "OTHER", self.cfg)
        self.assertIsNone(row)

    def test_signal_at_end_of_data_returns_none(self):
        sig = self._sig(idx=len(self.df) - 1)
        row = _execute_signal(sig, self.df, self.atr, "BANKING", self.cfg)
        self.assertIsNone(row)

    def test_min_economic_position_filter_rejects_uneconomic_signal(self):
        # Aggressive ratio (50x) forces rejection: even a healthy target
        # can't be 50x the brokerage on a small notional.
        cfg = EvaluatorConfig(notional_inr=5_000,
                              min_target_to_cost_ratio=50.0)
        sig = self._sig(idx=20, target=0.5)
        row = _execute_signal(sig, self.df, self.atr, "BANKING", cfg)
        self.assertIsNone(row,
            "target reward 50x below fee floor should be filtered")

    def test_min_economic_position_filter_disabled_when_ratio_zero(self):
        # Same uneconomic fixture but ratio=0 -> filter disabled -> trade.
        cfg = EvaluatorConfig(notional_inr=5_000,
                              min_target_to_cost_ratio=0.0)
        sig = self._sig(idx=20, target=0.5)
        row = _execute_signal(sig, self.df, self.atr, "BANKING", cfg)
        self.assertIsNotNone(row,
            "ratio=0.0 should disable the economic filter")

    def test_min_economic_position_filter_passes_large_position(self):
        # ₹2L notional with default target_atr=2.0 — reward easily clears
        # 3x cost. Must trade.
        cfg = EvaluatorConfig(notional_inr=200_000,
                              min_target_to_cost_ratio=3.0)
        sig = self._sig(idx=20)
        row = _execute_signal(sig, self.df, self.atr, "BANKING", cfg)
        self.assertIsNotNone(row)


# ---------------------------------------------------------------------------
# Evaluator: end-to-end
# ---------------------------------------------------------------------------

class ArsenalEvaluatorTests(unittest.TestCase):

    def _bundle(self):
        # Two assets, both with bars from 10:00 IST.
        df_a = _bars_ist(80, start_ist="10:00", base_price=100.0)
        df_b = _bars_ist(80, start_ist="10:00", base_price=500.0)
        return _Bundle(assets={
            "HDFCBANK": _Asset(base_df=df_a),
            "TCS":      _Asset(base_df=df_b),
        })

    def test_evaluator_requires_at_least_one_alpha(self):
        with self.assertRaises(ValueError):
            ArsenalEvaluator([])

    def test_evaluator_rejects_non_alpha(self):
        with self.assertRaises(TypeError):
            ArsenalEvaluator(["not an alpha"])    # type: ignore[list-item]

    def test_run_produces_trades_for_in_session_signals(self):
        # Signals at bar 20 and 30 should produce trades (both are well within
        # the session window when start_ist="10:00").
        alpha = _DummyAlpha("dummy", {
            "HDFCBANK": [
                {"decision_idx": 20, "side": "long"},
                {"decision_idx": 30, "side": "short"},
            ],
            "TCS": [{"decision_idx": 25, "side": "long"}],
        })
        evaluator = ArsenalEvaluator([alpha])
        trades = evaluator.run(self._bundle())
        self.assertEqual(len(trades), 3)
        self.assertTrue((trades["alpha_name"] == "dummy").all())
        # Both sides present.
        self.assertEqual(set(trades["side"]), {"long", "short"})
        # Regime tags should be attached.
        self.assertIn("regime_session", trades.columns)
        self.assertIn("regime_side", trades.columns)

    def test_run_filters_signals_past_1430(self):
        # Build alpha that fires at a bar past 14:30 IST in the synthetic data.
        # Bars are 5min from 10:00 IST -> bar 60 = 15:00 IST -> entry at 15:05
        # past cutoff (14:30 = bar 54 if we want entry exactly there).
        alpha = _DummyAlpha("late", {
            "HDFCBANK": [{"decision_idx": 60, "side": "long"}],
        })
        evaluator = ArsenalEvaluator([alpha])
        trades = evaluator.run(self._bundle())
        # The signal should be filtered out.
        self.assertEqual(len(trades), 0)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

class AggregationHelperTests(unittest.TestCase):

    def _trades(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "alpha_name": rng.choice(["a", "b"], n),
            "symbol": rng.choice(["X", "Y"], n),
            "decision_idx": rng.integers(0, 100, n),
            "side": rng.choice(["long", "short"], n),
            "regime_session": rng.choice(["morning", "midday"], n),
            "regime_side": rng.choice(["long", "short"], n),
            "net_r": rng.normal(0.05, 1.0, n),
            "net_pnl": rng.normal(50.0, 200.0, n),
            "cost_inr": rng.uniform(2, 10, n),
        })

    def test_per_alpha_summary_groups_and_computes_ci(self):
        trades = self._trades(300)
        s = ArsenalEvaluator.per_alpha_summary(trades)
        self.assertEqual(set(s["alpha_name"]), {"a", "b"})
        for _, r in s.iterrows():
            self.assertGreater(r["trades"], 0)
            self.assertLess(r["ci95_lo"], r["mean_R"])
            self.assertGreater(r["ci95_hi"], r["mean_R"])

    def test_per_alpha_summary_empty_input(self):
        s = ArsenalEvaluator.per_alpha_summary(pd.DataFrame())
        self.assertTrue(s.empty)

    def test_per_regime_summary_stratifies(self):
        trades = self._trades(400)
        s = ArsenalEvaluator.per_regime_summary(trades, "regime_session")
        # Expect 4 (alpha, session) cells -> 4 rows.
        self.assertEqual(len(s), 4)
        for _, r in s.iterrows():
            self.assertIn(r["regime"], {"morning", "midday"})

    def test_pairwise_combinations_finds_overlaps(self):
        # Hand-construct alpha-a and alpha-b trades that overlap at known
        # (symbol, decision_idx) pairs.
        rows = []
        for i in range(50):
            rows.append({"alpha_name": "a", "symbol": "X", "decision_idx": i,
                          "net_r": 0.5, "net_pnl": 50.0})
            rows.append({"alpha_name": "b", "symbol": "X", "decision_idx": i,
                          "net_r": 0.3, "net_pnl": 30.0})
        # And 50 non-overlapping a-only signals
        for i in range(50, 100):
            rows.append({"alpha_name": "a", "symbol": "X", "decision_idx": i,
                          "net_r": -0.2, "net_pnl": -20.0})
        trades = pd.DataFrame(rows)
        pw = ArsenalEvaluator.pairwise_combinations(trades, min_overlap=10)
        self.assertEqual(len(pw), 1)
        # Combined = avg of 0.5 + 0.3 = 0.4
        self.assertAlmostEqual(pw.iloc[0]["combined_mean_R"], 0.4, places=5)
        self.assertEqual(pw.iloc[0]["overlap_n"], 50)


if __name__ == "__main__":
    unittest.main()
