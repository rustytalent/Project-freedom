"""Tests for liqpool.arsenal.null_tests — permutation null harness.

Hard to test deeply without a real bundle, but we CAN verify:
  * NullResult dataclass schema is stable.
  * _shuffled_signals_random_times preserves count/side/barriers and only
    changes decision_idx, bounded by per-symbol bar length.
  * _shuffled_signals_sign_flip flips the right fraction in expectation.
  * The end-to-end time_shuffle_null / sign_flip_null run on a synthetic
    bundle without crashing and return a sensibly-shaped NullResult.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from liqpool.arsenal.base import Alpha, AlphaSignal
from liqpool.arsenal.evaluator import EvaluatorConfig
from liqpool.arsenal.null_tests import (
    NullResult,
    _shuffled_signals_random_times,
    _shuffled_signals_sign_flip,
    sign_flip_null,
    time_shuffle_null,
)


def _bars_ist(n, start_ist="10:00"):
    h, m = start_ist.split(":")
    ist_min = int(h) * 60 + int(m)
    utc_min = ist_min - (5 * 60 + 30)
    if utc_min < 0:
        utc_min += 24 * 60
    utc_h, utc_m = divmod(utc_min, 60)
    start_utc = f"2026-05-26 {utc_h:02d}:{utc_m:02d}"
    idx = pd.date_range(start_utc, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
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


class _FixedAlpha(Alpha):
    """Emits a known set of signals every time it's queried."""
    def __init__(self, sigs_by_symbol):
        self._sigs = sigs_by_symbol
    @property
    def name(self):
        return "fixed_alpha"
    def candidates(self, *, symbol, df_base, atr_series, extra=None):
        out: List[AlphaSignal] = []
        for s in self._sigs.get(symbol, []):
            idx = s["idx"]
            if idx >= len(df_base):
                continue
            out.append(AlphaSignal(
                alpha_name=self.name, symbol=symbol,
                decision_at=df_base.index[idx], decision_idx=idx,
                side=s["side"], entry_reference=float(df_base["close"].iloc[idx]),
                stop_atr=0.5, target_atr=2.0, horizon_bars=12,
            ))
        return out


# ---------------------------------------------------------------------------
# Shuffle helpers
# ---------------------------------------------------------------------------

class ShuffleHelperTests(unittest.TestCase):

    def _sigs(self, n=10, symbol="X"):
        return [
            AlphaSignal(
                alpha_name="x", symbol=symbol,
                decision_at=pd.Timestamp("2026-05-26 04:30"),
                decision_idx=i, side="long" if i % 2 == 0 else "short",
                entry_reference=100.0, stop_atr=0.5, target_atr=2.0,
                horizon_bars=12,
            )
            for i in range(n)
        ]

    def test_random_time_shuffle_preserves_count_and_barriers(self):
        sigs = self._sigs(10)
        rng = np.random.default_rng(0)
        sh = _shuffled_signals_random_times(sigs, {"X": 100}, rng)
        self.assertEqual(len(sh), len(sigs))
        for orig, new in zip(sigs, sh):
            self.assertEqual(orig.side, new.side)
            self.assertEqual(orig.stop_atr, new.stop_atr)
            self.assertEqual(orig.target_atr, new.target_atr)
            self.assertEqual(orig.horizon_bars, new.horizon_bars)
            self.assertEqual(orig.symbol, new.symbol)

    def test_random_time_shuffle_indices_in_range(self):
        sigs = self._sigs(20)
        rng = np.random.default_rng(0)
        sh = _shuffled_signals_random_times(sigs, {"X": 50}, rng)
        for s in sh:
            self.assertGreaterEqual(s.decision_idx, 0)
            self.assertLess(s.decision_idx, 49)         # high=n_bars-1

    def test_random_time_shuffle_skips_unknown_symbol(self):
        sigs = self._sigs(5, symbol="UNKNOWN")
        rng = np.random.default_rng(0)
        sh = _shuffled_signals_random_times(sigs, {"X": 100}, rng)
        self.assertEqual(sh, [])

    def test_sign_flip_zero_fraction_keeps_all(self):
        sigs = self._sigs(20)
        rng = np.random.default_rng(0)
        sh = _shuffled_signals_sign_flip(sigs, flip_fraction=0.0, rng=rng)
        self.assertEqual([s.side for s in sh], [s.side for s in sigs])

    def test_sign_flip_full_fraction_inverts_all(self):
        sigs = self._sigs(20)
        rng = np.random.default_rng(0)
        sh = _shuffled_signals_sign_flip(sigs, flip_fraction=1.0, rng=rng)
        for orig, new in zip(sigs, sh):
            expected = "short" if orig.side == "long" else "long"
            self.assertEqual(new.side, expected)

    def test_sign_flip_partial_is_approximately_fraction(self):
        sigs = self._sigs(2000)
        rng = np.random.default_rng(0)
        sh = _shuffled_signals_sign_flip(sigs, flip_fraction=0.5, rng=rng)
        flipped = sum(1 for o, n in zip(sigs, sh) if o.side != n.side)
        # ~1000 +/- noise; 800-1200 is comfortable.
        self.assertGreater(flipped, 800)
        self.assertLess(flipped, 1200)


# ---------------------------------------------------------------------------
# End-to-end null tests on a synthetic bundle
# ---------------------------------------------------------------------------

class NullTestEndToEndTests(unittest.TestCase):
    """Smoke tests — null harness runs without crashing and returns the
    documented schema."""

    def _bundle(self):
        df_a = _bars_ist(100, "10:00")
        df_b = _bars_ist(100, "10:00")
        return _Bundle(assets={
            "HDFCBANK": _Asset(base_df=df_a),
            "TCS":      _Asset(base_df=df_b),
        })

    def test_time_shuffle_null_runs_and_returns_schema(self):
        alpha = _FixedAlpha({
            "HDFCBANK": [{"idx": 25, "side": "long"},
                          {"idx": 40, "side": "short"}],
            "TCS":      [{"idx": 30, "side": "long"}],
        })
        cfg = EvaluatorConfig(notional_inr=50_000)
        nr = time_shuffle_null(alpha, self._bundle(), cfg, n_trials=10, seed=1)
        self.assertIsInstance(nr, NullResult)
        self.assertEqual(nr.alpha_name, "fixed_alpha")
        self.assertEqual(nr.test_name, "time_shuffle")
        self.assertGreaterEqual(nr.n_trials, 0)
        self.assertEqual(nr.actual_n_trades, 3)

    def test_sign_flip_null_runs_and_returns_schema(self):
        alpha = _FixedAlpha({
            "HDFCBANK": [{"idx": 25, "side": "long"},
                          {"idx": 40, "side": "short"}],
            "TCS":      [{"idx": 30, "side": "long"}],
        })
        cfg = EvaluatorConfig(notional_inr=50_000)
        nr = sign_flip_null(alpha, self._bundle(), cfg, n_trials=10,
                             flip_fraction=0.5, seed=1)
        self.assertEqual(nr.test_name, "sign_flip")
        self.assertGreaterEqual(nr.n_trials, 0)

    def test_null_test_with_no_signals_returns_safe_default(self):
        alpha = _FixedAlpha({})           # produces zero signals
        cfg = EvaluatorConfig()
        nr = time_shuffle_null(alpha, self._bundle(), cfg, n_trials=5, seed=1)
        self.assertEqual(nr.actual_n_trades, 0)
        # p-value should be NaN when there were no actual signals.
        import math
        self.assertTrue(math.isnan(nr.null_p_value))


if __name__ == "__main__":
    unittest.main()
