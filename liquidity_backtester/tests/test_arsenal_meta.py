"""Tests for the meta-alpha combinators (AND / OR / WEIGHTED).

We use fixture alphas that emit a known set of (symbol, idx, side) tuples,
then verify each combinator selects the right rows. No real data needed.
"""
from __future__ import annotations

import unittest
from typing import Dict, List

import pandas as pd

from liqpool.arsenal.base import Alpha, AlphaSignal
from liqpool.arsenal.meta import (
    MetaANDAlpha,
    MetaORAlpha,
    MetaWeightedAlpha,
    _merge_geometry_max_horizon_min_risk,
    _merge_geometry_min,
)


class _ScriptedAlpha(Alpha):
    """Emits a fixed list of signals regardless of inputs."""

    def __init__(self, name: str, rows: List[Dict]):
        self._n = name
        self._rows = rows

    @property
    def name(self) -> str:
        return self._n

    def candidates(self, *, symbol, df_base, atr_series, extra=None):
        out: List[AlphaSignal] = []
        for r in self._rows:
            if r["symbol"] != symbol:
                continue
            out.append(AlphaSignal(
                alpha_name=self._n, symbol=r["symbol"],
                decision_at=pd.Timestamp("2026-05-26 04:30"),
                decision_idx=int(r["idx"]),
                side=r["side"],
                entry_reference=float(r.get("entry", 100.0)),
                stop_atr=float(r.get("stop", 0.5)),
                target_atr=float(r.get("target", 2.0)),
                horizon_bars=int(r.get("horizon", 24)),
                confidence=float(r.get("confidence", 0.5)),
            ))
        return out


def _df():
    idx = pd.date_range("2026-05-26 04:30", periods=200, freq="5min")
    return pd.DataFrame({
        "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
        "volume": 1000,
    }, index=idx)


# ---------------------------------------------------------------------------
# Geometry merge helpers
# ---------------------------------------------------------------------------

class GeometryMergeTests(unittest.TestCase):

    def _sigs(self, geos):
        return [
            AlphaSignal(
                alpha_name=f"a{i}", symbol="X",
                decision_at=pd.Timestamp("2026-05-26 04:30"),
                decision_idx=10, side="long",
                entry_reference=100.0,
                stop_atr=g[0], target_atr=g[1], horizon_bars=g[2],
            )
            for i, g in enumerate(geos)
        ]

    def test_min_merge_picks_tightest_smallest_shortest(self):
        g = _merge_geometry_min(self._sigs([(0.5, 2.0, 24), (0.3, 1.5, 12)]))
        self.assertEqual(g, {"stop_atr": 0.3, "target_atr": 1.5, "horizon_bars": 12})

    def test_max_horizon_min_risk_picks_min_stop_max_target_min_horizon(self):
        g = _merge_geometry_max_horizon_min_risk(
            self._sigs([(0.5, 2.0, 24), (0.3, 3.0, 12)])
        )
        self.assertEqual(g, {"stop_atr": 0.3, "target_atr": 3.0, "horizon_bars": 12})


# ---------------------------------------------------------------------------
# MetaANDAlpha
# ---------------------------------------------------------------------------

class MetaANDAlphaTests(unittest.TestCase):

    def test_requires_at_least_two_components(self):
        with self.assertRaises(ValueError):
            MetaANDAlpha([_ScriptedAlpha("a", [])])

    def test_emits_only_at_shared_decision_idx(self):
        # A fires at idx {10, 20, 30}; B fires at idx {20, 30, 40}.
        # Intersection: {20, 30}.
        a = _ScriptedAlpha("a", [
            {"symbol": "X", "idx": 10, "side": "long"},
            {"symbol": "X", "idx": 20, "side": "long"},
            {"symbol": "X", "idx": 30, "side": "long"},
        ])
        b = _ScriptedAlpha("b", [
            {"symbol": "X", "idx": 20, "side": "long"},
            {"symbol": "X", "idx": 30, "side": "long"},
            {"symbol": "X", "idx": 40, "side": "long"},
        ])
        meta = MetaANDAlpha([a, b])
        sigs = meta.candidates(symbol="X", df_base=_df(), atr_series=None)
        self.assertEqual([s.decision_idx for s in sigs], [20, 30])
        for s in sigs:
            self.assertEqual(s.alpha_name, meta.name)

    def test_require_same_side_drops_disagreements(self):
        a = _ScriptedAlpha("a", [{"symbol": "X", "idx": 20, "side": "long"}])
        b = _ScriptedAlpha("b", [{"symbol": "X", "idx": 20, "side": "short"}])
        meta = MetaANDAlpha([a, b], require_same_side=True)
        sigs = meta.candidates(symbol="X", df_base=_df(), atr_series=None)
        self.assertEqual(sigs, [])

    def test_allow_disagreement_when_require_same_side_false(self):
        a = _ScriptedAlpha("a", [{"symbol": "X", "idx": 20, "side": "long"}])
        b = _ScriptedAlpha("b", [{"symbol": "X", "idx": 20, "side": "short"}])
        meta = MetaANDAlpha([a, b], require_same_side=False)
        sigs = meta.candidates(symbol="X", df_base=_df(), atr_series=None)
        self.assertEqual(len(sigs), 1)
        # Anchor side (first component).
        self.assertEqual(sigs[0].side, "long")

    def test_merged_geometry_is_conservative_min(self):
        a = _ScriptedAlpha("a", [{"symbol": "X", "idx": 20, "side": "long",
                                    "stop": 0.5, "target": 2.0, "horizon": 40}])
        b = _ScriptedAlpha("b", [{"symbol": "X", "idx": 20, "side": "long",
                                    "stop": 0.75, "target": 1.5, "horizon": 12}])
        meta = MetaANDAlpha([a, b])
        sigs = meta.candidates(symbol="X", df_base=_df(), atr_series=None)
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].stop_atr, 0.5)            # min
        self.assertEqual(sigs[0].target_atr, 1.5)          # min
        self.assertEqual(sigs[0].horizon_bars, 12)         # min

    def test_confidence_is_component_average(self):
        a = _ScriptedAlpha("a", [{"symbol": "X", "idx": 20, "side": "long",
                                    "confidence": 0.4}])
        b = _ScriptedAlpha("b", [{"symbol": "X", "idx": 20, "side": "long",
                                    "confidence": 0.8}])
        meta = MetaANDAlpha([a, b])
        sigs = meta.candidates(symbol="X", df_base=_df(), atr_series=None)
        self.assertAlmostEqual(sigs[0].confidence, 0.6, places=6)

    def test_no_overlap_returns_empty(self):
        a = _ScriptedAlpha("a", [{"symbol": "X", "idx": 10, "side": "long"}])
        b = _ScriptedAlpha("b", [{"symbol": "X", "idx": 20, "side": "long"}])
        meta = MetaANDAlpha([a, b])
        self.assertEqual(meta.candidates(symbol="X", df_base=_df(),
                                           atr_series=None), [])


# ---------------------------------------------------------------------------
# MetaORAlpha
# ---------------------------------------------------------------------------

class MetaORAlphaTests(unittest.TestCase):

    def test_emits_union_dedup_to_highest_confidence(self):
        a = _ScriptedAlpha("a", [
            {"symbol": "X", "idx": 10, "side": "long", "confidence": 0.3},
            {"symbol": "X", "idx": 20, "side": "long", "confidence": 0.6},
        ])
        b = _ScriptedAlpha("b", [
            {"symbol": "X", "idx": 20, "side": "long", "confidence": 0.8},
            {"symbol": "X", "idx": 30, "side": "short", "confidence": 0.5},
        ])
        meta = MetaORAlpha([a, b])
        sigs = meta.candidates(symbol="X", df_base=_df(), atr_series=None)
        # Union: {10, 20, 30}.
        self.assertEqual([s.decision_idx for s in sigs], [10, 20, 30])
        # idx=20 dedup'd to b (confidence 0.8 > 0.6).
        idx20 = next(s for s in sigs if s.decision_idx == 20)
        self.assertEqual(idx20.state["source_alpha"], "b")
        self.assertAlmostEqual(idx20.confidence, 0.8, places=6)
        # All re-tagged with meta alpha name.
        for s in sigs:
            self.assertEqual(s.alpha_name, meta.name)


# ---------------------------------------------------------------------------
# MetaWeightedAlpha
# ---------------------------------------------------------------------------

class MetaWeightedAlphaTests(unittest.TestCase):

    def test_threshold_filters_by_weighted_confidence(self):
        # A confidence 0.6 weight 1; B confidence 0.6 weight 1 -> total 1.2
        # threshold=1.0 -> emit. threshold=1.5 -> reject.
        a = _ScriptedAlpha("a", [{"symbol": "X", "idx": 20, "side": "long",
                                    "confidence": 0.6}])
        b = _ScriptedAlpha("b", [{"symbol": "X", "idx": 20, "side": "long",
                                    "confidence": 0.6}])
        # Threshold 1.0
        meta = MetaWeightedAlpha([a, b], weights=[1.0, 1.0], threshold=1.0)
        sigs = meta.candidates(symbol="X", df_base=_df(), atr_series=None)
        self.assertEqual(len(sigs), 1)
        # Threshold 1.5
        meta2 = MetaWeightedAlpha([a, b], weights=[1.0, 1.0], threshold=1.5)
        self.assertEqual(meta2.candidates(symbol="X", df_base=_df(),
                                            atr_series=None), [])

    def test_components_can_fire_individually(self):
        # B doesn't fire at idx=10; A does with confidence 0.7, weight 1.
        # If threshold <= 0.7, the alpha emits even though B didn't fire.
        a = _ScriptedAlpha("a", [{"symbol": "X", "idx": 10, "side": "long",
                                    "confidence": 0.7}])
        b = _ScriptedAlpha("b", [])
        meta = MetaWeightedAlpha([a, b], weights=[1.0, 1.0], threshold=0.5)
        sigs = meta.candidates(symbol="X", df_base=_df(), atr_series=None)
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].state["n_firing"], 1)

    def test_invalid_weights_raises(self):
        a = _ScriptedAlpha("a", [])
        b = _ScriptedAlpha("b", [])
        with self.assertRaises(ValueError):
            MetaWeightedAlpha([a, b], weights=[1.0])

    def test_default_threshold_is_half_weights_sum(self):
        a = _ScriptedAlpha("a", [])
        b = _ScriptedAlpha("b", [])
        c = _ScriptedAlpha("c", [])
        meta = MetaWeightedAlpha([a, b, c], weights=[1.0, 1.0, 1.0])
        self.assertEqual(meta.threshold, 1.5)


if __name__ == "__main__":
    unittest.main()
