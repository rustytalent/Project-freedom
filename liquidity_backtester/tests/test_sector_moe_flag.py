"""Pin the train_sector_experts flag of SectorMoERespectModel.

Per the cleanup analysis (2026-06-03), 4 of 5 sectors typically get a
dynamic MoE weight of 0.0 every retrain — the per-sector experts are
trained then thrown away. The ``train_sector_experts`` flag (default
False at the Config level, default True at the fit() call signature for
back-compat) lets us skip the per-sector loop entirely.

Pinned contracts:
  * Default fit() trains experts as before (back-compat for direct callers).
  * fit(train_sector_experts=False) skips per-sector training; all sector
    weights are 0.0; sector_stats marks each sector "experts_disabled".
  * Config().train_sector_experts is False — the pipeline default.
  * Global model behaviour is identical regardless of the flag.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from liqpool.config import Config
from liqpool.ml_model import SectorMoERespectModel
from liqpool.pools import Pool


def _synth_pool(symbol: str, side: str = "low") -> Pool:
    """Minimal Pool object that flows through sector_of(asset)."""
    return Pool(
        side=side,
        price_low=100.0, price_high=101.0,
        formed_at=pd.Timestamp("2026-04-01"),
        available_at=pd.Timestamp("2026-04-01"),
        contributors=[],
        score=1.0,
        tfs=["base"],
        asset=symbol,
    )


def _synth_training_set(n_per_sector: int = 80):
    """Build a synthetic training set spread across two sectors so the
    expert loop has enough samples per sector to consider training."""
    symbols_by_sector = {
        "BANKING": ["HDFCBANK", "ICICIBANK"],
        "IT":      ["TCS",       "INFY"],
    }
    pools: List[Pool] = []
    rows = []
    rng = np.random.default_rng(7)
    for sector, syms in symbols_by_sector.items():
        for sym in syms:
            for _ in range(n_per_sector // len(syms)):
                pools.append(_synth_pool(sym))
                rows.append({"f1": rng.normal(), "f2": rng.normal(),
                              "f3": rng.normal()})
    X = pd.DataFrame(rows)
    y = (rng.random(len(X)) > 0.5).astype(int)
    return X, y, pools


class ConfigDefaultTests(unittest.TestCase):

    def test_config_default_is_experts_off(self):
        cfg = Config()
        self.assertFalse(cfg.train_sector_experts,
            "pipeline default should be False — experts off")


class FitFlagBehaviourTests(unittest.TestCase):

    def test_experts_off_skips_per_sector_training(self):
        X, y, pools = _synth_training_set(n_per_sector=80)
        model = SectorMoERespectModel()
        model.fit(X, y, pools, train_sector_experts=False)
        # No sector models trained.
        self.assertEqual(len(model.sector_models), 0,
            f"experts_off should produce 0 sector_models, "
            f"got {len(model.sector_models)}")
        # Every sector seen in train_pools must still appear in
        # sector_stats with status experts_disabled (downstream code
        # iterates this dict).
        for sector in ("BANKING", "IT"):
            self.assertIn(sector, model.sector_stats)
            self.assertEqual(
                model.sector_stats[sector]["status"], "experts_disabled")
            self.assertEqual(model.sector_weights.get(sector, 0.0), 0.0)

    def test_experts_on_runs_per_sector_loop(self):
        X, y, pools = _synth_training_set(n_per_sector=80)
        model = SectorMoERespectModel()
        model.fit(X, y, pools, train_sector_experts=True)
        # The per-sector loop ran — each sector has a non-"experts_disabled"
        # status (could be 'trained' or 'skipped' for class imbalance, but
        # NOT 'experts_disabled').
        for sector in ("BANKING", "IT"):
            self.assertIn(sector, model.sector_stats)
            self.assertNotEqual(
                model.sector_stats[sector]["status"], "experts_disabled",
                f"sector {sector} status was experts_disabled in "
                f"train_sector_experts=True path")

    def test_global_model_identical_across_flag(self):
        # The global PoolRespectModel should be identical regardless of
        # the per-sector flag — they fit on the same X/y with the same seed.
        X, y, pools = _synth_training_set(n_per_sector=80)
        m_off = SectorMoERespectModel().fit(
            X, y, pools, train_sector_experts=False, seed=42)
        m_on = SectorMoERespectModel().fit(
            X, y, pools, train_sector_experts=True, seed=42)
        self.assertAlmostEqual(m_off.global_model.val_auc,
                                m_on.global_model.val_auc, places=4,
            msg="global model val_auc must be flag-independent")
        self.assertAlmostEqual(m_off.global_model.val_brier,
                                m_on.global_model.val_brier, places=4)

    def test_back_compat_default_keeps_experts_on(self):
        # Direct callers that don't pass the flag MUST get the legacy
        # behaviour (experts trained). Only the Config-level default
        # is False; the function signature stays back-compat True.
        X, y, pools = _synth_training_set(n_per_sector=80)
        model = SectorMoERespectModel().fit(X, y, pools)
        # Same as experts_on.
        for sector in ("BANKING", "IT"):
            self.assertNotEqual(
                model.sector_stats[sector]["status"], "experts_disabled")


if __name__ == "__main__":
    unittest.main()
