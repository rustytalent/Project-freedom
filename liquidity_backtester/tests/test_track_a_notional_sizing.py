import numpy as np
import pandas as pd

from analysis.run_phase4_track_a_pretouch_sweep import (
    _quantity_for_entry,
    _sizing_matches,
    _target_notional,
)
from liqpool.execution_simulator_v2 import RupeeTargetExecutionConfig


def test_target_notional_zero_disables_notional_sizing():
    assert _target_notional(100_000.0) == 100_000.0
    assert _target_notional(0.0) is None
    assert _target_notional(None) is None


def test_quantity_for_entry_uses_floor_lot_from_notional():
    assert _quantity_for_entry(1_250.0, quantity=1, notional_inr=100_000.0) == 80
    assert _quantity_for_entry(1_250.0, quantity=7, notional_inr=None) == 7
    assert _quantity_for_entry(np.nan, quantity=7, notional_inr=None) == 7


def test_sizing_matches_rejects_stale_qty_one_cache_under_notional():
    stale = pd.DataFrame({"quantity": [1, 1], "net_r": [0.1, -0.2]})
    fresh = pd.DataFrame({
        "quantity": [80, 81],
        "target_notional_inr": [100_000.0, 100_000.0],
    })
    assert not _sizing_matches(stale, quantity=1, notional_inr=100_000.0)
    assert _sizing_matches(fresh, quantity=1, notional_inr=100_000.0)


def test_sizing_matches_fixed_quantity_rejects_notional_cache():
    notional_cache = pd.DataFrame({
        "quantity": [80, 80],
        "target_notional_inr": [100_000.0, 100_000.0],
    })
    fixed_qty = pd.DataFrame({"quantity": [7, 7]})
    assert not _sizing_matches(notional_cache, quantity=7, notional_inr=None)
    assert _sizing_matches(fixed_qty, quantity=7, notional_inr=None)


def test_sizing_matches_rupee_target_config_and_rejects_notional_cache():
    cfg = RupeeTargetExecutionConfig(
        required_reward_inr=600.0,
        min_per_share_move=6.0,
        max_notional=200_000.0,
        min_notional=30_000.0,
        stop_ratio=0.5,
    )
    notional_cache = pd.DataFrame({
        "sizing_mode": ["notional"],
        "quantity": [80],
        "target_notional_inr": [100_000.0],
    })
    rupee_cache = pd.DataFrame({
        "sizing_mode": ["rupee_target", "rupee_target"],
        "rupee_required_reward_inr": [600.0, 600.0],
        "rupee_min_per_share_move": [6.0, 6.0],
        "rupee_max_notional": [200_000.0, 200_000.0],
        "rupee_min_notional": [30_000.0, 30_000.0],
        "rupee_stop_ratio": [0.5, 0.5],
        "rupee_target_allow_beyond_pool": [False, False],
    })

    assert not _sizing_matches(notional_cache, quantity=1, notional_inr=None, rupee_target=cfg)
    assert _sizing_matches(rupee_cache, quantity=1, notional_inr=None, rupee_target=cfg)
    assert not _sizing_matches(
        rupee_cache,
        quantity=1,
        notional_inr=None,
        rupee_target=cfg,
        rupee_target_allow_beyond_pool=True,
    )
