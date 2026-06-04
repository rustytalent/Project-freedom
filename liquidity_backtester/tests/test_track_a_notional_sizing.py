import numpy as np
import pandas as pd

from analysis.run_phase4_track_a_pretouch_sweep import (
    _quantity_for_entry,
    _sizing_matches,
    _target_notional,
)


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
