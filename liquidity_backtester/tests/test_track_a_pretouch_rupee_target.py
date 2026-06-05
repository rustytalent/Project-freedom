import pandas as pd

from analysis.run_phase4_track_a_pretouch_sweep import Geometry, _simulate_one
from liqpool.execution_simulator_v2 import RupeeTargetExecutionConfig


def _base_df() -> pd.DataFrame:
    idx = pd.date_range("2026-05-26 09:15", periods=6, freq="5min")
    return pd.DataFrame({
        "open": [999.0, 1000.0, 1001.0, 1006.2, 1005.0, 1004.0],
        "high": [1000.0, 1002.0, 1006.5, 1007.0, 1005.5, 1004.5],
        "low": [998.0, 999.0, 1000.5, 1005.0, 1003.0, 1002.0],
        "close": [999.5, 1001.0, 1006.0, 1005.5, 1004.0, 1003.0],
        "volume": [1000.0] * 6,
    }, index=idx)


def _candidate(price_low: float = 1010.0) -> pd.Series:
    return pd.Series({
        "symbol": "TEST",
        "sector": "AUTO",
        "fold": 0,
        "pool_idx": 7,
        "bar_idx": 0,
        "ts": "2026-05-26 09:15:00",
        "pool_side": "above",
        "price_low": price_low,
        "price_high": price_low + 1.0,
        "distance_atr": 6.0,
        "p_touch": 0.80,
        "p_up": 0.75,
        "p_direction_to_pool": 0.75,
        "pool_quality": 0.45,
        "touch_label": 1,
        "direction_target_to_pool": 1,
        "headline_factor": "EQHL",
        "tf_bucket": "4+",
    })


def test_pretouch_rupee_target_sets_exact_geometry_and_quantity():
    cfg = RupeeTargetExecutionConfig(
        required_reward_inr=600.0,
        min_per_share_move=6.0,
        max_notional=200_000.0,
        min_notional=30_000.0,
        stop_ratio=0.5,
    )

    trade = _simulate_one(
        _candidate(),
        base_df=_base_df(),
        atr_values=pd.Series([2.0] * 6, index=_base_df().index),
        intrabar_1m=None,
        geometry=Geometry(target_fraction=1.0, stop_atr_mult=1.0, max_hold_bars=4),
        data_timestamps_utc=False,
        session_exit_time="15:10",
        use_1m_resolution=False,
        slippage_model="flat",
        base_slippage_bps=0.0,
        exchange="NSE",
        quantity=1,
        notional_inr=None,
        rupee_target=cfg,
        rupee_target_allow_beyond_pool=False,
    )

    assert trade is not None
    assert trade["sizing_mode"] == "rupee_target"
    assert trade["direction"] == "UP"
    assert trade["quantity"] == 100
    assert trade["target"] == 1006.0
    assert trade["stop"] == 997.0
    assert trade["target_move_inr"] == 6.0
    assert trade["target_reward_inr"] == 600.0
    assert trade["planned_entry_notional_inr"] == 100_000.0
    assert trade["exit_reason"] == "target"


def test_pretouch_rupee_target_rejects_target_beyond_pool_by_default():
    cfg = RupeeTargetExecutionConfig(
        required_reward_inr=600.0,
        min_per_share_move=6.0,
        max_notional=200_000.0,
        min_notional=30_000.0,
        stop_ratio=0.5,
    )

    blocked = _simulate_one(
        _candidate(price_low=1004.0),
        base_df=_base_df(),
        atr_values=pd.Series([2.0] * 6, index=_base_df().index),
        intrabar_1m=None,
        geometry=Geometry(target_fraction=1.0, stop_atr_mult=1.0, max_hold_bars=4),
        data_timestamps_utc=False,
        session_exit_time="15:10",
        use_1m_resolution=False,
        slippage_model="flat",
        base_slippage_bps=0.0,
        exchange="NSE",
        quantity=1,
        notional_inr=None,
        rupee_target=cfg,
        rupee_target_allow_beyond_pool=False,
    )

    allowed = _simulate_one(
        _candidate(price_low=1004.0),
        base_df=_base_df(),
        atr_values=pd.Series([2.0] * 6, index=_base_df().index),
        intrabar_1m=None,
        geometry=Geometry(target_fraction=1.0, stop_atr_mult=1.0, max_hold_bars=4),
        data_timestamps_utc=False,
        session_exit_time="15:10",
        use_1m_resolution=False,
        slippage_model="flat",
        base_slippage_bps=0.0,
        exchange="NSE",
        quantity=1,
        notional_inr=None,
        rupee_target=cfg,
        rupee_target_allow_beyond_pool=True,
    )

    assert blocked is None
    assert allowed is not None
    assert allowed["target"] == 1006.0
