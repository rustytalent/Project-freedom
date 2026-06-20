"""Tests for executor_v4.economics — fees-FIRST EV gate."""
from __future__ import annotations

import pytest

from liqpool.research.belief.executor_v4.economics import (
    ExecutionEconomicsConfig,
    NIFTY_LOT_SIZE,
    NIFTY_MAX_LOSS_PER_TRADE,
    daily_bleed_blocker,
    estimate_slippage,
    expected_value_after_costs,
    fees_for_round_trip,
    minimum_profitable_premium_delta,
    should_accelerate_exit,
)


def test_founder_calibrated_defaults():
    """Founder confirmed (2026-06-19): lot=65, max_loss=₹1000, daily=₹5k floor."""
    assert NIFTY_LOT_SIZE == 65
    assert NIFTY_MAX_LOSS_PER_TRADE == 1000.0
    cfg = ExecutionEconomicsConfig()
    assert cfg.max_daily_bleed == 5000.0
    assert cfg.brokerage_flat_per_leg == 20.0
    assert cfg.gst_pct == 0.18
    assert cfg.min_edge_multiple >= 1.0


def test_fees_breakdown_components_sum_to_total():
    fees = fees_for_round_trip(
        entry_premium=100.0, exit_premium=120.0,
        lots=1, is_long=True,
    )
    components = (fees.brokerage + fees.stt + fees.exchange + fees.sebi
                   + fees.stamp_duty + fees.gst)
    assert components == pytest.approx(fees.total)


def test_fees_long_round_trip_nifty_one_lot_in_documented_range():
    """For a long round-trip on an ATM-ish NIFTY weekly (₹100 → ₹110)
    with lot=65, Zerodha fees come to ~₹50 (brokerage ₹40 + ~₹13 turnover
    + GST). The founder's quoted ~₹100/lot includes slippage on top of
    this — the executor's slippage layer adds the rest separately."""
    fees = fees_for_round_trip(
        entry_premium=100.0, exit_premium=110.0, lots=1, is_long=True,
    )
    assert 40.0 < fees.total < 100.0, (
        f"Fees ₹{fees.total:.2f} outside ₹40-100 calculated range for 1-lot NIFTY"
    )
    # Brokerage dominates a 1-lot trade.
    assert fees.brokerage == 40.0
    # GST should be ~18% of (brokerage + exchange + sebi), so ~₹7-8.
    assert 5.0 < fees.gst < 10.0


def test_fees_scale_with_lots_but_brokerage_flat():
    """Brokerage is flat per ORDER (₹20 entry + ₹20 exit), not per LOT.
    Only the turnover-pct fees scale with lots."""
    f1 = fees_for_round_trip(entry_premium=100.0, exit_premium=120.0,
                              lots=1, is_long=True)
    f4 = fees_for_round_trip(entry_premium=100.0, exit_premium=120.0,
                              lots=4, is_long=True)
    assert f4.total > f1.total                # more lots = more total fees
    assert f4.total < f1.total * 4            # but less than 4× (brokerage flat)
    # Sanity: 4-lot total ought to be ~30%+ higher than 1-lot.
    assert f4.total > f1.total * 1.25


def test_stt_only_applies_to_sell_side():
    """For a long, STT is on the exit premium. For a short, on the entry."""
    long_fees = fees_for_round_trip(
        entry_premium=100.0, exit_premium=120.0, lots=1, is_long=True,
    )
    short_fees = fees_for_round_trip(
        entry_premium=120.0, exit_premium=100.0, lots=1, is_long=False,
    )
    # Long sells at 120, short sells at 120 too — STT should match.
    assert long_fees.stt == pytest.approx(short_fees.stt)


def test_slippage_increases_with_dangerous_spread():
    clean = estimate_slippage(lots=1, friendliness=1.0, spread_state="clean")
    dangerous = estimate_slippage(lots=1, friendliness=0.3, spread_state="dangerous")
    assert dangerous > clean * 3


def test_minimum_profitable_premium_delta_is_positive_and_finite():
    delta = minimum_profitable_premium_delta(
        entry_premium=100.0, lots=1, is_long=True,
    )
    assert delta > 0
    assert delta < 5.0   # for ATM-ish premium, fees+slip floor is ~₹0.50-3 per share


def test_ev_approves_positive_expectancy_trade():
    """A 60/40 trade with reasonable target/stop and good edge should pass."""
    ev = expected_value_after_costs(
        entry_premium=100.0,
        expected_exit_premium=140.0,    # +40% premium move on win
        prob_target=0.55,
        prob_stop=0.45,
        stop_premium=70.0,              # -30% on loss
        lots=2, is_long=True,
        friendliness=0.9, spread_state="clean",
    )
    assert ev.approve, f"Should approve; got rejected with: {ev.reasons}"
    assert ev.expected_profit_rupees > 0
    assert ev.edge_multiple >= 1.5


def test_ev_refuses_when_fees_eat_expected_profit():
    """Tight target + low probability = fees eat any profit."""
    ev = expected_value_after_costs(
        entry_premium=10.0,             # cheap deep-OTM option
        expected_exit_premium=10.4,     # 4% gain
        prob_target=0.45,
        prob_stop=0.55,
        stop_premium=9.5,
        lots=1, is_long=True,
    )
    assert not ev.approve
    assert any("edge" in r.lower() or "profit" in r.lower() for r in ev.reasons)


def test_ev_refuses_negative_expectancy():
    ev = expected_value_after_costs(
        entry_premium=100.0,
        expected_exit_premium=110.0,
        prob_target=0.30,
        prob_stop=0.70,
        stop_premium=80.0,
        lots=1, is_long=True,
    )
    assert not ev.approve


def test_exit_acceleration_fires_on_meaningful_drop_with_thesis_decay():
    """Founder's exact rule: when loser is forming, cut even if fees ratio
    looks bad. Premium down 25%, thesis confidence decayed 0.25 → cut now."""
    accel = should_accelerate_exit(
        entry_premium=100.0,
        current_premium=75.0,    # 25% drop
        stop_premium=50.0,       # would lose 50% if held to stop
        bars_held=5,
        lots=2, is_long=True,
        thesis_confidence_decay=0.25,
    )
    assert accel.accelerate, f"Should accelerate; got: {accel.reason}"


def test_exit_acceleration_holds_when_loss_shallow_or_thesis_intact():
    """Either the drop is shallow (under 20% threshold) OR thesis is still
    intact — both are valid hold reasons. The point is the acceleration
    doesn't fire prematurely."""
    accel = should_accelerate_exit(
        entry_premium=100.0,
        current_premium=82.0,        # 18% drop — below the 20% trigger
        stop_premium=50.0,
        bars_held=5,
        lots=1, is_long=True,
        thesis_confidence_decay=0.05,
    )
    assert not accel.accelerate

    # Even with a deeper drop, an intact thesis prevents acceleration.
    accel_intact = should_accelerate_exit(
        entry_premium=100.0, current_premium=75.0,
        stop_premium=50.0, bars_held=5, lots=1, is_long=True,
        thesis_confidence_decay=0.05,
    )
    assert not accel_intact.accelerate
    assert "thesis still mostly intact" in accel_intact.reason


def test_exit_acceleration_holds_during_min_age():
    accel = should_accelerate_exit(
        entry_premium=100.0,
        current_premium=70.0,
        stop_premium=50.0,
        bars_held=0,            # before min hold
        lots=1, is_long=True,
        thesis_confidence_decay=0.30,
    )
    assert not accel.accelerate
    assert "too early" in accel.reason


def test_daily_bleed_blocker_triggers_at_floor():
    cfg = ExecutionEconomicsConfig(max_daily_bleed=5000.0)
    assert daily_bleed_blocker(-4500.0, cfg) is None
    blocker = daily_bleed_blocker(-5100.0, cfg)
    assert blocker is not None
    assert "daily bleed" in blocker


def test_fees_breakdown_to_dict_serializable():
    fees = fees_for_round_trip(entry_premium=100.0, exit_premium=120.0,
                                 lots=2, is_long=True)
    import json
    json.dumps(fees.to_dict())


def test_ev_decision_to_dict_serializable():
    ev = expected_value_after_costs(
        entry_premium=100.0, expected_exit_premium=140.0,
        prob_target=0.5, prob_stop=0.5,
        stop_premium=70.0, lots=2, is_long=True,
    )
    import json
    json.dumps(ev.to_dict())


def test_config_validation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ExecutionEconomicsConfig(lot_size=0)
    with pytest.raises(ValueError):
        ExecutionEconomicsConfig(min_edge_multiple=0.5)
