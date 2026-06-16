"""Tests for the Belief Engine moneyness identity layer."""
from __future__ import annotations

import pytest

from liqpool.research.belief.moneyness import (
    MoneynessSlot,
    build_chain_slots,
    classify_moneyness,
    detect_identity_anomaly,
    expected_abs_delta,
    moneyness_behavior,
)


SPOT = 23000.0
STEP = 50.0


def test_call_above_spot_is_otm_below_is_itm():
    atm = classify_moneyness(SPOT, 23000, "CE", STEP)
    assert atm.level == 0 and atm.label == "CE_ATM"
    otm1 = classify_moneyness(SPOT, 23050, "CE", STEP)
    assert otm1.level == 1 and otm1.label == "CE_OTM1"
    itm2 = classify_moneyness(SPOT, 22900, "CE", STEP)
    assert itm2.level == -2 and itm2.label == "CE_ITM2"


def test_put_moneyness_is_inverted():
    # For puts, strikes ABOVE spot are ITM, BELOW are OTM.
    itm1 = classify_moneyness(SPOT, 23050, "PE", STEP)
    assert itm1.level == -1 and itm1.label == "PE_ITM1"
    otm2 = classify_moneyness(SPOT, 22900, "PE", STEP)
    assert otm2.level == 2 and otm2.label == "PE_OTM2"


def test_signed_delta_direction():
    ce = classify_moneyness(SPOT, 22900, "CE", STEP)   # ITM2 call
    pe = classify_moneyness(SPOT, 23100, "PE", STEP)   # ITM2 put
    assert ce.expected_signed_delta > 0
    assert pe.expected_signed_delta < 0
    assert ce.expected_abs_delta == pytest.approx(pe.expected_abs_delta)


def test_deep_itm_is_future_like():
    deep = classify_moneyness(SPOT, 22750, "CE", STEP)   # ITM5
    assert deep.is_future_like
    assert deep.behavior == "future_like"


def test_atm_is_gamma_regime():
    atm = classify_moneyness(SPOT, 23000, "CE", STEP)
    assert atm.behavior == "gamma_atm"
    assert not atm.is_future_like


def test_deep_otm_is_lottery():
    far = classify_moneyness(SPOT, 23250, "CE", STEP)   # OTM5
    assert far.behavior == "lottery_otm"


def test_expected_abs_delta_monotonic_and_extrapolates():
    # Deeper ITM → higher delta; deeper OTM → lower delta; monotone.
    levels = list(range(-8, 9))
    deltas = [expected_abs_delta(l) for l in levels]
    # Sorted by level ascending (ITM→OTM) the delta must be non-increasing.
    assert all(a >= b - 1e-9 for a, b in zip(deltas, deltas[1:]))
    # Extrapolation stays in (0,1).
    assert 0.0 < expected_abs_delta(-8) <= 0.99
    assert 0.0 < expected_abs_delta(8) <= expected_abs_delta(5)


def test_moneyness_behavior_bands():
    assert moneyness_behavior(0.95) == "future_like"
    assert moneyness_behavior(0.70) == "directional_itm"
    assert moneyness_behavior(0.50) == "gamma_atm"
    assert moneyness_behavior(0.25) == "convex_otm"
    assert moneyness_behavior(0.05) == "lottery_otm"


def test_build_chain_slots_22_contracts_sorted():
    strikes = [SPOT + STEP * i for i in range(-5, 6)]
    contracts = [(k, "CE") for k in strikes] + [(k, "PE") for k in strikes]
    slots = build_chain_slots(SPOT, contracts, STEP)
    assert len(slots) == 22
    ce_levels = [s.level for s in slots if s.option_type == "CE"]
    pe_levels = [s.level for s in slots if s.option_type == "PE"]
    assert ce_levels == sorted(ce_levels)
    assert pe_levels == sorted(pe_levels)
    # Every level from -5..+5 present on each side.
    assert set(ce_levels) == set(range(-5, 6))
    assert set(pe_levels) == set(range(-5, 6))


def test_identity_anomaly_flags_itm_behaving_like_atm():
    """The founder's case: a 2-ITM call (expected |Δ|≈0.73, directional)
    whose realized effective delta has dropped to ~0.50 (ATM-like) must be
    flagged abnormal — something is being engineered in that strike."""
    slot = classify_moneyness(SPOT, 22900, "CE", STEP)   # ITM2
    anom = detect_identity_anomaly(slot, realized_abs_delta=0.50)
    assert anom["is_anomaly"]
    assert anom["behaving_like"] == "gamma_atm"
    assert anom["expected_behavior"] == "directional_itm"
    assert anom["delta_gap"] < 0   # realized weaker than expected


def test_identity_anomaly_silent_when_normal():
    slot = classify_moneyness(SPOT, 22900, "CE", STEP)   # ITM2, expected 0.73
    anom = detect_identity_anomaly(slot, realized_abs_delta=0.71)
    assert not anom["is_anomaly"]


def test_identity_anomaly_flags_otm_behaving_like_future():
    """An OTM contract suddenly moving 1:1 with spot (|Δ|→0.9) is a strong
    tell — deep money flowing into a cheap strike."""
    slot = classify_moneyness(SPOT, 23100, "CE", STEP)   # OTM2, expected 0.27
    anom = detect_identity_anomaly(slot, realized_abs_delta=0.90)
    assert anom["is_anomaly"]
    assert anom["behaving_like"] == "future_like"
    assert anom["delta_gap"] > 0


def test_classify_validates_inputs():
    with pytest.raises(ValueError, match="option_type"):
        classify_moneyness(SPOT, 23000, "XX", STEP)
    with pytest.raises(ValueError, match="strike_step"):
        classify_moneyness(SPOT, 23000, "CE", 0)


def test_slot_to_dict_is_json_serializable():
    slot = classify_moneyness(SPOT, 23000, "CE", STEP)
    import json
    json.dumps(slot.to_dict())
