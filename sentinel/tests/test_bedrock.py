"""Bedrock tests — moneyness identity, shadow ledger, orchestration
spine, profit-lock ratchet. The four pieces every later organ depends
on, so they are pinned hard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel.moneyness import (
    MoneynessKey, atm_strike, build_universe, moneyness_key,
    strike_step_for, universe_symbols,
)
from sentinel.orchestration import Orchestrator, Signal, Tier
from sentinel.profit_lock import ProfitLock
from sentinel.shadow_ledger import (
    Context, Hypothesis, Identity, ShadowLedger, compact_to_parquet,
    new_event, read_session, KIND_VIRTUAL, KIND_ACTUAL,
)


# ---------------------------------------------------------------------------
# Moneyness
# ---------------------------------------------------------------------------

def test_atm_strike_snaps_to_grid():
    assert atm_strike(25023, 50) == 25000
    assert atm_strike(25030, 50) == 25050


def test_call_otm_is_positive_offset():
    # spot 25000, step 50. A 25100 CE is 2 strikes OTM -> +2.
    k = moneyness_key("NIFTY", "CE", 25100, 25000, 50)
    assert k.offset == 2 and k.label == "OTM+2_CE"


def test_put_otm_is_positive_offset_too():
    # The sign flip: a 24900 PE is 2 strikes OTM for a PUT -> +2.
    k = moneyness_key("NIFTY", "PE", 24900, 25000, 50)
    assert k.offset == 2 and k.label == "OTM+2_PE"


def test_call_itm_is_negative_offset():
    k = moneyness_key("NIFTY", "CE", 24900, 25000, 50)
    assert k.offset == -2 and k.label.startswith("ITM")


def test_moneyness_key_survives_expiry_rollover():
    """The Thursday solution: a fresh-expiry ATM call gets the SAME key
    as last expiry's ATM call, even though the strike level differs."""
    # last week spot 24000, ATM call strike 24000
    k_old = moneyness_key("NIFTY", "CE", 24000, 24000, 50)
    # this week spot 25000, ATM call strike 25000 — different LEVEL
    k_new = moneyness_key("NIFTY", "CE", 25000, 25000, 50)
    assert k_old == k_new                      # identical behaviour identity
    assert k_old.as_str() == "NIFTY:CE:+0"


def test_strike_step_inferred_from_chain():
    assert strike_step_for("NIFTY", [25000, 25050, 25100, 25150]) == 50
    assert strike_step_for("BANKNIFTY", [52000, 52100, 52200]) == 100
    # No observations -> prior.
    assert strike_step_for("NIFTY", None) == 50


def test_build_universe_selects_atm_plus_minus_5():
    spot = 25000.0
    chain = []
    for k in range(-8, 9):                     # build a wide chain
        strike = 25000 + k * 50
        chain.append((f"NIFTY{int(strike)}CE", "CE", strike))
        chain.append((f"NIFTY{int(strike)}PE", "PE", strike))
    uni = build_universe("NIFTY", spot, chain, half_width=5)
    # 11 strikes (-5..+5) x 2 types = 22 contracts.
    assert len(uni) == 22
    syms = universe_symbols(uni)
    assert len(syms) == 22
    # The +8 strike is outside the window and excluded.
    assert "NIFTY25400CE" not in syms


# ---------------------------------------------------------------------------
# Shadow ledger
# ---------------------------------------------------------------------------

def _identity(sym="NIFTY25000CE", premium=180.0):
    return Identity(instrument=sym, option_type="CE", strike=25000,
                    expiry="2026-06-18", moneyness_key="NIFTY:CE:+0",
                    underlying_price=25000, premium=premium, delta=0.5)


def _hypothesis():
    return Hypothesis(
        expected_underlying_move=80.0, expected_premium=220.0,
        expected_horizon_min=15.0, suggested_entry=180.0,
        suggested_stop=150.0, suggested_target=230.0, confidence=0.62,
        reason_codes=["second_pullback", "gamma_expansion_early"],
        invalidation_conditions=["spot_below_24950"])


def test_event_requires_reason_codes():
    """The laboratory does not log a trade it cannot explain."""
    bad = Hypothesis(0, 0, 0, 0, 0, 0, 0.5, reason_codes=[])
    with pytest.raises(ValueError, match="reason_codes"):
        new_event(KIND_VIRTUAL, "2026-06-13", _identity(), Context(), bad)


def test_event_without_hypothesis_is_allowed():
    # A pure market-observation event (no trade) needs no reasons.
    ev = new_event(KIND_VIRTUAL, "2026-06-13", _identity(), Context(), None)
    assert ev.hypothesis is None


def test_ledger_write_and_read_roundtrip(tmp_path):
    led = ShadowLedger(tmp_path)
    ev = new_event(KIND_VIRTUAL, "2026-06-13", _identity(), Context(trend="up"),
                   _hypothesis(), scientist="second_pullback_scientist")
    led.write(ev)
    rows = read_session(tmp_path, "2026-06-13")
    assert len(rows) == 1
    assert rows[0]["identity"]["moneyness_key"] == "NIFTY:CE:+0"
    assert rows[0]["context"]["trend"] == "up"
    assert rows[0]["hypothesis"]["reason_codes"] == [
        "second_pullback", "gamma_expansion_early"]


def test_journey_stamps_checkpoints_and_completes(tmp_path):
    led = ShadowLedger(tmp_path)
    ev = new_event(KIND_VIRTUAL, "2026-06-13", _identity(premium=180),
                   Context(), _hypothesis())
    # premium rises to 230 (hits target) at 6 min, decays to 200 by 60.
    ShadowLedger.stamp_journey(ev, 1.0, 185)
    ShadowLedger.stamp_journey(ev, 6.0, 232)     # target 230 hit
    ShadowLedger.stamp_journey(ev, 60.0, 200)
    j = ev.journey
    assert j.outcomes["t+1"] == 185
    assert j.outcomes["t+5"] == 232              # 5-min checkpoint filled at 6'
    assert j.time_to_target_min == 6.0
    assert j.mfe == 232 and j.mae == 185
    assert j.complete is True
    assert j.premium_decay == round(200 - 180, 2)


def test_ledger_last_write_wins_on_update(tmp_path):
    led = ShadowLedger(tmp_path)
    ev = new_event(KIND_ACTUAL, "2026-06-13", _identity(), Context(),
                   _hypothesis())
    led.write(ev)
    ShadowLedger.stamp_journey(ev, 60.0, 250)
    ev.judgment.was_entry_good = True
    led.update(ev)
    rows = read_session(tmp_path, "2026-06-13")
    assert len(rows) == 1                         # collapsed, not duplicated
    assert rows[0]["judgment"]["was_entry_good"] is True
    assert rows[0]["journey"]["complete"] is True


def test_compact_to_parquet(tmp_path):
    led = ShadowLedger(tmp_path)
    for i in range(3):
        ev = new_event(KIND_VIRTUAL, "2026-06-13",
                       _identity(sym=f"S{i}"), Context(trend="up"),
                       _hypothesis())
        ShadowLedger.stamp_journey(ev, 60.0, 200 + i)
        led.write(ev)
    out = compact_to_parquet(tmp_path, "2026-06-13")
    assert out is not None and out.exists()
    import pandas as pd
    df = pd.read_parquet(out)
    assert len(df) == 3
    assert "identity.moneyness_key" in df.columns
    assert "journey.t+60" in df.columns


# ---------------------------------------------------------------------------
# Orchestration spine
# ---------------------------------------------------------------------------

def _spine():
    sinks = {"ledger": [], "surface": [], "exec": []}
    o = Orchestrator(
        ledger_sink=lambda s: sinks["ledger"].append(s),
        surface_sink=lambda s: sinks["surface"].append(s),
        execution_sink=lambda s: (sinks["exec"].append(s), "EXECUTED")[1],
    )
    return o, sinks


def test_shadow_signal_only_ledgered():
    o, sinks = _spine()
    o.route(Signal("some_model", Tier.SHADOW, "context", {"x": 1}))
    assert len(sinks["ledger"]) == 1
    assert sinks["surface"] == [] and sinks["exec"] == []


def test_trusted_signal_ledgered_and_surfaced():
    o, sinks = _spine()
    o.set_ceiling("recommender", Tier.TRUSTED)
    o.route(Signal("recommender", Tier.TRUSTED, "recommendation",
                   {"sym": "X"}, reason="fell 40% near money"))
    assert len(sinks["ledger"]) == 1 and len(sinks["surface"]) == 1
    assert sinks["exec"] == []


def test_ungraduated_model_clamped_below_execution():
    """A model that REQUESTS execution but hasn't graduated is clamped —
    the core safety property. It must NOT reach the execution sink."""
    o, sinks = _spine()
    # no ceiling set -> defaults to SHADOW
    res = o.route(Signal("greedy_model", Tier.EXECUTION, "exit",
                         {"sym": "X"}, reason="i'm very confident"))
    assert res is None
    assert sinks["exec"] == []
    assert o.clamped == 1


def test_hardwired_exit_actors_reach_execution():
    o, sinks = _spine()
    res = o.route(Signal("trailing_stop", Tier.EXECUTION, "exit",
                         {"sym": "X", "qty": 75}, reason="cushion hit"))
    assert res == "EXECUTED"
    assert len(sinks["exec"]) == 1


def test_curator_can_promote_a_source():
    o, sinks = _spine()
    o.set_ceiling("proven_scientist", Tier.TRUSTED)
    o.route(Signal("proven_scientist", Tier.TRUSTED, "recommendation",
                   {}, reason="validated over 40 sessions"))
    assert len(sinks["surface"]) == 1
    # But even promoted to TRUSTED it still can't execute.
    res = o.route(Signal("proven_scientist", Tier.EXECUTION, "exit", {},
                         reason="wants to trade"))
    assert res is None and o.clamped == 1


def test_trusted_signal_without_reason_rejected():
    with pytest.raises(ValueError, match="reason"):
        Signal("x", Tier.TRUSTED, "recommendation", {})


def test_execution_globally_disableable():
    sinks = {"exec": []}
    o = Orchestrator(execution_sink=lambda s: sinks["exec"].append(s),
                     allow_execution=False)
    res = o.route(Signal("trailing_stop", Tier.EXECUTION, "exit", {},
                         reason="cushion hit"))
    assert res is None and sinks["exec"] == []      # kill-switch behaviour


# ---------------------------------------------------------------------------
# Profit-lock ratchet
# ---------------------------------------------------------------------------

def test_founder_example_numbers():
    """peak 11500, lock_ratio 0.539 -> locked 6200, current 8000 ->
    floating 1800. Exactly the founder's worked example."""
    lock = ProfitLock(mode="ratio", lock_ratio=0.5391, activation_floor=1000)
    lock.update(11500)               # establish the peak
    lock.update(8000)                # fall back to current 8000
    s = lock.snapshot()
    assert abs(s["locked_pnl"] - 6200) < 5
    assert abs(s["floating_pnl"] - 1800) < 5
    assert abs(s["worst_case_day_profit"] - 6200) < 5


def test_lock_ratchets_up_never_down():
    lock = ProfitLock(lock_ratio=0.5, activation_floor=1000)
    lock.update(4000)                # locked 2000
    locked_at_4k = lock.snapshot()["locked_pnl"]
    lock.update(10000)               # locked 5000
    locked_at_10k = lock.snapshot()["locked_pnl"]
    lock.update(6000)                # peak stays 10000 -> locked stays 5000
    locked_after_fall = lock.snapshot()["locked_pnl"]
    assert locked_at_4k == 2000
    assert locked_at_10k == 5000
    assert locked_after_fall == 5000     # did NOT fall with the P&L


def test_lock_fires_when_current_hits_floor():
    lock = ProfitLock(lock_ratio=0.5, activation_floor=1000)
    assert lock.update(10000) is False   # peak 10000, locked 5000
    assert lock.update(7000) is False    # above floor
    assert lock.update(5000) is True     # hit the locked floor -> fire
    assert lock.update(3000) is False    # fires only once


def test_lock_does_not_activate_below_floor():
    lock = ProfitLock(lock_ratio=0.5, activation_floor=2000)
    lock.update(1500)                    # peak below activation floor
    s = lock.snapshot()
    assert s["activated"] is False
    assert s["locked_pnl"] == 0.0        # nothing locked yet


def test_buffer_mode():
    """'never give back more than Rs 3000 from peak.'"""
    lock = ProfitLock(mode="buffer", give_back_buffer=3000,
                      activation_floor=1000)
    lock.update(10000)                   # locked = 10000 - 3000 = 7000
    assert lock.snapshot()["locked_pnl"] == 7000
    assert lock.update(7000) is True     # gave back exactly 3000 -> fire


def test_invalid_configs_rejected():
    with pytest.raises(ValueError):
        ProfitLock(mode="ratio", lock_ratio=1.5)
    with pytest.raises(ValueError):
        ProfitLock(mode="buffer", give_back_buffer=0)
    with pytest.raises(ValueError):
        ProfitLock(mode="nonsense")
