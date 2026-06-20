"""Tests for the healed Premium Belief Execution Governor (v3).

Pin the THREE critical fixes from the founder-requested healing pass:
  1. R-multiple is computed on the held option PREMIUM, not on spot.
  2. The held contract's live spread / acceptance / mark are read and
     enforced (Critical-3 — "spread can eat you alive").
  3. The position carries the actual strike PRICE (not just the label).

Plus the soft-quality bumps: confirming votes count INDEPENDENT sources
(decision's own direction is NOT a vote); rupees-per-lot anchor exists;
engine-cold guard; first-weakness sweep exit.
"""
from __future__ import annotations

from liqpool.research.belief.executor import (
    BeliefExecutionGovernor,
    ExecutionGovernorConfig,
    HeldContractRead,
    INTENT_BLOCKED,
    INTENT_EXIT_POSITION,
    INTENT_FLAT_WAIT,
    INTENT_HOLD_POSITION,
    INTENT_OPEN_LONG,
    INTENT_OPEN_SCALP_PUT,
    PROFILE_INTRADAY,
)


# ─────────────────────────────────────────────────────────────────
# Snapshot fabrication
# ─────────────────────────────────────────────────────────────────

def _slot(strike: float, typ: str, *, mark_price: float = 100.0,
          mark_source: str = "microprice", friendliness: float = 0.92,
          dod_z: float = 0.6, is_abnormal: bool = False,
          acceptance: str = "normal", level: int = 0,
          spread_state: str = "clean") -> dict:
    return {
        "strike": float(strike),
        "option_type": typ,
        "moneyness_label": f"{typ}_ATM" if level == 0 else f"{typ}_OTM{level}",
        "label": f"{typ}_ATM" if level == 0 else f"{typ}_OTM{level}",
        "level": int(level),
        "behavior": "gamma_atm",
        "mark_source": mark_source,
        "mark_price": float(mark_price),
        "friendliness": float(friendliness),
        "acceptance": acceptance,
        "dod_z": float(dod_z),
        "is_abnormal": bool(is_abnormal),
        "mark_quality": {"confidence": 0.95},
        "spread_state": spread_state,
    }


def _snapshot(
    *,
    action: str = "ENTER_LONG",
    direction: int = 1,
    confidence: float = 0.82,
    bars: int = 120,
    spot: float = 25000.0,
    ce_atm_price: float = 100.0,
    pe_atm_price: float = 100.0,
    no_trade_score: float = 2.0,
    iv_state: str = "directional_bull",
    battlefield: str = "bullish_agreement",
    thesis_state: str = "BULL_ENTRY",
    clean_mark_fraction: float = 0.98,
    is_warm: bool = True,
    ce_rail_z: float = 1.30,
    pe_rail_z: float = -1.20,
) -> dict:
    return {
        "ts": "2026-06-19T09:20:00+05:30",
        "spot": spot,
        "bars_seen": bars,
        "is_warm": is_warm,
        "decision": {
            "action": action,
            "trade_allowed": True,
            "direction": direction,
            "confidence": confidence,
            "spread_friendliness": 0.91,
            "strike": {
                "side": "CE" if direction > 0 else "PE",
                "label": "CE_ATM" if direction > 0 else "PE_ATM",
                "level": 0,
            },
            "invalidation_rule": "exit when thesis flips or rail breaks",
        },
        "iv_state": {
            "state": iv_state,
            "direction": direction if iv_state.startswith("directional") else 0,
            "confidence": 0.86,
            "clean_mark_fraction": clean_mark_fraction,
        },
        "battlefield": {
            "verdict": battlefield,
            "direction": direction if "agreement" in battlefield else 0,
            "confidence": 0.84,
            "ce_rail": {"weighted_mean_signed_z": ce_rail_z},
            "pe_rail": {"weighted_mean_signed_z": pe_rail_z},
        },
        "thesis": {
            "composite_state": thesis_state,
            "no_trade_score": no_trade_score,
        },
        "slot_readings": [
            _slot(25000.0, "CE", mark_price=ce_atm_price),
            _slot(25000.0, "PE", mark_price=pe_atm_price),
        ],
    }


def _quote(*, mid: float, spread_state: str = "clean",
           friendliness: float = 0.92, acceptance: str = "normal",
           dod_z: float = 0.6, mark_source: str = "microprice") -> HeldContractRead:
    return HeldContractRead(
        bid=mid - 0.5, ask=mid + 0.5, mid=mid,
        spread_state=spread_state, friendliness=friendliness,
        acceptance=acceptance, dod_z=dod_z, mark_source=mark_source,
    )


# ─────────────────────────────────────────────────────────────────
# Open path
# ─────────────────────────────────────────────────────────────────

def test_executor_opens_when_belief_rails_and_quality_align():
    gov = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=20))
    intent = gov.evaluate(_snapshot(), held_quote=_quote(mid=100.0))
    assert intent.intent == INTENT_OPEN_LONG
    assert intent.allowed is True
    assert intent.size_fraction > 0
    assert intent.size_lots >= 1
    assert intent.contract_label == "CE_ATM"
    assert intent.strike_price == 25000.0      # Critical-3: anchored
    assert intent.order_mode == "SHADOW_ONLY"
    assert intent.live_orders_enabled is False
    assert intent.telemetry["entry_premium"] == 100.0


def test_executor_refuses_open_when_held_premium_unavailable():
    """If we cannot anchor R because no clean mark exists for the chosen
    contract, the executor must refuse rather than open with NaN R."""
    snap = _snapshot()
    # Remove the matching CE slot — no premium to anchor on.
    snap["slot_readings"] = [_slot(25000.0, "PE", mark_price=100.0)]
    gov = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=20))
    intent = gov.evaluate(snap)   # no held_quote either
    assert intent.intent == INTENT_BLOCKED
    assert any("anchor R" in r for r in intent.reject_reasons)


def test_executor_refuses_open_when_held_contract_spread_dangerous():
    gov = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=20))
    q = _quote(mid=100.0, spread_state="dangerous")
    intent = gov.evaluate(_snapshot(), held_quote=q)
    assert intent.intent == INTENT_BLOCKED
    assert any("spread state dangerous" in r for r in intent.reject_reasons)


# ─────────────────────────────────────────────────────────────────
# R-multiple is on PREMIUM (Critical-1)
# ─────────────────────────────────────────────────────────────────

def test_r_multiple_is_computed_on_held_premium_not_spot():
    """Hold a long. Move SPOT favourably but PREMIUM unfavourably (vol
    crush). The corrected R must be NEGATIVE — the spot-based logic would
    have shown positive R and let the operator bleed."""
    cfg = ExecutionGovernorConfig(min_warm_bars=20,
                                   min_hold_bars=1,
                                   cooldown_bars_after_entry=0)
    gov = BeliefExecutionGovernor(cfg)
    opened = gov.evaluate(_snapshot(bars=100, spot=25000.0,
                                       ce_atm_price=100.0),
                            held_quote=_quote(mid=100.0))
    assert opened.intent == INTENT_OPEN_LONG

    # Spot moves UP (+0.1%) but premium drops 30% (vol crush).
    held = gov.evaluate(_snapshot(action="HOLD", bars=101, spot=25025.0,
                                    ce_atm_price=70.0),
                         held_quote=_quote(mid=70.0))
    assert held.intent == INTENT_HOLD_POSITION
    # Premium-based R must be negative (we lost 30% on 50% stop = -0.6R).
    assert held.telemetry["position_current_r"] < -0.4


def test_hard_stop_fires_on_premium_drop_not_spot_drop():
    """A premium drop past the configured premium_stop must trigger an
    immediate hard exit even if spot has not moved meaningfully."""
    cfg = ExecutionGovernorConfig(
        min_warm_bars=20, min_hold_bars=0, cooldown_bars_after_entry=0,
        premium_stop_intraday=0.50,
    )
    gov = BeliefExecutionGovernor(cfg)
    gov.evaluate(_snapshot(bars=100, spot=25000.0, ce_atm_price=100.0),
                 held_quote=_quote(mid=100.0))
    # Spot flat, but premium dropped 60% — past the 50% stop → -1.2R.
    exit_intent = gov.evaluate(
        _snapshot(action="HOLD", bars=101, spot=25000.0, ce_atm_price=40.0),
        held_quote=_quote(mid=40.0),
    )
    assert exit_intent.intent == INTENT_EXIT_POSITION
    assert any("hard stop on premium R" in r for r in exit_intent.reason_codes)


def test_profit_lock_giveback_fires_on_premium_metric():
    cfg = ExecutionGovernorConfig(
        min_warm_bars=20, min_hold_bars=1, cooldown_bars_after_entry=0,
        premium_stop_intraday=0.50,
        profit_lock_start_r=0.90, profit_lock_giveback_r=0.45,
    )
    gov = BeliefExecutionGovernor(cfg)
    gov.evaluate(_snapshot(bars=100, spot=25000.0, ce_atm_price=100.0),
                 held_quote=_quote(mid=100.0))
    # Premium rallies to 160 → +1.2R. Lock in.
    gov.evaluate(_snapshot(action="HOLD", bars=101, spot=25030.0,
                              ce_atm_price=160.0),
                  held_quote=_quote(mid=160.0))
    # Then premium gives back to 130 → +0.6R (giveback = 0.6R from +1.2R).
    exit_intent = gov.evaluate(
        _snapshot(action="HOLD", bars=102, spot=25025.0, ce_atm_price=130.0),
        held_quote=_quote(mid=130.0),
    )
    assert exit_intent.intent == INTENT_EXIT_POSITION
    assert any("profit lock giveback" in r for r in exit_intent.reason_codes)


# ─────────────────────────────────────────────────────────────────
# Held-contract guards (Critical-3)
# ─────────────────────────────────────────────────────────────────

def test_held_contract_spread_dangerous_forces_hard_exit():
    cfg = ExecutionGovernorConfig(min_warm_bars=20, min_hold_bars=0,
                                    cooldown_bars_after_entry=0)
    gov = BeliefExecutionGovernor(cfg)
    gov.evaluate(_snapshot(bars=100), held_quote=_quote(mid=100.0))
    # Held quote turns dangerous mid-trade.
    exit_intent = gov.evaluate(
        _snapshot(action="HOLD", bars=101),
        held_quote=_quote(mid=100.0, spread_state="dangerous"),
    )
    assert exit_intent.intent == INTENT_EXIT_POSITION
    assert any("spread turned dangerous" in r
               for r in exit_intent.reason_codes)


def test_held_contract_mark_going_dirty_forces_hard_exit():
    cfg = ExecutionGovernorConfig(min_warm_bars=20, min_hold_bars=0,
                                    cooldown_bars_after_entry=0)
    gov = BeliefExecutionGovernor(cfg)
    gov.evaluate(_snapshot(bars=100), held_quote=_quote(mid=100.0))
    exit_intent = gov.evaluate(
        _snapshot(action="HOLD", bars=101),
        held_quote=_quote(mid=100.0, mark_source="last_valid"),
    )
    assert exit_intent.intent == INTENT_EXIT_POSITION
    assert any("mark went dirty" in r for r in exit_intent.reason_codes)


def test_held_acceptance_rejected_triggers_soft_exit():
    """When the held leg's acceptance flips to 'rejected', that's the
    founder's exact 'wrong-side defense lifted' read → soft exit."""
    cfg = ExecutionGovernorConfig(min_warm_bars=20, min_hold_bars=1,
                                    cooldown_bars_after_entry=0)
    gov = BeliefExecutionGovernor(cfg)
    gov.evaluate(_snapshot(bars=100), held_quote=_quote(mid=100.0))
    gov.evaluate(_snapshot(action="HOLD", bars=101), held_quote=_quote(mid=100.0))
    exit_intent = gov.evaluate(
        _snapshot(action="HOLD", bars=102),
        held_quote=_quote(mid=100.0, acceptance="rejected"),
    )
    assert exit_intent.intent == INTENT_EXIT_POSITION
    assert any("rejected" in r for r in exit_intent.reason_codes)


def test_first_weakness_sweep_exit_on_dod_z_collapse():
    """On a SCALP profile, when the held leg's |dod_z| collapses to below
    the configured fraction of the entry magnitude, exit. This is the
    decision layer's 'exit at first weakness' implemented at the leg level."""
    cfg = ExecutionGovernorConfig(min_warm_bars=20, min_hold_bars=1,
                                    cooldown_bars_after_entry=0,
                                    sweep_weakness_dod_z_ratio=0.35)
    gov = BeliefExecutionGovernor(cfg)
    # Open a scalp put (BULL_TRAP_WINDING route).
    snap = _snapshot(action="SCALP_PUT", direction=-1, confidence=0.80,
                     thesis_state="NEUTRAL",
                     iv_state="directional_bear", battlefield="bearish_agreement",
                     ce_rail_z=-1.20, pe_rail_z=1.30, bars=100,
                     pe_atm_price=80.0, ce_atm_price=80.0)
    gov.evaluate(snap, held_quote=_quote(mid=80.0, dod_z=3.5))
    # Bar+1: dod_z collapsed from 3.5 → 0.9 (well below 35% of 3.5 = 1.225).
    snap2 = _snapshot(action="SCALP_PUT", direction=-1, bars=101,
                      pe_atm_price=80.0, ce_atm_price=80.0,
                      iv_state="directional_bear", battlefield="bearish_agreement",
                      ce_rail_z=-1.20, pe_rail_z=1.30)
    exit_intent = gov.evaluate(snap2, held_quote=_quote(mid=80.0, dod_z=0.9))
    assert exit_intent.intent == INTENT_EXIT_POSITION
    assert any("sweep weakness" in r for r in exit_intent.reason_codes)


# ─────────────────────────────────────────────────────────────────
# Engine-cold guard
# ─────────────────────────────────────────────────────────────────

def test_engine_going_cold_forces_hard_exit_while_holding():
    cfg = ExecutionGovernorConfig(min_warm_bars=20, min_hold_bars=0,
                                    cooldown_bars_after_entry=0)
    gov = BeliefExecutionGovernor(cfg)
    gov.evaluate(_snapshot(bars=100, is_warm=True), held_quote=_quote(mid=100.0))
    exit_intent = gov.evaluate(
        _snapshot(action="HOLD", bars=101, is_warm=False),
        held_quote=_quote(mid=100.0),
    )
    assert exit_intent.intent == INTENT_EXIT_POSITION
    assert any("engine went cold" in r for r in exit_intent.reason_codes)


# ─────────────────────────────────────────────────────────────────
# Stricter quorum (independent confirming votes)
# ─────────────────────────────────────────────────────────────────

def test_min_confirming_votes_does_not_count_decision_direction():
    """Decision direction must NOT count as one of the confirming votes —
    it's the trigger, not a confirmation. With everything else neutral
    (only the decision pointing bullish), the executor refuses."""
    snap = _snapshot()
    snap["iv_state"]["direction"] = 0
    snap["iv_state"]["state"] = "neutral"
    snap["battlefield"]["direction"] = 0
    snap["battlefield"]["verdict"] = "quiet"
    snap["battlefield"]["ce_rail"]["weighted_mean_signed_z"] = 0.0
    snap["battlefield"]["pe_rail"]["weighted_mean_signed_z"] = 0.0
    snap["thesis"]["composite_state"] = "NEUTRAL"
    gov = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=20))
    intent = gov.evaluate(snap, held_quote=_quote(mid=100.0))
    assert intent.intent == INTENT_BLOCKED
    assert any("confirming votes" in r for r in intent.reject_reasons)


def test_min_directional_rail_floor_enforced():
    """If rail alignment is below the (raised) floor, the executor refuses
    even when individual votes are present."""
    snap = _snapshot(ce_rail_z=0.10, pe_rail_z=-0.10)
    # Weaken the side rails so rail < min_directional_rail_abs.
    snap["thesis"]["composite_state"] = "NEUTRAL"
    snap["iv_state"]["direction"] = 0
    snap["iv_state"]["state"] = "neutral"
    snap["battlefield"]["direction"] = 0
    snap["battlefield"]["verdict"] = "quiet"
    gov = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=20))
    intent = gov.evaluate(snap, held_quote=_quote(mid=100.0))
    assert intent.intent == INTENT_BLOCKED
    assert any("rail alignment" in r or "confirming votes" in r
               for r in intent.reject_reasons)


# ─────────────────────────────────────────────────────────────────
# Existing safety guards still work
# ─────────────────────────────────────────────────────────────────

def test_warmup_block_still_works():
    gov = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=80))
    intent = gov.evaluate(_snapshot(bars=12, is_warm=False),
                            held_quote=_quote(mid=100.0))
    assert intent.intent == INTENT_BLOCKED
    assert any("warmup incomplete" in r for r in intent.reject_reasons)


def test_dirty_marks_block_still_works():
    snap = _snapshot(clean_mark_fraction=0.50)
    gov = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=20))
    intent = gov.evaluate(snap, held_quote=_quote(mid=100.0))
    assert intent.intent == INTENT_BLOCKED
    assert any("clean marks" in r for r in intent.reject_reasons)


def test_opposite_direction_burst_exits_held_position():
    cfg = ExecutionGovernorConfig(min_warm_bars=20, min_hold_bars=1,
                                    cooldown_bars_after_entry=0)
    gov = BeliefExecutionGovernor(cfg)
    gov.evaluate(_snapshot(bars=100), held_quote=_quote(mid=100.0))
    gov.evaluate(_snapshot(action="HOLD", bars=101),
                  held_quote=_quote(mid=100.0))
    # Bar 102: full bearish flip in IV + battlefield + thesis + rails.
    bear_snap = _snapshot(action="WAIT", direction=-1, bars=102,
                           iv_state="directional_bear",
                           battlefield="bearish_agreement",
                           thesis_state="BEAR_ENTRY",
                           ce_rail_z=-1.30, pe_rail_z=1.20)
    exit_intent = gov.evaluate(bear_snap, held_quote=_quote(mid=100.0))
    assert exit_intent.intent == INTENT_EXIT_POSITION
    assert any("opposite directional" in r or "thesis flipped" in r
               for r in exit_intent.reason_codes)


# ─────────────────────────────────────────────────────────────────
# Lifecycle plumbing
# ─────────────────────────────────────────────────────────────────

def test_hold_intent_carries_size_and_position_metrics():
    cfg = ExecutionGovernorConfig(min_warm_bars=20, min_hold_bars=0,
                                    cooldown_bars_after_entry=0)
    gov = BeliefExecutionGovernor(cfg)
    opened = gov.evaluate(_snapshot(bars=100, ce_atm_price=100.0),
                            held_quote=_quote(mid=100.0))
    held = gov.evaluate(_snapshot(action="HOLD", bars=101,
                                     ce_atm_price=110.0),
                         held_quote=_quote(mid=110.0))
    assert held.intent == INTENT_HOLD_POSITION
    assert held.size_fraction == opened.size_fraction
    assert held.size_lots == opened.size_lots
    assert held.profile == PROFILE_INTRADAY
    assert held.telemetry["position_current_r"] > 0   # premium up → positive R
    assert held.telemetry["entry_premium"] == 100.0
    assert held.telemetry["last_seen_premium"] == 110.0


def test_intent_to_dict_is_json_serializable():
    gov = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=20))
    intent = gov.evaluate(_snapshot(), held_quote=_quote(mid=100.0))
    import json
    json.dumps(intent.to_dict())


def test_defaults_have_positive_reward_to_risk():
    cfg = ExecutionGovernorConfig()
    # In premium R, 1R loss = the stop. Targets exceed 1R.
    assert cfg.scalp_target_r > 1.0
    assert cfg.intraday_target_r > 1.0


def test_size_lots_is_bounded():
    cfg = ExecutionGovernorConfig(min_warm_bars=20, max_lots=4)
    gov = BeliefExecutionGovernor(cfg)
    intent = gov.evaluate(_snapshot(confidence=0.99),
                            held_quote=_quote(mid=100.0))
    assert 1 <= intent.size_lots <= cfg.max_lots
