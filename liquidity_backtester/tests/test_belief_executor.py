from __future__ import annotations

from liqpool.research.belief.executor import (
    INTENT_BLOCKED,
    INTENT_EXIT_POSITION,
    INTENT_HOLD_POSITION,
    INTENT_OPEN_LONG,
    BeliefExecutionGovernor,
    ExecutionGovernorConfig,
)


def _slot(strike: float = 25000.0, typ: str = "CE", *, dirty: bool = False):
    return {
        "strike": strike,
        "option_type": typ,
        "moneyness_label": "ATM",
        "level": 0,
        "behavior": "normal",
        "mark_source": "ltp" if dirty else "microprice",
        "friendliness": 0.92,
        "acceptance": 1.0,
        "dod_z": 0.65,
        "is_abnormal": False,
        "mark_quality": {"confidence": 0.95},
    }


def _snapshot(
    *,
    action: str = "ENTER_LONG",
    direction: int = 1,
    confidence: float = 0.82,
    bars: int = 120,
    spot: float = 25000.0,
    dirty: bool = False,
    no_trade_score: float = 2.0,
):
    bull = direction > 0
    return {
        "ts": "2026-06-17T09:20:00+05:30",
        "spot": spot,
        "bars_seen": bars,
        "is_warm": bars >= 20,
        "decision": {
            "action": action,
            "trade_allowed": True,
            "direction": direction,
            "confidence": confidence,
            "spread_friendliness": 0.91,
            "strike": {
                "side": "CE" if bull else "PE",
                "label": "CE_ATM" if bull else "PE_ATM",
                "level": 0,
            },
            "invalidation_rule": "exit when thesis flips or rail breaks",
        },
        "iv_state": {
            "state": "directional_bull" if bull else "directional_bear",
            "direction": direction,
            "confidence": 0.86,
            "clean_mark_fraction": 0.50 if dirty else 0.98,
        },
        "battlefield": {
            "verdict": "bullish_agreement" if bull else "bearish_agreement",
            "direction": direction,
            "confidence": 0.84,
            "ce_rail": {"weighted_mean_signed_z": 1.0 if bull else -0.85},
            "pe_rail": {"weighted_mean_signed_z": -0.80 if bull else 1.10},
        },
        "thesis": {
            "composite_state": "BULL_ENTRY" if bull else "BEAR_ENTRY",
            "no_trade_score": no_trade_score,
        },
        "slot_readings": [
            _slot(25000.0, "CE", dirty=dirty),
            _slot(25000.0, "PE", dirty=dirty),
        ],
    }


def test_executor_opens_when_belief_rails_and_quality_align():
    governor = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=20))
    intent = governor.evaluate(_snapshot())

    assert intent.intent == INTENT_OPEN_LONG
    assert intent.allowed is True
    assert intent.size_fraction > 0
    assert intent.contract_label == "CE_ATM"
    assert intent.order_mode == "SHADOW_ONLY"
    assert intent.live_orders_enabled is False
    assert intent.telemetry["directional_votes"] >= 2


def test_executor_defaults_have_positive_reward_to_risk():
    cfg = ExecutionGovernorConfig()

    assert cfg.scalp_target_r > cfg.stop_r
    assert cfg.intraday_target_r > cfg.stop_r


def test_executor_hold_carries_size_and_serialized_entry_context():
    governor = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=20))
    opened = governor.evaluate(_snapshot(action="ENTER_LONG", direction=1, bars=100, spot=25000.0))
    assert opened.intent == INTENT_OPEN_LONG

    held = governor.evaluate(_snapshot(action="HOLD", direction=1, confidence=0.80, bars=101, spot=25010.0))

    assert held.intent == INTENT_HOLD_POSITION
    assert held.size_fraction == opened.size_fraction
    assert held.size_fraction > 0
    assert held.state["entry_context"]["thesis_state"] == "BULL_ENTRY"
    assert held.state["entry_context"]["directional_votes"] >= 2
    assert held.telemetry["position_current_r"] > 0


def test_executor_profit_lock_exits_after_giveback():
    cfg = ExecutionGovernorConfig(
        min_warm_bars=20,
        min_hold_bars=1,
        cooldown_bars_after_entry=0,
    )
    governor = BeliefExecutionGovernor(cfg)
    opened = governor.evaluate(_snapshot(action="ENTER_LONG", direction=1, bars=100, spot=25000.0))
    assert opened.intent == INTENT_OPEN_LONG

    peak_spot = 25000.0 * (1.0 + cfg.adverse_spot_stop_pct * 1.20)
    held = governor.evaluate(_snapshot(action="HOLD", direction=1, confidence=0.90, bars=101, spot=peak_spot))
    assert held.intent == INTENT_HOLD_POSITION

    giveback_spot = 25000.0 * (1.0 + cfg.adverse_spot_stop_pct * 0.70)
    exit_intent = governor.evaluate(_snapshot(action="HOLD", direction=1, confidence=0.82, bars=102, spot=giveback_spot))

    assert exit_intent.intent == INTENT_EXIT_POSITION
    assert any("profit lock giveback" in reason for reason in exit_intent.reason_codes)


def test_executor_blocks_dirty_marks_before_entry():
    governor = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=20))
    intent = governor.evaluate(_snapshot(dirty=True))

    assert intent.intent == INTENT_BLOCKED
    assert intent.allowed is False
    assert any("clean marks" in reason for reason in intent.reject_reasons)


def test_executor_blocks_until_warmup_completes():
    governor = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=80))
    intent = governor.evaluate(_snapshot(bars=12))

    assert intent.intent == INTENT_BLOCKED
    assert any("warmup incomplete" in reason for reason in intent.reject_reasons)


def test_executor_exits_when_position_story_flips():
    governor = BeliefExecutionGovernor(ExecutionGovernorConfig(min_warm_bars=20))
    opened = governor.evaluate(_snapshot(action="ENTER_LONG", direction=1, bars=100, spot=25000.0))
    assert opened.intent == INTENT_OPEN_LONG

    exit_intent = governor.evaluate(_snapshot(action="WAIT", direction=-1, bars=104, spot=24995.0))

    assert exit_intent.intent == INTENT_EXIT_POSITION
    assert exit_intent.allowed is True
    assert any("opposite directional vote" in reason for reason in exit_intent.reason_codes)
