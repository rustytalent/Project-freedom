from __future__ import annotations

from liqpool.research.belief.executor import (
    INTENT_BLOCKED,
    INTENT_EXIT_POSITION,
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
