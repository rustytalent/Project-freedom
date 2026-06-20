"""Tests for executor_v4.cockpit — operator dashboard adapter."""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4.cockpit import (
    ACT_ENTER,
    ACT_EXIT,
    ACT_HOLD,
    ACT_REFUSE,
    build_cockpit_snapshot,
)
from liqpool.research.belief.executor_v4.manager import (
    PortfolioManager,
)


def _mk(*, bars: int = 100, action: str = "HOLD", direction: int = 1,
        thesis: str = "HOLD_BULL", iv: str = "directional_bull",
        confidence: float = 0.82,
        spot: float = 23000.0) -> dict:
    ts = pd.Timestamp("2026-06-19 10:00") + pd.Timedelta(seconds=bars)
    slot_readings = []
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "CE", "level": level,
            "label": f"CE_{'ATM' if level == 0 else (f'OTM{level}' if level > 0 else f'ITM{abs(level)}')}",
            "acceptance": "normal", "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "is_abnormal": False, "dod_z": 1.5, "mark_price": 100.0,
        })
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "PE", "level": level,
            "label": f"PE_{'ATM' if level == 0 else (f'OTM{level}' if level > 0 else f'ITM{abs(level)}')}",
            "acceptance": "normal", "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "is_abnormal": False, "dod_z": -1.0, "mark_price": 100.0,
        })
    return {
        "ts": ts, "spot": spot, "bars_seen": bars, "is_warm": True,
        "thesis": {"composite_state": thesis,
                   "bull_thesis_score": 70.0, "bear_thesis_score": 10.0,
                   "no_trade_score": 5.0},
        "iv_state": {"state": iv, "direction": direction, "confidence": 0.85,
                     "clean_mark_fraction": 0.98, "net_intent_z": 2.5},
        "battlefield": {
            "verdict": "bullish_agreement", "direction": direction,
            "confidence": 0.8,
            "ce_rail": {"weighted_mean_signed_z": 1.5,
                        "dispersion_score": 0.10,
                        "epicenter_label": "CE_ATM",
                        "epicenter_level": 0},
            "pe_rail": {"weighted_mean_signed_z": -1.0,
                        "dispersion_score": 0.10,
                        "epicenter_label": "PE_ATM",
                        "epicenter_level": 0},
        },
        "decision": {"action": action, "direction": direction,
                     "confidence": confidence, "trade_allowed": True,
                     "strike": {"side": "CE", "level": 0,
                                 "label": "CE_ATM"}},
        "winding": {"zone": "NO_WINDING"},
        "bull_state": {"state_index": 2},
        "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


def _warmup(mgr: PortfolioManager, n: int = 90) -> None:
    for i in range(n):
        mgr.evaluate(_mk(bars=100 + i, action="HOLD"))


def test_cockpit_hold_card():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="HOLD"))
    snap = build_cockpit_snapshot(intent.to_dict())
    assert snap.action_card["kind"] == ACT_HOLD
    assert snap.action_card["headline"] == "HOLD"


def test_cockpit_enter_card_carries_entry_details():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    snap = build_cockpit_snapshot(intent.to_dict())
    assert snap.action_card["kind"] == ACT_ENTER
    assert "OPEN" in snap.action_card["headline"]
    assert snap.action_card["premium"] is not None
    assert snap.action_card["aggregator_decision"] in ("ACCEPT", "SIZE_DOWN")


def test_cockpit_exit_card_after_engine_exit():
    mgr = PortfolioManager()
    _warmup(mgr)
    mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    intent = mgr.evaluate(_mk(bars=201, action="EXIT"))
    snap = build_cockpit_snapshot(intent.to_dict())
    assert snap.action_card["kind"] == ACT_EXIT
    assert "EXIT" in snap.action_card["headline"]


def test_cockpit_refuse_card_when_entry_blocked():
    mgr = PortfolioManager()
    # No warmup history → MTF alignment / data quality refuses entry.
    intent = mgr.evaluate(_mk(bars=10, action="ENTER_LONG"))
    snap = build_cockpit_snapshot(intent.to_dict())
    assert snap.action_card["kind"] == ACT_REFUSE
    assert "REFUSE" in snap.action_card["headline"]
    assert snap.action_card.get("all_reasons")


def test_cockpit_web_panel_carries_top_scenarios():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="HOLD"))
    snap = build_cockpit_snapshot(intent.to_dict())
    assert "n_active" in snap.web_panel
    assert "directional_consensus" in snap.web_panel
    assert "top_scenarios" in snap.web_panel


def test_cockpit_mm_panel_has_intent_and_guidance():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="HOLD"))
    snap = build_cockpit_snapshot(intent.to_dict())
    assert "dominant_intent" in snap.mm_panel
    assert "operator_guidance" in snap.mm_panel


def test_cockpit_fat_tail_dial_carries_score_and_action():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="HOLD"))
    snap = build_cockpit_snapshot(intent.to_dict())
    assert "tail_score" in snap.fat_tail_dial
    assert "action" in snap.fat_tail_dial
    assert snap.fat_tail_dial["action"] in (
        "SCALE_UP", "NORMAL", "HEDGE", "REFUSE")


def test_cockpit_risk_panel_has_kill_switches():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="HOLD"))
    snap = build_cockpit_snapshot(intent.to_dict())
    assert "kill_switches" in snap.risk_panel
    assert isinstance(snap.risk_panel["kill_switches"], list)


def test_cockpit_positions_panel_after_open():
    mgr = PortfolioManager()
    _warmup(mgr)
    mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    intent = mgr.evaluate(_mk(bars=201, action="HOLD"))
    snap = build_cockpit_snapshot(intent.to_dict())
    assert snap.positions_panel["n_open"] == 1
    assert snap.positions_panel["open"]


def test_cockpit_pnl_panel_tracks_realized():
    mgr = PortfolioManager()
    _warmup(mgr)
    mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    intent = mgr.evaluate(_mk(bars=201, action="EXIT"))
    snap = build_cockpit_snapshot(intent.to_dict())
    assert "daily_pnl_rupees" in snap.pnl_panel
    assert "cumulative_fees_rupees" in snap.pnl_panel


def test_cockpit_explainer_text_non_empty():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    snap = build_cockpit_snapshot(intent.to_dict())
    assert "Tick" in snap.explainer_text


def test_cockpit_serializable():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    snap = build_cockpit_snapshot(intent.to_dict())
    import json
    json.dumps(snap.to_dict(), default=str)
