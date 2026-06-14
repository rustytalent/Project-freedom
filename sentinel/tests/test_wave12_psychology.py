"""Wave 12 tests — the behavioral engine.

Each detector is grounded in a citation (Kahneman, Steenbarger, Tversky,
Shefrin & Statman, Gilovich-Vallone-Tversky, Tharp, Elster). Tests pin:
  * each detector fires on the right pattern and stays silent otherwise
  * citations + reason codes carry through
  * TiltIndex bands + EWMA decay are honest
  * IntentionContract violates correctly + spine refuses non-exit orders
    in RED band
  * MindReport rolls up the session
  * /api/state, /api/psychology, /api/intention, /api/mind_report shapes
  * cockpit ships the panel
"""
from __future__ import annotations

import importlib
import json
import time

import pytest
from fastapi.testclient import TestClient

from sentinel.live_publisher import LivePublisher
from sentinel.psychology import (
    AnchoringDetector, BiasEvent, DispositionEffectDetector,
    FomoEntryDetector, HotHandDetector, IntentionContract, MindReport,
    PsychologyEngine, RevengeTradeDetector, RiskBudgetDetector,
    TiltIndex, TiltState, TradeAction,
)


def _A(ts, action, **kw):
    return TradeAction(ts_ist=ts, action=action, **kw)


# ---------------------------------------------------------------------------
# RevengeTradeDetector — Steenbarger 2009
# ---------------------------------------------------------------------------

def test_revenge_fires_on_quick_entry_after_loss():
    actions = [
        _A("10:00:00", "EXIT", symbol="X", qty=75, pnl=-1500.0),
        _A("10:02:30", "ENTRY", symbol="Y", qty=75, premium=180.0),
    ]
    ev = RevengeTradeDetector().scan(actions, "10:02:31")
    assert ev is not None and ev.bias == "REVENGE"
    assert "Steenbarger" in ev.citation
    assert ev.severity > 0.1
    assert "min after closing" in ev.evidence


def test_revenge_silent_when_delay_exceeds_window():
    actions = [
        _A("10:00:00", "EXIT", symbol="X", qty=75, pnl=-2000.0),
        _A("10:30:00", "ENTRY", symbol="Y", qty=75, premium=180.0),
    ]
    assert RevengeTradeDetector().scan(actions, "10:30:01") is None


def test_revenge_silent_when_previous_exit_was_profit():
    actions = [
        _A("10:00:00", "EXIT", symbol="X", qty=75, pnl=+1500.0),
        _A("10:02:00", "ENTRY", symbol="Y", qty=75),
    ]
    assert RevengeTradeDetector().scan(actions, "10:02:01") is None


def test_revenge_severity_higher_with_escalated_size():
    small = [_A("10:00:00", "EXIT", symbol="X", qty=75, pnl=-2000.0),
             _A("10:01:00", "ENTRY", symbol="Y", qty=75)]
    big = [_A("10:00:00", "EXIT", symbol="X", qty=75, pnl=-2000.0),
           _A("10:01:00", "ENTRY", symbol="Y", qty=300)]
    s_ev = RevengeTradeDetector().scan(small, "10:01:01")
    b_ev = RevengeTradeDetector().scan(big, "10:01:01")
    assert s_ev and b_ev
    assert b_ev.severity > s_ev.severity


# ---------------------------------------------------------------------------
# FomoEntryDetector — Lo 2017
# ---------------------------------------------------------------------------

def test_fomo_fires_when_chase_after_fast_rally():
    spot_hist = [("10:00:00", 25000.0), ("10:05:00", 25080.0), ("10:08:00", 25180.0)]
    actions = [_A("10:08:30", "ENTRY", symbol="NIFTY25000CE", qty=75)]
    ev = FomoEntryDetector(lookback_min=10, move_pct=0.4).scan(
        actions, spot_hist, "10:08:31")
    assert ev is not None and ev.bias == "FOMO"
    assert "Lo" in ev.citation
    assert "after spot moved" in ev.evidence


def test_fomo_silent_in_flat_tape():
    spot_hist = [("10:00:00", 25000.0), ("10:05:00", 25002.0), ("10:08:00", 25001.0)]
    actions = [_A("10:08:30", "ENTRY", symbol="NIFTY25000CE", qty=75)]
    assert FomoEntryDetector().scan(actions, spot_hist, "10:08:31") is None


# ---------------------------------------------------------------------------
# HotHandDetector — Gilovich/Vallone/Tversky 1985
# ---------------------------------------------------------------------------

def test_hot_hand_fires_after_three_wins_then_size_up():
    actions = [
        _A("10:00:00", "EXIT", qty=75, pnl=+500),
        _A("10:10:00", "EXIT", qty=75, pnl=+800),
        _A("10:20:00", "EXIT", qty=75, pnl=+1200),
        _A("10:25:00", "ENTRY", qty=225),
    ]
    ev = HotHandDetector().scan(actions, "10:25:01")
    assert ev is not None and ev.bias == "HOT_HAND"
    assert "Gilovich" in ev.citation


def test_hot_hand_silent_with_loss_in_streak():
    actions = [
        _A("10:00:00", "EXIT", qty=75, pnl=+500),
        _A("10:10:00", "EXIT", qty=75, pnl=-200),
        _A("10:20:00", "EXIT", qty=75, pnl=+800),
        _A("10:25:00", "ENTRY", qty=225),
    ]
    assert HotHandDetector().scan(actions, "10:25:01") is None


def test_hot_hand_silent_when_size_not_escalated():
    actions = [
        _A("10:00:00", "EXIT", qty=75, pnl=+500),
        _A("10:10:00", "EXIT", qty=75, pnl=+800),
        _A("10:20:00", "EXIT", qty=75, pnl=+1200),
        _A("10:25:00", "ENTRY", qty=75),
    ]
    assert HotHandDetector().scan(actions, "10:25:01") is None


# ---------------------------------------------------------------------------
# DispositionEffectDetector — Shefrin & Statman 1985
# ---------------------------------------------------------------------------

def test_disposition_fires_when_losers_held_much_longer_than_winners():
    completed = [
        _A("10:00:00", "EXIT", pnl=+500,  extras={"hold_minutes": 8}),
        _A("10:10:00", "EXIT", pnl=+700,  extras={"hold_minutes": 6}),
        _A("10:30:00", "EXIT", pnl=-900,  extras={"hold_minutes": 45}),
        _A("11:00:00", "EXIT", pnl=-400,  extras={"hold_minutes": 38}),
    ]
    ev = DispositionEffectDetector().scan(completed, "11:00:01")
    assert ev is not None and ev.bias == "DISPOSITION"
    assert "Shefrin" in ev.citation
    assert ev.extras["hold_ratio"] > 1.6


def test_disposition_silent_when_holds_balanced():
    completed = [
        _A("10:00:00", "EXIT", pnl=+500,  extras={"hold_minutes": 10}),
        _A("10:10:00", "EXIT", pnl=+700,  extras={"hold_minutes": 12}),
        _A("10:30:00", "EXIT", pnl=-900,  extras={"hold_minutes": 11}),
        _A("11:00:00", "EXIT", pnl=-400,  extras={"hold_minutes": 13}),
    ]
    assert DispositionEffectDetector().scan(completed, "11:00:01") is None


# ---------------------------------------------------------------------------
# AnchoringDetector — Tversky & Kahneman 1974
# ---------------------------------------------------------------------------

def test_anchoring_fires_on_round_number_cushion():
    """₹100 cushion on a ₹180 premium = anchored to round number."""
    actions = [_A("10:00:00", "TRAIL_ARMED", premium=180.0,
                  extras={"cushion_rupees": 100.0, "premium": 180.0})]
    ev = AnchoringDetector().scan(actions, "10:00:01")
    assert ev is not None and ev.bias == "ANCHORING"
    assert "Tversky" in ev.citation


def test_anchoring_silent_on_atr_derived_cushion():
    actions = [_A("10:00:00", "TRAIL_ARMED", premium=180.0,
                  extras={"cushion_rupees": 37.5, "premium": 180.0})]
    assert AnchoringDetector().scan(actions, "10:00:01") is None


# ---------------------------------------------------------------------------
# RiskBudgetDetector — Tharp 2007
# ---------------------------------------------------------------------------

def test_risk_budget_fires_when_loss_past_hard_limit():
    ev = RiskBudgetDetector(hard_loss_rupees=3000).scan(
        day_pnl=-3500, open_positions=2, now_ist="11:00:00")
    assert ev is not None and ev.bias == "RISK_BUDGET"
    assert "Tharp" in ev.citation


def test_risk_budget_silent_when_within_limit():
    ev = RiskBudgetDetector(hard_loss_rupees=3000).scan(
        day_pnl=-1500, open_positions=2, now_ist="11:00:00")
    assert ev is None


# ---------------------------------------------------------------------------
# TiltIndex + bands
# ---------------------------------------------------------------------------

def test_tilt_starts_green_and_climbs_on_severe_event():
    t = TiltIndex(spike_gain=50)
    assert t.peek() < 30 and TiltIndex.band_for(t.peek()) == "GREEN"
    t.observe(BiasEvent(bias="REVENGE", severity=0.9, detected_at_ist="t",
                         citation="x", evidence="y"))
    v = t.peek()
    assert v > 30
    assert TiltIndex.band_for(v) in ("AMBER", "RED", "CIRCUIT")


def test_tilt_bands_cover_full_range():
    assert TiltIndex.band_for(0) == "GREEN"
    assert TiltIndex.band_for(45) == "AMBER"
    assert TiltIndex.band_for(70) == "RED"
    assert TiltIndex.band_for(95) == "CIRCUIT"


def test_tilt_records_active_biases():
    t = TiltIndex()
    for b in ("REVENGE", "FOMO", "HOT_HAND"):
        t.observe(BiasEvent(bias=b, severity=0.7, detected_at_ist="t",
                             citation="x", evidence="y"))
    assert set(t.biases_active()) == {"REVENGE", "FOMO", "HOT_HAND"}


# ---------------------------------------------------------------------------
# IntentionContract — Ulysses pattern (Elster 2000)
# ---------------------------------------------------------------------------

def test_intention_unset_never_violates():
    ic = IntentionContract()
    assert ic.check(current_pnl=-99999, open_positions=10,
                    now_ist_hms="15:30:00") is None


def test_intention_max_loss_breach():
    ic = IntentionContract(set_at_ist="09:15", max_day_loss_rupees=3000)
    reason = ic.check(current_pnl=-3500, open_positions=2,
                       now_ist_hms="10:00:00")
    assert reason and "exceeded committed max" in reason


def test_intention_no_trade_after_breach_only_with_open_positions():
    ic = IntentionContract(set_at_ist="09:15", max_day_loss_rupees=5000,
                            no_trade_after_ist="15:15")
    # no positions open at 15:20 -> no breach
    assert ic.check(current_pnl=0, open_positions=0,
                    now_ist_hms="15:20:00") is None
    # positions open at 15:20 -> breach
    assert ic.check(current_pnl=0, open_positions=1,
                    now_ist_hms="15:20:00") is not None


# ---------------------------------------------------------------------------
# PsychologyEngine — orchestration
# ---------------------------------------------------------------------------

def test_engine_publishes_bias_to_bus_and_updates_tilt():
    pub = LivePublisher()
    eng = PsychologyEngine(publisher=pub, session="2026-06-13")
    eng.record(_A("10:00:00", "EXIT", symbol="X", qty=75, pnl=-2000))
    eng.record(_A("10:02:00", "ENTRY", symbol="Y", qty=300))
    events = eng.cycle()
    assert any(e.bias == "REVENGE" for e in events)
    # bus has it (TRUSTED → mirrored)
    assert "OPERATOR|psychology" in pub.current()
    # tilt has climbed off zero
    assert eng.state().index > 0


def test_engine_dedupes_repeated_bias():
    pub = LivePublisher()
    eng = PsychologyEngine(publisher=pub, session="2026-06-13")
    eng.record(_A("10:00:00", "EXIT", symbol="X", qty=75, pnl=-2000))
    eng.record(_A("10:02:00", "ENTRY", symbol="Y", qty=300))
    first = eng.cycle()
    second = eng.cycle()
    assert any(e.bias == "REVENGE" for e in first)
    assert not any(e.bias == "REVENGE" for e in second)


def test_engine_blocks_execution_at_red_band():
    eng = PsychologyEngine()
    # Inject several high-severity events to push tilt past 60
    for _ in range(3):
        eng.tilt.observe(BiasEvent(
            bias="REVENGE", severity=0.95, detected_at_ist="t",
            citation="x", evidence="y"))
    assert eng.tilt.peek() >= 60
    assert eng.should_block_execution() is True


def test_engine_intention_violation_fires_bias():
    eng = PsychologyEngine()
    eng.set_intention(max_day_loss_rupees=2000)
    events = eng.cycle(day_pnl=-3000, open_positions=1)
    assert any(e.bias == "INTENTION_VIOLATED" for e in events)
    assert eng.intention.violated is True


def test_mind_report_persists_to_journal(tmp_path):
    path = tmp_path / "mind.jsonl"
    eng = PsychologyEngine(journal_path=path, session="2026-06-13")
    eng.record(_A("10:00:00", "EXIT", pnl=-2000, qty=75))
    eng.record(_A("10:02:00", "ENTRY", qty=300))
    eng.cycle()
    report = eng.mind_report()
    assert isinstance(report, MindReport)
    assert report.session == "2026-06-13"
    assert report.biases_by_type.get("REVENGE", 0) >= 1
    assert "Steenbarger" in " ".join(report.citations)
    # persisted as one JSONL row
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert rows and rows[0]["session"] == "2026-06-13"


# ---------------------------------------------------------------------------
# Server integration
# ---------------------------------------------------------------------------

@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    import sentinel.server as server
    importlib.reload(server)
    return server


def test_state_includes_psychology_and_intention(srv):
    core = srv.CORE
    core._tick_portfolio()
    core._tick_psychology()
    with TestClient(srv.app) as c:
        s = c.get("/api/state").json()
        assert "psychology" in s and "intention" in s
        assert "index" in s["psychology"]
        assert "band" in s["psychology"]


def test_intention_endpoint_sets_and_violates(srv):
    """POST /api/intention commits the contract; subsequent _tick_psychology
    with a P&L past the threshold flips violated=True."""
    with TestClient(srv.app) as c:
        r = c.post("/api/intention",
                   json={"max_day_loss_rupees": 1000,
                          "no_trade_after_ist": "15:15"})
        assert r.status_code == 200
        # force a violation
        srv.CORE.portfolio.total_pnl = -2000
        srv.CORE._tick_psychology()
        st = c.get("/api/psychology").json()
        assert st["intention"]["violated"] is True
        assert st["intention"]["violation_reason"]


def test_psychology_endpoint_returns_tilt_state(srv):
    with TestClient(srv.app) as c:
        r = c.get("/api/psychology")
        assert r.status_code == 200
        p = r.json()
        assert "tilt" in p and "intention" in p
        assert "execution_blocked" in p


def test_mind_report_endpoint(srv):
    with TestClient(srv.app) as c:
        r = c.get("/api/psychology/mind_report")
        assert r.status_code == 200
        body = r.json()
        assert "session" in body and "biases_by_type" in body
        assert "citations" in body


def test_spine_refuses_non_exit_orders_when_tilt_red(srv):
    """The behavioral wall has teeth: trip tilt past 60, then a
    research-grade EXECUTION attempt is rejected at the spine."""
    core = srv.CORE
    # push tilt past 60
    from sentinel.psychology import BiasEvent
    for _ in range(3):
        core.psychology.tilt.observe(BiasEvent(
            bias="REVENGE", severity=0.95, detected_at_ist="t",
            citation="x", evidence="y"))
    assert core.psychology.should_block_execution() is True
    # graduate a research source to TRUSTED and have it try to execute
    from sentinel.orchestration import Signal, Tier
    core.orchestrator.set_ceiling("rogue", Tier.EXECUTION)
    orders_before = len(core.account.orders_log)
    result = core.orchestrator.route(Signal(
        source="rogue", tier=Tier.EXECUTION, kind="exit",
        payload={"symbol": "X", "side": "SELL", "qty": 75},
        reason="research grade signal"))
    assert result is None
    assert len(core.account.orders_log) == orders_before


def test_spine_still_lets_hard_wired_exits_through_at_red_tilt(srv):
    """Trailing stop and profit lock are exit-only safety actors —
    they must STILL flow through, otherwise tilt could trap an
    operator inside a losing position."""
    core = srv.CORE
    core._tick_portfolio()  # populate demo positions
    from sentinel.psychology import BiasEvent
    for _ in range(3):
        core.psychology.tilt.observe(BiasEvent(
            bias="REVENGE", severity=0.95, detected_at_ist="t",
            citation="x", evidence="y"))
    assert core.psychology.should_block_execution() is True
    orders_before = len(core.account.orders_log)
    from sentinel.orchestration import Signal, Tier
    core.orchestrator.route(Signal(
        source="trailing_stop", tier=Tier.EXECUTION, kind="exit",
        payload={"symbol": "X", "side": "SELL", "qty": 75},
        reason="trail hit"))
    assert len(core.account.orders_log) > orders_before


def test_cockpit_ships_mind_panel(srv):
    with TestClient(srv.app) as c:
        html = c.get("/").text
    for marker in (
        "Mind · behavioral engine", "TiltIndex",
        "tilt_dial", "tilt_band_chip",
        "biases_active", "bias_feed",
        "intention_status",
        "renderTilt(", "setIntention(",
        "Steenbarger", "Tversky", "Shefrin",
    ):
        assert marker in html
