"""Wave 13 tests — Option premium chart + Position lifecycle viewer +
Trade Quality Scorecard + Mistake Detector.

The post-hoc audit layer: premium history per symbol, scorecard +
mistakes scanned from completed journeys, dedicated endpoints, cockpit
panels rendered.
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentinel.journey_audit import (
    Mistake, MistakeDetector, TradeQualityScorecard, grade_for,
    publish_mistakes,
)
from sentinel.live_publisher import LivePublisher
from sentinel.premium_tracker import (
    PremiumPoint, PremiumTracker, derive_metrics,
)
from sentinel.shadow_ledger import (
    Context, Hypothesis, Identity, KIND_VIRTUAL, ShadowLedger, new_event,
)


# ---------------------------------------------------------------------------
# PremiumTracker
# ---------------------------------------------------------------------------

def test_premium_tracker_records_history_and_metrics():
    pt = PremiumTracker()
    for i, prem in enumerate([180, 181, 179, 183, 185, 184, 188]):
        pt.update("NIFTY25000CE", prem, ts_ist=f"10:{i:02d}:00")
    s = pt.get("NIFTY25000CE")
    assert s is not None
    assert s.open_premium == 180.0 and s.last_premium == 188.0
    assert s.high == 188.0 and s.low == 179.0
    assert s.return_pct > 0
    assert len(s.history) == 7


def test_premium_tracker_ignores_bad_inputs():
    pt = PremiumTracker()
    pt.update("X", 0)
    pt.update("X", -5)
    pt.update("X", None)
    assert pt.get("X") is None


def test_premium_tracker_bounded_history():
    pt = PremiumTracker(max_per_symbol=4)
    for i in range(10):
        pt.update("X", 100 + i, ts_ist=f"10:00:{i:02d}")
    assert len(pt.get("X").history) == 4


def test_derive_metrics_velocity_positive_on_rising_premium():
    pts = [PremiumPoint(ts_ist=f"10:{i:02d}:00", premium=180 + i * 2)
           for i in range(8)]
    m = derive_metrics(pts)
    assert m["velocity_per_min"] > 0
    assert m["return_pct"] > 0


def test_derive_metrics_velocity_negative_on_falling():
    pts = [PremiumPoint(ts_ist=f"10:{i:02d}:00", premium=200 - i * 3)
           for i in range(8)]
    m = derive_metrics(pts)
    assert m["velocity_per_min"] < 0


def test_premium_tracker_select_default_picks_largest_qty():
    pt = PremiumTracker()
    pt.update("A", 100); pt.update("B", 50); pt.update("C", 25)
    class P:
        def __init__(self, q): self.quantity = q
    chosen = pt.select_default({"A": P(75), "B": P(150), "C": P(0)})
    assert chosen == "B"


# ---------------------------------------------------------------------------
# TradeQualityScorecard
# ---------------------------------------------------------------------------

def _completed_row(reason_codes, entry, sug_entry, sug_stop, sug_target,
                   mfe, mae, t_target=None, t_stop=None, outcome=None,
                   event_id="EV_x", instrument="NIFTY25000CE"):
    return {
        "event_id": event_id,
        "scientist": "second_pullback",
        "kind": KIND_VIRTUAL,
        "ts_utc": "2026-06-13T05:00:00Z",
        "identity": {"instrument": instrument, "premium": entry,
                      "option_type": "CE", "strike": 25000},
        "hypothesis": {"suggested_entry": sug_entry, "suggested_stop": sug_stop,
                        "suggested_target": sug_target,
                        "reason_codes": reason_codes,
                        "invalidation_conditions": []},
        "journey": {"complete": True, "mfe": mfe, "mae": mae,
                     "time_to_target_min": t_target,
                     "time_to_stop_min": t_stop,
                     "outcomes": {"t+60": outcome if outcome is not None else entry}},
        "judgment": {},
    }


def test_scorecard_grades_a_clean_winner():
    row = _completed_row(
        reason_codes=["gamma_expansion", "vwap_reclaim", "atm_pref", "low_iv", "trend_aligned"],
        entry=100.0, sug_entry=100.0, sug_stop=90.0, sug_target=130.0,
        mfe=132.0, mae=98.0, t_target=12.0, outcome=128.0)
    score = TradeQualityScorecard().score(row)
    assert score.breakdown.reason_codes == 20
    assert score.breakdown.entry_timing == 20
    assert score.breakdown.exit_quality == 20
    assert score.breakdown.r_multiple > 15
    assert score.grade in ("A", "B")
    assert score.breakdown.total >= 80


def test_scorecard_penalizes_no_reason_codes():
    row = _completed_row(
        reason_codes=[], entry=100, sug_entry=100, sug_stop=90, sug_target=130,
        mfe=110, mae=92, outcome=105)
    score = TradeQualityScorecard().score(row)
    assert score.breakdown.reason_codes == 0
    assert any("unexplained" in n for n in score.notes)


def test_scorecard_penalizes_off_entry():
    row = _completed_row(
        reason_codes=["x"], entry=120, sug_entry=100, sug_stop=90, sug_target=130,
        mfe=125, mae=118, outcome=120)
    score = TradeQualityScorecard().score(row)
    assert score.breakdown.entry_timing < 10
    assert any("off the suggested" in n for n in score.notes)


def test_scorecard_session_summary_distribution():
    rows = [
        _completed_row(["a","b","c","d","e"], 100, 100, 90, 130, 132, 98, 10, outcome=130, event_id=f"EV_{i}")
        for i in range(3)
    ] + [
        _completed_row(["a"], 100, 100, 90, 130, 105, 90, None, 8, outcome=92, event_id=f"EV_b{i}")
        for i in range(2)
    ]
    scores = TradeQualityScorecard().score_session(rows)
    summary = TradeQualityScorecard().session_summary(scores)
    assert summary["count"] == 5
    assert "average" in summary and summary["average"] > 0
    assert sum(summary["grade_distribution"].values()) == 5


def test_grade_for_boundaries():
    assert grade_for(86) == "A"
    assert grade_for(85) == "A"
    assert grade_for(70) == "B"
    assert grade_for(55) == "C"
    assert grade_for(40) == "D"
    assert grade_for(0)  == "F"


# ---------------------------------------------------------------------------
# MistakeDetector
# ---------------------------------------------------------------------------

def test_mistake_detector_finds_gave_back_peak():
    """Peak was +30, exit gave back to +2 → gave-back-peak."""
    row = _completed_row(
        reason_codes=["x"], entry=100, sug_entry=100, sug_stop=90, sug_target=140,
        mfe=130, mae=99, outcome=102)
    mistakes = MistakeDetector().scan_row(row)
    assert any(m.pattern == "GAVE_BACK_PEAK" for m in mistakes)


def test_mistake_detector_finds_target_too_far():
    """MFE 115 never reached 140 target → target-too-far."""
    row = _completed_row(
        reason_codes=["x"], entry=100, sug_entry=100, sug_stop=90, sug_target=140,
        mfe=115, mae=95, outcome=108)
    mistakes = MistakeDetector().scan_row(row)
    assert any(m.pattern == "TARGET_TOO_FAR" for m in mistakes)


def test_mistake_detector_finds_stop_too_tight():
    """Stopped at 90 (MAE 89), but move ran to 135 past 120 target → stop-too-tight."""
    row = _completed_row(
        reason_codes=["x"], entry=100, sug_entry=100, sug_stop=90, sug_target=120,
        mfe=135, mae=89, t_stop=5, outcome=110)
    mistakes = MistakeDetector().scan_row(row)
    assert any(m.pattern == "STOP_TOO_TIGHT" for m in mistakes)


def test_mistake_detector_finds_no_trail_on_winner():
    """MFE - entry = +30 * 75 = ₹2250 > threshold, no target/stop hit → no trail."""
    row = _completed_row(
        reason_codes=["x"], entry=100, sug_entry=100, sug_stop=80, sug_target=200,
        mfe=130, mae=99, outcome=108)
    mistakes = MistakeDetector().scan_row(row)
    assert any(m.pattern == "NO_TRAIL_ON_WINNER" for m in mistakes)


def test_mistake_detector_finds_ignored_invalidation():
    row = _completed_row(
        reason_codes=["x"], entry=100, sug_entry=100, sug_stop=90, sug_target=130,
        mfe=110, mae=80, outcome=85)
    row["hypothesis"]["invalidation_conditions"] = ["spot below 25100"]
    mistakes = MistakeDetector().scan_row(row)
    assert any(m.pattern == "IGNORED_INVALIDATION" for m in mistakes)


def test_mistake_detector_silent_on_clean_winner():
    row = _completed_row(
        reason_codes=["x"], entry=100, sug_entry=100, sug_stop=90, sug_target=130,
        mfe=132, mae=98, t_target=10, outcome=128)
    mistakes = MistakeDetector().scan_row(row)
    assert all(m.pattern != "GAVE_BACK_PEAK" for m in mistakes)
    assert all(m.pattern != "STOP_TOO_TIGHT" for m in mistakes)
    assert all(m.pattern != "IGNORED_INVALIDATION" for m in mistakes)


def test_publish_mistakes_streams_through_bus():
    pub = LivePublisher()
    row = _completed_row(
        reason_codes=["x"], entry=100, sug_entry=100, sug_stop=90, sug_target=140,
        mfe=115, mae=95, outcome=108)
    mistakes = MistakeDetector().scan_row(row)
    n = publish_mistakes(mistakes, pub)
    assert n >= 1
    # bus contains the trade_mistake key under the right asset
    cur = pub.current()
    assert any("trade_mistake" in k for k in cur.keys())


# ---------------------------------------------------------------------------
# Server integration — endpoints + cockpit markup
# ---------------------------------------------------------------------------

@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    import sentinel.server as server
    importlib.reload(server)
    return server


def _seed_completed_session(journal_dir: Path, session: str, n=3):
    led = ShadowLedger(journal_dir)
    for i in range(n):
        ident = Identity(instrument=f"NIFTY{25000+i*50}CE", option_type="CE",
                          strike=25000+i*50, expiry=None,
                          moneyness_key="NIFTY:CE:+0",
                          underlying_price=25000, premium=100, delta=0.5)
        hyp = Hypothesis(expected_underlying_move=30, expected_premium=130,
                          expected_horizon_min=15, suggested_entry=100,
                          suggested_stop=90, suggested_target=130,
                          confidence=0.6, reason_codes=["test", "vwap"])
        ev = new_event(KIND_VIRTUAL, session, ident, Context(), hyp,
                        scientist="test_sci")
        ShadowLedger.stamp_journey(ev, 5.0, 130)
        ShadowLedger.stamp_journey(ev, 12.0, 132)
        ShadowLedger.stamp_journey(ev, 60.0, 108)
        led.write(ev)


def test_premium_chart_endpoint_and_state(srv):
    core = srv.CORE
    core._tick_portfolio()
    # feed a few premium ticks
    for sym in core.portfolio.positions:
        for prem in [180.0, 181.5, 183.0, 185.0]:
            core.premium_tracker.update(sym, prem)
        break
    with TestClient(srv.app) as c:
        s = c.get("/api/state").json()
        assert "premium_chart" in s
        chart = c.get("/api/premium_chart").json()
        assert "series" in chart and chart["series"] is not None
        # explicit pin
        sym = list(core.portfolio.positions)[0]
        pin = c.post("/api/premium_chart",
                     json={"symbol": sym}).json()
        assert pin["symbol"] == sym


def test_premium_chart_endpoint_handles_unknown_symbol(srv):
    with TestClient(srv.app) as c:
        r = c.get("/api/premium_chart?symbol=DOES_NOT_EXIST")
        assert r.status_code == 200
        assert r.json()["series"] is None


def test_journeys_endpoint_scores_completed_rows(srv, tmp_path):
    _seed_completed_session(tmp_path, srv.CORE.session, n=3)
    with TestClient(srv.app) as c:
        r = c.get("/api/journeys").json()
        assert r["items"]
        scored = [it for it in r["items"] if "score" in it]
        assert scored
        assert "total" in scored[0]["score"]["breakdown"]


def test_scorecard_endpoint_returns_summary(srv, tmp_path):
    _seed_completed_session(tmp_path, srv.CORE.session, n=4)
    with TestClient(srv.app) as c:
        r = c.get("/api/scorecard").json()
        assert r["summary"]["count"] >= 4
        assert "average" in r["summary"]
        assert "grade_distribution" in r["summary"]


def test_mistakes_endpoint_returns_detected(srv, tmp_path):
    """Seeded journeys have MFE just hitting target — they should pass
    most checks, but some patterns may still fire depending on R; just
    confirm the endpoint shape."""
    _seed_completed_session(tmp_path, srv.CORE.session, n=2)
    with TestClient(srv.app) as c:
        r = c.get("/api/mistakes").json()
        assert "mistakes" in r and isinstance(r["mistakes"], list)


def test_tick_journey_audit_publishes_signals(srv, tmp_path):
    """The 45-second audit pump scans the ledger and publishes new
    mistake signals exactly once per event_id."""
    _seed_completed_session(tmp_path, srv.CORE.session, n=2)
    core = srv.CORE
    # Override the seeded outcomes so a mistake actually fires
    # (the seeded data lands on target; force a gave-back-peak).
    from sentinel.shadow_ledger import read_session
    rows = read_session(tmp_path, core.session)
    assert rows
    # rewrite the outcome to give back the move so GAVE_BACK_PEAK fires
    for r in rows:
        r["journey"]["mfe"] = 140
        r["journey"]["outcomes"]["t+60"] = 102
        r["hypothesis"]["suggested_target"] = 200
        # write back to the JSONL
        led = ShadowLedger(tmp_path)
        ev_path = tmp_path / f"ledger_{core.session}.jsonl"
        ev_path.write_text("\n".join(__import__("json").dumps(r) for r in rows))
        break
    core._tick_journey_audit()
    cur = core.publisher.current()
    assert any("trade_mistake" in k for k in cur.keys())
    # second call must NOT publish duplicates for the same event_ids
    before = len(core.publisher.history())
    core._tick_journey_audit()
    after = len(core.publisher.history())
    assert after == before


def test_cockpit_ships_premium_journey_panels(srv):
    with TestClient(srv.app) as c:
        html = c.get("/").text
    for marker in (
        "Option premium · live", "prem_chart", "prem_tip",
        "drawPremiumChart(", "setPremiumSymbol(",
        "Journeys", "scorecard", "score_summary", "journey_list",
        "loadJourneys(", "renderJourneyList(", "renderScoreSummary(",
        "GRADE",
    ):
        assert marker in html
