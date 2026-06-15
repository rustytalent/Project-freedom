"""Wave 21 — Indian-market differentiation + help + admin endpoints.

Pins:
  * Tax-impact math itemizes STT/GST/STCG/brokerage/SEBI/stamp correctly
  * STT applied only on sell side; CGT only on gains
  * Holiday + Muhurat + expiry-day-shift logic
  * /api/help carries the panel + citation dicts
  * /api/version + /api/tax/* + /api/session_context + /api/admin/plan_change
"""
from __future__ import annotations

import importlib
from datetime import date

import pytest
from fastapi.testclient import TestClient

from sentinel.india_tax import (
    DEFAULT_BROKERAGE_RUPEES, STT_OPTIONS_SELL_PCT, NSE_HOLIDAYS_2026,
    is_nse_holiday, is_nse_open, is_weekend, next_trading_day,
    options_trade_tax_impact, session_context,
)


# ─────────────────────────────────────────────────────────────────
# Tax math
# ─────────────────────────────────────────────────────────────────

def test_tax_breakdown_itemizes_costs():
    br = options_trade_tax_impact(buy_premium=100.0, sell_premium=130.0,
                                   lots=1, lot_size=75)
    assert br.gross_pnl == 30 * 75
    assert br.brokerage == 40.0      # ₹20 × 2 legs
    assert br.stt > 0                 # SELL-side STT
    assert br.gst > 0                 # 18% on brokerage + exchange + SEBI
    assert br.capital_gains_tax > 0   # gross was positive
    assert br.net_pnl < br.gross_pnl
    # net = gross - total_charges
    assert abs(br.net_pnl - (br.gross_pnl - br.total_charges)) < 0.01


def test_tax_no_capital_gains_on_loss():
    """STCG only applies on positive gross P&L."""
    br = options_trade_tax_impact(buy_premium=130.0, sell_premium=100.0,
                                   lots=1, lot_size=75)
    assert br.gross_pnl < 0
    assert br.capital_gains_tax == 0


def test_tax_apply_capital_gains_off_for_business_income():
    """An operator filing under 44AD doesn't pay CGT at trade time."""
    br = options_trade_tax_impact(buy_premium=100.0, sell_premium=130.0,
                                   lots=1, lot_size=75,
                                   apply_capital_gains=False)
    assert br.capital_gains_tax == 0


def test_tax_stt_proportional_to_sell_turnover():
    """STT is 0.0625% of SELL premium × quantity."""
    br = options_trade_tax_impact(buy_premium=100.0, sell_premium=200.0,
                                   lots=1, lot_size=75)
    expected = 200.0 * 75 * STT_OPTIONS_SELL_PCT
    assert abs(br.stt - round(expected, 2)) < 0.01


def test_tax_zero_quantity_falls_back_to_lots_x_lotsize():
    br = options_trade_tax_impact(buy_premium=100.0, sell_premium=110.0,
                                   quantity=0, lots=2, lot_size=75)
    # 2 lots × 75 = 150 units
    assert br.gross_pnl == 10 * 150


# ─────────────────────────────────────────────────────────────────
# Calendar
# ─────────────────────────────────────────────────────────────────

def test_known_holidays_recognised():
    """Hand-picked 2026 NSE holidays."""
    assert is_nse_holiday(date(2026, 1, 26)) == "Republic Day"
    assert is_nse_holiday(date(2026, 10, 20)).startswith("Diwali")


def test_weekend_is_not_open():
    saturday = date(2026, 6, 13)
    assert is_weekend(saturday) is True
    assert is_nse_open(saturday) is False


def test_next_trading_day_skips_holiday_and_weekend():
    """26 Jan 2026 (Mon, Republic Day) → next is Tue 27 Jan."""
    assert next_trading_day(date(2026, 1, 26)) == date(2026, 1, 27)


def test_session_context_today_shape():
    ctx = session_context()
    for key in ("today_iso", "today_open", "next_trading_day",
                 "expiry_today", "upcoming_holidays"):
        assert key in ctx


def test_session_context_muhurat_flag():
    """Diwali 2026-10-20 carries the Muhurat session payload."""
    ctx = session_context(date(2026, 10, 20))
    assert ctx["muhurat_session"] is not None
    assert "session_open_ist" in ctx["muhurat_session"]


def test_session_context_thursday_expiry():
    """A random Thursday (not a holiday) is an expiry day."""
    # 2026-06-04 is a Thursday and not on the holiday list
    ctx = session_context(date(2026, 6, 4))
    assert ctx["expiry_today"] is True


def test_session_context_wednesday_shifts_when_thursday_holiday():
    """If Thursday is a holiday, Wednesday becomes the expiry day."""
    # Place a fake holiday on a Thursday for the test
    holiday_thur = date(2026, 4, 2)        # Thursday
    NSE_HOLIDAYS_2026[holiday_thur.isoformat()] = "test_holiday"
    try:
        ctx = session_context(date(2026, 4, 1))   # Wednesday
        assert ctx["expiry_today"] is True
    finally:
        del NSE_HOLIDAYS_2026[holiday_thur.isoformat()]


def test_session_context_upcoming_holidays_window():
    ctx = session_context(date(2026, 1, 25))
    labels = [h["label"] for h in ctx["upcoming_holidays"]]
    assert "Republic Day" in labels


# ─────────────────────────────────────────────────────────────────
# Help payload
# ─────────────────────────────────────────────────────────────────

def test_help_payload_has_panel_and_citations():
    from sentinel.help_text import help_payload
    p = help_payload()
    assert "panels" in p and "citations" in p
    # Some core panels we promise to document
    for panel in ("spot_chart", "crux", "mind", "intention",
                   "profit_lock", "trust_spine", "tax_breakdown",
                   "session_context"):
        assert panel in p["panels"]
        assert p["panels"][panel]["title"]
        assert p["panels"][panel]["what"]
    # Citations point to real refs we cite throughout the cockpit
    for cite in ("tharp_2007", "steenbarger_2009",
                  "tversky_kahneman_1974", "elster_2000", "hull_2017",
                  "boyle_1977", "gatheral_2004"):
        assert cite in p["citations"]


# ─────────────────────────────────────────────────────────────────
# Server endpoints
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    from sentinel.byok import GLOBAL_STORE
    from sentinel.rate_limit import reset_all
    GLOBAL_STORE._blobs.clear(); reset_all()
    import sentinel.server as server
    importlib.reload(server)
    return server


def test_tax_endpoint_returns_breakdown(srv):
    with TestClient(srv.app) as c:
        r = c.post("/api/tax/options_trade",
                    json={"buy_premium": 100, "sell_premium": 130,
                           "lots": 1, "lot_size": 75})
        assert r.status_code == 200
        body = r.json()
        assert body["gross_pnl"] == 30 * 75
        assert body["net_pnl"] < body["gross_pnl"]


def test_session_context_endpoint(srv):
    with TestClient(srv.app) as c:
        r = c.get("/api/session_context")
        assert r.status_code == 200
        body = r.json()
        assert "today_open" in body


def test_help_endpoint(srv):
    with TestClient(srv.app) as c:
        r = c.get("/api/help")
        body = r.json()
        assert "panels" in body
        assert body["panels"]["spot_chart"]["title"]


def test_version_endpoint(srv):
    with TestClient(srv.app) as c:
        body = c.get("/api/version").json()
        assert body["name"] == "sentinel"
        assert "wave" in body
        assert body["modules_self_declared"] >= 30


def test_admin_plan_change_records_audit(srv):
    with TestClient(srv.app) as c:
        r = c.post("/api/admin/plan_change",
                    json={"user_id": "u-42",
                           "new_plan": "PRO",
                           "stripe_event_id": "evt_123",
                           "stripe_event_type": "customer.subscription.updated"})
        assert r.status_code == 200
        assert r.json()["new_plan"] == "PRO"
    # audit trail recorded the change
    audit = (srv.CORE.cfg.journal_dir / "audit.jsonl").read_text()
    assert "plan_change" in audit
    assert "u-42" in audit
