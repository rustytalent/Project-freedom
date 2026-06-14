"""Wave 11 tests — NIFTY constituent board (the founder's 'is this move
real?' panel).

Pins:
  * move_quality() verdict classifier
  * ConstituentBoard streaming state, snapshot shape, ledger mirror
  * DemoFeed produces movement
  * /api/state and /api/equity_board both expose the board
  * cockpit ships the panel + sparkline renderer
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from sentinel.equity_layer import (
    NIFTY_TOP10_WEIGHTS, move_quality, top10_contextual_layer,
)
from sentinel.live_equity import (
    TOP10, ConstituentBoard, DemoFeed, _DEMO_BOOK,
)
from sentinel.live_publisher import LivePublisher
from sentinel.shadow_ledger import ShadowLedger, read_session


# ---------------------------------------------------------------------------
# move_quality verdicts
# ---------------------------------------------------------------------------

def _ctx(returns_pct, idx):
    return top10_contextual_layer(returns_pct, idx)


def test_move_quality_strong_when_breadth_and_heavyweights_align():
    """Every name +1% with index +0.9 → STRONG, conf high."""
    returns = {s: 1.0 for s in NIFTY_TOP10_WEIGHTS}
    q = move_quality(_ctx(returns, 0.9))
    assert q["verdict"] == "STRONG"
    assert q["confidence"] >= 0.8
    assert any("breadth_confirms" in c for c in q["reason_codes"])


def test_move_quality_manipulated_when_concentrated():
    """Two heavyweights doing all the work, others flat → MANIPULATED."""
    returns = {s: 0.0 for s in NIFTY_TOP10_WEIGHTS}
    returns["RELIANCE"] = 3.0
    returns["HDFCBANK"] = 2.5
    q = move_quality(_ctx(returns, 0.55))
    assert q["verdict"] == "MANIPULATED"
    assert q["metrics"]["top2_share_of_move"] > 0.55
    assert "RELIANCE" in q["headline"] or "HDFCBANK" in q["headline"]


def test_move_quality_rotation_when_dispersed_but_flat():
    """Big gainers + big losers cancelling → ROTATION."""
    returns = {s: 0.0 for s in NIFTY_TOP10_WEIGHTS}
    returns["RELIANCE"] = 2.5
    returns["INFY"] = 2.0
    returns["HDFCBANK"] = -2.5
    returns["ICICIBANK"] = -2.0
    q = move_quality(_ctx(returns, 0.05))
    assert q["verdict"] == "ROTATION"


def test_move_quality_consolidation_when_everything_quiet():
    returns = {s: 0.05 for s in NIFTY_TOP10_WEIGHTS}
    q = move_quality(_ctx(returns, 0.04))
    assert q["verdict"] == "CONSOLIDATION"


def test_move_quality_fragile_when_breadth_weak():
    """Index up but only 4 names up, narrow leadership → FRAGILE."""
    returns = {s: -0.1 for s in NIFTY_TOP10_WEIGHTS}
    for s in ["RELIANCE", "HDFCBANK", "INFY", "TCS"]:
        returns[s] = 0.8
    q = move_quality(_ctx(returns, 0.25))
    assert q["verdict"] == "FRAGILE"
    assert "narrow_participation" in q["reason_codes"] or "unaligned_heavyweights" in q["reason_codes"]


# ---------------------------------------------------------------------------
# ConstituentBoard streaming
# ---------------------------------------------------------------------------

def test_board_tick_populates_per_stock_state():
    board = ConstituentBoard()
    open_quotes = {s: p.base for s, p in _DEMO_BOOK.items()}
    board.tick(open_quotes)
    moved = {s: p.base * 1.005 for s, p in _DEMO_BOOK.items()}    # +0.5%
    snap = board.tick(moved)
    assert snap["stocks"]                         # all 10 populated
    for row in snap["stocks"]:
        assert row["symbol"] in TOP10
        assert abs(row["return_pct"] - 0.5) < 0.01
        assert row["history"]                     # sparkline data captured
    # weight-adjusted contribution sums close to the average return
    assert abs(snap["top10_contribution_pct"] - 0.5 * sum(NIFTY_TOP10_WEIGHTS.values())) < 0.05
    assert "verdict" in snap["quality"]


def test_board_publishes_one_signal_per_cycle_via_bus(tmp_path):
    sl = ShadowLedger(tmp_path)
    pub = LivePublisher(shadow_ledger=sl, session="2026-06-13")
    board = ConstituentBoard(publisher=pub)
    quotes = {s: p.base for s, p in _DEMO_BOOK.items()}
    board.tick(quotes)
    moved = {s: p.base * 1.01 for s, p in _DEMO_BOOK.items()}
    board.tick(moved)
    # constituent_board key present on the bus, TRUSTED tier
    cur = pub.current()
    assert "NIFTY|constituent_board" in cur
    sig = cur["NIFTY|constituent_board"]
    assert sig.trust_tier == "TRUSTED"
    assert sig.confidence > 0
    # TRUSTED + conf >= 0.5 → mirrored to the canonical substrate
    rows = read_session(tmp_path, "2026-06-13")
    assert rows and rows[0]["scientist"] == "constituent_board"


def test_board_does_not_publish_with_empty_quotes():
    pub = LivePublisher()
    board = ConstituentBoard(publisher=pub)
    board.tick({})                                # nothing populates
    assert "NIFTY|constituent_board" not in pub.current()


def test_demo_feed_walks_prices_deterministically():
    f1 = DemoFeed(seed=42); f2 = DemoFeed(seed=42)
    a = [f1.next() for _ in range(5)]
    b = [f2.next() for _ in range(5)]
    assert a == b                                 # reproducible
    # later ticks differ from open
    open_prices = {s: p.base for s, p in _DEMO_BOOK.items()}
    later = a[-1]
    assert any(abs(later[s] - open_prices[s]) > 1e-6 for s in open_prices)


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


def test_state_carries_constituent_board(srv):
    """A few _tick_board cycles populate the board, /api/state surfaces it."""
    core = srv.CORE
    for _ in range(3):
        core._tick_board()
    with TestClient(srv.app) as c:
        s = c.get("/api/state").json()
        board = s["constituent_board"]
        assert board["stocks"]
        assert "verdict" in board["quality"]
        assert {row["symbol"] for row in board["stocks"]} == set(TOP10)


def test_equity_board_endpoint(srv):
    """GET /api/equity_board returns the live snapshot."""
    core = srv.CORE
    core._tick_board(); core._tick_board()
    with TestClient(srv.app) as c:
        r = c.get("/api/equity_board")
        assert r.status_code == 200
        board = r.json()
        assert "regime" in board and "quality" in board


def test_cockpit_ships_board_panel(srv):
    """Markup contract: panel title, table, verdict slot, JS hooks."""
    with TestClient(srv.app) as c:
        html = c.get("/").text
    for marker in (
        "Constituent board",
        "board_meta", "board_verdict", "board_breadth", "board_table",
        "renderBoard(", "sparkSvg(", "MOVE QUALITY",
        "STRONG", "FRAGILE", "MANIPULATED", "ROTATION", "CONSOLIDATION",
    ):
        assert marker in html
