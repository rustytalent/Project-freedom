"""Wave 5 tests — the Orchestrator wired into the server hot path.

The safety invariant made runtime-real: EVERY order placement now routes
through the trust-tier spine, and ONLY the two hard-wired EXECUTION
actors (trailing_stop, profit_lock) ever reach the order path. A
research source can request EXECUTION and the spine clamps it to SHADOW
— no order, no matter how confident.

These tests drive the live loop deterministically (demo mode) and assert
on both the placed orders (DemoAccount.orders_log) and the spine's own
routing stats.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from sentinel.orchestration import Signal, Tier


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    import sentinel.server as server
    importlib.reload(server)
    return server


# ---------------------------------------------------------------------------
# EXECUTION path: trailing stop fires THROUGH the spine
# ---------------------------------------------------------------------------

def test_trail_exit_routes_through_orchestrator_and_places_order(srv):
    core = srv.CORE
    # Arm a per-position trail, then drive ticks that give back past cushion.
    core.trails.arm("OPT", "long", 75, 30.0, 200.0)
    for p in (220, 250, 240, 219):       # gave back 31 from peak 250
        core.trails.on_tick("OPT", p)
    # An order was actually placed via the demo account.
    assert len(core.account.orders_log) == 1
    assert core.account.orders_log[0]["tradingsymbol"] == "OPT"
    # The spine saw an EXECUTION-tier exit from trailing_stop.
    assert core.orchestrator.stats()["routed"]["EXECUTION"] >= 1
    assert any(s["source"] == "trailing_stop" and s["tier"] == "EXECUTION"
               for s in core.routed_signals)


def test_profit_lock_fire_routes_through_orchestrator(srv):
    core = srv.CORE
    core._tick_portfolio()               # populate positions from demo
    assert core.portfolio.positions
    from sentinel.profit_lock import ProfitLock
    core.profit_lock = ProfitLock(mode="buffer", give_back_buffer=500.0,
                                  activation_floor=100.0)
    core.profit_lock.update(5000.0)      # establish a peak of 5000
    orders_before = len(core.account.orders_log)
    # Freeze the portfolio so refresh() can't overwrite our injected P&L,
    # then inject a give-back of 1000 (> 500 buffer) -> the lock must fire.
    core.portfolio.refresh = lambda *a, **k: None
    core.portfolio.total_pnl = 4000.0
    core._tick_portfolio()
    # profit_lock fired -> EXECUTION exit(s) routed from profit_lock, and an
    # order placed for each open position.
    assert any(s["source"] == "profit_lock" for s in core.routed_signals)
    assert len(core.account.orders_log) > orders_before


# ---------------------------------------------------------------------------
# The wall: an ungraduated source is CLAMPED, never executes
# ---------------------------------------------------------------------------

def test_ungraduated_source_requesting_execution_is_clamped(srv):
    core = srv.CORE
    orders_before = len(core.account.orders_log)
    clamped_before = core.orchestrator.clamped
    # A rogue research source asks for EXECUTION on a real symbol.
    result = core.orchestrator.route(Signal(
        source="rogue_scientist", tier=Tier.EXECUTION, kind="exit",
        payload={"symbol": "OPT", "side": "SELL", "qty": 75},
        reason="I am very confident"))
    # No order placed; the spine clamped it to SHADOW.
    assert result is None
    assert len(core.account.orders_log) == orders_before
    assert core.orchestrator.clamped == clamped_before + 1


def test_graduated_trusted_source_surfaces_but_never_executes(srv):
    core = srv.CORE
    orders_before = len(core.account.orders_log)
    core.orchestrator.set_ceiling("my_scientist", Tier.TRUSTED)
    # Even after graduating to TRUSTED, an EXECUTION request is clamped to
    # TRUSTED (surfaces, no order).
    core.orchestrator.route(Signal(
        source="my_scientist", tier=Tier.EXECUTION, kind="exit",
        payload={"symbol": "OPT", "side": "SELL", "qty": 75},
        reason="confident scientist"))
    assert len(core.account.orders_log) == orders_before    # no order
    # It surfaced (TRUSTED reaches the surface sink).
    assert any(s["source"] == "my_scientist" for s in core.surfaced_signals)


# ---------------------------------------------------------------------------
# Kill switch closes the execution gate at the spine level
# ---------------------------------------------------------------------------

def test_killswitch_blocks_execution_at_the_spine(srv):
    core = srv.CORE
    with TestClient(srv.app) as client:
        r = client.post("/api/killswitch")
        assert r.json()["killed"] is True
    assert core.orchestrator.allow_execution is False
    orders_before = len(core.account.orders_log)
    # Even the hard-wired actor cannot place an order now.
    core.orchestrator.route(Signal(
        source="trailing_stop", tier=Tier.EXECUTION, kind="exit",
        payload={"symbol": "OPT", "side": "SELL", "qty": 75},
        reason="trail hit after kill"))
    assert len(core.account.orders_log) == orders_before


# ---------------------------------------------------------------------------
# Graduate endpoint: the curator's lever, EXECUTION reserved
# ---------------------------------------------------------------------------

def test_graduate_endpoint_sets_ceiling(srv):
    with TestClient(srv.app) as client:
        r = client.post("/api/graduate",
                        json={"source": "second_pullback", "tier": "TRUSTED"})
        assert r.status_code == 200
        assert r.json()["ceiling"] == "TRUSTED"
        assert (r.json()["stats"]["ceilings"]["second_pullback"] == "TRUSTED")


def test_graduate_endpoint_refuses_execution(srv):
    with TestClient(srv.app) as client:
        r = client.post("/api/graduate",
                        json={"source": "second_pullback", "tier": "EXECUTION"})
        assert r.status_code == 400
        assert "reserved" in r.json()["detail"].lower()


def test_graduate_endpoint_rejects_bad_tier(srv):
    with TestClient(srv.app) as client:
        r = client.post("/api/graduate",
                        json={"source": "x", "tier": "SUPERUSER"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /api/state exposes the spine
# ---------------------------------------------------------------------------

def test_state_exposes_orchestrator_stats(srv):
    core = srv.CORE
    core._tick_portfolio()
    core._tick_recommendations()
    with TestClient(srv.app) as client:
        s = client.get("/api/state").json()
        assert "orchestrator" in s
        assert "routed" in s["orchestrator"]
        assert "routed_signals" in s
        # maximizer is graduated to TRUSTED at startup
        assert s["orchestrator"]["ceilings"].get("maximizer") == "TRUSTED"


def test_maximizer_and_recommender_are_graduated_at_startup(srv):
    core = srv.CORE
    ceilings = core.orchestrator.stats()["ceilings"]
    assert ceilings.get("maximizer") == "TRUSTED"
    assert ceilings.get("dip_recommender") == "TRUSTED"
