"""Wave 17 launch-prep tests — auth plug, health endpoints, preflight gate,
source-tagged overlays.

These pin the contract Codex's Supabase + Google Auth side plugs into:
  * register_verifier(fn) is the ONE integration point
  * dev tokens keep local mode working without Supabase
  * /healthz returns 200 always, /readyz reports degraded
  * /api/preflight gates Crux TRADE verdicts behind operator commitment
  * model_zones payload carries `source` for L/S labelling on overlays
"""
from __future__ import annotations

import base64
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from sentinel.auth import (
    AuthContext, make_dev_token, register_verifier, reset_verifier,
    verify_token,
)


# ---------------------------------------------------------------------------
# Auth contract
# ---------------------------------------------------------------------------

def test_verify_token_returns_none_for_empty():
    reset_verifier()
    assert verify_token("") is None
    assert verify_token(None) is None


def test_dev_token_round_trip():
    reset_verifier()
    token = make_dev_token(plan="PRO", user_id="u_42", email="x@y")
    ctx = verify_token(token)
    assert ctx is not None
    assert ctx.user_id == "u_42"
    assert ctx.plan == "PRO"
    assert ctx.email == "x@y"
    assert ctx.is_demo is True


def test_register_verifier_overrides_default():
    """Codex's production verifier wins over the dev fallback."""
    reset_verifier()
    called = {}
    def fake(token):
        called["token"] = token
        return AuthContext(user_id="prod-user", plan="QUANT")
    register_verifier(fake)
    try:
        ctx = verify_token("any-jwt")
        assert ctx.user_id == "prod-user"
        assert ctx.plan == "QUANT"
        assert called["token"] == "any-jwt"
    finally:
        reset_verifier()


def test_register_verifier_failure_is_handled():
    """A broken verifier returns None rather than crashing the endpoint."""
    reset_verifier()
    def boom(_token):
        raise RuntimeError("network down")
    register_verifier(boom)
    try:
        assert verify_token("anything") is None
    finally:
        reset_verifier()


def test_special_demo_token_returns_founder_ctx():
    reset_verifier()
    ctx = verify_token("demo")
    assert ctx is not None
    assert ctx.plan == "FOUNDER"
    assert ctx.is_demo is True
    assert ctx.is_founder is True


# ---------------------------------------------------------------------------
# Server integration — health + preflight + overlays
# ---------------------------------------------------------------------------

@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    import sentinel.server as server
    importlib.reload(server)
    return server


def test_healthz_always_returns_200(srv):
    with TestClient(srv.app) as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_readyz_reports_ready_in_demo_mode(srv):
    with TestClient(srv.app) as c:
        r = c.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["mode"] == "demo"
        assert body["session"]
        assert body["uptime_seconds"] >= 0


def test_preflight_unacknowledged_by_default(srv):
    with TestClient(srv.app) as c:
        r = c.get("/api/preflight", headers={"X-Sentinel-Plan": "FOUNDER"})
        body = r.json()
        assert body["acknowledged"] is False
        assert body["ack"] is None


def test_preflight_ack_records_state(srv):
    with TestClient(srv.app) as c:
        r = c.post("/api/preflight/ack",
                    headers={"X-Sentinel-Plan": "FOUNDER"},
                    json={"accepted_max_loss_rupees": 2500,
                           "expected_regime": "chop",
                           "notes": "RBI policy today"})
        assert r.status_code == 200
        ack = r.json()["ack"]
        assert ack["accepted_max_loss_rupees"] == 2500
        assert ack["expected_regime"] == "chop"
        # state surfaces it
        r2 = c.get("/api/preflight", headers={"X-Sentinel-Plan": "FOUNDER"})
        assert r2.json()["acknowledged"] is True


def test_preflight_zero_max_loss_rejected(srv):
    with TestClient(srv.app) as c:
        r = c.post("/api/preflight/ack",
                    headers={"X-Sentinel-Plan": "FOUNDER"},
                    json={"accepted_max_loss_rupees": 0})
        assert r.status_code == 400


def test_preflight_ack_also_sets_intention_contract(srv):
    """Committing the preflight should also seed the Ulysses-pattern
    intention contract so the psychology engine enforces the budget."""
    core = srv.CORE
    assert core.psychology.intention.is_set is False
    with TestClient(srv.app) as c:
        c.post("/api/preflight/ack",
                headers={"X-Sentinel-Plan": "FOUNDER"},
                json={"accepted_max_loss_rupees": 3000})
    assert core.psychology.intention.is_set is True
    assert core.psychology.intention.max_day_loss_rupees == 3000


def test_state_carries_preflight_block(srv):
    with TestClient(srv.app) as c:
        s = c.get("/api/state").json()
        assert "preflight" in s
        assert "acknowledged" in s["preflight"]


def test_crux_trade_downgraded_to_watch_without_preflight(srv):
    """Launch safety gate: until the operator commits, Crux can NOT
    surface a TRADE verdict. It downgrades to WATCH with a clear
    'preflight_not_acknowledged' contradiction."""
    from sentinel.live_publisher import ModelSignal, now_ist_hms
    core = srv.CORE
    core.preflight_ack = None
    # seed the bus with strong bull consensus so compose returns TRADE
    for tag in ("reaction_model", "proximity_model"):
        core.publisher.publish(ModelSignal(
            ts_ist=now_ist_hms(), asset="NIFTY", model=tag,
            signal="upside reaction in progress", confidence=0.85,
            trust_tier="LOGGED", reason_codes=["upside_reaction"]))
    core._tick_crux()
    # The preflight gate downgrades the verdict to WATCH and tags the
    # reason on the contradicting field so the cockpit can show it.
    assert core.crux_verdict.verdict == "WATCH"
    assert "preflight" in (core.crux_verdict.contradicting or "").lower()


def test_crux_trade_allowed_after_preflight(srv):
    core = srv.CORE
    with TestClient(srv.app) as c:
        c.post("/api/preflight/ack",
                headers={"X-Sentinel-Plan": "FOUNDER"},
                json={"accepted_max_loss_rupees": 3000})
    # publish a strong bull consensus so compose returns TRADE
    from sentinel.live_publisher import ModelSignal, now_ist_hms
    for tag in ("reaction_model", "proximity_model"):
        core.publisher.publish(ModelSignal(
            ts_ist=now_ist_hms(), asset="NIFTY", model=tag,
            signal="upside reaction in progress", confidence=0.78,
            trust_tier="LOGGED",
            reason_codes=["upside_reaction"]))
    core._tick_crux()
    # not asserting verdict=='TRADE' specifically (depends on AVOID census)
    # only that the preflight gate doesn't suppress it any more
    assert core.crux_verdict.contradicting != "preflight_not_acknowledged"


def test_model_zones_payload_carries_source(srv):
    """Source field lets the cockpit label [L] vs [S] on overlays."""
    core = srv.CORE
    from sentinel.live_publisher import ModelSignal, now_ist_hms
    # one from sentinel side, one from liqpool side
    core.publisher.publish(ModelSignal(
        ts_ist=now_ist_hms(), asset="NIFTY", model="proximity_model",
        signal="near low", confidence=0.7, trust_tier="LOGGED",
        zone=(25000, 25050), extras={}))
    core.publisher.publish(ModelSignal(
        ts_ist=now_ist_hms(), asset="NIFTY", model="direction_model",
        signal="upside 70%", confidence=0.7, trust_tier="LOGGED",
        zone=(25100, 25200), extras={"source": "liqpool"}))
    with TestClient(srv.app) as c:
        zones = c.get("/api/state").json()["model_zones"]
    found = {z["model"]: z["source"] for z in zones}
    assert found.get("proximity_model") in ("sentinel", None)
    assert found.get("direction_model") == "liqpool"
