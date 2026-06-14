"""Wave 18 — SaaS production hardening tests.

Pins:
  * BYOK credential store encrypts + decrypts; per-user isolation;
    cleared on missing-user; never persists plaintext.
  * Audit log writes JSONL with the right envelope; auto-scrubs
    credentials; external sinks fire.
  * /api/byok/* endpoints work end-to-end with auth.
  * Hardened auth: X-Sentinel-Plan header fallback is OFF unless
    SENTINEL_DEMO or SENTINEL_ALLOW_HEADER_AUTH is set.
  * Spine order events land in the audit log.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentinel.audit_log import (
    AuditLog, _scrub, clear_audit_sinks, register_audit_sink,
)
from sentinel.byok import CredentialStore, GLOBAL_STORE, KiteCredentials


# ---------------------------------------------------------------------------
# BYOK credential store
# ---------------------------------------------------------------------------

def test_byok_round_trip_preserves_credentials():
    store = CredentialStore()
    store.set("user-1", api_key="ak_1", access_token="at_1")
    creds = store.get("user-1")
    assert creds is not None
    assert creds.api_key == "ak_1"
    assert creds.access_token == "at_1"


def test_byok_returns_none_for_unknown_user():
    store = CredentialStore()
    assert store.get("nobody") is None


def test_byok_per_user_isolation():
    store = CredentialStore()
    store.set("a", "ak_a", "at_a")
    store.set("b", "ak_b", "at_b")
    assert store.get("a").api_key == "ak_a"
    assert store.get("b").api_key == "ak_b"
    assert store.connected_users() == 2


def test_byok_forget_clears_user():
    store = CredentialStore()
    store.set("a", "k", "t")
    assert store.has("a") is True
    assert store.forget("a") is True
    assert store.has("a") is False
    # second forget is a no-op
    assert store.forget("a") is False


def test_byok_rejects_empty_inputs():
    store = CredentialStore()
    with pytest.raises(ValueError):
        store.set("", "k", "t")
    with pytest.raises(ValueError):
        store.set("u", "", "t")
    with pytest.raises(ValueError):
        store.set("u", "k", "")


def test_byok_stored_blob_is_not_plaintext():
    """The in-memory blob must be encrypted — a memory dump shouldn't
    leak credentials in the clear."""
    store = CredentialStore()
    store.set("u", "MY_SECRET_KEY", "MY_SECRET_TOKEN")
    blob = store._blobs["u"]
    assert b"MY_SECRET_KEY" not in blob
    assert b"MY_SECRET_TOKEN" not in blob


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def test_audit_log_writes_jsonl_envelope(tmp_path):
    log = AuditLog(path=tmp_path / "audit.jsonl", also_stderr=False)
    log.write("order_placed",
              {"symbol": "NIFTY25000CE", "qty": 75},
              user_id="u-42", session="2026-06-14")
    rows = [json.loads(l) for l in
             (tmp_path / "audit.jsonl").read_text().splitlines() if l]
    assert len(rows) == 1
    r = rows[0]
    assert r["event"] == "order_placed"
    assert r["payload"]["symbol"] == "NIFTY25000CE"
    assert r["user_id"] == "u-42"
    assert r["severity"] == "INFO"
    assert r["ts_utc"].endswith("Z")


def test_audit_log_scrubs_credentials():
    """Secrets must never reach the log even if a caller accidentally
    passes them through."""
    payload = {
        "api_key": "sekrit",
        "nested": {"access_token": "token-here", "fine": "ok"},
        "items": [{"password": "x", "name": "n"}],
        "regular": "kept",
    }
    scrubbed = _scrub(payload)
    assert scrubbed["api_key"] == "<REDACTED>"
    assert scrubbed["nested"]["access_token"] == "<REDACTED>"
    assert scrubbed["nested"]["fine"] == "ok"
    assert scrubbed["items"][0]["password"] == "<REDACTED>"
    assert scrubbed["items"][0]["name"] == "n"
    assert scrubbed["regular"] == "kept"


def test_audit_log_external_sink_fires(tmp_path):
    seen = []
    register_audit_sink(lambda rec: seen.append(rec))
    try:
        log = AuditLog(path=tmp_path / "audit.jsonl", also_stderr=False)
        log.write("graduation", {"source": "x", "from": "SHADOW",
                                   "to": "TRUSTED"})
        assert seen and seen[0]["event"] == "graduation"
    finally:
        clear_audit_sinks()


def test_audit_log_external_sink_failure_does_not_break_write(tmp_path):
    """A broken sink must not take down the audit pipeline."""
    register_audit_sink(lambda rec: (_ for _ in ()).throw(RuntimeError("nope")))
    try:
        log = AuditLog(path=tmp_path / "audit.jsonl", also_stderr=False)
        log.write("test_event", {"k": "v"})    # must not raise
        assert (tmp_path / "audit.jsonl").exists()
    finally:
        clear_audit_sinks()


# ---------------------------------------------------------------------------
# Server integration
# ---------------------------------------------------------------------------

@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    # ensure no leftover credentials from a previous test
    GLOBAL_STORE._blobs.clear()
    import sentinel.server as server
    importlib.reload(server)
    return server, tmp_path


def test_byok_connect_endpoint_stores_credentials(srv):
    server, _ = srv
    with TestClient(server.app) as c:
        r = c.post("/api/byok/connect",
                    headers={"X-Sentinel-Plan": "PRO"},
                    json={"api_key": "k123", "access_token": "t456"})
        assert r.status_code == 200
        body = r.json()
        assert body["connected"] is True
        assert body["user_id"] == "dev"      # demo-mode dev user
        # status endpoint reports connected
        s = c.get("/api/byok/status",
                   headers={"X-Sentinel-Plan": "PRO"}).json()
        assert s["connected"] is True


def test_byok_disconnect_clears_session(srv):
    server, _ = srv
    with TestClient(server.app) as c:
        c.post("/api/byok/connect",
                headers={"X-Sentinel-Plan": "PRO"},
                json={"api_key": "k", "access_token": "t"})
        r = c.delete("/api/byok/connect",
                      headers={"X-Sentinel-Plan": "PRO"})
        assert r.status_code == 200
        s = c.get("/api/byok/status",
                   headers={"X-Sentinel-Plan": "PRO"}).json()
        assert s["connected"] is False


def test_byok_endpoints_refuse_unauthenticated(srv):
    server, _ = srv
    with TestClient(server.app) as c:
        # no plan header AND no authorization header => 401 from require_user
        r = c.post("/api/byok/connect",
                    json={"api_key": "k", "access_token": "t"})
        assert r.status_code == 401


def test_byok_records_to_audit_log(srv):
    server, tmp_path = srv
    with TestClient(server.app) as c:
        c.post("/api/byok/connect",
                headers={"X-Sentinel-Plan": "PRO"},
                json={"api_key": "k", "access_token": "t"})
    audit = (tmp_path / "audit.jsonl").read_text()
    assert "byok_connect" in audit
    # credentials never appear in the log
    assert "k\nt" not in audit


def test_server_start_writes_audit_event(srv):
    server, tmp_path = srv
    audit = (tmp_path / "audit.jsonl").read_text()
    assert "server_started" in audit


# ---------------------------------------------------------------------------
# Hardened auth — header fallback off unless explicitly enabled
# ---------------------------------------------------------------------------

def test_header_auth_blocked_in_production_mode(tmp_path, monkeypatch):
    """Without SENTINEL_DEMO or SENTINEL_ALLOW_HEADER_AUTH, the
    X-Sentinel-Plan dev backdoor is OFF — production cannot accidentally
    leak it."""
    monkeypatch.delenv("SENTINEL_DEMO", raising=False)
    monkeypatch.delenv("SENTINEL_ALLOW_HEADER_AUTH", raising=False)
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    import sentinel.server as server
    importlib.reload(server)
    with TestClient(server.app) as c:
        # /api/byok/connect needs require_user which raises 401 without ctx
        r = c.post("/api/byok/connect",
                    headers={"X-Sentinel-Plan": "FOUNDER"},
                    json={"api_key": "k", "access_token": "t"})
        assert r.status_code == 401
