"""Wave 20 — notification filter + rate limiter.

Pins:
  * Notifier routes the right audit events to Alerts and ignores noise
  * Renderers produce well-formed plain + HTML bodies
  * Transports fire; broken transport doesn't take down the pipeline
  * TokenBucketLimiter allows the burst, throttles past it, refills
    over time, and evicts beyond max_keys
  * Rate-limited endpoints return 429 with retry hint
"""
from __future__ import annotations

import importlib
import time

import pytest
from fastapi.testclient import TestClient

from sentinel.audit_log import AuditLog, clear_audit_sinks
from sentinel.notify import (
    Alert, GLOBAL_NOTIFIER, Notifier,
)
from sentinel.rate_limit import (
    PRESETS, TokenBucketLimiter, allow, reset_all,
)


# ─────────────────────────────────────────────────────────────────
# Notifier
# ─────────────────────────────────────────────────────────────────

def test_notifier_routes_intention_violated_to_push(tmp_path):
    n = Notifier()
    n.handle_audit({"event": "intention_violated",
                     "payload": {"reason": "max loss breached"},
                     "user_id": "u1", "severity": "ERROR"})
    delivered = n.delivered()
    assert len(delivered) == 1
    a = delivered[0]
    assert a.channel == "push"
    assert "intention" in a.subject.lower()
    assert "max loss breached" in a.body_plain
    assert "intention" in a.body_html.lower()


def test_notifier_ignores_non_actionable_events():
    n = Notifier()
    for event in ("server_started", "byok_connect", "replay_started",
                   "graduation", "preflight_ack", "order_placed"):
        n.handle_audit({"event": event, "payload": {}, "user_id": ""})
    assert n.delivered() == []


def test_notifier_renders_tilt_red():
    n = Notifier()
    n.handle_audit({"event": "tilt_red",
                     "payload": {"tilt_band": "RED"}, "user_id": "u"})
    assert n.delivered()
    assert "RED" in n.delivered()[0].subject


def test_notifier_renders_mind_report_ready():
    n = Notifier()
    n.handle_audit({"event": "mind_report_ready",
                     "payload": {"tilt_peak": 47.2,
                                  "most_common_bias": "REVENGE"},
                     "user_id": "u"})
    delivered = n.delivered()
    assert delivered and delivered[0].channel == "email"
    assert "Mind Report" in delivered[0].subject
    assert "REVENGE" in delivered[0].body_plain


def test_notifier_renders_order_blocked():
    n = Notifier()
    n.handle_audit({"event": "order_blocked",
                     "payload": {"symbol": "NIFTY25000CE",
                                  "reason": "red_tilt"},
                     "user_id": "u"})
    assert n.delivered()
    assert "NIFTY25000CE" in n.delivered()[0].body_plain


def test_notifier_transport_fires():
    n = Notifier()
    sent: list = []
    n.register_transport(lambda a: sent.append(a))
    n.handle_audit({"event": "intention_violated",
                     "payload": {"reason": "x"},
                     "user_id": "u"})
    assert len(sent) == 1
    assert sent[0].channel == "push"


def test_notifier_broken_transport_does_not_break_pipeline():
    n = Notifier()
    def broken(_): raise RuntimeError("smtp down")
    delivered_good = []
    n.register_transport(broken)
    n.register_transport(lambda a: delivered_good.append(a))
    n.handle_audit({"event": "intention_violated",
                     "payload": {"reason": "x"}, "user_id": "u"})
    # the good transport still fired even though the broken one threw
    assert len(delivered_good) == 1


def test_notifier_attach_to_audit_routes_real_events(tmp_path):
    """End-to-end: AuditLog.write -> notifier sees the event."""
    clear_audit_sinks()
    n = Notifier()
    n.attach_to_audit()
    try:
        log = AuditLog(path=tmp_path / "a.jsonl", also_stderr=False)
        log.write("intention_violated", {"reason": "real-end-to-end"},
                  user_id="u")
        assert n.delivered()
    finally:
        clear_audit_sinks()


# ─────────────────────────────────────────────────────────────────
# Rate limiter
# ─────────────────────────────────────────────────────────────────

def test_token_bucket_allows_within_burst():
    lim = TokenBucketLimiter(capacity=5, refill_per_sec=0.1)
    for _ in range(5):
        assert lim.allow("k") is True
    assert lim.allow("k") is False        # burst exhausted
    assert lim.remaining("k") < 1.0


def test_token_bucket_refills_over_time():
    lim = TokenBucketLimiter(capacity=3, refill_per_sec=100.0)
    for _ in range(3):
        assert lim.allow("k") is True
    assert lim.allow("k") is False
    time.sleep(0.05)                       # at 100/s, that's 5 tokens
    assert lim.allow("k") is True


def test_token_bucket_isolation_between_keys():
    lim = TokenBucketLimiter(capacity=2, refill_per_sec=0.1)
    lim.allow("user_A"); lim.allow("user_A")
    assert lim.allow("user_A") is False
    # different key still has full bucket
    assert lim.allow("user_B") is True


def test_token_bucket_reset_clears_key():
    lim = TokenBucketLimiter(capacity=1, refill_per_sec=0.001)
    lim.allow("k")
    assert lim.allow("k") is False
    lim.reset("k")
    assert lim.allow("k") is True


def test_token_bucket_lru_evicts_beyond_max_keys():
    lim = TokenBucketLimiter(capacity=1, refill_per_sec=0.001, max_keys=3)
    for i in range(5):
        lim.allow(f"k{i}")
    # the first 2 keys must have been evicted
    assert len(lim._buckets) == 3


def test_presets_have_expected_relative_strictness():
    """Account/heavy must be slower than polling/public."""
    assert PRESETS["account"].refill_per_sec < PRESETS["polling"].refill_per_sec
    assert PRESETS["heavy"].refill_per_sec < PRESETS["polling"].refill_per_sec


def test_allow_helper_returns_remaining_count():
    reset_all()
    ok, remaining = allow("public", "ip-1")
    assert ok is True
    assert remaining >= 0
    reset_all()


# ─────────────────────────────────────────────────────────────────
# Server integration — rate limit on /api/byok/connect
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    from sentinel.byok import GLOBAL_STORE
    GLOBAL_STORE._blobs.clear()
    reset_all()
    import sentinel.server as server
    importlib.reload(server)
    return server, tmp_path


def test_byok_connect_rate_limited(srv):
    """The 'account' preset is capacity=10. The 11th hit returns 429."""
    server, _ = srv
    body = {"api_key": "k", "access_token": "t"}
    headers = {"X-Sentinel-Plan": "PRO"}
    with TestClient(server.app) as c:
        # Burn through the burst
        for _ in range(10):
            r = c.post("/api/byok/connect", headers=headers, json=body)
            assert r.status_code == 200
        # 11th = rate-limited
        r = c.post("/api/byok/connect", headers=headers, json=body)
        assert r.status_code == 429
        body_err = r.json()["detail"]
        assert body_err["error"] == "rate_limited"
        assert body_err["preset"] == "account"
        assert body_err["retry_after_seconds"] >= 1


def test_state_not_rate_limited_in_demo(srv):
    """/api/state has no rate limit attached so the polling cockpit
    keeps working. (Polling preset would also allow 60 burst, but the
    point here is the pin: state never throttles.)"""
    server, _ = srv
    with TestClient(server.app) as c:
        # 30 rapid hits all succeed
        for _ in range(30):
            assert c.get("/api/state").status_code == 200


def test_notifier_attached_at_server_boot(srv):
    """Server constructor calls GLOBAL_NOTIFIER.attach_to_audit() so a
    real audit event lands in the notifier's delivered queue."""
    server, tmp_path = srv
    GLOBAL_NOTIFIER._delivered.clear()
    server.CORE.audit.write("intention_violated",
                             {"reason": "boot-time test"},
                             user_id="u-test")
    assert any(a.event == "intention_violated"
               for a in GLOBAL_NOTIFIER.delivered())
