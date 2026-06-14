"""Launch-day smoke test — the soul test from the Mother's Audit, run
top-to-bottom in one test so a CI failure here means launch is at
risk.

Walks: preflight → live signals on the bus → constituent board +
psychology tick → Crux composer → ledger export → night-trainer adapter
read. Both halves of the project, end to end.
"""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentinel.live_publisher import ModelSignal, now_ist_hms


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    import sentinel.server as server
    importlib.reload(server)
    return server, tmp_path


def test_launch_day_smoke(srv):
    """Walk a full simulated session through the entire stack."""
    server, tmp_path = srv
    core = server.CORE

    # ── 1. Health probes are up before anything else ─────────────────
    with TestClient(server.app) as c:
        assert c.get("/healthz").json()["status"] == "ok"
        assert c.get("/readyz").json()["status"] == "ready"

        # ── 2. State carries every wave's payload ───────────────────
        s = c.get("/api/state").json()
        for key in ("portfolio", "orchestrator", "psychology", "intention",
                    "preflight", "crux", "model_zones",
                    "constituent_board", "premium_chart"):
            assert key in s, f"/api/state missing: {key}"

        # ── 3. Pre-flight is unacknowledged on a fresh session ──────
        assert s["preflight"]["acknowledged"] is False

        # ── 4. Operator commits preflight; intention contract seeded ─
        ack = c.post(
            "/api/preflight/ack",
            headers={"X-Sentinel-Plan": "FOUNDER"},
            json={"accepted_max_loss_rupees": 3000,
                  "expected_regime": "trending"})
        assert ack.status_code == 200
        assert core.psychology.intention.is_set
        assert core.psychology.intention.max_day_loss_rupees == 3000

        # ── 5. liqpool live signals land via the JSONL tail ──────────
        liqpool_signals = (tmp_path / "journal" / "liqpool_live_signals.jsonl")
        liqpool_signals.parent.mkdir(parents=True, exist_ok=True)
        with liqpool_signals.open("a") as f:
            for tag, prob in [("direction_model", 0.78),
                               ("proximity_model", 0.71)]:
                f.write(json.dumps({
                    "ts_ist": now_ist_hms(), "asset": "NIFTY",
                    "model": tag,
                    "signal": f"liqpool head fired {prob:.0%}",
                    "confidence": prob, "trust_tier": "LOGGED",
                    "reason_codes": [tag], "source": "liqpool",
                }) + "\n")
        n_polled = core.liqpool_tail.poll()
        assert n_polled == 2
        # bus now carries the cross-codebase signals
        cur = core.publisher.current()
        assert "NIFTY|direction_model" in cur
        assert "NIFTY|proximity_model" in cur

        # ── 6. Drive a few cycles so models, board, crux all run ────
        for _ in range(5):
            core._tick_board()
            core._tick_psychology()
            core._tick_crux()

        # ── 7. Crux composed a verdict; with preflight set TRADE is allowed
        assert core.crux_verdict is not None
        assert core.crux_verdict.contradicting != "preflight_not_acknowledged"

        # ── 8. Constituent board snapshot has the 10 stocks ──────────
        board = c.get("/api/equity_board").json()
        assert board["stocks"]
        assert len(board["stocks"]) == 10

        # ── 9. Spine + psychology endpoints respond ──────────────────
        assert c.get("/api/psychology").json()["tilt"]["band"] in (
            "GREEN", "AMBER", "RED", "CIRCUIT")
        assert c.get("/api/crux").json()["verdict"] in (
            "EXIT_NOW", "TRAIL_UP", "HOLD", "WATCH", "TRADE",
            "DO_NOT_CHASE", "WAIT")

    # ── 10. Ledger export reads ledger files; the night-trainer adapter
    #         can ingest them via load_sentinel_as_shadow_frame ───────
    # First write a synthetic completed journey to the ledger.
    from sentinel.shadow_ledger import (
        Context, Hypothesis, Identity, KIND_VIRTUAL, ShadowLedger,
        new_event,
    )
    led = ShadowLedger(tmp_path / "journal")
    ident = Identity(
        instrument="NIFTY25000CE", option_type="CE", strike=25000,
        expiry=None, moneyness_key="NIFTY:CE:+0",
        underlying_price=25000, premium=180)
    hyp = Hypothesis(
        expected_underlying_move=50, expected_premium=210,
        expected_horizon_min=15, suggested_entry=180,
        suggested_stop=160, suggested_target=220, confidence=0.65,
        reason_codes=["gamma_expansion"])
    ev = new_event(KIND_VIRTUAL, core.session, ident, Context(), hyp,
                    scientist="test_sci", trust_tier="TRUSTED")
    ShadowLedger.stamp_journey(ev, 5.0, 205)
    ShadowLedger.stamp_journey(ev, 60.0, 215)
    led.write(ev)

    # liqpool's adapter reads it
    from liqpool.sentinel_adapter import load_sentinel_as_shadow_frame
    frame = load_sentinel_as_shadow_frame(
        journal_dir=tmp_path / "journal",
        export_dir=None)
    assert frame is not None
    assert len(frame) >= 1
    # the cross-codebase column shape matches what the flywheel expects
    for col in ("shadow_id", "event_kind", "trading_date_ist",
                 "decision_context", "regime_tags",
                 "counterfactual_outcome"):
        assert col in frame.columns
    # TRUSTED tier propagates into regime_tags so per-tier bucketing works
    assert any("TRUSTED" in tags for tags in frame["regime_tags"])
