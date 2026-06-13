"""Wave 7 tests — trust_tier on persisted events + TrustPromotionRecord.

Closes the last two Sentinel-local items from Codex's queue: every
ledger event carries the trust tier it was emitted at, and every
graduation decision is an auditable record.
"""
from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient

from sentinel.advisor import Suggestion, SuggestionLedger
from sentinel.ledger_export import to_decision_event
from sentinel.orchestration import Tier, TrustPromotionRecord
from sentinel.shadow_ledger import (
    Context, Hypothesis, Identity, KIND_VIRTUAL, ShadowLedger, new_event,
    read_session,
)


# ---------------------------------------------------------------------------
# trust_tier on persisted events
# ---------------------------------------------------------------------------

def test_event_defaults_to_shadow_tier(tmp_path):
    sl = ShadowLedger(tmp_path)
    ident = Identity(instrument="X", option_type="CE", strike=25000,
                     expiry=None, moneyness_key="NIFTY:CE:+0",
                     underlying_price=25000, premium=100)
    hyp = Hypothesis(expected_underlying_move=10, expected_premium=110,
                     expected_horizon_min=15, suggested_entry=100,
                     suggested_stop=90, suggested_target=120, confidence=0.6,
                     reason_codes=["test"])
    ev = new_event(KIND_VIRTUAL, "2026-06-13", ident, Context(), hyp)
    assert ev.trust_tier == "SHADOW"            # scientist hypotheses start SHADOW
    sl.write(ev)
    rows = read_session(tmp_path, "2026-06-13")
    assert rows[0]["trust_tier"] == "SHADOW"


def test_explicit_tier_round_trips(tmp_path):
    sl = ShadowLedger(tmp_path)
    ident = Identity(instrument="X", option_type="CE", strike=25000,
                     expiry=None, moneyness_key="NIFTY:CE:+0",
                     underlying_price=25000, premium=100)
    ev = new_event(KIND_VIRTUAL, "2026-06-13", ident, Context(), None,
                   trust_tier="EXECUTION")
    sl.write(ev)
    rows = read_session(tmp_path, "2026-06-13")
    assert rows[0]["trust_tier"] == "EXECUTION"


def test_canonical_suggestion_is_trusted_tier(tmp_path):
    sl = ShadowLedger(tmp_path)
    led = SuggestionLedger(tmp_path / "s.jsonl", shadow_ledger=sl,
                           session="2026-06-13")
    led.record_many([Suggestion(
        suggestion_id="s1", rule_id="R1", tradingsymbol="NIFTY25000CE",
        action="ARM_TRAIL", reason="x", premium_at_suggestion=180,
        option_type="CE", strike=25000, underlying="NIFTY")])
    rows = read_session(tmp_path, "2026-06-13")
    assert rows[0]["trust_tier"] == "TRUSTED"   # suggestions are operator-facing


def test_trust_tier_flows_through_ledger_export(tmp_path):
    sl = ShadowLedger(tmp_path)
    led = SuggestionLedger(tmp_path / "s.jsonl", shadow_ledger=sl,
                           session="2026-06-13")
    led.record_many([Suggestion(
        suggestion_id="s1", rule_id="R1", tradingsymbol="NIFTY25000CE",
        action="ARM_TRAIL", reason="x", premium_at_suggestion=180,
        option_type="CE", strike=25000, underlying="NIFTY")])
    rows = read_session(tmp_path, "2026-06-13")
    ev = to_decision_event(rows[0])
    assert ev.trust_tier == "TRUSTED"


# ---------------------------------------------------------------------------
# TrustPromotionRecord (unit)
# ---------------------------------------------------------------------------

def test_promotion_record_classifies_direction():
    up = TrustPromotionRecord.build("sci", Tier.SHADOW, Tier.TRUSTED)
    assert up.decision == "promoted" and up.from_tier == "SHADOW"
    down = TrustPromotionRecord.build("sci", Tier.TRUSTED, Tier.LOGGED)
    assert down.decision == "demoted"
    held = TrustPromotionRecord.build("sci", Tier.LOGGED, Tier.LOGGED)
    assert held.decision == "held"


def test_promotion_record_keeps_only_known_evidence():
    rec = TrustPromotionRecord.build(
        "sci", Tier.SHADOW, Tier.TRUSTED, ts_utc="t",
        n=42, hit_rate=0.61, leakage_status="CLEAN",
        garbage_field="ignored")          # unknown evidence dropped
    row = rec.to_row()
    assert row["n"] == 42 and row["hit_rate"] == 0.61
    assert row["leakage_status"] == "CLEAN"
    assert "garbage_field" not in row


# ---------------------------------------------------------------------------
# /api/graduate persists a record (integration)
# ---------------------------------------------------------------------------

@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    import sentinel.server as server
    importlib.reload(server)
    return server


def test_graduate_records_promotion_and_persists(srv, tmp_path):
    with TestClient(srv.app) as client:
        r = client.post("/api/graduate", json={
            "source": "second_pullback", "tier": "TRUSTED",
            "n": 35, "hit_rate": 0.58, "evidence_window": "2026-06-01..13",
            "leakage_status": "CLEAN"})
        assert r.status_code == 200
        promo = r.json()["promotion"]
        assert promo["decision"] == "promoted"
        assert promo["from_tier"] == "SHADOW" and promo["to_tier"] == "TRUSTED"
        assert promo["n"] == 35 and promo["leakage_status"] == "CLEAN"
    # persisted to the journal as append-only JSONL
    path = tmp_path / "trust_promotions.jsonl"
    assert path.exists()
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert rows and rows[0]["source"] == "second_pullback"
    # surfaced in /api/state
    with TestClient(srv.app) as client:
        s = client.get("/api/state").json()
        assert "promotions" in s


def test_graduate_demotion_is_recorded(srv):
    with TestClient(srv.app) as client:
        # maximizer starts at TRUSTED (graduated at startup) -> demote to LOGGED
        r = client.post("/api/graduate",
                        json={"source": "maximizer", "tier": "LOGGED"})
        assert r.json()["promotion"]["decision"] == "demoted"
