"""Wave 4 tests — closes the remaining open items in the handoff doc:

  * curator fair-value misprice finding (§6.5 complete)
  * SaaS enforcement on the FastAPI endpoints (§6.6 complete)
  * research-context bridge (Codex Problem S3 — sentinel/liqpool_bridge.py)
  * live-to-research adapter (Codex Problem S4 — sentinel/ledger_export.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentinel.curator import Curator
from sentinel.ledger_export import (
    KIND_MAP, export_session, to_decision_event,
)
from sentinel.liqpool_bridge import (
    ResearchContextPack, ZoneLevel, load_research_pack,
    merge_into_snapshot_context,
)
from sentinel.shadow_ledger import (
    Context, Hypothesis, Identity, KIND_VIRTUAL, ShadowLedger,
    new_event, read_session,
)


# ---------------------------------------------------------------------------
# Curator: fair-value misprice finding
# ---------------------------------------------------------------------------

def _seed_session(tmp_path, n, entry_premium, target=120.0, complete=True):
    led = ShadowLedger(tmp_path)
    for i in range(n):
        ident = Identity(instrument=f"S{i}", option_type="CE", strike=25000,
                         expiry=None, moneyness_key="NIFTY:CE:+0",
                         underlying_price=25000, premium=entry_premium, delta=0.5)
        hyp = Hypothesis(expected_underlying_move=60, expected_premium=target,
                         expected_horizon_min=15, suggested_entry=entry_premium,
                         suggested_stop=80.0, suggested_target=target,
                         confidence=0.6, reason_codes=["test"])
        ev = new_event(KIND_VIRTUAL, "2026-06-13", ident, Context(), hyp)
        if complete:
            ShadowLedger.stamp_journey(ev, 1.0, 110)
            ShadowLedger.stamp_journey(ev, 5.0, 130)
            ShadowLedger.stamp_journey(ev, 60.0, 115)
        led.write(ev)
    return read_session(tmp_path, "2026-06-13")


def test_curator_fair_value_misprice_flags_rich_entries(tmp_path):
    rows = _seed_session(tmp_path, n=40, entry_premium=120.0)
    # fair value is 100 -> entries at 120 are +20% rich (>5% threshold)
    cur = Curator(fair_value_resolver=lambda r: 100.0,
                   misprice_threshold=0.05)
    rep = cur.judge_rows("2026-06-13", rows)
    f = next(x for x in rep.findings if x.key == "fair_value_misprice_bias")
    assert f.value > 0.05
    assert "rich" in f.statement.lower()
    assert f.confidence == "firm"


def test_curator_fair_value_misprice_says_healthy_when_aligned(tmp_path):
    rows = _seed_session(tmp_path, n=40, entry_premium=100.0)
    cur = Curator(fair_value_resolver=lambda r: 100.0)
    rep = cur.judge_rows("2026-06-13", rows)
    f = next(x for x in rep.findings if x.key == "fair_value_misprice_bias")
    assert abs(f.value) < 0.05
    assert "healthy" in f.statement.lower()


def test_curator_no_finding_when_no_resolver(tmp_path):
    rows = _seed_session(tmp_path, n=40, entry_premium=120.0)
    cur = Curator()         # no resolver
    rep = cur.judge_rows("2026-06-13", rows)
    assert not any(f.key == "fair_value_misprice_bias" for f in rep.findings)


def test_curator_misprice_stamps_per_event_judgment(tmp_path):
    rows = _seed_session(tmp_path, n=10, entry_premium=120.0)
    cur = Curator(fair_value_resolver=lambda r: 100.0)
    cur.judge_rows("2026-06-13", rows)
    # judgment stamped on each row
    stamped = [r for r in rows
               if "misprice_vs_fair_value_pct" in r.get("judgment", {})]
    assert len(stamped) == len(rows)
    assert abs(stamped[0]["judgment"]["misprice_vs_fair_value_pct"] - 0.20) < 1e-6


# ---------------------------------------------------------------------------
# Research-context bridge (liqpool_bridge.py)
# ---------------------------------------------------------------------------

def test_load_research_pack_returns_empty_when_no_artifacts(tmp_path):
    pack = load_research_pack(session_date="2026-06-13", root=tmp_path)
    assert pack.is_empty
    assert pack.session_date == "2026-06-13"


def test_load_research_pack_reads_research_context_json(tmp_path):
    session = "2026-06-13"
    d = tmp_path / session
    d.mkdir()
    (d / "research_context.json").write_text(json.dumps({
        "run_id": "rb_2026_06_13_001",
        "direction_probability": 0.62,
        "reaction_probability": 0.48,
        "proximity": {"h12": 0.40, "h36": 0.71},
        "sector_regime": "BANK_HOT",
        "avoid": ["earnings_window"],
        "zones": [
            {"label": "PDH", "price": 25180, "side": "resistance",
             "horizon_minutes": 60, "reach_probability": 0.55},
        ],
    }))
    pack = load_research_pack(session_date=session, root=tmp_path)
    assert not pack.is_empty
    assert pack.source_run_id == "rb_2026_06_13_001"
    assert pack.direction_probability == 0.62
    assert pack.proximity_h12 == 0.40
    assert pack.sector_regime == "BANK_HOT"
    assert pack.avoid_flags == ["earnings_window"]
    assert pack.key_zones and pack.key_zones[0].label == "PDH"


def test_load_research_pack_falls_back_to_latest(tmp_path):
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "daily_brief.json").write_text(json.dumps({
        "direction": {"up": 0.55},
        "reaction": {"fire": 0.42},
    }))
    pack = load_research_pack(session_date="2099-01-01", root=tmp_path)
    assert pack.direction_probability == 0.55
    assert pack.reaction_probability == 0.42


def test_merge_into_snapshot_context_populates_scientist_fields():
    pack = ResearchContextPack(
        direction_probability=0.7,
        reaction_probability=0.5,
        sector_regime="IT_HOT",
        avoid_flags=["earnings"],
        proximity_h12=0.6,
        source_run_id="rb_xyz",
        key_zones=[ZoneLevel("PDL", 24900, "support", 60, 0.4)],
    )
    ctx = merge_into_snapshot_context(pack, {})
    assert ctx["model_signal"] == 0.7
    assert ctx["reaction_model"] == 0.5
    assert ctx["sector_regime"] == "IT_HOT"
    assert ctx["avoid_flags"] == ["earnings"]
    assert ctx["proximity_h12"] == 0.6
    assert ctx["key_zones"][0]["label"] == "PDL"
    assert ctx["research_pack_id"] == "rb_xyz"


def test_load_research_pack_tolerates_corrupt_payload(tmp_path):
    session = "2026-06-13"
    d = tmp_path / session
    d.mkdir()
    (d / "research_context.json").write_text("not json {")
    pack = load_research_pack(session_date=session, root=tmp_path)
    assert pack.is_empty


# ---------------------------------------------------------------------------
# Live-to-research adapter (ledger_export.py)
# ---------------------------------------------------------------------------

def test_decision_event_maps_sentinel_kinds():
    """Every Sentinel event kind has a DecisionEvent type in KIND_MAP."""
    expected = {"actual", "virtual", "rejected", "alternative",
                "model_suggestion", "counterfactual"}
    assert expected <= set(KIND_MAP)


def test_to_decision_event_carries_identity_and_journey(tmp_path):
    rows = _seed_session(tmp_path, n=2, entry_premium=100.0)
    ev = to_decision_event(rows[0])
    assert ev.event_type == "VIRTUAL_DECISION"
    assert ev.option_type == "CE"
    assert ev.entry_premium == 100.0
    assert ev.confidence == 0.6
    assert ev.reason_codes == ["test"]
    assert ev.journey_complete is True
    assert ev.realized_outcome_t60 == 115


def test_export_session_writes_jsonl_with_one_row_per_event(tmp_path):
    _seed_session(tmp_path, n=5, entry_premium=100.0)
    out = tmp_path / "out"
    path = export_session(tmp_path, "2026-06-13", out)
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 5
    row = json.loads(lines[0])
    assert row["source"] == "sentinel"
    assert row["event_type"] == "VIRTUAL_DECISION"
    # event_id present and unique
    ids = {json.loads(l)["event_id"] for l in lines}
    assert len(ids) == 5


def test_export_session_handles_empty_session(tmp_path):
    """Empty session writes a zero-row file (so downstream can distinguish
    'not exported' from 'exported with no events')."""
    out = tmp_path / "out"
    path = export_session(tmp_path, "2099-01-01", out)
    assert path.exists() and path.read_text() == ""


def test_export_session_omits_none_fields():
    """DecisionEvent.to_row() drops None/empty so the research-side schema
    sees clean columns."""
    ev = to_decision_event({"event_id": "x", "kind": "rejected",
                              "session": "2026-06-13", "ts_utc": "...",
                              "identity": {"instrument": "FOO"}})
    row = ev.to_row()
    assert "event_id" in row and "event_type" in row
    assert "strike" not in row    # was None -> omitted


# ---------------------------------------------------------------------------
# SaaS enforcement on the FastAPI server
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient with SENTINEL_DEMO=1, no dashboard token. The module-
    level CFG/CORE/app are rebuilt by reload() so previously-loaded test
    config (e.g. SENTINEL_TOKEN from test_trails_server) doesn't leak in."""
    import importlib
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    from sentinel import server as srv
    importlib.reload(srv)
    with TestClient(srv.app) as c:
        yield c


def _retail(headers=None):
    h = {"X-Sentinel-Plan": "RETAIL"}
    if headers:
        h.update(headers)
    return h


def _pro(headers=None):
    h = {"X-Sentinel-Plan": "PRO"}
    if headers:
        h.update(headers)
    return h


def test_me_plan_returns_entitlement_summary(client):
    r = client.get("/api/me/plan", headers=_retail())
    assert r.status_code == 200
    body = r.json()
    assert body["plan"] == "RETAIL"
    assert body["level"] == 1
    assert "features" in body and "rv.close_to_close" in body["features"]


def test_saas_catalog_marks_allowed_per_plan(client):
    r = client.get("/api/saas/catalog", headers=_retail())
    rows = {row["feature"]: row for row in r.json()["catalog"]}
    assert rows["rv.close_to_close"]["allowed"] is True
    assert rows["rv.yang_zhang"]["allowed"] is False    # QUANT-only


def test_audit_endpoint_retail_strips_greek_book(client):
    body = {
        "legs": [{"option_type": "CE", "strike": 25000, "qty": 75,
                  "premium": 180}],
        "spot": 25000,
    }
    r = client.post("/api/audit", headers=_retail(), json=body)
    assert r.status_code == 200
    p = r.json()
    assert p["detected_strategy"] == "LONG_CALL"
    # RETAIL -> greek_book + risk_flags stripped
    assert "net_greeks" not in p
    assert "flags" not in p


def test_audit_endpoint_pro_includes_greek_book_and_flags(client):
    body = {
        "legs": [{"option_type": "CE", "strike": 25000, "qty": -75,
                  "premium": 180}],
        "spot": 25000,
        "regime": "chop",
    }
    r = client.post("/api/audit", headers=_pro(), json=body)
    p = r.json()
    assert "net_greeks" in p
    assert "flags" in p
    # naked short call -> uncapped-upside flag must fire
    assert any("uncapped" in f.lower() for f in p["flags"])


def test_stress_full_matrix_requires_pro(client):
    body = {
        "legs": [{"tradingsymbol": "NIFTY25000CE", "qty": 75, "premium": 180,
                   "delta": 0.5, "gamma": 0.001, "theta_per_day": -5.0,
                   "vega_per_pct": 15.0}],
        "spot": 25000,
    }
    # RETAIL cannot request the full matrix
    r = client.post("/api/stress", headers=_retail(), json=body)
    assert r.status_code == 402
    detail = r.json()["detail"]
    assert detail["required_tier"] == "PRO"
    # PRO can
    r2 = client.post("/api/stress", headers=_pro(), json=body)
    assert r2.status_code == 200
    assert len(r2.json()["matrix"]) >= 5


def test_stress_single_scenario_works_for_pro(client):
    body = {
        "legs": [{"tradingsymbol": "NIFTY25000CE", "qty": 75, "premium": 180,
                   "delta": 0.5, "gamma": 0.001, "theta_per_day": -5.0,
                   "vega_per_pct": 15.0}],
        "spot": 25000,
        "scenario": "COVID_MAR_2020",
    }
    r = client.post("/api/stress", headers=_pro(), json=body)
    assert r.status_code == 200
    p = r.json()
    assert p["scenario"] == "COVID_MAR_2020"
    # Endpoint shape + gate test only — sign correctness for parametric vs
    # BS-reprice paths is pinned in test_wave3.test_stress_test_bs_reprice_path.
    assert "portfolio_pnl" in p and isinstance(p["portfolio_pnl"], (int, float))


def test_var_cornish_fisher_requires_pro(client):
    body = {"pnls": [i * 10 - 500 for i in range(100)],
            "alpha": 0.99, "method": "cornish_fisher"}
    r = client.post("/api/var", headers=_retail(), json=body)
    assert r.status_code == 402
    r2 = client.post("/api/var", headers=_pro(), json=body)
    assert r2.status_code == 200
    assert "var" in r2.json()


def test_var_historical_allowed_for_retail(client):
    body = {"pnls": list(range(-50, 51)), "alpha": 0.95, "method": "historical"}
    r = client.post("/api/var", headers=_retail(), json=body)
    assert r.status_code == 200


def test_vol_cone_requires_pro(client):
    body = {"closes": [25000 + i for i in range(120)]}
    r = client.post("/api/vol_cone", headers=_retail(), json=body)
    assert r.status_code == 402
    r2 = client.post("/api/vol_cone", headers=_pro(), json=body)
    assert r2.status_code == 200


def test_equity_context_retail_gets_scalar(client):
    body = {"stock_returns_pct": {"RELIANCE": 2.0, "HDFCBANK": 1.5},
            "index_return_pct": 0.1}
    r = client.post("/api/equity_context", headers=_retail(), json=body)
    p = r.json()
    assert r.status_code == 200
    assert "weightage_divergence" in p
    assert "leaders" not in p     # full context is PRO


def test_equity_context_pro_gets_full_pack(client):
    body = {"stock_returns_pct": {"RELIANCE": 2.0, "HDFCBANK": 1.5},
            "index_return_pct": 0.1}
    r = client.post("/api/equity_context", headers=_pro(), json=body)
    p = r.json()
    assert "leaders" in p and "contributions" in p


def test_build_endpoint_unknown_intent_400(client):
    body = {"intent": "FREE_MONEY_PLEASE", "spot": 25000,
            "chain": [], "iv": 0.15}
    r = client.post("/api/build", headers=_pro(), json=body)
    assert r.status_code == 400
