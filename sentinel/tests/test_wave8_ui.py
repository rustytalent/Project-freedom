"""Wave 8 tests — UI serving + the console's customer endpoints.

We can't drive the browser here, but we CAN assert: both pages are
served, they reference the endpoints they call, and every endpoint the
console hits works end-to-end through the real FastAPI app (demo mode).
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    import sentinel.server as server
    importlib.reload(server)
    return server


# ---------------------------------------------------------------------------
# Both pages are served
# ---------------------------------------------------------------------------

def test_operator_cockpit_served(srv):
    with TestClient(srv.app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert "Sentinel" in r.text
        # the new trust-spine panel is present
        assert "Trust spine" in r.text
        assert "/console" in r.text          # link to the customer console


def test_customer_console_served(srv):
    with TestClient(srv.app) as c:
        r = c.get("/console")
        assert r.status_code == 200
        assert "Strategy Auditor" in r.text
        assert "Crisis Stress" in r.text
        # references the endpoints it calls
        for ep in ("/api/audit", "/api/build", "/api/stress",
                   "/api/var", "/api/vol_cone", "/api/me/plan",
                   "/api/equity_context"):
            assert ep in r.text


def test_design_system_stylesheet_served(srv):
    """Bloomberg-grade UI needs the shared design tokens; /sentinel.css
    serves them as text/css and both pages reference it."""
    with TestClient(srv.app) as c:
        css = c.get("/sentinel.css")
        assert css.status_code == 200
        assert css.headers["content-type"].startswith("text/css")
        # design tokens that downstream styling depends on
        for token in ("--amber", "--bg-0", ".panel", ".chip", ".cmdk"):
            assert token in css.text
        # tab content panes — inactive views MUST be display:none, else
        # all five console tabs render at once (caught by a live screenshot)
        assert ".view{" in css.text or ".view {" in css.text
        assert "display: none" in css.text and ".view.on" in css.text
        # both pages link it
        for path in ("/", "/console"):
            page = c.get(path).text
            assert "/sentinel.css" in page


def test_cockpit_has_command_palette_and_status_bar(srv):
    """The polished cockpit ships a command palette (⌘K), a bottom
    status bar, and a visible market-hours indicator."""
    with TestClient(srv.app) as c:
        html = c.get("/").text
        assert "cmdk" in html                   # palette wired
        assert "statusbar" in html              # bottom bar
        assert "NSE" in html                    # market-hours chip
        assert "Trust spine" in html            # spine panel survives


def test_console_has_equity_tab_and_palette(srv):
    with TestClient(srv.app) as c:
        html = c.get("/console").text
        assert "Equity Context" in html         # 5th tab
        assert "cmdk" in html                   # palette wired here too
        # the tabs list the established institutional surfaces
        for t in ("Strategy Auditor", "Strategy Builder", "Crisis Stress",
                  "Risk", "Equity Context"):
            assert t in html


# ---------------------------------------------------------------------------
# The console's endpoints work end-to-end (PRO plan)
# ---------------------------------------------------------------------------

PRO = {"X-Sentinel-Plan": "PRO"}


def test_console_audit_roundtrip(srv):
    with TestClient(srv.app) as c:
        r = c.post("/api/audit", headers=PRO, json={
            "legs": [{"option_type": "CE", "strike": 25000, "qty": 75,
                      "premium": 180, "delta": 0.5, "theta_per_day": -5},
                     {"option_type": "CE", "strike": 25200, "qty": -75,
                      "premium": 90, "delta": 0.3, "theta_per_day": -3}],
            "spot": 25000})
        assert r.status_code == 200
        p = r.json()
        assert p["detected_strategy"] == "BULL_CALL_SPREAD"
        assert p["payoff_curve"] and "net_greeks" in p   # PRO sees greeks


def test_console_build_roundtrip(srv):
    chain = []
    for off in range(-30, 31):
        k = 25000 + off * 50
        chain.append({"option_type": "CE", "strike": k, "premium": max(5, 200 - off * 6)})
        chain.append({"option_type": "PE", "strike": k, "premium": max(5, 200 + off * 6)})
    with TestClient(srv.app) as c:
        r = c.post("/api/build", headers=PRO, json={
            "intent": "VOL_BUY", "spot": 25000, "chain": chain, "iv": 0.15})
        assert r.status_code == 200
        assert r.json()["audit"]["detected_strategy"] == "LONG_STRADDLE"


def test_console_stress_matrix_roundtrip(srv):
    with TestClient(srv.app) as c:
        r = c.post("/api/stress", headers=PRO, json={
            "legs": [{"tradingsymbol": "NIFTY25000CE", "qty": 75, "premium": 180,
                      "delta": 0.5, "gamma": 0.001, "theta_per_day": -5,
                      "vega_per_pct": 15}],
            "spot": 25000})
        assert r.status_code == 200
        assert len(r.json()["matrix"]) >= 5


def test_console_var_and_cone_roundtrip(srv):
    with TestClient(srv.app) as c:
        v = c.post("/api/var", headers=PRO, json={
            "pnls": [i * 10 - 500 for i in range(60)],
            "alpha": 0.99, "method": "cornish_fisher"})
        assert v.status_code == 200 and "var" in v.json()
        cone = c.post("/api/vol_cone", headers=PRO, json={
            "closes": [25000 + i for i in range(150)]})
        assert cone.status_code == 200 and "windows" in cone.json()


def test_console_retail_blocked_shows_upsell_shape(srv):
    """A RETAIL plan hitting a PRO feature gets the 402 the console UI
    renders as an upsell."""
    with TestClient(srv.app) as c:
        r = c.post("/api/stress", headers={"X-Sentinel-Plan": "RETAIL"}, json={
            "legs": [{"tradingsymbol": "X", "qty": 75, "premium": 180}],
            "spot": 25000})
        assert r.status_code == 402
        d = r.json()["detail"]
        assert d["required_tier"] == "PRO" and d["your_tier"] == "RETAIL"
