"""Sentinel trails engine + API smoke tests (demo mode)."""
from __future__ import annotations

from datetime import time as dtime
from pathlib import Path
from typing import Any, Dict, List

import pytest

from sentinel.trails import TrailEngine


class FakeExit:
    def __init__(self, fail: bool = False) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.fail = fail

    def __call__(self, symbol: str, side: str, qty: int) -> Dict[str, Any]:
        self.calls.append({"symbol": symbol, "side": side, "qty": qty})
        if self.fail:
            raise RuntimeError("broker down")
        return {"order_id": f"OK-{len(self.calls)}", "status": "simulated"}


def _engine(tmp_path: Path, fail=False,
            flatten=dtime(23, 59)) -> tuple[TrailEngine, FakeExit]:
    ex = FakeExit(fail)
    return TrailEngine(tmp_path / "t.jsonl", ex, auto_flatten_ist=flatten), ex


# -- per-position trails ------------------------------------------------

def test_long_trail_fires_on_giveback(tmp_path):
    eng, ex = _engine(tmp_path)
    eng.arm("OPT", "long", 75, 30.0, 200.0)
    for p in (220, 250, 240, 219):       # gave back 31 from 250
        eng.on_tick("OPT", p)
    assert len(ex.calls) == 1
    assert ex.calls[0]["side"] == "SELL"


def test_short_trail_fires_on_rise_from_trough(tmp_path):
    eng, ex = _engine(tmp_path)
    eng.arm("OPT", "short", 75, 30.0, 100.0)
    for p in (90, 60, 80, 91):           # rose 31 from trough 60
        eng.on_tick("OPT", p)
    assert len(ex.calls) == 1
    assert ex.calls[0]["side"] == "BUY"


def test_bad_ticks_ignored(tmp_path):
    eng, ex = _engine(tmp_path)
    eng.arm("OPT", "long", 10, 30.0, 200.0)
    eng.on_tick("OPT", 250.0)
    eng.on_tick("OPT", 0.0)
    eng.on_tick("OPT", -5.0)
    eng.on_tick("OPT", 245.0)            # give-back 5 < 30
    assert ex.calls == []


def test_no_double_fire(tmp_path):
    eng, ex = _engine(tmp_path)
    eng.arm("OPT", "long", 10, 30.0, 200.0)
    for p in (250, 200, 150, 100):
        eng.on_tick("OPT", p)
    assert len(ex.calls) == 1


def test_cancel_prevents_fire(tmp_path):
    eng, ex = _engine(tmp_path)
    t = eng.arm("OPT", "long", 10, 30.0, 200.0)
    assert eng.cancel(t.trail_id)
    eng.on_tick("OPT", 100.0)
    assert ex.calls == []


def test_squareoff_flattens_with_reason(tmp_path):
    eng, ex = _engine(tmp_path, flatten=dtime(0, 0))
    eng.arm("OPT", "long", 10, 999.0, 200.0)
    eng.on_tick("OPT", 199.0)
    assert len(ex.calls) == 1
    rows = (tmp_path / "t.jsonl").read_text().strip().splitlines()
    import json
    assert json.loads(rows[-1])["exit_reason"] == "SQUAREOFF"


def test_journal_recovery(tmp_path):
    eng1, _ = _engine(tmp_path)
    t = eng1.arm("OPT", "long", 10, 30.0, 200.0)
    eng2, _ = _engine(tmp_path)              # fresh engine, same journal
    assert any(x.trail_id == t.trail_id for x in eng2.active())


def test_broker_failure_lands_error_state(tmp_path):
    eng, ex = _engine(tmp_path, fail=True)
    eng.arm("OPT", "long", 10, 30.0, 200.0)
    for p in (250, 200):
        eng.on_tick("OPT", p)
    import json
    rows = [json.loads(l) for l in
            (tmp_path / "t.jsonl").read_text().strip().splitlines()]
    assert rows[-1]["state"] == "ERROR"


# -- portfolio lock ----------------------------------------------------------

def test_portfolio_lock_flattens_everything(tmp_path):
    eng, ex = _engine(tmp_path)
    eng.arm("CE1", "long", 75, 500.0, 100.0)
    eng.arm_portfolio(cushion_rupees=2000.0)
    positions = [{"tradingsymbol": "CE1", "quantity": 75},
                 {"tradingsymbol": "PE1", "quantity": -75}]
    assert eng.on_portfolio_pnl(5000.0, positions) is False   # peak set
    assert eng.on_portfolio_pnl(4000.0, positions) is False   # gb 1000 < 2000
    assert eng.on_portfolio_pnl(2900.0, positions) is True    # gb 2100 fires
    # Long flattened with SELL, short with BUY.
    sides = {c["symbol"]: c["side"] for c in ex.calls}
    assert sides == {"CE1": "SELL", "PE1": "BUY"}
    # Per-position trails closed out too.
    assert eng.active() == []


def test_portfolio_lock_fires_once(tmp_path):
    eng, ex = _engine(tmp_path)
    eng.arm_portfolio(1000.0)
    pos = [{"tradingsymbol": "X", "quantity": 10}]
    eng.on_portfolio_pnl(3000.0, pos)
    assert eng.on_portfolio_pnl(1000.0, pos) is True
    assert eng.on_portfolio_pnl(-5000.0, pos) is False        # already fired
    assert len(ex.calls) == 1


# -- API smoke (demo mode) ------------------------------------------------------

def test_server_state_endpoint_demo(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    import importlib
    import sentinel.server as srv
    importlib.reload(srv)
    from fastapi.testclient import TestClient
    with TestClient(srv.app) as client:
        # Drive one refresh cycle manually for determinism.
        srv.CORE._tick_portfolio()
        srv.CORE._tick_recommendations()
        r = client.get("/api/state")
        assert r.status_code == 200
        s = r.json()
        assert s["demo"] is True
        assert len(s["portfolio"]["positions"]) == 2
        assert s["portfolio"]["funds"]["cash"] > 0
        # Pair detected (demo holds a CE and a PE on NIFTY).
        assert len(s["portfolio"]["pairs"]) == 1
        # Scenario curve exists.
        assert len(s["scenario"]["points"]) > 10
        # Recommendations populated from the demo chain.
        assert isinstance(s["recommendations"]["items"], list)

        # Arm a trail via the API and see it in state.
        sym = s["portfolio"]["positions"][0]["tradingsymbol"]
        r2 = client.post("/api/trails", json={
            "tradingsymbol": sym, "side": "long",
            "quantity": 75, "cushion_rupees": 25.0})
        assert r2.status_code == 200
        r3 = client.get("/api/state")
        assert len(r3.json()["trails"]) == 1

        # Kill switch cancels it and refuses orders.
        r4 = client.post("/api/killswitch")
        assert r4.json()["killed"] is True
        assert client.get("/api/state").json()["trails"] == []


def test_server_auth_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.setenv("SENTINEL_TOKEN", "secret123")
    import importlib
    import sentinel.server as srv
    importlib.reload(srv)
    from fastapi.testclient import TestClient
    with TestClient(srv.app) as client:
        assert client.get("/api/state").status_code == 401
        ok = client.get("/api/state",
                        headers={"X-Sentinel-Token": "secret123"})
        assert ok.status_code == 200
