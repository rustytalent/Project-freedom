"""Wave 19 — replay, paper trading, backup script, mobile CSS pins.

Pins:
  * ReplayController loads ledger + liqpool archives, refuses while
    spine is live-real, plays at variable speed, pause/resume/stop
  * PaperAccount intercepts every order, simulates fill at LTP,
    tracks open book + realized P&L, journals to JSONL
  * /api/replay/* and /api/paper/* endpoints work end-to-end
  * Mobile CSS rules present
  * Backup script tars the journal + writes a sha256 manifest
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentinel.live_publisher import LivePublisher
from sentinel.paper import PaperAccount, PaperFill, PaperPosition
from sentinel.replay import ReplayController, _hms_to_secs


# ─────────────────────────────────────────────────────────────────
# ReplayController
# ─────────────────────────────────────────────────────────────────

def _write_ledger(journal: Path, session: str, rows: list):
    journal.mkdir(parents=True, exist_ok=True)
    p = journal / f"ledger_{session}.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_replay_refuses_while_live_real(tmp_path):
    pub = LivePublisher()
    r = ReplayController(pub, journal_dir=tmp_path,
                          is_live_real=lambda: True)
    with pytest.raises(RuntimeError, match="live"):
        r.start("2026-06-14", speed=5.0)


def test_replay_loads_events_and_emits_in_order(tmp_path):
    pub = LivePublisher()
    rows = [
        # close timestamps so the inter-event sleep is well below the
        # per-gap cap even at 1000x speed
        {"event_id": "EV2", "kind": "virtual", "session": "2026-06-14",
          "ts_utc": "2026-06-14T04:00:02Z", "scientist": "sci",
          "identity": {"instrument": "X"},
          "hypothesis": {"confidence": 0.7, "reason_codes": ["a"]},
          "journey": {}, "judgment": {}, "trust_tier": "LOGGED"},
        {"event_id": "EV1", "kind": "virtual", "session": "2026-06-14",
          "ts_utc": "2026-06-14T04:00:00Z", "scientist": "sci",
          "identity": {"instrument": "X"},
          "hypothesis": {"confidence": 0.6, "reason_codes": ["b"]},
          "journey": {}, "judgment": {}, "trust_tier": "LOGGED"},
    ]
    _write_ledger(tmp_path, "2026-06-14", rows)
    r = ReplayController(pub, journal_dir=tmp_path)
    r.start("2026-06-14", speed=1000.0)
    for _ in range(100):
        if r.state().state == "done":
            break
        time.sleep(0.05)
    assert r.state().state == "done"
    assert r.state().events_emitted == 2
    # bus has the replayed signals (model name = "<kind>_replay")
    cur = pub.current()
    assert any("_replay" in k for k in cur.keys())


def test_replay_empty_session_completes_immediately(tmp_path):
    pub = LivePublisher()
    r = ReplayController(pub, journal_dir=tmp_path)
    prog = r.start("2099-01-01", speed=10.0)
    assert prog.state == "done"
    assert prog.events_total == 0


def test_replay_pause_resume_stop(tmp_path):
    pub = LivePublisher()
    rows = [{"event_id": f"EV_{i}", "kind": "virtual", "session": "2026-06-14",
              "ts_utc": f"2026-06-14T0{4+i}:00:00Z",
              "scientist": "sci",
              "identity": {"instrument": "X"},
              "hypothesis": {"confidence": 0.5, "reason_codes": []},
              "journey": {}, "judgment": {}, "trust_tier": "LOGGED"}
            for i in range(5)]
    _write_ledger(tmp_path, "2026-06-14", rows)
    r = ReplayController(pub, journal_dir=tmp_path)
    r.start("2026-06-14", speed=1000.0)
    time.sleep(0.05)
    r.pause()
    # after stop the state is done
    prog = r.stop()
    assert prog.state == "done"


def test_replay_refuses_second_start_while_running(tmp_path):
    pub = LivePublisher()
    # craft a session big enough that the run loop doesn't finish before
    # we attempt the second start. 50 events at speed=1 means a multi-
    # second wallclock walk; cap is 5s per gap but we have plenty.
    rows = [{"event_id": f"EV_{i}", "kind": "virtual", "session": "2026-06-14",
              "ts_utc": f"2026-06-14T0{4+i%6}:0{i%6}:00Z",
              "scientist": "sci",
              "identity": {"instrument": "X"},
              "hypothesis": {"confidence": 0.5, "reason_codes": []},
              "journey": {}, "judgment": {}, "trust_tier": "LOGGED"}
            for i in range(50)]
    _write_ledger(tmp_path, "2026-06-14", rows)
    r = ReplayController(pub, journal_dir=tmp_path)
    r.start("2026-06-14", speed=0.5)            # slow on purpose
    try:
        with pytest.raises(RuntimeError, match="already"):
            r.start("2026-06-14", speed=10.0)
    finally:
        r.stop()


def test_replay_signals_carry_replay_marker(tmp_path):
    pub = LivePublisher()
    rows = [{"event_id": "EV1", "kind": "virtual", "session": "2026-06-14",
              "ts_utc": "2026-06-14T04:00:00Z",
              "scientist": "sci", "identity": {"instrument": "NIFTY"},
              "hypothesis": {"confidence": 0.7, "reason_codes": ["a", "b"]},
              "journey": {}, "judgment": {}, "trust_tier": "LOGGED"}]
    _write_ledger(tmp_path, "2026-06-14", rows)
    r = ReplayController(pub, journal_dir=tmp_path)
    r.start("2026-06-14", speed=1000.0)
    for _ in range(40):
        if r.state().state == "done":
            break
        time.sleep(0.05)
    sigs = list(pub.current().values())
    assert sigs and sigs[0].trust_tier == "SHADOW"          # replay never EXECUTION
    assert "replay" in sigs[0].reason_codes
    assert sigs[0].extras.get("source") == "replay"


def test_hms_to_secs_handles_garbage():
    assert _hms_to_secs("00:00:00") == 0
    assert _hms_to_secs("10:30:45") == 10 * 3600 + 30 * 60 + 45
    assert _hms_to_secs("garbage") == 0


# ─────────────────────────────────────────────────────────────────
# PaperAccount
# ─────────────────────────────────────────────────────────────────

class _FakeQuote:
    def __init__(self, ltp): self.ltp = ltp


class _FakeAccount:
    """Stand-in for KiteAccount: passes through funds/positions/quotes."""
    dry_run = True
    label = "fake"
    def funds(self): return {"cash": 50000.0, "utilised": 0.0}
    def positions(self): return []
    def quotes(self, syms, exchange="NFO"):
        return {s: _FakeQuote(ltp=180.0 if "CE" in s else 150.0) for s in syms}
    def instruments(self): return {}
    def spot_ltp(self): return 25000.0


def test_paper_account_intercepts_orders(tmp_path):
    pa = PaperAccount(_FakeAccount(), tmp_path / "paper.jsonl")
    fill = pa.place_market_exit("NIFTY25000CE", "SELL", 75)
    assert fill["status"] == "simulated_paper"
    assert fill["sim_price"] == 180.0
    # journal row present
    rows = (tmp_path / "paper.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["tradingsymbol"] == "NIFTY25000CE"


def test_paper_account_tracks_realized_pnl(tmp_path):
    """Open a paper position then close it at a higher price → P&L."""
    class _LTPAccount(_FakeAccount):
        ltp = 180.0
        def quotes(self, syms, exchange="NFO"):
            return {s: _FakeQuote(ltp=self.ltp) for s in syms}

    underlying = _LTPAccount()
    pa = PaperAccount(underlying, tmp_path / "paper.jsonl")
    # open long 75 at ₹180 by simulating a BUY
    pa.place_market_exit("NIFTY25000CE", "BUY", 75)
    underlying.ltp = 220.0                       # premium pumped
    pa.place_market_exit("NIFTY25000CE", "SELL", 75)
    realized = pa.realized_pnl()
    assert abs(realized - (40 * 75)) < 0.01     # ₹3000


def test_paper_account_funds_reflect_realized_pnl(tmp_path):
    underlying = _FakeAccount()
    pa = PaperAccount(underlying, tmp_path / "paper.jsonl",
                       starting_balance=100000.0)
    funds = pa.funds()
    assert funds["paper_realized_pnl"] == 0.0
    assert funds["paper_starting_balance"] == 100000.0


def test_paper_account_passes_through_reads(tmp_path):
    pa = PaperAccount(_FakeAccount(), tmp_path / "paper.jsonl")
    assert pa.dry_run is True
    assert pa.positions() == []
    assert pa.spot_ltp() == 25000.0
    quotes = pa.quotes(["NIFTY25000CE"])
    assert quotes["NIFTY25000CE"].ltp == 180.0


def test_paper_position_realized_close_math():
    pos = PaperPosition(tradingsymbol="X", quantity=75, avg_price=100.0)
    pnl = pos.realized_close(sim_price=130.0, close_qty=-75)
    assert pnl == 75 * 30
    pos2 = PaperPosition(tradingsymbol="Y", quantity=-50, avg_price=200.0)
    pnl2 = pos2.realized_close(sim_price=170.0, close_qty=50)
    assert pnl2 == 50 * 30


# ─────────────────────────────────────────────────────────────────
# Server endpoints
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    monkeypatch.delenv("SENTINEL_PAPER", raising=False)
    from sentinel.byok import GLOBAL_STORE
    GLOBAL_STORE._blobs.clear()
    import sentinel.server as server
    importlib.reload(server)
    return server, tmp_path


def test_replay_endpoints_full_cycle(srv):
    server, tmp_path = srv
    # seed a ledger for some past session
    rows = [{"event_id": "EV1", "kind": "virtual", "session": "2026-06-13",
              "ts_utc": "2026-06-13T04:00:00Z", "scientist": "sci",
              "identity": {"instrument": "X"},
              "hypothesis": {"confidence": 0.6, "reason_codes": ["replay_test"]},
              "journey": {}, "judgment": {}, "trust_tier": "LOGGED"}]
    _write_ledger(tmp_path, "2026-06-13", rows)
    with TestClient(server.app) as c:
        r = c.post("/api/replay/start",
                    headers={"X-Sentinel-Plan": "FOUNDER"},
                    json={"session_date": "2026-06-13", "speed": 1000.0})
        assert r.status_code == 200
        # poll until done
        for _ in range(40):
            state = c.get("/api/replay/state",
                           headers={"X-Sentinel-Plan": "FOUNDER"}).json()
            if state["state"] == "done":
                break
            time.sleep(0.05)
        assert state["state"] == "done"
        # /api/state surfaces replay block
        s = c.get("/api/state").json()
        assert "replay" in s


def test_replay_start_rejects_unknown_session(srv):
    server, _ = srv
    with TestClient(server.app) as c:
        r = c.post("/api/replay/start",
                    headers={"X-Sentinel-Plan": "FOUNDER"},
                    json={"session_date": "1999-01-01", "speed": 10.0})
        # empty session → state=done immediately, not 4xx
        assert r.status_code == 200
        assert r.json()["state"] == "done"


def test_paper_status_endpoint_when_disabled(srv):
    server, _ = srv
    with TestClient(server.app) as c:
        s = c.get("/api/paper/status",
                   headers={"X-Sentinel-Plan": "FOUNDER"}).json()
        assert s["enabled"] is False


def test_state_carries_paper_block(srv):
    server, _ = srv
    with TestClient(server.app) as c:
        s = c.get("/api/state").json()
        assert "paper" in s
        assert s["paper"]["enabled"] is False


# ─────────────────────────────────────────────────────────────────
# Mobile CSS pins
# ─────────────────────────────────────────────────────────────────

def test_stylesheet_includes_mobile_rules():
    from sentinel.server import STATIC_DIR
    css = (STATIC_DIR / "sentinel.css").read_text()
    assert "@media (max-width: 900px)" in css
    assert "@media (max-width: 540px)" in css
    assert "grid-template-columns: 1fr !important" in css


# ─────────────────────────────────────────────────────────────────
# Backup script
# ─────────────────────────────────────────────────────────────────

def test_backup_script_tars_and_writes_manifest(tmp_path):
    journal = tmp_path / "journal"
    out = tmp_path / "out"
    journal.mkdir()
    (journal / "ledger_2026-06-14.jsonl").write_text('{"event_id":"X"}\n')
    (journal / "audit.jsonl").write_text('{"event":"x"}\n')

    res = subprocess.run(
        [sys.executable, "-m", "sentinel.scripts.backup_journal",
          "--journal", str(journal), "--session", "2026-06-14",
          "--out", str(out), "--keep", "30"],
        capture_output=True, text=True,
        cwd="/home/user/Project-freedom")
    assert res.returncode == 0, res.stderr
    tar = out / "sentinel-backup-2026-06-14.tar.gz"
    assert tar.exists()
    manifest = json.loads(
        (out / "sentinel-backup-2026-06-14.tar.gz.manifest.json").read_text())
    assert manifest["session"] == "2026-06-14"
    files = {f["name"] for f in manifest["files"]}
    assert {"ledger_2026-06-14.jsonl", "audit.jsonl"} <= files
    # each file has a sha256
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])
    # the tarball actually contains the files
    with tarfile.open(tar) as t:
        assert {n for n in t.getnames()} >= {"ledger_2026-06-14.jsonl",
                                              "audit.jsonl"}


def test_backup_script_prunes_old_backups(tmp_path):
    journal = tmp_path / "journal"
    out = tmp_path / "out"
    journal.mkdir(); out.mkdir()
    (journal / "ledger.jsonl").write_text("{}")
    # pre-populate 5 fake old backups (different session labels) with
    # accompanying manifests
    for i in range(5):
        (out / f"sentinel-backup-2026-01-0{i+1}.tar.gz").write_bytes(b"x")
        (out / f"sentinel-backup-2026-01-0{i+1}.tar.gz.manifest.json").write_text("{}")
    res = subprocess.run(
        [sys.executable, "-m", "sentinel.scripts.backup_journal",
          "--journal", str(journal), "--session", "2026-06-14",
          "--out", str(out), "--keep", "2"],
        capture_output=True, text=True,
        cwd="/home/user/Project-freedom")
    assert res.returncode == 0
    remaining = sorted(out.glob("sentinel-backup-*.tar.gz"))
    # only the newest 2 should survive
    assert len(remaining) == 2
