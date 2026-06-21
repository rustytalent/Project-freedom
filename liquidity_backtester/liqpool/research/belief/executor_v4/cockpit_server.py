"""HTTP / SSE server for the cockpit.

A tiny stdlib-only HTTP server that exposes the current cockpit
snapshot via three endpoints:

  GET /                  → HTML viewer (built-in)
  GET /api/cockpit       → latest cockpit snapshot as JSON
  GET /api/journeys      → all open + closed position journeys as JSON
  GET /api/journey/<id>  → single position journey
  GET /api/stream        → Server-Sent Events stream (one event per
                            cockpit update)
  GET /health            → health_report(runner) JSON

The server is driven by a CockpitFeed — the V4Runner pushes updates to
the feed via ``feed.publish(cockpit_dict)``; the SSE endpoint streams
them to all connected clients.

Threading: the HTTP server runs in a background thread so it doesn't
block the trading loop. Updates are stored in a thread-safe latest-
snapshot slot + a bounded ring buffer for replay.

Usage:

    from liqpool.research.belief.executor_v4.cockpit_server import (
        CockpitServer,
    )
    server = CockpitServer(runner, port=8765)
    server.start()
    # ... run trading loop ...
    server.stop()

Then open http://localhost:8765/ in a browser.
"""
from __future__ import annotations

import collections
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .cockpit import build_cockpit_snapshot
from .journey import build_all_journeys, build_journey, render_journey
from .monday_bonuses import health_report


log = logging.getLogger(__name__)


# ── Feed ──────────────────────────────────────────────────────────


class CockpitFeed:
    """Thread-safe latest-snapshot + ring buffer + subscriber notify."""

    def __init__(self, *, history_size: int = 256) -> None:
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._latest: Optional[Dict[str, Any]] = None
        self._history: Deque[Tuple[int, Dict[str, Any]]] = collections.deque(
            maxlen=history_size)
        self._seq = 0

    def publish(self, cockpit_dict: Dict[str, Any]) -> int:
        with self._cv:
            self._seq += 1
            self._latest = cockpit_dict
            self._history.append((self._seq, cockpit_dict))
            self._cv.notify_all()
            return self._seq

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest

    def history(self) -> List[Tuple[int, Dict[str, Any]]]:
        with self._lock:
            return list(self._history)

    def wait_for(self, last_seen_seq: int, timeout: float = 10.0
                   ) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Block until an event after ``last_seen_seq`` is available.
        Returns (seq, payload) or None on timeout."""
        with self._cv:
            end = time.time() + timeout
            while True:
                # Find the next event after last_seen_seq.
                for s, payload in self._history:
                    if s > last_seen_seq:
                        return (s, payload)
                remaining = end - time.time()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=remaining)


# ── HTTP handler ──────────────────────────────────────────────────


def _make_handler(feed: CockpitFeed, runner_ref: Callable[[], Any]):
    """Closure-based handler factory. ``runner_ref`` returns the live
    V4Runner instance (so we can serve health + journeys)."""

    class CockpitHandler(BaseHTTPRequestHandler):
        # Reduce log noise — log only on error.
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            try:
                if self.path == "/" or self.path == "/index.html":
                    self._serve_viewer()
                    return
                if self.path == "/api/cockpit":
                    self._serve_cockpit_json()
                    return
                if self.path == "/api/journeys":
                    self._serve_journeys()
                    return
                if self.path.startswith("/api/journey/"):
                    pid = self.path[len("/api/journey/"):]
                    self._serve_single_journey(pid)
                    return
                if self.path == "/api/stream":
                    self._serve_sse()
                    return
                if self.path == "/health":
                    self._serve_health()
                    return
                if self.path == "/api/controls":
                    self._serve_controls_get()
                    return
                self._send_text(404, "not found\n")
            except Exception as exc:
                log.exception("handler error: %s", exc)
                try:
                    self._send_text(500, f"internal error: {exc}\n")
                except Exception:
                    pass

        def do_POST(self) -> None:
            try:
                # Read body if any.
                length = int(self.headers.get("Content-Length") or 0)
                body = (self.rfile.read(length).decode("utf-8")
                          if length else "")
                # CONTROL endpoints — these MUTATE live state.
                if self.path == "/api/execution/enable":
                    self._control_execution_enable(body)
                    return
                if self.path == "/api/execution/disable":
                    self._control_execution_disable(body)
                    return
                if self.path == "/api/execution/kill":
                    self._control_execution_kill(body)
                    return
                if self.path == "/api/calibrator/pause":
                    self._control_calibrator_pause(body)
                    return
                if self.path == "/api/calibrator/resume":
                    self._control_calibrator_resume(body)
                    return
                if self.path == "/api/calibrator/rollback":
                    self._control_calibrator_rollback(body)
                    return
                self._send_text(404, "control endpoint not found\n")
            except Exception as exc:
                log.exception("control handler error: %s", exc)
                try:
                    self._send_text(500, f"internal error: {exc}\n")
                except Exception:
                    pass

        # ── Control handlers (LIVE TOGGLES) ───────────────────────

        def _serve_controls_get(self) -> None:
            runner = runner_ref()
            if runner is None:
                self._send_json(503, {"error": "runner not attached"})
                return
            out = {
                "broker_name": type(runner.broker).__name__,
                "broker_is_live": bool(getattr(runner.broker, "is_live", False)),
                "broker_killed": bool(getattr(runner.broker, "killed", False)),
                "calibrator_paused": False,
                "calibrator_enabled": False,
            }
            try:
                out["calibrator_paused"] = bool(
                    runner.manager.live_calibrator.paused)
                out["calibrator_enabled"] = bool(
                    runner.manager.live_calibrator.cfg.enabled)
            except Exception:
                pass
            self._send_json(200, out)

        def _control_execution_enable(self, body: str) -> None:
            """ENABLE real execution. Requires explicit confirm flag in body."""
            runner = runner_ref()
            if runner is None:
                self._send_json(503, {"error": "runner not attached"})
                return
            # Demand explicit confirmation phrase.
            data = self._parse_body(body)
            if data.get("confirm") != "I_UNDERSTAND_REAL_MONEY":
                self._send_json(400, {
                    "error": "must POST {'confirm': 'I_UNDERSTAND_REAL_MONEY'}",
                    "reason": "safety guard against accidental live toggle",
                })
                return
            broker = runner.broker
            broker.killed = False
            # Flip Kite confirm_real if applicable.
            if hasattr(broker, "cfg") and hasattr(broker.cfg, "confirm_real"):
                broker.cfg.confirm_real = True
                broker.is_live = True
                broker.dry_run = False
            self._send_json(200, {
                "ok": True,
                "broker_is_live": bool(getattr(broker, "is_live", False)),
            })

        def _control_execution_disable(self, body: str) -> None:
            runner = runner_ref()
            if runner is None:
                self._send_json(503, {"error": "runner not attached"})
                return
            broker = runner.broker
            if hasattr(broker, "cfg") and hasattr(broker.cfg, "confirm_real"):
                broker.cfg.confirm_real = False
                broker.is_live = False
                broker.dry_run = True
            self._send_json(200, {"ok": True,
                                     "broker_is_live": False})

        def _control_execution_kill(self, body: str) -> None:
            """Emergency kill — block ALL future order placements."""
            runner = runner_ref()
            if runner is None:
                self._send_json(503, {"error": "runner not attached"})
                return
            try:
                runner.broker.kill_switch("operator kill via UI")
            except Exception:
                runner.broker.killed = True
            self._send_json(200, {"ok": True, "killed": True})

        def _control_calibrator_pause(self, body: str) -> None:
            runner = runner_ref()
            if runner is None:
                self._send_json(503, {"error": "runner not attached"})
                return
            try:
                runner.manager.live_calibrator.pause("operator")
                self._send_json(200, {"ok": True, "paused": True})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def _control_calibrator_resume(self, body: str) -> None:
            runner = runner_ref()
            if runner is None:
                self._send_json(503, {"error": "runner not attached"})
                return
            try:
                runner.manager.live_calibrator.resume()
                self._send_json(200, {"ok": True, "paused": False})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def _control_calibrator_rollback(self, body: str) -> None:
            runner = runner_ref()
            if runner is None:
                self._send_json(503, {"error": "runner not attached"})
                return
            try:
                ok = runner.manager.live_calibrator.rollback_last_applied()
                self._send_json(200, {"ok": ok})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def _parse_body(self, body: str) -> Dict[str, Any]:
            if not body:
                return {}
            try:
                return json.loads(body) or {}
            except Exception:
                return {}

        def _serve_viewer(self) -> None:
            body = _DEFAULT_VIEWER_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_cockpit_json(self) -> None:
            latest = feed.latest()
            self._send_json(200, latest if latest else {"empty": True})

        def _serve_journeys(self) -> None:
            runner = runner_ref()
            if runner is None:
                self._send_json(503, {"error": "runner not attached"})
                return
            journeys = build_all_journeys(runner.manager)
            self._send_json(200, {
                "journeys": [j.to_dict() for j in journeys],
                "count": len(journeys),
            })

        def _serve_single_journey(self, pid: str) -> None:
            runner = runner_ref()
            if runner is None:
                self._send_json(503, {"error": "runner not attached"})
                return
            j = build_journey(runner.manager, pid)
            if j is None:
                self._send_json(404, {"error": f"position {pid} not found"})
                return
            self._send_json(200, j.to_dict())

        def _serve_health(self) -> None:
            runner = runner_ref()
            if runner is None:
                self._send_json(503, {"error": "runner not attached"})
                return
            self._send_json(200, health_report(runner))

        def _serve_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last_seq = 0
            # Send the latest immediately if any.
            latest = feed.latest()
            if latest is not None:
                self._write_sse_event(last_seq + 1, latest)
            # Then loop, blocking on the condvar.
            while True:
                try:
                    event = feed.wait_for(last_seq, timeout=10.0)
                    if event is None:
                        # Keep-alive
                        try:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                        continue
                    seq, payload = event
                    self._write_sse_event(seq, payload)
                    last_seq = seq
                except (BrokenPipeError, ConnectionResetError):
                    return

        def _write_sse_event(self, seq: int, payload: Dict[str, Any]
                                ) -> None:
            body = json.dumps(payload, default=str)
            try:
                self.wfile.write(
                    f"id: {seq}\nevent: cockpit\ndata: {body}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_text(self, status: int, text: str) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, data: Any) -> None:
            body = json.dumps(data, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

    return CockpitHandler


# ── Server wrapper ────────────────────────────────────────────────


class CockpitServer:
    """Background HTTP server for the cockpit feed.

    Plug it into the V4Runner by wrapping the runner's on_tick(...) and
    calling ``feed.publish(...)`` after each tick.
    """

    def __init__(self, runner: Any, *, host: str = "127.0.0.1",
                  port: int = 8765) -> None:
        self.runner = runner
        self.feed = CockpitFeed()
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.runner_ref = lambda: self.runner

    def start(self) -> None:
        handler_cls = _make_handler(self.feed, self.runner_ref)
        self._server = ThreadingHTTPServer((self.host, self.port),
                                              handler_cls)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="CockpitServer",
            daemon=True,
        )
        self._thread.start()
        log.info("cockpit server listening on http://%s:%d/",
                  self.host, self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def publish_from_intent(self, intent_dict: Dict[str, Any]) -> None:
        """Convert an intent dict to a cockpit snapshot and publish."""
        cockpit = build_cockpit_snapshot(intent_dict)
        self.feed.publish(cockpit.to_dict())

    def publish(self, cockpit_dict: Dict[str, Any]) -> None:
        self.feed.publish(cockpit_dict)


# ── Embedded viewer HTML ──────────────────────────────────────────


_DEFAULT_VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Premium Belief v4 — cockpit</title>
<style>
  :root {
    --bg:#0a0e14; --panel:#10151c; --accent:#5dffd0; --text:#cad3df;
    --muted:#5a6172; --green:#7dff8e; --red:#ff7d8e; --yellow:#fff19a;
    --bull:#7dff8e; --bear:#ff7d8e;
  }
  body { margin:0; padding:24px; background:var(--bg); color:var(--text);
          font-family: ui-monospace, "JetBrains Mono", monospace;
          font-size: 13px; line-height: 1.5; }
  h1 { color: var(--accent); font-size: 18px; margin: 0 0 8px 0;
        letter-spacing: 2px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr;
           gap: 16px; max-width: 1400px; }
  .panel { background: var(--panel); padding: 16px; border-radius: 8px;
            border-left: 3px solid var(--accent); }
  .panel h2 { margin: 0 0 12px 0; font-size: 12px;
               color: var(--accent); letter-spacing: 2px; text-transform: uppercase; }
  .kv { display:grid; grid-template-columns: 200px 1fr; gap: 4px 12px; }
  .kv .k { color: var(--muted); }
  .kv .v { color: var(--text); }
  .bar { display:inline-block; width:140px; height:14px;
          background: #1c2330; border-radius: 2px; vertical-align: middle;
          overflow: hidden; margin-right: 8px; }
  .bar-fill { height: 100%; background: var(--accent); }
  .green { color: var(--green); }
  .red { color: var(--red); }
  .yellow { color: var(--yellow); }
  .muted { color: var(--muted); }
  .panel.full { grid-column: span 2; }
  .position-row { padding: 6px 0; border-bottom: 1px solid #1c2330; }
  .position-row:last-child { border-bottom: none; }
  .badge { display:inline-block; padding:2px 6px; border-radius:3px;
            font-size:11px; margin-left: 6px;
            background: #1c2330; color: var(--muted); }
  .badge.bull { background: rgba(125,255,142,.12); color: var(--bull); }
  .badge.bear { background: rgba(255,125,142,.12); color: var(--bear); }
  .pulse { animation: pulse 2s infinite; }
  @keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.4; }
  }
  .status-dot { display: inline-block; width: 8px; height: 8px;
                 border-radius: 50%; background: var(--green);
                 margin-right: 6px; vertical-align: middle; }
  .footer { margin-top: 24px; color: var(--muted); font-size: 11px; }
  pre { white-space: pre-wrap; word-break: break-word; margin: 0;
         font-family: inherit; font-size: 12px; }
  .control-bar { margin-top: 12px; padding: 12px; background: var(--panel);
                  border-radius: 8px; display: flex; align-items: center;
                  gap: 10px; flex-wrap: wrap; }
  .btn { padding: 8px 14px; border: none; border-radius: 4px;
          font-family: inherit; font-size: 13px; cursor: pointer;
          font-weight: bold; letter-spacing: 1px; }
  .btn-paper { background: #2a4d3a; color: var(--green); }
  .btn-live { background: var(--bear); color: #fff; }
  .btn-kill { background: #6b0d18; color: #fff; }
  .btn-kill:hover { background: #8b0d18; }
  .btn-neutral { background: #2a3340; color: var(--text); }
  .btn:hover { filter: brightness(1.15); }
</style>
</head>
<body>
  <h1>◆ PREMIUM BELIEF v4 — LIVE COCKPIT</h1>
  <div id="status" class="muted"><span class="status-dot pulse"></span>connecting…</div>
  <div class="control-bar">
    <button id="btn-live-toggle" class="btn btn-paper">PAPER MODE</button>
    <button id="btn-kill" class="btn btn-kill">⛔ EMERGENCY KILL</button>
    <button id="btn-calib-toggle" class="btn btn-neutral">CALIB: …</button>
    <button id="btn-rollback" class="btn btn-neutral">↩ ROLLBACK</button>
    <span id="control-status" class="muted" style="margin-left: 12px;"></span>
  </div>
  <div class="grid" style="margin-top: 12px">
    <div class="panel">
      <h2>ACTION</h2>
      <pre id="action_card">—</pre>
    </div>
    <div class="panel">
      <h2>P&amp;L</h2>
      <div class="kv">
        <div class="k">Daily</div><div class="v" id="pnl_daily">—</div>
        <div class="k">Fees</div><div class="v" id="pnl_fees">—</div>
        <div class="k">Win rate</div><div class="v" id="pnl_win">—</div>
      </div>
    </div>
    <div class="panel">
      <h2>PROBABILITY WEB</h2>
      <div class="kv">
        <div class="k">Consensus</div><div class="v" id="web_consensus">—</div>
        <div class="k">Tail mass</div><div class="v" id="web_tail">—</div>
        <div class="k">Chop mass</div><div class="v" id="web_chop">—</div>
        <div class="k">Dominant strategy</div><div class="v" id="web_dom_strat">—</div>
      </div>
      <h2 style="margin-top:14px;">Top scenarios</h2>
      <div id="web_scenarios">—</div>
    </div>
    <div class="panel">
      <h2>MM MIND</h2>
      <div class="kv">
        <div class="k">Dominant intent</div><div class="v" id="mm_intent">—</div>
        <div class="k">Probability</div><div class="v" id="mm_prob">—</div>
        <div class="k">Bias</div><div class="v" id="mm_bias">—</div>
        <div class="k">Vol view</div><div class="v" id="mm_vol_view">—</div>
        <div class="k">Confidence</div><div class="v" id="mm_conf">—</div>
      </div>
      <pre id="mm_guidance" class="muted" style="margin-top:8px;">—</pre>
    </div>
    <div class="panel">
      <h2>FAT-TAIL</h2>
      <div class="kv">
        <div class="k">Score</div><div class="v" id="tail_score">—</div>
        <div class="k">Action</div><div class="v" id="tail_action">—</div>
      </div>
    </div>
    <div class="panel">
      <h2>CROWD MIRROR</h2>
      <div class="kv">
        <div class="k">We look retail?</div><div class="v" id="crowd_retail">—</div>
        <div class="k">Similarity score</div><div class="v" id="crowd_score">—</div>
        <div class="k">ATM CE density</div><div class="v" id="crowd_ce">—</div>
        <div class="k">ATM PE density</div><div class="v" id="crowd_pe">—</div>
      </div>
    </div>
    <div class="panel full">
      <h2>RISK / GREEKS</h2>
      <div class="kv">
        <div class="k">Net Δ</div><div class="v" id="risk_delta">—</div>
        <div class="k">Net V</div><div class="v" id="risk_vega">—</div>
        <div class="k">Net Γ</div><div class="v" id="risk_gamma">—</div>
        <div class="k">Net Θ/day</div><div class="v" id="risk_theta">—</div>
        <div class="k">Premium at risk</div><div class="v" id="risk_par">—</div>
        <div class="k">Drawdown R</div><div class="v" id="risk_dd">—</div>
      </div>
      <div id="risk_kills" style="margin-top:8px;"></div>
    </div>
    <div class="panel full">
      <h2>OPEN POSITIONS</h2>
      <div id="positions">—</div>
    </div>
    <div class="panel full">
      <h2>ACTIVE PATTERNS</h2>
      <div id="patterns">—</div>
    </div>
  </div>
  <div class="footer">SSE source: <code>/api/stream</code>. Latest snapshot: <span id="ts">—</span></div>
<script>
function fmt_pnl(v) {
  if (v == null) return "—";
  var n = Number(v);
  var sign = n > 0 ? "+" : "";
  var cls = n > 0 ? "green" : (n < 0 ? "red" : "");
  return `<span class="${cls}">${sign}₹${n.toLocaleString("en-IN", {maximumFractionDigits: 0})}</span>`;
}
function fmt_pct(v) {
  if (v == null) return "—";
  return (Number(v) * 100).toFixed(1) + "%";
}
function fmt_bar(v, low, high) {
  v = Math.max(low, Math.min(high, Number(v)));
  var pct = (v - low) / (high - low) * 100;
  return `<span class="bar"><span class="bar-fill" style="width:${pct.toFixed(0)}%"></span></span>`;
}
function render(snap) {
  document.getElementById("ts").textContent = snap.ts || "—";
  document.getElementById("status").innerHTML = '<span class="status-dot"></span>live';
  // Action card
  var ac = snap.action_card || {};
  var headline = ac.headline || "—";
  var detail = "";
  if (ac.kind === "ENTER") {
    detail = `\nscore ${(ac.aggregator_score||0).toFixed(2)} | size×${(ac.size_multiplier||0).toFixed(2)}`;
    detail += `\nstop ₹${ac.stop_premium} target ₹${ac.target_premium}`;
  } else if (ac.kind === "EXIT") {
    detail = `\nrealized ${fmt_pnl(ac.realized_rupees)} (${(ac.realized_r||0).toFixed(2)}R) ${ac.bars_held}b`;
    detail += `\nreason: ${ac.exit_reason || ""}`;
  } else if (ac.kind === "REFUSE") {
    detail = "\n" + (ac.all_reasons || []).slice(0, 2).map(r => `→ ${r}`).join("\n");
  }
  document.getElementById("action_card").innerHTML = `[${ac.kind}] ${headline}${detail}`;
  // PnL
  var p = snap.pnl_panel || {};
  document.getElementById("pnl_daily").innerHTML = fmt_pnl(p.daily_pnl_rupees);
  document.getElementById("pnl_fees").innerHTML = `₹${(p.cumulative_fees_rupees||0).toLocaleString()}`;
  document.getElementById("pnl_win").innerHTML = p.win_rate != null ? fmt_pct(p.win_rate) : "—";
  // Web
  var w = snap.web_panel || {};
  document.getElementById("web_consensus").innerHTML =
    fmt_bar(w.directional_consensus||0, -1, 1) + " " + (w.directional_consensus||0).toFixed(2);
  document.getElementById("web_tail").innerHTML =
    fmt_bar(w.tail_mass||0, 0, 1) + " " + (w.tail_mass||0).toFixed(2);
  document.getElementById("web_chop").innerHTML =
    fmt_bar(w.chop_mass||0, 0, 1) + " " + (w.chop_mass||0).toFixed(2);
  document.getElementById("web_dom_strat").textContent = w.dominant_strategy_class || "—";
  var top = (w.top_scenarios || []).slice(0, 5).map(s =>
    `<div class="position-row">
       <strong>${s.name}</strong>
       <span class="badge ${s.direction > 0 ? "bull" : (s.direction < 0 ? "bear" : "")}">
         ${s.direction > 0 ? "↑ +1" : (s.direction < 0 ? "↓ -1" : "·  0")}
       </span>
       <span class="muted"> ${s.family} </span>
       <span class="green" style="float:right;">${(s.probability||0).toFixed(3)}</span>
     </div>`).join("");
  document.getElementById("web_scenarios").innerHTML = top || "<em class='muted'>none active</em>";
  // MM
  var mm = snap.mm_panel || {};
  document.getElementById("mm_intent").textContent = mm.dominant_intent || "—";
  document.getElementById("mm_prob").textContent = fmt_pct(mm.dominant_probability);
  document.getElementById("mm_bias").innerHTML = mm.implied_bias > 0
    ? '<span class="green">+1 ↑</span>'
    : mm.implied_bias < 0 ? '<span class="red">-1 ↓</span>'
                          : '<span class="muted">0 ·</span>';
  document.getElementById("mm_vol_view").textContent = mm.implied_volatility_view || "—";
  document.getElementById("mm_conf").textContent = fmt_pct(mm.confidence);
  document.getElementById("mm_guidance").textContent = mm.operator_guidance || "—";
  // Fat-tail
  var ft = snap.fat_tail_dial || {};
  document.getElementById("tail_score").innerHTML =
    fmt_bar(ft.tail_score||0, 0, 1) + " " + (ft.tail_score||0).toFixed(2);
  document.getElementById("tail_action").innerHTML = ft.action || "NORMAL";
  // Crowd
  var c = snap.crowd_panel || {};
  document.getElementById("crowd_retail").innerHTML = c.we_look_like_retail
    ? '<span class="red">YES — MM target risk</span>'
    : '<span class="green">no</span>';
  document.getElementById("crowd_score").textContent = (c.retail_similarity_score||0).toFixed(2);
  document.getElementById("crowd_ce").textContent = fmt_pct(c.crowd_density_long_atm_ce);
  document.getElementById("crowd_pe").textContent = fmt_pct(c.crowd_density_long_atm_pe);
  // Risk
  var r = snap.risk_panel || {};
  document.getElementById("risk_delta").textContent = (r.net_delta||0).toFixed(2);
  document.getElementById("risk_vega").textContent = (r.net_vega||0).toFixed(2);
  document.getElementById("risk_gamma").textContent = (r.net_gamma||0).toFixed(4);
  document.getElementById("risk_theta").textContent = "₹" + (r.net_theta||0).toFixed(2);
  document.getElementById("risk_par").innerHTML = `₹${(r.total_premium_at_risk_rupees||0).toLocaleString()}`;
  document.getElementById("risk_dd").textContent = (r.portfolio_drawdown_r||0).toFixed(2) + "R";
  var kills = (r.kill_switches || []).map(k => `<div class="red">⚠ ${k}</div>`).join("");
  document.getElementById("risk_kills").innerHTML = kills;
  // Positions
  var pos = snap.positions_panel || {};
  var openHtml = (pos.open || []).map(p =>
    `<div class="position-row">
       <strong>${p.contract_label}</strong>
       <span class="badge ${p.direction > 0 ? "bull" : "bear"}">${p.direction > 0 ? "long" : "short"} ${p.size_lots}L</span>
       last ₹${(p.last_premium||0).toFixed(2)}
       <span class="muted"> best ${(p.best_r||0).toFixed(2)}R worst ${(p.worst_r||0).toFixed(2)}R</span>
     </div>`).join("");
  document.getElementById("positions").innerHTML = openHtml || "<em class='muted'>no open positions</em>";
  // Patterns
  var pat = snap.patterns_panel || {};
  var pHtml = (pat.patterns || []).map(x =>
    `<div class="position-row">
       <strong>${x.name}</strong>
       <span class="badge">conf ${(x.confidence||0).toFixed(2)}</span>
       <span class="muted"> intent: ${x.implied_mm_intent}</span>
     </div>`).join("");
  document.getElementById("patterns").innerHTML = pHtml || "<em class='muted'>no patterns</em>";
}
// ── Control bar (LIVE TOGGLES) ────────────────────────────────
async function refreshControls() {
  try {
    const r = await fetch("/api/controls");
    const j = await r.json();
    const liveBtn = document.getElementById("btn-live-toggle");
    if (j.broker_is_live) {
      liveBtn.textContent = "● LIVE MONEY (click to stop)";
      liveBtn.className = "btn btn-live";
    } else {
      liveBtn.textContent = "PAPER MODE — click to GO LIVE";
      liveBtn.className = "btn btn-paper";
    }
    const calibBtn = document.getElementById("btn-calib-toggle");
    calibBtn.textContent = j.calibrator_paused ? "CALIB: PAUSED ▶ resume" : "CALIB: live ⏸ pause";
    document.getElementById("control-status").textContent =
      `broker=${j.broker_name} | killed=${j.broker_killed}`;
  } catch (e) { /* server not ready */ }
}
async function postControl(path, body) {
  try {
    const r = await fetch(path, {method: "POST",
      headers: {"Content-Type": "application/json"},
      body: body ? JSON.stringify(body) : "" });
    const j = await r.json();
    if (!r.ok) alert("control failed: " + JSON.stringify(j));
  } catch (e) { alert("control error: " + e); }
  refreshControls();
}
document.getElementById("btn-live-toggle").addEventListener("click", async () => {
  // Check current state to know which way to toggle.
  const r = await fetch("/api/controls"); const j = await r.json();
  if (j.broker_is_live) {
    if (!confirm("Disable real execution? Orders will revert to paper.")) return;
    await postControl("/api/execution/disable");
  } else {
    const phrase = prompt(
      "About to ENABLE REAL EXECUTION. Type EXACTLY:\n  I_UNDERSTAND_REAL_MONEY\n\nThis is your only safety guard.");
    if (phrase !== "I_UNDERSTAND_REAL_MONEY") {
      alert("not enabled — phrase did not match");
      return;
    }
    await postControl("/api/execution/enable", {confirm: "I_UNDERSTAND_REAL_MONEY"});
  }
});
document.getElementById("btn-kill").addEventListener("click", async () => {
  if (!confirm("EMERGENCY KILL — block all future order placements?")) return;
  await postControl("/api/execution/kill");
});
document.getElementById("btn-calib-toggle").addEventListener("click", async () => {
  const r = await fetch("/api/controls"); const j = await r.json();
  if (j.calibrator_paused) await postControl("/api/calibrator/resume");
  else await postControl("/api/calibrator/pause");
});
document.getElementById("btn-rollback").addEventListener("click", async () => {
  if (!confirm("Rollback the most recent applied weight change?")) return;
  await postControl("/api/calibrator/rollback");
});
refreshControls();
setInterval(refreshControls, 5000);

var es = new EventSource("/api/stream");
es.addEventListener("cockpit", e => {
  try { render(JSON.parse(e.data)); } catch (err) { console.error(err); }
});
es.addEventListener("error", () => {
  document.getElementById("status").innerHTML = '<span class="status-dot" style="background:#ff7d8e"></span>disconnected — retrying';
});
// Initial fetch in case SSE hasn't kicked in yet.
fetch("/api/cockpit").then(r => r.json()).then(j => {
  if (j && !j.empty) render(j);
});
</script>
</body>
</html>
"""
