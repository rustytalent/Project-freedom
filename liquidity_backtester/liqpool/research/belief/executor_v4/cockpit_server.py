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
        <div class="k">Consensus (raw)</div><div class="v" id="web_consensus">—</div>
        <div class="k">Consensus (horizon-wt)</div><div class="v" id="web_consensus_hw">—</div>
        <div class="k">Tail mass</div><div class="v" id="web_tail">—</div>
        <div class="k">Chop mass</div><div class="v" id="web_chop">—</div>
        <div class="k">Dominant strategy</div><div class="v" id="web_dom_strat">—</div>
      </div>
      <h2 style="margin-top:14px;">Top scenarios</h2>
      <div id="web_scenarios">—</div>
    </div>
    <div class="panel full">
      <h2>PULSING WEB (all active scenarios)</h2>
      <div class="muted" style="font-size: 11px; margin-bottom: 8px;">
        Bubble size = probability × √horizon.
        Color: <span class="green">bull</span> /
        <span class="red">bear</span> /
        <span style="color:#9ca8c0;">chop</span> /
        <span style="color:#c78aff;">tail-risk</span> /
        <span style="color:#fff19a;">manipulation</span>.
        Trail shows last 12 probability samples; growing = brightening.
      </div>
      <svg id="web_svg" width="100%" height="380" viewBox="0 0 1000 380"
           style="background: radial-gradient(circle at 50% 50%, #0c1320 0%, #050810 100%); border-radius: 6px;"></svg>
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
      <h2>LIVE CAPITAL (account)</h2>
      <div class="kv">
        <div class="k">Starting / Net</div><div class="v" id="cap_starting">—</div>
        <div class="k">Available cash</div><div class="v" id="cap_avail">—</div>
        <div class="k">Used margin</div><div class="v" id="cap_used">—</div>
        <div class="k">Current total</div><div class="v" id="cap_total">—</div>
        <div class="k">Daily P&amp;L (manager)</div><div class="v" id="cap_daily">—</div>
      </div>
    </div>
    <div class="panel full">
      <h2>OPEN POSITIONS</h2>
      <div id="positions">—</div>
    </div>
    <div class="panel full">
      <h2>LIVE TRADES (last 15)</h2>
      <div id="trades">—</div>
    </div>
    <div class="panel full">
      <h2>ACTIVE PATTERNS</h2>
      <div id="patterns">—</div>
    </div>
    <div class="panel">
      <h2>LATENCY (rolling 60 ticks)</h2>
      <div class="kv">
        <div class="k">Total p50</div><div class="v" id="lat_p50">—</div>
        <div class="k">Total p95</div><div class="v" id="lat_p95">—</div>
        <div class="k">Total max</div><div class="v" id="lat_max">—</div>
        <div class="k">Manager p95</div><div class="v" id="lat_mgr">—</div>
        <div class="k">Broker p95</div><div class="v" id="lat_brk">—</div>
        <div class="k">Persist p95</div><div class="v" id="lat_per">—</div>
      </div>
    </div>
    <div class="panel">
      <h2>SLIPPAGE TRACKER</h2>
      <div class="kv">
        <div class="k">Records</div><div class="v" id="slip_n">—</div>
        <div class="k">Mean bps</div><div class="v" id="slip_mean">—</div>
        <div class="k">Median bps</div><div class="v" id="slip_med">—</div>
        <div class="k">Worst bps</div><div class="v" id="slip_worst">—</div>
      </div>
    </div>
    <div class="panel full">
      <h2>STRATEGY ATTRIBUTION (closed trades)</h2>
      <div id="attribution">—</div>
    </div>
    <div class="panel full">
      <h2>DEV-OF-DEV — 22-SLOT dod_z HEATMAP</h2>
      <div class="muted" style="font-size: 11px; margin-bottom: 8px;">
        Per-slot dod_z (deviation-of-deviation). <span class="green">green</span> = unusually friendly (premium compressed below model),
        <span class="red">red</span> = unusually distorted (premium inflated). Centre column = ATM.
      </div>
      <svg id="dod_svg" width="100%" height="120" viewBox="0 0 1000 120"
           style="background: radial-gradient(circle at 50% 50%, #0c1320 0%, #050810 100%); border-radius: 6px;"></svg>
    </div>
    <div class="panel">
      <h2>SUBSTRATE — VELOCITIES &amp; REGIME</h2>
      <div class="kv">
        <div class="k">Regime stability</div><div class="v" id="sub_regime">—</div>
        <div class="k">Epicenter</div><div class="v" id="sub_epi">—</div>
        <div class="k">Epi migration</div><div class="v" id="sub_epi_mig">—</div>
        <div class="k">CE signed-z v / a</div><div class="v" id="sub_ce_va">—</div>
        <div class="k">PE signed-z v / a</div><div class="v" id="sub_pe_va">—</div>
        <div class="k">Net intent v / a</div><div class="v" id="sub_net_va">—</div>
        <div class="k">Thesis bull v</div><div class="v" id="sub_bull_v">—</div>
        <div class="k">Thesis bear v</div><div class="v" id="sub_bear_v">—</div>
        <div class="k">Dispersion velocity</div><div class="v" id="sub_disp_v">—</div>
        <div class="k">Dominant side</div><div class="v" id="sub_dom_side">—</div>
      </div>
    </div>
    <div class="panel">
      <h2>MULTI-TIMEFRAME ALIGNMENT</h2>
      <div class="kv">
        <div class="k">Long alignment</div><div class="v" id="mtf_long">—</div>
        <div class="k">Short alignment</div><div class="v" id="mtf_short">—</div>
      </div>
      <h2 style="margin-top:12px;">Per-TF bias</h2>
      <div id="mtf_per_tf">—</div>
    </div>
    <div class="panel full">
      <h2>LIVE CALIBRATION (last close)</h2>
      <div class="kv">
        <div class="k">Proposed / Applied</div><div class="v" id="calib_pa">—</div>
        <div class="k">Train / Val loss</div><div class="v" id="calib_loss">—</div>
        <div class="k">Rejected reason</div><div class="v" id="calib_reason">—</div>
      </div>
      <div id="calib_weights" style="margin-top: 8px;">—</div>
      <pre id="calib_notes" class="muted" style="margin-top: 6px;"></pre>
    </div>
    <div class="panel">
      <h2>WEIGHT EVOLUTION</h2>
      <div class="kv">
        <div class="k">Adaptability</div><div class="v" id="wevo_adapt">—</div>
        <div class="k">Snapshots seen</div><div class="v" id="wevo_n">—</div>
        <div class="k">Coord. drift</div><div class="v" id="wevo_drift">—</div>
        <div class="k">Val-loss trend</div><div class="v" id="wevo_vlt">—</div>
        <div class="k">Most drifting</div><div class="v" id="wevo_mdr">—</div>
      </div>
      <div id="wevo_warnings" class="red" style="margin-top: 6px;"></div>
    </div>
    <div class="panel">
      <h2>ADAPTIVE EXIT (portfolio coordinator)</h2>
      <div class="kv">
        <div class="k">Cluster bull / bear</div><div class="v" id="exit_cluster">—</div>
        <div class="k">Winning side</div><div class="v" id="exit_winning">—</div>
        <div class="k">Approved this tick</div><div class="v" id="exit_appr">—</div>
        <div class="k">Deferred this tick</div><div class="v" id="exit_def">—</div>
        <div class="k">Weakest thesis pos</div><div class="v" id="exit_weak">—</div>
      </div>
      <div id="exit_per_position" style="margin-top: 6px;">—</div>
      <pre id="exit_notes" class="muted" style="margin-top: 6px;"></pre>
    </div>
    <div class="panel">
      <h2>PROJECTION (intent vs realized)</h2>
      <div class="kv">
        <div class="k">Records</div><div class="v" id="proj_n">—</div>
        <div class="k">Divergence</div><div class="v" id="proj_div">—</div>
      </div>
      <div id="proj_recent" style="margin-top: 6px;">—</div>
    </div>
    <div class="panel full">
      <h2>HEDGE PROPOSAL</h2>
      <div class="kv">
        <div class="k">Proposed?</div><div class="v" id="hedge_proposed">—</div>
        <div class="k">Total premium</div><div class="v" id="hedge_premium">—</div>
      </div>
      <div id="hedge_proposals" style="margin-top: 6px;">—</div>
      <pre id="hedge_reasons" class="muted" style="margin-top: 6px;"></pre>
    </div>
    <div class="panel full">
      <h2>REHEARSAL ENSEMBLE — conditional-kNN off-policy evaluation</h2>
      <div class="muted" style="font-size: 11px; margin-bottom: 8px;">
        At each entry consideration we look at the K most-similar past trades and ask
        "what would the analogues say happens if we enter with these parameters?" Per-perturbation
        rows show alternate entry variants — when a row's mean R is meaningfully higher than
        as_proposed, the ensemble flags TUNE.
      </div>
      <div class="kv">
        <div class="k">Booted?</div><div class="v" id="reh_boot">—</div>
        <div class="k">Observations in ring</div><div class="v" id="reh_n">—</div>
        <div class="k">History days loaded</div><div class="v" id="reh_days">—</div>
        <div class="k">Rehearsal score</div><div class="v" id="reh_score">—</div>
        <div class="k">Confidence</div><div class="v" id="reh_conf">—</div>
        <div class="k">Analogues retrieved</div><div class="v" id="reh_anal">—</div>
        <div class="k">Action</div><div class="v" id="reh_act">—</div>
        <div class="k">Best perturbation</div><div class="v" id="reh_best">—</div>
      </div>
      <h2 style="margin-top:14px;">Per-perturbation outcome distribution</h2>
      <div id="reh_perturbations">—</div>
      <h2 style="margin-top:14px;">Top distance-metric weights</h2>
      <div id="reh_weights">—</div>
      <pre id="reh_notes" class="muted" style="margin-top: 6px;"></pre>
    </div>
    <div class="panel full">
      <h2>REGIME-AWARE CALIBRATION (yesterday → today)</h2>
      <div class="kv">
        <div class="k">Bootstrap ran?</div><div class="v" id="boot_ran">—</div>
        <div class="k">Days loaded</div><div class="v" id="boot_days">—</div>
        <div class="k">Observations replayed</div><div class="v" id="boot_obs">—</div>
        <div class="k">Updates applied at startup</div><div class="v" id="boot_upd">—</div>
        <div class="k">Today's dominant family</div><div class="v" id="reg_today">—</div>
        <div class="k">Confidence floor nudge</div><div class="v" id="reg_nudge">—</div>
      </div>
      <div id="reg_family_rows" style="margin-top: 8px;">—</div>
      <pre id="boot_notes" class="muted" style="margin-top: 6px;"></pre>
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
// ── Pulsing probability web (founder ask 2026-06-22) ─────────────
// SVG bubble chart: every active scenario is a node. Size = probability
// × √horizon. Color = family + direction. Glow trail = last few
// probability samples (rising = brightens; falling = dims).
function colorForScenario(sc) {
  var fam = sc.family || "";
  if (fam === "directional") {
    return sc.direction > 0 ? "#7dff8e" : (sc.direction < 0 ? "#ff7d8e" : "#cad3df");
  }
  if (fam === "chop") return "#9ca8c0";
  if (fam === "fat_tail") return "#c78aff";
  if (fam === "manipulation") return "#fff19a";
  if (fam === "neutral") return "#5dffd0";
  return "#cad3df";
}
function render_pulsing_web(scenarios) {
  var svg = document.getElementById("web_svg");
  if (!svg) return;
  // Clear existing children.
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!scenarios || scenarios.length === 0) {
    var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", 500); t.setAttribute("y", 190);
    t.setAttribute("text-anchor", "middle"); t.setAttribute("fill", "#5a6172");
    t.textContent = "no active scenarios";
    svg.appendChild(t);
    return;
  }
  // Spiral layout: place scenarios on a logarithmic spiral, sorted by
  // probability × horizon (the strongest sit centre, weakest on the rim).
  // Avoids overlap better than a grid for varying counts.
  var sorted = scenarios.slice().sort(function(a, b) {
    var sa = (a.probability||0) * Math.sqrt(a.horizon_bars||1);
    var sb = (b.probability||0) * Math.sqrt(b.horizon_bars||1);
    return sb - sa;
  });
  var cx = 500, cy = 190;
  var W = 1000, H = 380;
  for (var i = 0; i < sorted.length; i++) {
    var s = sorted[i];
    var prob = s.probability || 0;
    var h = s.horizon_bars || 1;
    var weight = prob * Math.sqrt(h);
    var radius = Math.max(8, Math.min(60, 6 + 60 * weight));
    // Spiral coords.
    var angle = i * 0.72;
    var r = 30 + 20 * Math.sqrt(i + 1);
    var x = cx + r * Math.cos(angle);
    var y = cy + r * Math.sin(angle) * 0.55;     // squash vertically
    x = Math.max(radius + 4, Math.min(W - radius - 4, x));
    y = Math.max(radius + 4, Math.min(H - radius - 4, y));
    var color = colorForScenario(s);
    // ── Glow trail from recent_probabilities ─────────────────
    var rp = s.recent_probabilities || [];
    var rising = rp.length >= 2 && rp[rp.length-1] > rp[0];
    var glow_factor = 1.0;
    if (rp.length >= 2) {
      var first = rp[0], last = rp[rp.length-1];
      var delta = last - first;
      glow_factor = 1.0 + 6 * Math.abs(delta);     // 1..3 typical
    }
    // Outer glow circle (only if probability is meaningful).
    if (prob >= 0.1) {
      var glow = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      glow.setAttribute("cx", x); glow.setAttribute("cy", y);
      glow.setAttribute("r", radius * 1.6 * glow_factor);
      glow.setAttribute("fill", color);
      glow.setAttribute("opacity", (0.10 + 0.20 * prob).toFixed(3));
      glow.setAttribute("filter", "blur(4px)");
      svg.appendChild(glow);
    }
    // Main bubble.
    var c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", x); c.setAttribute("cy", y);
    c.setAttribute("r", radius);
    c.setAttribute("fill", color);
    c.setAttribute("opacity", (0.30 + 0.55 * prob).toFixed(3));
    c.setAttribute("stroke", rising ? "#fff" : color);
    c.setAttribute("stroke-width", rising ? "2" : "1");
    svg.appendChild(c);
    // Label inside if bubble is big enough.
    if (radius >= 14) {
      var lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lbl.setAttribute("x", x); lbl.setAttribute("y", y - 2);
      lbl.setAttribute("text-anchor", "middle");
      lbl.setAttribute("fill", "#0a0e14");
      lbl.setAttribute("font-size", radius >= 24 ? "11" : "9");
      lbl.setAttribute("font-weight", "bold");
      lbl.textContent = (s.name || "").slice(0, 14);
      svg.appendChild(lbl);
      var pl = document.createElementNS("http://www.w3.org/2000/svg", "text");
      pl.setAttribute("x", x); pl.setAttribute("y", y + 10);
      pl.setAttribute("text-anchor", "middle");
      pl.setAttribute("fill", "#0a0e14");
      pl.setAttribute("font-size", "10");
      pl.textContent = prob.toFixed(2) + " · " + h + "b";
      svg.appendChild(pl);
    }
  }
}
// ── Dev-of-dev heatmap (founder restoration 2026-06-22) ─────────
// Renders the 22-slot dod_z heatmap as a horizontal strip. CE slots on
// the right, PE slots on the left (matches the strike ladder).
function render_dod_heatmap(values, labels) {
  var svg = document.getElementById("dod_svg");
  if (!svg) return;
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!values || values.length === 0) {
    var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", 500); t.setAttribute("y", 60);
    t.setAttribute("text-anchor", "middle"); t.setAttribute("fill", "#5a6172");
    t.textContent = "no slot readings";
    svg.appendChild(t);
    return;
  }
  var n = values.length;
  var W = 1000, H = 120;
  var pad = 20;
  var cellW = (W - 2*pad) / n;
  var cellH = 60;
  var y0 = 25;
  // Find max abs value for colour scaling.
  var maxAbs = 1.5;
  for (var i = 0; i < n; i++) maxAbs = Math.max(maxAbs, Math.abs(values[i]));
  for (var i = 0; i < n; i++) {
    var v = values[i];
    var norm = Math.max(-1, Math.min(1, v / maxAbs));
    // Diverging palette: positive = red (distorted up), negative = green (friendly).
    var r, g, b;
    if (norm >= 0) {
      r = Math.round(20 + 235 * norm); g = Math.round(120 - 60 * norm); b = Math.round(140 - 60 * norm);
    } else {
      r = Math.round(20 - 0 * norm); g = Math.round(120 + 135 * (-norm)); b = Math.round(140 - 50 * (-norm));
    }
    var color = "rgb(" + r + "," + g + "," + b + ")";
    var x = pad + i * cellW;
    var cell = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    cell.setAttribute("x", x); cell.setAttribute("y", y0);
    cell.setAttribute("width", cellW - 1); cell.setAttribute("height", cellH);
    cell.setAttribute("fill", color);
    cell.setAttribute("rx", 2);
    svg.appendChild(cell);
    // Label below.
    var lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
    lbl.setAttribute("x", x + cellW/2); lbl.setAttribute("y", y0 + cellH + 14);
    lbl.setAttribute("text-anchor", "middle"); lbl.setAttribute("fill", "#5a6172");
    lbl.setAttribute("font-size", "9");
    lbl.textContent = (labels && labels[i]) ? labels[i] : "";
    svg.appendChild(lbl);
    // Value inside the cell when cell is wide enough.
    if (cellW >= 28) {
      var val = document.createElementNS("http://www.w3.org/2000/svg", "text");
      val.setAttribute("x", x + cellW/2); val.setAttribute("y", y0 + cellH/2 + 4);
      val.setAttribute("text-anchor", "middle"); val.setAttribute("fill", "#0a0e14");
      val.setAttribute("font-size", "10");
      val.setAttribute("font-weight", "bold");
      val.textContent = v.toFixed(1);
      svg.appendChild(val);
    }
  }
}
function fmt_v(v, digits) {
  if (v == null) return "—";
  digits = digits == null ? 3 : digits;
  var n = Number(v);
  var sign = n > 0 ? "+" : "";
  var cls = n > 0 ? "green" : (n < 0 ? "red" : "muted");
  return '<span class="' + cls + '">' + sign + n.toFixed(digits) + '</span>';
}
function render_substrate(sub) {
  var s = sub || {};
  document.getElementById("sub_regime").innerHTML =
    fmt_bar(s.regime_stability_index||0, 0, 1) + " " + (s.regime_stability_index||0).toFixed(2);
  document.getElementById("sub_epi").textContent =
    (s.epicenter_label || "—") + " (lvl " + (s.epicenter_level || 0) + ")";
  document.getElementById("sub_epi_mig").textContent = (s.epicenter_migration_distance||0).toFixed(2);
  document.getElementById("sub_ce_va").innerHTML =
    fmt_v(s.ce_signed_z_velocity, 3) + " / " + fmt_v(s.ce_signed_z_acceleration, 3);
  document.getElementById("sub_pe_va").innerHTML =
    fmt_v(s.pe_signed_z_velocity, 3) + " / " + fmt_v(s.pe_signed_z_acceleration, 3);
  document.getElementById("sub_net_va").innerHTML =
    fmt_v(s.net_intent_velocity, 3) + " / " + fmt_v(s.net_intent_acceleration, 3);
  document.getElementById("sub_bull_v").innerHTML = fmt_v(s.thesis_bull_velocity, 2);
  document.getElementById("sub_bear_v").innerHTML = fmt_v(s.thesis_bear_velocity, 2);
  document.getElementById("sub_disp_v").innerHTML = fmt_v(s.dispersion_velocity, 3);
  var dom = s.thesis_velocity_dominant_side || "flat";
  var domCls = dom === "bull" ? "green" : (dom === "bear" ? "red" : "muted");
  document.getElementById("sub_dom_side").innerHTML =
    '<span class="' + domCls + '">' + dom + '</span>';
}
function render_mtf(mtf) {
  var m = mtf || {};
  var okL = m.alignment_ok_long;
  var okS = m.alignment_ok_short;
  document.getElementById("mtf_long").innerHTML =
    '<span class="' + (okL ? "green" : "red") + '">' + (okL ? "OK" : "block") + '</span> '
    + fmt_bar(m.alignment_score_long||0, 0, 1) + " " + (m.alignment_score_long||0).toFixed(2)
    + ' <span class="muted">(' + (m.confirmation_count_long||0) + '/3 confirm)</span>';
  document.getElementById("mtf_short").innerHTML =
    '<span class="' + (okS ? "green" : "red") + '">' + (okS ? "OK" : "block") + '</span> '
    + fmt_bar(m.alignment_score_short||0, 0, 1) + " " + (m.alignment_score_short||0).toFixed(2)
    + ' <span class="muted">(' + (m.confirmation_count_short||0) + '/3 confirm)</span>';
  var per = m.per_timeframe || {};
  var rows = ["L1","L5","L15","L60"].map(function(level){
    var view = per[level] || {};
    var bias = Number(view.direction_bias || 0);
    var biasCls = bias > 0 ? "bull" : (bias < 0 ? "bear" : "");
    var biasTxt = bias > 0 ? "↑ +1" : (bias < 0 ? "↓ -1" : "·  0");
    return '<div class="position-row"><strong>' + level + '</strong>'
      + ' <span class="badge ' + biasCls + '">' + biasTxt + '</span>'
      + ' <span class="muted">' + (view.dominant_thesis || "") + ' · ' + (view.dominant_battlefield || "") + '</span>'
      + ' <span class="muted" style="float:right">n=' + (view.n_samples||0) + '</span></div>';
  }).join("");
  document.getElementById("mtf_per_tf").innerHTML = rows;
}
function render_calibration(calib) {
  var c = calib || {};
  if (!c.has_event) {
    document.getElementById("calib_pa").innerHTML = '<span class="muted">no event yet</span>';
    document.getElementById("calib_loss").textContent = "—";
    document.getElementById("calib_reason").textContent = "—";
    document.getElementById("calib_weights").innerHTML = "";
    document.getElementById("calib_notes").textContent = "";
    return;
  }
  var paTxt = '<span class="' + (c.applied ? "green" : "muted") + '">'
    + (c.applied ? "APPLIED" : (c.proposed ? "proposed" : "no-op")) + '</span>';
  if (c.paused) paTxt += ' <span class="yellow">[paused]</span>';
  document.getElementById("calib_pa").innerHTML = paTxt;
  document.getElementById("calib_loss").textContent =
    "train " + Number(c.train_loss||0).toFixed(4) + " / val " + Number(c.val_loss||0).toFixed(4);
  document.getElementById("calib_reason").textContent = c.rejected_reason || "—";
  var pre = c.pre_weights || {}, post = c.post_weights || {}, d = c.deltas || {};
  var keys = Object.keys(pre);
  var html = '<table style="width:100%;font-size:12px;border-collapse:collapse;">'
    + '<tr style="color:#5a6172;text-align:left;"><th>weight</th><th>pre</th><th>post</th><th>Δ</th></tr>';
  keys.forEach(function(k){
    var p1 = Number(pre[k]||0), p2 = Number(post[k]||0), dd = Number(d[k]||0);
    var ddCls = dd > 0 ? "green" : (dd < 0 ? "red" : "muted");
    html += '<tr><td>' + k + '</td>'
      + '<td>' + p1.toFixed(3) + '</td>'
      + '<td>' + p2.toFixed(3) + '</td>'
      + '<td class="' + ddCls + '">' + (dd >= 0 ? "+" : "") + dd.toFixed(4) + '</td></tr>';
  });
  html += '</table>';
  document.getElementById("calib_weights").innerHTML = html;
  document.getElementById("calib_notes").textContent = (c.notes || []).join("\n");
}
function render_weight_evolution(wevo) {
  var w = wevo || {};
  document.getElementById("wevo_adapt").innerHTML =
    fmt_bar(w.adaptability_index||0, 0, 1) + " " + (w.adaptability_index||0).toFixed(2);
  document.getElementById("wevo_n").textContent = w.n_snapshots || 0;
  document.getElementById("wevo_drift").innerHTML = fmt_v(w.coordinated_drift_score, 3);
  document.getElementById("wevo_vlt").innerHTML = fmt_v(w.val_loss_trend, 4);
  document.getElementById("wevo_mdr").textContent = w.most_drifting_weight || "—";
  document.getElementById("wevo_warnings").innerHTML =
    (w.warnings || []).map(function(x){ return "⚠ " + x; }).join("<br>");
}
function render_exit_decision(ex) {
  var e = ex || {};
  if (!e.has_decision) {
    document.getElementById("exit_cluster").innerHTML = '<span class="muted">no held position</span>';
    document.getElementById("exit_winning").textContent = "—";
    document.getElementById("exit_appr").textContent = "—";
    document.getElementById("exit_def").textContent = "—";
    document.getElementById("exit_weak").textContent = "—";
    document.getElementById("exit_per_position").innerHTML = "";
    document.getElementById("exit_notes").textContent = "";
    return;
  }
  document.getElementById("exit_cluster").innerHTML =
    '<span class="green">' + (e.cluster_bullish_count||0) + '</span> / '
    + '<span class="red">' + (e.cluster_bearish_count||0) + '</span>';
  var ws = e.regime_winning_side;
  document.getElementById("exit_winning").innerHTML =
    ws > 0 ? '<span class="green">↑ bull</span>'
    : (ws < 0 ? '<span class="red">↓ bear</span>' : '<span class="muted">·</span>');
  document.getElementById("exit_appr").textContent =
    (e.modifications_approved_this_tick || []).map(function(p){return p.slice(0,8);}).join(", ") || "—";
  document.getElementById("exit_def").textContent =
    (e.modifications_deferred_this_tick || []).map(function(p){return p.slice(0,8);}).join(", ") || "—";
  document.getElementById("exit_weak").textContent =
    (e.weakest_thesis_position_id || "").slice(0,8) || "—";
  var rows = (e.per_position || []).slice(0, 8).map(function(pp){
    var act = pp.action || pp.mode || "";
    var actCls = String(act).indexOf("KILL") >= 0 || String(act).indexOf("EXIT") >= 0 ? "red"
                : (String(act).indexOf("HOLD") >= 0 ? "green" : "yellow");
    return '<div class="position-row"><strong class="' + actCls + '">' + act + '</strong> '
      + ' <span class="muted">' + (pp.position_id || "").slice(0,8) + '</span>'
      + ' <span class="muted">' + (pp.reason || pp.note || "") + '</span></div>';
  }).join("");
  document.getElementById("exit_per_position").innerHTML = rows;
  document.getElementById("exit_notes").textContent = (e.notes || []).join("\n");
}
function render_projection(p) {
  var pp = p || {};
  document.getElementById("proj_n").textContent = pp.tape_size || 0;
  var mr = Number(pp.mean_realized_r||0);
  var mrCls = mr > 0 ? "green" : (mr < 0 ? "red" : "muted");
  document.getElementById("proj_div").innerHTML =
    '<span class="' + mrCls + '">' + (mr>=0?"+":"") + mr.toFixed(2) + 'R avg</span>';
  document.getElementById("proj_recent").innerHTML =
    '<div class="position-row">'
    + '<span class="green">target ' + (pp.n_target_hits || 0) + '</span> · '
    + '<span class="red">stop ' + (pp.n_stop_hits || 0) + '</span> · '
    + '<span class="muted">neither ' + (pp.n_neither || 0) + '</span>'
    + '</div>';
}
function render_hedge(h) {
  var hh = h || {};
  document.getElementById("hedge_proposed").innerHTML =
    hh.proposed ? '<span class="yellow">yes — review</span>' : '<span class="muted">no</span>';
  document.getElementById("hedge_premium").textContent = "₹" + Number(hh.total_premium_rupees||0).toLocaleString("en-IN", {maximumFractionDigits: 0});
  var rows = (hh.proposals || []).map(function(p){
    return '<div class="position-row"><strong>' + (p.contract_label || p.label || "") + '</strong> '
      + ' <span class="badge">' + (p.kind || p.side || "") + '</span>'
      + ' <span class="muted">×' + (p.size_lots || 1) + 'L</span>'
      + ' <span class="muted" style="float:right">@ ₹' + Number(p.entry_premium||p.premium||0).toFixed(2) + '</span></div>';
  }).join("");
  document.getElementById("hedge_proposals").innerHTML = rows || '<em class="muted">none</em>';
  document.getElementById("hedge_reasons").textContent = (hh.reasons || []).join("\n");
}
function render_bootstrap(boot, regime) {
  var b = boot || {}, r = regime || {};
  document.getElementById("boot_ran").innerHTML =
    b.ran ? '<span class="green">yes</span>' : '<span class="muted">no</span>';
  document.getElementById("boot_days").textContent = b.days_loaded || 0;
  document.getElementById("boot_obs").textContent = b.observations_replayed || 0;
  document.getElementById("boot_upd").textContent = b.updates_applied || 0;
  document.getElementById("reg_today").textContent = r.today_dominant_family || "—";
  var adj = Number(r.today_confidence_adjustment || 0);
  var adjCls = adj > 0 ? "red" : (adj < 0 ? "green" : "muted");
  document.getElementById("reg_nudge").innerHTML =
    '<span class="' + adjCls + '">' + (adj > 0 ? "+" : "") + adj.toFixed(2) + '</span>'
    + ' <span class="muted">(positive = tighter floor; negative = looser)</span>';
  var byFam = r.by_family || {};
  var rows = Object.keys(byFam).map(function(fam){
    var v = byFam[fam] || {};
    var avgR = Number(v.avg_realized_r||0);
    var avgCls = avgR > 0 ? "green" : (avgR < 0 ? "red" : "muted");
    return '<div class="position-row"><strong>' + fam + '</strong>'
      + ' <span class="muted">n=' + (v.n_trades||0) + ' wins=' + (v.wins||0) + '/' + (v.losses||0) + '</span>'
      + ' <span class="muted">win-rate ' + ((v.win_rate||0)*100).toFixed(1) + '%</span>'
      + ' <span class="' + avgCls + '" style="float:right">avgR ' + (avgR>=0?"+":"") + avgR.toFixed(2) + '</span></div>';
  }).join("");
  document.getElementById("reg_family_rows").innerHTML = rows || '<em class="muted">no history yet (first day)</em>';
  document.getElementById("boot_notes").textContent =
    ((b.notes || []).concat(r.notes || [])).join("\n");
}
function render_rehearsal(rh) {
  var r = rh || {};
  document.getElementById("reh_boot").innerHTML =
    r.booted ? '<span class="green">yes</span>' : '<span class="muted">no</span>';
  document.getElementById("reh_n").textContent = r.n_observations_in_ring || 0;
  document.getElementById("reh_days").textContent = r.history_days_loaded || 0;
  if (!r.has_decision || !r.ran) {
    document.getElementById("reh_score").innerHTML = '<span class="muted">no decision yet</span>';
    document.getElementById("reh_conf").textContent = "—";
    document.getElementById("reh_anal").textContent = "—";
    document.getElementById("reh_act").textContent = r.deferral_reason || "—";
    document.getElementById("reh_best").textContent = "—";
    document.getElementById("reh_perturbations").innerHTML =
      '<em class="muted">no rehearsal yet (need ≥ 12 observations + entry consideration)</em>';
  } else {
    var score = Number(r.rehearsal_score || 0.50);
    var cls = score > 0.55 ? "green" : (score < 0.45 ? "red" : "muted");
    document.getElementById("reh_score").innerHTML =
      fmt_bar(score, 0, 1) + ' <span class="' + cls + '">' + score.toFixed(2) + '</span>';
    document.getElementById("reh_conf").innerHTML = fmt_bar(r.confidence || 0, 0, 1) + " " + (r.confidence||0).toFixed(2);
    document.getElementById("reh_anal").textContent = r.n_analogues_total + " (mean similarity " + (r.mean_similarity||0).toFixed(2) + ")";
    var act = r.recommended_action || "PROCEED";
    var actCls = act === "DEFER" ? "red" : (act === "TUNE" ? "yellow" : "green");
    document.getElementById("reh_act").innerHTML = '<span class="' + actCls + '">' + act + '</span>';
    document.getElementById("reh_best").innerHTML =
      '<strong>' + (r.best_perturbation_name || "—") + '</strong>'
      + ' <span class="muted">(current: ' + (r.current_perturbation_name || "as_proposed") + ')</span>';
    // Per-perturbation rows.
    var rows = (r.per_perturbation || []).map(function(p){
      var mean = Number(p.mean_r||0);
      var meanCls = mean > 0.10 ? "green" : (mean < -0.10 ? "red" : "muted");
      var pProf = Number(p.p_profit||0);
      var profCls = pProf > 0.55 ? "green" : (pProf < 0.45 ? "red" : "muted");
      var isBest = p.name === r.best_perturbation_name;
      var isCurr = p.name === r.current_perturbation_name;
      var prefix = (isBest ? "★ " : "") + (isCurr ? "● " : "");
      return '<div class="position-row">'
        + '<strong>' + prefix + (p.name || "") + '</strong>'
        + ' <span class="muted">n=' + (p.n_analogues||0) + '</span>'
        + ' <span class="' + meanCls + '"> mean R ' + (mean>=0?"+":"") + mean.toFixed(2) + '</span>'
        + ' <span class="muted"> ±' + Number(p.std_r||0).toFixed(2) + '</span>'
        + ' <span class="' + profCls + '"> P(profit) ' + (pProf*100).toFixed(0) + '%</span>'
        + ' <span class="muted" style="float:right">target ' + ((p.p_target||0)*100).toFixed(0) + '% / stop ' + ((p.p_stop||0)*100).toFixed(0) + '%</span>'
        + (p.notes ? '<div class="muted" style="margin-left:18px;font-size:11px;">' + p.notes + '</div>' : '')
        + '</div>';
    }).join("");
    document.getElementById("reh_perturbations").innerHTML = rows;
  }
  // Feature weights — show the top 8 by magnitude (most informative
  // features under the current outcome correlation).
  var fw = r.feature_weights || {};
  var ents = Object.keys(fw).map(function(k){ return [k, Number(fw[k] || 0)]; });
  ents.sort(function(a,b){ return b[1] - a[1]; });
  var top = ents.slice(0, 8);
  document.getElementById("reh_weights").innerHTML = top.map(function(kv){
    return '<div class="position-row"><strong>' + kv[0] + '</strong>'
      + ' <span class="muted" style="float:right">' + kv[1].toFixed(2) + '</span></div>';
  }).join("") || '<em class="muted">weights not yet computed (needs ≥ 30 obs)</em>';
  document.getElementById("reh_notes").textContent = (r.notes || []).join("\n");
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
  document.getElementById("web_consensus_hw").innerHTML =
    fmt_bar(w.directional_consensus_horizon_weighted||0, -1, 1)
    + " " + (w.directional_consensus_horizon_weighted||0).toFixed(2);
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
  render_pulsing_web(w.all_scenarios || []);
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
  // ── Live capital panel ─────────────────────────────────────
  var cap = snap.capital_panel || {};
  document.getElementById("cap_starting").textContent = "₹" + (cap.starting_capital_rupees || 0).toLocaleString("en-IN");
  document.getElementById("cap_avail").textContent = "₹" + (cap.available_rupees || 0).toLocaleString("en-IN");
  document.getElementById("cap_used").textContent = "₹" + (cap.used_margin_rupees || 0).toLocaleString("en-IN");
  document.getElementById("cap_total").textContent = "₹" + (cap.current_total_rupees || 0).toLocaleString("en-IN");
  document.getElementById("cap_daily").innerHTML = fmt_pnl(cap.daily_pnl_rupees);
  // ── Live trades tape ───────────────────────────────────────
  var tradesObj = snap.trades_panel || {};
  var tradesHtml = (tradesObj.trades || []).map(t => {
    var dir = t.direction > 0 ? "long" : (t.direction < 0 ? "short" : "·");
    var kindLabel = t.kind === "OPEN" ? "→ OPEN" : "← CLOSE";
    var kindColor = t.kind === "OPEN" ? "green" : "yellow";
    if (t.kind === "CLOSE") {
      return `<div class="position-row">
        <strong class="${kindColor}">${kindLabel}</strong>
        ${t.contract_label} ${dir} ${t.size_lots}L
        @ ₹${(t.exit_premium||0).toFixed(2)}
        <span class="muted">[${(t.realized_r||0).toFixed(2)}R | ${fmt_pnl(t.realized_rupees)}]</span>
        <span class="muted" style="float:right">${t.exit_reason || ""}</span>
      </div>`;
    }
    return `<div class="position-row">
      <strong class="${kindColor}">${kindLabel}</strong>
      ${t.contract_label} ${dir} ${t.size_lots}L
      @ ₹${(t.entry_premium||0).toFixed(2)}
      <span class="muted" style="float:right">strategy: ${t.strategy || ""}</span>
    </div>`;
  }).join("");
  document.getElementById("trades").innerHTML = tradesHtml || "<em class='muted'>no recent trades</em>";
  // ── Latency panel ──────────────────────────────────────────
  var lat = snap.latency_panel || {};
  function fmt_ms(v) {
    if (v == null) return "—";
    var n = Number(v);
    var cls = n > 1000 ? "red" : (n > 500 ? "yellow" : "green");
    return `<span class="${cls}">${n.toFixed(1)} ms</span>`;
  }
  document.getElementById("lat_p50").innerHTML = fmt_ms(lat.total_ms_p50);
  document.getElementById("lat_p95").innerHTML = fmt_ms(lat.total_ms_p95);
  document.getElementById("lat_max").innerHTML = fmt_ms(lat.total_ms_max);
  document.getElementById("lat_mgr").innerHTML = fmt_ms(lat.manager_ms_p95);
  document.getElementById("lat_brk").innerHTML = fmt_ms(lat.broker_ms_p95);
  document.getElementById("lat_per").innerHTML = fmt_ms(lat.persist_ms_p95);
  // ── Slippage tracker ───────────────────────────────────────
  var slip = (snap.attribution_panel || {}).slippage_tracker || {};
  document.getElementById("slip_n").textContent = slip.n_records || 0;
  document.getElementById("slip_mean").textContent = slip.mean_bps != null ? Number(slip.mean_bps).toFixed(1) : "—";
  document.getElementById("slip_med").textContent = slip.median_bps != null ? Number(slip.median_bps).toFixed(1) : "—";
  document.getElementById("slip_worst").textContent = slip.worst_bps != null ? Number(slip.worst_bps).toFixed(1) : "—";
  // ── Strategy attribution panel ─────────────────────────────
  var att = snap.attribution_panel || {};
  var attRows = (att.rows || []).slice(0, 10).map(r => {
    var pnl = Number(r.total_realized_rupees || 0);
    var pnlClass = pnl > 0 ? "green" : (pnl < 0 ? "red" : "muted");
    var multClass = r.size_multiplier_now < 1 ? "red" : (r.size_multiplier_now > 1 ? "green" : "muted");
    return `<div class="position-row">
      <strong>${r.strategy}</strong>
      <span class="muted">n=${r.n_trades} wins=${r.wins}/${r.losses}</span>
      <span class="muted"> avgR ${(r.avg_realized_r||0).toFixed(2)}</span>
      <span class="${pnlClass}" style="float:right">
        ${pnl >= 0 ? "+" : ""}₹${pnl.toLocaleString("en-IN", {maximumFractionDigits: 0})}
        <span class="${multClass}">[size × ${(r.size_multiplier_now||1).toFixed(2)}]</span>
      </span>
    </div>`;
  }).join("");
  document.getElementById("attribution").innerHTML = attRows || "<em class='muted'>no closed trades yet</em>";
  // ── Restoration panels (founder 2026-06-22) ──────────────────
  var dod = snap.dod_heatmap_panel || {};
  render_dod_heatmap(dod.values || [], dod.labels || []);
  render_substrate(snap.substrate_panel || {});
  render_mtf(snap.mtf_panel || {});
  render_calibration(snap.calibration_panel || {});
  render_weight_evolution(snap.weight_evolution_panel || {});
  render_exit_decision(snap.exit_decision_panel || {});
  render_projection(snap.projection_panel || {});
  render_hedge(snap.hedge_panel || {});
  render_bootstrap(snap.bootstrap_panel || {}, snap.regime_history_panel || {});
  render_rehearsal(snap.rehearsal_panel || {});
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
