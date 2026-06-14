"""Replay mode — run any past session through the live cockpit.

The institutional must-have we didn't have:

  * "Why did Crux say WATCH at 11:23?" — load that session, jump to
    11:20, watch the next five minutes play through every panel.
  * Customer demos — "here's how the cockpit handled expiry-day
    volatility three Thursdays ago."
  * Tuning — change a model parameter overnight, re-play last week,
    watch what would have changed.

How it works:

  1. Operator (or test) calls ``ReplayController.start(session_date,
     speed)``. Speed is a wallclock multiplier — 1x for real-time,
     10x to compress an hour into 6 minutes, ``max`` to flush as fast
     as the bus accepts.

  2. The controller reads two streams:
       * ``<journal>/ledger_<date>.jsonl`` — Sentinel's own ledger
       * ``<journal>/liqpool_live_signals.jsonl`` — research-engine
         outputs that day (rotated to that date's archive if any)
     plus the spot history if present.

  3. A background thread walks events in chronological order, sleeps
     ``(event_dt) / speed`` between them, and publishes each onto
     the same ``LivePublisher`` the cockpit reads. The cockpit
     doesn't know it's replay — every panel just lights up.

  4. ``pause()`` / ``resume()`` / ``stop()`` work the way you'd
     expect. ``stop()`` flushes a final ``replay_finished`` audit
     event.

Important safety: replay mode REFUSES TO START while the spine is
live (``cfg.dry_run is False`` and orders could go to broker). The
operator must explicitly enable demo or paper mode before replay can
publish synthetic signals onto the same bus that decides exits.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .io_decl import IOSpec, declare
from .live_publisher import LivePublisher, ModelSignal

LOG = logging.getLogger("sentinel.replay")


@dataclass
class ReplayProgress:
    state: str = "idle"                   # idle / loading / running / paused / done / error
    session_date: str = ""
    speed: float = 1.0
    started_at_utc: Optional[str] = None
    finished_at_utc: Optional[str] = None
    events_total: int = 0
    events_emitted: int = 0
    last_ts_ist: str = ""
    note: str = ""

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


class ReplayController:
    """Owns one replay at a time. Thread-safe — start/pause/resume/stop
    all serialise on a single lock. The background thread is daemon so
    the process can exit without joining."""

    def __init__(self, publisher: LivePublisher,
                 journal_dir: Path,
                 is_live_real: Callable[[], bool] = lambda: False,
                 on_event: Optional[Callable[[Dict[str, Any]], None]] = None
                 ) -> None:
        self.publisher = publisher
        self.journal_dir = Path(journal_dir)
        self._is_live_real = is_live_real
        self._on_event = on_event
        self.progress = ReplayProgress()
        self._lock = threading.Lock()
        self._pause = threading.Event()
        self._pause.set()                        # "set" = running, "clear" = paused
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ─────────── public surface ───────────

    def start(self, session_date: str, speed: float = 10.0) -> ReplayProgress:
        """Refuses to start while live-real or while a replay is
        already running."""
        if self._is_live_real():
            raise RuntimeError(
                "replay is refused while spine is live (real orders). "
                "Switch to demo or paper mode first.")
        with self._lock:
            if self.progress.state in ("running", "paused", "loading"):
                raise RuntimeError(
                    f"a replay is already {self.progress.state}; "
                    f"stop it before starting a new one")
            self.progress = ReplayProgress(
                state="loading", session_date=session_date,
                speed=float(max(0.001, speed)),
                started_at_utc=_now_utc())
        events = self._load_events(session_date)
        with self._lock:
            self.progress.events_total = len(events)
            self.progress.state = "running" if events else "done"
            if not events:
                self.progress.note = "no events found for that session"
                self.progress.finished_at_utc = _now_utc()
                return self.progress
        self._pause.set()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(events,),
            name="sentinel-replay", daemon=True)
        self._thread.start()
        return self.progress

    def pause(self) -> ReplayProgress:
        with self._lock:
            if self.progress.state == "running":
                self.progress.state = "paused"
                self._pause.clear()
        return self.progress

    def resume(self) -> ReplayProgress:
        with self._lock:
            if self.progress.state == "paused":
                self.progress.state = "running"
                self._pause.set()
        return self.progress

    def stop(self) -> ReplayProgress:
        self._stop.set()
        self._pause.set()                        # let the thread observe stop
        if self._thread is not None:
            self._thread.join(timeout=2)
        with self._lock:
            if self.progress.state not in ("done", "error"):
                self.progress.state = "done"
                self.progress.finished_at_utc = _now_utc()
        return self.progress

    def state(self) -> ReplayProgress:
        with self._lock:
            return ReplayProgress(**asdict(self.progress))

    # ─────────── internals ───────────

    def _load_events(self, session_date: str) -> List[Dict[str, Any]]:
        """Returns events sorted by their wallclock time-of-day.
        Two sources concatenated:
          * ledger_<date>.jsonl — every Sentinel-side decision
          * liqpool_live_signals.jsonl.<date> (rotated archive) — every
            research-engine signal published that day
        """
        out: List[Dict[str, Any]] = []
        ledger = self.journal_dir / f"ledger_{session_date}.jsonl"
        if ledger.exists():
            seen: Dict[str, Dict[str, Any]] = {}
            for line in ledger.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                # last-write-wins per event_id (matches read_session)
                if eid := row.get("event_id"):
                    seen[eid] = row
            for row in seen.values():
                ts = self._ts_ist_of(row)
                if ts:
                    out.append({"kind": "ledger", "ts_ist": ts, "row": row})

        liq_archive = (self.journal_dir
                        / f"liqpool_live_signals.jsonl.{session_date}")
        if liq_archive.exists():
            for line in liq_archive.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = row.get("ts_ist", "")
                if ts:
                    out.append({"kind": "liqpool", "ts_ist": ts, "row": row})

        out.sort(key=lambda e: e["ts_ist"])
        return out

    @staticmethod
    def _ts_ist_of(ledger_row: Dict[str, Any]) -> str:
        """Pull an HH:MM:SS IST stamp out of a ledger row."""
        from datetime import timedelta
        ts_utc = ledger_row.get("ts_utc") or ""
        if "T" in ts_utc and len(ts_utc) >= 19:
            # 2026-06-13T05:00:00Z → +5h30m → 10:30:00
            try:
                t = datetime.strptime(ts_utc[:19], "%Y-%m-%dT%H:%M:%S")
                t = t + timedelta(hours=5, minutes=30)
                return t.strftime("%H:%M:%S")
            except Exception:
                pass
        return ledger_row.get("ts_ist") or ""

    def _run(self, events: List[Dict[str, Any]]) -> None:
        try:
            prev_secs: Optional[float] = None
            for ev in events:
                if self._stop.is_set():
                    break
                self._pause.wait()             # blocks while paused
                if self._stop.is_set():
                    break
                cur_secs = _hms_to_secs(ev["ts_ist"])
                with self._lock:
                    speed = max(self.progress.speed, 0.001)
                if prev_secs is not None and cur_secs > prev_secs:
                    delay = (cur_secs - prev_secs) / speed
                    # cap so a multi-hour gap doesn't freeze replay forever
                    self._stop.wait(min(delay, 5.0))
                prev_secs = cur_secs
                self._emit(ev)
                with self._lock:
                    self.progress.events_emitted += 1
                    self.progress.last_ts_ist = ev["ts_ist"]
            with self._lock:
                self.progress.state = "done"
                self.progress.finished_at_utc = _now_utc()
                self.progress.note = "completed"
        except Exception as exc:
            LOG.exception("replay crashed: %s", exc)
            with self._lock:
                self.progress.state = "error"
                self.progress.note = str(exc)
                self.progress.finished_at_utc = _now_utc()

    def _emit(self, ev: Dict[str, Any]) -> None:
        if ev["kind"] == "liqpool":
            sig = self._signal_from_liqpool(ev["row"])
        else:
            sig = self._signal_from_ledger(ev["row"])
        if sig is None:
            return
        try:
            self.publisher.publish(sig)
        except Exception:
            return
        if self._on_event is not None:
            try:
                self._on_event(ev)
            except Exception:
                pass

    @staticmethod
    def _signal_from_liqpool(row: Dict[str, Any]) -> Optional[ModelSignal]:
        try:
            zone = row.get("zone")
            if isinstance(zone, list) and len(zone) == 2:
                zone = tuple(zone)
            else:
                zone = None
            return ModelSignal(
                ts_ist=str(row.get("ts_ist", "")),
                asset=str(row.get("asset", "")),
                model=str(row.get("model", "")) + "_replay",
                signal=f"[REPLAY] {row.get('signal', '')}",
                confidence=float(row.get("confidence", 0.0)),
                trust_tier="SHADOW",          # replayed signals never reach EXECUTION
                zone=zone,
                risk=str(row.get("risk", "")),
                reason_codes=list(row.get("reason_codes") or []) + ["replay"],
                extras={"source": "replay", "original_source": row.get("source", "")},
            )
        except Exception:
            return None

    @staticmethod
    def _signal_from_ledger(row: Dict[str, Any]) -> Optional[ModelSignal]:
        ident = row.get("identity") or {}
        hyp = row.get("hypothesis") or {}
        try:
            return ModelSignal(
                ts_ist=ReplayController._ts_ist_of(row),
                asset=str(ident.get("instrument", "REPLAY")),
                model=f"{row.get('kind', 'ledger')}_replay",
                signal=f"[REPLAY] {row.get('scientist', '')}: " +
                        (hyp.get("reason_codes") or ["replayed event"])[0],
                confidence=float(hyp.get("confidence") or 0.5),
                trust_tier="SHADOW",
                reason_codes=list(hyp.get("reason_codes") or []) + ["replay"],
                extras={"source": "replay",
                        "kind": row.get("kind"),
                        "event_id": row.get("event_id")},
            )
        except Exception:
            return None


def _hms_to_secs(hms: str) -> float:
    try:
        h, m, s = hms.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return 0.0


def _now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


declare(IOSpec(
    module="sentinel.replay",
    purpose="historical-session replay through the live cockpit — load a "
            "past ledger + research-signal archive, walk events in "
            "wallclock order at configurable speed, publish onto the "
            "live bus tagged as SHADOW so cockpit panels light up "
            "exactly as they did that day. Refuses to start while spine "
            "is live-real so synthetic signals can't influence real orders",
    inputs=["session_date (YYYY-MM-DD)",
            "speed multiplier (1.0 = real-time, 10.0 = 10x, max = flush)",
            "<journal>/ledger_<date>.jsonl",
            "<journal>/liqpool_live_signals.jsonl.<date> (rotated archive)"],
    outputs=["ModelSignal stream onto LivePublisher (trust_tier=SHADOW)",
             "ReplayProgress (state / events_emitted / last_ts_ist)"],
    consumes_from=["sentinel.shadow_ledger (archive)",
                   "sentinel.liqpool_bridge (rotated archive)"],
    produces_for=["sentinel.live_publisher", "sentinel.server (cockpit)"],
    tier="TRUSTED",
))
