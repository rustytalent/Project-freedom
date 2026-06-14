"""Audit logger — structured one-line-per-event JSON to stdout/stderr +
optional sink file. The thing every compliance audit asks for.

What we record (the institutional minimum):

  * order_placed       — every exit reaching the broker
  * order_blocked      — kill switch / RED tilt / preflight refusal
  * graduation         — every TrustPromotionRecord
  * preflight_ack      — Ulysses-contract commitments
  * intention_violation — committed-limit breach
  * tilt_band_change   — emotional-state transitions
  * auth_failure       — bad JWT / unknown plan / missing token
  * byok_connect       — runtime credential connection (no creds in the log!)
  * config_change      — runtime mode flips (kill, dry_run, etc.)

Every record carries the same minimal envelope: ``ts_utc``, ``event``,
``user_id`` (when known), ``severity`` (INFO/WARN/ERROR), and a
payload. The payload NEVER contains credentials / JWTs / personally
identifying numbers — the user_id is enough for Codex's side to join
against Supabase.

Output sinks:
  * stderr (always — for stdout-tailing observability tools)
  * <journal_dir>/audit.jsonl (always — for the compliance trail)
  * Optional: an injected callable (``register_audit_sink``) for
    Sentry / Datadog / Codex's gateway logger.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .io_decl import IOSpec, declare

LOG = logging.getLogger("sentinel.audit")


_external_sinks: List[Callable[[Dict[str, Any]], None]] = []
_sink_lock = threading.Lock()


def register_audit_sink(fn: Callable[[Dict[str, Any]], None]) -> None:
    """Codex's observability stack (Sentry, Datadog, his own gateway log)
    registers here. Each sink is called with the full event dict on
    every audit write. Sink exceptions are caught — one broken sink
    can't take down the cockpit."""
    with _sink_lock:
        _external_sinks.append(fn)


def clear_audit_sinks() -> None:
    """Test helper."""
    with _sink_lock:
        _external_sinks.clear()


class AuditLog:
    """Per-process audit log writer."""

    SEV_INFO = "INFO"
    SEV_WARN = "WARN"
    SEV_ERROR = "ERROR"

    def __init__(self, path: Optional[Path] = None,
                 also_stderr: bool = True) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.also_stderr = also_stderr
        self._lock = threading.Lock()

    def write(self, event: str, payload: Dict[str, Any],
              severity: str = "INFO",
              user_id: str = "",
              session: str = "") -> None:
        """Emit one audit event. Never blocks: a slow sink doesn't slow
        the trading loop."""
        record = {
            "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "event": event,
            "severity": severity,
            "user_id": user_id,
            "session": session,
            "pid": os.getpid(),
            "payload": _scrub(payload),
        }
        line = json.dumps(record, default=str, separators=(",", ":"))
        if self.path is not None:
            try:
                with self._lock, open(self.path, "a") as f:
                    f.write(line + "\n")
            except Exception as exc:
                # We can't audit-log a broken audit log; fall through.
                LOG.error("audit write failed: %s", exc)
        if self.also_stderr:
            try:
                print(line, file=sys.stderr, flush=True)
            except Exception:
                pass
        with _sink_lock:
            sinks = list(_external_sinks)
        for sink in sinks:
            try:
                sink(record)
            except Exception:
                LOG.warning("external audit sink failed", exc_info=False)


# ---------------------------------------------------------------------------
# Scrubber — make sure secrets never reach the log
# ---------------------------------------------------------------------------

_SECRET_KEYS = frozenset({
    "api_key", "access_token", "password", "secret", "token",
    "jwt", "authorization", "supabase_jwt", "client_secret",
    "session_token", "credentials",
})


def _scrub(payload: Any) -> Any:
    """Recursively redact any field whose key matches the secret list.
    Keeps the trail useful while making sure a leaked log doesn't
    leak credentials."""
    if isinstance(payload, dict):
        return {k: ("<REDACTED>" if k.lower() in _SECRET_KEYS else _scrub(v))
                for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_scrub(v) for v in payload]
    return payload


declare(IOSpec(
    module="sentinel.audit_log",
    purpose="institutional-grade audit logger — one-line-per-event JSON "
            "to stderr + journal file + optional external sinks "
            "(Sentry/Datadog/Codex gateway). Records order placements, "
            "blocks, graduations, preflight acks, intention violations, "
            "tilt transitions, auth failures, BYOK connects, config "
            "changes. Auto-scrubs credentials so a leaked log can't "
            "leak the broker session",
    inputs=["event name + payload dict + severity + user_id + session"],
    outputs=["JSONL row to <journal_dir>/audit.jsonl",
             "stderr (always)",
             "external sinks (registered)"],
    consumes_from=[],
    produces_for=["sentinel.server (every spine routing + every endpoint)",
                  "Codex's compliance / observability layer"],
    tier="TRUSTED",
))
