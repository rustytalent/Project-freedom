"""ShadowLedger — full-tape persistence for offline analysis + calibration.

Founder 2026-06-22 follow-up: "you have to make a layer where all this
data or shadow ledger or whatever gets stored and gets used so we can
use your data to calibrate." This is that layer.

Writes one JSONL file per IST date under the VPS state directory.
Each line is a single event of one of these kinds:

  * ``tick``       — manager observed a snapshot; we save a compact
                       summary (no per-tick depth payload to keep file
                       size manageable, but full slot_readings)
  * ``mm_intent``  — the fused MMIntent for that tick
  * ``decision``   — aggregator decision (accept/refuse + reasons)
  * ``open``       — a position opened
  * ``close``      — a position closed with realised_r
  * ``override``   — operator override set / cleared

Buffered + flushed every N events to limit fsync overhead. Tape rotates
at midnight IST. Old files are NEVER auto-deleted — the operator
manages retention.

The pull-down workflow:

    # operator on VPS
    tar czf shadow_ledger_2026-06-22.tgz \\
        state_dir/shadow_ledger_2026-06-22.jsonl
    # sends to me, I read offline and propose detector retunes
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


IST = timezone(timedelta(hours=5, minutes=30))


def _today_ist() -> str:
    return datetime.now(IST).date().isoformat()


@dataclass
class ShadowLedgerConfig:
    enabled: bool = True
    flush_every_n_events: int = 50
    keep_per_tick_slot_readings: bool = True
    # Cap on the in-memory write buffer; auto-flush when crossed even
    # if flush_every_n_events hasn't been reached.
    max_buffer_size: int = 256


class ShadowLedger:
    """Per-IST-date append-only JSONL of full-tape events.

    Thread-safety: NONE. The manager calls these from its single tick
    thread. If you ever multi-thread the manager, wrap the methods in a
    lock.
    """

    FILE_PREFIX = "shadow_ledger_"
    FILE_SUFFIX = ".jsonl"

    def __init__(self, *,
                  state_dir: Path,
                  cfg: Optional[ShadowLedgerConfig] = None,
                  ) -> None:
        self.cfg = cfg or ShadowLedgerConfig()
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Bounded buffer; we flush before it overflows.
        self._buffer: Deque[Dict[str, Any]] = deque(
            maxlen=max(64, self.cfg.max_buffer_size))
        self._n_since_flush: int = 0
        self._current_date: str = _today_ist()
        # Counters surfaced via summary().
        self._counters: Dict[str, int] = {
            "ticks": 0, "mm_intents": 0, "decisions": 0,
            "opens": 0, "closes": 0, "overrides": 0, "flushes": 0,
        }

    # ── Path helpers ──────────────────────────────────────────────

    def path_for(self, date_str: str) -> Path:
        return self.state_dir / (
            f"{self.FILE_PREFIX}{date_str}{self.FILE_SUFFIX}")

    def path_for_today(self) -> Path:
        return self.path_for(_today_ist())

    # ── Event recording ───────────────────────────────────────────

    def record_tick(self, *,
                       bar_index: int,
                       snapshot: Dict[str, Any],
                       ) -> None:
        if not self.cfg.enabled:
            return
        cfg = self.cfg
        compact: Dict[str, Any] = {
            "kind": "tick",
            "ts": _now_iso(),
            "bar_index": int(bar_index),
            "snapshot_ts": str(snapshot.get("ts") or ""),
            "spot": float(snapshot.get("spot") or 0.0),
            "thesis": snapshot.get("thesis"),
            "iv_state": snapshot.get("iv_state"),
            "battlefield": snapshot.get("battlefield"),
            "decision": snapshot.get("decision"),
        }
        if cfg.keep_per_tick_slot_readings:
            compact["slot_readings"] = snapshot.get("slot_readings")
        self._enqueue(compact)
        self._counters["ticks"] += 1

    def record_mm_intent(self, *,
                            bar_index: int,
                            mm_intent_dict: Dict[str, Any],
                            ) -> None:
        if not self.cfg.enabled:
            return
        self._enqueue({
            "kind": "mm_intent",
            "ts": _now_iso(),
            "bar_index": int(bar_index),
            "mm_intent": dict(mm_intent_dict or {}),
        })
        self._counters["mm_intents"] += 1

    def record_decision(self, *,
                          bar_index: int,
                          decision_dict: Dict[str, Any],
                          ) -> None:
        if not self.cfg.enabled:
            return
        self._enqueue({
            "kind": "decision",
            "ts": _now_iso(),
            "bar_index": int(bar_index),
            "decision": dict(decision_dict or {}),
        })
        self._counters["decisions"] += 1

    def record_open(self, *,
                       bar_index: int,
                       position_id: str,
                       hypothesis_dict: Dict[str, Any],
                       ) -> None:
        if not self.cfg.enabled:
            return
        self._enqueue({
            "kind": "open",
            "ts": _now_iso(),
            "bar_index": int(bar_index),
            "position_id": str(position_id),
            "hypothesis": dict(hypothesis_dict or {}),
        })
        self._counters["opens"] += 1

    def record_close(self, *,
                        bar_index: int,
                        position_id: str,
                        outcome_dict: Dict[str, Any],
                        ) -> None:
        if not self.cfg.enabled:
            return
        self._enqueue({
            "kind": "close",
            "ts": _now_iso(),
            "bar_index": int(bar_index),
            "position_id": str(position_id),
            "outcome": dict(outcome_dict or {}),
        })
        self._counters["closes"] += 1

    def record_override(self, *,
                           override_dict: Optional[Dict[str, Any]],
                           action: str,
                           ) -> None:
        if not self.cfg.enabled:
            return
        self._enqueue({
            "kind": "override",
            "ts": _now_iso(),
            "action": str(action),    # 'set' or 'clear'
            "override": dict(override_dict or {}),
        })
        self._counters["overrides"] += 1

    # ── Internals ─────────────────────────────────────────────────

    def _enqueue(self, event: Dict[str, Any]) -> None:
        # Date rollover.
        today = _today_ist()
        if today != self._current_date:
            self.flush()
            self._current_date = today
        self._buffer.append(event)
        self._n_since_flush += 1
        if self._n_since_flush >= self.cfg.flush_every_n_events:
            self.flush()

    def flush(self) -> int:
        """Append the buffer to today's file. Returns events written."""
        if not self._buffer:
            return 0
        path = self.path_for_today()
        written = 0
        try:
            with open(path, "a") as f:
                for event in self._buffer:
                    f.write(json.dumps(event, default=str) + "\n")
                    written += 1
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            self._buffer.clear()
            self._n_since_flush = 0
            self._counters["flushes"] += 1
        except Exception:
            # Persistence must never break the trading loop.
            pass
        return written

    def close(self) -> None:
        self.flush()

    # ── Cockpit summary ───────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        path = self.path_for_today()
        size_bytes = 0
        try:
            if path.exists():
                size_bytes = path.stat().st_size
        except OSError:
            pass
        return {
            "enabled": self.cfg.enabled,
            "current_path": str(path),
            "current_date": self._current_date,
            "today_size_bytes": int(size_bytes),
            "buffer_size": len(self._buffer),
            "counters": dict(self._counters),
        }


def _now_iso() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")
