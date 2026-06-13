"""Leakage guard — the founder's "0.9 AUC: real or leaking?" alarm.

The principle the founder stated: a high score can be good OR bad. If a
metric jumps suspiciously fast, suspect leakage. But if it is high AND
the real-money / re-run expectancy is also improving, the high score is
real. So this guard never trusts a score in isolation — it cross-checks
the score's jump against the calibration re-run's expectancy delta.

Verdicts:
  CLEAN        score moved within a believable per-session band
  SUSPICIOUS   score jumped fast but expectancy did NOT improve -> likely leak
  EARNED       score jumped fast AND expectancy improved -> trust it
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_decl import IOSpec, declare

# A per-session win-rate / AUC jump beyond this is "fast".
SUSPICIOUS_JUMP = 0.15


@dataclass
class LeakageVerdict:
    metric: str
    previous: Optional[float]
    current: float
    jump: Optional[float]
    expectancy_improved: bool
    verdict: str
    note: str


class LeakageGuard:
    def __init__(self, history_path: Optional[Path] = None) -> None:
        self.history_path = Path(history_path) if history_path else None
        self._hist: Dict[str, List[float]] = self._load()

    def _load(self) -> Dict[str, List[float]]:
        if self.history_path and self.history_path.exists():
            try:
                return json.loads(self.history_path.read_text())
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        if self.history_path:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps(self._hist))

    def check(self, metric: str, current: float,
              expectancy_improved: bool) -> LeakageVerdict:
        prev_list = self._hist.get(metric, [])
        previous = prev_list[-1] if prev_list else None
        jump = (current - previous) if previous is not None else None
        if jump is None:
            verdict, note = "CLEAN", "first observation; nothing to compare"
        elif abs(jump) <= SUSPICIOUS_JUMP:
            verdict, note = "CLEAN", f"moved {jump:+.3f}, within believable band"
        elif expectancy_improved:
            verdict = "EARNED"
            note = (f"jumped {jump:+.3f} AND re-run expectancy improved — "
                    f"the high score is backed by real outcome gains")
        else:
            verdict = "SUSPICIOUS"
            note = (f"jumped {jump:+.3f} but re-run expectancy did NOT improve "
                    f"— treat as possible leakage; do not promote")
        # record
        self._hist.setdefault(metric, []).append(round(float(current), 4))
        self._hist[metric] = self._hist[metric][-60:]
        self._save()
        return LeakageVerdict(metric, previous, round(current, 4), jump,
                              expectancy_improved, verdict, note)


declare(IOSpec(
    module="sentinel.leakage_guard",
    purpose="cross-checks metric jumps against re-run expectancy: "
            "CLEAN / EARNED / SUSPICIOUS (never trusts a score alone)",
    inputs=["metric value (e.g. win-rate/AUC)",
            "expectancy_improved flag from sentinel.calibration"],
    outputs=["sentinel.leakage_guard.LeakageVerdict",
             "file:<journal>/leakage_history.json"],
    consumes_from=["sentinel.curator", "sentinel.calibration"],
    produces_for=["sentinel.reports", "sentinel.orchestration"],
    tier="LOGGED",
))
