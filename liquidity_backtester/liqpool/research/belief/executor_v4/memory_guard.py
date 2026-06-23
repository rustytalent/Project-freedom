"""MemoryGuard — live-session RSS diagnostics + hard ceiling.

Founder 2026-06-22 RAM-leak triage: the live Kite session grew from
27.6 GB to 27.9 GB in 20 seconds at ~35 minutes. That trajectory is a
guaranteed OOM. This module gives the live loop two things:

  1. **Diagnostics** — RSS in MB, deltas, and the last sample interval,
     sampled at most ``sample_interval_seconds``. The diagnostic block
     is cheap (1 syscall) and is logged to the cockpit so the operator
     can see RAM in real time.
  2. **Hard ceiling** — if RSS crosses ``hard_limit_mb``, the guard
     raises ``MemoryCeilingExceeded``. The live runner catches this,
     flushes any pending work, and exits cleanly. Better a controlled
     shutdown than an OOM-killed container.

We try ``psutil`` first (most accurate), fall back to ``/proc/self/status``
on Linux (zero-dep), and degrade silently on platforms where neither
works.

The guard is purely advisory in tests — you can construct it with
``hard_limit_mb=None`` to disable the ceiling and only collect
diagnostics.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class MemoryCeilingExceeded(RuntimeError):
    """Raised by MemoryGuard when RSS crosses the hard ceiling."""


def _rss_mb_via_psutil() -> Optional[float]:
    try:
        import psutil
    except Exception:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def _rss_mb_via_proc() -> Optional[float]:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        # VmRSS is in kB.
                        return float(parts[1]) / 1024.0
    except (FileNotFoundError, PermissionError, OSError):
        return None
    return None


def get_rss_mb() -> Optional[float]:
    """Best-effort current RSS in MB. Returns None when unavailable."""
    v = _rss_mb_via_psutil()
    if v is not None:
        return v
    return _rss_mb_via_proc()


@dataclass
class MemoryGuardConfig:
    sample_interval_seconds: float = 5.0     # how often to sample RSS
    soft_warn_mb: float = 4_096.0            # log a warning above this
    hard_limit_mb: Optional[float] = 12_288.0  # 12 GB — abort the loop
    history_cap: int = 240                   # ~20 minutes at 5s sampling


@dataclass
class MemoryGuard:
    """Sampled RSS diagnostics + ceiling enforcement."""
    cfg: MemoryGuardConfig = field(default_factory=MemoryGuardConfig)
    samples: List[Dict[str, float]] = field(default_factory=list)
    _last_sample_ts: float = 0.0
    _last_rss_mb: Optional[float] = None

    def check(self) -> Optional[Dict[str, float]]:
        """Sample RSS if the interval has elapsed; return the latest
        sample (or None when the interval hasn't elapsed)."""
        now = time.time()
        if now - self._last_sample_ts < self.cfg.sample_interval_seconds:
            return None
        rss = get_rss_mb()
        self._last_sample_ts = now
        if rss is None:
            return None
        delta = (rss - self._last_rss_mb
                 if self._last_rss_mb is not None else 0.0)
        sample = {"ts": now, "rss_mb": round(rss, 1),
                    "delta_mb": round(delta, 1)}
        self.samples.append(sample)
        cap = max(20, int(self.cfg.history_cap))
        if len(self.samples) > cap:
            self.samples = self.samples[-cap:]
        self._last_rss_mb = rss
        # Hard ceiling — raise so the live loop can shut down cleanly.
        if (self.cfg.hard_limit_mb is not None
                and rss > self.cfg.hard_limit_mb):
            raise MemoryCeilingExceeded(
                f"RSS {rss:.0f} MB > hard ceiling "
                f"{self.cfg.hard_limit_mb:.0f} MB")
        return sample

    def latest(self) -> Optional[Dict[str, float]]:
        return self.samples[-1] if self.samples else None

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {"n_samples": 0}
        rss_now = self.samples[-1]["rss_mb"]
        rss_min = min(s["rss_mb"] for s in self.samples)
        rss_max = max(s["rss_mb"] for s in self.samples)
        # Trend over the last few samples (positive = growing).
        tail = self.samples[-8:]
        trend = (tail[-1]["rss_mb"] - tail[0]["rss_mb"]
                 if len(tail) >= 2 else 0.0)
        return {
            "n_samples": len(self.samples),
            "rss_now_mb": rss_now,
            "rss_min_mb": rss_min,
            "rss_max_mb": rss_max,
            "rss_trend_recent_mb": round(trend, 1),
            "soft_warn_mb": self.cfg.soft_warn_mb,
            "hard_limit_mb": self.cfg.hard_limit_mb,
            "above_soft_warn": rss_now > self.cfg.soft_warn_mb,
        }
