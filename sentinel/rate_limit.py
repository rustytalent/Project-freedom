"""Token-bucket rate limiter — defends the API from scrape / brute force.

This is the in-process layer that runs alongside whatever rate-limiting
Codex's gateway adds. Two reasons we still need it server-side:

  1. The gateway might be misconfigured / under load — defense in depth.
  2. Some attack paths (a slow trickle from many IPs) are easier to
     defend after the auth step where we know the user_id.

Two limiters:

  * IPRateLimiter — keyed by client IP, used for unauthenticated routes
    (/healthz / /readyz / login flow). Cheap, defends against scraping.

  * UserRateLimiter — keyed by AuthContext.user_id, used after auth.
    Different buckets for different actions: e.g. /api/byok/connect at
    5/min (account-touching) vs /api/state at 60/min (polling).

Token-bucket math: each key has a bucket with capacity C, refilled at
R tokens/sec. A request takes 1 token; refusal when bucket is empty.
Memory cost: ~80 bytes per key; LRU-evicted at MAX_KEYS to bound.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .io_decl import IOSpec, declare


_MAX_KEYS = 10_000               # eviction floor; tune via env later


@dataclass
class Bucket:
    tokens: float
    last_refill_monotonic: float


class TokenBucketLimiter:
    """Generic per-key token bucket. Thread-safe; bounded memory."""

    def __init__(self, capacity: float, refill_per_sec: float,
                 max_keys: int = _MAX_KEYS) -> None:
        self.capacity = float(capacity)
        self.refill = float(refill_per_sec)
        self.max_keys = max_keys
        self._lock = threading.Lock()
        self._buckets: "OrderedDict[str, Bucket]" = OrderedDict()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """Returns True if the request fits; False if throttled."""
        now = time.monotonic()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = Bucket(tokens=self.capacity,
                            last_refill_monotonic=now)
            else:
                self._buckets.move_to_end(key)
            # refill
            elapsed = now - b.last_refill_monotonic
            b.tokens = min(self.capacity,
                            b.tokens + elapsed * self.refill)
            b.last_refill_monotonic = now
            if b.tokens >= cost:
                b.tokens -= cost
                self._buckets[key] = b
                # LRU evict
                while len(self._buckets) > self.max_keys:
                    self._buckets.popitem(last=False)
                return True
            self._buckets[key] = b
            return False

    def reset(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)

    def remaining(self, key: str) -> float:
        """Read-only peek; for tests + observability."""
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                return self.capacity
            now = time.monotonic()
            elapsed = now - b.last_refill_monotonic
            return min(self.capacity, b.tokens + elapsed * self.refill)


# ─────────────────────────────────────────────────────────────────
# Default presets — tuned for a single-operator launch; can be
# overridden via env or per-user later.
# ─────────────────────────────────────────────────────────────────

@dataclass
class RatePreset:
    name: str
    capacity: float            # burst size
    refill_per_sec: float      # sustained rate


PRESETS: Dict[str, RatePreset] = {
    # Light public routes — generous so the LB's health check doesn't trip
    "public":      RatePreset("public",      capacity=120, refill_per_sec=60),
    # Authenticated polling — /api/state at 2s default; allow burst of 30
    "polling":     RatePreset("polling",     capacity=60,  refill_per_sec=30),
    # Account-touching — BYOK connect / preflight / kill — small + slow
    "account":     RatePreset("account",     capacity=10,  refill_per_sec=1),
    # Heavy compute — Monte Carlo, stress matrix — quota by minute
    "heavy":       RatePreset("heavy",       capacity=20,  refill_per_sec=1.0/30),
}


# Process-wide singleton limiters per preset
_LIMITERS: Dict[str, TokenBucketLimiter] = {
    name: TokenBucketLimiter(p.capacity, p.refill_per_sec)
    for name, p in PRESETS.items()
}


def limiter_for(preset: str) -> TokenBucketLimiter:
    return _LIMITERS[preset]


def allow(preset: str, key: str) -> Tuple[bool, float]:
    """Convenience used by FastAPI dependencies. Returns
    (allowed, remaining_tokens) for honest error messages."""
    lim = _LIMITERS[preset]
    ok = lim.allow(key)
    return ok, lim.remaining(key)


def reset_all() -> None:
    """Test helper."""
    for lim in _LIMITERS.values():
        lim.reset()


declare(IOSpec(
    module="sentinel.rate_limit",
    purpose="token-bucket rate limiter — defense-in-depth at the API "
            "layer. Four presets: public (LB-friendly), polling "
            "(authenticated read), account (BYOK/preflight/kill), heavy "
            "(Monte Carlo / stress). Bounded-memory LRU-evicted keys "
            "stop a key-explosion attack",
    inputs=["preset name (public|polling|account|heavy)",
            "key (IP or user_id)"],
    outputs=["allow/deny + remaining tokens for honest 429 messages"],
    consumes_from=[],
    produces_for=["sentinel.server (rate-limited endpoints)"],
    tier="TRUSTED",
))
