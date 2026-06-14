"""Live Model Publisher — the spine ChatGPT was right about.

The founder's diagnosis, exactly: until Sentinel shows what the research
brain is thinking live, it's just another panel. The fix is a clean
contract — every research model (reaction, proximity, liquidity, quality,
post-reaction, manipulation, ...) publishes a structured ``ModelSignal``
to one in-memory bus; the dashboard reads from the bus.

This module is intentionally tiny:

  * ``ModelSignal`` is the schema (matches the founder's
    timestamp/asset/model/signal/confidence/zone/risk/reason_codes
    shape; matches Codex's DecisionEvent fields).
  * ``LivePublisher`` is the bus — thread-safe, last-write-wins per
    (asset, model) key, with a bounded history ring for the brief feed.
  * Optional ShadowLedger mirror: every TRUSTED-or-higher signal is
    canonicalised as a ``model_suggestion`` event so the curator /
    calibration / flywheel can train on the same stream the dashboard
    surfaces. ONE substrate, again.

Models live in ``sentinel.live_models``. They don't import this module's
publisher directly — the server wires them — so a model can be tested
in isolation and the bus can be swapped (e.g. for a Kafka adapter)
without touching model code.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from .io_decl import IOSpec, declare
from .shadow_ledger import (
    Context, Identity, KIND_MODEL_SUGGESTION, ShadowLedger, new_event,
)

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass(frozen=True)
class ModelSignal:
    """One live model output. Frozen so the bus can hand the same instance
    to many readers without defensive copies."""
    ts_ist: str                                  # HH:MM:SS in IST
    asset: str                                   # NIFTY / RELIANCE / ATM_CE / ...
    model: str                                   # reaction_model / proximity_model / ...
    signal: str                                  # human-readable verdict
    confidence: float                            # 0..1
    trust_tier: str = "SHADOW"                   # SHADOW / LOGGED / TRUSTED
    zone: Optional[Tuple[float, float]] = None   # spot band the signal applies to
    risk: str = ""                               # invalidation condition
    reason_codes: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.asset}|{self.model}"

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        # tuples don't survive JSON cleanly; flatten the zone
        if self.zone is not None:
            d["zone"] = list(self.zone)
        return d


class LivePublisher:
    """Thread-safe in-memory pub/sub. Last-write-wins per (asset, model);
    bounded history ring keeps the live brief feed light."""

    def __init__(self, shadow_ledger: Optional[ShadowLedger] = None,
                 session: str = "", history: int = 80) -> None:
        self._lock = threading.Lock()
        self._latest: Dict[str, ModelSignal] = {}
        self._history: Deque[ModelSignal] = deque(maxlen=history)
        self.shadow_ledger = shadow_ledger
        self.session = session
        # cheap suppression so the same signal text doesn't spam the feed
        # every poll cycle — only re-publish when the SIGNAL or CONFIDENCE
        # crosses a meaningful step.
        self._last_published: Dict[str, Tuple[str, float]] = {}

    # -- producer side ----------------------------------------------------

    def publish(self, sig: ModelSignal) -> bool:
        """Returns True if the signal was a meaningful change worth
        showing the operator. Cheap dedupe: same (signal text, confidence
        rounded to 0.05) within the same minute is suppressed."""
        with self._lock:
            prev = self._last_published.get(sig.key)
            step = round(sig.confidence * 20) / 20.0     # 0.05 buckets
            this = (sig.signal, step)
            if prev == this:
                # still update latest so /api/live/signals shows fresh ts;
                # but don't append to the brief feed or mirror to ledger.
                self._latest[sig.key] = sig
                return False
            self._last_published[sig.key] = this
            self._latest[sig.key] = sig
            self._history.appendleft(sig)
        # Canonical mirror — only for TRUSTED-or-higher; SHADOW research
        # noise stays on the bus, off the substrate.
        if (self.shadow_ledger is not None and sig.trust_tier != "SHADOW"
                and sig.confidence >= 0.5):
            self._canonicalise(sig)
        return True

    def _canonicalise(self, sig: ModelSignal) -> None:
        try:
            ident = Identity(
                instrument=sig.asset, option_type="", strike=0.0,
                expiry=None, moneyness_key="", underlying_price=0.0,
                premium=0.0,
            )
            ev = new_event(KIND_MODEL_SUGGESTION,
                           self.session or _ist_today(),
                           ident, Context(), None,
                           scientist=sig.model,
                           trust_tier=sig.trust_tier)
            # carry the signal payload onto the judgment notes so the
            # row is self-describing without a separate join.
            ev.judgment.notes = sig.signal
            ev.context.model_signal = sig.confidence
            self.shadow_ledger.write(ev)
        except Exception:
            pass  # canonical mirror is best-effort

    # -- consumer side ----------------------------------------------------

    def current(self) -> Dict[str, ModelSignal]:
        with self._lock:
            return dict(self._latest)

    def history(self, limit: int = 20) -> List[ModelSignal]:
        with self._lock:
            return list(self._history)[:limit]

    def latest_per_asset(self, asset: str) -> List[ModelSignal]:
        with self._lock:
            return [s for s in self._latest.values() if s.asset == asset]


# -- helpers ------------------------------------------------------------

def now_ist_hms() -> str:
    return datetime.now(IST).strftime("%H:%M:%S")


def _ist_today() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


declare(IOSpec(
    module="sentinel.live_publisher",
    purpose="the live model bus — ModelSignal contract + LivePublisher; "
            "every research model output flows through here, dashboard "
            "reads from here, TRUSTED+ signals mirror to ShadowLedger "
            "(the founder's missing 'live model publisher' layer)",
    inputs=["ModelSignal(ts, asset, model, signal, confidence, zone, "
            "risk, reason_codes, trust_tier) from any live model"],
    outputs=["dict[(asset|model)] -> latest ModelSignal",
             "history ring for the brief feed",
             "model_suggestion events in sentinel.shadow_ledger (TRUSTED+)"],
    consumes_from=["sentinel.live_models"],
    produces_for=["sentinel.server (dashboard)", "sentinel.shadow_ledger",
                  "sentinel.curator", "sentinel.ledger_export"],
    tier="TRUSTED",
))
