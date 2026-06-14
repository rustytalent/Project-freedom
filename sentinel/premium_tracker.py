"""Premium tracker — per-symbol LTP history so the cockpit can chart
ATM CE / ATM PE / held-position premium just like the spot.

The founder's observation: the OPTION PREMIUM chart matters more than
the index chart, because when you trade options, the premium is what
you actually pay/receive. A flat spot with expanding ATM CE premium =
the market is bidding for upside vol; the option is responding even
though the spot isn't.

This module is intentionally small:

  * ``PremiumTracker`` — thread-safe map of symbol -> bounded deque of
    (ts_ist, premium) records, plus derived metrics: %change from open,
    velocity (per-min slope), recent peak / trough.
  * ``derive_metrics(history)`` — pure function for tests.

The cockpit picks the held-position-with-largest-qty by default; the
operator can override via the API.
"""
from __future__ import annotations

import statistics
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from .io_decl import IOSpec, declare

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class PremiumPoint:
    ts_ist: str
    premium: float


@dataclass
class PremiumSeries:
    symbol: str
    open_premium: float = 0.0
    last_premium: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    return_pct: float = 0.0
    velocity_per_min: float = 0.0
    history: Deque[PremiumPoint] = field(default_factory=lambda: deque(maxlen=180))

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["history"] = [asdict(p) for p in self.history]
        if self.low == float("inf"):
            d["low"] = 0.0
        return d


def derive_metrics(history: Iterable[PremiumPoint]) -> Dict[str, float]:
    """Compute velocity (per-minute slope across the last ~60s of ticks)
    and the basic high/low/return numbers. Pure — for tests."""
    pts = list(history)
    if len(pts) < 2:
        return {"velocity_per_min": 0.0, "high": 0.0, "low": 0.0,
                "return_pct": 0.0}
    premiums = [p.premium for p in pts]
    high, low = max(premiums), min(premiums)
    ret = (premiums[-1] - premiums[0]) / premiums[0] * 100.0 if premiums[0] > 0 else 0.0
    # velocity: simple regression over the most recent 30 points
    recent = pts[-30:]
    if len(recent) < 2:
        velocity = 0.0
    else:
        t0 = _to_seconds(recent[0].ts_ist)
        xs = [(_to_seconds(p.ts_ist) - t0) / 60.0 for p in recent]
        ys = [p.premium for p in recent]
        mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        velocity = num / den if den > 0 else 0.0
    return {"velocity_per_min": round(velocity, 3),
            "high": round(high, 2), "low": round(low, 2),
            "return_pct": round(ret, 2)}


def _to_seconds(ts_ist: str) -> float:
    """HH:MM:SS -> seconds-since-midnight; tolerant of bad input."""
    try:
        h, m, s = ts_ist.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return 0.0


class PremiumTracker:
    """Bounded per-symbol LTP history, thread-safe."""

    def __init__(self, max_per_symbol: int = 180) -> None:
        self._lock = threading.Lock()
        self._series: Dict[str, PremiumSeries] = {}
        self._cap = max_per_symbol

    def update(self, symbol: str, premium: float,
               ts_ist: Optional[str] = None) -> None:
        if premium is None or premium <= 0:
            return
        ts = ts_ist or datetime.now(IST).strftime("%H:%M:%S")
        with self._lock:
            s = self._series.get(symbol)
            if s is None:
                s = PremiumSeries(symbol=symbol, open_premium=float(premium))
                s.history = deque(maxlen=self._cap)
                self._series[symbol] = s
            s.history.append(PremiumPoint(ts_ist=ts, premium=float(premium)))
            s.last_premium = float(premium)
            metrics = derive_metrics(s.history)
            s.high = metrics["high"]
            s.low = metrics["low"]
            s.return_pct = metrics["return_pct"]
            s.velocity_per_min = metrics["velocity_per_min"]

    def get(self, symbol: str) -> Optional[PremiumSeries]:
        with self._lock:
            return self._series.get(symbol)

    def known_symbols(self) -> List[str]:
        with self._lock:
            return list(self._series)

    def select_default(self, positions: Dict[str, Any]) -> Optional[str]:
        """Pick the held position with the largest absolute qty as the
        default symbol the premium chart shows. Falls back to the most
        recently observed symbol."""
        best, best_qty = None, 0
        for sym, v in positions.items():
            qty = abs(getattr(v, "quantity", 0))
            if qty > best_qty and sym in self._series:
                best, best_qty = sym, qty
        if best is not None:
            return best
        syms = self.known_symbols()
        return syms[-1] if syms else None

    def snapshot(self, symbol: Optional[str]) -> Optional[Dict[str, Any]]:
        s = self.get(symbol) if symbol else None
        if s is None:
            return None
        return s.to_row()


declare(IOSpec(
    module="sentinel.premium_tracker",
    purpose="per-symbol premium history + velocity/high/low/return — "
            "feeds the option-premium chart on the cockpit. The founder "
            "asked for this: a flat spot with rising ATM premium is the "
            "first sign vol is being bid; you can't see that on a spot "
            "chart alone",
    inputs=["symbol + LTP per quote cycle"],
    outputs=["PremiumSeries snapshot (history + metrics) by symbol"],
    consumes_from=["sentinel.kite_client"],
    produces_for=["sentinel.server (cockpit option chart)"],
    tier="TRUSTED",
))
