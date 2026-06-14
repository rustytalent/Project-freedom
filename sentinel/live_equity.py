"""Live NIFTY constituent board — the founder's 'is the index move real?'
panel.

The thesis ChatGPT was right about: NIFTY can be +0.35% with HDFCBANK
and ICICIBANK weak — that's a **fragile** rally. Retail traders don't
have this; institutional desks live by it.

This module is small on purpose:

  * ``TOP10`` — the canonical 10 names (matches NIFTY_TOP10_WEIGHTS).
  * ``StockTick`` — one symbol's live state (LTP, day-open, recent
    history for the sparkline, derived %change).
  * ``ConstituentBoard`` — holds per-symbol ``StockTick`` deques,
    consumes a ``{symbol: ltp}`` dict each cycle, computes the
    weight-adjusted contribution + breadth + regime + move-quality
    verdict by delegating to ``equity_layer``, publishes a
    ``ModelSignal`` to the bus.
  * ``DemoFeed`` — a deterministic synthetic feed so the cockpit shows
    real movement in demo mode without Kite. The walk has per-stock
    personality (trend bias + vol) so the board looks alive, not noisy.

The board is its own module, not bolted onto equity_layer, because
equity_layer is the pure stateless analytics (Q2-2026 weights, regime
classifier); this is the *streaming* surface that calls into it.
"""
from __future__ import annotations

import math
import random
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional

from .equity_layer import (
    NIFTY_TOP10_WEIGHTS, move_quality, top10_contextual_layer,
)
from .io_decl import IOSpec, declare
from .live_publisher import LivePublisher, ModelSignal, now_ist_hms

TOP10: List[str] = list(NIFTY_TOP10_WEIGHTS.keys())


@dataclass
class StockTick:
    symbol: str
    weight: float
    day_open: float = 0.0
    ltp: float = 0.0
    return_pct: float = 0.0
    contribution_pct: float = 0.0      # weight * return_pct (in %)
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["history"] = list(self.history)
        return d


class ConstituentBoard:
    """Holds streaming state per top-10 symbol; publishes a single
    ``constituent_board`` model signal per cycle whose payload is the
    full board snapshot. Thread-safe (one lock around the dict)."""

    def __init__(self, publisher: Optional[LivePublisher] = None,
                 weights: Optional[Mapping[str, float]] = None) -> None:
        self._lock = threading.Lock()
        self.weights: Dict[str, float] = dict(weights or NIFTY_TOP10_WEIGHTS)
        self.stocks: Dict[str, StockTick] = {
            sym: StockTick(symbol=sym, weight=w)
            for sym, w in self.weights.items()
        }
        self.publisher = publisher
        self.last_regime: str = ""
        self.last_quality: Dict[str, Any] = {}

    def tick(self, quotes: Mapping[str, float],
              index_return_pct: Optional[float] = None) -> Dict[str, Any]:
        """Feed one cycle of {symbol: ltp}. First call per symbol sets the
        day-open; subsequent calls update ltp + recompute %change +
        sparkline. Returns the snapshot dict the dashboard renders."""
        with self._lock:
            for sym, ltp in quotes.items():
                s = self.stocks.get(sym)
                if s is None or ltp is None or ltp <= 0:
                    continue
                if s.day_open <= 0:
                    s.day_open = float(ltp)
                s.ltp = float(ltp)
                s.return_pct = round(
                    (s.ltp - s.day_open) / s.day_open * 100.0, 2)
                s.contribution_pct = round(s.weight * s.return_pct, 3)
                s.history.append(float(ltp))

            # Use the supplied index return if given; otherwise estimate it
            # from the top-10's weight-adjusted contribution (good enough
            # in demo — desks pipe in the real index in prod).
            returns_pct = {sym: s.return_pct for sym, s in self.stocks.items()}
            if index_return_pct is None:
                index_return_pct = sum(s.contribution_pct for s in self.stocks.values())

            ctx = top10_contextual_layer(returns_pct, index_return_pct, self.weights)
            quality = move_quality(ctx)
            self.last_regime = ctx.regime
            self.last_quality = quality

            snapshot = {
                "regime": ctx.regime,
                "index_return_pct": ctx.index_return_pct,
                "top10_contribution_pct": ctx.top10_summed_contribution_pct,
                "breadth_up": ctx.breadth_up,
                "breadth_down": ctx.breadth_down,
                "quality": quality,
                "stocks": [self.stocks[sym].to_row() for sym in TOP10
                           if self.stocks[sym].day_open > 0],
                "leaders": [vars(c) for c in ctx.leaders],
                "laggards": [vars(c) for c in ctx.laggards],
                "effective_date": ctx.effective_date,
            }

        if self.publisher is not None and snapshot["stocks"]:
            # Publish as a TRUSTED signal so it mirrors to the ShadowLedger
            # — this is the institutional-grade context, not research noise.
            self.publisher.publish(ModelSignal(
                ts_ist=now_ist_hms(), asset="NIFTY",
                model="constituent_board",
                signal=f"{quality['verdict']}: {quality['headline']}",
                confidence=quality["confidence"],
                trust_tier="TRUSTED",
                reason_codes=quality["reason_codes"],
                extras={
                    "regime": ctx.regime,
                    "breadth_up": ctx.breadth_up,
                    "breadth_down": ctx.breadth_down,
                    "leaders": [c.symbol for c in ctx.leaders],
                    "laggards": [c.symbol for c in ctx.laggards],
                },
            ))
        return snapshot


# ---------------------------------------------------------------------------
# DemoFeed — synthetic price walks per top-10 personality
# ---------------------------------------------------------------------------

@dataclass
class _Personality:
    base: float                # opening price
    vol: float                 # per-tick volatility (fractional)
    drift: float               # per-tick mean drift (fractional)


# Approximate session-open prices + a personality profile so the demo
# board looks alive without being random soup. Numbers are illustrative;
# operators with live Kite swap this out for `quote(...)`.
_DEMO_BOOK: Dict[str, _Personality] = {
    "RELIANCE":  _Personality(base=2880.0,  vol=0.0008,  drift=0.00010),
    "HDFCBANK":  _Personality(base=1620.0,  vol=0.0006,  drift=-0.00006),
    "ICICIBANK": _Personality(base=1180.0,  vol=0.0009,  drift=0.00008),
    "INFY":      _Personality(base=1850.0,  vol=0.0010,  drift=0.00012),
    "TCS":       _Personality(base=3960.0,  vol=0.0007,  drift=0.00004),
    "ITC":       _Personality(base=470.0,   vol=0.0005,  drift=-0.00002),
    "LT":        _Personality(base=3580.0,  vol=0.0011,  drift=0.00009),
    "AXISBANK":  _Personality(base=1140.0,  vol=0.0010,  drift=-0.00010),
    "KOTAKBANK": _Personality(base=1760.0,  vol=0.0008,  drift=-0.00012),
    "SBIN":      _Personality(base=820.0,   vol=0.0012,  drift=0.00015),
}


class DemoFeed:
    """Deterministic synthetic feed for the top-10 — used in demo mode so
    the constituent board has real movement to render. Each symbol has a
    personality (trend + vol); a single RNG keeps runs reproducible."""

    def __init__(self, seed: int = 1729) -> None:
        self._rng = random.Random(seed)
        self._state: Dict[str, float] = {sym: p.base for sym, p in _DEMO_BOOK.items()}

    def next(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for sym, p in _DEMO_BOOK.items():
            cur = self._state[sym]
            shock = self._rng.gauss(0.0, p.vol)
            new = max(0.1, cur * math.exp(p.drift + shock))
            self._state[sym] = new
            out[sym] = round(new, 2)
        return out


declare(IOSpec(
    module="sentinel.live_equity",
    purpose="streaming NIFTY constituent board — feeds top-10 quotes "
            "through equity_layer, computes contribution + breadth + "
            "regime + move-quality verdict, publishes one TRUSTED "
            "constituent_board signal per cycle to the live bus",
    inputs=["{symbol: ltp} from Kite top-10 quotes (or DemoFeed)",
            "optional explicit index_return_pct"],
    outputs=["board snapshot dict (regime + quality + per-stock rows + "
             "leaders / laggards + sparkline history)",
             "ModelSignal published to LivePublisher"],
    consumes_from=["sentinel.equity_layer", "sentinel.kite_client"],
    produces_for=["sentinel.live_publisher", "sentinel.server (cockpit)",
                  "sentinel.shadow_ledger"],
    tier="TRUSTED",
))
