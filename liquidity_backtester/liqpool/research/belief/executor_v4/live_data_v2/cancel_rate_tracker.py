"""CancelRateTracker — depth-level persistence + cancel rate.

The founder's anti-spoofing structural defense: only count orders
that survive K consecutive ticks. We compare depth between consecutive
ticks at the same instrument and classify each level as:

  * **persistent**  — same (price, ~quantity) observed for ≥ K ticks
  * **fresh**       — appeared this tick, no history
  * **cancelled**   — was present last tick, gone this tick

Outputs per instrument:

  * **persistence_score** — fraction of recent depth levels that have
    survived ``persistence_threshold_ticks`` ticks. High = real book.
    Low = spoofing-rich.
  * **cancel_rate** — cancellations per tick over a rolling window.

The CancelRateDetector consumes these per-instrument signals to detect
spoofing regimes (high cancel rate + low persistence = spoofs
dominant, damp other detectors that read book depth).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CancelRateReading:
    """Per-instrument cancel-rate diagnostics."""
    tradingsymbol: str
    persistence_score: float       # 0..1, higher = more persistent book
    cancel_rate_per_tick: float    # avg cancellations per recent tick
    fresh_rate_per_tick: float
    n_ticks_observed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tradingsymbol": self.tradingsymbol,
            "persistence_score": round(self.persistence_score, 4),
            "cancel_rate_per_tick": round(self.cancel_rate_per_tick, 4),
            "fresh_rate_per_tick": round(self.fresh_rate_per_tick, 4),
            "n_ticks_observed": int(self.n_ticks_observed),
        }


@dataclass
class _PerInstrumentState:
    last_levels: List[Tuple[float, int, str]]   # (price, qty, side)
    persistence_counts: Dict[Tuple[float, str], int]
    recent_cancels: Deque[int]                    # cancels per tick
    recent_freshes: Deque[int]
    n_ticks: int = 0


class CancelRateTracker:
    """Per-instrument depth persistence + cancel rate."""

    def __init__(self, *,
                  persistence_threshold_ticks: int = 3,
                  cancel_window_ticks: int = 30,
                  qty_match_pct: float = 0.10,
                  ) -> None:
        self.persistence_threshold_ticks = int(persistence_threshold_ticks)
        self.cancel_window_ticks = int(cancel_window_ticks)
        self.qty_match_pct = float(qty_match_pct)
        self._states: Dict[str, _PerInstrumentState] = {}

    def reset(self) -> None:
        self._states.clear()

    def observe(self, *,
                  tradingsymbol: str,
                  depth_buy: List[Any],
                  depth_sell: List[Any],
                  ) -> CancelRateReading:
        """Record this tick's depth levels. Returns the updated reading."""
        sym = str(tradingsymbol)
        state = self._states.get(sym)
        if state is None:
            state = _PerInstrumentState(
                last_levels=[], persistence_counts={},
                recent_cancels=deque(
                    maxlen=self.cancel_window_ticks),
                recent_freshes=deque(
                    maxlen=self.cancel_window_ticks),
                n_ticks=0,
            )
            self._states[sym] = state
        # Normalise current depth into (price, qty, side) tuples.
        cur: List[Tuple[float, int, str]] = []
        for d in (depth_buy or []):
            try:
                cur.append((float(d.get("price") or d.get("p") or 0.0),
                              int(d.get("quantity")
                                  or d.get("qty") or 0),
                              "buy"))
            except (TypeError, ValueError, AttributeError):
                continue
        for d in (depth_sell or []):
            try:
                cur.append((float(d.get("price") or d.get("p") or 0.0),
                              int(d.get("quantity")
                                  or d.get("qty") or 0),
                              "sell"))
            except (TypeError, ValueError, AttributeError):
                continue
        # Compare with last_levels: classify cancel / fresh / persistent.
        prev_keys = {(p, side) for (p, _q, side) in state.last_levels}
        cur_keys = {(p, side) for (p, _q, side) in cur}
        cancelled = prev_keys - cur_keys
        fresh = cur_keys - prev_keys
        state.recent_cancels.append(len(cancelled))
        state.recent_freshes.append(len(fresh))
        # Persistence: increment count for levels still present.
        new_counts: Dict[Tuple[float, str], int] = {}
        for (price, side) in cur_keys:
            new_counts[(price, side)] = (
                state.persistence_counts.get((price, side), 0) + 1)
        state.persistence_counts = new_counts
        state.last_levels = cur
        state.n_ticks += 1

        # Persistence score: fraction of current levels that have
        # survived persistence_threshold_ticks consecutive ticks.
        if not cur_keys:
            persistence = 0.0
        else:
            survivors = sum(1 for v in new_counts.values()
                            if v >= self.persistence_threshold_ticks)
            persistence = float(survivors) / float(len(cur_keys))
        # Rates (per-tick averages over the window).
        cancel_rate = (sum(state.recent_cancels)
                       / max(1, len(state.recent_cancels)))
        fresh_rate = (sum(state.recent_freshes)
                      / max(1, len(state.recent_freshes)))
        return CancelRateReading(
            tradingsymbol=sym,
            persistence_score=persistence,
            cancel_rate_per_tick=cancel_rate,
            fresh_rate_per_tick=fresh_rate,
            n_ticks_observed=state.n_ticks,
        )

    def reading_for(self, tradingsymbol: str
                     ) -> Optional[CancelRateReading]:
        state = self._states.get(tradingsymbol)
        if state is None:
            return None
        return self.observe(
            tradingsymbol=tradingsymbol,
            depth_buy=[{"price": p, "quantity": q}
                         for (p, q, side) in state.last_levels
                         if side == "buy"],
            depth_sell=[{"price": p, "quantity": q}
                          for (p, q, side) in state.last_levels
                          if side == "sell"],
        )

    def aggregate_persistence(self) -> float:
        """Cross-instrument average persistence score."""
        readings = [
            self.observe(
                tradingsymbol=sym,
                depth_buy=[{"price": p, "quantity": q}
                             for (p, q, side) in s.last_levels
                             if side == "buy"],
                depth_sell=[{"price": p, "quantity": q}
                              for (p, q, side) in s.last_levels
                              if side == "sell"],
            )
            for sym, s in self._states.items() if s.last_levels
        ]
        if not readings:
            return 1.0
        return sum(r.persistence_score for r in readings) / len(readings)

    def aggregate_cancel_rate(self) -> float:
        readings = [
            self.observe(
                tradingsymbol=sym,
                depth_buy=[{"price": p, "quantity": q}
                             for (p, q, side) in s.last_levels
                             if side == "buy"],
                depth_sell=[{"price": p, "quantity": q}
                              for (p, q, side) in s.last_levels
                              if side == "sell"],
            )
            for sym, s in self._states.items() if s.last_levels
        ]
        if not readings:
            return 0.0
        return sum(r.cancel_rate_per_tick for r in readings) / len(readings)
