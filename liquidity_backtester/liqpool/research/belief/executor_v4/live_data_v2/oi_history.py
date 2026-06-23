"""OIHistoryTracker — per-strike open interest velocity + acceleration.

OI tells us who is writing and who is unwinding. A strike whose OI is
climbing while spot moves AWAY from it is a strike being defended (more
writers piling in). A strike whose OI is collapsing while spot
approaches it is being unwound (writers running for cover).

The tracker maintains a short bounded history per (strike, option_type)
and exposes:

  * **velocity** — Δ OI per minute
  * **acceleration** — Δ velocity per minute
  * **net direction signal** — combined CE/PE OI velocity signed by
    rail (CE writers = bearish; PE writers = bullish)

Bounded by ``history_bars`` so no leak. State is reset cleanly at the
start of each session (the live runner calls ``reset()``).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class OIVelocityReading:
    """One per-strike velocity / acceleration sample."""
    strike: float
    option_type: str
    oi: float
    oi_velocity_per_min: float    # Δ OI per minute
    oi_acceleration_per_min2: float
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strike": float(self.strike),
            "option_type": self.option_type,
            "oi": float(self.oi),
            "oi_velocity_per_min": round(self.oi_velocity_per_min, 2),
            "oi_acceleration_per_min2": round(
                self.oi_acceleration_per_min2, 4),
            "n_samples": int(self.n_samples),
        }


class OIHistoryTracker:
    """Per-(strike, option_type) bounded OI history + velocity/accel."""

    def __init__(self, *, history_bars: int = 60,
                  velocity_window_bars: int = 5,
                  ) -> None:
        self.history_bars = int(history_bars)
        self.velocity_window_bars = int(velocity_window_bars)
        # Map: (strike, option_type) → deque of (ts_seconds, oi).
        self._history: Dict[Tuple[float, str], Deque[Tuple[float, float]]] = {}

    def reset(self) -> None:
        self._history.clear()

    def observe(self, *,
                  strike: float,
                  option_type: str,
                  oi: float,
                  ts: Optional[float] = None,
                  ) -> OIVelocityReading:
        """Record a per-strike OI sample. Returns the latest reading."""
        key = (float(strike), str(option_type))
        history = self._history.setdefault(key, deque(maxlen=self.history_bars))
        now = float(ts) if ts is not None else time.time()
        history.append((now, float(oi)))
        return self.reading_for(strike, option_type)

    def reading_for(self, strike: float, option_type: str
                     ) -> OIVelocityReading:
        key = (float(strike), str(option_type))
        history = self._history.get(key)
        if not history:
            return OIVelocityReading(
                strike=float(strike), option_type=str(option_type),
                oi=0.0, oi_velocity_per_min=0.0,
                oi_acceleration_per_min2=0.0, n_samples=0,
            )
        if len(history) < 2:
            return OIVelocityReading(
                strike=float(strike), option_type=str(option_type),
                oi=history[-1][1], oi_velocity_per_min=0.0,
                oi_acceleration_per_min2=0.0,
                n_samples=len(history),
            )
        # Velocity: Δ OI over the velocity window converted to per-minute.
        window = list(history)[-self.velocity_window_bars:] \
                  if len(history) >= self.velocity_window_bars \
                  else list(history)
        ts0, oi0 = window[0]
        ts1, oi1 = window[-1]
        elapsed_min = max(1e-6, (ts1 - ts0) / 60.0)
        velocity = (oi1 - oi0) / elapsed_min
        # Acceleration: compare first-half vs second-half average velocity.
        accel = 0.0
        if len(window) >= 4:
            mid = len(window) // 2
            t_a0, oi_a0 = window[0]
            t_a1, oi_a1 = window[mid]
            t_b0, oi_b0 = window[mid]
            t_b1, oi_b1 = window[-1]
            vel_a = (oi_a1 - oi_a0) / max(1e-6, (t_a1 - t_a0) / 60.0)
            vel_b = (oi_b1 - oi_b0) / max(1e-6, (t_b1 - t_b0) / 60.0)
            accel = (vel_b - vel_a)
        return OIVelocityReading(
            strike=float(strike), option_type=str(option_type),
            oi=oi1,
            oi_velocity_per_min=velocity,
            oi_acceleration_per_min2=accel,
            n_samples=len(history),
        )

    def all_readings(self) -> List[OIVelocityReading]:
        return [self.reading_for(strike, opt)
                for (strike, opt) in self._history.keys()]

    def rail_net_velocity(self) -> Dict[str, float]:
        """Sum OI velocity per rail. Negative CE rail velocity means
        CE writers unwinding (bullish); positive means new CE writers
        (bearish). Mirror for PE."""
        out: Dict[str, float] = {"CE": 0.0, "PE": 0.0}
        for r in self.all_readings():
            if r.option_type in out:
                out[r.option_type] += r.oi_velocity_per_min
        return out
