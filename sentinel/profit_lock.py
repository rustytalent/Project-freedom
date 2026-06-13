"""Profit Lock — the ratcheting day-profit floor that fights greed.

The founder's psychology, made mechanical:

    entry            -> all profit is UNLOCKED (floating, can vanish)
    profit rises     -> the lock ratchets UP behind it (trailing)
    peak rises       -> LOCKED profit rises with it, never falls
    price falls       -> UNLOCKED (floating) profit disappears first
    current hits lock -> position flattened; LOCKED profit is realized

Worked example (the founder's numbers):
    current  = 8000
    peak     = 11500
    locked   = 6200       (the ratcheting floor)
    floating = 1800       (= current - locked, the part still at risk)
    -> if a sudden fall happens, the MOST you can lose back is the 1800;
       your worst-case realized day profit is the locked 6200.

The lock is a trailing stop on CUMULATIVE day P&L, not on any single
position. Arming it answers "this is the profit I am taking today,
irrespective of how high it goes or how hard it falls."

Two floor models (operator picks):
  ratio  : locked = peak * lock_ratio        (default; "keep 54% of peak")
  buffer : locked = peak - give_back_buffer  ("never give back > Rs X from peak")
Both ratchet: locked only ever rises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProfitLockState:
    armed: bool = False
    fired: bool = False
    mode: str = "ratio"               # ratio | buffer
    lock_ratio: float = 0.5           # used when mode == ratio
    give_back_buffer: float = 0.0     # used when mode == buffer (rupees)
    activation_floor: float = 1000.0  # don't lock until peak clears this
    current_pnl: float = 0.0
    peak_pnl: float = 0.0
    locked_pnl: float = 0.0           # the ratcheting floor; never falls
    activated: bool = False

    @property
    def floating_pnl(self) -> float:
        """The part still at risk = current minus locked, never negative."""
        return max(0.0, self.current_pnl - self.locked_pnl)

    @property
    def worst_case_day_profit(self) -> float:
        """If it all fell right now, what gets realized = the locked floor
        once activated, else whatever the current is (nothing locked yet)."""
        return self.locked_pnl if self.activated else min(self.current_pnl, 0.0)


class ProfitLock:
    """Stateful ratchet. Feed it cumulative day P&L each tick; it tells
    you when to flatten everything."""

    def __init__(self, mode: str = "ratio", lock_ratio: float = 0.5,
                 give_back_buffer: float = 0.0,
                 activation_floor: float = 1000.0) -> None:
        if mode not in ("ratio", "buffer"):
            raise ValueError("mode must be 'ratio' or 'buffer'")
        if mode == "ratio" and not (0.0 < lock_ratio < 1.0):
            raise ValueError("lock_ratio must be in (0,1)")
        if mode == "buffer" and give_back_buffer <= 0:
            raise ValueError("give_back_buffer must be > 0 in buffer mode")
        self.s = ProfitLockState(
            armed=True, mode=mode, lock_ratio=lock_ratio,
            give_back_buffer=give_back_buffer,
            activation_floor=activation_floor,
        )

    def _floor_for(self, peak: float) -> float:
        if self.s.mode == "ratio":
            return peak * self.s.lock_ratio
        return peak - self.s.give_back_buffer

    def update(self, current_pnl: float) -> bool:
        """Feed cumulative day P&L. Returns True the instant the lock
        fires (caller flattens everything). Fires at most once."""
        s = self.s
        if not s.armed or s.fired:
            return False
        s.current_pnl = float(current_pnl)
        if current_pnl > s.peak_pnl:
            s.peak_pnl = float(current_pnl)
        # Activate only once peak has cleared the activation floor — we
        # don't lock trivial profits and choke a trade before it breathes.
        if not s.activated and s.peak_pnl >= s.activation_floor:
            s.activated = True
        if s.activated:
            candidate = self._floor_for(s.peak_pnl)
            # Ratchet: floor only ever rises, and never above current peak.
            s.locked_pnl = max(s.locked_pnl, max(0.0, candidate))
            # Fire when current falls to/through the locked floor.
            if s.current_pnl <= s.locked_pnl:
                s.fired = True
                return True
        return False

    def snapshot(self) -> dict:
        s = self.s
        return {
            "armed": s.armed, "fired": s.fired, "activated": s.activated,
            "mode": s.mode, "lock_ratio": s.lock_ratio,
            "give_back_buffer": s.give_back_buffer,
            "current_pnl": round(s.current_pnl, 2),
            "peak_pnl": round(s.peak_pnl, 2),
            "locked_pnl": round(s.locked_pnl, 2),
            "floating_pnl": round(s.floating_pnl, 2),
            "worst_case_day_profit": round(s.worst_case_day_profit, 2),
        }

    def disarm(self) -> None:
        self.s.armed = False
