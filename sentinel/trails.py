"""Tick-driven trailing-stop engine + portfolio-level profit lock.

Port of the research repo's trailing bot, adapted for Sentinel:
  * on_tick(symbol, ltp) driven by the quote loop (no internal poller)
  * exits go through a callable executor (the account's
    place_market_exit), so the engine is broker-agnostic and the
    tests run on fakes
  * adds the PORTFOLIO trail: protect total day P&L — when total
    P&L gives back ≥ cushion from its session peak, flatten every
    open MIS option position. Armed explicitly, like everything.

The per-position contract is unchanged from the spec the founder set:
rupee cushion per position, peak tracked from arm-time premium,
exits at market, journal-backed crash recovery, no double fire.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

IST = timezone(timedelta(hours=5, minutes=30))
MIS_AUTO_FLATTEN_IST = dtime(15, 14)
MIN_CUSHION = 0.05

ExitFn = Callable[[str, str, int], Dict[str, Any]]   # (symbol, side, qty)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Trail:
    trail_id: str
    tradingsymbol: str
    side: str                       # side of the OPEN position: long|short
    quantity: int
    cushion_rupees: float
    peak_premium: float             # trough for shorts
    last_premium: float
    state: str = "ARMED"            # ARMED | EXITED | CANCELLED | ERROR
    exit_reason: Optional[str] = None     # TRIGGER | SQUAREOFF | PORTFOLIO
    exit_order_id: Optional[str] = None
    exit_error: Optional[str] = None
    armed_at_utc: str = field(default_factory=_utc_now)

    def serialise(self) -> Dict[str, Any]:
        return asdict(self)


class TrailEngine:
    def __init__(self, journal_path: Path, exit_fn: ExitFn,
                 auto_flatten_ist: dtime = MIS_AUTO_FLATTEN_IST) -> None:
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.exit_fn = exit_fn
        self.auto_flatten_ist = auto_flatten_ist
        self._trails: Dict[str, Trail] = {}
        self._lock = threading.Lock()
        # Portfolio trail
        self.portfolio_cushion: Optional[float] = None
        self.portfolio_peak_pnl: float = float("-inf")
        self.portfolio_fired: bool = False
        self._recover()

    # -- journal ---------------------------------------------------------

    def _write(self, t: Trail) -> None:
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(t.serialise()) + "\n")

    def _recover(self) -> None:
        if not self.journal_path.exists():
            return
        latest: Dict[str, Trail] = {}
        for line in self.journal_path.read_text().splitlines():
            try:
                d = json.loads(line)
                latest[d["trail_id"]] = Trail(**d)
            except Exception:
                continue
        with self._lock:
            self._trails = {tid: t for tid, t in latest.items()
                            if t.state == "ARMED"}

    # -- per-position API ----------------------------------------------------

    def arm(self, tradingsymbol: str, side: str, quantity: int,
            cushion_rupees: float, current_premium: float) -> Trail:
        if side not in ("long", "short"):
            raise ValueError("side must be long|short")
        if quantity <= 0 or current_premium <= 0:
            raise ValueError("quantity and premium must be positive")
        if cushion_rupees < MIN_CUSHION:
            raise ValueError(f"cushion must be >= {MIN_CUSHION}")
        t = Trail(
            trail_id=f"TR_{tradingsymbol}_{int(time.time()*1000)}",
            tradingsymbol=tradingsymbol, side=side, quantity=int(quantity),
            cushion_rupees=float(cushion_rupees),
            peak_premium=float(current_premium),
            last_premium=float(current_premium),
        )
        with self._lock:
            self._trails[t.trail_id] = t
        self._write(t)
        return t

    def cancel(self, trail_id: str) -> bool:
        with self._lock:
            t = self._trails.get(trail_id)
            if t is None or t.state != "ARMED":
                return False
            t.state = "CANCELLED"
            self._trails.pop(trail_id, None)
        self._write(t)
        return True

    def update_cushion(self, trail_id: str, cushion_rupees: float) -> bool:
        if cushion_rupees < MIN_CUSHION:
            raise ValueError("cushion too small")
        with self._lock:
            t = self._trails.get(trail_id)
            if t is None or t.state != "ARMED":
                return False
            t.cushion_rupees = float(cushion_rupees)
        self._write(t)
        return True

    def active(self) -> List[Trail]:
        with self._lock:
            return list(self._trails.values())

    # -- tick path -------------------------------------------------------------

    def on_tick(self, tradingsymbol: str, ltp: float,
                now_ist: Optional[dtime] = None) -> None:
        """Feed one premium tick. Bad ticks (<=0) are ignored entirely."""
        if ltp is None or ltp <= 0:
            return
        now_ist = now_ist or datetime.now(IST).time()
        flatten = now_ist >= self.auto_flatten_ist
        with self._lock:
            trails = [t for t in self._trails.values()
                      if t.tradingsymbol == tradingsymbol and t.state == "ARMED"]
        for t in trails:
            t.last_premium = float(ltp)
            if t.side == "long":
                if ltp > t.peak_premium:
                    t.peak_premium = float(ltp)
                trigger = (t.peak_premium - ltp) >= t.cushion_rupees
            else:
                if ltp < t.peak_premium:
                    t.peak_premium = float(ltp)
                trigger = (ltp - t.peak_premium) >= t.cushion_rupees
            if flatten:
                self._fire(t, "SQUAREOFF")
            elif trigger:
                self._fire(t, "TRIGGER")
            else:
                self._write(t)

    # -- portfolio trail ----------------------------------------------------------

    def arm_portfolio(self, cushion_rupees: float) -> None:
        if cushion_rupees < MIN_CUSHION:
            raise ValueError("cushion too small")
        self.portfolio_cushion = float(cushion_rupees)
        self.portfolio_peak_pnl = float("-inf")
        self.portfolio_fired = False

    def cancel_portfolio(self) -> None:
        self.portfolio_cushion = None
        self.portfolio_fired = False

    def on_portfolio_pnl(self, total_pnl: float,
                         open_positions: List[Dict[str, Any]]) -> bool:
        """Feed the live total day P&L. When the give-back from session
        peak >= cushion, flatten every open position passed in. Returns
        True when the portfolio lock fired this call."""
        if self.portfolio_cushion is None or self.portfolio_fired:
            return False
        if total_pnl > self.portfolio_peak_pnl:
            self.portfolio_peak_pnl = float(total_pnl)
        give_back = self.portfolio_peak_pnl - total_pnl
        if give_back < self.portfolio_cushion:
            return False
        self.portfolio_fired = True
        for p in open_positions:
            qty = int(p.get("quantity") or 0)
            if qty == 0:
                continue
            side = "SELL" if qty > 0 else "BUY"
            try:
                self.exit_fn(str(p["tradingsymbol"]), side, abs(qty))
            except Exception:
                continue
        # Also close out any armed per-position trails (positions gone).
        with self._lock:
            for t in list(self._trails.values()):
                t.state = "EXITED"
                t.exit_reason = "PORTFOLIO"
                self._write(t)
            self._trails.clear()
        return True

    # -- exit -------------------------------------------------------------

    def _fire(self, t: Trail, reason: str) -> None:
        with self._lock:
            if t.state != "ARMED":
                return
            t.state = "EXITED"          # provisional; ERROR on failure
            t.exit_reason = reason
            self._trails.pop(t.trail_id, None)
        side = "SELL" if t.side == "long" else "BUY"
        try:
            resp = self.exit_fn(t.tradingsymbol, side, t.quantity)
            t.exit_order_id = str(resp.get("order_id", ""))
        except Exception as exc:
            t.state = "ERROR"
            t.exit_error = str(exc)
        self._write(t)
