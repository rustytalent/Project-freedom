"""Premium Belief paper ledger.

This is intentionally separate from :mod:`sentinel.paper.PaperAccount`.
``PaperAccount`` wraps real Kite order placement.  The Premium Belief engine
publishes a shadow decision stream first, so the cockpit needs a lightweight
virtual book that can mark those decisions to spot without touching Kite.

The ledger consumes ``premium_belief_engine`` rows from the LivePublisher and
keeps a one-lot index-point portfolio:

* LONG means the belief engine is carrying a call-side / bullish exposure.
* SHORT means put-side / bearish exposure.
* P&L is measured as NIFTY points * lot_size * lots.

It is a visibility layer, not an execution venue.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .live_publisher import ModelSignal

IST = timezone(timedelta(hours=5, minutes=30))


_LONG_ACTIONS = {
    "ENTER_LONG", "SCALP_CALL", "BUY_CALL", "LONG", "CALL", "BULL_ENTRY",
}
_SHORT_ACTIONS = {
    "ENTER_SHORT", "SCALP_PUT", "BUY_PUT", "SHORT", "PUT", "BEAR_ENTRY",
}
_EXIT_PREFIXES = ("EXIT", "FLAT")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _side_from_direction(direction: int) -> str:
    if direction > 0:
        return "LONG"
    if direction < 0:
        return "SHORT"
    return "FLAT"


def _now_ist_hms() -> str:
    return datetime.now(IST).strftime("%H:%M:%S")


def _stale_seconds(ts_ist: str) -> Optional[float]:
    if not ts_ist:
        return None
    try:
        h, m, s = [int(x) for x in ts_ist.split(":")[:3]]
        now = datetime.now(IST)
        ts = now.replace(hour=h, minute=m, second=s, microsecond=0)
        if ts > now + timedelta(hours=1):
            ts -= timedelta(days=1)
        return max(0.0, (now - ts).total_seconds())
    except Exception:
        return None


@dataclass
class BeliefPaperPosition:
    side: str
    entry_spot: float
    entry_ts_ist: str
    entry_signal: str
    entry_confidence: float
    lots: int = 1
    age_ticks: int = 0

    @property
    def direction(self) -> int:
        return 1 if self.side == "LONG" else -1 if self.side == "SHORT" else 0

    def unrealized_points(self, spot: float) -> float:
        return self.direction * (spot - self.entry_spot)


@dataclass
class BeliefPaperLedger:
    """One-lot virtual portfolio for the Premium Belief stream."""

    enabled: bool = False
    starting_balance: float = 100000.0
    lot_size: int = 65
    lots: int = 1
    min_confidence: float = 0.55
    require_allowed: bool = False
    trade_holds: bool = True
    journal_path: Optional[Path] = None
    position: Optional[BeliefPaperPosition] = None
    realized_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    n_trades: int = 0
    peak_equity: float = 100000.0
    max_drawdown: float = 0.0
    last_spot: Optional[float] = None
    last_signal: Optional[str] = None
    last_event: str = "INIT"
    last_reason: str = ""
    last_reject_reasons: List[str] = field(default_factory=list)
    last_key: str = ""
    last_ts_ist: str = ""
    last_executor: Dict[str, Any] = field(default_factory=dict)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_env(cls, journal_dir: Path) -> "BeliefPaperLedger":
        # Backward compatible: SENTINEL_PAPER=1 should finally make the
        # Premium Belief paper ledger visible in demo, even though the old
        # Kite PaperAccount cannot wrap DemoAccount.
        enabled_default = os.environ.get("SENTINEL_PAPER") == "1"
        enabled = _env_bool("SENTINEL_BELIEF_PAPER", enabled_default)
        balance = _as_float(
            os.environ.get("SENTINEL_BELIEF_PAPER_BALANCE")
            or os.environ.get("SENTINEL_PAPER_BALANCE"),
            100000.0,
        )
        return cls(
            enabled=enabled,
            starting_balance=balance,
            lot_size=max(1, _as_int(os.environ.get("SENTINEL_BELIEF_LOT_SIZE"), 65)),
            lots=max(1, _as_int(os.environ.get("SENTINEL_BELIEF_LOTS"), 1)),
            min_confidence=max(0.0, min(1.0, _as_float(
                os.environ.get("SENTINEL_BELIEF_MIN_CONFIDENCE"), 0.55))),
            require_allowed=_env_bool("SENTINEL_BELIEF_REQUIRE_ALLOWED", False),
            trade_holds=_env_bool("SENTINEL_BELIEF_TRADE_HOLDS", True),
            journal_path=journal_dir / "belief_paper_orders.jsonl",
        )

    def ingest(self, sig: ModelSignal) -> bool:
        """Consume one Premium Belief signal. Returns True if book changed."""
        if not self.enabled or sig.model != "premium_belief_engine":
            return False
        extras = dict(sig.extras or {})
        executor = dict(extras.get("executor") or {})
        spot = self._extract_spot(extras)
        if spot is None or spot <= 0:
            return False
        key = self._dedupe_key(sig, extras, spot)
        if key == self.last_key:
            return False
        self.last_key = key
        self.last_spot = spot
        self.last_signal = sig.signal
        self.last_ts_ist = sig.ts_ist or _now_ist_hms()
        self.last_executor = executor
        self.last_reason = self._reason_text(sig, executor)
        self.last_reject_reasons = list(executor.get("reject_reasons") or [])
        if self.position is not None:
            self.position.age_ticks += 1

        action = str(executor.get("action") or executor.get("intent")
                     or sig.signal or "").upper()
        direction = self._extract_direction(extras, executor)
        confidence = _as_float(executor.get("confidence"), sig.confidence)
        allowed = bool(executor.get("allowed", extras.get("trade_allowed", True)))

        self._mark_drawdown(spot)

        if self.require_allowed and not allowed:
            self.last_event = "BLOCKED_BY_EXECUTOR"
            return False
        if confidence < self.min_confidence and not self._is_exit(action, sig):
            self.last_event = "LOW_CONFIDENCE"
            return False

        desired = self._desired_side(action, direction)
        if desired == "FLAT":
            if self._is_exit(action, sig):
                return self._close(spot, self.last_ts_ist, sig.signal,
                                   confidence, "EXIT")
            self.last_event = "MARK"
            return False

        if self.position is None:
            return self._open(desired, spot, self.last_ts_ist, sig.signal,
                              confidence)
        if self.position.side != desired:
            closed = self._close(spot, self.last_ts_ist, sig.signal,
                                 confidence, "REVERSE_CLOSE")
            opened = self._open(desired, spot, self.last_ts_ist, sig.signal,
                                confidence, "REVERSE_OPEN")
            return closed or opened
        self.last_event = "HOLD_POSITION"
        return False

    def snapshot(self) -> Dict[str, Any]:
        spot = self.last_spot or (self.position.entry_spot if self.position else 0.0)
        unrealized_points = (self.position.unrealized_points(spot)
                             if self.position and spot else 0.0)
        unrealized_pnl = unrealized_points * self.lot_size * self.lots
        total_pnl = self.realized_pnl + unrealized_pnl
        equity = self.starting_balance + total_pnl
        win_rate = (self.wins / self.n_trades) if self.n_trades else 0.0
        executor = dict(self.last_executor or {})
        side = self.position.side if self.position else "FLAT"
        entry_spot = self.position.entry_spot if self.position else None
        entry_ts = self.position.entry_ts_ist if self.position else None
        age = self.position.age_ticks if self.position else 0
        executor_size = _as_float(executor.get("size_fraction"), 1.0)
        paper_size = executor_size if executor_size > 0 else 1.0
        return {
            "enabled": self.enabled,
            "mode": "INDEX_POINT_SHADOW",
            "shadow_only": True,
            "lot_size": self.lot_size,
            "quantity_lots": self.lots,
            "starting_balance": round(self.starting_balance, 2),
            "position_open": self.position is not None,
            "side": side,
            "entry_spot": entry_spot,
            "last_spot": spot or None,
            "entry_ts_ist": entry_ts,
            "last_ts_ist": self.last_ts_ist,
            "position_age_ticks": age,
            "unrealized_points": round(unrealized_points, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "equity": round(equity, 2),
            "n_trades": self.n_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(win_rate, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "last_event": self.last_event,
            "last_signal": self.last_signal,
            "last_reason": self.last_reason,
            "last_reject_reasons": self.last_reject_reasons,
            "executor_intent": executor.get("intent") or executor.get("action") or "",
            "executor_allowed": bool(executor.get("allowed", False)),
            "executor_size_fraction": _as_float(executor.get("size_fraction"), 0.0),
            "paper_size_fraction": paper_size if self.position else 0.0,
            "stale_seconds": _stale_seconds(self.last_ts_ist),
            "updated_at_ist": _now_ist_hms(),
            "recent_trades": list(self.trades[-12:]),
        }

    def _extract_spot(self, extras: Dict[str, Any]) -> Optional[float]:
        spot = _as_float(extras.get("spot"), 0.0)
        if spot > 0:
            return spot
        snap = extras.get("belief_snapshot") or {}
        spot = _as_float(snap.get("spot"), 0.0)
        return spot if spot > 0 else None

    def _extract_direction(self, extras: Dict[str, Any], executor: Dict[str, Any]) -> int:
        for value in [executor.get("direction"), extras.get("direction")]:
            direction = _as_int(value, 0)
            if direction:
                return 1 if direction > 0 else -1
        snap = extras.get("belief_snapshot") or {}
        decision = snap.get("decision") or {}
        direction = _as_int(decision.get("direction"), 0)
        return 1 if direction > 0 else -1 if direction < 0 else 0

    def _desired_side(self, action: str, direction: int) -> str:
        action_u = (action or "").upper()
        if action_u in _LONG_ACTIONS or "ENTER_LONG" in action_u or "CALL" in action_u:
            return "LONG"
        if action_u in _SHORT_ACTIONS or "ENTER_SHORT" in action_u or "PUT" in action_u:
            return "SHORT"
        if self.trade_holds and action_u.startswith("HOLD") and direction:
            return _side_from_direction(direction)
        return "FLAT"

    def _is_exit(self, action: str, sig: ModelSignal) -> bool:
        action_u = (action or "").upper()
        signal_u = (sig.signal or "").upper()
        reasons = " ".join([str(x).upper() for x in sig.reason_codes or []])
        return (
            action_u.startswith(_EXIT_PREFIXES)
            or signal_u.startswith(_EXIT_PREFIXES)
            or "EXIT_BULL" in reasons
            or "EXIT_BEAR" in reasons
        )

    def _open(self, side: str, spot: float, ts_ist: str, signal: str,
              confidence: float, event: str = "OPEN") -> bool:
        self.position = BeliefPaperPosition(
            side=side, entry_spot=spot, entry_ts_ist=ts_ist,
            entry_signal=signal, entry_confidence=confidence,
            lots=self.lots,
        )
        self.last_event = event
        self._journal({
            "event": event, "side": side, "spot": spot, "ts_ist": ts_ist,
            "signal": signal, "confidence": confidence, "lots": self.lots,
            "lot_size": self.lot_size,
        })
        return True

    def _close(self, spot: float, ts_ist: str, signal: str,
               confidence: float, event: str) -> bool:
        if self.position is None:
            self.last_event = "FLAT_EXIT_IGNORED"
            return False
        pos = self.position
        points = pos.unrealized_points(spot)
        pnl = points * self.lot_size * self.lots
        self.realized_pnl += pnl
        self.n_trades += 1
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1
        row = {
            "event": event, "side": pos.side, "entry_spot": pos.entry_spot,
            "exit_spot": spot, "points": round(points, 2),
            "pnl": round(pnl, 2), "ts_ist": ts_ist,
            "entry_ts_ist": pos.entry_ts_ist, "signal": signal,
            "confidence": confidence, "lots": self.lots,
            "lot_size": self.lot_size,
        }
        self.trades.append(row)
        self._journal(row)
        self.position = None
        self.last_event = event
        self._mark_drawdown(spot)
        return True

    def _mark_drawdown(self, spot: float) -> None:
        unrealized = 0.0
        if self.position is not None:
            unrealized = self.position.unrealized_points(spot) * self.lot_size * self.lots
        equity = self.starting_balance + self.realized_pnl + unrealized
        self.peak_equity = max(self.peak_equity, equity)
        self.max_drawdown = min(self.max_drawdown, equity - self.peak_equity)

    def _dedupe_key(self, sig: ModelSignal, extras: Dict[str, Any], spot: float) -> str:
        bars = extras.get("bars_seen") or (extras.get("belief_snapshot") or {}).get("bars_seen")
        return "|".join([
            sig.asset, sig.model, sig.ts_ist or "", str(bars or ""),
            f"{spot:.2f}", sig.signal or "",
        ])

    def _reason_text(self, sig: ModelSignal, executor: Dict[str, Any]) -> str:
        parts: List[str] = []
        parts.extend([str(x) for x in executor.get("reason_codes") or []])
        parts.extend([str(x) for x in executor.get("reject_reasons") or []])
        parts.extend([str(x) for x in sig.reason_codes or []])
        return "; ".join([p for p in parts if p])[:500]

    def _journal(self, row: Dict[str, Any]) -> None:
        if self.journal_path is None:
            return
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:
            # Paper journal must never take down Sentinel.
            pass
