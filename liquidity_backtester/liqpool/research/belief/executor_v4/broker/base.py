"""Broker adapter base class + order/fill data shapes.

The manager talks to a single ``BrokerAdapter`` interface; concrete
implementations (paper, Kite, etc.) live in sibling modules.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Order states.
STATE_PENDING = "PENDING"
STATE_FILLED = "FILLED"
STATE_PARTIAL = "PARTIAL"
STATE_REJECTED = "REJECTED"
STATE_CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class BrokerOrder:
    """A request to open or close a single leg."""
    client_order_id: str        # caller-generated unique id
    tradingsymbol: str          # e.g. NIFTY24JUN26023000CE
    exchange: str               # NFO / BFO / NSE
    side: str                   # BUY / SELL
    quantity: int               # SHARES, not lots — caller multiplies by lot_size
    order_type: str             # MARKET / LIMIT
    product: str                # MIS / NRML / CNC
    limit_price: Optional[float] = None
    tag: Optional[str] = None
    position_id: Optional[str] = None   # link back to v4 position

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class BrokerOrderResult:
    """Reply from the broker. The adapter handles the round-trip and
    surfaces a fill state."""
    client_order_id: str
    broker_order_id: str
    state: str                  # STATE_*
    filled_quantity: int
    avg_fill_price: float
    rejection_reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    ts_submitted: float = 0.0
    ts_filled: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["raw"] = dict(self.raw)
        return d

    @property
    def is_filled(self) -> bool:
        return self.state in (STATE_FILLED, STATE_PARTIAL) and self.filled_quantity > 0

    @property
    def is_rejected(self) -> bool:
        return self.state == STATE_REJECTED


@dataclass(frozen=True)
class BrokerOrderState:
    """Pollable state for an in-flight order."""
    broker_order_id: str
    state: str
    filled_quantity: int
    avg_fill_price: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerPosition:
    """Snapshot of a broker-side open position."""
    tradingsymbol: str
    exchange: str
    quantity: int                # signed: +long / -short
    avg_price: float
    last_price: float
    unrealized_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class BrokerAdapter:
    """Abstract base for any broker integration.

    Implementations must override:
      * ``place_order(order: BrokerOrder) -> BrokerOrderResult``
      * ``get_order_state(broker_order_id: str) -> Optional[BrokerOrderState]``
      * ``get_positions() -> List[BrokerPosition]``

    The base class provides:
      * ``is_live`` flag (False = paper / dry-run; True = real money)
      * Convenience helpers for risk-gating (e.g. global kill switch).
    """

    name: str = "base"
    is_live: bool = False

    def __init__(self, *, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self.killed = False
        self._submitted_ts: Dict[str, float] = {}

    # ── Required interface ────────────────────────────────────────────

    def place_order(self, order: BrokerOrder) -> BrokerOrderResult:
        raise NotImplementedError

    def get_order_state(self, broker_order_id: str) -> Optional[BrokerOrderState]:
        raise NotImplementedError

    def get_positions(self) -> List[BrokerPosition]:
        return []

    # ── Optional helpers ─────────────────────────────────────────────

    def kill_switch(self, reason: str = "operator triggered") -> None:
        """Disable order placement until further notice."""
        self.killed = True
        self._last_kill_reason = reason

    def healthcheck(self) -> Dict[str, Any]:
        """Returns adapter health diagnostics."""
        return {
            "name": self.name,
            "is_live": self.is_live,
            "dry_run": self.dry_run,
            "killed": self.killed,
        }

    def _mk_client_id(self, position_id: str, tag: str = "") -> str:
        ts = int(time.time() * 1000)
        return f"v4-{position_id[:8]}-{tag}-{ts}"
