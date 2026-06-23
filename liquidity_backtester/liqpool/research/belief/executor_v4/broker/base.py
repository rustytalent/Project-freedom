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

    def get_capital(self) -> Dict[str, float]:
        """Return account capital: starting / available / used / current.

        Default implementation returns zeros — subclasses (paper + Kite)
        override with the real numbers. Always returns a dict so the UI
        and tests can render it without conditional shape checks.
        """
        return {
            "starting_capital_rupees": 0.0,
            "available_rupees": 0.0,
            "used_margin_rupees": 0.0,
            "current_total_rupees": 0.0,
        }

    # ── Multi-leg bundle support (founder 2026-06-22 Tier-2) ─────────

    def place_multi_leg_bundle(self, bundle: Any, *,
                                  lot_size: int,
                                  position_id: Optional[str] = None,
                                  ) -> Any:
        """Submit every leg of a MultiLegBundle and reconcile partial
        fills.

        The default implementation fans the legs out through
        ``place_order`` sequentially: each leg becomes a LIMIT order
        using the leg's ``limit_price`` (or its ``estimated_premium`` as
        a fallback). After all results are in, partial-fill
        reconciliation is invoked — if any leg failed, the legs that
        successfully filled are reversed.

        Returns a ``BundleReconciliationOutcome`` dataclass; the manager
        reads ``outcome.outcome`` (OK / ROLLED_BACK / REJECTED) to
        decide whether the bundle is OPEN or needs to be discarded.
        """
        # Local imports keep the broker.base file free of multi_leg
        # cross-deps at import time. The broker symbols are local to
        # this module, so we use them directly.
        from ..multi_leg.reconciliation import (
            BundleReconciliationOutcome, RECON_REJECTED,
            reconcile_partial_fills,
        )

        if self.killed:
            return BundleReconciliationOutcome(
                outcome=RECON_REJECTED, bundle=bundle,
                filled_legs=[], rejected_legs=list(range(len(bundle.legs))),
                notes=["broker kill switch active"],
            )

        results: List[BrokerOrderResult] = []
        for i, leg in enumerate(bundle.legs):
            client_id = self._mk_client_id(
                position_id or bundle.bundle_id,
                tag=f"leg{i}",
            )
            limit = (leg.limit_price if leg.limit_price is not None
                     else leg.estimated_premium)
            if limit is None or limit <= 0:
                results.append(BrokerOrderResult(
                    client_order_id=client_id,
                    broker_order_id="",
                    state=STATE_REJECTED,
                    filled_quantity=0, avg_fill_price=0.0,
                    rejection_reason="leg missing limit/estimated_premium",
                ))
                continue
            order = BrokerOrder(
                client_order_id=client_id,
                tradingsymbol=leg.tradingsymbol,
                exchange="NFO",
                side=leg.side,
                quantity=int(leg.lots * lot_size),
                order_type="LIMIT",
                product="MIS",
                limit_price=float(limit),
                tag=f"bundle:{bundle.structure_class}:{leg.role}",
                position_id=position_id or bundle.bundle_id,
            )
            try:
                r = self.place_order(order)
            except Exception as exc:
                r = BrokerOrderResult(
                    client_order_id=client_id,
                    broker_order_id="",
                    state=STATE_REJECTED,
                    filled_quantity=0, avg_fill_price=0.0,
                    rejection_reason=f"adapter exception: {exc}",
                )
            results.append(r)

        outcome = reconcile_partial_fills(
            bundle=bundle, results=results, lot_size=lot_size,
        )
        # If we need to roll back, submit the reverse orders now.
        if outcome.reverse_orders:
            for rl in outcome.reverse_orders:
                limit = (rl.limit_price if rl.limit_price is not None
                         else rl.estimated_premium)
                if limit is None or limit <= 0:
                    outcome.notes.append(
                        f"reverse leg {rl.role} skipped: no limit price")
                    continue
                client_id = self._mk_client_id(
                    position_id or bundle.bundle_id,
                    tag=f"reverse_{rl.role}",
                )
                rev_order = BrokerOrder(
                    client_order_id=client_id,
                    tradingsymbol=rl.tradingsymbol,
                    exchange="NFO",
                    side=rl.side,
                    quantity=int(rl.lots * lot_size),
                    order_type="LIMIT",
                    product="MIS",
                    limit_price=float(limit),
                    tag=f"bundle_reverse:{bundle.structure_class}:{rl.role}",
                    position_id=position_id or bundle.bundle_id,
                )
                try:
                    rev_result = self.place_order(rev_order)
                    outcome.notes.append(
                        f"reverse {rl.role}: state={rev_result.state}")
                except Exception as exc:
                    outcome.notes.append(
                        f"reverse {rl.role} EXCEPTION: {exc} — "
                        f"OPEN ALERT: partial bundle position remains!")
        return outcome

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
