"""Paper broker — fills happen at the requested premium with zero slippage.

This is the SAFE default. Tracks order ledger in memory so the cockpit
shows realistic fill activity, but no real money moves.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import (
    BrokerAdapter,
    BrokerOrder,
    BrokerOrderResult,
    BrokerOrderState,
    BrokerPosition,
    STATE_FILLED,
    STATE_REJECTED,
)


@dataclass
class _PaperFill:
    order: BrokerOrder
    avg_price: float
    ts: float
    broker_order_id: str


class PaperBrokerAdapter(BrokerAdapter):
    """Paper trading adapter — instant fills at the limit price (or a
    caller-provided ``mark_price`` lookup for market orders)."""

    name = "paper"
    is_live = False

    def __init__(self, *,
                  mark_lookup=None,
                  default_slippage_bps: float = 0.0,
                  reject_when_killed: bool = True,
                  starting_capital_rupees: float = 50_000.0) -> None:
        super().__init__(dry_run=True)
        self.starting_capital_rupees = float(starting_capital_rupees)
        self.realized_pnl_rupees = 0.0
        self.mark_lookup = mark_lookup    # callable: tradingsymbol -> price
        self.default_slippage_bps = default_slippage_bps
        self.reject_when_killed = reject_when_killed
        self.fills: List[_PaperFill] = []
        self._positions: Dict[str, int] = {}     # tradingsymbol -> signed shares
        self._avg_price: Dict[str, float] = {}
        self._last_price: Dict[str, float] = {}

    def update_mark(self, tradingsymbol: str, price: float) -> None:
        """Optionally let the runner push a fresh quote in for unrealized
        P&L tracking."""
        self._last_price[tradingsymbol] = price

    def place_order(self, order: BrokerOrder) -> BrokerOrderResult:
        ts_sub = time.time()
        if self.killed and self.reject_when_killed:
            return BrokerOrderResult(
                client_order_id=order.client_order_id,
                broker_order_id="",
                state=STATE_REJECTED,
                filled_quantity=0, avg_fill_price=0.0,
                rejection_reason="broker kill switch active",
                ts_submitted=ts_sub,
            )

        # Determine fill price.
        price: Optional[float] = None
        if order.order_type == "LIMIT" and order.limit_price is not None:
            price = order.limit_price
        elif self.mark_lookup is not None:
            try:
                price = float(self.mark_lookup(order.tradingsymbol) or 0)
            except Exception:
                price = None
        if price is None or price <= 0:
            return BrokerOrderResult(
                client_order_id=order.client_order_id,
                broker_order_id="",
                state=STATE_REJECTED,
                filled_quantity=0, avg_fill_price=0.0,
                rejection_reason="no mark available for paper fill",
                ts_submitted=ts_sub,
            )

        # Apply slippage to market orders.
        slip = 1.0
        if order.order_type == "MARKET" and self.default_slippage_bps > 0:
            bps = self.default_slippage_bps / 10000.0
            slip = (1.0 + bps) if order.side == "BUY" else (1.0 - bps)
        fill_price = max(0.01, price * slip)

        broker_id = f"PAPER-{int(ts_sub * 1000)}-{order.client_order_id[-6:]}"
        self.fills.append(_PaperFill(order=order, avg_price=fill_price,
                                       ts=ts_sub, broker_order_id=broker_id))

        # Update internal position.
        signed_qty = order.quantity if order.side == "BUY" else -order.quantity
        prev_qty = self._positions.get(order.tradingsymbol, 0)
        prev_avg = self._avg_price.get(order.tradingsymbol, 0.0)
        new_qty = prev_qty + signed_qty
        if prev_qty == 0 or (prev_qty * signed_qty < 0
                              and abs(signed_qty) >= abs(prev_qty)):
            # Open / flip
            self._avg_price[order.tradingsymbol] = fill_price
        else:
            # Adjust avg: weighted of existing + new
            total_value = prev_avg * abs(prev_qty) + fill_price * abs(signed_qty)
            total_qty = abs(prev_qty) + abs(signed_qty)
            self._avg_price[order.tradingsymbol] = total_value / max(1, total_qty)
        self._positions[order.tradingsymbol] = new_qty
        if new_qty == 0:
            self._avg_price.pop(order.tradingsymbol, None)

        ts_fill = time.time()
        return BrokerOrderResult(
            client_order_id=order.client_order_id,
            broker_order_id=broker_id,
            state=STATE_FILLED,
            filled_quantity=order.quantity,
            avg_fill_price=fill_price,
            ts_submitted=ts_sub,
            ts_filled=ts_fill,
            raw={"mode": "paper", "applied_slippage_bps": self.default_slippage_bps},
        )

    def get_order_state(self, broker_order_id: str) -> Optional[BrokerOrderState]:
        for f in self.fills:
            if f.broker_order_id == broker_order_id:
                return BrokerOrderState(
                    broker_order_id=broker_order_id,
                    state=STATE_FILLED,
                    filled_quantity=f.order.quantity,
                    avg_fill_price=f.avg_price,
                )
        return None

    def get_positions(self) -> List[BrokerPosition]:
        out: List[BrokerPosition] = []
        for sym, qty in self._positions.items():
            if qty == 0:
                continue
            avg = self._avg_price.get(sym, 0.0)
            last = self._last_price.get(sym, avg)
            unrealized = (last - avg) * qty
            out.append(BrokerPosition(
                tradingsymbol=sym, exchange="NFO",
                quantity=qty, avg_price=avg, last_price=last,
                unrealized_pnl=unrealized,
            ))
        return out

    def get_capital(self) -> Dict[str, float]:
        # Used margin = sum of |qty| × avg over open positions.
        used = 0.0
        for sym, qty in self._positions.items():
            avg = self._avg_price.get(sym, 0.0)
            used += abs(qty) * max(avg, 0.0)
        available = max(0.0, self.starting_capital_rupees
                         + self.realized_pnl_rupees - used)
        current_total = (self.starting_capital_rupees
                          + self.realized_pnl_rupees)
        return {
            "starting_capital_rupees": round(self.starting_capital_rupees, 2),
            "available_rupees": round(available, 2),
            "used_margin_rupees": round(used, 2),
            "current_total_rupees": round(current_total, 2),
        }

    def credit_realized_pnl(self, amount: float) -> None:
        """Hook used by V4Runner.on_tick to feed manager.daily_pnl deltas
        back into the paper broker so its capital projection reflects
        the actual ledger outcomes."""
        if amount is None:
            return
        try:
            self.realized_pnl_rupees += float(amount)
        except (TypeError, ValueError):
            pass

    def healthcheck(self) -> Dict[str, Any]:
        base = super().healthcheck()
        base.update({
            "n_fills": len(self.fills),
            "n_open_symbols": sum(1 for q in self._positions.values() if q != 0),
        })
        return base
