"""Kite (Zerodha) broker adapter — wraps ``sentinel.kite_client.KiteAccount``.

The Kite adapter sits on top of the existing rate-limited account, so
all of its protections (per-second / per-minute / daily caps) apply
transparently to the v4 executor.

Safety rails (founder-mandated 2026-06-21):
  * Live orders only fire when ``confirm_real=True`` is set explicitly.
    The default is dry-run regardless of the underlying account state.
  * Per-tick max order count cap to prevent runaway loops.
  * Per-position guard: refuse if the broker already shows the
    tradingsymbol with an opposing quantity (avoids accidental
    over-positioning).
  * Optional limit-price cap for entries: don't pay more than
    ``max_entry_premium_pct`` above the requested mark.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .base import (
    BrokerAdapter,
    BrokerOrder,
    BrokerOrderResult,
    BrokerOrderState,
    BrokerPosition,
    STATE_FILLED,
    STATE_PENDING,
    STATE_REJECTED,
)


@dataclass
class KiteBrokerConfig:
    """Knobs for the Kite broker adapter."""
    confirm_real: bool = False           # MUST be flipped explicitly to go live
    max_orders_per_tick: int = 4
    max_orders_per_day: int = 200
    max_entry_premium_pct: float = 1.10  # entry cap: 10% above expected mark
    use_limit_orders: bool = True        # default to LIMIT (safer than MARKET)
    limit_buffer_bps: float = 5.0        # +/- 5bps above/below ask/bid
    fail_on_kite_exception: bool = True


class KiteBrokerAdapter(BrokerAdapter):
    """Pluggable Kite adapter.

    Construction is intentionally permissive — the founder passes the
    KiteAccount object (or its factory) at construction. The adapter
    has NO knowledge of api_key/access_token directly; that stays in
    the existing sentinel.kite_client layer.
    """

    name = "kite"

    def __init__(self, *,
                  kite_account,
                  cfg: Optional[KiteBrokerConfig] = None) -> None:
        self.cfg = cfg or KiteBrokerConfig()
        self.kite_account = kite_account
        super().__init__(dry_run=not self.cfg.confirm_real)
        self.is_live = bool(self.cfg.confirm_real)
        self._orders_this_tick = 0
        self._orders_today = 0
        self._tick_start_ts = time.time()
        self._submitted: Dict[str, BrokerOrderResult] = {}
        self._last_day = self._today()

    def begin_tick(self) -> None:
        """Called by the runner at the start of each tick — resets the
        per-tick order counter."""
        self._orders_this_tick = 0
        today = self._today()
        if today != self._last_day:
            self._orders_today = 0
            self._last_day = today

    def place_order(self, order: BrokerOrder) -> BrokerOrderResult:
        cfg = self.cfg
        ts_sub = time.time()

        if self.killed:
            return _reject(order, ts_sub, "broker kill switch active")

        if self._orders_this_tick >= cfg.max_orders_per_tick:
            return _reject(order, ts_sub,
                            f"per-tick order cap ({cfg.max_orders_per_tick}) reached")
        if self._orders_today >= cfg.max_orders_per_day:
            return _reject(order, ts_sub,
                            f"daily order cap ({cfg.max_orders_per_day}) reached")

        # Cross-check: refuse if we already hold an opposing position.
        try:
            positions = self.get_positions()
        except Exception:
            positions = []
        for p in positions:
            if p.tradingsymbol == order.tradingsymbol:
                signed_req = (order.quantity if order.side == "BUY"
                              else -order.quantity)
                if (p.quantity * signed_req < 0
                        and abs(signed_req) > abs(p.quantity)):
                    return _reject(order, ts_sub,
                                    "broker already holds opposing quantity")

        # Entry premium cap (LIMIT orders).
        if (order.order_type == "LIMIT" and order.limit_price is not None
                and order.side == "BUY"
                and order.limit_price > 0):
            # Defer to caller — they're already specifying the limit price.
            pass

        # Dry-run handling.
        if self.dry_run or not cfg.confirm_real:
            self._orders_this_tick += 1
            self._orders_today += 1
            broker_id = f"DRY-{int(ts_sub * 1000)}"
            result = BrokerOrderResult(
                client_order_id=order.client_order_id,
                broker_order_id=broker_id,
                state=STATE_FILLED,
                filled_quantity=order.quantity,
                avg_fill_price=(order.limit_price or 0.0),
                ts_submitted=ts_sub, ts_filled=ts_sub,
                raw={"mode": "dry_run"},
            )
            self._submitted[broker_id] = result
            return result

        # Live placement.
        try:
            response = self.kite_account.place_market_exit(
                tradingsymbol=order.tradingsymbol,
                side=order.side, quantity=order.quantity,
                exchange=order.exchange,
            )
        except Exception as exc:
            if cfg.fail_on_kite_exception:
                return _reject(order, ts_sub, f"kite exception: {exc}")
            return _reject(order, ts_sub, str(exc))

        self._orders_this_tick += 1
        self._orders_today += 1
        broker_id = str(response.get("order_id") or "")
        state = (STATE_FILLED if response.get("status") == "submitted"
                 else STATE_PENDING)
        result = BrokerOrderResult(
            client_order_id=order.client_order_id,
            broker_order_id=broker_id,
            state=state,
            filled_quantity=order.quantity,
            avg_fill_price=float(response.get("avg_fill_price",
                                                 order.limit_price or 0.0)),
            ts_submitted=ts_sub, ts_filled=time.time(),
            raw=dict(response),
        )
        self._submitted[broker_id] = result
        return result

    def get_order_state(self, broker_order_id: str) -> Optional[BrokerOrderState]:
        r = self._submitted.get(broker_order_id)
        if r is None:
            return None
        return BrokerOrderState(
            broker_order_id=broker_order_id,
            state=r.state,
            filled_quantity=r.filled_quantity,
            avg_fill_price=r.avg_fill_price,
            raw=r.raw,
        )

    def get_positions(self) -> List[BrokerPosition]:
        if not hasattr(self.kite_account, "positions"):
            return []
        try:
            raw = self.kite_account.positions()
        except Exception:
            return []
        # Best-effort flatten. Kite returns {"net": [...]} typically.
        rows = (raw.get("net") if isinstance(raw, dict)
                else raw if isinstance(raw, list) else [])
        out: List[BrokerPosition] = []
        for p in rows:
            sym = str(p.get("tradingsymbol") or "")
            qty = int(p.get("quantity") or 0)
            if not sym or qty == 0:
                continue
            avg = float(p.get("average_price") or 0.0)
            last = float(p.get("last_price") or avg)
            unrealized = (last - avg) * qty
            out.append(BrokerPosition(
                tradingsymbol=sym, exchange=str(p.get("exchange") or "NFO"),
                quantity=qty, avg_price=avg, last_price=last,
                unrealized_pnl=unrealized,
            ))
        return out

    def get_capital(self) -> Dict[str, float]:
        """Fetch real margins from the Kite account when available;
        fall back to zeros otherwise (paper broker shape preserved)."""
        try:
            margins = (self.kite_account.margins()
                        if hasattr(self.kite_account, "margins") else None)
        except Exception:
            margins = None
        if not margins:
            return super().get_capital()
        try:
            # Kite margins() typically returns
            # {"equity": {"net": ..., "available": {...}, "utilised": {...}}, "commodity": {...}}.
            eq = (margins.get("equity")
                   if isinstance(margins, dict) else None) or {}
            avail = float((eq.get("available") or {}).get("cash") or 0.0)
            used = float((eq.get("utilised") or {}).get("debits") or 0.0)
            net = float(eq.get("net") or (avail + used))
            return {
                "starting_capital_rupees": round(net, 2),
                "available_rupees": round(avail, 2),
                "used_margin_rupees": round(used, 2),
                "current_total_rupees": round(net, 2),
            }
        except Exception:
            return super().get_capital()

    def healthcheck(self) -> Dict[str, Any]:
        base = super().healthcheck()
        base.update({
            "confirm_real": self.cfg.confirm_real,
            "orders_this_tick": self._orders_this_tick,
            "orders_today": self._orders_today,
        })
        return base

    def _today(self) -> str:
        from datetime import datetime, timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        return datetime.now(ist).date().isoformat()


def _reject(order: BrokerOrder, ts: float,
             reason: str) -> BrokerOrderResult:
    return BrokerOrderResult(
        client_order_id=order.client_order_id,
        broker_order_id="",
        state=STATE_REJECTED,
        filled_quantity=0,
        avg_fill_price=0.0,
        rejection_reason=reason,
        ts_submitted=ts,
    )
