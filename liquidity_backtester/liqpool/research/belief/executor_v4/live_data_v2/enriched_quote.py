"""EnrichedQuote — the full microstructure picture per contract.

Captures everything Kite gives us in a single quote payload — 5 levels
of depth on each side, total pending order quantities, OI, last trade
size + time, average price. Used by the new microstructure detectors
which need the unflattened picture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DepthLevel:
    """One side, one level of the L2 book."""
    price: float
    quantity: int
    orders: int

    def to_dict(self) -> Dict[str, Any]:
        return {"price": float(self.price),
                  "quantity": int(self.quantity),
                  "orders": int(self.orders)}

    @classmethod
    def from_kite(cls, raw: Optional[Dict[str, Any]]) -> "DepthLevel":
        r = raw or {}
        return cls(
            price=float(r.get("price") or 0.0),
            quantity=int(r.get("quantity") or 0),
            orders=int(r.get("orders") or 0),
        )


@dataclass
class EnrichedQuote:
    """One contract's full microstructure snapshot."""
    tradingsymbol: str
    ltp: float                    # last traded price
    last_quantity: int            # size of the last trade
    last_trade_time_iso: str = ""  # last trade timestamp (str for JSON)
    average_price: float = 0.0    # VWAP
    volume: float = 0.0
    oi: float = 0.0
    oi_day_high: float = 0.0
    oi_day_low: float = 0.0
    # Total pending order quantities across the chain — the high-level
    # bid/ask pressure that the founder asked us to capture.
    buy_quantity_total: int = 0
    sell_quantity_total: int = 0
    # Five levels each side; index 0 = top of book.
    depth_buy: List[DepthLevel] = field(default_factory=list)
    depth_sell: List[DepthLevel] = field(default_factory=list)

    # ── Derived properties (computed without re-allocation) ────────

    @property
    def bid(self) -> float:
        return self.depth_buy[0].price if self.depth_buy else 0.0

    @property
    def ask(self) -> float:
        return self.depth_sell[0].price if self.depth_sell else 0.0

    @property
    def bid_qty(self) -> int:
        return self.depth_buy[0].quantity if self.depth_buy else 0

    @property
    def ask_qty(self) -> int:
        return self.depth_sell[0].quantity if self.depth_sell else 0

    @property
    def spread(self) -> float:
        if not (self.depth_buy and self.depth_sell):
            return 0.0
        return max(0.0, self.depth_sell[0].price - self.depth_buy[0].price)

    @property
    def mid(self) -> float:
        if not (self.depth_buy and self.depth_sell):
            return self.ltp
        return (self.depth_buy[0].price + self.depth_sell[0].price) / 2.0

    @property
    def total_depth_buy_qty(self) -> int:
        return sum(d.quantity for d in self.depth_buy)

    @property
    def total_depth_sell_qty(self) -> int:
        return sum(d.quantity for d in self.depth_sell)

    @property
    def depth_orders_buy(self) -> int:
        return sum(d.orders for d in self.depth_buy)

    @property
    def depth_orders_sell(self) -> int:
        return sum(d.orders for d in self.depth_sell)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tradingsymbol": self.tradingsymbol,
            "ltp": float(self.ltp),
            "last_quantity": int(self.last_quantity),
            "last_trade_time_iso": self.last_trade_time_iso,
            "average_price": float(self.average_price),
            "volume": float(self.volume),
            "oi": float(self.oi),
            "oi_day_high": float(self.oi_day_high),
            "oi_day_low": float(self.oi_day_low),
            "buy_quantity_total": int(self.buy_quantity_total),
            "sell_quantity_total": int(self.sell_quantity_total),
            "depth_buy": [d.to_dict() for d in self.depth_buy],
            "depth_sell": [d.to_dict() for d in self.depth_sell],
            # Derived for cockpit / downstream consumers.
            "bid": float(self.bid),
            "ask": float(self.ask),
            "spread": float(self.spread),
            "mid": float(self.mid),
            "total_depth_buy_qty": int(self.total_depth_buy_qty),
            "total_depth_sell_qty": int(self.total_depth_sell_qty),
        }


def extract_pressure_ratio(quote: EnrichedQuote) -> float:
    """High-level bid/ask pressure ratio in [-1, +1].

    +1.0 = all pending orders on buy side, -1.0 = all on sell side,
    0.0 = balanced. Computed from buy_quantity_total + sell_quantity_total
    (the chain-wide totals that Kite reports separately from depth).
    Falls back to top-5-depth-summed quantities when the chain totals
    are zero.
    """
    buy_t = quote.buy_quantity_total
    sell_t = quote.sell_quantity_total
    if buy_t == 0 and sell_t == 0:
        buy_t = quote.total_depth_buy_qty
        sell_t = quote.total_depth_sell_qty
    total = buy_t + sell_t
    if total == 0:
        return 0.0
    return float(buy_t - sell_t) / float(total)
