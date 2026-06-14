"""Paper trading — real live ticks, simulated fills, no broker risk.

The differentiation between RETAIL ("watch the cockpit work") and PRO
("trade with it"): a PaperAccount that reads real Kite quotes (so the
operator sees actual market microstructure) but every market-exit
call is fulfilled by a simulated fill at the last LTP. Journal lands
to ``<journal>/paper_orders.jsonl`` so the operator can review the
day's hypothetical book.

Three modes the server can run in:

  * ``DemoAccount``   — fully synthetic (random walk; for UI demos)
  * ``PaperAccount``  — real LTP feed, simulated fills (THIS module)
  * ``KiteAccount``   — real LTP feed, real broker fills

Paper mode wraps an underlying account that provides quotes/funds/
positions/instruments — usually a ``KiteAccount`` configured with the
operator's credentials but with ``dry_run=True``. That way every read
(quotes, funds, chain) is real-market, only writes (orders) are
intercepted.

Audit notes:
  * Every simulated fill writes to ``paper_orders.jsonl`` with a
    unique paper_order_id, the simulated price (LTP at submit
    time), and the symbol/side/qty.
  * `funds()` is adjusted down by the realized P&L of paper trades
    so the operator sees their hypothetical equity curve, not the
    real account's. Set ``track_pnl=False`` to keep funds raw.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_decl import IOSpec, declare

LOG = logging.getLogger("sentinel.paper")


@dataclass
class PaperFill:
    paper_order_id: str
    ts_utc: str
    tradingsymbol: str
    side: str
    quantity: int
    sim_price: float
    realized_pnl: float = 0.0          # against the operator's open paper book
    note: str = ""

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperPosition:
    """One open paper-trade position the simulator tracks."""
    tradingsymbol: str
    quantity: int                     # signed
    avg_price: float                  # weighted-avg entry the simulator tracks

    def realized_close(self, sim_price: float, close_qty: int) -> float:
        """Compute realized P&L when closing ``close_qty`` units at
        ``sim_price``. ``close_qty`` is signed — positive when we're
        BUYing back a short, negative when we're SELLing a long."""
        # Direction the position is being closed in matches the sign of
        # quantity (long pos => negative close_qty (sell), short pos =>
        # positive close_qty (buy)).
        units_closed = abs(close_qty)
        if self.quantity > 0:
            pnl = (sim_price - self.avg_price) * units_closed
        else:
            pnl = (self.avg_price - sim_price) * units_closed
        return float(pnl)


class PaperAccount:
    """Wraps an underlying account; intercepts ``place_market_exit``
    and writes a simulated fill instead. Every other call delegates to
    the underlying account so positions/funds/instruments/quotes are
    real-market reads."""

    def __init__(self, underlying: Any,
                 journal_path: Path,
                 starting_balance: float = 100000.0,
                 track_pnl: bool = True) -> None:
        self.underlying = underlying
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Per-symbol paper position bookkeeping (separate from any real
        # holdings the underlying account knows about).
        self._book: Dict[str, PaperPosition] = {}
        self._realized_pnl: float = 0.0
        self._starting_balance = float(starting_balance)
        self._track_pnl = track_pnl
        self._orders_log: List[Dict[str, Any]] = []

    # ── identity ──────────────────────────────────────────────────
    @property
    def dry_run(self) -> bool:                          # paper is dry by definition
        return True

    @property
    def label(self) -> str:
        return f"paper({getattr(self.underlying, 'label', 'real')})"

    @property
    def orders_log(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._orders_log)

    # ── read-through to the underlying account ────────────────────
    def funds(self) -> Dict[str, float]:
        try:
            raw = self.underlying.funds() or {}
        except Exception as exc:
            LOG.warning("paper: underlying.funds() failed: %s", exc)
            raw = {}
        if not self._track_pnl:
            return raw
        cash = float(raw.get("cash", self._starting_balance))
        with self._lock:
            equity = cash + self._realized_pnl
            return {
                "cash": equity,
                "utilised": raw.get("utilised", 0.0),
                "paper_realized_pnl": round(self._realized_pnl, 2),
                "paper_starting_balance": self._starting_balance,
            }

    def positions(self) -> List[Dict[str, Any]]:
        return self.underlying.positions()

    def quotes(self, *args, **kwargs):
        return self.underlying.quotes(*args, **kwargs)

    def chain_quotes(self, *args, **kwargs):
        if hasattr(self.underlying, "chain_quotes"):
            return self.underlying.chain_quotes(*args, **kwargs)
        return self.quotes(*args, **kwargs)

    def instruments(self):
        return self.underlying.instruments()

    def spot_ltp(self) -> float:
        return self.underlying.spot_ltp()

    # ── the ONE write path — simulated only ───────────────────────
    def place_market_exit(self, tradingsymbol: str, side: str,
                          quantity: int, exchange: str = "NFO"
                          ) -> Dict[str, Any]:
        """Simulates a fill at the latest LTP of ``tradingsymbol``.
        Updates paper book + writes a row to ``paper_orders.jsonl``."""
        sim_price = self._lookup_ltp(tradingsymbol)
        if sim_price <= 0:
            raise RuntimeError(
                f"paper fill aborted: no live LTP for {tradingsymbol}")
        oid = f"PAPER-{int(time.time() * 1000)}"
        signed_qty = quantity if side.upper() == "BUY" else -quantity
        realized = self._book_update(tradingsymbol, signed_qty, sim_price)
        fill = PaperFill(
            paper_order_id=oid,
            ts_utc=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            tradingsymbol=tradingsymbol, side=side.upper(),
            quantity=quantity, sim_price=sim_price, realized_pnl=realized,
            note="paper",
        )
        self._journal_fill(fill)
        row = {
            "order_id": oid, "status": "simulated_paper",
            "tradingsymbol": tradingsymbol, "side": side.upper(),
            "quantity": quantity, "sim_price": sim_price,
            "realized_pnl": realized,
        }
        with self._lock:
            self._orders_log.append(row)
        return row

    # ── internals ─────────────────────────────────────────────────
    def _lookup_ltp(self, tradingsymbol: str) -> float:
        try:
            q = self.underlying.quotes([tradingsymbol])
        except Exception:
            return 0.0
        if not q:
            return 0.0
        for v in q.values():
            ltp = getattr(v, "ltp", None) or (
                v.get("ltp") if isinstance(v, dict) else None)
            if ltp and float(ltp) > 0:
                return float(ltp)
        return 0.0

    def _book_update(self, sym: str, signed_qty: int, sim_price: float) -> float:
        """Returns realized P&L from this fill (0 on opening trade)."""
        with self._lock:
            pos = self._book.get(sym)
            realized = 0.0
            if pos is None or pos.quantity == 0:
                self._book[sym] = PaperPosition(
                    tradingsymbol=sym, quantity=signed_qty,
                    avg_price=sim_price)
            elif (pos.quantity > 0 and signed_qty > 0) or (pos.quantity < 0 and signed_qty < 0):
                # adding to existing position — recompute avg
                total = pos.quantity + signed_qty
                weighted_total = (pos.avg_price * abs(pos.quantity)
                                  + sim_price * abs(signed_qty))
                pos.avg_price = weighted_total / abs(total)
                pos.quantity = total
            else:
                # closing / flipping
                closing = min(abs(pos.quantity), abs(signed_qty))
                realized = pos.realized_close(
                    sim_price, closing if pos.quantity > 0 else -closing)
                self._realized_pnl += realized
                remaining = abs(pos.quantity) - closing
                if remaining == 0:
                    sign_left = abs(signed_qty) - closing
                    if sign_left > 0:
                        # flipped
                        self._book[sym] = PaperPosition(
                            tradingsymbol=sym,
                            quantity=(sign_left if signed_qty > 0 else -sign_left),
                            avg_price=sim_price)
                    else:
                        del self._book[sym]
                else:
                    pos.quantity = (remaining if pos.quantity > 0 else -remaining)
            return realized

    def _journal_fill(self, fill: PaperFill) -> None:
        try:
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(fill.to_row()) + "\n")
        except Exception as exc:
            LOG.warning("paper: journal write failed: %s", exc)

    def realized_pnl(self) -> float:
        with self._lock:
            return self._realized_pnl

    def open_book(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [asdict(p) for p in self._book.values()]


declare(IOSpec(
    module="sentinel.paper",
    purpose="paper trading — real live ticks, simulated fills, no broker "
            "risk. Wraps a KiteAccount (or any account); reads quotes/"
            "positions/funds/chain unchanged; intercepts place_market_exit "
            "to simulate the fill at the last LTP, write a row to "
            "paper_orders.jsonl, track realized P&L. The differentiation "
            "between RETAIL ('watch') and PRO ('trade')",
    inputs=["underlying account (KiteAccount or DemoAccount)",
            "journal_path for paper_orders.jsonl",
            "starting_balance + track_pnl flags"],
    outputs=["simulated fills written to paper_orders.jsonl",
             "PaperAccount.realized_pnl() + open_book() for the cockpit"],
    consumes_from=["sentinel.kite_client (read-through)"],
    produces_for=["sentinel.server (paper mode)",
                  "sentinel.trails (exit path)"],
    tier="TRUSTED",
))
