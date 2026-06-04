"""Track 5: Zerodha Kite Connect adapter.

SAFETY DESIGN:
  - dry_run=True is the DEFAULT. All write operations are logged but never sent to Kite
    unless dry_run is explicitly disabled.
  - Read operations (positions, LTP, holdings) work in either mode.
  - Even with dry_run=False, every order placement requires the caller to acknowledge
    via explicit `confirm=True` flag — no implicit auto-execution.

CREDENTIALS:
  - Never hardcode api_key / api_secret / access_token in source files.
  - Pass via constructor args, env vars (KITE_API_KEY, KITE_ACCESS_TOKEN), or a config file
    OUTSIDE the repo.
  - access_token expires daily (~ at midnight). The caller is responsible for refresh —
    typically via a short login script that handles the Kite OAuth flow once per morning.

USAGE:
    from liqpool.broker_zerodha import ZerodhaBroker, ZerodhaConfig
    broker = ZerodhaBroker(ZerodhaConfig(
        api_key=os.environ["KITE_API_KEY"],
        access_token=os.environ["KITE_ACCESS_TOKEN"],
        dry_run=True,                                 # default — safe
    ))
    positions = broker.get_positions()                 # works in dry run too
    ltp = broker.get_ltp(["NSE:HDFCBANK"])             # works in dry run too
    result = broker.place_bracket_order(               # dry-runs unless confirm=True
        tradingsymbol="HDFCBANK", exchange="NSE",
        side="BUY", quantity=10,
        entry_price=763.04, stop=761.34, target=766.44,
    )

The KiteConnect library is an optional dependency — this module imports it lazily so users
not yet trading live can still use the journal / sizing / analysis modules.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
import time


@dataclass
class ZerodhaConfig:
    api_key: str
    access_token: str
    dry_run: bool = True
    """Default ON: orders are simulated, never sent. Set to False explicitly to go live."""
    product: str = "MIS"
    """MIS = intraday margin (Indian retail leverage). CNC = cash-and-carry (long-term)."""
    variety: str = "regular"
    exchange_default: str = "NSE"
    fail_closed_on_read_error: bool = True
    """In live mode, broker read failures raise instead of returning empty fallbacks."""


@dataclass
class OrderResult:
    order_id: str
    status: str                       # "simulated" | "submitted" | "rejected" | "complete"
    dry_run: bool = True
    raw_response: Optional[Dict] = None
    error: Optional[str] = None


class ZerodhaBroker:
    """Lazy-import wrapper around Kite Connect. Safe by default."""

    def __init__(self, config: ZerodhaConfig):
        self.config = config
        self._kite = None
        # Lazy connection: we only instantiate KiteConnect when a method that needs it is called.

    def _conn(self):
        if self._kite is None:
            try:
                from kiteconnect import KiteConnect
            except ImportError as e:
                raise ImportError(
                    "Zerodha integration needs `kiteconnect`. Install with "
                    "`pip install kiteconnect`."
                ) from e
            self._kite = KiteConnect(api_key=self.config.api_key)
            self._kite.set_access_token(self.config.access_token)
        return self._kite

    def _read_failure(self, method: str, exc: Exception, fallback):
        msg = f"[zerodha] {method} failed: {exc}"
        print(msg)
        if self.config.fail_closed_on_read_error and not self.config.dry_run:
            raise RuntimeError(msg) from exc
        return fallback

    # ------------------------------------------------------------------
    # READ-ONLY  (work in both dry and live modes)
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict]:
        """Returns Kite's `net` positions list."""
        try:
            return list(self._conn().positions().get("net", []))
        except Exception as e:
            return self._read_failure("get_positions", e, [])

    def get_holdings(self) -> List[Dict]:
        try:
            return list(self._conn().holdings())
        except Exception as e:
            return self._read_failure("get_holdings", e, [])

    def get_ltp(self, instruments: List[str]) -> Dict[str, float]:
        """`instruments` = ["NSE:HDFCBANK", "NSE:ICICIBANK", ...]. Returns {sym: ltp}.
        Kite's response has shape {sym: {instrument_token, last_price}}; we flatten."""
        try:
            resp = self._conn().ltp(instruments)
            return {k: float(v.get("last_price", 0.0)) for k, v in resp.items()}
        except Exception as e:
            return self._read_failure("get_ltp", e, {})

    def get_orders(self) -> List[Dict]:
        try:
            return list(self._conn().orders())
        except Exception as e:
            return self._read_failure("get_orders", e, [])

    def get_margins(self) -> Dict:
        try:
            return dict(self._conn().margins())
        except Exception as e:
            return self._read_failure("get_margins", e, {})

    # ------------------------------------------------------------------
    # WRITE  (gated by dry_run + confirm)
    # ------------------------------------------------------------------

    def place_bracket_order(self, *, tradingsymbol: str, side: str, quantity: int,
                             entry_price: float, stop: float, target: float,
                             exchange: Optional[str] = None,
                             confirm: bool = False) -> OrderResult:
        """Place a Cover Order with stop-loss + target. Returns OrderResult.

        Kite supports several order varieties. We use REGULAR with SL-M trigger separately
        when bracket orders aren't available for the segment.

        `side`: "BUY" or "SELL".
        `confirm`: must be True AND dry_run False for the order to actually transmit.
        """
        exchange = exchange or self.config.exchange_default

        if side not in ("BUY", "SELL"):
            return OrderResult("", "rejected", dry_run=self.config.dry_run,
                                error=f"invalid side {side!r}")

        order_summary = {
            "tradingsymbol": tradingsymbol, "exchange": exchange, "side": side,
            "quantity": quantity, "entry": entry_price, "stop": stop, "target": target,
            "product": self.config.product,
        }

        # DRY-RUN: log + return simulated order
        if self.config.dry_run or not confirm:
            reason = "dry_run=True" if self.config.dry_run else "confirm=False"
            sim_id = f"SIM-{int(time.time() * 1000)}"
            print(f"[zerodha] SIMULATED ({reason}): {order_summary}")
            return OrderResult(sim_id, "simulated", dry_run=True,
                                raw_response=order_summary)

        # LIVE PLACEMENT — REGULAR entry order, then IMMEDIATELY attach a protective SL-M leg.
        # A "bracket" must never leave a naked, unprotected position: if the stop leg can't be
        # placed we cancel the entry and report a rejection. (Kite removed CO/BO for most
        # segments in 2020; entry + follow-up SL-M is the supported pattern now.)
        exit_side = "SELL" if side == "BUY" else "BUY"
        try:
            kite = self._conn()

            # Main entry: limit order at entry_price.
            entry_resp = kite.place_order(
                variety=self.config.variety,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=side,
                quantity=int(quantity),
                product=self.config.product,
                order_type="LIMIT",
                price=float(entry_price),
            )
            entry_order_id = entry_resp if isinstance(entry_resp, str) else \
                              entry_resp.get("order_id") or ""
        except Exception as e:
            return OrderResult("", "rejected", dry_run=False, error=f"entry failed: {e}",
                                raw_response=order_summary)

        # Protective stop-loss leg (SL-M at the stop trigger, exit side).
        try:
            sl_resp = kite.place_order(
                variety=self.config.variety,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=exit_side,
                quantity=int(quantity),
                product=self.config.product,
                order_type="SL-M",
                trigger_price=float(stop),
            )
            sl_order_id = sl_resp if isinstance(sl_resp, str) else sl_resp.get("order_id") or ""
        except Exception as e:
            # Could not protect the position — undo the entry so we never sit naked.
            try:
                kite.cancel_order(variety=self.config.variety, order_id=entry_order_id)
                undo = f"entry {entry_order_id} cancelled"
            except Exception as ce:
                undo = f"WARNING: entry {entry_order_id} could NOT be cancelled: {ce}"
            return OrderResult(entry_order_id, "rejected", dry_run=False,
                                error=f"stop-loss leg failed ({e}); {undo}",
                                raw_response={"entry_order_id": entry_order_id, **order_summary})

        # NOTE: target is left to the caller's monitor loop (or a GTT) — the SL leg is the
        # non-negotiable protection; the target is an optimisation.
        return OrderResult(
            entry_order_id, "submitted", dry_run=False,
            raw_response={"entry_order_id": entry_order_id, "sl_order_id": sl_order_id,
                          **order_summary},
        )

    def place_stop_loss(self, *, tradingsymbol: str, side: str, quantity: int,
                         trigger_price: float, exchange: Optional[str] = None,
                         confirm: bool = False) -> OrderResult:
        """Place an SL-M order to protect an existing position. `side` is the EXIT side
        (opposite of the entry — SELL if you bought, BUY if you sold).

        Typically called once the entry order has filled."""
        exchange = exchange or self.config.exchange_default
        summary = {"tradingsymbol": tradingsymbol, "exchange": exchange,
                    "side": side, "quantity": quantity, "trigger": trigger_price,
                    "product": self.config.product, "order_type": "SL-M"}
        if self.config.dry_run or not confirm:
            sim_id = f"SIM-SL-{int(time.time() * 1000)}"
            print(f"[zerodha] SIMULATED SL: {summary}")
            return OrderResult(sim_id, "simulated", dry_run=True, raw_response=summary)
        try:
            kite = self._conn()
            resp = kite.place_order(
                variety=self.config.variety, exchange=exchange,
                tradingsymbol=tradingsymbol, transaction_type=side,
                quantity=int(quantity), product=self.config.product,
                order_type="SL-M", trigger_price=float(trigger_price),
            )
            oid = resp if isinstance(resp, str) else resp.get("order_id", "")
            return OrderResult(oid, "submitted", dry_run=False, raw_response={**summary,
                                                                              "order_id": oid})
        except Exception as e:
            return OrderResult("", "rejected", dry_run=False, error=str(e), raw_response=summary)

    def cancel_order(self, order_id: str, variety: Optional[str] = None,
                      confirm: bool = False) -> bool:
        if self.config.dry_run or not confirm:
            print(f"[zerodha] SIMULATED cancel: {order_id}")
            return True
        try:
            self._conn().cancel_order(variety=variety or self.config.variety,
                                       order_id=order_id)
            return True
        except Exception as e:
            print(f"[zerodha] cancel_order failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Convenience: build a broker from environment variables (the safe default)
# ---------------------------------------------------------------------------

def broker_from_env(dry_run: bool = True) -> Optional[ZerodhaBroker]:
    """Construct a ZerodhaBroker from KITE_API_KEY and KITE_ACCESS_TOKEN env vars. Returns
    None if either is missing (callers can then run in journal-only mode)."""
    api_key = os.environ.get("KITE_API_KEY")
    token = os.environ.get("KITE_ACCESS_TOKEN")
    if not api_key or not token:
        return None
    return ZerodhaBroker(ZerodhaConfig(api_key=api_key, access_token=token, dry_run=dry_run))


# ---------------------------------------------------------------------------
# Standalone Kite login script — generates an access_token from request_token.
# Used once per morning. See examples/zerodha_login.py.
# ---------------------------------------------------------------------------

def generate_access_token(api_key: str, api_secret: str, request_token: str) -> str:
    """Exchange a request_token (from Kite's OAuth redirect URL) for an access_token.
    Caller saves this and exports as KITE_ACCESS_TOKEN before running the live runner."""
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)
    return data["access_token"]
