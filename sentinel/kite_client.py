"""Kite Connect client wrapper — rate-limited, dry-run-safe, demo-able.

Three layers in one module:

  RateLimiter   — token buckets per Kite endpoint class. Encodes the
                  documented Kite Connect v3 limits (VERIFY against
                  your plan; kite.trade blocks automated doc fetch):
                    quote/ltp/ohlc :  1 req/s
                    historical     :  3 req/s
                    orders         : 10 req/s, 200/min, 3000/day
                    everything else: 10 req/s
                  Sentinel self-caps far below these.

  KiteAccount   — one account's REST surface: funds, positions,
                  orders, quotes, instruments dump (cached daily),
                  market exit orders. dry_run simulates placements.

  DemoAccount   — synthetic account with a CE+PE pair on random-walk
                  premiums, so the full dashboard runs with zero
                  credentials and the market closed. Same interface.

The KiteTicker WebSocket (3 connections/key, 3000 instruments/conn)
is a planned upgrade; v1 polls quotes at 1 req/s in batches (Kite
allows up to 500 instruments per quote call, 1000 per ltp call),
which is comfortably enough for one account's positions + one
recommendation chain.
"""
from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """Blocking token bucket per endpoint class."""

    LIMITS = {
        "quote": 1.0,        # req/s
        "historical": 3.0,
        "orders": 10.0,
        "other": 10.0,
    }

    def __init__(self) -> None:
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.orders_today = 0
        self.orders_this_minute: List[float] = []

    def acquire(self, kind: str) -> None:
        rate = self.LIMITS.get(kind, 10.0)
        min_gap = 1.0 / rate
        with self._lock:
            now = time.monotonic()
            last = self._last.get(kind, 0.0)
            wait = (last + min_gap) - now
            if wait > 0:
                time.sleep(wait)
            self._last[kind] = time.monotonic()

    def register_order(self, max_per_day: int) -> None:
        """Raises if Sentinel's self-cap or Kite's 200/min would be hit."""
        now = time.monotonic()
        with self._lock:
            self.orders_this_minute = [t for t in self.orders_this_minute
                                       if now - t < 60.0]
            if len(self.orders_this_minute) >= 180:   # cushion under 200/min
                raise RuntimeError("order/min budget nearly exhausted; refusing")
            if self.orders_today >= max_per_day:
                raise RuntimeError(
                    f"sentinel self-cap reached ({max_per_day} orders today)")
            self.orders_this_minute.append(now)
            self.orders_today += 1


# ---------------------------------------------------------------------------
# Shared shapes
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    tradingsymbol: str
    ltp: float
    day_high: float = 0.0
    day_low: float = 0.0
    volume: float = 0.0
    oi: float = 0.0
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class InstrumentMeta:
    tradingsymbol: str
    name: str                 # underlying, e.g. NIFTY
    strike: float
    instrument_type: str      # CE | PE | EQ | FUT
    expiry: Optional[datetime]
    lot_size: int
    exchange: str = "NFO"


# ---------------------------------------------------------------------------
# Real account
# ---------------------------------------------------------------------------

class KiteAccount:
    def __init__(self, label: str, api_key: str, access_token: str,
                 dry_run: bool, limiter: Optional[RateLimiter] = None,
                 max_orders_per_day: int = 50) -> None:
        self.label = label
        self.dry_run = dry_run
        self.limiter = limiter or RateLimiter()
        self.max_orders_per_day = max_orders_per_day
        self._kite = None
        self._api_key, self._token = api_key, access_token
        self._instruments: Dict[str, InstrumentMeta] = {}
        self._instruments_loaded_on: Optional[str] = None

    # -- connection -----------------------------------------------------

    def _conn(self):
        if self._kite is None:
            from kiteconnect import KiteConnect
            self._kite = KiteConnect(api_key=self._api_key)
            self._kite.set_access_token(self._token)
        return self._kite

    # -- account state ----------------------------------------------------

    def funds(self) -> Dict[str, float]:
        self.limiter.acquire("other")
        m = self._conn().margins()
        eq = m.get("equity", {}) or {}
        avail = (eq.get("available") or {})
        util = (eq.get("utilised") or {})
        return {
            "cash": float(avail.get("live_balance", avail.get("cash", 0.0)) or 0.0),
            "utilised": float(util.get("debits", 0.0) or 0.0),
            "net": float(eq.get("net", 0.0) or 0.0),
        }

    def positions(self) -> List[Dict[str, Any]]:
        self.limiter.acquire("other")
        return list((self._conn().positions() or {}).get("net", []))

    def orders(self) -> List[Dict[str, Any]]:
        self.limiter.acquire("other")
        return list(self._conn().orders() or [])

    # -- quotes ------------------------------------------------------------

    def quotes(self, symbols: List[str], exchange: str = "NFO") -> Dict[str, Quote]:
        """Batched full quotes (<=500/call per Kite)."""
        out: Dict[str, Quote] = {}
        for i in range(0, len(symbols), 450):
            batch = [f"{exchange}:{s}" for s in symbols[i:i + 450]]
            self.limiter.acquire("quote")
            raw = self._conn().quote(batch)
            for key, q in (raw or {}).items():
                sym = key.split(":", 1)[-1]
                ohlc = q.get("ohlc") or {}
                depth = q.get("depth") or {}
                buy = (depth.get("buy") or [{}])
                sell = (depth.get("sell") or [{}])
                out[sym] = Quote(
                    tradingsymbol=sym,
                    ltp=float(q.get("last_price") or 0.0),
                    day_high=float(ohlc.get("high") or 0.0),
                    day_low=float(ohlc.get("low") or 0.0),
                    volume=float(q.get("volume") or 0.0),
                    oi=float(q.get("oi") or 0.0),
                    bid=float((buy[0] or {}).get("price") or 0.0),
                    ask=float((sell[0] or {}).get("price") or 0.0),
                )
        return out

    def spot_ltp(self, key: str = "NSE:NIFTY 50") -> Optional[float]:
        self.limiter.acquire("quote")
        try:
            q = self._conn().ltp([key])
            return float(q.get(key, {}).get("last_price") or 0.0) or None
        except Exception:
            return None

    # -- instruments dump ----------------------------------------------------

    def instruments(self, exchange: str = "NFO") -> Dict[str, InstrumentMeta]:
        """Daily-cached instruments dump → metadata by tradingsymbol."""
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if self._instruments_loaded_on == today and self._instruments:
            return self._instruments
        self.limiter.acquire("other")
        rows = self._conn().instruments(exchange) or []
        meta: Dict[str, InstrumentMeta] = {}
        for r in rows:
            try:
                exp = r.get("expiry")
                if isinstance(exp, str) and exp:
                    exp = datetime.fromisoformat(exp)
                if isinstance(exp, datetime) and exp.tzinfo is None:
                    exp = exp.replace(tzinfo=IST)
                meta[r["tradingsymbol"]] = InstrumentMeta(
                    tradingsymbol=r["tradingsymbol"],
                    name=str(r.get("name") or ""),
                    strike=float(r.get("strike") or 0.0),
                    instrument_type=str(r.get("instrument_type") or ""),
                    expiry=exp if isinstance(exp, datetime) else None,
                    lot_size=int(r.get("lot_size") or 0),
                    exchange=str(r.get("exchange") or exchange),
                )
            except Exception:
                continue
        self._instruments = meta
        self._instruments_loaded_on = today
        return meta

    # -- orders ---------------------------------------------------------------

    def place_market_exit(self, tradingsymbol: str, side: str, quantity: int,
                          exchange: str = "NFO") -> Dict[str, Any]:
        summary = {"tradingsymbol": tradingsymbol, "side": side,
                   "quantity": quantity, "exchange": exchange,
                   "order_type": "MARKET", "product": "MIS"}
        if self.dry_run:
            return {**summary, "order_id": f"SIM-{int(time.time()*1000)}",
                    "status": "simulated"}
        self.limiter.register_order(self.max_orders_per_day)
        self.limiter.acquire("orders")
        kite = self._conn()
        oid = kite.place_order(
            variety=kite.VARIETY_REGULAR, exchange=exchange,
            tradingsymbol=tradingsymbol, transaction_type=side,
            quantity=int(quantity), product="MIS", order_type="MARKET",
        )
        return {**summary, "order_id": str(oid), "status": "submitted"}


# ---------------------------------------------------------------------------
# Demo account — full dashboard with zero credentials
# ---------------------------------------------------------------------------

class DemoAccount:
    """Synthetic single account: long one NIFTY CE and one NIFTY PE,
    premiums random-walking around a drifting synthetic spot. Lets the
    operator run and feel the entire dashboard on a weekend."""

    label = "demo"
    dry_run = True

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self.spot = 25000.0
        now = datetime.now(IST)
        days_ahead = (3 - now.weekday()) % 7 or 7      # next Thursday
        self.expiry = (now + timedelta(days=days_ahead)).replace(
            hour=15, minute=30, second=0, microsecond=0)
        self._legs = {
            "NIFTYDEMO25000CE": {"strike": 25000.0, "type": "CE",
                                 "premium": 180.0, "qty": 75, "avg": 150.0},
            "NIFTYDEMO24800PE": {"strike": 24800.0, "type": "PE",
                                 "premium": 95.0, "qty": 75, "avg": 110.0},
        }
        self.orders_log: List[Dict[str, Any]] = []

    def tick(self) -> None:
        drift = self._rng.gauss(0.3, 8.0)
        self.spot = max(1000.0, self.spot + drift)
        for leg in self._legs.values():
            direction = 1.0 if leg["type"] == "CE" else -1.0
            move = direction * drift * 0.45 + self._rng.gauss(0, 1.2)
            leg["premium"] = max(0.5, leg["premium"] + move)

    def funds(self) -> Dict[str, float]:
        return {"cash": 187_450.0, "utilised": 62_550.0, "net": 250_000.0}

    def positions(self) -> List[Dict[str, Any]]:
        self.tick()
        rows = []
        for sym, leg in self._legs.items():
            rows.append({
                "tradingsymbol": sym, "exchange": "NFO", "product": "MIS",
                "quantity": leg["qty"], "average_price": leg["avg"],
                "last_price": round(leg["premium"], 2),
                "pnl": round((leg["premium"] - leg["avg"]) * leg["qty"], 2),
            })
        return rows

    def orders(self) -> List[Dict[str, Any]]:
        return list(self.orders_log)

    def quotes(self, symbols: List[str], exchange: str = "NFO") -> Dict[str, Quote]:
        out = {}
        for s in symbols:
            leg = self._legs.get(s)
            if leg is None:
                continue
            p = leg["premium"]
            out[s] = Quote(s, ltp=round(p, 2),
                           day_high=round(p * 1.25, 2),
                           day_low=round(p * 0.8, 2),
                           volume=1_000_000, oi=2_000_000,
                           bid=round(p - 0.4, 2), ask=round(p + 0.4, 2))
        return out

    def spot_ltp(self, key: str = "NSE:NIFTY 50") -> Optional[float]:
        return round(self.spot, 2)

    def instruments(self, exchange: str = "NFO") -> Dict[str, InstrumentMeta]:
        out = {}
        for sym, leg in self._legs.items():
            out[sym] = InstrumentMeta(sym, "NIFTY", leg["strike"], leg["type"],
                                      self.expiry, 75)
        # A demo chain around spot for the recommender.
        for k in range(-6, 7):
            strike = round((self.spot + k * 100) / 100) * 100
            for t in ("CE", "PE"):
                s = f"NIFTYDEMO{int(strike)}{t}"
                if s not in out:
                    out[s] = InstrumentMeta(s, "NIFTY", float(strike), t,
                                            self.expiry, 75)
        return out

    def chain_quotes(self, metas: List[InstrumentMeta]) -> Dict[str, Quote]:
        """Synthetic chain premiums via BS at a fixed IV + noise."""
        from .greeks import bs_price
        t_years = max((self.expiry - datetime.now(IST)).total_seconds(), 3600) / (365 * 86400)
        out = {}
        for m in metas:
            fair = bs_price(self.spot, m.strike, t_years, 0.14, m.instrument_type)
            noise = self._rng.uniform(0.92, 1.08)
            p = max(0.5, fair * noise)
            out[m.tradingsymbol] = Quote(
                m.tradingsymbol, ltp=round(p, 2),
                day_high=round(p * self._rng.uniform(1.1, 1.6), 2),
                day_low=round(p * 0.75, 2),
                volume=self._rng.randint(50_000, 3_000_000),
                oi=self._rng.randint(100_000, 5_000_000),
                bid=round(p - 0.5, 2), ask=round(p + 0.5, 2))
        return out

    def place_market_exit(self, tradingsymbol: str, side: str, quantity: int,
                          exchange: str = "NFO") -> Dict[str, Any]:
        row = {"tradingsymbol": tradingsymbol, "side": side,
               "quantity": quantity, "exchange": exchange,
               "order_type": "MARKET", "product": "MIS",
               "order_id": f"DEMO-{int(time.time()*1000)}",
               "status": "simulated"}
        self.orders_log.append(row)
        return row
