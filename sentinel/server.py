"""Sentinel server — FastAPI app + the live loop.

One background thread drives everything on a budget that respects
Kite's documented limits with slack to spare:

  every ~1s   quotes for held symbols (one batched call)
  every 15s   positions + funds refresh, portfolio enrichment,
              maximizer run, portfolio-trail check
  every 30s   ledger resolution pass
  every 60s   recommendation chain refresh (one batched quote call)

Auth: if SENTINEL_TOKEN is set, every /api request must carry it in
the X-Sentinel-Token header. The dashboard prompts once and stores
it in localStorage.

Run:
    SENTINEL_DEMO=1 uvicorn sentinel.server:app --port 8800   # demo
    uvicorn sentinel.server:app --host 127.0.0.1 --port 8800  # live (dry-run)
    SENTINEL_CONFIRM_REAL=1 uvicorn ...                       # live (real orders)
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .advisor import Maximizer, SuggestionLedger, recommend_dips
from .config import SentinelConfig
from .kite_client import DemoAccount, KiteAccount, InstrumentMeta
from .portfolio import PortfolioState
from .trails import TrailEngine

LOG = logging.getLogger("sentinel")
IST = timezone(timedelta(hours=5, minutes=30))
STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class Sentinel:
    def __init__(self, cfg: SentinelConfig) -> None:
        self.cfg = cfg
        self.killed = False
        self.activity: List[Dict[str, str]] = []
        if cfg.demo_mode or not cfg.accounts:
            self.account = DemoAccount()
            self.demo = True
        else:
            a = cfg.accounts[0]
            self.account = KiteAccount(a.label, a.api_key, a.access_token,
                                       dry_run=cfg.dry_run,
                                       max_orders_per_day=cfg.max_orders_per_day)
            self.demo = False
        cfg.journal_dir.mkdir(parents=True, exist_ok=True)
        self.trails = TrailEngine(
            journal_path=cfg.journal_dir / "trails.jsonl",
            exit_fn=self._exit_fn,
        )
        self.portfolio = PortfolioState()
        self.ledger = SuggestionLedger(
            journal_path=cfg.journal_dir / "suggestions.jsonl",
            resolve_minutes=cfg.ledger_resolve_minutes,
        )
        self.maximizer = Maximizer(self.ledger)
        self.suggestions: List[Dict[str, Any]] = []
        self.recommendations: Dict[str, Any] = {"direction": "NEUTRAL",
                                                "items": []}
        self.direction_override: Optional[str] = None
        self._last_quotes: Dict[str, float] = {}
        self._chain_metas: List[InstrumentMeta] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- plumbing -----------------------------------------------------

    def log(self, msg: str) -> None:
        self.activity.insert(0, {
            "at": datetime.now(IST).strftime("%H:%M:%S"), "msg": msg})
        del self.activity[200:]
        LOG.info(msg)

    def _exit_fn(self, symbol: str, side: str, qty: int) -> Dict[str, Any]:
        if self.killed:
            raise RuntimeError("kill switch active; order refused")
        resp = self.account.place_market_exit(symbol, side, qty)
        self.log(f"EXIT {side} {qty} {symbol} -> {resp.get('status')} "
                 f"({resp.get('order_id')})")
        return resp

    # -- background loop -------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop,
                                        name="sentinel-loop", daemon=True)
        self._thread.start()
        self.log(f"sentinel started (demo={self.demo}, "
                 f"dry_run={getattr(self.account, 'dry_run', True)})")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        last_pf, last_ledger, last_reco = 0.0, 0.0, 0.0
        while not self._stop.is_set():
            try:
                now = time.monotonic()
                self._tick_quotes()
                if now - last_pf > self.cfg.poll_portfolio_seconds:
                    self._tick_portfolio()
                    last_pf = now
                if now - last_ledger > 30.0:
                    self.ledger.resolve_due(self._last_quotes)
                    last_ledger = now
                if now - last_reco > 60.0:
                    self._tick_recommendations()
                    last_reco = now
            except Exception as exc:
                LOG.exception("loop tick failed: %s", exc)
            self._stop.wait(self.cfg.poll_quote_seconds)

    def _held_symbols(self) -> List[str]:
        return [v.tradingsymbol for v in self.portfolio.positions.values()] \
            or [p.get("tradingsymbol") for p in self._safe_positions()]

    def _safe_positions(self) -> List[Dict[str, Any]]:
        try:
            return self.account.positions()
        except Exception as exc:
            self.log(f"positions fetch failed: {exc}")
            return []

    def _tick_quotes(self) -> None:
        syms = [s for s in self._held_symbols() if s]
        if not syms:
            return
        try:
            quotes = self.account.quotes(syms)
        except Exception as exc:
            self.log(f"quotes failed: {exc}")
            return
        for sym, q in quotes.items():
            self._last_quotes[sym] = q.ltp
            self.trails.on_tick(sym, q.ltp)

    def _tick_portfolio(self) -> None:
        try:
            raw = self._safe_positions()
            funds = self.account.funds()
            spot = self.account.spot_ltp()
            metas = self.account.instruments()
            syms = [str(p.get("tradingsymbol")) for p in raw
                    if p.get("tradingsymbol")]
            quotes = self.account.quotes(syms) if syms else {}
        except Exception as exc:
            self.log(f"portfolio refresh failed: {exc}")
            return
        self.portfolio.refresh(raw, quotes, metas, spot, funds)
        armed = [t.tradingsymbol for t in self.trails.active()]
        sugg = self.maximizer.run(self.portfolio, armed)
        self.suggestions = [vars(s) for s in sugg]
        fired = self.trails.on_portfolio_pnl(
            self.portfolio.total_pnl,
            [{"tradingsymbol": v.tradingsymbol, "quantity": v.quantity}
             for v in self.portfolio.positions.values()])
        if fired:
            self.log(f"PORTFOLIO LOCK fired at total P&L "
                     f"₹{self.portfolio.total_pnl:,.0f}")

    def _tick_recommendations(self) -> None:
        spot = self.portfolio.spot or self.account.spot_ltp()
        if not spot:
            return
        try:
            metas_all = self.account.instruments()
        except Exception:
            return
        # Nearest expiry chain for the configured underlying, ±6% strikes.
        chain = [m for m in metas_all.values()
                 if m.name == self.cfg.underlying
                 and m.instrument_type in ("CE", "PE")
                 and m.strike and abs(m.strike - spot) / spot <= 0.06]
        if not chain:
            return
        expiries = sorted({m.expiry for m in chain if m.expiry})
        if expiries:
            chain = [m for m in chain if m.expiry == expiries[0]]
        self._chain_metas = chain
        try:
            if hasattr(self.account, "chain_quotes"):
                quotes = self.account.chain_quotes(chain)        # demo path
            else:
                quotes = self.account.quotes(
                    [m.tradingsymbol for m in chain])
        except Exception as exc:
            self.log(f"chain quotes failed: {exc}")
            return
        direction = self.direction_override or self._auto_direction()
        recs = recommend_dips(direction, spot, chain, quotes)
        self.recommendations = {
            "direction": direction,
            "auto": self.direction_override is None,
            "items": [vars(r) for r in recs],
        }

    _spot_history: List[float] = []

    def _auto_direction(self) -> str:
        if self.portfolio.spot:
            self._spot_history.append(self.portfolio.spot)
            del self._spot_history[:-30]
        if len(self._spot_history) >= 10:
            ret = (self._spot_history[-1] / self._spot_history[0]) - 1.0
            if ret > 0.0015:
                return "UP"
            if ret < -0.0015:
                return "DOWN"
        return "NEUTRAL"


# ---------------------------------------------------------------------------
# FastAPI wiring
# ---------------------------------------------------------------------------

CFG = SentinelConfig.from_env()
CORE = Sentinel(CFG)
app = FastAPI(title="Sentinel", docs_url=None, redoc_url=None)


@app.on_event("startup")
def _startup() -> None:
    CORE.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    CORE.stop()


def auth(x_sentinel_token: Optional[str] = Header(default=None)) -> None:
    if CFG.dash_token and x_sentinel_token != CFG.dash_token:
        raise HTTPException(401, "bad or missing X-Sentinel-Token")


class ArmBody(BaseModel):
    tradingsymbol: str
    side: str = "long"
    quantity: int
    cushion_rupees: float


class CushionBody(BaseModel):
    cushion_rupees: float


class DirectionBody(BaseModel):
    direction: Optional[str] = None    # UP | DOWN | NEUTRAL | None=auto


class PortfolioTrailBody(BaseModel):
    cushion_rupees: float


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state", dependencies=[Depends(auth)])
def state() -> JSONResponse:
    return JSONResponse({
        "demo": CORE.demo,
        "dry_run": bool(getattr(CORE.account, "dry_run", True)),
        "killed": CORE.killed,
        "portfolio": CORE.portfolio.snapshot(),
        "scenario": CORE.portfolio.scenario_curve(),
        "trails": [t.serialise() for t in CORE.trails.active()],
        "portfolio_trail": {
            "cushion": CORE.trails.portfolio_cushion,
            "peak_pnl": (None if CORE.trails.portfolio_peak_pnl == float("-inf")
                         else CORE.trails.portfolio_peak_pnl),
            "fired": CORE.trails.portfolio_fired,
        },
        "suggestions": CORE.suggestions,
        "ledger": CORE.ledger.stats(),
        "recommendations": CORE.recommendations,
        "activity": CORE.activity[:50],
    })


@app.get("/api/whatif", dependencies=[Depends(auth)])
def whatif(move_points: float = 0.0) -> JSONResponse:
    return JSONResponse({"move_points": move_points,
                         "legs": CORE.portfolio.what_if(move_points)})


@app.post("/api/trails", dependencies=[Depends(auth)])
def arm_trail(body: ArmBody) -> JSONResponse:
    premium = CORE._last_quotes.get(body.tradingsymbol)
    if premium is None:
        v = CORE.portfolio.positions.get(body.tradingsymbol)
        premium = v.ltp if v else None
    if not premium:
        raise HTTPException(400, "no live premium for that symbol yet")
    t = CORE.trails.arm(body.tradingsymbol, body.side, body.quantity,
                        body.cushion_rupees, premium)
    CORE.log(f"trail armed on {body.tradingsymbol} cushion "
             f"₹{body.cushion_rupees:,.0f} @ {premium}")
    return JSONResponse(t.serialise())


@app.delete("/api/trails/{trail_id}", dependencies=[Depends(auth)])
def cancel_trail(trail_id: str) -> JSONResponse:
    ok = CORE.trails.cancel(trail_id)
    if ok:
        CORE.log(f"trail cancelled: {trail_id}")
    return JSONResponse({"cancelled": ok})


@app.post("/api/trails/{trail_id}/cushion", dependencies=[Depends(auth)])
def update_cushion(trail_id: str, body: CushionBody) -> JSONResponse:
    ok = CORE.trails.update_cushion(trail_id, body.cushion_rupees)
    return JSONResponse({"updated": ok})


@app.post("/api/portfolio_trail", dependencies=[Depends(auth)])
def arm_portfolio_trail(body: PortfolioTrailBody) -> JSONResponse:
    CORE.trails.arm_portfolio(body.cushion_rupees)
    CORE.log(f"portfolio lock armed: protect peak minus "
             f"₹{body.cushion_rupees:,.0f}")
    return JSONResponse({"armed": True})


@app.delete("/api/portfolio_trail", dependencies=[Depends(auth)])
def cancel_portfolio_trail() -> JSONResponse:
    CORE.trails.cancel_portfolio()
    CORE.log("portfolio lock cancelled")
    return JSONResponse({"cancelled": True})


@app.post("/api/direction", dependencies=[Depends(auth)])
def set_direction(body: DirectionBody) -> JSONResponse:
    if body.direction not in (None, "UP", "DOWN", "NEUTRAL"):
        raise HTTPException(400, "direction must be UP|DOWN|NEUTRAL|null")
    CORE.direction_override = body.direction
    CORE._tick_recommendations()
    return JSONResponse({"direction": body.direction or "auto"})


@app.post("/api/killswitch", dependencies=[Depends(auth)])
def killswitch() -> JSONResponse:
    CORE.killed = True
    n = len(CORE.trails.active())
    for t in CORE.trails.active():
        CORE.trails.cancel(t.trail_id)
    CORE.trails.cancel_portfolio()
    CORE.log(f"KILL SWITCH: {n} trail(s) cancelled, orders disabled "
             f"until restart")
    return JSONResponse({"killed": True, "trails_cancelled": n})
