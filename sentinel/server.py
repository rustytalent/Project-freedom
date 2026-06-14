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

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .advisor import Maximizer, SuggestionLedger, recommend_dips
from .auditor import StrategyLeg, audit_strategy
from .config import SentinelConfig
from .equity_layer import top10_contextual_layer
from .institutional import (
    expected_shortfall, historical_var, parametric_var, vol_cone,
)
from .kite_client import DemoAccount, KiteAccount, InstrumentMeta
from .audit_log import AuditLog
from .auth import AuthContext, verify_token
from .byok import GLOBAL_STORE, KiteCredentials
from .crux_signal import CruxVerdict, compose_from_state
from .notify import GLOBAL_NOTIFIER
from .paper import PaperAccount
from .rate_limit import allow as _rate_allow
from .replay import ReplayController
from .journey_audit import (
    Mistake, MistakeDetector, TradeQualityScorecard, publish_mistakes,
)
from .live_equity import ConstituentBoard, DemoFeed
from .live_models import DEFAULT_MODELS, LiveModelPool, Tick
from .liqpool_bridge import LiveSignalsTail
from .live_publisher import LivePublisher, ModelSignal
from .monte_carlo import LegPayoff, realised_sigma_per_day, simulate
from .orchestration import Orchestrator, Signal, Tier, TrustPromotionRecord
from .premium_tracker import PremiumTracker
from .psychology import PsychologyEngine, TradeAction
from .shadow_ledger import read_session
from .portfolio import PortfolioState
from .profit_lock import ProfitLock
from .shadow_ledger import ShadowLedger
from .saas import (
    FEATURE_CATALOG, FOUNDER, RETAIL, features_for, gate, plan_summary,
)
from .strategy_builder import ChainQuote, INTENTS, build as build_strategy
from .stress import SCENARIOS, stress_test, stress_test_all
from .institutional import LegExposure
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
        # The trust-tier spine: EVERY order placement is routed through the
        # orchestrator, and only EXECUTION-tier signals reach the execution
        # sink. The two hard-wired exit actors (trailing_stop, profit_lock)
        # are EXECUTION by identity; everything else is clamped to SHADOW
        # until the curator graduates it. This makes the wall runtime-real,
        # not just conceptual.
        self.routed_signals: List[Dict[str, Any]] = []
        self.surfaced_signals: List[Dict[str, Any]] = []
        self.orchestrator = Orchestrator(
            ledger_sink=self._sig_ledger,
            surface_sink=self._sig_surface,
            execution_sink=self._sig_execute,
            show_logged=False,
            allow_execution=True,
        )
        # Operator-facing rule engines are decision-support, not execution:
        # graduate them to TRUSTED so they surface, but they can never reach
        # the order path (only EXECUTION can, and only the two exit actors
        # are allowed to claim it).
        self.orchestrator.set_ceiling("maximizer", Tier.TRUSTED)
        self.orchestrator.set_ceiling("dip_recommender", Tier.TRUSTED)
        self.trails = TrailEngine(
            journal_path=cfg.journal_dir / "trails.jsonl",
            exit_fn=self._trail_exit,
        )
        self.portfolio = PortfolioState()
        # The canonical substrate: every live decision (suggestions today,
        # scientist hypotheses next) lands here. SuggestionLedger keeps its
        # scoring role but mirrors each suggestion into this ShadowLedger as
        # a model_suggestion event, so there is ONE system of record.
        self.session = datetime.now(IST).strftime("%Y-%m-%d")
        self.promotions: List[Dict[str, Any]] = []
        self._promotions_path = cfg.journal_dir / "trust_promotions.jsonl"
        self.shadow_ledger = ShadowLedger(cfg.journal_dir)
        self.ledger = SuggestionLedger(
            journal_path=cfg.journal_dir / "suggestions.jsonl",
            resolve_minutes=cfg.ledger_resolve_minutes,
            shadow_ledger=self.shadow_ledger,
            session=self.session,
        )
        # The Live Model Publisher — the founder's missing live brain.
        # Every research signal flows through here; the cockpit reads from
        # here; TRUSTED+ signals mirror to ShadowLedger.
        self.publisher = LivePublisher(shadow_ledger=self.shadow_ledger,
                                       session=self.session)
        self.live_models = LiveModelPool(DEFAULT_MODELS)
        self._tick_hist: List[Tick] = []
        # NIFTY constituent board (Wave 11). In demo mode we feed a
        # deterministic synthetic top-10 walker so the panel comes alive
        # without Kite quotes; live mode plugs into account.quotes().
        self.constituent_board = ConstituentBoard(publisher=self.publisher)
        self._equity_feed: Optional[DemoFeed] = DemoFeed() if self.demo else None
        self._board_snapshot: Dict[str, Any] = {}
        # Behavioral engine (Wave 12). Drives 6 citation-bearing bias
        # detectors + a TiltIndex composite + the operator's Ulysses-
        # contract; spine refuses non-hard-wired EXECUTION upgrades while
        # tilt is in RED or worse.
        self.psychology = PsychologyEngine(
            publisher=self.publisher, session=self.session,
            journal_path=cfg.journal_dir / "mind_reports.jsonl",
        )
        # Wave 13 — per-symbol premium history + post-hoc audit.
        self.premium_tracker = PremiumTracker()
        self.scorecard = TradeQualityScorecard()
        self.mistake_detector = MistakeDetector()
        # The premium symbol the cockpit charts. None = auto-pick the
        # largest held position; the operator can override via the API.
        self.premium_symbol: Optional[str] = None
        self._mistake_event_ids: set = set()    # dedupe per-row publications
        # Wave 14 — Crux meta-signal composer (latest verdict on the bus)
        self.crux_verdict: Optional[CruxVerdict] = None
        # Wave 15 — research-engine live signals listener. Tails the
        # JSONL liqpool's live_inference writes; each new row becomes a
        # ModelSignal on Sentinel's bus.
        liqpool_signals_path = cfg.journal_dir / "liqpool_live_signals.jsonl"
        self.liqpool_tail = LiveSignalsTail(
            liqpool_signals_path, publisher=self.publisher)
        # Launch-ready preflight: track whether the operator has
        # acknowledged the pre-market brief for this session. The cockpit
        # gates Crux verdicts on this so a fresh-login operator can't
        # blast into the day without committing.
        self.preflight_ack: Optional[Dict[str, Any]] = None
        self.boot_ts = time.monotonic()
        # Wave 18 — institutional audit trail + BYOK credential store.
        self.audit = AuditLog(path=cfg.journal_dir / "audit.jsonl")
        self.byok = GLOBAL_STORE
        # Notifier hook (Wave 20) — every audit event is filtered through
        # _RENDERERS; Codex's email/push transports register on
        # GLOBAL_NOTIFIER and receive Alerts for actionable events only.
        GLOBAL_NOTIFIER.attach_to_audit()
        # Wave 19 — paper mode + replay
        paper_env = os.environ.get("SENTINEL_PAPER", "") == "1"
        if paper_env and not self.demo:
            # Wrap the underlying account in PaperAccount so every order
            # is simulated. funds/positions/quotes/chain stay real.
            self.account = PaperAccount(
                underlying=self.account,
                journal_path=cfg.journal_dir / "paper_orders.jsonl",
                starting_balance=float(os.environ.get(
                    "SENTINEL_PAPER_BALANCE", "100000")),
            )
            self.paper_enabled = True
        else:
            self.paper_enabled = False
        self.replay = ReplayController(
            publisher=self.publisher,
            journal_dir=cfg.journal_dir,
            is_live_real=lambda: (not self.demo
                                   and not self.paper_enabled
                                   and not getattr(self.account, "dry_run", True)),
        )
        self.audit.write("server_started",
                          {"mode": "demo" if self.demo else (
                              "dry_run" if getattr(self.account, "dry_run", True)
                              else "live"),
                            "session": self.session,
                            "demo_mode": self.demo},
                          session=self.session)
        self.maximizer = Maximizer(self.ledger)
        # Ratcheting day-profit lock (the founder's locked/floating model).
        self.profit_lock: Optional[ProfitLock] = None
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

    # -- orchestrator sinks ----------------------------------------------
    # The spine calls these; route order is ledger -> surface -> execution.

    def _sig_ledger(self, sig: Signal) -> None:
        """Every routed signal is recorded — the substrate sees all."""
        self.routed_signals.insert(0, {
            "at": datetime.now(IST).strftime("%H:%M:%S"),
            "source": sig.source, "tier": sig.tier.name,
            "kind": sig.kind, "reason": sig.reason,
        })
        del self.routed_signals[100:]

    def _sig_surface(self, sig: Signal) -> None:
        """Operator-facing signals (TRUSTED+, or LOGGED when show_logged)."""
        self.surfaced_signals.insert(0, {
            "at": datetime.now(IST).strftime("%H:%M:%S"),
            "source": sig.source, "kind": sig.kind, "reason": sig.reason,
        })
        del self.surfaced_signals[50:]

    def _sig_execute(self, sig: Signal) -> Optional[Dict[str, Any]]:
        """The ONE gated path to an order. Reached only by EXECUTION-tier
        signals the spine permitted. Hard-wired exit actors (trailing_stop,
        profit_lock) always pass; everything else is also gated by the
        behavioral engine — RED tilt or worse refuses non-exit orders so
        the operator cannot route around their own committed limits."""
        hard = sig.source in ("trailing_stop", "profit_lock")
        if not hard and self.psychology.should_block_execution():
            band = self.psychology.state().band
            self.log(f"[psychology] BLOCKED {sig.source} order — "
                     f"tilt band {band}")
            self.audit.write("order_blocked",
                              {"source": sig.source, "reason": "red_tilt",
                                "tilt_band": band,
                                "symbol": sig.payload.get("symbol"),
                                "qty": sig.payload.get("qty")},
                              severity=AuditLog.SEV_WARN,
                              session=self.session)
            return None
        p = sig.payload
        result = self._exit_fn(p["symbol"], p["side"], int(p["qty"]))
        self.audit.write("order_placed",
                          {"source": sig.source, "symbol": p["symbol"],
                            "side": p["side"], "qty": p["qty"],
                            "order_id": (result or {}).get("order_id", ""),
                            "status": (result or {}).get("status", "")},
                          severity=AuditLog.SEV_INFO,
                          session=self.session)
        return result

    def record_promotion(self, rec: TrustPromotionRecord) -> None:
        """Persist a trust-promotion decision (append-only) and keep a
        recent window for the dashboard."""
        row = rec.to_row()
        self.promotions.insert(0, row)
        del self.promotions[50:]
        try:
            with open(self._promotions_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass

    def _trail_exit(self, symbol: str, side: str, qty: int) -> Dict[str, Any]:
        """TrailEngine's injected exit_fn — instead of placing an order
        directly, it asks the spine to route an EXECUTION-tier exit. The
        trailing stop and portfolio trail are both 'trailing_stop' (a
        hard-wired EXECUTION actor). Returns the order response (or {} if
        the spine blocked it, e.g. kill switch) so TrailEngine's order_id
        read never crashes."""
        resp = self.orchestrator.route(Signal(
            source="trailing_stop", tier=Tier.EXECUTION, kind="exit",
            payload={"symbol": symbol, "side": side, "qty": qty},
            reason="trailing stop / portfolio lock hit"))
        # Record the EXIT so the disposition / revenge detectors have
        # material to read.
        pos = self.portfolio.positions.get(symbol)
        self.psychology.record(TradeAction(
            ts_ist=datetime.now(IST).strftime("%H:%M:%S"),
            action="EXIT", symbol=symbol, qty=int(qty),
            pnl=float(pos.pnl) if pos else 0.0,
            premium=float(pos.ltp) if pos else 0.0,
            spot=float(self.portfolio.spot or 0.0),
        ))
        return resp or {}

    def _tick_psychology(self) -> None:
        """Run the behavioral engine once per cycle: record spot, fire
        detectors, update tilt."""
        if self.portfolio.spot:
            self.psychology.record_spot(
                datetime.now(IST).strftime("%H:%M:%S"),
                float(self.portfolio.spot))
        events = self.psychology.cycle(
            day_pnl=float(self.portfolio.total_pnl),
            open_positions=len(self.portfolio.positions))
        for ev in events:
            self.log(f"[psychology] {ev.bias} (sev {ev.severity:.0%}) — "
                     f"{ev.evidence}")

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
        last_pf, last_ledger, last_reco, last_audit = 0.0, 0.0, 0.0, 0.0
        while not self._stop.is_set():
            try:
                now = time.monotonic()
                self._tick_quotes()
                self._tick_board()
                self._tick_models()
                self.liqpool_tail.poll()
                self._tick_psychology()
                self._tick_crux()
                if now - last_pf > self.cfg.poll_portfolio_seconds:
                    self._tick_portfolio()
                    last_pf = now
                if now - last_ledger > 30.0:
                    self.ledger.resolve_due(self._last_quotes)
                    last_ledger = now
                if now - last_reco > 60.0:
                    self._tick_recommendations()
                    last_reco = now
                if now - last_audit > 45.0:
                    self._tick_journey_audit()
                    last_audit = now
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
            self.premium_tracker.update(sym, q.ltp)

    def _tick_crux(self) -> None:
        """Run the Crux meta-signal composer once per cycle. Combines
        every live signal + tilt + intention + position state into one
        operator-facing verdict, published as a TRUSTED 'crux' signal."""
        positions = list(self.portfolio.positions.values())
        has_winner = any(getattr(p, "pnl", 0) > 1000 for p in positions)
        has_loser  = any(getattr(p, "pnl", 0) < -1000 for p in positions)
        # Preflight gate: if the operator hasn't acknowledged the day's
        # plan, Crux is allowed to surface HOLD/WATCH/WAIT/EXIT signals
        # (defensive) but TRADE is downgraded to WATCH. Implemented by
        # tagging the composer's open_positions=0 case differently —
        # simplest: post-process the verdict.
        self.crux_verdict = compose_from_state(
            live_signals=self.publisher.current(),
            tilt_band=self.psychology.state().band,
            tilt_index=self.psychology.state().index,
            intention_violated=bool(self.psychology.intention.violated),
            open_positions=len(positions),
            open_pnl=float(self.portfolio.total_pnl),
            has_winner=has_winner, has_loser=has_loser,
            killed=bool(self.killed),
            publisher=self.publisher,
        )
        if (self.preflight_ack is None
                and self.crux_verdict is not None
                and self.crux_verdict.verdict == "TRADE"):
            # Downgrade TRADE to WATCH; record the reason so the cockpit
            # shows the operator why.
            self.crux_verdict = type(self.crux_verdict)(
                verdict="WATCH",
                confidence=min(0.6, self.crux_verdict.confidence),
                rationale=("TRADE verdict suppressed — pre-market "
                           "acknowledgement required before live entries"),
                supporting=list(self.crux_verdict.supporting),
                contradicting="preflight_not_acknowledged",
                inputs=dict(self.crux_verdict.inputs),
                ts_ist=self.crux_verdict.ts_ist,
            )

    def _tick_journey_audit(self) -> None:
        """Score every completed journey + scan for mistakes. Cheap; runs
        on a longer interval than the quote loop. New mistakes publish
        as TRUSTED trade_mistake signals (deduped by event_id)."""
        try:
            rows = read_session(self.cfg.journal_dir, self.session)
        except Exception:
            return
        new_mistakes: List[Mistake] = []
        for row in rows:
            if not (row.get("journey") or {}).get("complete"):
                continue
            eid = str(row.get("event_id", ""))
            if eid in self._mistake_event_ids:
                continue
            self._mistake_event_ids.add(eid)
            new_mistakes.extend(self.mistake_detector.scan_row(row))
        if new_mistakes:
            publish_mistakes(new_mistakes, self.publisher)
            for m in new_mistakes:
                self.log(f"[mistake] {m.pattern} on {m.symbol} — {m.citation}")

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
        for s in self.suggestions:
            # maximizer is graduated to TRUSTED -> these surface but can
            # never reach the order path (only EXECUTION can).
            self.orchestrator.route(Signal(
                source="maximizer", tier=Tier.TRUSTED, kind="recommendation",
                payload=s,
                reason=str(s.get("rationale") or s.get("action")
                           or "maximizer suggestion")))
        fired = self.trails.on_portfolio_pnl(
            self.portfolio.total_pnl,
            [{"tradingsymbol": v.tradingsymbol, "quantity": v.quantity}
             for v in self.portfolio.positions.values()])
        if fired:
            self.log(f"PORTFOLIO LOCK fired at total P&L "
                     f"₹{self.portfolio.total_pnl:,.0f}")
        # Ratcheting profit lock: feed cumulative day P&L; on fire,
        # flatten every open position (the locked profit becomes realized).
        if self.profit_lock is not None and self.profit_lock.update(
                self.portfolio.total_pnl):
            snap = self.profit_lock.snapshot()
            self.log(f"PROFIT LOCK fired: locking ₹{snap['locked_pnl']:,.0f} "
                     f"(peak ₹{snap['peak_pnl']:,.0f})")
            reason = (f"profit lock fired, locking ₹{snap['locked_pnl']:,.0f} "
                      f"(peak ₹{snap['peak_pnl']:,.0f})")
            for v in list(self.portfolio.positions.values()):
                if self.killed:
                    break
                side = "SELL" if v.quantity > 0 else "BUY"
                try:
                    self.orchestrator.route(Signal(
                        source="profit_lock", tier=Tier.EXECUTION, kind="exit",
                        payload={"symbol": v.tradingsymbol, "side": side,
                                 "qty": abs(v.quantity)},
                        reason=reason))
                except Exception:
                    continue

    def _tick_board(self) -> None:
        """Pump the NIFTY constituent board. Demo mode uses DemoFeed;
        live mode reads top-10 quotes from the broker."""
        if self._equity_feed is not None:
            quotes = self._equity_feed.next()
        else:
            try:
                live = self.account.quotes(
                    [f"NSE:{s}" for s in self.constituent_board.weights])
                quotes = {k.split(":")[-1]: q.ltp for k, q in live.items()}
            except Exception:
                return
        try:
            self._board_snapshot = self.constituent_board.tick(quotes)
        except Exception as exc:
            self.log(f"constituent board tick failed: {exc}")

    def _tick_models(self) -> None:
        """Drive every live research model on the latest spot tick. Each
        non-None output goes to the publisher; deduped publishes mirror
        to the shadow ledger and become rows on the brief feed."""
        spot = self.portfolio.spot or self.account.spot_ltp()
        if not spot:
            return
        self._tick_hist.append(Tick(ts=time.monotonic(), spot=float(spot)))
        # bound history to ~10 minutes at 1-2s polling
        del self._tick_hist[:-600]
        if len(self._tick_hist) < 5:
            return
        for sig in self.live_models.run(self._tick_hist, asset="NIFTY"):
            if self.publisher.publish(sig):
                self.log(f"[{sig.model}] {sig.signal} "
                         f"(conf {sig.confidence:.0%})")

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
        if recs:
            self.orchestrator.route(Signal(
                source="dip_recommender", tier=Tier.TRUSTED, kind="recommendation",
                payload={"direction": direction, "n": len(recs)},
                reason=f"{len(recs)} dip recommendation(s), bias {direction}"))

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


_DEV_HEADER_FALLBACK_ALLOWED = (
    os.environ.get("SENTINEL_DEMO", "") == "1"
    or os.environ.get("SENTINEL_ALLOW_HEADER_AUTH", "") == "1"
)


def resolve_auth(
    authorization: Optional[str] = Header(default=None),
    x_sentinel_plan: Optional[str] = Header(default=None),
) -> Optional[AuthContext]:
    """Auth resolver — production reads ``Authorization: Bearer <jwt>``
    (Codex's Supabase token).

    The legacy ``X-Sentinel-Plan: FOUNDER`` header is ONLY honoured when
    ``SENTINEL_DEMO=1`` or ``SENTINEL_ALLOW_HEADER_AUTH=1`` is set —
    otherwise prod can't accidentally ship the dev backdoor.

    Returns None when no valid credential present; endpoints that need
    auth check for None and return 401."""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        return verify_token(token)
    if x_sentinel_plan and _DEV_HEADER_FALLBACK_ALLOWED:
        return AuthContext(user_id="dev",
                           email="dev@local",
                           plan=x_sentinel_plan.upper(),
                           is_demo=True)
    return None


def require_user(ctx: Optional[AuthContext] = Depends(resolve_auth)
                 ) -> AuthContext:
    """FastAPI dependency for endpoints that need an authenticated user
    (not just a plan). 401 on no valid credential."""
    if ctx is None:
        raise HTTPException(401, "authentication required")
    return ctx


def _client_key(req) -> str:
    """Best-effort client key for IP-based limits. Codex's gateway
    should set X-Forwarded-For or X-Real-IP; we fall back to the
    socket peer."""
    return (req.headers.get("X-Real-IP")
            or (req.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
            or (req.client.host if req.client else "")
            or "_unknown")


def enforce_rate(preset: str, request, ctx: Optional[AuthContext] = None) -> None:
    """Inline rate-limit check called at the top of an endpoint.
    Raises 429 with retry hint on throttle. Caller picks the key:
    user_id when authenticated, IP when anonymous."""
    key = ctx.user_id if (ctx and ctx.user_id) else _client_key(request)
    ok, remaining = _rate_allow(preset, key)
    if not ok:
        raise HTTPException(429, {
            "error": "rate_limited",
            "preset": preset,
            "retry_after_seconds": max(1, int(60 / max(1, remaining))),
        })


def resolve_plan(x_sentinel_plan: Optional[str] = Header(default=None)) -> str:
    """Plan resolver — header-driven for SaaS gating. If unset, the
    dashboard auth implies FOUNDER (the user's own view). A signed
    plan-token JWT can replace this in v2; the contract is just the
    string returned here."""
    if x_sentinel_plan:
        return x_sentinel_plan.upper()
    return FOUNDER


def require(feature: str, plan: str) -> None:
    """Raise 402 unless ``plan`` covers ``feature``."""
    g = gate(plan, feature)
    if not g.allowed:
        raise HTTPException(402, {
            "error": "payment_required",
            "feature": feature,
            "required_tier": g.required_tier,
            "your_tier": g.your_tier,
            "reason": g.reason,
        })


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


# ─────────────────────────────────────────────────────────────────
# Health / observability — for whatever load balancer Codex puts in
# front. Both unauthenticated by design.
# ─────────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz() -> JSONResponse:
    """Liveness probe. The process is up; this returns 200 even if the
    market loop is stuck. For traffic-routing decisions, use /readyz."""
    return JSONResponse({"status": "ok"})


@app.get("/readyz")
def readyz() -> JSONResponse:
    """Readiness probe. The hot loop is alive (background thread up),
    the broker connection responded recently, and the shadow ledger
    is writable. Returns 503 if any of those fails so the LB can shed
    traffic."""
    issues = []
    if CORE._thread is None or not CORE._thread.is_alive():
        issues.append("loop_thread_dead")
    try:
        # round-trip a noop quote so we know the broker layer responds
        _ = CORE.account.funds()
    except Exception as exc:
        issues.append(f"broker:{type(exc).__name__}")
    try:
        # confirm ledger dir is writable
        (CFG.journal_dir / ".readyz_probe").touch()
    except Exception:
        issues.append("ledger_dir_unwritable")
    uptime = round(time.monotonic() - CORE.boot_ts, 1)
    if issues:
        return JSONResponse(
            {"status": "degraded", "issues": issues, "uptime_seconds": uptime},
            status_code=503)
    return JSONResponse({
        "status": "ready",
        "uptime_seconds": uptime,
        "mode": "demo" if CORE.demo else ("dry_run" if getattr(CORE.account, "dry_run", True) else "live"),
        "session": CORE.session,
        "killed": CORE.killed,
        "ledger_events_today": len(CORE.routed_signals),
    })


# ─────────────────────────────────────────────────────────────────
# Preflight — the Ulysses-pattern commitment before live trading
# (Mother's Audit §3.5)
# ─────────────────────────────────────────────────────────────────

class PreflightBody(BaseModel):
    read_brief: bool = True
    accepted_max_loss_rupees: float
    expected_regime: str = ""             # "trending" | "chop" | "no_view"
    no_revenge_trading: bool = True
    notes: str = ""


# ─────────────────────────────────────────────────────────────────
# Replay — load a past session, watch it play through the cockpit
# ─────────────────────────────────────────────────────────────────

class ReplayStartBody(BaseModel):
    session_date: str
    speed: float = 10.0


@app.post("/api/replay/start", dependencies=[Depends(auth)])
def replay_start(body: ReplayStartBody) -> JSONResponse:
    try:
        prog = CORE.replay.start(body.session_date, body.speed)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    CORE.audit.write("replay_started", body.dict(), session=CORE.session)
    return JSONResponse(prog.to_row())


@app.post("/api/replay/pause", dependencies=[Depends(auth)])
def replay_pause() -> JSONResponse:
    return JSONResponse(CORE.replay.pause().to_row())


@app.post("/api/replay/resume", dependencies=[Depends(auth)])
def replay_resume() -> JSONResponse:
    return JSONResponse(CORE.replay.resume().to_row())


@app.post("/api/replay/stop", dependencies=[Depends(auth)])
def replay_stop() -> JSONResponse:
    prog = CORE.replay.stop()
    CORE.audit.write("replay_stopped", prog.to_row(), session=CORE.session)
    return JSONResponse(prog.to_row())


@app.get("/api/replay/state", dependencies=[Depends(auth)])
def replay_state() -> JSONResponse:
    return JSONResponse(CORE.replay.state().to_row())


# ─────────────────────────────────────────────────────────────────
# Paper trading — read-only quotes, simulated fills
# ─────────────────────────────────────────────────────────────────

@app.get("/api/paper/status", dependencies=[Depends(auth)])
def paper_status() -> JSONResponse:
    """Surface paper-mode state for the cockpit: enabled, realized
    P&L, open book of simulated positions."""
    if not CORE.paper_enabled:
        return JSONResponse({"enabled": False})
    pa = CORE.account                          # PaperAccount
    return JSONResponse({
        "enabled": True,
        "starting_balance": pa._starting_balance,
        "realized_pnl": pa.realized_pnl(),
        "open_book": pa.open_book(),
        "n_orders": len(pa.orders_log),
    })


# ─────────────────────────────────────────────────────────────────
# BYOK — bring-your-own-Kite-key (SaaS multi-tenant unlock)
# ─────────────────────────────────────────────────────────────────

class ByokConnectBody(BaseModel):
    api_key: str
    access_token: str


@app.post("/api/byok/connect", dependencies=[Depends(auth)])
def byok_connect(body: ByokConnectBody, request: Request,
                 ctx: AuthContext = Depends(require_user)) -> JSONResponse:
    """Customer connects their own Kite developer credentials. Stored
    encrypted in-memory keyed by ``user_id``. Cleared on process restart
    — Codex's persistence layer re-populates from Supabase as users
    come online (he holds the AT-REST master key)."""
    enforce_rate("account", request, ctx)
    try:
        CORE.byok.set(ctx.user_id, body.api_key, body.access_token)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    CORE.audit.write("byok_connect",
                      {"connected_users": CORE.byok.connected_users()},
                      severity=AuditLog.SEV_INFO,
                      user_id=ctx.user_id, session=CORE.session)
    return JSONResponse({
        "ok": True,
        "user_id": ctx.user_id,
        "connected": True,
        "connected_users_total": CORE.byok.connected_users(),
    })


@app.delete("/api/byok/connect", dependencies=[Depends(auth)])
def byok_disconnect(ctx: AuthContext = Depends(require_user)) -> JSONResponse:
    """Customer disconnects their Kite session (logout / token revoke)."""
    removed = CORE.byok.forget(ctx.user_id)
    CORE.audit.write("byok_disconnect", {"removed": removed},
                      user_id=ctx.user_id, session=CORE.session)
    return JSONResponse({"ok": True, "removed": removed})


@app.get("/api/byok/status", dependencies=[Depends(auth)])
def byok_status(ctx: AuthContext = Depends(require_user)) -> JSONResponse:
    """Whether the caller has Kite credentials connected this process."""
    return JSONResponse({
        "user_id": ctx.user_id,
        "connected": CORE.byok.has(ctx.user_id),
    })


@app.get("/api/preflight", dependencies=[Depends(auth)])
def preflight_status() -> JSONResponse:
    return JSONResponse({
        "acknowledged": CORE.preflight_ack is not None,
        "ack": CORE.preflight_ack,
        "session": CORE.session,
    })


@app.post("/api/preflight/ack", dependencies=[Depends(auth)])
def preflight_ack(body: PreflightBody) -> JSONResponse:
    """The operator commits to the day's plan before live signals stream.
    Sentinel doesn't refuse cockpit access without this, but the Crux
    composer reads ``preflight_ack`` and will not surface a TRADE
    verdict until the operator has explicitly committed. This is the
    behavioral wall ChatGPT and the founder talked about — the
    pre-market read happens, the commitment is made, THEN the day
    starts."""
    if body.accepted_max_loss_rupees <= 0:
        raise HTTPException(400, "accepted_max_loss_rupees must be > 0")
    CORE.preflight_ack = {
        "at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "accepted_max_loss_rupees": body.accepted_max_loss_rupees,
        "expected_regime": body.expected_regime,
        "no_revenge_trading": body.no_revenge_trading,
        "read_brief": body.read_brief,
        "notes": body.notes,
    }
    # Also commit a matching intention contract so the psychology engine
    # can enforce the limits (Ulysses pattern).
    try:
        CORE.psychology.set_intention(
            max_day_loss_rupees=float(body.accepted_max_loss_rupees),
            notes=body.notes or "set via preflight",
        )
    except Exception:
        pass
    CORE.log(f"preflight ACK: max loss ₹{body.accepted_max_loss_rupees:,.0f}"
             + (f", regime {body.expected_regime}" if body.expected_regime else ""))
    return JSONResponse({"ok": True, "ack": CORE.preflight_ack})


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/console")
def customer_console() -> FileResponse:
    """Customer-facing SaaS console — auditor, builder, stress, risk.
    Plan-gated (RETAIL/PRO/QUANT) via the X-Sentinel-Plan header."""
    return FileResponse(STATIC_DIR / "console.html")


@app.get("/sentinel.css")
def stylesheet() -> FileResponse:
    return FileResponse(STATIC_DIR / "sentinel.css", media_type="text/css")


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
        "profit_lock": (CORE.profit_lock.snapshot()
                        if CORE.profit_lock is not None else None),
        "suggestions": CORE.suggestions,
        "ledger": CORE.ledger.stats(),
        "recommendations": CORE.recommendations,
        "orchestrator": CORE.orchestrator.stats(),
        "routed_signals": CORE.routed_signals[:20],
        "promotions": CORE.promotions[:20],
        "live_signals": {k: v.to_row() for k, v in CORE.publisher.current().items()},
        "live_feed": [s.to_row() for s in CORE.publisher.history(limit=18)],
        "spot_history": [{"t": round(t.ts, 1), "spot": t.spot}
                          for t in CORE._tick_hist[-180:]],
        "constituent_board": CORE._board_snapshot,
        "psychology": CORE.psychology.state().to_row(),
        "intention": CORE.psychology.intention.to_row(),
        "premium_chart": _premium_chart_payload(),
        "preflight": {
            "acknowledged": CORE.preflight_ack is not None,
            "ack": CORE.preflight_ack,
        },
        "paper": {
            "enabled": CORE.paper_enabled,
            "realized_pnl": (CORE.account.realized_pnl()
                              if CORE.paper_enabled else 0.0),
        },
        "replay": CORE.replay.state().to_row(),
        "crux": CORE.crux_verdict.to_row() if CORE.crux_verdict else None,
        "model_zones": _model_zones_payload(),
        "activity": CORE.activity[:50],
    })


class IntentionBody(BaseModel):
    max_day_loss_rupees: float
    target_day_profit_rupees: float = 0.0
    max_positions: int = 0
    no_trade_after_ist: str = "15:15"
    notes: str = ""


@app.post("/api/intention", dependencies=[Depends(auth)])
def set_intention(body: IntentionBody) -> JSONResponse:
    """The operator's pre-market commitment — Ulysses contract pattern
    (Elster 2000). The behavioral engine checks it every cycle; if it's
    breached, INTENTION_VIOLATED fires and the spine refuses non-hard-
    wired EXECUTION upgrades."""
    if body.max_day_loss_rupees <= 0:
        raise HTTPException(400, "max_day_loss_rupees must be > 0")
    intent = CORE.psychology.set_intention(
        max_day_loss_rupees=body.max_day_loss_rupees,
        target_day_profit_rupees=body.target_day_profit_rupees,
        max_positions=body.max_positions,
        no_trade_after_ist=body.no_trade_after_ist,
        notes=body.notes,
    )
    CORE.log(f"intention set: max loss ₹{body.max_day_loss_rupees:,.0f}, "
             f"flat by {body.no_trade_after_ist}")
    return JSONResponse(intent.to_row())


@app.get("/api/psychology", dependencies=[Depends(auth)])
def psychology_state() -> JSONResponse:
    """The TiltIndex + active biases + recent bias events."""
    return JSONResponse({
        "tilt": CORE.psychology.state().to_row(),
        "intention": CORE.psychology.intention.to_row(),
        "execution_blocked": CORE.psychology.should_block_execution(),
    })


@app.get("/api/psychology/mind_report", dependencies=[Depends(auth)])
def mind_report() -> JSONResponse:
    """The end-of-session reflection: tilt avg/peak, biases-by-type,
    intention status, citations. Also appended to
    <journal>/mind_reports.jsonl for the next-day review."""
    return JSONResponse(CORE.psychology.mind_report().to_row())


def _model_zones_payload() -> List[Dict[str, Any]]:
    """Collect every live model signal that carries a zone and emit it
    for the cockpit's spot-chart overlay layer. The cockpit's switcher
    lets the operator toggle which models render."""
    out: List[Dict[str, Any]] = []
    for sig in CORE.publisher.current().values():
        if sig.zone is None:
            continue
        out.append({
            "model": sig.model, "asset": sig.asset,
            "ts_ist": sig.ts_ist, "signal": sig.signal,
            "confidence": sig.confidence,
            "zone_low": sig.zone[0], "zone_high": sig.zone[1],
            "reason_codes": list(sig.reason_codes),
            "source": sig.extras.get("source") if sig.extras else "sentinel",
        })
    return out


class MonteCarloLegBody(BaseModel):
    symbol: str
    qty: int
    entry_premium: float
    delta: float = 0.0
    gamma: float = 0.0
    theta_per_day: float = 0.0


class MonteCarloBody(BaseModel):
    legs: Optional[List[MonteCarloLegBody]] = None     # None = auto from book
    spot: Optional[float] = None                        # None = current
    horizons_min: List[int] = [5, 15, 30]
    n_paths: int = 1000
    target_pnl_rupees: float = 1500.0
    stop_pnl_rupees: float = 1500.0
    seed: int = 7


@app.post("/api/monte_carlo", dependencies=[Depends(auth)])
def monte_carlo(body: MonteCarloBody, request: Request,
                plan: str = Depends(resolve_plan)) -> JSONResponse:
    """1000-path GBM probe over the held book (or supplied legs).
    Returns p(profit/target/stop) + R-multiple distribution per horizon.
    Tier: PRO (institutional analytics)."""
    enforce_rate("heavy", request)
    require("var.cornish_fisher", plan)        # PRO-grade scenario probe
    if body.legs:
        legs = [LegPayoff(symbol=l.symbol, qty=l.qty,
                          entry_premium=l.entry_premium,
                          delta=l.delta, gamma=l.gamma,
                          theta_per_day=l.theta_per_day)
                for l in body.legs]
    else:
        legs = []
        for p in CORE.portfolio.positions.values():
            legs.append(LegPayoff(
                symbol=p.tradingsymbol, qty=p.quantity,
                entry_premium=float(p.ltp),
                delta=float(p.delta or 0.0),
                gamma=0.0,
                theta_per_day=float(p.theta_per_day or 0.0),
            ))
    spot = body.spot if body.spot else CORE.portfolio.spot or 25000.0
    # realised σ from the recent spot history (annualised)
    prices = [t.spot for t in CORE._tick_hist[-120:]]
    sigma = realised_sigma_per_day(prices) if len(prices) > 10 else 0.15
    result = simulate(
        legs, spot=float(spot), sigma_annualised=sigma,
        horizons_min=tuple(body.horizons_min), n_paths=body.n_paths,
        target_pnl_rupees=body.target_pnl_rupees,
        stop_pnl_rupees=body.stop_pnl_rupees,
        seed=body.seed,
    )
    return JSONResponse(result.to_row())


@app.get("/api/crux", dependencies=[Depends(auth)])
def crux_verdict() -> JSONResponse:
    """The fused Crux verdict — one operator-facing call."""
    return JSONResponse(CORE.crux_verdict.to_row() if CORE.crux_verdict else {})


def _premium_chart_payload() -> Dict[str, Any]:
    """The default premium chart payload: tracker snapshot for the
    selected (or auto-picked) symbol + list of all known symbols."""
    sym = CORE.premium_symbol or CORE.premium_tracker.select_default(
        CORE.portfolio.positions)
    snap = CORE.premium_tracker.snapshot(sym) if sym else None
    return {
        "symbol": sym,
        "available": CORE.premium_tracker.known_symbols(),
        "series": snap,
    }


class PremiumSymbolBody(BaseModel):
    symbol: Optional[str] = None        # None = auto-pick


@app.post("/api/premium_chart", dependencies=[Depends(auth)])
def set_premium_symbol(body: PremiumSymbolBody) -> JSONResponse:
    """Pin the cockpit's option-premium chart to a specific symbol; pass
    null to revert to auto-picking the largest held position."""
    CORE.premium_symbol = body.symbol
    return JSONResponse(_premium_chart_payload())


@app.get("/api/premium_chart", dependencies=[Depends(auth)])
def get_premium_chart(symbol: Optional[str] = None) -> JSONResponse:
    """Read the premium tracker for ?symbol= (or the auto/pinned default)."""
    if symbol:
        snap = CORE.premium_tracker.snapshot(symbol)
        return JSONResponse({"symbol": symbol,
                              "available": CORE.premium_tracker.known_symbols(),
                              "series": snap})
    return JSONResponse(_premium_chart_payload())


@app.get("/api/journeys", dependencies=[Depends(auth)])
def journeys(limit: int = 30) -> JSONResponse:
    """Recent ledger journeys, newest first. Each row is the full
    five-block event (identity + context + hypothesis + journey +
    judgment) plus the trade-quality score when complete."""
    try:
        rows = read_session(CORE.cfg.journal_dir, CORE.session)
    except Exception:
        rows = []
    # newest first
    rows = list(reversed(rows))[: max(1, limit)]
    scored = []
    for r in rows:
        item = {"row": r}
        if (r.get("journey") or {}).get("complete"):
            item["score"] = CORE.scorecard.score(r).to_row()
        scored.append(item)
    return JSONResponse({"session": CORE.session, "items": scored})


@app.get("/api/scorecard", dependencies=[Depends(auth)])
def scorecard_summary() -> JSONResponse:
    """Session-level scorecard summary: average, grade distribution,
    best / worst journey."""
    try:
        rows = read_session(CORE.cfg.journal_dir, CORE.session)
    except Exception:
        rows = []
    scores = CORE.scorecard.score_session(rows)
    return JSONResponse({
        "session": CORE.session,
        "scores": [s.to_row() for s in scores],
        "summary": CORE.scorecard.session_summary(scores),
    })


@app.get("/api/mistakes", dependencies=[Depends(auth)])
def mistakes_endpoint() -> JSONResponse:
    """Every detected mistake this session (newest first)."""
    try:
        rows = read_session(CORE.cfg.journal_dir, CORE.session)
    except Exception:
        rows = []
    mistakes = CORE.mistake_detector.scan_session(rows)
    return JSONResponse({"session": CORE.session,
                          "mistakes": [m.to_row() for m in reversed(mistakes)]})


@app.get("/api/equity_board", dependencies=[Depends(auth)])
def equity_board() -> JSONResponse:
    """The live NIFTY constituent board snapshot — regime, move-quality
    verdict, per-stock contributions, leaders, laggards. Same payload
    /api/state carries under `constituent_board`."""
    return JSONResponse(CORE._board_snapshot or {})


@app.get("/api/live/signals", dependencies=[Depends(auth)])
def live_signals(asset: Optional[str] = None) -> JSONResponse:
    """The live model bus, exposed. Returns the latest signal per
    (asset, model) plus a recent history ring for the brief feed.
    Optional ``asset`` filters to one symbol."""
    current = CORE.publisher.current()
    hist = CORE.publisher.history(limit=40)
    if asset:
        current = {k: v for k, v in current.items() if v.asset == asset}
        hist = [s for s in hist if s.asset == asset]
    return JSONResponse({
        "current": {k: v.to_row() for k, v in current.items()},
        "history": [s.to_row() for s in hist],
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
    # feed the anchoring detector: round-number cushions on small-
    # premium contracts are the textbook anchoring tell.
    CORE.psychology.record(TradeAction(
        ts_ist=datetime.now(IST).strftime("%H:%M:%S"),
        action="TRAIL_ARMED", symbol=body.tradingsymbol,
        qty=int(body.quantity), premium=float(premium),
        extras={"cushion_rupees": float(body.cushion_rupees),
                "premium": float(premium)},
    ))
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


class ProfitLockBody(BaseModel):
    mode: str = "ratio"               # ratio | buffer
    lock_ratio: float = 0.5
    give_back_buffer: float = 0.0
    activation_floor: float = 1000.0


@app.post("/api/profit_lock", dependencies=[Depends(auth)])
def arm_profit_lock(body: ProfitLockBody) -> JSONResponse:
    """Arm the ratcheting day-profit lock. The locked floor rises with
    every new peak and never falls; when live P&L falls to the floor,
    all positions flatten and the locked profit is realized."""
    try:
        CORE.profit_lock = ProfitLock(
            mode=body.mode, lock_ratio=body.lock_ratio,
            give_back_buffer=body.give_back_buffer,
            activation_floor=body.activation_floor)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    CORE.log(f"profit lock armed ({body.mode}): "
             f"{'keep '+str(int(body.lock_ratio*100))+'% of peak' if body.mode=='ratio' else 'max give-back ₹'+str(int(body.give_back_buffer))}")
    return JSONResponse(CORE.profit_lock.snapshot())


@app.delete("/api/profit_lock", dependencies=[Depends(auth)])
def cancel_profit_lock() -> JSONResponse:
    if CORE.profit_lock is not None:
        CORE.profit_lock.disarm()
        CORE.profit_lock = None
    CORE.log("profit lock disarmed")
    return JSONResponse({"disarmed": True})


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
    # Close the spine's execution gate too: even a hard-wired exit actor
    # cannot place an order while the kill switch is active.
    CORE.orchestrator.allow_execution = False
    n = len(CORE.trails.active())
    for t in CORE.trails.active():
        CORE.trails.cancel(t.trail_id)
    CORE.trails.cancel_portfolio()
    CORE.log(f"KILL SWITCH: {n} trail(s) cancelled, orders disabled "
             f"until restart")
    return JSONResponse({"killed": True, "trails_cancelled": n})


class GraduateBody(BaseModel):
    source: str
    tier: str          # SHADOW | LOGGED | TRUSTED (EXECUTION is reserved)
    # optional curator evidence behind the decision (auditable)
    evidence_window: Optional[str] = None
    n: Optional[int] = None
    hit_rate: Optional[float] = None
    expectancy: Optional[float] = None
    drawdown: Optional[float] = None
    calibration_error: Optional[float] = None
    leakage_status: Optional[str] = None
    note: Optional[str] = None


@app.post("/api/graduate", dependencies=[Depends(auth)])
def graduate(body: GraduateBody) -> JSONResponse:
    """The curator's lever, made operable: promote/demote a signal
    source's maximum tier. EXECUTION is refused for any source — only the
    two hard-wired exit actors (trailing_stop, profit_lock) may ever
    reach the order path, and they don't need a ceiling entry. Every
    promotion is recorded as an auditable TrustPromotionRecord."""
    try:
        tier = Tier[body.tier.upper()]
    except KeyError:
        raise HTTPException(400, f"tier must be one of "
                                 f"{[t.name for t in Tier]}")
    if tier >= Tier.EXECUTION:
        raise HTTPException(
            400, "EXECUTION is reserved for trailing_stop / profit_lock; "
                 "no other source may be graduated to the order path")
    from_tier = CORE.orchestrator.ceiling_of(body.source)
    CORE.orchestrator.set_ceiling(body.source, tier)
    evidence = {k: v for k, v in {
        "evidence_window": body.evidence_window, "n": body.n,
        "hit_rate": body.hit_rate, "expectancy": body.expectancy,
        "drawdown": body.drawdown, "calibration_error": body.calibration_error,
        "leakage_status": body.leakage_status, "note": body.note,
    }.items() if v is not None}
    rec = TrustPromotionRecord.build(
        body.source, from_tier, tier,
        ts_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **evidence)
    CORE.record_promotion(rec)
    CORE.log(f"{rec.decision} {body.source}: {from_tier.name} -> {tier.name}")
    return JSONResponse({"source": body.source, "ceiling": tier.name,
                         "promotion": rec.to_row(),
                         "stats": CORE.orchestrator.stats()})


# ---------------------------------------------------------------------------
# Customer SaaS surfaces — each gated through sentinel.saas.gate()
# ---------------------------------------------------------------------------

@app.get("/api/me/plan", dependencies=[Depends(auth)])
def me_plan(plan: str = Depends(resolve_plan)) -> JSONResponse:
    """The customer's own entitlement summary."""
    return JSONResponse(plan_summary(plan))


@app.get("/api/saas/catalog", dependencies=[Depends(auth)])
def saas_catalog(plan: str = Depends(resolve_plan)) -> JSONResponse:
    """Every paid surface, its required tier, and which the caller can access."""
    feats = features_for(plan)
    return JSONResponse({
        "plan": plan,
        "catalog": [
            {"feature": f, "required_tier": tier,
             "allowed": f in feats}
            for f, tier in sorted(FEATURE_CATALOG.items())
        ],
    })


class AuditLegBody(BaseModel):
    option_type: str
    strike: float
    qty: int
    premium: float
    delta: float = 0.0
    gamma: float = 0.0
    theta_per_day: float = 0.0
    vega_per_pct: float = 0.0


class AuditBody(BaseModel):
    legs: List[AuditLegBody]
    spot: float
    regime: Optional[str] = None


@app.post("/api/audit", dependencies=[Depends(auth)])
def api_audit(body: AuditBody,
              plan: str = Depends(resolve_plan)) -> JSONResponse:
    """Multi-leg payoff audit + Hull-taxonomy detection."""
    require("auditor.payoff_curve", plan)
    legs = [StrategyLeg(
        option_type=l.option_type, strike=l.strike, qty=l.qty,
        premium=l.premium, delta=l.delta, gamma=l.gamma,
        theta_per_day=l.theta_per_day, vega_per_pct=l.vega_per_pct,
    ) for l in body.legs]
    res = audit_strategy(legs, spot_now=body.spot, regime=body.regime)
    # The Greek book + risk flags are PRO-only — strip them for RETAIL.
    payload = {
        "detected_strategy": res.detected_strategy,
        "max_profit_rupees": res.max_profit_rupees,
        "max_loss_rupees": res.max_loss_rupees,
        "breakeven_points": res.breakeven_points,
        "payoff_curve": res.payoff_curve,
        "notes": res.notes,
    }
    if gate(plan, "auditor.greek_book").allowed:
        payload["net_greeks"] = res.net_greeks
    if gate(plan, "auditor.risk_flags").allowed:
        payload["flags"] = res.flags
    return JSONResponse(payload)


class ChainQuoteBody(BaseModel):
    option_type: str
    strike: float
    premium: float
    delta: float = 0.0
    gamma: float = 0.0
    theta_per_day: float = 0.0
    vega_per_pct: float = 0.0


class BuildBody(BaseModel):
    intent: str
    spot: float
    chain: List[ChainQuoteBody]   # chain quotes carry no position qty
    iv: float = 0.15
    t_years: float = 7 / 365
    qty: int = 75
    strike_step: float = 50.0
    regime: Optional[str] = None


@app.post("/api/build", dependencies=[Depends(auth)])
def api_build(body: BuildBody,
              plan: str = Depends(resolve_plan)) -> JSONResponse:
    """Multi-leg strategy builder for a customer intent."""
    # RETAIL gets one build per session via builder.one_per_session;
    # PRO+ gets unlimited via builder.unlimited.
    if not gate(plan, "builder.unlimited").allowed:
        require("builder.one_per_session", plan)
    if body.intent not in INTENTS:
        raise HTTPException(400, f"intent must be one of {INTENTS}")
    chain = [ChainQuote(
        option_type=c.option_type, strike=c.strike, premium=c.premium,
        delta=c.delta, gamma=c.gamma,
        theta_per_day=c.theta_per_day, vega_per_pct=c.vega_per_pct,
    ) for c in body.chain]
    return JSONResponse(build_strategy(
        body.intent, body.spot, chain, iv=body.iv,
        t_years=body.t_years, qty=body.qty,
        strike_step=body.strike_step, regime=body.regime,
    ))


class StressLegBody(BaseModel):
    tradingsymbol: str
    qty: int
    premium: float
    delta: float = 0.0
    gamma: float = 0.0
    theta_per_day: float = 0.0
    vega_per_pct: float = 0.0


class StressBody(BaseModel):
    legs: List[StressLegBody]
    spot: float
    scenario: Optional[str] = None     # None -> full matrix
    base_iv: float = 0.15
    t_years_remaining: float = 7 / 365


@app.post("/api/stress", dependencies=[Depends(auth)])
def api_stress(body: StressBody,
               plan: str = Depends(resolve_plan)) -> JSONResponse:
    """Crisis stress test — single scenario or full matrix."""
    legs = [LegExposure(
        tradingsymbol=l.tradingsymbol, qty=l.qty, premium=l.premium,
        delta=l.delta, gamma=l.gamma,
        theta_per_day=l.theta_per_day, vega_per_pct=l.vega_per_pct,
    ) for l in body.legs]
    if body.scenario is None:
        require("stress.full_matrix", plan)
        return JSONResponse({
            "matrix": stress_test_all(legs, body.spot,
                                       t_years_remaining=body.t_years_remaining,
                                       base_iv=body.base_iv),
        })
    require("stress.single_scenario", plan)
    if body.scenario not in SCENARIOS:
        raise HTTPException(400, f"unknown scenario: {body.scenario}; "
                                 f"available: {sorted(SCENARIOS)}")
    r = stress_test(legs, body.spot, body.scenario,
                    t_years_remaining=body.t_years_remaining,
                    base_iv=body.base_iv)
    return JSONResponse({
        "scenario": r.scenario, "date": r.date, "narrative": r.narrative,
        "spot_before": r.spot_before, "spot_after": r.spot_after,
        "spot_shock_pct": r.spot_shock_pct, "iv_shock_abs": r.iv_shock_abs,
        "portfolio_pnl": r.portfolio_pnl,
        "portfolio_pnl_pct": r.portfolio_pnl_pct,
        "margin_at_risk_rupees": r.margin_at_risk_rupees,
        "legs": [vars(l) for l in r.legs],
    })


class EquityBody(BaseModel):
    stock_returns_pct: Dict[str, float]
    index_return_pct: float


@app.post("/api/equity_context", dependencies=[Depends(auth)])
def api_equity_context(body: EquityBody,
                        plan: str = Depends(resolve_plan)) -> JSONResponse:
    """NIFTY top-10 regime classifier — full pack at PRO, scalar at RETAIL."""
    ctx = top10_contextual_layer(body.stock_returns_pct, body.index_return_pct)
    if gate(plan, "equity.full_context").allowed:
        return JSONResponse({
            "regime": ctx.regime,
            "index_return_pct": ctx.index_return_pct,
            "top10_summed_contribution_pct": ctx.top10_summed_contribution_pct,
            "breadth_up": ctx.breadth_up,
            "breadth_down": ctx.breadth_down,
            "leaders": [vars(c) for c in ctx.leaders],
            "laggards": [vars(c) for c in ctx.laggards],
            "contributions": [vars(c) for c in ctx.contributions],
            "effective_date": ctx.effective_date,
        })
    require("equity.divergence_scalar", plan)
    return JSONResponse({
        "regime": ctx.regime,
        "weightage_divergence": round(
            ctx.top10_summed_contribution_pct - ctx.index_return_pct, 3),
    })


class VarBody(BaseModel):
    pnls: List[float]
    alpha: float = 0.99
    method: str = "historical"        # "historical" | "cornish_fisher" | "es"


@app.post("/api/var", dependencies=[Depends(auth)])
def api_var(body: VarBody,
            plan: str = Depends(resolve_plan)) -> JSONResponse:
    """VaR / Expected Shortfall on a supplied P&L series."""
    if body.method == "historical":
        require("var.historical", plan)
        return JSONResponse(historical_var(body.pnls, alpha=body.alpha))
    if body.method == "cornish_fisher":
        require("var.cornish_fisher", plan)
        return JSONResponse(parametric_var(body.pnls, alpha=body.alpha))
    if body.method == "es":
        require("expected_shortfall", plan)
        return JSONResponse(expected_shortfall(body.pnls, alpha=body.alpha))
    raise HTTPException(400, "method must be historical|cornish_fisher|es")


class VolConeBody(BaseModel):
    closes: List[float]
    windows: Optional[List[int]] = None
    percentiles: Optional[List[int]] = None


@app.post("/api/vol_cone", dependencies=[Depends(auth)])
def api_vol_cone(body: VolConeBody,
                  plan: str = Depends(resolve_plan)) -> JSONResponse:
    """Burghardt & Lane vol cone for a series of closes."""
    require("vol_cone", plan)
    return JSONResponse(vol_cone(
        body.closes,
        windows=tuple(body.windows) if body.windows else (10, 20, 30, 60, 90),
        percentiles=tuple(body.percentiles) if body.percentiles else (10, 25, 50, 75, 90),
    ))
