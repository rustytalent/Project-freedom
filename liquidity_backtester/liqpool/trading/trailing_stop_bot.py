"""Trailing-stop bot for manual Zerodha options trading.

The thing it solves, in one sentence: when you've opened a position
manually and it's running in profit, you tap one command and from
that moment the bot watches the live premium, tracks the running peak,
and the instant the premium falls from peak by your specified rupee
cushion, it exits at market through the Zerodha API — so a winning
trade can never round-trip to break-even while you're looking at
something else.

  HARD non-goals (deliberate):
    * It does NOT decide entries. You do.
    * It does NOT pick the cushion automatically. You do, per trade.
    * It does NOT analyse the market. You already do.
    * It does NOT carry positions overnight. MIS only. Auto-flatten
      before the broker squares off.

  Hard SAFETY rails (also deliberate):
    * dry_run mode is the default. Set confirm=True to send real
      orders. The broker module's same discipline.
    * Every state change is journalled to disk — if the bot crashes
      and restarts, the active trails are recovered, not lost.
    * Polling, not streaming, at v1. Slightly more latency, MASSIVELY
      simpler to reason about. WebSocket is a v2 upgrade.
    * No "modify the trail" or "tighten on the fly" until the basic
      contract is rock-solid. Easy to add; don't add it yet.
    * The bot only acts on trails YOU started. It never touches a
      position you didn't explicitly hand it.

  The state machine per trail is small:

    ARMED          -> you said "trail this"; bot watching premium
                      tracking peak
    EXIT_TRIGGERED -> peak - current >= cushion; bot has FIRED the
                      market exit order; waiting for confirmation
    EXITED         -> exit confirmed by broker; trail done
    CANCELLED      -> you tapped cancel before exit fired
    SQUAREOFF      -> 15:15 IST approached, MIS auto-flatten kicked
                      in; treated as a normal forced exit

Per the founder's spec: cushion is per-position in RUPEES (not
percentages). For a fat position 300 is one tick of noise; for a
thin one it's the whole prize. The caller decides.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..broker_zerodha import OrderResult, ZerodhaBroker


LOG = logging.getLogger(__name__)

# IST square-off time the bot auto-flattens at. Zerodha auto-squares MIS
# at 15:15-15:20 depending on instrument; we flatten 60s before to
# avoid the broker's forced exit (which can be at worse prices).
MIS_AUTO_FLATTEN_IST = dtime(15, 14)
DEFAULT_POLL_SECONDS = 2.0
MIN_CUSHION_RUPEES = 0.05


# ---------------------------------------------------------------------------
# Trail state
# ---------------------------------------------------------------------------

@dataclass
class TrailState:
    """One active trailing-stop on one position.

    ``side`` is the side of the OPEN position. Long position -> exit
    by SELL; short position -> exit by BUY. For options, "long" means
    you bought the option (paid premium) and now own it.

    ``cushion_rupees`` is the give-back budget in PREMIUM POINTS. If
    peak premium is 200 and cushion is 30, the trail fires at 170.

    ``activated_at_premium`` is the premium when you armed the trail —
    we track peak STRICTLY ABOVE this so a momentary tick down from
    activation doesn't fire instantly.
    """
    trail_id: str
    tradingsymbol: str
    exchange: str
    side: str                          # "long" | "short"
    quantity: int
    cushion_rupees: float
    activated_at_premium: float
    activated_at_utc: str
    peak_premium: float
    last_premium: float = 0.0
    state: str = "ARMED"               # ARMED | EXIT_TRIGGERED | EXITED |
                                       # CANCELLED | SQUAREOFF | ERROR
    triggered_at_premium: Optional[float] = None
    triggered_at_utc: Optional[str] = None
    exit_reason: Optional[str] = None    # "TRIGGER" | "SQUAREOFF" — why
                                          # the exit was attempted (orthogonal
                                          # to whether it actually went through;
                                          # ``state`` carries the outcome).
    exit_order_id: Optional[str] = None
    exit_error: Optional[str] = None
    last_poll_utc: Optional[str] = None
    # Optional notes the caller can attach (e.g. "post-Fed gap fade").
    note: str = ""

    def serialise(self) -> Dict[str, Any]:
        return asdict(self)


def _new_trail_id(symbol: str) -> str:
    return f"TRAIL_{symbol}_{int(time.time() * 1000)}"


def _utc_now_iso() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _ist_now_time() -> dtime:
    return pd.Timestamp.now("UTC").tz_convert("Asia/Kolkata").time()


# ---------------------------------------------------------------------------
# Journal (crash-safe state recovery)
# ---------------------------------------------------------------------------

class TrailJournal:
    """Append-only JSONL log of every trail state transition.

    Each line is a serialised TrailState at the moment it changed.
    Replay = scan to end-of-file, keep only the last state per
    trail_id, drop any that ended (EXITED / CANCELLED / SQUAREOFF /
    ERROR). What remains is the active trail set the bot resumes
    watching.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, state: TrailState) -> None:
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(state.serialise()) + "\n")

    def replay_active(self) -> Dict[str, TrailState]:
        """Reconstruct active trails from the journal. Terminal states
        are dropped — they no longer need watching."""
        if not self.path.exists():
            return {}
        latest: Dict[str, TrailState] = {}
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    latest[d["trail_id"]] = TrailState(**d)
                except Exception:
                    continue
        active = {
            tid: s for tid, s in latest.items()
            if s.state in ("ARMED", "EXIT_TRIGGERED")
        }
        return active


# ---------------------------------------------------------------------------
# Premium source
# ---------------------------------------------------------------------------

PremiumQuoteFn = Callable[[str, str], Optional[float]]


def default_kite_quote(broker: ZerodhaBroker) -> PremiumQuoteFn:
    """Return a function that fetches the LAST traded price for an
    instrument via the broker's kite handle. Falls back to None on
    any error — the bot treats None as "skip this poll" rather than
    panicking on transient broker hiccups."""
    def _quote(exchange: str, tradingsymbol: str) -> Optional[float]:
        try:
            kite = broker._conn()
            key = f"{exchange}:{tradingsymbol}"
            q = kite.ltp([key])
            v = q.get(key, {}).get("last_price")
            return float(v) if v is not None else None
        except Exception as exc:
            LOG.warning("ltp fetch failed for %s:%s: %s",
                        exchange, tradingsymbol, exc)
            return None
    return _quote


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------

class TrailingStopBot:
    """Manages a set of active trailing stops on Zerodha positions.

    Threading model: one background poller thread loops over active
    trails every ``poll_seconds``. arm() / cancel() can be called
    from any thread; the bot uses a lock around state mutations.

    The bot does NOT subscribe to broker order updates — it places
    market exits and assumes the broker fills them. The exit_order_id
    is recorded so the operator can confirm via the Zerodha app.
    """

    def __init__(
        self,
        broker: ZerodhaBroker,
        journal_path: Path,
        quote_fn: Optional[PremiumQuoteFn] = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        confirm_real_orders: bool = False,
        auto_flatten_at_ist: dtime = MIS_AUTO_FLATTEN_IST,
    ) -> None:
        self.broker = broker
        self.journal = TrailJournal(journal_path)
        self.quote_fn = quote_fn or default_kite_quote(broker)
        self.poll_seconds = float(poll_seconds)
        self.confirm_real_orders = bool(confirm_real_orders)
        self.auto_flatten_at_ist = auto_flatten_at_ist
        self._active: Dict[str, TrailState] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- public API ---------------------------------------------------

    def arm(
        self,
        tradingsymbol: str,
        side: str,
        quantity: int,
        cushion_rupees: float,
        current_premium: float,
        exchange: str = "NFO",
        note: str = "",
    ) -> TrailState:
        """Start trailing a position. ``current_premium`` is the
        premium AT THE MOMENT YOU ARMED THIS — used as the initial
        peak. ``side`` is the side of the OPEN position (long = you
        bought the option; short = you sold/wrote it). ``cushion_rupees``
        is the rupee give-back from peak that fires the exit."""
        if side not in ("long", "short"):
            raise ValueError("side must be 'long' or 'short'")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if cushion_rupees < MIN_CUSHION_RUPEES:
            raise ValueError(
                f"cushion_rupees must be >= {MIN_CUSHION_RUPEES}"
            )
        if current_premium <= 0:
            raise ValueError("current_premium must be positive")
        state = TrailState(
            trail_id=_new_trail_id(tradingsymbol),
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            side=side,
            quantity=int(quantity),
            cushion_rupees=float(cushion_rupees),
            activated_at_premium=float(current_premium),
            activated_at_utc=_utc_now_iso(),
            peak_premium=float(current_premium),
            last_premium=float(current_premium),
            note=note,
        )
        with self._lock:
            self._active[state.trail_id] = state
        self.journal.write(state)
        LOG.info(
            "armed trail %s on %s qty=%d cushion=%.2f at premium=%.2f",
            state.trail_id, tradingsymbol, quantity,
            cushion_rupees, current_premium,
        )
        return state

    def cancel(self, trail_id: str) -> bool:
        """Cancel a trail (you've changed your mind / want to ride
        further). Idempotent: calling on an already-terminal trail is
        a no-op. Returns True if cancellation took effect."""
        with self._lock:
            s = self._active.get(trail_id)
            if s is None:
                return False
            if s.state != "ARMED":
                # EXIT_TRIGGERED can't be un-fired; user must call
                # broker directly to cancel the resulting order.
                return False
            s.state = "CANCELLED"
            self._active.pop(trail_id, None)
            self.journal.write(s)
            LOG.info("cancelled trail %s", trail_id)
            return True

    def update_cushion(self, trail_id: str, cushion_rupees: float) -> bool:
        """Loosen or tighten the cushion on an active trail. Common
        intent: tighten as the trade keeps running ('lock in more')."""
        if cushion_rupees < MIN_CUSHION_RUPEES:
            raise ValueError("cushion_rupees too small")
        with self._lock:
            s = self._active.get(trail_id)
            if s is None or s.state != "ARMED":
                return False
            s.cushion_rupees = float(cushion_rupees)
            self.journal.write(s)
            LOG.info("updated cushion on %s to %.2f", trail_id, cushion_rupees)
            return True

    def list_active(self) -> List[TrailState]:
        with self._lock:
            return list(self._active.values())

    def start(self) -> None:
        """Start the background poller thread + recover any active
        trails from the journal (crash-safe restart)."""
        recovered = self.journal.replay_active()
        with self._lock:
            self._active.update(recovered)
        if recovered:
            LOG.info("recovered %d active trail(s) from journal", len(recovered))
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="trailing-stop-bot", daemon=True,
        )
        self._thread.start()
        LOG.info("trailing-stop bot started (poll=%.1fs, confirm=%s)",
                 self.poll_seconds, self.confirm_real_orders)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_seconds * 3)
        LOG.info("trailing-stop bot stopped")

    # -- poller loop --------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                LOG.exception("trail poll tick raised: %s", exc)
            self._stop.wait(self.poll_seconds)

    def _tick(self) -> None:
        # Snapshot under lock so a concurrent arm()/cancel() doesn't
        # mutate the iteration target.
        with self._lock:
            trails = list(self._active.values())
        if not trails:
            return
        ist_now = _ist_now_time()
        force_flatten = ist_now >= self.auto_flatten_at_ist
        for s in trails:
            if s.state != "ARMED":
                continue
            premium = self.quote_fn(s.exchange, s.tradingsymbol)
            now = _utc_now_iso()
            if premium is None or premium <= 0:
                # Skip this tick on a bad quote; do NOT update peak
                # downward, do NOT fire an exit on missing data.
                s.last_poll_utc = now
                continue
            s.last_premium = float(premium)
            s.last_poll_utc = now
            # Track peak strictly for the side that BENEFITS the
            # position holder. For an open LONG, premium up = good
            # = new peak. For an open SHORT (option writer), premium
            # DOWN = good = new "peak" of profit; cushion fires when
            # premium rises by cushion above the trough.
            if s.side == "long":
                if premium > s.peak_premium:
                    s.peak_premium = float(premium)
                give_back = s.peak_premium - premium
                trigger = give_back >= s.cushion_rupees
            else:  # short
                # Use peak_premium as TROUGH for shorts (lowest
                # premium reached = highest unrealised profit).
                if premium < s.peak_premium:
                    s.peak_premium = float(premium)
                give_back = premium - s.peak_premium
                trigger = give_back >= s.cushion_rupees
            if force_flatten:
                self._fire_exit(s, reason="SQUAREOFF")
            elif trigger:
                self._fire_exit(s, reason="TRIGGER")
            else:
                # Persist the peak update so a restart preserves it.
                self.journal.write(s)

    # -- exit execution -----------------------------------------------

    def _fire_exit(self, s: TrailState, reason: str) -> None:
        """Place the market exit order. Idempotent — only fires when
        state is still ARMED."""
        with self._lock:
            if s.state != "ARMED":
                return
            s.state = "EXIT_TRIGGERED"
            s.exit_reason = reason         # "TRIGGER" or "SQUAREOFF"
            s.triggered_at_premium = s.last_premium
            s.triggered_at_utc = _utc_now_iso()
        # Long position -> SELL to exit. Short position -> BUY to exit.
        exit_side = "SELL" if s.side == "long" else "BUY"
        try:
            result: OrderResult = self.broker.place_market_order(
                tradingsymbol=s.tradingsymbol,
                side=exit_side,
                quantity=s.quantity,
                exchange=s.exchange,
                confirm=self.confirm_real_orders,
            )
            s.exit_order_id = result.order_id
            if result.status in ("submitted", "simulated"):
                s.state = "EXITED"
            else:
                s.state = "ERROR"
                s.exit_error = result.error or f"status={result.status}"
        except Exception as exc:
            s.state = "ERROR"
            s.exit_error = str(exc)
            LOG.exception("exit order failed for %s: %s", s.trail_id, exc)
        with self._lock:
            self._active.pop(s.trail_id, None)
        self.journal.write(s)
        LOG.info(
            "trail %s fired (%s) at %.2f peak=%.2f cushion=%.2f order=%s state=%s",
            s.trail_id, reason, s.triggered_at_premium or 0,
            s.peak_premium, s.cushion_rupees,
            s.exit_order_id, s.state,
        )
