"""Track 5b: event-driven post-touch confirmation engine.

The morning runner identifies tradeable POOLS. This engine watches those pools live during the
session and only fires a trade alert when one of three confirmation events lands:

  1. REJECTION WICK
     A bar with a long wick into the zone whose body closes BACK OUTSIDE. Classic SMC
     entry signal — price tested the level, was rejected, sellers (or buyers) showed up.
     Trigger: wick_length / ATR >= rejection_min_wick_atr, AND close is on the outside.

  2. SWEEP + RECLAIM
     A bar closes THROUGH the zone (the touch grabs liquidity), but within K bars
     another bar closes back INSIDE the zone. Stop-hunt that didn't hold — smart money
     pattern.

  3. BREAKDOWN (failure)
     A bar closes through by >= failure_close_through_atr × ATR, AND another bar within
     the failure-confirmation window also closes past. Pool is broken; abandon.

States:
  ARMED       — pool added at morning, no touch yet
  TOUCHED     — at least one bar entered the zone after arming
  TRIGGERED   — a confirmation event fired; emit a trade alert
  FAILED      — breakdown confirmed; pool is broken
  EXPIRED     — touch_window_bars passed after first touch with no clear signal

The engine is data-source-agnostic — feed it bars from any source (yfinance, Kite
historical_data, Kite WebSocket-built 5m bars). For live trading via Kite, the user
plugs in a fetcher that returns the latest bars per symbol.

Backtest replay mode: feed historical bars sequentially to verify state-machine logic
against real OHLCV. Used as a sanity check before going live.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Iterable
import json
import time
import pandas as pd
import numpy as np

from .pools import Pool
from .indicators import atr


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EventEngineConfig:
    rejection_min_wick_atr: float = 0.5
    """Wick must be at least this many ATRs into the zone to count as rejection."""

    rejection_max_close_through_atr: float = 0.15
    """Body close must not be more than this far past the zone (= no breakdown disguised
    as rejection)."""

    reclaim_within_bars: int = 6
    """After a close-through, allow this many bars for a reclaim back into the zone."""

    failure_close_through_atr: float = 0.5
    """Close past the zone by this many ATRs counts as a breakdown candidate."""

    failure_confirm_bars: int = 2
    """Need this many bars total closing past for a confirmed failure (1 = single-bar break;
    2 = next bar also closes past — recommended)."""

    touch_window_bars: int = 30
    """After first touch, give the pool this many bars to produce a signal before expiring."""

    poll_interval_sec: int = 30
    """How often the live loop fetches fresh bars."""


# ---------------------------------------------------------------------------
# Trigger event + pool state machine
# ---------------------------------------------------------------------------

@dataclass
class TriggerEvent:
    timestamp: str
    symbol: str
    pool_id: str
    trigger_type: str             # 'rejection_wick' | 'sweep_and_reclaim'
    side: str                      # 'buy' (pool below) or 'sell' (pool above)
    entry_price: float
    stop: float
    target: float
    q: float
    t_today: float
    ev: float
    reaction_atr: float = 0.0
    reaction_bars: int = 0
    extras: Dict = field(default_factory=dict)


@dataclass
class PoolState:
    pool_id: str
    symbol: str
    sector: str

    # Pool geometry — duplicated from Pool so state can persist independently
    pool_low: float
    pool_high: float
    side: str                      # 'buy' or 'sell'

    # Predictions at arming time
    q: float
    t_today: float
    ev: float
    dir_tag: str

    # Pre-computed trade plan
    entry_price: float
    stop_price: float
    target_price: float

    # State machine
    state: str = "ARMED"           # ARMED | TOUCHED | TRIGGERED | FAILED | EXPIRED
    armed_at: str = ""
    touched_at: Optional[str] = None
    bars_since_touch: int = 0

    # Tracking through touch window
    max_close_through_atr_since_touch: float = 0.0
    first_strong_break_idx: Optional[int] = None     # bars_since_touch at first strong break
    sweep_extreme_price: Optional[float] = None      # most extreme close-through price seen
    confirmed_strong_break_bars: int = 0
    close_through_streak: int = 0

    # Outcome
    trigger: Optional[Dict] = None     # serialised TriggerEvent
    fail_reason: Optional[str] = None
    expire_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PoolState":
        return cls(**d)


def make_pool_state(pool: Pool, symbol: str, sector: str, q: float, t_today: float,
                     ev: float, dir_tag: str, current_price: float, atr_proxy: float,
                     stop_buffer_atr: float = 0.7) -> PoolState:
    """Build a PoolState from a Pool + the trade-plan inputs used in live_run.py.
    The entry/stop/target follow the same rules used in live_run.py for consistency.
    `stop_buffer_atr` parks the stop beyond the decisive-break distance so a swept-and-reclaimed
    pool isn't stopped out during the sweep — keep it equal to live_run's STOP_BUFFER_ATR."""
    side = "buy" if pool.price_high < current_price else "sell"
    if side == "buy":
        entry = pool.price_high
        stop = pool.price_low - stop_buffer_atr * atr_proxy
        target = pool.price_high + 2 * atr_proxy
    else:
        entry = pool.price_low
        stop = pool.price_high + stop_buffer_atr * atr_proxy
        target = pool.price_low - 2 * atr_proxy
    return PoolState(
        pool_id=f"{symbol}@{pool.formed_at}",
        symbol=symbol, sector=sector,
        pool_low=float(pool.price_low), pool_high=float(pool.price_high),
        side=side, q=float(q), t_today=float(t_today), ev=float(ev), dir_tag=dir_tag,
        entry_price=float(entry), stop_price=float(stop), target_price=float(target),
        armed_at=pd.Timestamp.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# Pure trigger-detection functions (testable in isolation)
# ---------------------------------------------------------------------------

def _touch_bar(side: str, bar_low: float, bar_high: float,
                pool_low: float, pool_high: float) -> bool:
    """For BUY (pool below price): touch when bar_low <= pool_high.
    For SELL (pool above price): touch when bar_high >= pool_low."""
    if side == "buy":
        return bar_low <= pool_high
    return bar_high >= pool_low


def _close_through_atr(side: str, close: float, pool_low: float, pool_high: float,
                        atr_val: float) -> float:
    """Returns ATR-units that `close` is BEYOND the pool. 0 if inside or wrong side.
       For BUY pool: through means close < pool_low (broken support).
       For SELL pool: through means close > pool_high (broken resistance)."""
    a = max(atr_val, 1e-9)
    if side == "buy":
        if close < pool_low:
            return (pool_low - close) / a
    else:
        if close > pool_high:
            return (close - pool_high) / a
    return 0.0


def _rejection_wick_atr(side: str, bar_low: float, bar_high: float, bar_close: float,
                         pool_low: float, pool_high: float, atr_val: float,
                         max_close_through_atr: float) -> float:
    """Returns wick length in ATRs if this bar is a rejection wick. 0 otherwise.

    BUY pool (support):  bar_low touches pool zone (<= pool_high), bar_close > pool_high,
                          AND bar_close is not more than max_close_through_atr below the
                          high (i.e., body firmly above the zone).
    SELL pool (resistance): bar_high touches pool, bar_close < pool_low, body firmly below.
    """
    a = max(atr_val, 1e-9)
    if side == "buy":
        if bar_low > pool_high:                 # didn't actually touch
            return 0.0
        if bar_close <= pool_high:               # closed back inside or below — not rejection
            return 0.0
        # Body must close back NEAR the zone edge, not run away above it. A bar that closes
        # far above pool_high is a breakout, not a rejection-off-support — and entering at
        # pool_high would mean chasing far below the close.
        if (bar_close - pool_high) > max_close_through_atr * a:
            return 0.0
        wick = pool_high - bar_low
        return wick / a if wick > 0 else 0.0
    else:
        if bar_high < pool_low:
            return 0.0
        if bar_close >= pool_low:
            return 0.0
        if (pool_low - bar_close) > max_close_through_atr * a:
            return 0.0
        wick = bar_high - pool_low
        return wick / a if wick > 0 else 0.0


# ---------------------------------------------------------------------------
# State machine — advance ONE pool given a new bar
# ---------------------------------------------------------------------------

def advance_one_bar(state: PoolState, bar_ts: pd.Timestamp, bar_o: float, bar_h: float,
                     bar_l: float, bar_c: float, atr_val: float,
                     cfg: EventEngineConfig) -> Optional[TriggerEvent]:
    """Mutate `state` based on the new bar. Returns a TriggerEvent if the pool just
    transitioned to TRIGGERED, else None."""
    if state.state in ("TRIGGERED", "FAILED", "EXPIRED"):
        return None

    in_pool = _touch_bar(state.side, bar_l, bar_h, state.pool_low, state.pool_high)
    if state.state == "ARMED":
        if not in_pool:
            return None
        # First touch
        state.state = "TOUCHED"
        state.touched_at = str(bar_ts)
        state.bars_since_touch = 0

    # We're now in TOUCHED. Increment counters.
    if state.state == "TOUCHED":
        state.bars_since_touch += 1

        # Check 1: REJECTION WICK
        wick_atr = _rejection_wick_atr(state.side, bar_l, bar_h, bar_c,
                                         state.pool_low, state.pool_high, atr_val,
                                         cfg.rejection_max_close_through_atr)
        if wick_atr >= cfg.rejection_min_wick_atr:
            ev = TriggerEvent(
                timestamp=str(bar_ts), symbol=state.symbol, pool_id=state.pool_id,
                trigger_type="rejection_wick", side=state.side,
                entry_price=state.entry_price, stop=state.stop_price,
                target=state.target_price,
                q=state.q, t_today=state.t_today, ev=state.ev,
                reaction_atr=wick_atr, reaction_bars=state.bars_since_touch,
                extras={"wick_atr": float(wick_atr), "bar_close": float(bar_c)},
            )
            state.state = "TRIGGERED"
            state.trigger = asdict(ev)
            return ev

        # Track close-through magnitude
        ct = _close_through_atr(state.side, bar_c, state.pool_low, state.pool_high, atr_val)
        if ct > 0:
            state.close_through_streak += 1
        else:
            state.close_through_streak = 0
        if ct > state.max_close_through_atr_since_touch:
            state.max_close_through_atr_since_touch = ct
        # Record the FIRST strong close-through bar and the most extreme close-through price,
        # so the reclaim window is measured from the break (not from the touch) and a reclaim
        # entry can be protected with a stop beyond the actual sweep extreme.
        if ct >= cfg.failure_close_through_atr:
            if state.first_strong_break_idx is None:
                state.first_strong_break_idx = state.bars_since_touch
            if state.side == "buy":
                state.sweep_extreme_price = (bar_c if state.sweep_extreme_price is None
                                             else min(state.sweep_extreme_price, bar_c))
            else:
                state.sweep_extreme_price = (bar_c if state.sweep_extreme_price is None
                                             else max(state.sweep_extreme_price, bar_c))

        # Check 2: SWEEP + RECLAIM
        # A strong close-through occurred, then a bar closes back INSIDE the zone within
        # reclaim_within_bars OF THE BREAK (measured from the recorded first-break bar).
        if state.first_strong_break_idx is not None:
            inside = state.pool_low <= bar_c <= state.pool_high
            bars_since_break = state.bars_since_touch - state.first_strong_break_idx
            if inside and 0 < bars_since_break <= cfg.reclaim_within_bars:
                # Protect the reclaim entry with a stop just beyond the sweep extreme.
                if state.side == "buy":
                    sweep_stop = (state.sweep_extreme_price - 0.25 * atr_val
                                  if state.sweep_extreme_price is not None else state.stop_price)
                    reclaim_stop = min(state.stop_price, sweep_stop)
                else:
                    sweep_stop = (state.sweep_extreme_price + 0.25 * atr_val
                                  if state.sweep_extreme_price is not None else state.stop_price)
                    reclaim_stop = max(state.stop_price, sweep_stop)
                ev = TriggerEvent(
                    timestamp=str(bar_ts), symbol=state.symbol, pool_id=state.pool_id,
                    trigger_type="sweep_and_reclaim", side=state.side,
                    entry_price=bar_c,    # enter at the reclaim close
                    stop=float(reclaim_stop), target=state.target_price,
                    q=state.q, t_today=state.t_today, ev=state.ev,
                    reaction_atr=state.max_close_through_atr_since_touch,
                    reaction_bars=state.bars_since_touch,
                    extras={"max_through_atr": float(state.max_close_through_atr_since_touch),
                            "bars_since_break": int(bars_since_break)},
                )
                state.state = "TRIGGERED"
                state.trigger = asdict(ev)
                return ev

        # Check 3: BREAKDOWN (failure)
        # Need failure_confirm_bars consecutive close-throughs at >= failure_close_through_atr
        # OR one strong + N more closes outside (the close_through_streak).
        if (ct >= cfg.failure_close_through_atr and
                state.close_through_streak >= cfg.failure_confirm_bars):
            state.state = "FAILED"
            state.fail_reason = (f"close through {ct:.2f}ATR for "
                                  f"{state.close_through_streak} bars")
            return None

        # Expire after touch_window_bars
        if state.bars_since_touch >= cfg.touch_window_bars:
            state.state = "EXPIRED"
            state.expire_reason = (f"no signal in {cfg.touch_window_bars} bars after touch")
            return None

    return None


# ---------------------------------------------------------------------------
# Engine — orchestrates many pool-states across multiple symbols
# ---------------------------------------------------------------------------

class EventDrivenEngine:
    """Holds the active PoolState set, advances them as new bars arrive, returns
    fired TriggerEvents per call."""

    def __init__(self, cfg: Optional[EventEngineConfig] = None):
        self.cfg = cfg or EventEngineConfig()
        self.states_by_symbol: Dict[str, List[PoolState]] = {}

    def arm(self, state: PoolState) -> None:
        self.states_by_symbol.setdefault(state.symbol, []).append(state)

    def active_symbols(self) -> List[str]:
        out = []
        for sym, sts in self.states_by_symbol.items():
            if any(s.state in ("ARMED", "TOUCHED") for s in sts):
                out.append(sym)
        return out

    def process_bars(self, symbol: str, bars: pd.DataFrame,
                      atr_series: pd.Series) -> List[TriggerEvent]:
        """Advance every state of `symbol` through every bar in `bars`. Returns ordered
        TriggerEvents fired in this call. `bars` must have columns open/high/low/close
        indexed by timestamp. `atr_series` aligned to bars.index gives per-bar ATR."""
        events = []
        states = self.states_by_symbol.get(symbol, [])
        if not states:
            return events
        for ts, row in bars.iterrows():
            a = float(atr_series.loc[ts]) if ts in atr_series.index else float(atr_series.iloc[-1])
            for st in states:
                ev = advance_one_bar(st, ts, float(row["open"]), float(row["high"]),
                                       float(row["low"]), float(row["close"]), a, self.cfg)
                if ev is not None:
                    events.append(ev)
        return events

    def summary(self) -> Dict:
        out = {"by_state": {"ARMED": 0, "TOUCHED": 0, "TRIGGERED": 0,
                            "FAILED": 0, "EXPIRED": 0},
                "n_pools": 0}
        for sym, sts in self.states_by_symbol.items():
            for s in sts:
                out["by_state"][s.state] = out["by_state"].get(s.state, 0) + 1
                out["n_pools"] += 1
        return out

    # ----- Persistence -----

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cfg": asdict(self.cfg),
                    "states_by_symbol": {sym: [s.to_dict() for s in sts]
                                          for sym, sts in self.states_by_symbol.items()}}
        p.write_text(json.dumps(payload, default=str, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "EventDrivenEngine":
        d = json.loads(Path(path).read_text())
        eng = cls(EventEngineConfig(**d.get("cfg", {})))
        for sym, sts in d.get("states_by_symbol", {}).items():
            eng.states_by_symbol[sym] = [PoolState.from_dict(s) for s in sts]
        return eng


# ---------------------------------------------------------------------------
# Backtest replay — verify the engine works by replaying a historical bar series
# ---------------------------------------------------------------------------

def replay_backtest(engine: EventDrivenEngine, symbol: str, df_5m: pd.DataFrame,
                     atr_period: int = 14) -> List[TriggerEvent]:
    """Feed every bar from df_5m through the engine for `symbol`. Returns all triggers
    fired. Useful for sanity-checking the state machine on real historical data."""
    atr_series = atr(df_5m, atr_period).bfill()
    return engine.process_bars(symbol, df_5m, atr_series)


# ---------------------------------------------------------------------------
# Live polling loop — yfinance source (15m delay; for Kite use a custom fetcher)
# ---------------------------------------------------------------------------

DataFetcher = Callable[[str], Optional[pd.DataFrame]]


def yfinance_recent_5m(period_days: int = 2) -> DataFetcher:
    """Returns a fetcher that pulls the last N days of 5m bars from yfinance. Use this
    for development / Colab. yfinance has ~15-min delay on intraday for retail — not
    suitable for low-latency live trading. For production, use a kite_recent_5m() fetcher."""
    import yfinance as yf

    def fetch(symbol: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(symbol, period=f"{period_days}d", interval="5m",
                              progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns={c: str(c).lower() for c in df.columns})
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            return df[["open", "high", "low", "close"]].sort_index()
        except Exception as e:
            print(f"[event_engine] yfinance fetch failed for {symbol}: {e}")
            return None

    return fetch


def run_live_loop(engine: EventDrivenEngine, fetcher: DataFetcher,
                   on_trigger: Callable[[TriggerEvent], None],
                   end_at: Optional[pd.Timestamp] = None,
                   max_iterations: Optional[int] = None) -> None:
    """Main live loop. Polls every `engine.cfg.poll_interval_sec` until `end_at` or
    `max_iterations`. `on_trigger` is called for each TriggerEvent fired.

    The fetcher is plug-and-play: yfinance, Kite Connect, or any source returning a DataFrame
    of 5m bars with open/high/low/close columns. Each fetch returns ALL recent bars; we drop
    the final (still-forming) bar and only advance the state machine on CLOSED bars after the
    last one already processed (tracked per symbol in `last_processed`)."""
    iteration = 0
    last_processed: Dict[str, pd.Timestamp] = {}

    while True:
        if max_iterations is not None and iteration >= max_iterations:
            break
        if end_at is not None and pd.Timestamp.utcnow() >= end_at:
            print(f"[event_engine] reached end_at={end_at}, stopping")
            break

        active = engine.active_symbols()
        if not active:
            print(f"[event_engine] no active pools — exiting loop")
            break

        for sym in active:
            df = fetcher(sym)
            if df is None or df.empty:
                continue
            # The most recent bar from an intraday feed is the in-progress candle. Acting on it
            # would trigger on incomplete data, so process only fully-closed bars (drop the last).
            closed = df.iloc[:-1] if len(df) > 1 else df.iloc[:0]
            if closed.empty:
                continue
            last_ts = last_processed.get(sym)
            new_bars = closed.loc[closed.index > last_ts] if last_ts is not None else closed
            if new_bars.empty:
                continue
            try:
                atr_series = atr(df, 14).bfill()
            except Exception:
                atr_series = pd.Series(1.0, index=df.index)
            events = engine.process_bars(sym, new_bars, atr_series)
            for ev in events:
                on_trigger(ev)
            # Advance to the last CLOSED bar processed (NOT the dropped forming bar, so it gets
            # picked up once it closes on a later poll).
            last_processed[sym] = new_bars.index[-1]

        iteration += 1
        time.sleep(engine.cfg.poll_interval_sec)
