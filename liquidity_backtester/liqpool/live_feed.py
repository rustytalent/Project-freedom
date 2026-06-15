"""Live Kite WebSocket producer — the blocker.

The path that turns "REST polling at 1s" into "tick-by-tick over the
KiteTicker WebSocket → 5-min bars → state features → inference server
→ Sentinel cockpit." Closes Codex's #1 launch blocker.

What this module owns:

  * MinuteBarAggregator   — aggregates raw ticks into N-minute OHLCV
                            bars; emits at bar close
  * BarStateFeatureBuilder — a SLIM rolling-window feature builder
                            that produces what live_inference's heads
                            need (returns, ATR proxy, momentum,
                            session-position, recent high/low)
  * KiteFeedProducer       — wraps KiteTicker; subscribes to one or
                            more instrument tokens; pumps ticks into
                            the aggregator; on bar close calls the
                            inference server
  * LiveFeed              — orchestrator with start/stop, heartbeat,
                            reconnect counter, last-tick timestamp

Honest scope: this is a working live path for the MVP. The full
StateFeaturizer needs pools + multi-TF + sector context wired too;
this slim builder gives the bundle ENOUGH features to score, with
the documented gap that prediction quality won't fully match the
training-time backtest until the rest of the feature surfaces are
streamed in. Track in MOTHERS_AUDIT §3.2 follow-up.

The producer NEVER touches the trust spine. It only publishes signals
onto the JSONL the Sentinel-side LiveSignalsTail reads. Spine + auth
+ everything else flows through Sentinel exactly as it does in REST
mode.
"""
from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

from liqpool.contracts.signals import ModelSignal
from liqpool.live_inference import (
    JsonlPublisher, LiveInferenceBundle, LiveInferenceServer,
)

LOG = logging.getLogger("liqpool.live_feed")
IST = timezone(timedelta(hours=5, minutes=30))


# ─────────────────────────────────────────────────────────────────
# Bar aggregator — ticks → N-minute OHLCV
# ─────────────────────────────────────────────────────────────────

@dataclass
class Bar:
    """One closed N-minute OHLCV bar."""
    symbol: str
    bar_start_ts: float          # epoch seconds at bar open
    bar_end_ts: float            # epoch seconds at bar close
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    n_ticks: int = 0

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


class MinuteBarAggregator:
    """Per-symbol tick-to-bar aggregator. The KiteTicker callback feeds
    raw ticks; this emits closed Bars at the boundary.

    Bar boundaries are aligned to wall-clock minute ticks so a 5-min
    bar that opens at 09:15:00 closes at 09:20:00, the next opens
    immediately, etc. That matches the way the warehouse's resampled
    parquet was built — training and inference reference the same
    bar windows."""

    def __init__(self, timeframe_minutes: int = 5,
                 on_bar_close: Optional[Callable[[Bar], None]] = None) -> None:
        self.timeframe_seconds = timeframe_minutes * 60
        self._on_close = on_bar_close
        # symbol → (bar_start_ts, mutable accumulator)
        self._active: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _bar_start(self, ts: float) -> float:
        """Align ts down to the nearest bar boundary."""
        return (int(ts) // self.timeframe_seconds) * self.timeframe_seconds

    def feed(self, symbol: str, ts: float, price: float,
             volume: float = 0.0) -> List[Bar]:
        """Feed one tick. Returns the list of bars that closed AS A
        RESULT of this tick (usually 0; 1 when the tick straddled a
        boundary)."""
        if price <= 0 or ts <= 0:
            return []
        closed: List[Bar] = []
        with self._lock:
            bar_start = self._bar_start(ts)
            cur = self._active.get(symbol)
            if cur is None:
                self._active[symbol] = {
                    "start": bar_start, "open": price, "high": price,
                    "low": price, "close": price, "volume": float(volume),
                    "n": 1,
                }
                return []
            if bar_start > cur["start"]:
                # This tick rolls into a NEW bar. Close the previous one.
                closed.append(Bar(
                    symbol=symbol, bar_start_ts=cur["start"],
                    bar_end_ts=cur["start"] + self.timeframe_seconds,
                    open=cur["open"], high=cur["high"], low=cur["low"],
                    close=cur["close"], volume=cur["volume"],
                    n_ticks=cur["n"],
                ))
                self._active[symbol] = {
                    "start": bar_start, "open": price, "high": price,
                    "low": price, "close": price, "volume": float(volume),
                    "n": 1,
                }
            else:
                cur["high"] = max(cur["high"], price)
                cur["low"] = min(cur["low"], price)
                cur["close"] = price
                cur["volume"] += float(volume)
                cur["n"] += 1
        if self._on_close is not None:
            for b in closed:
                try:
                    self._on_close(b)
                except Exception as exc:
                    LOG.warning("on_bar_close raised: %s", exc)
        return closed

    def force_close_all(self) -> List[Bar]:
        """End-of-session helper — close every currently-open bar
        without waiting for the next tick (e.g. at 15:30 IST)."""
        closed: List[Bar] = []
        with self._lock:
            for symbol, cur in list(self._active.items()):
                closed.append(Bar(
                    symbol=symbol, bar_start_ts=cur["start"],
                    bar_end_ts=cur["start"] + self.timeframe_seconds,
                    open=cur["open"], high=cur["high"], low=cur["low"],
                    close=cur["close"], volume=cur["volume"],
                    n_ticks=cur["n"],
                ))
            self._active.clear()
        return closed


# ─────────────────────────────────────────────────────────────────
# Slim state-feature builder — what the bundle's heads need to score
# ─────────────────────────────────────────────────────────────────

class BarStateFeatureBuilder:
    """A small rolling-bar feature builder for live inference.

    What this gives you:
      * log-returns at 1 / 6 / 24 / 78 bars
      * intraday range/ATR proxy (high-low / close)
      * session position (bar index since 09:15 IST)
      * close-vs-recent-mean z-score
      * up/down streak length

    What this DOESN'T give you: pool-aware features (width_atr,
    n_contributors, age_bars), sector regime, multi-timeframe AVWAP,
    FRVP, etc. Those wire from the full StateFeaturizer in a
    follow-up sprint. For MVP, the bundle scores honestly on what's
    here; downgrade confidence accordingly via the inference
    server's `publish_floor`.
    """

    def __init__(self, max_history: int = 120) -> None:
        self._history: Deque[Bar] = deque(maxlen=max_history)
        self._lock = threading.Lock()

    def push(self, bar: Bar) -> None:
        with self._lock:
            self._history.append(bar)

    def features(self) -> Dict[str, float]:
        with self._lock:
            bars = list(self._history)
        if len(bars) < 2:
            return {"insufficient_history": 1.0}
        closes = [b.close for b in bars]
        last = closes[-1]
        out: Dict[str, float] = {}
        # log-returns over configured lookbacks
        for lb in (1, 6, 24, 78):
            if len(closes) > lb and closes[-lb - 1] > 0:
                out[f"ret_{lb}"] = _log_ret(closes[-lb - 1], last)
            else:
                out[f"ret_{lb}"] = 0.0
        # range proxy
        recent = bars[-min(14, len(bars)):]
        atr_proxy = (
            sum((b.high - b.low) for b in recent) / len(recent) / max(last, 1e-9)
        )
        out["range_atr_proxy"] = atr_proxy
        # close-vs-recent-mean z-score
        win = closes[-min(50, len(closes)):]
        if len(win) >= 4 and statistics.pstdev(win) > 0:
            out["zscore_close_50"] = (last - statistics.mean(win)) / statistics.pstdev(win)
        else:
            out["zscore_close_50"] = 0.0
        # session position (since 09:15 IST = epoch start 33300s into day)
        last_ts = bars[-1].bar_end_ts
        ist_dt = datetime.fromtimestamp(last_ts, tz=IST)
        session_open_min = 9 * 60 + 15
        cur_min = ist_dt.hour * 60 + ist_dt.minute
        out["session_minutes_in"] = max(0, cur_min - session_open_min)
        # up/down streak length (small but useful). Use explicit indexing
        # so negative-slice arithmetic doesn't break on short series.
        recent_closes = closes[-10:]
        streak, dirn = 0, 0
        for i in range(1, len(recent_closes)):
            a, b = recent_closes[i - 1], recent_closes[i]
            if b > a:
                streak = streak + 1 if dirn >= 0 else 1
                dirn = 1
            elif b < a:
                streak = streak + 1 if dirn <= 0 else 1
                dirn = -1
        out["streak"] = float(streak * (1 if dirn > 0 else -1 if dirn < 0 else 0))
        return out


def _log_ret(prev: float, cur: float) -> float:
    import math
    if prev <= 0 or cur <= 0:
        return 0.0
    return math.log(cur / prev)


# ─────────────────────────────────────────────────────────────────
# LiveFeed — the orchestrator
# ─────────────────────────────────────────────────────────────────

@dataclass
class FeedHeartbeat:
    state: str = "idle"                  # idle / running / reconnecting / stopped / error
    started_at_utc: Optional[str] = None
    last_tick_utc: Optional[str] = None
    last_bar_close_utc: Optional[str] = None
    last_published_utc: Optional[str] = None
    n_ticks: int = 0
    n_bars: int = 0
    n_signals_published: int = 0
    n_reconnects: int = 0
    last_error: str = ""

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


class LiveFeed:
    """Owns the bundle + bar aggregator + feature builder + inference
    server + JSONL sink. Codex's wiring code constructs one of these
    per process, calls start() once, lets the WS thread run.

    KiteTicker integration is via the on_ticks_callback parameter so
    this module has no hard dep on kiteconnect. The launch wiring
    snippet (see CODEX_BACKLOG §1.1) imports KiteTicker and points
    `kt.on_ticks = feed.on_ticks` before calling `kt.connect`."""

    HEAD_NAMES = ("direction", "quality", "proximity_h12",
                  "proximity_h25", "proximity_h78")

    def __init__(self, bundle: LiveInferenceBundle,
                 publisher_path: Path,
                 asset: str = "NIFTY",
                 timeframe_minutes: int = 5,
                 publish_floor: float = 0.30,
                 enforce_mis: bool = True,
                 audit_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None
                 ) -> None:
        self.bundle = bundle
        self.asset = asset
        self.audit_sink = audit_sink or (lambda evt, payload: None)
        self.feature_builder = BarStateFeatureBuilder()
        self.aggregator = MinuteBarAggregator(
            timeframe_minutes=timeframe_minutes,
            on_bar_close=self._on_bar_close)
        self.publisher = JsonlPublisher(publisher_path)
        self.server = LiveInferenceServer(
            bundle=bundle, publisher=self.publisher, asset=asset,
            publish_floor=publish_floor, enforce_mis=enforce_mis)
        self.heartbeat = FeedHeartbeat()
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────
    def start(self) -> None:
        with self._lock:
            self.heartbeat.state = "running"
            self.heartbeat.started_at_utc = _utcnow()
        self.audit_sink("live_feed_started",
                         {"asset": self.asset,
                          "bundle_version": self.bundle.bundle_version,
                          "head_count": len(self.bundle.heads)})

    def stop(self) -> None:
        # flush any open bars first so the operator sees the close
        closed = self.aggregator.force_close_all()
        for b in closed:
            self._on_bar_close(b)
        with self._lock:
            self.heartbeat.state = "stopped"
        self.audit_sink("live_feed_stopped", self.heartbeat.to_row())

    def mark_reconnect(self, reason: str = "") -> None:
        with self._lock:
            self.heartbeat.n_reconnects += 1
            self.heartbeat.state = "reconnecting"
            if reason:
                self.heartbeat.last_error = reason
        self.audit_sink("live_feed_reconnect",
                         {"reason": reason,
                          "n_reconnects": self.heartbeat.n_reconnects})

    def mark_back_running(self) -> None:
        with self._lock:
            self.heartbeat.state = "running"

    # ── tick path ─────────────────────────────────────────────────
    def on_ticks(self, ticks: Sequence[Dict[str, Any]]) -> None:
        """KiteTicker callback signature. Each tick is a dict at minimum
        carrying ``last_price`` and one of ``timestamp`` (epoch s) or
        ``last_trade_time`` (datetime). ``tradingsymbol`` is OPTIONAL
        — when a feed is subscribed to a single symbol, we fall back
        to ``self.asset``."""
        now = time.time()
        with self._lock:
            self.heartbeat.last_tick_utc = _utcnow()
        for t in ticks:
            ts = _coerce_ts(t.get("timestamp") or t.get("last_trade_time")
                              or now)
            price = float(t.get("last_price") or t.get("ltp") or 0.0)
            volume = float(t.get("volume_traded") or t.get("volume") or 0.0)
            symbol = str(t.get("tradingsymbol") or t.get("symbol")
                          or self.asset)
            self.aggregator.feed(symbol, ts, price, volume)
            with self._lock:
                self.heartbeat.n_ticks += 1

    def _on_bar_close(self, bar: Bar) -> None:
        """Bar closed — push to the feature builder, run inference,
        publish whatever fires."""
        self.feature_builder.push(bar)
        with self._lock:
            self.heartbeat.n_bars += 1
            self.heartbeat.last_bar_close_utc = _utcnow()
        features = self.feature_builder.features()
        if features.get("insufficient_history"):
            return
        # one feature vector goes to every head we discovered in the bundle.
        features_per_head: Dict[str, Dict[str, Any]] = {}
        for head in self.bundle.head_names():
            features_per_head[head] = features
        fired = self.server.on_tick(features_per_head,
                                      asset=self.asset,
                                      extras={"bar_close_ts": bar.bar_end_ts,
                                              "bundle_version":
                                                  self.bundle.bundle_version})
        with self._lock:
            self.heartbeat.n_signals_published += len(fired)
            if fired:
                self.heartbeat.last_published_utc = _utcnow()


# ─────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_ts(t: Any) -> float:
    """KiteTicker hands timestamps as either epoch seconds, datetime,
    or pandas Timestamp. Normalise to epoch seconds."""
    if t is None:
        return time.time()
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, datetime):
        return t.timestamp()
    try:
        return float(t.timestamp())                    # pandas Timestamp
    except Exception:
        try:
            return float(t)
        except Exception:
            return time.time()


# Module purpose (liqpool has no io_decl registry; this is documentation):
#   live Kite WebSocket producer — turns tick-by-tick KiteTicker feed
#   into 5-min OHLCV bars, builds slim state features per bar close,
#   runs the loaded model bundle, publishes signals via JsonlPublisher
#   onto the file Sentinel's LiveSignalsTail reads. Closes the #1
#   launch blocker (REST polling at 1s -> tick-by-tick live signals).
