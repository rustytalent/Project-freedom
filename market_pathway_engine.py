#!/usr/bin/env python3
"""
market_pathway_engine.py
=========================================================================
A standalone, dependency-free, multi-timeframe market-structure and
pathway-projection engine.

WHAT THIS IS
------------
This file is a self-contained research tool. It does not predict candles.
It builds a layered *context* of how price is structured across several
timeframes (1min, 5min, 15min, 1h, 4h, 1D — configurable), reconciles that
context into a single set of price levels that every timeframe agrees on,
scores how "engineered" the most recent session's liquidity was (i.e.
whether smart money likely swept stops and accumulated/distributed
inventory), and from that produces a small set of *plausible pathways* —
labeled sequences of structural "legs" (expansion, retracement, liquidity
sweep, consolidation, distribution, reversal) rendered as smooth curves —
for a chosen set of gap-up / gap-down scenarios. A human then looks at the
handful of rendered pathways and decides which one the market is tracing.

This is deliberately NOT a black box. Every number the engine produces
(a swing, a liquidity level, a sweep, a bias score, a pathway) is backed
by an explicit, inspectable rule, and every pathway ships with a written
narrative of *why* it was generated. Read `build_context_map()` and
`Pathway.narrative` if you want the "why", not just the "what".

WHY THIS FILE IS STANDALONE
----------------------------
The repository already has a substantial research package for liquidity-
pool detection (`liquidity_backtester/`). This module intentionally does
NOT import anything from it, and nothing in the repository imports this
module. It has one dependency policy: the Python 3.9+ standard library,
full stop. numpy/pandas/matplotlib are NOT required — swing detection,
ATR, percentiles, spline smoothing and SVG rendering are all implemented
by hand below. `kiteconnect` (Zerodha Kite Connect) is used ONLY if you
choose --mode live, and it is imported lazily so its absence never breaks
--mode synthetic or --mode csv.

You can run this file today, with no setup, no API keys, no network:

    python3 market_pathway_engine.py --demo

That runs the full pipeline against an internally-generated synthetic
dataset (a random walk seeded with realistic liquidity-sweep and
trend/consolidation regimes) and writes an SVG chart + JSON context map to
./market_pathway_output/.

To run it against real data via your Zerodha Kite Connect subscription:

    export KITE_API_KEY=...
    export KITE_ACCESS_TOKEN=...      # see the Kite login/OAuth flow docs
    python3 market_pathway_engine.py --mode live --symbol NIFTY 50 \
        --exchange NSE --instrument-token 256265 --lookback-days 60

(`--instrument-token` skips the instrument-master lookup; pass it if you
already know it. Without it the engine will call kite.instruments() once
and cache the resolved token to disk next to this file.)

THE CORE IDEA, IN ONE PARAGRAPH
--------------------------------
Every timeframe layer (1m..1D) gets: swing highs/lows, a market-structure
read (bullish / bearish / ranging, with the BOS/CHoCH that produced it),
an external range with a premium/discount read of where price currently
sits inside it, a set of liquidity levels (equal highs/lows, previous
day/week/month extremes, session extremes), a volatility-state read
(consolidating / expanding / distributing), and — critically — a pointer
to its immediate parent timeframe's range so you can see whether the 5m
structure is "a discount pullback inside a 1h premium leg inside a 4h
uptrend" or similar. All of those per-layer liquidity levels are then
merged, across every timeframe, into a single `LiquidityZone` ledger using
price-tolerance clustering — the same zone gets a higher confluence score
the more independent timeframes point at it. Pathways are only allowed to
target zones from that shared ledger (or an explicitly-labeled unconfluenced
Fibonacci projection), which is what guarantees that a pathway's "price
reached X" reads the same way on the 5m chart as it does on the 4h chart —
by construction, not by hope.

THE FOUR MODES
--------------
    (default)          generate tomorrow's gap scenarios + pathway charts
    --replay-days N    THE LEARNING LOOP: replay the last N completed
                       sessions; for each, generate scenarios from only
                       pre-open data, score every pathway against the tape
                       that actually printed (ATR-normalized curve RMSE),
                       and journal per-template match scores. Journal
                       history then tilts template ranking in every future
                       run via learned priors in [0.7, 1.3] — earned
                       preference, never total override.
    --track            LIVE TRACKER: mid-session, rank which of this
                       morning's pathways the day's tape is actually
                       tracing (scored over the elapsed session only),
                       with the realized tape overlaid on the chart.
    --login / --check-auth   Kite Connect daily OAuth + connectivity check.

Pathways are TIME-ANCHORED: each leg type carries a realistic session
duration (a stop-run sweep is minutes, midday consolidation eats hours)
and a time-of-day affinity (sweeps cluster at open/close, consolidation
at midday) that feeds plausibility. The x-axis is the actual NSE session
clock, 09:15 -> 15:30 IST.

The ledger includes, beyond swings/EQH/EQL/prior-period extremes: Fair
Value Gaps, order blocks, and volume-by-price high-volume nodes — all
merged through the same cross-timeframe confluence machinery.

Derivatives flow (--flow in live mode, or --flow-json for manual
injection) adds options put-call OI ratio + futures order-book depth
imbalance as a third input to gap probabilities, alongside the liquidity-
engineering bias and sentiment.

DISCLAIMER
----------
This is a decision-support / research-visualization tool. It does not
place trades, does not call any broker's order-entry endpoints, and the
gap/pathway probabilities it prints are transparent heuristics over the
detected structure — not a fitted statistical model and not investment
advice. The sentiment/news input is a manual pluggable hook
(`sentiment_provider`), not a live news feed, unless you wire one in.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

IST = timezone(timedelta(hours=5, minutes=30))

# =========================================================================
# 1. TIMEFRAME CONFIG
# =========================================================================

# Ordered fine -> coarse. Each entry: (name, minutes, kind)
# kind "intraday" = bucketed against the session anchor time within a day.
# kind "calendar" = bucketed against calendar day/week/month boundaries.
TIMEFRAME_SPECS: List[Tuple[str, int, str]] = [
    ("1min", 1, "intraday"),
    ("5min", 5, "intraday"),
    ("15min", 15, "intraday"),
    ("1h", 60, "intraday"),
    ("4h", 240, "intraday"),
    ("1D", 1440, "calendar_day"),
]

SESSION_OPEN = (9, 15)   # NSE cash session open, IST
SESSION_CLOSE = (15, 30)  # NSE cash session close, IST


# =========================================================================
# 2. DATA MODEL
# =========================================================================

@dataclass
class Candle:
    t: datetime
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0

    def to_dict(self) -> dict:
        return {"t": self.t.isoformat(), "o": self.o, "h": self.h,
                "l": self.l, "c": self.c, "v": self.v}


class CandleSeries:
    """Thin wrapper over a chronologically-sorted list of Candles with the
    handful of numeric helpers the rest of the engine needs. Pure stdlib —
    no numpy. Everything here is O(n) and written for clarity over speed;
    a few thousand candles per layer is the expected scale, not millions.
    """

    def __init__(self, candles: List[Candle]):
        self.candles = sorted(candles, key=lambda c: c.t)

    def __len__(self) -> int:
        return len(self.candles)

    def __getitem__(self, i) -> Candle:
        return self.candles[i]

    def __iter__(self):
        return iter(self.candles)

    def highs(self) -> List[float]:
        return [c.h for c in self.candles]

    def lows(self) -> List[float]:
        return [c.l for c in self.candles]

    def closes(self) -> List[float]:
        return [c.c for c in self.candles]

    def slice(self, start: Optional[datetime] = None,
              end: Optional[datetime] = None) -> "CandleSeries":
        out = [c for c in self.candles
               if (start is None or c.t >= start) and (end is None or c.t <= end)]
        return CandleSeries(out)

    def last_n(self, n: int) -> "CandleSeries":
        return CandleSeries(self.candles[-n:])

    def true_ranges(self) -> List[float]:
        trs = []
        prev_close = None
        for c in self.candles:
            if prev_close is None:
                trs.append(c.h - c.l)
            else:
                trs.append(max(c.h - c.l, abs(c.h - prev_close), abs(c.l - prev_close)))
            prev_close = c.c
        return trs

    def atr(self, n: int = 14) -> float:
        trs = self.true_ranges()
        if not trs:
            return 0.0
        window = trs[-n:] if len(trs) >= n else trs
        return sum(window) / len(window)

    def range_high_low(self) -> Tuple[float, float]:
        return max(self.highs()), min(self.lows())


def percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile, pure stdlib (no numpy)."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (pct / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


# =========================================================================
# 3. DATA PROVIDERS
# =========================================================================

class DataProvider:
    """Interface: fetch_base() must return 1-minute Candles, chronological,
    spanning `lookback_days` trading days up to `as_of`."""

    def fetch_base(self, as_of: datetime, lookback_days: int) -> List[Candle]:
        raise NotImplementedError


class SyntheticProvider(DataProvider):
    """Generates a 1-minute OHLCV series with the ingredients this engine is
    built to detect: daily trend/range regimes, injected liquidity sweeps
    (a wick punches through the prior day's high/low or an equal-high/low
    cluster, then price snaps back), and volatility regimes that alternate
    between consolidation and expansion. Fully deterministic given a seed,
    so --demo runs are reproducible. This exists so the whole pipeline is
    runnable and testable with zero credentials and zero network access.
    """

    def __init__(self, seed: int = 7, start_price: float = 22500.0):
        self.rng = random.Random(seed)
        self.start_price = start_price

    def fetch_base(self, as_of: datetime, lookback_days: int) -> List[Candle]:
        candles: List[Candle] = []
        price = self.start_price
        day = _prior_trading_day(as_of, lookback_days)
        prev_day_high = None
        prev_day_low = None

        while day.date() <= as_of.date():
            if day.weekday() >= 5:  # skip weekends
                day = day + timedelta(days=1)
                continue

            session_open = day.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1],
                                        second=0, microsecond=0)
            session_close = day.replace(hour=SESSION_CLOSE[0], minute=SESSION_CLOSE[1],
                                         second=0, microsecond=0)
            n_bars = int((session_close - session_open).total_seconds() // 60)

            regime = self.rng.choice(["trend_up", "trend_down", "range", "trend_up", "trend_down"])
            drift = {"trend_up": 0.9, "trend_down": -0.9, "range": 0.0}[regime]
            vol = self.rng.uniform(2.0, 6.0)

            # decide, ahead of time, whether today engineers a liquidity sweep
            # of yesterday's extreme before reversing - the signature this
            # engine's sweep/bias detectors are meant to pick up.
            sweep_plan = None
            if prev_day_high is not None and self.rng.random() < 0.35:
                side = self.rng.choice(["high", "low"])
                sweep_bar = self.rng.randint(int(n_bars * 0.1), int(n_bars * 0.4))
                sweep_plan = (side, sweep_bar)

            day_open = price
            day_high = day_open
            day_low = day_open
            t = session_open
            for i in range(n_bars):
                o = price
                step = self.rng.gauss(drift * 0.05, vol * 0.15)

                if sweep_plan and i == sweep_plan[1]:
                    side, _ = sweep_plan
                    target = prev_day_high if side == "high" else prev_day_low
                    if target is not None:
                        # punch through the level then start reverting
                        overshoot = vol * self.rng.uniform(1.5, 3.5)
                        price = target + overshoot if side == "high" else target - overshoot
                        drift = -drift if drift != 0 else (0.9 if side == "low" else -0.9)

                price = max(1.0, price + step)
                h = price + abs(self.rng.gauss(0, vol * 0.4))
                l = price - abs(self.rng.gauss(0, vol * 0.4))
                c = price
                v = max(0, self.rng.gauss(1500, 400))
                candles.append(Candle(t, o, max(h, o, c), min(l, o, c), c, v))
                day_high = max(day_high, h)
                day_low = min(day_low, l)
                t = t + timedelta(minutes=1)

            prev_day_high, prev_day_low = day_high, day_low
            day = day + timedelta(days=1)

        return candles


class CSVProvider(DataProvider):
    """Reads a local CSV with header: date,open,high,low,close,volume (any
    ISO-parsable date/datetime string). Use this to feed in an export from
    Kite, TradingView, or anywhere else, without wiring up live auth."""

    def __init__(self, path: str):
        self.path = path

    def fetch_base(self, as_of: datetime, lookback_days: int) -> List[Candle]:
        candles = []
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = datetime.fromisoformat(row["date"]).replace(tzinfo=IST) \
                    if datetime.fromisoformat(row["date"]).tzinfo is None \
                    else datetime.fromisoformat(row["date"])
                candles.append(Candle(
                    t, float(row["open"]), float(row["high"]),
                    float(row["low"]), float(row["close"]),
                    float(row.get("volume", 0) or 0),
                ))
        cutoff = as_of - timedelta(days=int(lookback_days * 1.6) + 5)
        return [c for c in candles if cutoff <= c.t <= as_of]


class ZerodhaProvider(DataProvider):
    """Live data via Zerodha Kite Connect. `kiteconnect` is imported lazily
    so its absence never breaks --mode synthetic/csv. Auth follows the same
    env-var convention as the rest of this repo's Kite tooling:
        KITE_API_KEY, KITE_ACCESS_TOKEN
    (access_token expires daily; regenerate it via Kite's OAuth flow before
    running this in live mode.)
    """

    TOKEN_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".market_pathway_instrument_cache.json")

    def __init__(self, symbol: str, exchange: str = "NSE",
                 instrument_token: Optional[int] = None,
                 api_key: Optional[str] = None, access_token: Optional[str] = None):
        self.symbol = symbol
        self.exchange = exchange
        self.instrument_token = instrument_token
        self.api_key = api_key or os.environ.get("KITE_API_KEY")
        self.access_token = access_token or os.environ.get("KITE_ACCESS_TOKEN")

    def _kite(self):
        try:
            from kiteconnect import KiteConnect
        except ImportError as e:
            raise ImportError(
                "Live mode needs the `kiteconnect` package: pip install kiteconnect"
            ) from e
        if not self.api_key or not self.access_token:
            raise RuntimeError(
                "Live mode needs KITE_API_KEY and KITE_ACCESS_TOKEN (env vars or "
                "constructor args). Generate an access_token via Kite's daily OAuth "
                "flow before running this."
            )
        kite = KiteConnect(api_key=self.api_key)
        kite.set_access_token(self.access_token)
        return kite

    def _resolve_token(self, kite) -> int:
        if self.instrument_token:
            return self.instrument_token
        cache = {}
        if os.path.exists(self.TOKEN_CACHE):
            try:
                with open(self.TOKEN_CACHE) as f:
                    cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                cache = {}
        key = f"{self.exchange}:{self.symbol}"
        if key in cache:
            return cache[key]
        for inst in kite.instruments(self.exchange):
            if inst.get("tradingsymbol") == self.symbol:
                cache[key] = inst["instrument_token"]
                try:
                    with open(self.TOKEN_CACHE, "w") as f:
                        json.dump(cache, f)
                except OSError:
                    pass
                return inst["instrument_token"]
        raise ValueError(f"Could not resolve instrument_token for {key}. "
                          f"Pass --instrument-token explicitly.")

    def fetch_base(self, as_of: datetime, lookback_days: int) -> List[Candle]:
        kite = self._kite()
        token = self._resolve_token(kite)
        start = _prior_trading_day(as_of, lookback_days)
        # Kite's minute-candle history endpoint caps at ~60 days per call;
        # chunk requests defensively.
        candles: List[Candle] = []
        chunk_start = start
        while chunk_start < as_of:
            chunk_end = min(chunk_start + timedelta(days=55), as_of)
            rows = kite.historical_data(token, chunk_start, chunk_end, "minute")
            for r in rows:
                candles.append(Candle(r["date"], r["open"], r["high"], r["low"],
                                       r["close"], r.get("volume", 0)))
            chunk_start = chunk_end + timedelta(minutes=1)
        return candles


def _prior_trading_day(as_of: datetime, lookback_days: int) -> datetime:
    d = as_of - timedelta(days=int(lookback_days * 1.6) + 5)  # pad for weekends
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


# =========================================================================
# 4. RESAMPLING (1-minute base -> every configured timeframe)
# =========================================================================

def _intraday_bucket(t: datetime, tf_minutes: int) -> datetime:
    """Bucket a timestamp to a timeframe boundary anchored at session open
    (09:15 IST), not clock-hour boundaries — a 5-minute NSE candle starts
    at 09:15, 09:20, ... not 09:00, 09:05."""
    anchor = t.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1], second=0, microsecond=0)
    delta_min = int((t - anchor).total_seconds() // 60)
    bucket_min = (delta_min // tf_minutes) * tf_minutes
    return anchor + timedelta(minutes=bucket_min)


def _calendar_day_bucket(t: datetime) -> datetime:
    return t.replace(hour=0, minute=0, second=0, microsecond=0)


def resample(base: List[Candle], tf_name: str, tf_minutes: int, kind: str) -> List[Candle]:
    if tf_minutes == 1:
        return sorted(base, key=lambda c: c.t)

    buckets: Dict[datetime, List[Candle]] = {}
    for c in sorted(base, key=lambda c: c.t):
        if kind == "intraday":
            key = _intraday_bucket(c.t, tf_minutes)
        else:  # calendar_day
            key = _calendar_day_bucket(c.t)
        buckets.setdefault(key, []).append(c)

    out = []
    for key in sorted(buckets):
        group = buckets[key]
        out.append(Candle(
            t=key,
            o=group[0].o,
            h=max(g.h for g in group),
            l=min(g.l for g in group),
            c=group[-1].c,
            v=sum(g.v for g in group),
        ))
    return out


def resample_weekly(daily: List[Candle]) -> List[Candle]:
    buckets: Dict[Tuple[int, int], List[Candle]] = {}
    for c in daily:
        iso_year, iso_week, _ = c.t.isocalendar()
        buckets.setdefault((iso_year, iso_week), []).append(c)
    out = []
    for key in sorted(buckets):
        g = buckets[key]
        out.append(Candle(g[0].t, g[0].o, max(x.h for x in g), min(x.l for x in g),
                           g[-1].c, sum(x.v for x in g)))
    return out


def resample_monthly(daily: List[Candle]) -> List[Candle]:
    buckets: Dict[Tuple[int, int], List[Candle]] = {}
    for c in daily:
        buckets.setdefault((c.t.year, c.t.month), []).append(c)
    out = []
    for key in sorted(buckets):
        g = buckets[key]
        out.append(Candle(g[0].t, g[0].o, max(x.h for x in g), min(x.l for x in g),
                           g[-1].c, sum(x.v for x in g)))
    return out


# =========================================================================
# 5. SWING / MARKET-STRUCTURE DETECTION
# =========================================================================

class SwingKind(Enum):
    HIGH = "high"
    LOW = "low"


@dataclass
class Swing:
    idx: int
    t: datetime
    price: float
    kind: SwingKind


def detect_swings(candles: List[Candle], left: int = 2, right: int = 2) -> List[Swing]:
    """Fractal swing detection: a bar is a swing high if its high is the
    strict max over `left` bars before and `right` bars after (swing low:
    symmetric on lows). This is the bedrock structural primitive everything
    else (BOS/CHoCH, EQH/EQL, ranges) is built from."""
    swings = []
    n = len(candles)
    for i in range(left, n - right):
        window_h = [candles[j].h for j in range(i - left, i + right + 1)]
        window_l = [candles[j].l for j in range(i - left, i + right + 1)]
        if candles[i].h == max(window_h) and window_h.count(candles[i].h) == 1:
            swings.append(Swing(i, candles[i].t, candles[i].h, SwingKind.HIGH))
        if candles[i].l == min(window_l) and window_l.count(candles[i].l) == 1:
            swings.append(Swing(i, candles[i].t, candles[i].l, SwingKind.LOW))
    return swings


class StructureBias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


@dataclass
class StructureEvent:
    t: datetime
    price: float
    kind: str          # "BOS" (break of structure, trend-confirming) or "CHoCH" (change of character)
    direction: StructureBias


def analyze_market_structure(candles: List[Candle], swings: List[Swing]
                              ) -> Tuple[StructureBias, List[StructureEvent]]:
    """Walks the swing sequence and classifies each new swing break as a
    BOS (break of structure — the prevailing trend takes out the prior
    same-direction swing, confirming continuation) or a CHoCH (change of
    character — price breaks the most recent opposite-direction swing,
    signalling the trend may be flipping). The returned bias is simply
    "whatever the last event implied"."""
    events: List[StructureEvent] = []
    bias = StructureBias.RANGING
    if len(swings) < 2:
        return bias, events

    last_high = None
    last_low = None
    for sw in swings:
        if sw.kind == SwingKind.HIGH:
            if last_high is not None and sw.price > last_high.price:
                kind = "BOS" if bias == StructureBias.BULLISH else "CHoCH"
                events.append(StructureEvent(sw.t, sw.price, kind, StructureBias.BULLISH))
                bias = StructureBias.BULLISH
            last_high = sw
        else:
            if last_low is not None and sw.price < last_low.price:
                kind = "BOS" if bias == StructureBias.BEARISH else "CHoCH"
                events.append(StructureEvent(sw.t, sw.price, kind, StructureBias.BEARISH))
                bias = StructureBias.BEARISH
            last_low = sw

    return bias, events


@dataclass
class RangeContext:
    high: float
    low: float
    mid: float
    current_price: float
    position_pct: float   # 0 = at range low, 100 = at range high
    state: str             # "premium" (>55), "discount" (<45), "equilibrium"


def compute_range_context(candles: List[Candle], lookback: int = 120) -> RangeContext:
    """The 'external range' this timeframe is currently trading inside:
    the highest-high / lowest-low over the trailing `lookback` bars. Where
    current price sits inside that range (premium vs discount, ICT-style,
    split around the 50% equilibrium) is one of the strongest per-layer
    context signals — a layer trading in discount of its own range is a
    very different regime than one trading in premium."""
    window = candles[-lookback:] if len(candles) >= lookback else candles
    hi = max(c.h for c in window)
    lo = min(c.l for c in window)
    mid = (hi + lo) / 2
    cur = candles[-1].c
    pct = 50.0 if hi == lo else (cur - lo) / (hi - lo) * 100.0
    if pct >= 55:
        state = "premium"
    elif pct <= 45:
        state = "discount"
    else:
        state = "equilibrium"
    return RangeContext(hi, lo, mid, cur, pct, state)


# =========================================================================
# 6. LIQUIDITY LEVELS
# =========================================================================

class LevelType(Enum):
    EQH = "equal_highs"
    EQL = "equal_lows"
    PDH = "prev_day_high"
    PDL = "prev_day_low"
    PWH = "prev_week_high"
    PWL = "prev_week_low"
    PMH = "prev_month_high"
    PML = "prev_month_low"
    SESSION_HIGH = "session_high"
    SESSION_LOW = "session_low"
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    FVG_BULL = "fvg_bullish"
    FVG_BEAR = "fvg_bearish"
    OB_BULL = "order_block_bullish"
    OB_BEAR = "order_block_bearish"
    HVN = "high_volume_node"


@dataclass
class LiquidityLevel:
    price: float
    type: LevelType
    timeframe: str
    t: datetime
    strength: float = 1.0   # base weight before cross-TF confluence multiplier


# Heavier weight = a heavier stop cluster / more consequential level.
_LEVEL_BASE_WEIGHT = {
    LevelType.EQH: 1.3, LevelType.EQL: 1.3,
    LevelType.PDH: 1.6, LevelType.PDL: 1.6,
    LevelType.PWH: 2.0, LevelType.PWL: 2.0,
    LevelType.PMH: 2.4, LevelType.PML: 2.4,
    LevelType.SESSION_HIGH: 1.0, LevelType.SESSION_LOW: 1.0,
    LevelType.SWING_HIGH: 0.8, LevelType.SWING_LOW: 0.8,
    LevelType.FVG_BULL: 1.1, LevelType.FVG_BEAR: 1.1,
    LevelType.OB_BULL: 1.2, LevelType.OB_BEAR: 1.2,
    LevelType.HVN: 1.4,
}


def detect_equal_highs_lows(swings: List[Swing], timeframe: str, tolerance: float
                             ) -> List[LiquidityLevel]:
    """Two or more swing highs (or lows) within `tolerance` price-units of
    each other read as a single, heavier resting-liquidity cluster (EQH /
    EQL) rather than two separate swings — that's where stops actually
    stack."""
    levels = []
    highs = sorted([s for s in swings if s.kind == SwingKind.HIGH], key=lambda s: s.price)
    lows = sorted([s for s in swings if s.kind == SwingKind.LOW], key=lambda s: s.price)

    for group, kind, ltype in ((highs, SwingKind.HIGH, LevelType.EQH),
                                (lows, SwingKind.LOW, LevelType.EQL)):
        used = [False] * len(group)
        for i in range(len(group)):
            if used[i]:
                continue
            cluster = [group[i]]
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                if abs(group[j].price - group[i].price) <= tolerance:
                    cluster.append(group[j])
                    used[j] = True
            if len(cluster) >= 2:
                avg_price = sum(s.price for s in cluster) / len(cluster)
                latest_t = max(s.t for s in cluster)
                strength = _LEVEL_BASE_WEIGHT[ltype] * (1 + 0.25 * (len(cluster) - 2))
                levels.append(LiquidityLevel(avg_price, ltype, timeframe, latest_t, strength))
    return levels


def detect_periodic_extremes(daily: List[Candle], weekly: List[Candle],
                              monthly: List[Candle], as_of_date, timeframe: str
                              ) -> List[LiquidityLevel]:
    """Previous day / week / month high & low — the classic external-
    liquidity reference points. `as_of_date` marks 'P day'; we want the
    completed P-1 day/week/month, not the in-progress one."""
    levels = []
    prior_days = [c for c in daily if c.t.date() < as_of_date]
    if prior_days:
        p = prior_days[-1]
        levels.append(LiquidityLevel(p.h, LevelType.PDH, timeframe, p.t, _LEVEL_BASE_WEIGHT[LevelType.PDH]))
        levels.append(LiquidityLevel(p.l, LevelType.PDL, timeframe, p.t, _LEVEL_BASE_WEIGHT[LevelType.PDL]))

    prior_weeks = [c for c in weekly if c.t.date() < as_of_date]
    if prior_weeks:
        p = prior_weeks[-1]
        levels.append(LiquidityLevel(p.h, LevelType.PWH, timeframe, p.t, _LEVEL_BASE_WEIGHT[LevelType.PWH]))
        levels.append(LiquidityLevel(p.l, LevelType.PWL, timeframe, p.t, _LEVEL_BASE_WEIGHT[LevelType.PWL]))

    prior_months = [c for c in monthly if c.t.date() < as_of_date]
    if prior_months:
        p = prior_months[-1]
        levels.append(LiquidityLevel(p.h, LevelType.PMH, timeframe, p.t, _LEVEL_BASE_WEIGHT[LevelType.PMH]))
        levels.append(LiquidityLevel(p.l, LevelType.PML, timeframe, p.t, _LEVEL_BASE_WEIGHT[LevelType.PML]))

    return levels


def detect_fvgs(candles: List[Candle], timeframe: str, atr: float,
                 keep_recent: int = 30) -> List[LiquidityLevel]:
    """Fair Value Gaps: the classic three-candle imbalance. A bullish FVG
    exists when candle i's low is strictly above candle i-2's high — the
    middle candle displaced so fast that a price band traded on only one
    side of the book. That unfilled band acts as a magnet/support on
    revisit. We anchor the level at the gap midpoint and require the gap
    to be at least 0.15 ATR wide so micro-gaps don't flood the ledger."""
    levels: List[LiquidityLevel] = []
    min_gap = atr * 0.15
    for i in range(2, len(candles)):
        a, c = candles[i - 2], candles[i]
        if c.l > a.h and (c.l - a.h) >= min_gap:
            levels.append(LiquidityLevel((c.l + a.h) / 2, LevelType.FVG_BULL, timeframe,
                                          c.t, _LEVEL_BASE_WEIGHT[LevelType.FVG_BULL]))
        elif c.h < a.l and (a.l - c.h) >= min_gap:
            levels.append(LiquidityLevel((c.h + a.l) / 2, LevelType.FVG_BEAR, timeframe,
                                          c.t, _LEVEL_BASE_WEIGHT[LevelType.FVG_BEAR]))
    return levels[-keep_recent:]


def detect_order_blocks(candles: List[Candle], timeframe: str, atr: float,
                         keep_recent: int = 30) -> List[LiquidityLevel]:
    """Order blocks: the last opposite-direction candle immediately before
    a displacement move (a candle whose body is >= 1.2 ATR). The logic:
    institutions filling size leave their footprint in that final counter-
    candle before price launches — its body midpoint is where unfilled
    institutional orders are presumed to rest, so revisits react there."""
    levels: List[LiquidityLevel] = []
    min_body = atr * 1.2
    for i in range(0, len(candles) - 1):
        prev, cur = candles[i], candles[i + 1]
        body = abs(cur.c - cur.o)
        if body < min_body:
            continue
        if cur.c > cur.o and prev.c < prev.o:      # bearish candle, then bullish displacement
            levels.append(LiquidityLevel((prev.o + prev.c) / 2, LevelType.OB_BULL, timeframe,
                                          prev.t, _LEVEL_BASE_WEIGHT[LevelType.OB_BULL]))
        elif cur.c < cur.o and prev.c > prev.o:    # bullish candle, then bearish displacement
            levels.append(LiquidityLevel((prev.o + prev.c) / 2, LevelType.OB_BEAR, timeframe,
                                          prev.t, _LEVEL_BASE_WEIGHT[LevelType.OB_BEAR]))
    return levels[-keep_recent:]


def detect_volume_nodes(candles: List[Candle], timeframe: str,
                         lookback: int = 300, n_bins: int = 24) -> List[LiquidityLevel]:
    """High-volume nodes from a volume-by-price histogram: bin the trailing
    window's closes into `n_bins` price buckets weighted by traded volume;
    a local-maximum bin carrying >= 1.5x the average bin volume marks a
    price the market repeatedly accepted (an old value area). Those act as
    consolidation magnets on revisit — this is the detector that encodes
    'which price levels had the most consolidation' directly."""
    window = candles[-lookback:] if len(candles) > lookback else candles
    if len(window) < 20:
        return []
    lo = min(c.l for c in window)
    hi = max(c.h for c in window)
    if hi <= lo:
        return []
    if not any(c.v > 0 for c in window):
        return []  # no volume data on this feed -> detector honestly abstains
    bin_w = (hi - lo) / n_bins
    vols = [0.0] * n_bins
    for c in window:
        idx = min(n_bins - 1, max(0, int((c.c - lo) / bin_w)))
        vols[idx] += c.v
    avg = sum(vols) / n_bins
    levels: List[LiquidityLevel] = []
    for i in range(1, n_bins - 1):
        if vols[i] > vols[i - 1] and vols[i] >= vols[i + 1] and vols[i] >= avg * 1.5:
            price = lo + (i + 0.5) * bin_w
            levels.append(LiquidityLevel(price, LevelType.HVN, timeframe, window[-1].t,
                                          _LEVEL_BASE_WEIGHT[LevelType.HVN]))
    return levels


# --- Liquidity sweeps -----------------------------------------------------

class SweepClassification(Enum):
    GRAB = "predatory_grab"        # swept then reversed hard: liquidity engineered, not a real breakout
    BREAKOUT = "genuine_breakout"  # swept and follow-through continued: real expansion


@dataclass
class SweepEvent:
    t: datetime
    level: LiquidityLevel
    side: str                # "buy_side" (swept a high) or "sell_side" (swept a low)
    swept_by: float          # points beyond the level the wick reached
    reversal_strength: float  # 0..1, how convincingly price reclaimed the level within the window
    est_volume: float
    classification: SweepClassification


def detect_liquidity_sweeps(candles: List[Candle], levels: List[LiquidityLevel],
                             reversal_window: int = 6) -> List[SweepEvent]:
    """A sweep: some candle's wick pierces a known liquidity level, and
    within `reversal_window` bars price closes back on the origin side of
    that level. We score `reversal_strength` by how much of the pierce got
    reclaimed, and classify GRAB (strong reclaim -> liquidity engineered,
    a probable trap) vs BREAKOUT (weak/no reclaim -> real continuation).
    `est_volume` is a heuristic proxy for "how much liquidity was taken" —
    true resting-order size isn't observable from OHLCV, so we approximate
    it as the traded volume on the piercing bar, which at least tracks
    participation at the level."""
    events = []
    idx_by_time = {c.t: i for i, c in enumerate(candles)}
    _HIGH_TYPES = (LevelType.EQH, LevelType.PDH, LevelType.PWH, LevelType.PMH,
                   LevelType.SESSION_HIGH, LevelType.SWING_HIGH)
    _LOW_TYPES = (LevelType.EQL, LevelType.PDL, LevelType.PWL, LevelType.PML,
                  LevelType.SESSION_LOW, LevelType.SWING_LOW)
    for lvl in levels:
        # FVG/OB/HVN are magnet/reaction zones, not stop clusters — a wick
        # through them isn't a "sweep" in the stop-run sense, so they are
        # excluded from sweep classification (they still shape the ledger).
        if lvl.type not in _HIGH_TYPES and lvl.type not in _LOW_TYPES:
            continue
        is_high = lvl.type in _HIGH_TYPES
        for i, c in enumerate(candles):
            if c.t <= lvl.t:
                continue  # only look for sweeps of a level after it was established
            pierced = (c.h > lvl.price) if is_high else (c.l < lvl.price)
            if not pierced:
                continue
            swept_by = (c.h - lvl.price) if is_high else (lvl.price - c.l)
            window = candles[i:i + reversal_window]
            if not window:
                continue
            reclaim_close = window[-1].c
            reclaimed = (reclaim_close < lvl.price) if is_high else (reclaim_close > lvl.price)
            if is_high:
                strength = max(0.0, min(1.0, (c.h - reclaim_close) / swept_by)) if swept_by > 0 else 0.0
            else:
                strength = max(0.0, min(1.0, (reclaim_close - c.l) / swept_by)) if swept_by > 0 else 0.0
            classification = SweepClassification.GRAB if (reclaimed and strength > 0.5) \
                else SweepClassification.BREAKOUT
            events.append(SweepEvent(
                t=c.t, level=lvl, side="buy_side" if is_high else "sell_side",
                swept_by=swept_by, reversal_strength=strength, est_volume=c.v,
                classification=classification,
            ))
            break  # one sweep per level is enough context; avoid double counting the same pierce
    return events


# --- Volatility / consolidation-vs-expansion state ------------------------

class VolState(Enum):
    CONSOLIDATION = "consolidation"
    EXPANSION = "expansion"
    DISTRIBUTION = "distribution"  # wide range, but closes clustering near one edge (a topping/bottoming signature)


def classify_volatility_state(candles: List[Candle], lookback: int = 30) -> VolState:
    """Range-width, normalized by its own trailing distribution (a
    percentile-rank, not a fixed threshold, so this adapts per-instrument
    and per-regime instead of hard-coding a point value). Low percentile ->
    consolidation (coiling). High percentile with closes bunched at one
    extreme of the range -> distribution. High percentile with closes
    spread through the range -> expansion."""
    if len(candles) < 10:
        return VolState.CONSOLIDATION
    window = candles[-lookback:] if len(candles) >= lookback else candles
    ranges = [c.h - c.l for c in window]
    current_range = ranges[-1] if ranges else 0.0
    rank = sum(1 for r in ranges if r <= current_range) / len(ranges) * 100.0

    hi = max(c.h for c in window)
    lo = min(c.l for c in window)
    span = hi - lo if hi > lo else 1.0
    closes_pos = [(c.c - lo) / span for c in window]
    close_spread = statistics.pstdev(closes_pos) if len(closes_pos) > 1 else 0.5

    if rank <= 40:
        return VolState.CONSOLIDATION
    if close_spread < 0.15:
        return VolState.DISTRIBUTION
    return VolState.EXPANSION


# =========================================================================
# 7. LAYER CONTEXT (one bundle per timeframe)
# =========================================================================

@dataclass
class LayerContext:
    timeframe: str
    candles: List[Candle]
    swings: List[Swing]
    bias: StructureBias
    structure_events: List[StructureEvent]
    range_ctx: RangeContext
    levels: List[LiquidityLevel]
    sweeps: List[SweepEvent]
    vol_state: VolState
    parent_timeframe: Optional[str] = None
    parent_relation: Optional[str] = None  # human-readable nesting description


def build_layer_context(tf_name: str, candles: List[Candle], daily: List[Candle],
                         weekly: List[Candle], monthly: List[Candle],
                         as_of_date, parent: Optional[LayerContext] = None) -> LayerContext:
    swings = detect_swings(candles, left=2, right=2)
    bias, events = analyze_market_structure(candles, swings)
    range_ctx = compute_range_context(candles, lookback=min(120, len(candles)))

    atr = CandleSeries(candles).atr(14)
    tolerance = max(atr * 0.15, 0.0001)

    levels = detect_equal_highs_lows(swings, tf_name, tolerance)
    levels += detect_periodic_extremes(daily, weekly, monthly, as_of_date, tf_name)
    if candles:
        session_day = candles[-1].t.date()
        today_candles = [c for c in candles if c.t.date() == session_day]
        if today_candles:
            levels.append(LiquidityLevel(max(c.h for c in today_candles), LevelType.SESSION_HIGH,
                                          tf_name, today_candles[-1].t, _LEVEL_BASE_WEIGHT[LevelType.SESSION_HIGH]))
            levels.append(LiquidityLevel(min(c.l for c in today_candles), LevelType.SESSION_LOW,
                                          tf_name, today_candles[-1].t, _LEVEL_BASE_WEIGHT[LevelType.SESSION_LOW]))
    for sw in swings[-40:]:
        lt = LevelType.SWING_HIGH if sw.kind == SwingKind.HIGH else LevelType.SWING_LOW
        levels.append(LiquidityLevel(sw.price, lt, tf_name, sw.t, _LEVEL_BASE_WEIGHT[lt]))

    levels += detect_fvgs(candles, tf_name, atr)
    levels += detect_order_blocks(candles, tf_name, atr)
    levels += detect_volume_nodes(candles, tf_name)

    sweeps = detect_liquidity_sweeps(candles, levels, reversal_window=6)
    vol_state = classify_volatility_state(candles, lookback=30)

    parent_relation = None
    if parent is not None:
        cur_price = candles[-1].c
        if parent.range_ctx.low <= cur_price <= parent.range_ctx.high:
            parent_relation = (
                f"{tf_name} price ({cur_price:.2f}) sits in the {range_ctx.state} of its own "
                f"range and in the {parent.range_ctx.state} ({parent.range_ctx.position_pct:.0f}%) "
                f"of the {parent.timeframe} range [{parent.range_ctx.low:.2f}, "
                f"{parent.range_ctx.high:.2f}], while {parent.timeframe} bias reads "
                f"{parent.bias.value}."
            )
        else:
            parent_relation = (
                f"{tf_name} price ({cur_price:.2f}) has traded OUTSIDE the last "
                f"{parent.timeframe} range [{parent.range_ctx.low:.2f}, {parent.range_ctx.high:.2f}] "
                f"— the {parent.timeframe} range itself is stale/expanding."
            )

    return LayerContext(
        timeframe=tf_name, candles=candles, swings=swings, bias=bias,
        structure_events=events, range_ctx=range_ctx, levels=levels, sweeps=sweeps,
        vol_state=vol_state, parent_timeframe=parent.timeframe if parent else None,
        parent_relation=parent_relation,
    )


def build_all_layers(base_1min: List[Candle], as_of: datetime) -> Dict[str, LayerContext]:
    """Orchestrates resampling + per-layer analysis, coarse-to-fine parent
    linking (4h's parent is 1D, 1h's parent is 4h, etc.), so each layer
    context carries a read on how it nests inside the layer above it."""
    base_1min = sorted(base_1min, key=lambda c: c.t)
    daily = resample(base_1min, "1D", 1440, "calendar_day")
    weekly = resample_weekly(daily)
    monthly = resample_monthly(daily)
    as_of_date = as_of.date()

    layers: Dict[str, LayerContext] = {}
    ordered_names = [name for name, _, _ in TIMEFRAME_SPECS]
    parent_ctx = None
    # Build coarse -> fine so each layer can reference its already-built parent.
    for name, minutes, kind in reversed(TIMEFRAME_SPECS):
        candles = resample(base_1min, name, minutes, kind)
        candles = [c for c in candles if c.t <= as_of]
        if not candles:
            continue
        ctx = build_layer_context(name, candles, daily, weekly, monthly, as_of_date, parent=parent_ctx)
        layers[name] = ctx
        parent_ctx = ctx

    return {name: layers[name] for name in ordered_names if name in layers}


# =========================================================================
# 8. LEVEL LEDGER — cross-timeframe confluence merge
# =========================================================================

@dataclass
class LiquidityZone:
    price_low: float
    price_high: float
    mid: float
    contributors: List[LiquidityLevel]
    confluence_score: float
    timeframes: List[str]

    def touches(self, price: float) -> bool:
        return self.price_low <= price <= self.price_high


def build_level_ledger(layers: Dict[str, LayerContext], tolerance_pct: float = 0.05
                        ) -> List[LiquidityZone]:
    """This is the coherence guarantee the whole engine is built around: we
    take every liquidity level from every timeframe, sort by price, and
    merge anything within `tolerance_pct`% of price into one canonical
    zone. A zone's confluence_score rewards agreement from MORE DISTINCT
    timeframes (not just more levels on one timeframe), so a 5m equal-high
    that overlaps a 1h swing high and a prior-day high scores far higher
    than any one of those alone. Because pathway generation (section 10)
    only ever targets zones from this ledger, a pathway can never claim a
    price on one timeframe that contradicts another timeframe's read of
    that same price — the level is the same object everywhere it appears.
    """
    all_levels: List[LiquidityLevel] = []
    for ctx in layers.values():
        all_levels.extend(ctx.levels)
    if not all_levels:
        return []

    ref_price = statistics.median(l.price for l in all_levels)
    tolerance = max(ref_price * (tolerance_pct / 100.0), 1e-6)

    all_levels.sort(key=lambda l: l.price)
    zones: List[LiquidityZone] = []
    cluster: List[LiquidityLevel] = [all_levels[0]]

    def flush(cluster: List[LiquidityLevel]):
        prices = [l.price for l in cluster]
        distinct_tfs = set(l.timeframe for l in cluster)
        base_strength = sum(l.strength for l in cluster)
        # diminishing returns per extra level on the SAME timeframe, but a
        # full multiplicative bump for every additional DISTINCT timeframe —
        # this is what makes multi-timeframe agreement worth more than
        # repetition on one timeframe.
        confluence_multiplier = 1.0 + 0.5 * (len(distinct_tfs) - 1)
        score = base_strength * confluence_multiplier
        zones.append(LiquidityZone(
            price_low=min(prices), price_high=max(prices),
            mid=sum(prices) / len(prices), contributors=list(cluster),
            confluence_score=round(score, 3), timeframes=sorted(distinct_tfs),
        ))

    for lvl in all_levels[1:]:
        # Bound cluster width against its START price, not the last-added
        # price - comparing against cluster[-1] lets dense levels "chain"
        # transitively across an unbounded span (classic single-linkage
        # degeneracy). Bounding against cluster[0] keeps every zone's total
        # width <= tolerance.
        if lvl.price - cluster[0].price <= tolerance:
            cluster.append(lvl)
        else:
            flush(cluster)
            cluster = [lvl]
    flush(cluster)

    zones.sort(key=lambda z: z.confluence_score, reverse=True)
    return zones


def compute_layer_alignment(price: float, layers: Dict[str, LayerContext]) -> Dict[str, dict]:
    """For a candidate pathway target price, read out where that price
    would sit on EVERY layer's own range (premium/discount %) and whether
    that reading is even reachable (inside that layer's plausible extended
    range). This is the explicit implementation of the requirement that
    'if the lower timeframe reaches X, the 4h view of that same X must
    read coherently too' — every leg target gets checked against every
    layer before the pathway is accepted."""
    out = {}
    for tf, ctx in layers.items():
        rc = ctx.range_ctx
        span = rc.high - rc.low
        extended_lo = rc.low - span * 0.5
        extended_hi = rc.high + span * 0.5
        pct = 50.0 if span == 0 else (price - rc.low) / span * 100.0
        out[tf] = {
            "position_pct": round(pct, 1),
            "reachable": extended_lo <= price <= extended_hi,
            "layer_bias": ctx.bias.value,
        }
    return out


# =========================================================================
# 9. LIQUIDITY ENGINEERING / SESSION BIAS
# =========================================================================

@dataclass
class SessionBias:
    score: float           # -1 (strong bearish/distribution bias) .. +1 (strong bullish/accumulation bias)
    label: str
    explanation: str


def compute_liquidity_engineering_bias(layers: Dict[str, LayerContext]) -> SessionBias:
    """Reads the most recent session's sweep events across all layers and
    scores how much of the day's liquidity-taking looks like engineered
    accumulation/distribution rather than a clean trend day. The intuition
    this encodes (directly from your brief): if sell-side liquidity was
    swept hard and then strongly reclaimed (a GRAB), that reads as smart
    money accumulating into weak hands' stops -> bullish bias for the next
    session (and vice-versa for buy-side grabs -> bearish/distribution
    bias). Genuine BREAKOUT sweeps (no reclaim) are treated as trend
    confirmation, not engineering, and contribute far less to this score.
    """
    weighted_bull = 0.0
    weighted_bear = 0.0
    notes = []

    for tf, ctx in layers.items():
        # weight sweeps on coarser timeframes more heavily - a 1D/4h grab
        # matters more to next-session bias than a 1-minute wick.
        tf_weight = {"1min": 0.3, "5min": 0.5, "15min": 0.7, "1h": 1.0, "4h": 1.4, "1D": 1.8}.get(tf, 1.0)
        for sw in ctx.sweeps[-8:]:  # most recent sweeps per layer
            if sw.classification != SweepClassification.GRAB:
                continue
            magnitude = sw.reversal_strength * math.log1p(max(sw.est_volume, 0)) * tf_weight
            if sw.side == "sell_side":
                weighted_bull += magnitude
                notes.append(f"{tf}: sell-side liquidity swept at {sw.level.price:.2f} "
                             f"({sw.level.type.value}) and reclaimed (strength "
                             f"{sw.reversal_strength:.2f}) -> bullish engineering signature")
            else:
                weighted_bear += magnitude
                notes.append(f"{tf}: buy-side liquidity swept at {sw.level.price:.2f} "
                             f"({sw.level.type.value}) and reclaimed (strength "
                             f"{sw.reversal_strength:.2f}) -> bearish/distribution signature")

    total = weighted_bull + weighted_bear
    score = 0.0 if total == 0 else (weighted_bull - weighted_bear) / total
    score = max(-1.0, min(1.0, score))

    if score > 0.25:
        label = "accumulation (gap-up leaning)"
    elif score < -0.25:
        label = "distribution (gap-down leaning)"
    else:
        label = "balanced / no clear engineering signature"

    explanation = (
        f"Weighted bullish engineering: {weighted_bull:.2f}; bearish: {weighted_bear:.2f}. "
        + (" | ".join(notes[-6:]) if notes else "No qualifying liquidity grabs found in the "
                                                 "recent window across any layer.")
    )
    return SessionBias(round(score, 3), label, explanation)


# =========================================================================
# 10. SENTIMENT HOOK (pluggable — not a live feed)
# =========================================================================

def default_sentiment_provider() -> float:
    """Returns 0.0 (neutral) by default. This is an explicit extension
    point: wire in a real news/sentiment source by passing a callable of
    the same signature (`() -> float in [-1, 1]`) to `run()`'s
    `sentiment_provider` argument, or pass --sentiment on the CLI to hard-
    set a value for one run. We do NOT fabricate a news score here."""
    return 0.0


# =========================================================================
# 10b. OPTIONS OI + MARKET-DEPTH FLOW (Kite-powered, or manually injected)
# =========================================================================

@dataclass
class FlowContext:
    """Derivatives/order-book flow read, condensed to one score in [-1,1].

    pcr               put-call OI ratio near the money. Indian-market
                      convention: heavy put OI = put WRITERS defending
                      levels below price (supportive/bullish); heavy call
                      OI = call writers capping upside (bearish). We map
                      pcr -> tanh((pcr - 1.0) * 2), so pcr 1.0 is neutral,
                      ~1.5 strongly bullish, ~0.6 strongly bearish.
    depth_imbalance   (total bid qty - total ask qty) / (bid + ask) from
                      the order book of the instrument's nearest future
                      (indices have no cash order book). Positive = more
                      resting demand visible.
    score             0.7 * pcr_bias + 0.3 * depth_imbalance, clamped.
    source            "kite" | "manual" | "none" — always shown in the
                      report so you know whether flow was real data.
    """
    pcr: Optional[float]
    depth_imbalance: float
    score: float
    source: str
    notes: str = ""


def neutral_flow() -> FlowContext:
    return FlowContext(pcr=None, depth_imbalance=0.0, score=0.0, source="none",
                        notes="no flow data supplied — flow term contributes nothing")


def _flow_score(pcr: Optional[float], depth_imbalance: float) -> float:
    pcr_bias = math.tanh((pcr - 1.0) * 2.0) if pcr is not None else 0.0
    raw = 0.7 * pcr_bias + 0.3 * max(-1.0, min(1.0, depth_imbalance))
    return round(max(-1.0, min(1.0, raw)), 3)


def flow_from_json(payload: str) -> FlowContext:
    """Manual injection for offline runs / other data sources:
    --flow-json '{"pcr": 1.32, "depth_imbalance": 0.15}'"""
    d = json.loads(payload)
    pcr = d.get("pcr")
    depth = float(d.get("depth_imbalance", 0.0))
    return FlowContext(pcr=pcr, depth_imbalance=depth,
                        score=_flow_score(pcr, depth), source="manual",
                        notes="values injected via --flow-json")


def fetch_kite_flow(symbol: str, exchange: str = "NSE",
                     option_name: Optional[str] = None,
                     strikes_pct_window: float = 0.03) -> FlowContext:
    """Pull near-the-money option OI and futures order-book depth from Kite
    and condense them into a FlowContext. Degrades gracefully: any missing
    piece (no NFO derivatives for the symbol, quote errors, no depth) just
    drops out of the score rather than failing the run.

    option_name: the NFO 'name' field (e.g. "NIFTY" for the NIFTY 50 index,
    "HDFCBANK" for the stock). Defaults to symbol with spaces/index suffixes
    stripped, which handles "NIFTY 50" -> "NIFTY".
    """
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        return FlowContext(None, 0.0, 0.0, "none",
                            "kiteconnect not installed — flow skipped")
    api_key = os.environ.get("KITE_API_KEY")
    token = os.environ.get("KITE_ACCESS_TOKEN")
    if not api_key or not token:
        return FlowContext(None, 0.0, 0.0, "none",
                            "KITE_API_KEY / KITE_ACCESS_TOKEN not set — flow skipped")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(token)

    name = option_name or symbol.replace(" 50", "").replace(" ", "")
    notes = []
    pcr = None
    depth_imbalance = 0.0

    try:
        spot = kite.ltp([f"{exchange}:{symbol}"])
        spot_price = list(spot.values())[0]["last_price"]
    except Exception as e:
        return FlowContext(None, 0.0, 0.0, "none", f"spot LTP failed: {e}")

    try:
        nfo = kite.instruments("NFO")
        opts = [i for i in nfo if i.get("name") == name and i.get("segment") == "NFO-OPT"]
        futs = [i for i in nfo if i.get("name") == name and i.get("segment") == "NFO-FUT"]

        if opts:
            nearest_expiry = min(o["expiry"] for o in opts)
            lo = spot_price * (1 - strikes_pct_window)
            hi = spot_price * (1 + strikes_pct_window)
            near = [o for o in opts
                    if o["expiry"] == nearest_expiry and lo <= o["strike"] <= hi]
            # kite.quote caps at ~500 instruments per call; NTM window is far below that
            keys = [f"NFO:{o['tradingsymbol']}" for o in near][:400]
            if keys:
                quotes = kite.quote(keys)
                put_oi = sum(q.get("oi", 0) for k, q in quotes.items() if k.endswith("PE"))
                call_oi = sum(q.get("oi", 0) for k, q in quotes.items() if k.endswith("CE"))
                if call_oi > 0:
                    pcr = round(put_oi / call_oi, 3)
                    notes.append(f"PCR {pcr} from {len(keys)} NTM contracts "
                                 f"(expiry {nearest_expiry}, ±{strikes_pct_window*100:.0f}% strikes)")
        if futs:
            fut = min(futs, key=lambda f: f["expiry"])
            fq = kite.quote([f"NFO:{fut['tradingsymbol']}"])
            depth = list(fq.values())[0].get("depth", {})
            bid_qty = sum(x.get("quantity", 0) for x in depth.get("buy", []))
            ask_qty = sum(x.get("quantity", 0) for x in depth.get("sell", []))
            if bid_qty + ask_qty > 0:
                depth_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
                notes.append(f"depth imbalance {depth_imbalance:+.2f} from {fut['tradingsymbol']}")
    except Exception as e:
        notes.append(f"derivatives fetch partial failure: {e}")

    return FlowContext(pcr=pcr, depth_imbalance=round(depth_imbalance, 3),
                        score=_flow_score(pcr, depth_imbalance), source="kite",
                        notes="; ".join(notes) or "no derivatives found for this name")


# =========================================================================
# 11. GAP BUCKET ENGINE
# =========================================================================

@dataclass
class GapBucket:
    points: float          # signed: +N = gap up, -N = gap down
    probability: float
    rationale: str


def compute_atr_daily(layers: Dict[str, LayerContext], n: int = 14) -> float:
    if "1D" in layers:
        return CandleSeries(layers["1D"].candles).atr(n)
    # fall back to the coarsest available layer, scaled up crudely
    coarsest = list(layers.values())[-1]
    return CandleSeries(coarsest.candles).atr(n)


def build_gap_buckets(atr: float, custom_points: Optional[List[float]] = None) -> List[float]:
    """Default buckets are ATR-derived fractions, rounded to a clean step
    (25 units) so they read like the round-number gaps traders actually
    talk about (+50/+100/+200/+250/+300-style). Pass --gap-points to
    override with your own explicit list (e.g. instrument-specific)."""
    if custom_points:
        pts = sorted(set(custom_points) | set(-p for p in custom_points))
        return [p for p in pts if p != 0]

    fractions = [0.15, 0.35, 0.6, 1.0, 1.5]
    step = 25.0
    raw = [round(atr * f / step) * step for f in fractions]
    raw = sorted(set(p for p in raw if p > 0))
    return [-p for p in reversed(raw)] + raw


def score_gap_probabilities(buckets: List[float], bias: SessionBias, sentiment: float,
                             atr: float, flow: Optional[FlowContext] = None) -> List[GapBucket]:
    """Heuristic (not statistically fitted) scoring: a bucket's raw score
    combines (a) alignment between the bucket's direction and the
    liquidity-engineering bias, (b) alignment with the sentiment input,
    (c) alignment with the derivatives-flow score (options OI put-call
    ratio + futures depth imbalance, when supplied), and (d) a magnitude
    penalty (bigger gaps are inherently less probable, all else equal).
    Scores are then softmax-normalized into a probability distribution
    over the buckets. This is transparent by design — replace
    `score_gap_probabilities` with a fitted model later without touching
    anything downstream, since callers only consume `GapBucket.probability`.
    """
    flow_score = flow.score if flow is not None else 0.0
    raw_scores = []
    for pts in buckets:
        direction = 1.0 if pts > 0 else -1.0
        bias_align = direction * bias.score
        sentiment_align = direction * sentiment
        flow_align = direction * flow_score
        magnitude_penalty = -abs(pts) / (atr * 2.0 + 1e-9)
        raw = (1.4 * bias_align + 0.8 * sentiment_align + 0.9 * flow_align
               + 0.6 * magnitude_penalty)
        raw_scores.append(raw)

    max_raw = max(raw_scores)
    exp_scores = [math.exp(r - max_raw) for r in raw_scores]
    total = sum(exp_scores)
    probs = [e / total for e in exp_scores]

    out = []
    for pts, p in zip(buckets, probs):
        direction_word = "gap up" if pts > 0 else "gap down"
        rationale = (
            f"{direction_word} of {abs(pts):.0f} pts: bias={bias.label} (score {bias.score:+.2f}), "
            f"sentiment={sentiment:+.2f}, flow={flow_score:+.2f}, magnitude={abs(pts)/atr:.2f}x ATR"
        )
        out.append(GapBucket(pts, round(p, 4), rationale))
    out.sort(key=lambda g: g.probability, reverse=True)
    return out


# =========================================================================
# 12. PATHWAY / SCENARIO GRAMMAR
# =========================================================================

class LegType(Enum):
    EXPANSION = "expansion"
    RETRACEMENT = "retracement"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    CONSOLIDATION = "consolidation"
    DISTRIBUTION = "distribution"
    REVERSAL = "reversal"


# Named narrative templates: each is a sequence of (LegType, direction_sign)
# where direction_sign is relative to the PRIMARY (gap) direction: +1 = with
# the gap direction, -1 = against it. These are hand-authored structural
# narratives grounded in common ICT/SMC session archetypes, not randomly
# generated, so every pathway tells a coherent story rather than a random
# walk of labels.
_TEMPLATES: List[Tuple[str, List[Tuple[LegType, int]]]] = [
    ("sweep_and_go", [
        (LegType.LIQUIDITY_SWEEP, -1), (LegType.REVERSAL, +1),
        (LegType.EXPANSION, +1), (LegType.CONSOLIDATION, 0), (LegType.EXPANSION, +1),
    ]),
    ("trend_day", [
        (LegType.CONSOLIDATION, 0), (LegType.EXPANSION, +1),
        (LegType.RETRACEMENT, -1), (LegType.EXPANSION, +1), (LegType.CONSOLIDATION, 0),
    ]),
    ("fade_and_reverse", [
        (LegType.EXPANSION, +1), (LegType.LIQUIDITY_SWEEP, +1),
        (LegType.REVERSAL, -1), (LegType.EXPANSION, -1), (LegType.DISTRIBUTION, -1),
    ]),
    ("double_sweep_range", [
        (LegType.LIQUIDITY_SWEEP, +1), (LegType.REVERSAL, -1),
        (LegType.LIQUIDITY_SWEEP, -1), (LegType.REVERSAL, +1), (LegType.CONSOLIDATION, 0),
    ]),
    ("choppy_consolidation", [
        (LegType.CONSOLIDATION, 0), (LegType.RETRACEMENT, -1),
        (LegType.CONSOLIDATION, 0), (LegType.RETRACEMENT, +1), (LegType.CONSOLIDATION, 0),
    ]),
    ("distribution_day", [
        (LegType.EXPANSION, +1), (LegType.CONSOLIDATION, 0),
        (LegType.DISTRIBUTION, +1), (LegType.REVERSAL, -1), (LegType.EXPANSION, -1),
    ]),
    ("clean_expansion", [
        (LegType.EXPANSION, +1), (LegType.RETRACEMENT, -1),
        (LegType.EXPANSION, +1), (LegType.RETRACEMENT, -1), (LegType.EXPANSION, +1),
    ]),
    ("liquidity_hunt_reversal", [
        (LegType.EXPANSION, +1), (LegType.LIQUIDITY_SWEEP, +1),
        (LegType.REVERSAL, -1), (LegType.EXPANSION, -1), (LegType.RETRACEMENT, +1),
    ]),
]


# --- Session-clock model ---------------------------------------------------
# Legs don't take equal time in a real session: a stop-run sweep is minutes,
# a midday consolidation eats hours. Each leg type gets a duration weight;
# a template's anchor times are the normalized cumulative durations, so the
# x-axis of every pathway is actual session time (09:15 -> 15:30 IST), not
# an abstract "leg index". On top of that, each leg type has a time-of-day
# affinity (sweeps cluster at the open and the close, consolidation lives
# in the midday lull, expansions favor the open drive and the closing hour)
# which feeds the plausibility score: a template whose consolidation lands
# at 12:30 is more believable than one that consolidates into the close.

SESSION_MINUTES = (SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]) - (SESSION_OPEN[0] * 60 + SESSION_OPEN[1])

_LEG_DURATION_WEIGHT = {
    LegType.EXPANSION: 0.18,
    LegType.RETRACEMENT: 0.12,
    LegType.LIQUIDITY_SWEEP: 0.08,
    LegType.CONSOLIDATION: 0.24,
    LegType.DISTRIBUTION: 0.15,
    LegType.REVERSAL: 0.10,
}


def session_clock(frac: float) -> str:
    """Map a session fraction (0..1) to an IST clock string, 09:15..15:30."""
    minutes = SESSION_OPEN[0] * 60 + SESSION_OPEN[1] + int(round(frac * SESSION_MINUTES))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def compute_time_anchors(template: List[Tuple[LegType, int]]) -> List[float]:
    """Cumulative normalized durations: len(template)+1 fractions from 0.0
    to 1.0, one anchor per leg boundary."""
    durs = [_LEG_DURATION_WEIGHT[kind] for kind, _ in template]
    total = sum(durs) or 1.0
    anchors = [0.0]
    acc = 0.0
    for d in durs:
        acc += d / total
        anchors.append(min(1.0, acc))
    anchors[-1] = 1.0
    return anchors


def _time_affinity_bonus(kind: LegType, mid_frac: float) -> float:
    """Small plausibility adjustment for how well a leg's session timing
    matches where that behavior empirically clusters in an NSE cash day."""
    if kind == LegType.LIQUIDITY_SWEEP:
        return 0.20 if (mid_frac < 0.25 or mid_frac > 0.75) else -0.10
    if kind == LegType.CONSOLIDATION:
        return 0.15 if 0.35 <= mid_frac <= 0.65 else -0.05
    if kind == LegType.EXPANSION:
        return 0.10 if (mid_frac < 0.35 or mid_frac > 0.70) else 0.0
    if kind == LegType.DISTRIBUTION:
        return 0.10 if mid_frac > 0.55 else -0.05
    return 0.0


@dataclass
class Leg:
    kind: LegType
    direction: int          # +1 up, -1 down, 0 flat/ranging
    start_price: float
    end_price: float
    target_zone: Optional[LiquidityZone]
    layer_alignment: Dict[str, dict]
    note: str
    t_start: float = 0.0    # session fraction at leg start (0 = 09:15, 1 = 15:30)
    t_end: float = 1.0


@dataclass
class Pathway:
    template_name: str
    gap: GapBucket
    legs: List[Leg]
    points: List[Tuple[float, float]]   # (time_fraction 0..1, price) anchor points
    plausibility_score: float
    narrative: str

    def to_dict(self) -> dict:
        return {
            "template": self.template_name,
            "gap_points": self.gap.points,
            "gap_probability": self.gap.probability,
            "plausibility_score": round(self.plausibility_score, 3),
            "narrative": self.narrative,
            "legs": [
                {
                    "kind": leg.kind.value, "direction": leg.direction,
                    "session_window": f"{session_clock(leg.t_start)}-{session_clock(leg.t_end)}",
                    "start_price": round(leg.start_price, 2), "end_price": round(leg.end_price, 2),
                    "target_confluence": (
                        {"score": leg.target_zone.confluence_score,
                         "timeframes": leg.target_zone.timeframes,
                         "price_range": [round(leg.target_zone.price_low, 2),
                                         round(leg.target_zone.price_high, 2)]}
                        if leg.target_zone else "projected (no cross-timeframe confluence)"
                    ),
                    "note": leg.note,
                }
                for leg in self.legs
            ],
        }


def _bias_alignment_bonus(direction: int, higher_tf_bias: StructureBias) -> float:
    if direction == 0:
        return 0.0
    if higher_tf_bias == StructureBias.BULLISH:
        return 0.4 if direction > 0 else -0.3
    if higher_tf_bias == StructureBias.BEARISH:
        return 0.4 if direction < 0 else -0.3
    return 0.0


def _pick_leg_target(current_price: float, direction: int, atr: float,
                      ledger: List[LiquidityZone], min_reach: float, max_reach: float
                      ) -> Tuple[float, Optional[LiquidityZone], str]:
    """Chooses the next anchor price for a leg. Prefers a ledger zone in
    the requested direction within [min_reach, max_reach] of current price
    (weighted toward higher-confluence zones), and falls back to a
    Fibonacci-style projection (explicitly labeled unconfluenced) if no
    zone qualifies."""
    if direction == 0:
        wob = atr * 0.15
        return current_price + random.uniform(-wob, wob), None, "ranging inside current value area"

    candidates = []
    for z in ledger:
        dist = (z.mid - current_price) * direction
        if min_reach <= dist <= max_reach:
            candidates.append(z)
    if candidates:
        candidates.sort(key=lambda z: z.confluence_score, reverse=True)
        top = candidates[: max(1, len(candidates) // 2 + 1)]
        chosen = random.choice(top)
        note = (f"targets a {len(chosen.timeframes)}-timeframe confluence zone "
                f"({', '.join(chosen.timeframes)}) around {chosen.mid:.2f}")
        return chosen.mid, chosen, note

    fib = random.choice([0.5, 0.618, 0.786, 1.0])
    projected = current_price + direction * (min_reach + fib * (max_reach - min_reach))
    return projected, None, f"no cross-timeframe confluence nearby — projected via {fib} extension of ATR range"


def generate_pathway(template_name: str, template: List[Tuple[LegType, int]],
                      gap: GapBucket, ledger: List[LiquidityZone],
                      layers: Dict[str, LayerContext], bias: SessionBias,
                      gap_open_price: float, prior_close: float) -> Pathway:
    coarsest_tf = list(layers.keys())[-1]
    higher_bias = layers[coarsest_tf].bias
    atr = compute_atr_daily(layers)

    legs: List[Leg] = []
    price = gap_open_price
    gap_direction = 1 if gap.points >= 0 else -1
    score = gap.probability * 3.0  # base score seeded from the gap's own probability
    anchors = compute_time_anchors(template)

    for li, (kind, rel_dir) in enumerate(template):
        direction = gap_direction * rel_dir if rel_dir != 0 else 0
        if kind == LegType.CONSOLIDATION:
            min_reach, max_reach = 0.0, atr * 0.2
        elif kind == LegType.RETRACEMENT:
            min_reach, max_reach = atr * 0.15, atr * 0.5
        elif kind == LegType.LIQUIDITY_SWEEP:
            min_reach, max_reach = atr * 0.1, atr * 0.4
        elif kind == LegType.DISTRIBUTION:
            min_reach, max_reach = atr * 0.05, atr * 0.3
        else:  # EXPANSION, REVERSAL
            min_reach, max_reach = atr * 0.3, atr * 1.1

        end_price, zone, note = _pick_leg_target(price, direction if direction != 0 else 1,
                                                   atr, ledger, min_reach, max_reach)
        if direction == 0:
            end_price, zone, note = _pick_leg_target(price, 0, atr, ledger, min_reach, max_reach)

        alignment = compute_layer_alignment(end_price, layers)
        unreachable_layers = [tf for tf, a in alignment.items() if not a["reachable"]]
        if unreachable_layers:
            score -= 0.5 * len(unreachable_layers)
            note += f" [WARNING: outside plausible range on {', '.join(unreachable_layers)}]"

        score += _bias_alignment_bonus(direction, higher_bias)
        if zone is not None:
            score += 0.15 * math.log1p(zone.confluence_score)

        t0, t1 = anchors[li], anchors[li + 1]
        score += _time_affinity_bonus(kind, (t0 + t1) / 2)

        legs.append(Leg(kind, direction, price, end_price, zone, alignment,
                         f"{kind.value}: {note}", t_start=t0, t_end=t1))
        price = end_price

    # net-displacement sanity: whole day shouldn't wildly exceed a plausible
    # multiple of ATR, or plausibility drops sharply.
    total_move = abs(price - gap_open_price)
    if total_move > atr * 3.0:
        score -= (total_move / atr - 3.0) * 0.5

    prices = [gap_open_price] + [leg.end_price for leg in legs]
    points = list(zip(anchors, prices))

    narrative_parts = [
        f"Gap {'up' if gap.points >= 0 else 'down'} {abs(gap.points):.0f} pts to open near "
        f"{gap_open_price:.2f} at {session_clock(0.0)} (prior close {prior_close:.2f})."
    ]
    for leg in legs:
        dir_word = {1: "up", -1: "down", 0: "sideways"}[leg.direction]
        narrative_parts.append(
            f"[{session_clock(leg.t_start)}-{session_clock(leg.t_end)}] "
            f"{leg.kind.value.replace('_', ' ').title()} {dir_word} toward "
            f"{leg.end_price:.2f} — {leg.note}"
        )
    narrative = " ".join(narrative_parts)

    return Pathway(template_name, gap, legs, points, score, narrative)


def generate_candidates(gap: GapBucket, ledger: List[LiquidityZone],
                         layers: Dict[str, LayerContext], bias: SessionBias,
                         prior_close: float, n_per_template: int = 2,
                         top_k: int = 3, seed: Optional[int] = None,
                         template_priors: Optional[Dict[str, float]] = None) -> List[Pathway]:
    if seed is not None:
        random.seed(seed)

    gap_open_price = prior_close + gap.points
    all_candidates: List[Pathway] = []

    # template_priors is the learning loop's feedback channel: replayed
    # sessions journal how well each template matched reality, and those
    # per-template weights (neutral = 1.0) tilt ranking here. Applied
    # additively so a negative raw score isn't perversely amplified.
    priors = template_priors or {}
    for name, template in _TEMPLATES:
        for _ in range(n_per_template):
            pw = generate_pathway(name, template, gap, ledger, layers, bias,
                                   gap_open_price, prior_close)
            pw.plausibility_score += (priors.get(name, 1.0) - 1.0) * 1.5
            if validate_pathway_coherence(pw, layers):
                all_candidates.append(pw)

    all_candidates.sort(key=lambda p: p.plausibility_score, reverse=True)
    return all_candidates[:top_k]


def validate_pathway_coherence(pathway: Pathway, layers: Dict[str, LayerContext]) -> bool:
    """Final safety-net check (the construction in generate_pathway already
    biases heavily toward coherent paths, but this is a hard gate): every
    leg's target must be reachable on at least a majority of layers, and
    price direction must actually move the stated way."""
    for leg in pathway.legs:
        reachable_count = sum(1 for a in leg.layer_alignment.values() if a["reachable"])
        if reachable_count < max(1, len(leg.layer_alignment) // 2):
            return False
        if leg.direction > 0 and leg.end_price < leg.start_price - 1e-6:
            return False
        if leg.direction < 0 and leg.end_price > leg.start_price + 1e-6:
            return False
    return True


# =========================================================================
# 13. SMOOTH CURVE MATH — pure-python Catmull-Rom -> sampled points
# =========================================================================

def _catmull_rom_point(p0, p1, p2, p3, t: float) -> Tuple[float, float]:
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
               + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
               + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
    y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
               + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
               + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
    return x, y


def sample_smooth_curve(points: List[Tuple[float, float]], samples_per_seg: int = 24
                         ) -> List[Tuple[float, float]]:
    """Catmull-Rom spline through the anchor points, sampled densely — a
    dependency-free stand-in for scipy's PCHIP/cubic interpolation. Pads
    the ends by reflecting the first/last segment so the curve doesn't
    need special-cased boundary handling."""
    if len(points) < 2:
        return points
    if len(points) == 2:
        (x0, y0), (x1, y1) = points
        return [(x0 + (x1 - x0) * i / samples_per_seg, y0 + (y1 - y0) * i / samples_per_seg)
                for i in range(samples_per_seg + 1)]

    padded = [points[0]] + points + [points[-1]]
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for s in range(samples_per_seg):
            t = s / samples_per_seg
            out.append(_catmull_rom_point(p0, p1, p2, p3, t))
    out.append(points[-1])
    return out


# =========================================================================
# 13b. PATHWAY vs REALITY — match scoring, journal, learned template priors
# =========================================================================
# This is the end-of-day feedback loop: after a session completes, we can
# ask "which of the pathways generated BEFORE the open actually traced
# closest to what the market did?", write the answer to a journal, and let
# accumulated journal history tilt future template ranking. Probabilities
# stay heuristic, but template preference becomes *earned* over time
# instead of authored.

def _interp_path(points: List[Tuple[float, float]], x: float) -> float:
    """Piecewise-linear interpolation over (t, price) points sorted by t."""
    if not points:
        return 0.0
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for i in range(1, len(points)):
        if points[i][0] >= x:
            (x0, y0), (x1, y1) = points[i - 1], points[i]
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def actual_day_path(day_candles: List[Candle]) -> List[Tuple[float, float]]:
    """The realized session as (session_fraction, close) points — the same
    coordinate system pathways live in, so the two are directly comparable."""
    if not day_candles:
        return []
    open_minutes = SESSION_OPEN[0] * 60 + SESSION_OPEN[1]
    out = []
    for c in day_candles:
        frac = ((c.t.hour * 60 + c.t.minute) - open_minutes) / SESSION_MINUTES
        out.append((max(0.0, min(1.0, frac)), c.c))
    return out


def score_curve_match(pathway: Pathway, actual: List[Tuple[float, float]],
                       atr: float, upto_frac: float = 1.0, n_samples: int = 64) -> float:
    """Similarity in (0, 1]: sample both curves on a shared time grid over
    [0, upto_frac], normalize prices by ATR (so the score is scale-free and
    comparable across instruments/days), and squash the RMSE through
    1/(1+rmse). 1.0 = pathway traced reality exactly; ~0.5 = off by one
    full ATR on average. `upto_frac < 1` scores only the elapsed part of
    the session — that's what live tracking uses."""
    if not actual or atr <= 0:
        return 0.0
    smooth = sample_smooth_curve(pathway.points, samples_per_seg=16)
    errs = []
    for i in range(n_samples):
        x = upto_frac * i / (n_samples - 1)
        p_path = _interp_path(smooth, x)
        p_real = _interp_path(actual, x)
        errs.append(((p_path - p_real) / atr) ** 2)
    rmse = math.sqrt(sum(errs) / len(errs))
    return round(1.0 / (1.0 + rmse), 4)


def journal_append(path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def load_journal(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_template_priors(path: str) -> Dict[str, float]:
    """Convert journal history into per-template weights centered on 1.0.
    Each replayed day records every template's best match score (0..1);
    a template's prior is 0.7 + 0.6 * (its average score), i.e. bounded in
    [0.7, 1.3] so no template is ever fully silenced or made dominant —
    the journal *tilts* ranking, it doesn't take it over. Templates with
    no history stay at exactly 1.0 (neutral)."""
    entries = load_journal(path)
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for e in entries:
        for name, s in (e.get("match_by_template") or {}).items():
            sums[name] = sums.get(name, 0.0) + float(s)
            counts[name] = counts.get(name, 0) + 1
    return {name: round(0.7 + 0.6 * (sums[name] / counts[name]), 4)
            for name in sums if counts[name] > 0}


# =========================================================================
# 14. RENDERING — self-contained SVG (no dependencies); optional matplotlib
# =========================================================================

_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706", "#0891b2"]


def render_pathways_svg(gap_scenarios: Dict[float, List[Pathway]],
                         ledger: List[LiquidityZone], prior_close: float,
                         out_path: str, width: int = 1200, height: int = 760,
                         actual_path: Optional[List[Tuple[float, float]]] = None,
                         title_suffix: str = "") -> None:
    """Renders every gap scenario's top pathways as smooth lines on one
    SVG chart, with liquidity-zone bands drawn behind them and the x-axis
    labeled in session time (09:15 -> 15:30 IST). If `actual_path` is
    given (track/replay modes), the realized tape is overlaid as a thick
    white line so you can see which projection the day is tracing.
    Self-contained (no JS, no external fonts/CDNs) — open in a browser."""
    all_prices = [prior_close]
    for pathways in gap_scenarios.values():
        for pw in pathways:
            all_prices.extend(p for _, p in pw.points)
    for z in ledger[:12]:
        all_prices.extend([z.price_low, z.price_high])
    if actual_path:
        all_prices.extend(p for _, p in actual_path)

    y_min, y_max = min(all_prices), max(all_prices)
    pad = (y_max - y_min) * 0.08 or 1.0
    y_min, y_max = y_min - pad, y_max + pad

    margin_l, margin_r, margin_t, margin_b = 90, 220, 50, 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def px(t: float) -> float:
        return margin_l + t * plot_w

    def py(price: float) -> float:
        return margin_t + (1 - (price - y_min) / (y_max - y_min)) * plot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
               f'font-family="Helvetica,Arial,sans-serif">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0b1220"/>')
    svg.append(f'<text x="{margin_l}" y="28" fill="#e5e7eb" font-size="18" font-weight="600">'
               f'Market Pathway Projections — gap scenarios from prior close {prior_close:.2f}'
               f'{" — " + title_suffix if title_suffix else ""}</text>')

    # gridlines + y-axis labels
    for i in range(6):
        gy = margin_t + plot_h * i / 5
        price = y_max - (y_max - y_min) * i / 5
        svg.append(f'<line x1="{margin_l}" y1="{gy:.1f}" x2="{width - margin_r}" y2="{gy:.1f}" '
                   f'stroke="#1f2937" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l - 10}" y="{gy + 4:.1f}" fill="#9ca3af" font-size="11" '
                   f'text-anchor="end">{price:.1f}</text>')

    # x-axis: session-time ticks (09:15 .. 15:30 IST)
    for i in range(7):
        frac = i / 6
        gx = px(frac)
        svg.append(f'<line x1="{gx:.1f}" y1="{margin_t}" x2="{gx:.1f}" y2="{margin_t + plot_h}" '
                   f'stroke="#1f2937" stroke-width="1"/>')
        svg.append(f'<text x="{gx:.1f}" y="{margin_t + plot_h + 18}" fill="#9ca3af" '
                   f'font-size="11" text-anchor="middle">{session_clock(frac)}</text>')

    # liquidity zone bands (top confluence zones only, to avoid clutter)
    for z in ledger[:10]:
        y0, y1 = py(z.price_high), py(z.price_low)
        opacity = min(0.35, 0.08 + 0.03 * len(z.timeframes))
        svg.append(f'<rect x="{margin_l}" y="{y0:.1f}" width="{plot_w}" height="{max(1, y1 - y0):.1f}" '
                   f'fill="#fbbf24" opacity="{opacity:.2f}"/>')
        svg.append(f'<text x="{width - margin_r + 6}" y="{(y0 + y1) / 2 + 3:.1f}" fill="#fbbf24" '
                   f'font-size="9">{z.mid:.0f} ({"+".join(z.timeframes)})</text>')

    # prior-close reference line
    py0 = py(prior_close)
    svg.append(f'<line x1="{margin_l}" y1="{py0:.1f}" x2="{width - margin_r}" y2="{py0:.1f}" '
               f'stroke="#e5e7eb" stroke-width="1" stroke-dasharray="4,4"/>')
    svg.append(f'<text x="{margin_l + 4}" y="{py0 - 4:.1f}" fill="#e5e7eb" font-size="10">prior close</text>')

    legend_y = margin_t
    color_i = 0
    for gap_points, pathways in sorted(gap_scenarios.items(), key=lambda kv: kv[0]):
        for pw in pathways:
            color = _PALETTE[color_i % len(_PALETTE)]
            color_i += 1
            curve = sample_smooth_curve(pw.points, samples_per_seg=24)
            path_d = " ".join(
                f'{"M" if i == 0 else "L"} {px(t):.1f} {py(p):.1f}'
                for i, (t, p) in enumerate(curve)
            )
            svg.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.2" '
                       f'stroke-linecap="round" opacity="0.92"/>')
            for t, p in pw.points:
                svg.append(f'<circle cx="{px(t):.1f}" cy="{py(p):.1f}" r="3" fill="{color}"/>')

            label = f'{"+" if gap_points >= 0 else ""}{gap_points:.0f} {pw.template_name} ' \
                    f'(p={pw.gap.probability:.2f}, score={pw.plausibility_score:.2f})'
            svg.append(f'<rect x="{width - margin_r + 4}" y="{legend_y - 10}" width="10" height="10" '
                       f'fill="{color}"/>')
            svg.append(f'<text x="{width - margin_r + 18}" y="{legend_y - 1}" fill="#e5e7eb" '
                       f'font-size="10">{label}</text>')
            legend_y += 16

    # realized-tape overlay (track / replay modes)
    if actual_path and len(actual_path) >= 2:
        path_d = " ".join(
            f'{"M" if i == 0 else "L"} {px(t):.1f} {py(p):.1f}'
            for i, (t, p) in enumerate(actual_path)
        )
        svg.append(f'<path d="{path_d}" fill="none" stroke="#f8fafc" stroke-width="3.2" '
                   f'stroke-linecap="round" opacity="0.95"/>')
        lt, lp = actual_path[-1]
        svg.append(f'<circle cx="{px(lt):.1f}" cy="{py(lp):.1f}" r="5" fill="#f8fafc"/>')
        svg.append(f'<text x="{px(lt) + 8:.1f}" y="{py(lp) - 8:.1f}" fill="#f8fafc" '
                   f'font-size="11" font-weight="600">actual {lp:.1f}</text>')

    svg.append(f'<text x="{margin_l}" y="{height - 12}" fill="#6b7280" font-size="10">'
               f'Illustrative structural pathways, not a candle-by-candle forecast. '
               f'Generated by market_pathway_engine.py</text>')
    svg.append("</svg>")

    with open(out_path, "w") as f:
        f.write("\n".join(svg))


def try_render_matplotlib(gap_scenarios: Dict[float, List[Pathway]],
                           ledger: List[LiquidityZone], prior_close: float,
                           out_path: str) -> bool:
    """Optional higher-fidelity PNG render if matplotlib happens to be
    installed. Returns False (and does nothing else) if it isn't — SVG
    rendering above is the guaranteed path and needs no third-party libs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(13, 8))
    for z in ledger[:10]:
        ax.axhspan(z.price_low, z.price_high, color="#f59e0b",
                   alpha=min(0.3, 0.08 + 0.03 * len(z.timeframes)))
    ax.axhline(prior_close, color="white", linestyle="--", linewidth=1, alpha=0.6)

    color_i = 0
    for gap_points, pathways in sorted(gap_scenarios.items(), key=lambda kv: kv[0]):
        for pw in pathways:
            curve = sample_smooth_curve(pw.points, samples_per_seg=24)
            xs = [t for t, _ in curve]
            ys = [p for _, p in curve]
            ax.plot(xs, ys, linewidth=2.2, color=_PALETTE[color_i % len(_PALETTE)],
                    label=f'{"+" if gap_points >= 0 else ""}{gap_points:.0f} {pw.template_name} '
                          f'(p={pw.gap.probability:.2f})')
            color_i += 1
    ax.set_title("Market Pathway Projections")
    ax.set_xlabel("session progress")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="#0b1220")
    plt.close(fig)
    return True


# =========================================================================
# 15. CONTEXT MAP / REPORT
# =========================================================================

def build_context_map(layers: Dict[str, LayerContext], ledger: List[LiquidityZone],
                       bias: SessionBias, gap_buckets: List[GapBucket],
                       scenarios: Dict[float, List[Pathway]], prior_close: float) -> dict:
    return {
        "generated_at": datetime.now(IST).isoformat(),
        "prior_close": prior_close,
        "layers": {
            tf: {
                "bias": ctx.bias.value,
                "range": {"high": ctx.range_ctx.high, "low": ctx.range_ctx.low,
                          "position_pct": ctx.range_ctx.position_pct, "state": ctx.range_ctx.state},
                "volatility_state": ctx.vol_state.value,
                "n_structure_events": len(ctx.structure_events),
                "last_structure_event": (
                    {"kind": ctx.structure_events[-1].kind, "direction": ctx.structure_events[-1].direction.value,
                     "price": ctx.structure_events[-1].price, "t": ctx.structure_events[-1].t.isoformat()}
                    if ctx.structure_events else None
                ),
                "n_liquidity_levels": len(ctx.levels),
                "n_sweeps": len(ctx.sweeps),
                "parent_relation": ctx.parent_relation,
            }
            for tf, ctx in layers.items()
        },
        "liquidity_engineering_bias": {
            "score": bias.score, "label": bias.label, "explanation": bias.explanation,
        },
        "level_ledger_top": [
            {"price_range": [z.price_low, z.price_high], "mid": z.mid,
             "confluence_score": z.confluence_score, "timeframes": z.timeframes}
            for z in ledger[:20]
        ],
        "gap_buckets": [
            {"points": g.points, "probability": g.probability, "rationale": g.rationale}
            for g in gap_buckets
        ],
        "pathways": {
            f"{pts:+.0f}": [pw.to_dict() for pw in pathways]
            for pts, pathways in scenarios.items()
        },
    }


# =========================================================================
# 16. ORCHESTRATION — shared analysis core + run / replay / track modes
# =========================================================================

@dataclass
class AnalysisResult:
    layers: Dict[str, LayerContext]
    ledger: List[LiquidityZone]
    bias: SessionBias
    gap_buckets: List[GapBucket]
    scenarios: Dict[float, List[Pathway]]
    prior_close: float
    atr: float
    flow: FlowContext


def _make_provider(mode: str, symbol: str, exchange: str,
                    instrument_token: Optional[int], csv_path: Optional[str],
                    seed: int) -> DataProvider:
    if mode == "live":
        return ZerodhaProvider(symbol, exchange, instrument_token)
    if mode == "csv":
        if not csv_path:
            raise ValueError("--csv PATH is required for --mode csv")
        return CSVProvider(csv_path)
    return SyntheticProvider(seed=seed)


def analyze_and_generate(base: List[Candle], as_of: datetime, *,
                          custom_gap_points: Optional[List[float]] = None,
                          sentiment: float = 0.0,
                          flow: Optional[FlowContext] = None,
                          template_priors: Optional[Dict[str, float]] = None,
                          top_k: int = 3, seed: int = 7,
                          verbose: bool = True) -> AnalysisResult:
    """The full context -> ledger -> bias -> gap -> pathways pipeline over
    a candle set, with NO file/IO side effects. run(), replay_run() and
    track_run() are all thin shells around this — which is exactly what
    makes replay honest: the replayed day is generated by the same code
    path, seeing only data that existed before that day's open."""
    flow = flow or neutral_flow()

    def say(msg: str):
        if verbose:
            print(msg)

    say("      building multi-timeframe layer contexts (1min..1D)...")
    layers = build_all_layers(base, as_of)
    if not layers:
        raise RuntimeError("No layers could be built from the supplied candles.")
    for tf, ctx in layers.items():
        say(f"      {tf:>5}: {len(ctx.candles):>5} candles | bias={ctx.bias.value:<8} | "
            f"range_state={ctx.range_ctx.state:<11} | vol={ctx.vol_state.value:<13} | "
            f"levels={len(ctx.levels):>3} | sweeps={len(ctx.sweeps)}")

    ledger = build_level_ledger(layers, tolerance_pct=0.05)
    say(f"      -> {len(ledger)} canonical liquidity zones" +
        (f" (top confluence: {ledger[0].confluence_score:.2f} across {ledger[0].timeframes})"
         if ledger else ""))

    bias = compute_liquidity_engineering_bias(layers)
    say(f"      -> session bias: {bias.label} (score {bias.score:+.2f})")
    if flow.source != "none":
        say(f"      -> flow [{flow.source}]: score {flow.score:+.2f} "
            f"(pcr={flow.pcr}, depth={flow.depth_imbalance:+.2f}) — {flow.notes}")

    atr = compute_atr_daily(layers)
    gap_points = build_gap_buckets(atr, custom_gap_points)
    gap_buckets = score_gap_probabilities(gap_points, bias, sentiment, atr, flow)
    prior_close = layers["1D"].candles[-1].c if "1D" in layers \
        else layers[list(layers.keys())[-1]].candles[-1].c

    scenarios: Dict[float, List[Pathway]] = {}
    for i, gap in enumerate(gap_buckets):
        candidates = generate_candidates(gap, ledger, layers, bias, prior_close,
                                          n_per_template=2, top_k=top_k, seed=seed + i,
                                          template_priors=template_priors)
        scenarios[gap.points] = candidates
        say(f"      gap {gap.points:+.0f} (p={gap.probability:.2f}): "
            f"{len(candidates)} pathway(s) kept")

    return AnalysisResult(layers, ledger, bias, gap_buckets, scenarios,
                           prior_close, atr, flow)


def _session_bounds(day: datetime) -> Tuple[datetime, datetime]:
    o = day.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1], second=0, microsecond=0)
    c = day.replace(hour=SESSION_CLOSE[0], minute=SESSION_CLOSE[1], second=0, microsecond=0)
    return o, c


def _resolve_flow(mode: str, symbol: str, exchange: str, use_flow: bool,
                   flow_json: Optional[str]) -> FlowContext:
    if flow_json:
        return flow_from_json(flow_json)
    if use_flow and mode == "live":
        return fetch_kite_flow(symbol, exchange)
    return neutral_flow()


def run(symbol: str, mode: str, as_of: datetime, lookback_days: int,
        csv_path: Optional[str] = None, exchange: str = "NSE",
        instrument_token: Optional[int] = None, seed: int = 7,
        custom_gap_points: Optional[List[float]] = None,
        sentiment_provider: Callable[[], float] = default_sentiment_provider,
        sentiment_override: Optional[float] = None,
        top_k: int = 3, out_dir: str = "market_pathway_output",
        use_flow: bool = False, flow_json: Optional[str] = None,
        journal_path: Optional[str] = None) -> dict:

    provider = _make_provider(mode, symbol, exchange, instrument_token, csv_path, seed)

    print(f"[1/5] Fetching base 1-minute data (mode={mode}, lookback={lookback_days}d)...")
    base = provider.fetch_base(as_of, lookback_days)
    if not base:
        raise RuntimeError("No candles returned — check symbol/lookback/credentials.")
    print(f"      -> {len(base)} base candles from {base[0].t} to {base[-1].t}")

    sentiment = sentiment_override if sentiment_override is not None else sentiment_provider()
    print(f"[2/5] Sentiment input: {sentiment:+.2f} "
          f"({'override' if sentiment_override is not None else 'provider'})")

    flow = _resolve_flow(mode, symbol, exchange, use_flow, flow_json)
    print(f"[3/5] Flow input [{flow.source}]: score {flow.score:+.2f} — {flow.notes}")

    journal_path = journal_path or os.path.join(out_dir, "journal.jsonl")
    priors = load_template_priors(journal_path)
    if priors:
        ranked = sorted(priors.items(), key=lambda kv: kv[1], reverse=True)
        print(f"[4/5] Learned template priors from {journal_path} "
              f"({len(load_journal(journal_path))} journaled sessions): "
              + ", ".join(f"{n}={w:.2f}" for n, w in ranked))
    else:
        print(f"[4/5] No journal history at {journal_path} — all template priors neutral "
              f"(run --replay-days N to build the learning loop's history).")

    print("[5/5] Analyzing + generating scenarios...")
    result = analyze_and_generate(base, as_of, custom_gap_points=custom_gap_points,
                                    sentiment=sentiment, flow=flow, template_priors=priors,
                                    top_k=top_k, seed=seed, verbose=True)

    os.makedirs(out_dir, exist_ok=True)
    svg_path = os.path.join(out_dir, "pathways.svg")
    render_pathways_svg(result.scenarios, result.ledger, result.prior_close, svg_path)
    png_path = os.path.join(out_dir, "pathways.png")
    rendered_png = try_render_matplotlib(result.scenarios, result.ledger,
                                          result.prior_close, png_path)

    context_map = build_context_map(result.layers, result.ledger, result.bias,
                                     result.gap_buckets, result.scenarios, result.prior_close)
    context_map["flow"] = {"source": result.flow.source, "score": result.flow.score,
                            "pcr": result.flow.pcr,
                            "depth_imbalance": result.flow.depth_imbalance,
                            "notes": result.flow.notes}
    context_map["template_priors"] = priors
    json_path = os.path.join(out_dir, "context_map.json")
    with open(json_path, "w") as f:
        json.dump(context_map, f, indent=2, default=str)

    print(f"\nDone. Wrote:\n  {svg_path}" + (f"\n  {png_path}" if rendered_png else "") +
          f"\n  {json_path}")
    return context_map


def replay_run(symbol: str, mode: str, as_of: datetime, lookback_days: int,
                replay_days: int, csv_path: Optional[str] = None, exchange: str = "NSE",
                instrument_token: Optional[int] = None, seed: int = 7,
                custom_gap_points: Optional[List[float]] = None, top_k: int = 3,
                out_dir: str = "market_pathway_output",
                journal_path: Optional[str] = None) -> List[dict]:
    """THE LEARNING LOOP. For each of the last `replay_days` completed
    sessions: rewind to the prior day's close, generate scenarios exactly
    as run() would have that evening (same code path, priors as they stood
    from earlier replayed days), then score every generated pathway
    against the session that actually printed. Each day appends a journal
    entry recording (a) actual vs predicted gap bucket, (b) every
    template's best match score, (c) the winning pathway. The journal then
    feeds load_template_priors() for all future runs — including later
    iterations of this same loop, so priors compound day over day in
    strict chronological order (no lookahead: day N's generation only ever
    sees journal entries from days < N)."""
    provider = _make_provider(mode, symbol, exchange, instrument_token, csv_path, seed)
    journal_path = journal_path or os.path.join(out_dir, "journal.jsonl")

    print(f"[replay] Fetching base data (mode={mode}, lookback={lookback_days}d)...")
    base = provider.fetch_base(as_of, lookback_days)
    if not base:
        raise RuntimeError("No candles returned.")
    daily = resample(base, "1D", 1440, "calendar_day")
    # completed sessions only, oldest -> newest, most recent `replay_days`
    days = [c.t for c in daily][-(replay_days + 1):]
    if len(days) < 2:
        raise RuntimeError(f"Need at least 2 daily candles to replay; got {len(days)}.")

    entries = []
    for di in range(1, len(days)):
        day = days[di]
        prev_day = days[di - 1]
        _, prev_close_t = _session_bounds(prev_day)
        day_open_t, day_close_t = _session_bounds(day)

        base_before = [c for c in base if c.t <= prev_close_t]
        day_candles = [c for c in base if day_open_t <= c.t <= day_close_t]
        if len(base_before) < 500 or len(day_candles) < 30:
            print(f"[replay] {day.date()}: skipped (insufficient data)")
            continue

        priors = load_template_priors(journal_path)
        result = analyze_and_generate(base_before, prev_close_t,
                                        custom_gap_points=custom_gap_points,
                                        sentiment=0.0, template_priors=priors,
                                        top_k=top_k, seed=seed + di, verbose=False)

        actual = actual_day_path(day_candles)
        actual_open = day_candles[0].o
        actual_gap = actual_open - result.prior_close
        # bucket whose gap size is closest to what actually printed
        nearest_gap = min(result.scenarios.keys(), key=lambda p: abs(p - actual_gap))
        predicted_top = result.gap_buckets[0].points if result.gap_buckets else 0.0
        gap_direction_hit = (actual_gap >= 0) == (predicted_top >= 0)

        match_by_template: Dict[str, float] = {}
        best: Optional[Tuple[str, float]] = None
        for pw in result.scenarios.get(nearest_gap, []):
            s = score_curve_match(pw, actual, result.atr)
            prev_best = match_by_template.get(pw.template_name, 0.0)
            match_by_template[pw.template_name] = max(prev_best, s)
            if best is None or s > best[1]:
                best = (pw.template_name, s)

        entry = {
            "date": str(day.date()), "symbol": symbol,
            "prior_close": round(result.prior_close, 2),
            "actual_open": round(actual_open, 2),
            "actual_gap": round(actual_gap, 2),
            "nearest_gap_bucket": nearest_gap,
            "predicted_top_bucket": predicted_top,
            "gap_direction_hit": gap_direction_hit,
            "session_bias_score": result.bias.score,
            "match_by_template": match_by_template,
            "best_template": best[0] if best else None,
            "best_match_score": best[1] if best else None,
        }
        journal_append(journal_path, entry)
        entries.append(entry)
        print(f"[replay] {day.date()}: gap {actual_gap:+.0f} (nearest bucket {nearest_gap:+.0f}, "
              f"direction {'HIT' if gap_direction_hit else 'MISS'}) | best template: "
              f"{best[0] if best else '-'} (match {best[1] if best else 0:.3f})")

    if entries:
        hits = sum(1 for e in entries if e["gap_direction_hit"])
        print(f"\n[replay] {len(entries)} sessions journaled -> {journal_path}")
        print(f"[replay] gap DIRECTION hit rate: {hits}/{len(entries)} "
              f"({hits / len(entries) * 100:.0f}%) — heuristic, small sample, treat accordingly")
        priors = load_template_priors(journal_path)
        for name, w in sorted(priors.items(), key=lambda kv: kv[1], reverse=True):
            print(f"[replay]   learned prior: {name:<24} {w:.3f}")
    return entries


def track_run(symbol: str, mode: str, as_of: datetime, lookback_days: int,
               csv_path: Optional[str] = None, exchange: str = "NSE",
               instrument_token: Optional[int] = None, seed: int = 7,
               custom_gap_points: Optional[List[float]] = None, top_k: int = 3,
               out_dir: str = "market_pathway_output",
               journal_path: Optional[str] = None,
               use_flow: bool = False, flow_json: Optional[str] = None,
               sentiment_override: Optional[float] = None) -> dict:
    """LIVE TRACKER. `as_of` is a moment INSIDE the current session. The
    engine rewinds to yesterday's close, generates this morning's scenarios
    (same code path as run(), so what you track is what you would have been
    handed pre-open), then scores each pathway against the tape that has
    actually printed so far — over the elapsed part of the session only —
    and tells you which projection the day is currently tracing. Re-run it
    whenever you want an updated read; in live mode each invocation
    re-fetches up-to-the-minute candles from Kite."""
    provider = _make_provider(mode, symbol, exchange, instrument_token, csv_path, seed)
    print(f"[track] Fetching base data through {as_of} (mode={mode})...")
    base = provider.fetch_base(as_of, lookback_days)
    if not base:
        raise RuntimeError("No candles returned.")

    day_open_t, day_close_t = _session_bounds(as_of)
    today = [c for c in base if day_open_t <= c.t <= min(as_of, day_close_t)]
    if len(today) < 5:
        raise RuntimeError(f"Only {len(today)} candles printed today by {as_of} — nothing to track yet.")
    daily = resample(base, "1D", 1440, "calendar_day")
    prev_days = [c.t for c in daily if c.t.date() < as_of.date()]
    if not prev_days:
        raise RuntimeError("No prior session in the data to anchor scenarios on.")
    _, prev_close_t = _session_bounds(prev_days[-1])
    base_before = [c for c in base if c.t <= prev_close_t]

    sentiment = sentiment_override if sentiment_override is not None else 0.0
    flow = _resolve_flow(mode, symbol, exchange, use_flow, flow_json)
    priors = load_template_priors(journal_path or os.path.join(out_dir, "journal.jsonl"))

    print("[track] Generating this morning's scenarios (data through yesterday's close)...")
    result = analyze_and_generate(base_before, prev_close_t,
                                    custom_gap_points=custom_gap_points,
                                    sentiment=sentiment, flow=flow,
                                    template_priors=priors, top_k=top_k,
                                    seed=seed, verbose=False)

    actual = actual_day_path(today)
    elapsed_frac = actual[-1][0] if actual else 0.0
    actual_open = today[0].o
    actual_gap = actual_open - result.prior_close
    nearest_gap = min(result.scenarios.keys(), key=lambda p: abs(p - actual_gap))

    print(f"[track] Session {as_of.date()} | open {actual_open:.2f} = gap {actual_gap:+.1f} "
          f"vs prior close {result.prior_close:.2f} (nearest bucket {nearest_gap:+.0f})")
    print(f"[track] Elapsed: {session_clock(0)} -> {session_clock(elapsed_frac)} "
          f"({elapsed_frac * 100:.0f}% of session)\n")

    ranked = []
    for gap_pts, pathways in result.scenarios.items():
        for pw in pathways:
            s = score_curve_match(pw, actual, result.atr, upto_frac=max(0.05, elapsed_frac))
            ranked.append((s, gap_pts, pw))
    ranked.sort(key=lambda x: x[0], reverse=True)

    print(f"{'match':>6}  {'gap':>5}  {'template':<24} narrative-so-far")
    for s, gap_pts, pw in ranked[:8]:
        marker = " <-- tracking" if (s, gap_pts) == (ranked[0][0], ranked[0][1]) else ""
        print(f"{s:>6.3f}  {gap_pts:+5.0f}  {pw.template_name:<24} "
              f"plausibility {pw.plausibility_score:.2f}{marker}")

    os.makedirs(out_dir, exist_ok=True)
    svg_path = os.path.join(out_dir, "track.svg")
    # draw only the actual-gap bucket's pathways so the overlay is readable
    render_pathways_svg({nearest_gap: result.scenarios[nearest_gap]}, result.ledger,
                         result.prior_close, svg_path, actual_path=actual,
                         title_suffix=f"LIVE TRACK {as_of.date()} @ {session_clock(elapsed_frac)}")
    report = {
        "as_of": as_of.isoformat(), "elapsed_frac": round(elapsed_frac, 3),
        "actual_gap": round(actual_gap, 2), "nearest_gap_bucket": nearest_gap,
        "ranking": [
            {"match": s, "gap_bucket": gp, "template": pw.template_name,
             "plausibility": round(pw.plausibility_score, 3), "narrative": pw.narrative}
            for s, gp, pw in ranked[:8]
        ],
    }
    json_path = os.path.join(out_dir, "track_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[track] Wrote:\n  {svg_path}\n  {json_path}")
    return report


# =========================================================================
# 17. KITE AUTH HELPERS — `--login` (morning OAuth flow) and `--check-auth`
# =========================================================================

def kite_login_flow(api_key: Optional[str] = None, api_secret: Optional[str] = None) -> int:
    """Interactive daily login. Kite Connect auth has three pieces with
    different lifetimes, and confusing them is the #1 source of 'am I even
    connected?' doubt:

      api_key / api_secret  -> permanent, from your app on developers.kite.trade
      request_token         -> single-use, ~few minutes, produced by the
                               browser login redirect
      access_token          -> what the engine actually uses; valid until
                               ~7:30 AM IST the NEXT day, then dead

    This flow: print the login URL -> you log in to Zerodha in a browser ->
    Kite redirects to your app's registered redirect URL with
    ?request_token=XXX -> paste that token here -> we exchange
    (request_token + api_secret) for today's access_token and print the
    export line. The api_secret never needs to exist in your shell after
    this exchange; the engine itself only ever reads KITE_API_KEY and
    KITE_ACCESS_TOKEN."""
    api_key = api_key or os.environ.get("KITE_API_KEY", "")
    api_secret = api_secret or os.environ.get("KITE_API_SECRET", "")
    if not api_key:
        print("error: set KITE_API_KEY (from your app on developers.kite.trade)")
        return 1
    if not api_secret:
        print("error: set KITE_API_SECRET (from the same app page)")
        return 1
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("error: kiteconnect not installed. Run: pip install kiteconnect")
        return 1

    kite = KiteConnect(api_key=api_key)
    print("\nStep 1 — open this URL in a browser and log in to Zerodha:\n")
    print(f"    {kite.login_url()}\n")
    print("Step 2 — after you authorise, Kite redirects to your app's registered")
    print("redirect URL with ?request_token=... in the address bar. Copy it fast —")
    print("request_tokens are single-use and expire in minutes.\n")
    try:
        request_token = input("request_token: ").strip()
    except EOFError:
        print("\nno request_token provided")
        return 1
    if not request_token:
        print("no request_token entered")
        return 1
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as e:
        print(f"\nLOGIN FAILED: {e}")
        print("Common causes: request_token already used or expired (redo the browser")
        print("step and paste immediately), or api_secret doesn't match this api_key.")
        return 1
    token = data.get("access_token", "")
    print("\nSUCCESS — logged in as "
          f"{data.get('user_name', '?')} ({data.get('user_id', '?')}).")
    print("Export today's token, then you're set until ~7:30 AM IST tomorrow:\n")
    print(f"    export KITE_ACCESS_TOKEN={token}\n")
    print("Verify any time with:  python3 market_pathway_engine.py --check-auth")
    return 0


def kite_check_auth() -> int:
    """The definitive 'am I connected correctly?' answer. Exit code 0 =
    yes, your key + token are valid RIGHT NOW and the engine's --mode live
    will work. Non-zero = no, with the specific reason printed. Checks, in
    order: env vars present -> kiteconnect importable -> profile() call
    succeeds (validates the token against Zerodha's servers) -> a real
    market-data call succeeds (validates data permissions, not just auth).
    """
    api_key = os.environ.get("KITE_API_KEY")
    token = os.environ.get("KITE_ACCESS_TOKEN")
    print("Kite Connect auth check")
    print(f"  KITE_API_KEY:      {'set (' + api_key[:4] + '...)' if api_key else 'MISSING'}")
    print(f"  KITE_ACCESS_TOKEN: {'set (' + token[:4] + '...)' if token else 'MISSING'}")
    if not api_key or not token:
        print("\nNOT CONNECTED — export both env vars first. Get today's access_token")
        print("via:  python3 market_pathway_engine.py --login")
        return 1
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("\nNOT CONNECTED — kiteconnect not installed: pip install kiteconnect")
        return 1
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(token)
    try:
        profile = kite.profile()
    except Exception as e:
        print(f"\nNOT CONNECTED — profile() rejected: {e}")
        print("If this says TokenException/403: your access_token has expired (they die")
        print("daily ~7:30 AM IST). Re-run --login to mint today's token.")
        return 1
    print(f"\n  profile OK: {profile.get('user_name', '?')} ({profile.get('user_id', '?')}), "
          f"broker={profile.get('broker', '?')}")
    try:
        ltp = kite.ltp(["NSE:NIFTY 50"])
        for sym, row in ltp.items():
            print(f"  market data OK: {sym} last price {row.get('last_price')}")
    except Exception as e:
        print(f"  market data check failed: {e}")
        print("  (auth is valid but the data call was refused — check your app's")
        print("   subscription/permissions on developers.kite.trade)")
        return 2
    print("\nCONNECTED — --mode live will work with these credentials.")
    return 0


# =========================================================================
# 18. CLI
# =========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Standalone multi-timeframe liquidity-structure & pathway-projection engine.")
    ap.add_argument("--demo", action="store_true",
                     help="Shortcut for --mode synthetic with sensible defaults; runs with zero setup.")
    ap.add_argument("--mode", choices=["synthetic", "csv", "live"], default="synthetic")
    ap.add_argument("--symbol", default="NIFTY 50", help="Tradingsymbol, used in --mode live")
    ap.add_argument("--exchange", default="NSE")
    ap.add_argument("--instrument-token", type=int, default=None)
    ap.add_argument("--csv", dest="csv_path", default=None, help="CSV path for --mode csv")
    ap.add_argument("--as-of", dest="as_of", default=None,
                     help="ISO datetime treated as 'P day' end (default: now, IST)")
    ap.add_argument("--lookback-days", type=int, default=45)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gap-points", type=float, nargs="*", default=None,
                     help="Explicit positive gap magnitudes, e.g. --gap-points 50 100 200 250 300 "
                          "(negatives are added automatically)")
    ap.add_argument("--sentiment", type=float, default=None,
                     help="Manual sentiment override in [-1,1]; default uses the neutral stub provider")
    ap.add_argument("--top-k", type=int, default=3, help="Pathways kept per gap bucket")
    ap.add_argument("--out-dir", default="market_pathway_output")
    ap.add_argument("--replay-days", type=int, default=None,
                     help="LEARNING LOOP: replay the last N completed sessions — generate each "
                          "day's scenarios from only pre-open data, score them against what "
                          "actually printed, and journal per-template match scores. The journal "
                          "then tilts template ranking in all future runs.")
    ap.add_argument("--track", action="store_true",
                     help="LIVE TRACKER: treat --as-of as a moment inside today's session; "
                          "generate this morning's scenarios from yesterday's close and rank "
                          "which pathway the tape printed so far is actually tracing.")
    ap.add_argument("--journal", dest="journal_path", default=None,
                     help="Path to the replay journal (default: <out-dir>/journal.jsonl)")
    ap.add_argument("--flow", action="store_true",
                     help="Live mode only: pull options OI (put-call ratio) + futures depth "
                          "from Kite and feed the flow score into gap probabilities.")
    ap.add_argument("--flow-json", default=None,
                     help='Manually inject flow data for any mode, e.g. '
                          '\'{"pcr": 1.32, "depth_imbalance": 0.15}\'')
    ap.add_argument("--login", action="store_true",
                     help="Run the daily Kite Connect OAuth flow: prints the login URL, "
                          "exchanges your request_token for today's access_token, and exits.")
    ap.add_argument("--check-auth", action="store_true",
                     help="Verify KITE_API_KEY/KITE_ACCESS_TOKEN against Zerodha's servers "
                          "(profile + a live data call) and exit. Exit code 0 = connected.")
    args = ap.parse_args()

    if args.login:
        sys.exit(kite_login_flow())
    if args.check_auth:
        sys.exit(kite_check_auth())

    mode = "synthetic" if args.demo else args.mode
    as_of = datetime.now(IST) if not args.as_of else datetime.fromisoformat(args.as_of)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=IST)

    if args.replay_days:
        replay_run(
            symbol=args.symbol, mode=mode, as_of=as_of, lookback_days=args.lookback_days,
            replay_days=args.replay_days, csv_path=args.csv_path, exchange=args.exchange,
            instrument_token=args.instrument_token, seed=args.seed,
            custom_gap_points=args.gap_points, top_k=args.top_k, out_dir=args.out_dir,
            journal_path=args.journal_path,
        )
        return

    if args.track:
        track_run(
            symbol=args.symbol, mode=mode, as_of=as_of, lookback_days=args.lookback_days,
            csv_path=args.csv_path, exchange=args.exchange,
            instrument_token=args.instrument_token, seed=args.seed,
            custom_gap_points=args.gap_points, top_k=args.top_k, out_dir=args.out_dir,
            journal_path=args.journal_path, use_flow=args.flow, flow_json=args.flow_json,
            sentiment_override=args.sentiment,
        )
        return

    run(
        symbol=args.symbol, mode=mode, as_of=as_of, lookback_days=args.lookback_days,
        csv_path=args.csv_path, exchange=args.exchange, instrument_token=args.instrument_token,
        seed=args.seed, custom_gap_points=args.gap_points, sentiment_override=args.sentiment,
        top_k=args.top_k, out_dir=args.out_dir, use_flow=args.flow, flow_json=args.flow_json,
        journal_path=args.journal_path,
    )


if __name__ == "__main__":
    main()
