"""Mark-price engine + data-quality layer (Belief Engine, Phase 2).

The founder's #1 correction: the engine must run on LIVE PRICE, not wick
or candle logic — and pure LTP must NOT be treated as truth, because a
stale / locked / crossed / off-book last-traded print will hallucinate
every downstream signal.

The fix is a quote-derived MARK PRICE with an explicit quality grade:

    1. microprice — preferred when the quote is good. The Stoikov
       microprice weights the ask by the bid size and the bid by the ask
       size:

           microprice = (ask·bid_qty + bid·ask_qty) / (bid_qty + ask_qty)

       Intuition: when the bid is much larger than the ask (many buyers,
       few sellers), tradable pressure sits near the ask, so the mark
       leans toward the ask. This captures order-book pressure that a
       plain mid misses.

    2. mid = (bid + ask) / 2 — used when depth is unreliable but the
       quote is still valid (not crossed/locked).

    3. LTP — used ONLY as a trade-print confirmation, never as the
       primary mark. ``ltp_confirms`` is True only when the LTP is fresh
       AND sits inside the bid-ask. A stale or out-of-book LTP is flagged
       and ignored for pricing.

Crossed (ask < bid) or locked (ask == bid) or non-positive quotes are
invalid; the engine falls back to the last good mark (if supplied) and
flags the bar dirty so downstream layers can refuse to trade on it.

Everything here is per-quote and deterministic. ``MarkPriceTracker`` adds
a thin stateful wrapper that remembers the last good mark per contract
for the invalid-quote fallback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class Quote:
    """One contract's top-of-book snapshot at a tick.

    ``ltp_age_s`` is seconds since the last trade print; large values mark
    the LTP stale. ``ts`` is carried through for logging only.
    """
    bid: float
    ask: float
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    ltp: float = float("nan")
    ltp_age_s: float = 0.0
    ts: Any = None


@dataclass(frozen=True)
class MarkPrice:
    """The clean mark for one contract plus its quality assessment."""
    price: float
    source: str          # "microprice" / "mid" / "last_valid" / "ltp" / "invalid"
    quality: float       # [0,1]
    quality_label: str   # "good" / "usable" / "poor" / "invalid"
    mid: float
    microprice: float
    spread: float
    spread_pct: float
    is_crossed: bool
    is_locked: bool
    ltp_confirms: bool
    flags: Tuple[str, ...] = ()

    @property
    def tradable(self) -> bool:
        """A convenience gate: the quote is clean enough to act on."""
        return self.source in ("microprice", "mid") and self.quality_label in ("good", "usable")

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["flags"] = list(self.flags)
        return d


@dataclass
class MarkPriceConfig:
    """Thresholds for the mark-price decision.

    Spread is judged as a fraction of the mid so the thresholds transfer
    across a ₹5 deep-OTM premium and a ₹400 deep-ITM premium."""
    max_spread_pct_good: float = 0.02      # ≤2% spread → eligible for microprice
    max_spread_pct_usable: float = 0.08    # ≤8% spread → mid is still usable
    depth_full_qty: float = 1000.0         # total qty at which depth_score saturates to 1
    min_depth_qty: float = 1.0             # below this, microprice is not trusted
    max_ltp_age_s: float = 2.0             # LTP older than this is stale
    microprice_quality_floor: float = 0.55  # below this quality → fall back to mid
    spread_weight: float = 0.6             # quality = w·spread + (1-w)·depth
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if not (0.0 <= self.spread_weight <= 1.0):
            raise ValueError("spread_weight must be in [0,1]")
        if self.max_spread_pct_usable < self.max_spread_pct_good:
            raise ValueError("max_spread_pct_usable must be >= max_spread_pct_good")


def _microprice(bid: float, ask: float, bid_qty: float, ask_qty: float,
                eps: float) -> float:
    total = bid_qty + ask_qty
    if total <= eps:
        return (bid + ask) / 2.0
    return (ask * bid_qty + bid * ask_qty) / total


def _quality_label(quality: float, source: str) -> str:
    if source in ("invalid", "last_valid", "ltp"):
        return "invalid" if source == "invalid" else "poor"
    if quality >= 0.75:
        return "good"
    if quality >= 0.45:
        return "usable"
    return "poor"


def compute_mark(quote: Quote,
                 cfg: Optional[MarkPriceConfig] = None,
                 last_valid_price: Optional[float] = None) -> MarkPrice:
    """Compute the clean mark for one contract from its quote.

    Returns a :class:`MarkPrice`. When the quote is invalid (crossed,
    locked, or non-positive), falls back to ``last_valid_price`` if given,
    else the LTP if finite, else NaN — always flagged so downstream layers
    can refuse the bar.
    """
    cfg = cfg or MarkPriceConfig()
    bid, ask = float(quote.bid), float(quote.ask)
    bid_qty, ask_qty = float(quote.bid_qty), float(quote.ask_qty)
    ltp = float(quote.ltp)
    flags = []

    is_crossed = ask < bid
    is_locked = ask == bid and ask > 0
    valid = (bid > 0.0) and (ask > 0.0) and (ask >= bid) and not is_crossed

    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else float("nan")
    micro = _microprice(bid, ask, bid_qty, ask_qty, cfg.eps) if (bid > 0 and ask > 0) else float("nan")
    spread = (ask - bid) if (bid > 0 and ask > 0) else float("nan")
    spread_pct = (spread / mid) if (isinstance(mid, float) and mid > 0 and not math.isnan(spread)) else float("nan")

    # LTP confirmation: fresh AND inside the book.
    ltp_fresh = math.isfinite(ltp) and quote.ltp_age_s <= cfg.max_ltp_age_s
    ltp_inside = math.isfinite(ltp) and bid <= ltp <= ask if valid else False
    ltp_confirms = bool(ltp_fresh and ltp_inside)
    if math.isfinite(ltp) and not ltp_fresh:
        flags.append("stale_ltp")
    if valid and math.isfinite(ltp) and not ltp_inside:
        flags.append("ltp_outside")

    if not valid:
        if is_crossed:
            flags.append("crossed")
        if is_locked:
            flags.append("locked")
        if bid <= 0 or ask <= 0:
            flags.append("nonpositive_quote")
        # Fallback chain: last good mark → fresh LTP → NaN.
        if last_valid_price is not None and math.isfinite(last_valid_price):
            price, source = float(last_valid_price), "last_valid"
        elif ltp_fresh:
            price, source = ltp, "ltp"
            flags.append("ltp_fallback")
        else:
            price, source = float("nan"), "invalid"
        return MarkPrice(
            price=price, source=source, quality=0.0,
            quality_label=_quality_label(0.0, source),
            mid=mid, microprice=micro, spread=spread,
            spread_pct=spread_pct if not math.isnan(spread_pct) else float("nan"),
            is_crossed=is_crossed, is_locked=is_locked,
            ltp_confirms=ltp_confirms, flags=tuple(flags),
        )

    # Quality score from spread tightness + depth.
    depth = bid_qty + ask_qty
    spread_score = max(0.0, 1.0 - (spread_pct / (cfg.max_spread_pct_usable + cfg.eps)))
    depth_score = min(1.0, depth / (cfg.depth_full_qty + cfg.eps))
    quality = cfg.spread_weight * spread_score + (1.0 - cfg.spread_weight) * depth_score
    quality = max(0.0, min(1.0, quality))

    if spread_pct > cfg.max_spread_pct_usable:
        flags.append("wide_spread")
    if depth < cfg.min_depth_qty:
        flags.append("thin_depth")
    if is_locked:
        flags.append("locked")

    # Source decision.
    can_micro = (depth >= cfg.min_depth_qty
                 and quality >= cfg.microprice_quality_floor
                 and spread_pct <= cfg.max_spread_pct_good)
    if can_micro:
        price, source = micro, "microprice"
    else:
        price, source = mid, "mid"

    return MarkPrice(
        price=float(price), source=source, quality=quality,
        quality_label=_quality_label(quality, source),
        mid=mid, microprice=micro, spread=spread, spread_pct=spread_pct,
        is_crossed=is_crossed, is_locked=is_locked,
        ltp_confirms=ltp_confirms, flags=tuple(flags),
    )


@dataclass
class MarkPriceTracker:
    """Stateful wrapper that remembers the last good mark per contract key,
    so an invalid quote can fall back to the most recent clean price."""
    cfg: MarkPriceConfig = field(default_factory=MarkPriceConfig)
    _last_good: Dict[Any, float] = field(default_factory=dict, init=False)

    def update(self, key: Any, quote: Quote) -> MarkPrice:
        last = self._last_good.get(key)
        mark = compute_mark(quote, self.cfg, last_valid_price=last)
        # Only remember marks that came from a clean quote.
        if mark.source in ("microprice", "mid") and math.isfinite(mark.price):
            self._last_good[key] = mark.price
        return mark

    def last_good(self, key: Any) -> Optional[float]:
        return self._last_good.get(key)

    def reset(self) -> None:
        self._last_good.clear()
