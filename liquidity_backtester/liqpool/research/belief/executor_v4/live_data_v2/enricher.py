"""Enricher — Kite quote payload → EnrichedQuote → slot_reading.

Two functions:

  * ``enrich_kite_quote(raw)`` — pure conversion, no I/O. The result
    holds full 5-level depth + OI + pending quantities + last trade
    metadata, ready to be fed into the microstructure detectors.
  * ``slot_reading_from_enriched(enriched, …)`` — produces a dict the
    manager's ``snapshot["slot_readings"]`` consumer can read directly,
    matching the legacy schema plus the new microstructure fields.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from .enriched_quote import DepthLevel, EnrichedQuote


def _normalise_last_trade_time(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def enrich_kite_quote(raw: Mapping[str, Any], *,
                          tradingsymbol: str = "") -> EnrichedQuote:
    """Pure Kite quote payload → EnrichedQuote dataclass.

    Accepts the dict shape Kite Connect returns from ``quote()``. Safe
    for partially-populated payloads — missing fields default to 0 or
    empty lists.
    """
    depth = raw.get("depth") or {}
    buy_levels: List[DepthLevel] = []
    sell_levels: List[DepthLevel] = []
    for lvl in (depth.get("buy") or []):
        buy_levels.append(DepthLevel.from_kite(lvl))
    for lvl in (depth.get("sell") or []):
        sell_levels.append(DepthLevel.from_kite(lvl))

    return EnrichedQuote(
        tradingsymbol=str(tradingsymbol
                           or raw.get("tradingsymbol") or ""),
        ltp=float(raw.get("last_price")
                   or raw.get("ltp") or 0.0),
        last_quantity=int(raw.get("last_quantity") or 0),
        last_trade_time_iso=_normalise_last_trade_time(
            raw.get("last_trade_time")),
        average_price=float(raw.get("average_price")
                              or raw.get("avg_price") or 0.0),
        volume=float(raw.get("volume") or 0.0),
        oi=float(raw.get("oi") or 0.0),
        oi_day_high=float(raw.get("oi_day_high") or 0.0),
        oi_day_low=float(raw.get("oi_day_low") or 0.0),
        buy_quantity_total=int(raw.get("buy_quantity") or 0),
        sell_quantity_total=int(raw.get("sell_quantity") or 0),
        depth_buy=buy_levels,
        depth_sell=sell_levels,
    )


def slot_reading_from_enriched(quote: EnrichedQuote, *,
                                   strike: float,
                                   option_type: str,
                                   level: int,
                                   label: str,
                                   moneyness_label: str = "",
                                   dod_z: float = 0.0,
                                   acceptance: str = "normal",
                                   friendliness: float = 1.0,
                                   spread_state: str = "clean",
                                   is_abnormal: bool = False,
                                   ) -> Dict[str, Any]:
    """Build the slot_reading dict the manager's slot loop consumes.

    Augments the legacy schema with microstructure fields the V2
    detectors read: ``depth_buy_levels``, ``depth_sell_levels``,
    ``oi``, ``oi_day_high``, ``buy_quantity_pending``,
    ``sell_quantity_pending``, ``last_trade_size``,
    ``total_depth_buy_qty``, ``total_depth_sell_qty``,
    ``mid``, ``spread``.

    Returning ``Dict[str, Any]`` (not a dataclass) keeps the legacy
    snapshot consumers untouched.
    """
    return {
        # Legacy schema fields.
        "strike": float(strike),
        "option_type": option_type,
        "level": int(level),
        "label": label,
        "moneyness_label": moneyness_label or label,
        "acceptance": acceptance,
        "friendliness": float(friendliness),
        "spread_state": spread_state,
        "mark_source": "microprice" if quote.spread > 0 else "ltp",
        "is_abnormal": bool(is_abnormal),
        "dod_z": float(dod_z),
        "mark_price": float(quote.mid if quote.mid > 0 else quote.ltp),
        # New microstructure fields (V2 detectors read these).
        "ltp": float(quote.ltp),
        "bid": float(quote.bid),
        "ask": float(quote.ask),
        "spread": float(quote.spread),
        "oi": float(quote.oi),
        "oi_day_high": float(quote.oi_day_high),
        "oi_day_low": float(quote.oi_day_low),
        "buy_quantity_pending": int(quote.buy_quantity_total),
        "sell_quantity_pending": int(quote.sell_quantity_total),
        "last_trade_size": int(quote.last_quantity),
        "last_trade_time_iso": quote.last_trade_time_iso,
        "total_depth_buy_qty": int(quote.total_depth_buy_qty),
        "total_depth_sell_qty": int(quote.total_depth_sell_qty),
        "depth_buy_levels": [d.to_dict() for d in quote.depth_buy],
        "depth_sell_levels": [d.to_dict() for d in quote.depth_sell],
    }
