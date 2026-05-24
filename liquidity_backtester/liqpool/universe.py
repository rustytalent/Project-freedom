"""Named research universes for local and cloud training runs."""
from __future__ import annotations

from typing import List


CORE25_SYMBOLS: List[str] = [
    "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
    "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "TVSMOTOR",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
]


def symbols_for_universe(name: str) -> List[str]:
    key = (name or "").strip().lower()
    if key in ("", "custom"):
        return []
    if key == "core25":
        return list(CORE25_SYMBOLS)
    raise ValueError(f"unknown universe '{name}'. Supported: core25, custom")
