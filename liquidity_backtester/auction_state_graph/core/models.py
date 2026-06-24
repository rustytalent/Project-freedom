"""Typed structures for auction-state replay.

The objects here are deliberately serializable and compact. They are designed
for replay/offline dashboards first, and live streaming later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


def _iso(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@dataclass
class Tick:
    ts: pd.Timestamp
    price: float
    volume: float = 0.0
    bid: float | None = None
    ask: float | None = None
    bid_qty: float | None = None
    ask_qty: float | None = None
    ce_price: float | None = None
    pe_price: float | None = None
    symbol: str = ""
    source_kind: str = "tick_replay"


@dataclass
class PriceBinFeature:
    bin_index: int
    low: float
    high: float
    tick_count: int
    time_spent_seconds: float
    approx_volume: float
    up_ticks: int
    down_ticks: int
    net_tick_delta: int
    urgency_net: float
    premium_bias: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NextState:
    name: str
    probability: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Story:
    title: str
    bullets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandleFeatures:
    symbol: str
    timeframe: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    tick_count: int
    source_kind: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    range: float
    body: float
    abs_body: float
    upper_wick: float
    lower_wick: float
    body_ratio: float
    close_location: float
    high_time: pd.Timestamp | None
    low_time: pd.Timestamp | None
    high_first: bool | None
    low_first: bool | None
    time_to_high_seconds: float | None
    time_to_low_seconds: float | None
    time_spent_top_quartile_seconds: float
    time_spent_bottom_quartile_seconds: float
    total_abs_path: float
    path_efficiency: float
    churn_ratio: float
    mfe_from_open: float
    mae_from_open: float
    upper_rejection_distance: float
    lower_reclaim_distance: float
    upper_rejection_speed: float
    lower_reclaim_speed: float
    failed_high_hold: bool
    failed_low_hold: bool
    acceptance_proxy: float
    urgency_plus: float
    urgency_minus: float
    urgency_net: float
    depth_available: bool
    ce_efficiency: float | None
    pe_efficiency: float | None
    premium_bias: float | None
    premium_compression: float | None
    price_bins: list[PriceBinFeature] = field(default_factory=list)
    label: str = "Unclassified"
    label_reasons: list[str] = field(default_factory=list)
    story: Story | None = None
    next_states: list[NextState] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_ts"] = _iso(self.start_ts)
        data["end_ts"] = _iso(self.end_ts)
        data["high_time"] = _iso(self.high_time)
        data["low_time"] = _iso(self.low_time)
        data["price_bins"] = [row.to_dict() for row in self.price_bins]
        data["story"] = self.story.to_dict() if self.story else None
        data["next_states"] = [row.to_dict() for row in self.next_states]
        return data
