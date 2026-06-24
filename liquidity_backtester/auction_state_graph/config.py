"""Configuration for the standalone Auction State Graph."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuctionGraphConfig:
    symbol: str = "DEMO"
    timeframe: str = "1m"
    max_candles: int = 300
    bin_count: int = 16
    max_replay_rows: int | None = None
    min_tick_count_for_story: int = 3
    dashboard_title: str = "Auction State Graph"

    def validate(self) -> None:
        if self.timeframe not in {"1m", "5m", "15m"}:
            raise ValueError("timeframe must be one of: 1m, 5m, 15m")
        if self.max_candles < 10:
            raise ValueError("max_candles must be at least 10")
        if self.bin_count < 4:
            raise ValueError("bin_count must be at least 4")
