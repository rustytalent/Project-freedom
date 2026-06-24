"""Build path-aware candle features from ticks or synthetic OHLC replay ticks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import AuctionGraphConfig
from .models import CandleFeatures, PriceBinFeature, Tick
from .premium_features import premium_response
from .ring_buffer import RingBuffer
from .urgency_features import urgency_from_ticks


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    if timeframe == "1m":
        return pd.Timedelta(minutes=1)
    if timeframe == "5m":
        return pd.Timedelta(minutes=5)
    if timeframe == "15m":
        return pd.Timedelta(minutes=15)
    raise ValueError(f"unsupported timeframe: {timeframe}")


def candle_bucket(ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    freq = {"1m": "min", "5m": "5min", "15m": "15min"}[timeframe]
    return pd.Timestamp(ts).floor(freq)


@dataclass
class CandleAccumulator:
    symbol: str
    timeframe: str
    start_ts: pd.Timestamp
    ticks: list[Tick] = field(default_factory=list)

    def add(self, tick: Tick) -> None:
        self.ticks.append(tick)

    def finalize(self, bin_count: int) -> CandleFeatures:
        if not self.ticks:
            raise ValueError("cannot finalize empty candle")

        ticks = sorted(self.ticks, key=lambda t: t.ts)
        prices = np.asarray([float(t.price) for t in ticks], dtype=float)
        volumes = np.asarray([float(t.volume or 0.0) for t in ticks], dtype=float)

        open_px = float(prices[0])
        close_px = float(prices[-1])
        high_px = float(np.max(prices))
        low_px = float(np.min(prices))
        range_px = high_px - low_px
        body = close_px - open_px
        abs_body = abs(body)
        upper_wick = high_px - max(open_px, close_px)
        lower_wick = min(open_px, close_px) - low_px
        body_ratio = abs_body / max(range_px, 1e-9)
        close_location = (close_px - low_px) / max(range_px, 1e-9)

        high_pos = int(np.argmax(prices))
        low_pos = int(np.argmin(prices))
        high_time = ticks[high_pos].ts
        low_time = ticks[low_pos].ts
        high_first = high_pos < low_pos if high_pos != low_pos else None
        low_first = low_pos < high_pos if high_pos != low_pos else None
        time_to_high = (high_time - ticks[0].ts).total_seconds()
        time_to_low = (low_time - ticks[0].ts).total_seconds()

        step_seconds = _tick_step_seconds(ticks)
        top_cut = low_px + 0.75 * range_px
        bottom_cut = low_px + 0.25 * range_px
        top_time = float(np.sum(prices >= top_cut) * step_seconds) if range_px > 0 else 0.0
        bottom_time = float(np.sum(prices <= bottom_cut) * step_seconds) if range_px > 0 else 0.0

        total_abs_path = float(np.sum(np.abs(np.diff(prices)))) if len(prices) > 1 else 0.0
        path_efficiency = abs(close_px - open_px) / max(total_abs_path, 1e-9)
        churn_ratio = total_abs_path / max(abs(close_px - open_px), range_px, 1e-9)
        mfe = high_px - open_px
        mae = open_px - low_px

        upper_rejection_distance = high_px - close_px
        lower_reclaim_distance = close_px - low_px
        duration_after_high = max((ticks[-1].ts - high_time).total_seconds(), 1e-3)
        duration_after_low = max((ticks[-1].ts - low_time).total_seconds(), 1e-3)
        upper_rejection_speed = upper_rejection_distance / duration_after_high
        lower_reclaim_speed = lower_reclaim_distance / duration_after_low
        failed_high_hold = bool(range_px > 0 and close_px < high_px - 0.35 * range_px)
        failed_low_hold = bool(range_px > 0 and close_px > low_px + 0.35 * range_px)
        acceptance_proxy = _acceptance_proxy(prices, open_px, close_px, range_px)

        u_plus, u_minus, u_net, depth_available = urgency_from_ticks(ticks)
        ce_eff, pe_eff, premium_bias, premium_compression = premium_response(ticks)
        bins = _price_bins(ticks, prices, volumes, bin_count, low_px, high_px)

        end_ts = self.start_ts + timeframe_delta(self.timeframe)
        source_kind = ticks[0].source_kind if ticks else "tick_replay"
        return CandleFeatures(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_ts=self.start_ts,
            end_ts=end_ts,
            tick_count=len(ticks),
            source_kind=source_kind,
            open=open_px,
            high=high_px,
            low=low_px,
            close=close_px,
            volume=float(np.sum(volumes)),
            range=range_px,
            body=body,
            abs_body=abs_body,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            body_ratio=body_ratio,
            close_location=close_location,
            high_time=high_time,
            low_time=low_time,
            high_first=high_first,
            low_first=low_first,
            time_to_high_seconds=float(time_to_high),
            time_to_low_seconds=float(time_to_low),
            time_spent_top_quartile_seconds=top_time,
            time_spent_bottom_quartile_seconds=bottom_time,
            total_abs_path=total_abs_path,
            path_efficiency=float(path_efficiency),
            churn_ratio=float(churn_ratio),
            mfe_from_open=float(mfe),
            mae_from_open=float(mae),
            upper_rejection_distance=float(upper_rejection_distance),
            lower_reclaim_distance=float(lower_reclaim_distance),
            upper_rejection_speed=float(upper_rejection_speed),
            lower_reclaim_speed=float(lower_reclaim_speed),
            failed_high_hold=failed_high_hold,
            failed_low_hold=failed_low_hold,
            acceptance_proxy=float(acceptance_proxy),
            urgency_plus=float(u_plus),
            urgency_minus=float(u_minus),
            urgency_net=float(u_net),
            depth_available=depth_available,
            ce_efficiency=ce_eff,
            pe_efficiency=pe_eff,
            premium_bias=premium_bias,
            premium_compression=premium_compression,
            price_bins=bins,
        )


def build_candles(ticks: list[Tick], config: AuctionGraphConfig) -> list[CandleFeatures]:
    config.validate()
    history: RingBuffer[CandleFeatures] = RingBuffer(config.max_candles)
    active: CandleAccumulator | None = None
    active_bucket: pd.Timestamp | None = None

    for tick in sorted(ticks, key=lambda t: t.ts):
        bucket = candle_bucket(tick.ts, config.timeframe)
        if active is None:
            active_bucket = bucket
            active = CandleAccumulator(config.symbol, config.timeframe, bucket)
        elif bucket != active_bucket:
            history.append(active.finalize(config.bin_count))
            active_bucket = bucket
            active = CandleAccumulator(config.symbol, config.timeframe, bucket)
        active.add(tick)

    if active is not None and active.ticks:
        history.append(active.finalize(config.bin_count))
    return history.to_list()


def _tick_step_seconds(ticks: list[Tick]) -> float:
    if len(ticks) < 2:
        return 0.0
    deltas = [
        max((cur.ts - prev.ts).total_seconds(), 0.0)
        for prev, cur in zip(ticks, ticks[1:])
    ]
    if not deltas:
        return 0.0
    return float(np.median(deltas))


def _acceptance_proxy(prices: np.ndarray, open_px: float, close_px: float, range_px: float) -> float:
    if range_px <= 0:
        return 0.0
    midpoint = (open_px + close_px) / 2.0
    if close_px >= open_px:
        accepted = np.mean(prices >= midpoint)
    else:
        accepted = np.mean(prices <= midpoint)
    return float(accepted)


def _price_bins(
    ticks: list[Tick],
    prices: np.ndarray,
    volumes: np.ndarray,
    bin_count: int,
    low_px: float,
    high_px: float,
) -> list[PriceBinFeature]:
    if high_px <= low_px:
        high_px = low_px + 1e-6
    edges = np.linspace(low_px, high_px, bin_count + 1)
    idxs = np.clip(np.searchsorted(edges, prices, side="right") - 1, 0, bin_count - 1)

    step_seconds = _tick_step_seconds(ticks)
    rows: list[PriceBinFeature] = []
    changes = np.diff(prices, prepend=prices[0])
    grouped: dict[int, list[int]] = defaultdict(list)
    for pos, idx in enumerate(idxs):
        grouped[int(idx)].append(pos)

    for idx in range(bin_count):
        positions = grouped.get(idx, [])
        if not positions:
            rows.append(
                PriceBinFeature(
                    bin_index=idx,
                    low=float(edges[idx]),
                    high=float(edges[idx + 1]),
                    tick_count=0,
                    time_spent_seconds=0.0,
                    approx_volume=0.0,
                    up_ticks=0,
                    down_ticks=0,
                    net_tick_delta=0,
                    urgency_net=0.0,
                    premium_bias=None,
                )
            )
            continue
        delta_slice = changes[positions]
        up_ticks = int(np.sum(delta_slice > 0))
        down_ticks = int(np.sum(delta_slice < 0))
        bin_ticks = [ticks[p] for p in positions]
        _, _, urgency_net, _ = urgency_from_ticks(bin_ticks)
        _, _, premium_bias, _ = premium_response(bin_ticks)
        rows.append(
            PriceBinFeature(
                bin_index=idx,
                low=float(edges[idx]),
                high=float(edges[idx + 1]),
                tick_count=len(positions),
                time_spent_seconds=float(len(positions) * step_seconds),
                approx_volume=float(np.sum(volumes[positions])),
                up_ticks=up_ticks,
                down_ticks=down_ticks,
                net_tick_delta=up_ticks - down_ticks,
                urgency_net=float(urgency_net),
                premium_bias=premium_bias,
            )
        )
    return rows
