"""Deterministic candle-state labels and next-state possibilities."""

from __future__ import annotations

import math

import numpy as np

from .models import CandleFeatures, NextState
from .story_engine import story_for_candle


def annotate_candles(candles: list[CandleFeatures]) -> list[CandleFeatures]:
    if not candles:
        return candles

    ranges = np.asarray([c.range for c in candles if c.range > 0], dtype=float)
    churns = np.asarray([c.churn_ratio for c in candles if math.isfinite(c.churn_ratio)], dtype=float)
    range_med = float(np.median(ranges)) if len(ranges) else 0.0
    churn_med = float(np.median(churns)) if len(churns) else 1.0

    for i, candle in enumerate(candles):
        label, reasons = classify_candle(candle, range_med, churn_med)
        candle.label = label
        candle.label_reasons = reasons
        recent = candles[max(0, i - 12): i + 1]
        candle.next_states = next_state_possibilities(candle, recent)
        candle.story = story_for_candle(candle, recent)
    return candles


def classify_candle(c: CandleFeatures, range_med: float, churn_med: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    is_large = c.range >= max(range_med * 1.15, 1e-9)
    is_small = c.range <= max(range_med * 0.55, 1e-9) if range_med > 0 else c.range == 0
    efficient = c.path_efficiency >= 0.55
    churny = c.churn_ratio >= max(churn_med * 1.25, 1.8)
    strong_body = c.body_ratio >= 0.55
    upper_reject = c.upper_wick >= max(c.range * 0.35, 1e-9) and c.close_location < 0.65
    lower_reclaim = c.lower_wick >= max(c.range * 0.35, 1e-9) and c.close_location > 0.35
    premium_compress = c.premium_compression is not None and c.premium_compression < -0.015

    if premium_compress and c.body_ratio < 0.35:
        reasons.append("premium shrank while price failed to travel cleanly")
        return "Premium Compression", reasons

    if is_small and churny:
        reasons.append("small candle with high internal path churn")
        return "No-Trade Churn", reasons

    if is_small and c.tick_count >= 3 and c.path_efficiency < 0.25:
        reasons.append("low range and low path efficiency")
        return "Dead Auction", reasons

    if upper_reject and lower_reclaim and churny:
        reasons.append("both ends were probed and neither side held cleanly")
        return "Two-Sided Fight", reasons

    if c.body > 0 and is_large and strong_body and efficient and c.urgency_net > 0.15:
        reasons.append("large positive body with efficient path and positive urgency")
        return "Clean Bullish Impulse", reasons

    if c.body < 0 and is_large and strong_body and efficient and c.urgency_net < -0.15:
        reasons.append("large negative body with efficient path and negative urgency")
        return "Clean Bearish Impulse", reasons

    if c.body > 0 and c.close_location > 0.6 and not efficient:
        reasons.append("up close but path needed too much churn")
        return "Weak Bullish Push", reasons

    if c.body < 0 and c.close_location < 0.4 and not efficient:
        reasons.append("down close but path needed too much churn")
        return "Weak Bearish Push", reasons

    if upper_reject and c.high_first is not False:
        reasons.append("upper extreme printed and price backed away")
        return "Upside Sweep Rejection", reasons

    if lower_reclaim and c.low_first is not False:
        reasons.append("lower extreme printed and price reclaimed")
        return "Downside Sweep Reclaim", reasons

    if upper_reject:
        reasons.append("supply absorbed the upper wick")
        return "Upper Absorption", reasons

    if lower_reclaim:
        reasons.append("demand absorbed the lower wick")
        return "Lower Absorption", reasons

    if c.failed_high_hold or c.failed_low_hold:
        reasons.append("extreme failed to hold through the candle close")
        return "Failed Continuation", reasons

    if churny and c.acceptance_proxy < 0.45:
        reasons.append("path is noisy and acceptance is weak")
        return "Trap Risk", reasons

    reasons.append("features do not show a decisive auction state")
    return "No-Trade Churn", reasons


def next_state_possibilities(candle: CandleFeatures, recent: list[CandleFeatures]) -> list[NextState]:
    recent_labels = [c.label for c in recent[-8:]]
    bull_memory = sum(label in {"Clean Bullish Impulse", "Weak Bullish Push", "Downside Sweep Reclaim"} for label in recent_labels)
    bear_memory = sum(label in {"Clean Bearish Impulse", "Weak Bearish Push", "Upside Sweep Rejection"} for label in recent_labels)
    churn_memory = sum(label in {"Two-Sided Fight", "No-Trade Churn", "Dead Auction", "Premium Compression"} for label in recent_labels)

    raw = {
        "acceptance_continues_up": 0.35 + 0.18 * max(candle.urgency_net, 0.0) + 0.05 * bull_memory,
        "acceptance_continues_down": 0.35 + 0.18 * max(-candle.urgency_net, 0.0) + 0.05 * bear_memory,
        "mean_reversion_or_absorption": 0.25 + 0.08 * candle.churn_ratio + 0.05 * churn_memory,
        "auction_stays_balanced": 0.22 + 0.08 * (1.0 - min(candle.path_efficiency, 1.0)),
    }
    if candle.label in {"Upside Sweep Rejection", "Upper Absorption"}:
        raw["mean_reversion_or_absorption"] += 0.35
    if candle.label in {"Downside Sweep Reclaim", "Lower Absorption"}:
        raw["mean_reversion_or_absorption"] += 0.35
    if candle.label == "Premium Compression":
        raw["auction_stays_balanced"] += 0.45
    if candle.label == "Clean Bullish Impulse":
        raw["acceptance_continues_up"] += 0.45
    if candle.label == "Clean Bearish Impulse":
        raw["acceptance_continues_down"] += 0.45

    total = sum(max(v, 1e-9) for v in raw.values())
    return [
        NextState(name=name, probability=max(value, 1e-9) / total, reason=_next_reason(name, candle))
        for name, value in sorted(raw.items(), key=lambda item: item[1], reverse=True)
    ]


def _next_reason(name: str, candle: CandleFeatures) -> str:
    if name == "acceptance_continues_up":
        return f"positive urgency {candle.urgency_net:+.2f} and close location {candle.close_location:.2f}"
    if name == "acceptance_continues_down":
        return f"negative urgency {candle.urgency_net:+.2f} and close location {candle.close_location:.2f}"
    if name == "mean_reversion_or_absorption":
        return f"wick rejection/reclaim and churn ratio {candle.churn_ratio:.2f}"
    return f"path efficiency {candle.path_efficiency:.2f} keeps the auction unresolved"
