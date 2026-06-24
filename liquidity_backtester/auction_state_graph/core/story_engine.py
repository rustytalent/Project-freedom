"""Measured story layer for candle path intelligence."""

from __future__ import annotations

from .models import CandleFeatures, Story


def story_for_candle(candle: CandleFeatures, recent: list[CandleFeatures]) -> Story:
    bullets: list[str] = []
    warnings: list[str] = []

    bullets.append(
        f"Path efficiency {candle.path_efficiency:.2f}; churn ratio {candle.churn_ratio:.2f}; "
        f"close location {candle.close_location:.2f} inside the candle."
    )
    if candle.high_first is True:
        bullets.append(
            f"High printed before low; upper rejection distance is {candle.upper_rejection_distance:.2f}."
        )
    elif candle.low_first is True:
        bullets.append(
            f"Low printed before high; lower reclaim distance is {candle.lower_reclaim_distance:.2f}."
        )
    else:
        bullets.append("High and low timing did not give a clean first-extreme clue.")

    if candle.depth_available:
        bullets.append(f"Depth-adjusted urgency net is {candle.urgency_net:+.2f}.")
    else:
        bullets.append(f"Replay urgency net is {candle.urgency_net:+.2f}; depth was unavailable.")

    if candle.premium_bias is not None:
        bullets.append(
            f"CE/PE premium bias is {candle.premium_bias:+.2f}; "
            f"compression is {candle.premium_compression:+.2%}."
        )
    else:
        bullets.append("CE/PE premium columns were absent, so premium behavior is not scored.")

    if candle.tick_count < 3:
        warnings.append("Very few ticks inside this candle; path label has low evidence.")
    if candle.label in {"Trap Risk", "Two-Sided Fight", "No-Trade Churn"}:
        warnings.append("Internal movement is noisy; treat the candle as an information state, not a clean directional read.")

    recent_labels = [c.label for c in recent[-5:]]
    if recent_labels:
        bullets.append("Recent state sequence: " + " -> ".join(recent_labels))
    return Story(title=candle.label, bullets=bullets, warnings=warnings)
