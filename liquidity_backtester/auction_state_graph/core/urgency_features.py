"""Urgency features from path and optional depth."""

from __future__ import annotations

from .models import Tick


def urgency_from_ticks(ticks: list[Tick]) -> tuple[float, float, float, bool]:
    """Return U+, U-, U-net, depth_available.

    The function degrades gracefully without depth. With depth, it uses a small
    imbalance adjustment. Without depth, it uses directional tick pressure and
    short-step velocity, which is available in replay mode.
    """
    if len(ticks) < 2:
        return 0.0, 0.0, 0.0, False

    up = 0.0
    down = 0.0
    depth_seen = False
    for prev, cur in zip(ticks, ticks[1:]):
        delta = cur.price - prev.price
        seconds = max((cur.ts - prev.ts).total_seconds(), 1e-3)
        velocity = abs(delta) / seconds
        depth_bonus = 1.0
        if cur.bid_qty is not None and cur.ask_qty is not None:
            total = max(float(cur.bid_qty) + float(cur.ask_qty), 1e-9)
            imbalance = (float(cur.bid_qty) - float(cur.ask_qty)) / total
            depth_seen = True
            if delta > 0:
                depth_bonus += max(imbalance, 0.0)
            elif delta < 0:
                depth_bonus += max(-imbalance, 0.0)
        impulse = abs(delta) * (1.0 + velocity) * depth_bonus
        if delta > 0:
            up += impulse
        elif delta < 0:
            down += impulse

    denom = max(up + down, 1e-9)
    u_plus = up / denom
    u_minus = down / denom
    return u_plus, u_minus, u_plus - u_minus, depth_seen
