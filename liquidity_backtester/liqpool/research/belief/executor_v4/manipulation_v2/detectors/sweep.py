"""SweepDetector — distinguishes stop-hunt sweeps from real directional flow.

A sweep is a fast price burst over a few ticks. It can be:
  * **stop_hunt** — burst followed by a reversal within K bars, *without*
    sustained net-intent confirmation. MM is gunning retail stops to
    flip positioning.
  * **real_flow** — burst confirmed by sustained net-intent direction
    AND lack of immediate reversal. Genuine directional pressure.

The classifier uses three signals:
  1. The engine's ``sweep_state.state_index`` (substrate already detects
     the raw burst).
  2. The ``iv_state.net_intent_z`` magnitude + direction — real flow
     keeps pushing; stop-hunts release.
  3. The post-sweep follow-through over a short rolling window — does
     spot keep moving the same way, or does it snap back?
"""
from __future__ import annotations

import statistics
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class SweepDetector(DetectorBase):
    name = "sweep"

    def __init__(self, *,
                  follow_through_window: int = 6,
                  reversal_within_bars: int = 4,
                  intent_z_floor: float = 1.5,
                  ) -> None:
        self.follow_through_window = int(follow_through_window)
        self.reversal_within_bars = int(reversal_within_bars)
        self.intent_z_floor = float(intent_z_floor)
        self._spot_history: Deque[Tuple[int, float]] = deque(
            maxlen=max(64, self.follow_through_window * 4))
        self._sweep_history: Deque[Tuple[int, int, float]] = deque(maxlen=32)

    def reset(self) -> None:
        self._spot_history.clear()
        self._sweep_history.clear()

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        spot = float(snapshot.get("spot") or 0.0)
        self._spot_history.append((bar_index, spot))

        sweep_state = _map(snapshot.get("sweep_state"))
        sweep_idx = int(sweep_state.get("state_index", 0) or 0)
        iv = _map(snapshot.get("iv_state"))
        net_intent_z = float(iv.get("net_intent_z") or 0.0)
        intent_direction = int(iv.get("direction") or 0)

        # The substrate's sweep_state.state_index is non-zero when a
        # burst has been detected within the last few bars. We snapshot
        # the burst and its direction here for the post-window classifier.
        if sweep_idx != 0:
            burst_direction = (
                int(_sign(intent_direction)) if intent_direction != 0
                else int(_sign(net_intent_z)))
            self._sweep_history.append(
                (bar_index, burst_direction, spot))

        # If we have a recent burst within the reversal window, classify
        # it. Otherwise the detector is quiet.
        if not self._sweep_history:
            return DetectorPosterior.quiet()
        recent_bar, recent_dir, recent_spot = self._sweep_history[-1]
        bars_since = bar_index - recent_bar
        if recent_dir == 0:
            return DetectorPosterior.quiet()
        if bars_since > self.follow_through_window:
            return DetectorPosterior.quiet()

        # Did spot reverse since the burst?
        reversed_now = False
        if bars_since <= self.reversal_within_bars:
            reversed_now = (
                recent_dir > 0 and spot < recent_spot - 0.1
            ) or (
                recent_dir < 0 and spot > recent_spot + 0.1
            )

        sustained_intent = (abs(net_intent_z) >= self.intent_z_floor
                            and int(_sign(net_intent_z)) == recent_dir)

        # Spot displacement since burst (in points).
        displacement = spot - recent_spot
        same_dir_displacement = displacement * recent_dir

        evidence = []
        if reversed_now and not sustained_intent:
            classification = "stop_hunt"
            # MM intent is OPPOSITE the burst — the real intent surfaces
            # in the reversal.
            direction = -recent_dir
            base_prob = 0.65
            if abs(net_intent_z) >= self.intent_z_floor and \
                    int(_sign(net_intent_z)) == -recent_dir:
                base_prob = 0.78
            evidence.append(
                f"sweep burst dir={recent_dir} reversed within "
                f"{bars_since}b without intent follow-through")
        elif sustained_intent and not reversed_now:
            classification = "real_flow"
            direction = recent_dir
            base_prob = 0.62 + min(0.18, 0.06 * (
                abs(net_intent_z) - self.intent_z_floor))
            evidence.append(
                f"sweep burst dir={recent_dir} confirmed by "
                f"intent_z={net_intent_z:+.2f}; spot moved "
                f"{same_dir_displacement:+.1f}pts in dir")
        elif same_dir_displacement > 0 and abs(net_intent_z) < self.intent_z_floor:
            # Still extending without intent confirmation — ambiguous,
            # lean toward stop-hunt but lower confidence.
            classification = "ambiguous_extension"
            direction = -recent_dir
            base_prob = 0.40
            evidence.append(
                "sweep extended but intent_z is weak — likely "
                "incomplete stop-hunt")
        else:
            return DetectorPosterior.quiet()

        # Magnitude: how big was the burst in standardised displacement?
        if len(self._spot_history) >= 20:
            recent_returns = []
            prev = None
            for _, s in list(self._spot_history)[-20:]:
                if prev is not None and prev > 0:
                    recent_returns.append((s - prev) / prev)
                prev = s
            if recent_returns:
                std = statistics.pstdev(recent_returns) or 1e-6
                # Magnitude of the burst itself.
                burst_return = (
                    abs(displacement) / max(1e-6, recent_spot))
                z = burst_return / std
            else:
                z = 0.0
        else:
            z = 0.0

        magnitude = DetectorMagnitude(
            z_score=z, raw_value=abs(displacement), units="points")
        confidence = max(0.1, min(0.95, base_prob))
        horizon = (3 if classification == "stop_hunt"
                   else 8 if classification == "real_flow"
                   else 4)
        return DetectorPosterior(
            fire=True, probability=base_prob, direction=direction,
            confidence=confidence, horizon_bars=horizon,
            magnitude=magnitude, evidence=evidence,
            classification=classification,
        )


def _map(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
