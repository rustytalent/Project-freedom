"""LocalHighTracker — peak detection + revisit probability.

The founder's "sine theta wave" intuition: option premium pulses
through local highs and lows. The question that matters is NOT
"will it revisit?" but "will the next revisit be high enough, before
the thesis decays?"

This module:

  1. Detects local extremes (peaks + valleys) using a window-based
     detector — NOT naive max-in-window, which produces ghost peaks
     on every new high. We require a confirmed turn (k bars below
     the local high) before crowning it.
  2. Tracks the trajectory of recent local highs: rising / flat /
     falling. The founder's exact words: "if highs are rising → hold
     target; flattening → prepare to shade; falling → reduce
     aggressively."
  3. Estimates revisit probability for an arbitrary target price
     using the recent revisit frequency around comparable price
     bands.

The output ``RevisitForecast`` is the input to the modification gate:
"is the next favourable revisit alive enough to keep waiting for ₹10
profit, or should we shade down to ₹7?"
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class LocalHighTrackerConfig:
    """Knobs for peak detection + revisit forecasting."""
    # History window
    history_bars: int = 120
    # Peak detection (confirmed turn-down required)
    confirmation_bars: int = 2     # need k bars below high to confirm peak
    min_peak_spacing_bars: int = 3 # don't crown peaks too close together
    # Revisit forecasting
    revisit_lookback_bars: int = 60
    revisit_band_pct: float = 0.005   # ±0.5% band counts as "same level"
    # Trajectory analysis
    trajectory_window_peaks: int = 4  # look at last N peaks for trajectory


# ── Data shapes ───────────────────────────────────────────────────


@dataclass
class LocalExtreme:
    """One detected local peak or valley."""
    bar_index: int
    price: float
    kind: str        # "peak" / "valley"
    confirmed_at_bar: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RevisitForecast:
    """Per-target forecast of revisit dynamics."""
    target_price: float
    revisit_probability: float       # 0..1 chance of reaching target soon
    expected_bars_to_revisit: float
    confidence: float                # 0..1, scales with sample size
    # Trajectory signals
    high_trajectory: str             # "rising" / "flat" / "falling" / "insufficient"
    low_trajectory: str
    recent_peak_price: Optional[float]
    recent_peak_bar: Optional[int]
    bars_since_last_peak: Optional[int]
    # Concrete metrics
    peak_count_in_window: int
    revisits_to_band_in_window: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_price": round(self.target_price, 4),
            "revisit_probability": round(self.revisit_probability, 3),
            "expected_bars_to_revisit": round(self.expected_bars_to_revisit, 1),
            "confidence": round(self.confidence, 3),
            "high_trajectory": self.high_trajectory,
            "low_trajectory": self.low_trajectory,
            "recent_peak_price": (round(self.recent_peak_price, 4)
                                    if self.recent_peak_price is not None else None),
            "recent_peak_bar": self.recent_peak_bar,
            "bars_since_last_peak": self.bars_since_last_peak,
            "peak_count_in_window": self.peak_count_in_window,
            "revisits_to_band_in_window": self.revisits_to_band_in_window,
            "notes": list(self.notes),
        }


# ── Tracker ────────────────────────────────────────────────────────


class LocalHighTracker:
    """Stateful peak/valley detector with revisit forecasting.

    Call ``observe(bar_index, price)`` once per bar. Then query
    ``forecast_revisit(target_price)`` whenever the exit engine needs
    to make a decision.

    The detector uses a confirmed-turn rule: a local high is crowned
    only after we see ``confirmation_bars`` ticks at lower prices.
    This eliminates the ghost peaks that naive max-in-window produces
    on every new high.
    """

    def __init__(self,
                  cfg: Optional[LocalHighTrackerConfig] = None) -> None:
        self.cfg = cfg or LocalHighTrackerConfig()
        self._history: Deque[Tuple[int, float]] = deque(
            maxlen=self.cfg.history_bars)
        self._peaks: Deque[LocalExtreme] = deque(maxlen=20)
        self._valleys: Deque[LocalExtreme] = deque(maxlen=20)
        # Working state for confirmed-turn detection
        self._pending_peak: Optional[Tuple[int, float]] = None
        self._pending_valley: Optional[Tuple[int, float]] = None
        self._confirm_counter_peak = 0
        self._confirm_counter_valley = 0

    def observe(self, bar_index: int, price: float) -> None:
        """Ingest one bar's premium."""
        cfg = self.cfg
        if not (math.isfinite(price) and price > 0):
            return
        self._history.append((bar_index, price))
        if len(self._history) < 3:
            return

        # Look at the last 3 samples (a, b, c).
        (a_bar, a_price), (b_bar, b_price), (c_bar, c_price) = (
            self._history[-3], self._history[-2], self._history[-1])

        # Tentative peak: b is local max (a < b > c)
        if a_price < b_price > c_price:
            # Start (or update) pending peak.
            if (self._pending_peak is None
                    or b_price > self._pending_peak[1]):
                self._pending_peak = (b_bar, b_price)
                self._confirm_counter_peak = 0

        # Tentative valley: b is local min (a > b < c)
        if a_price > b_price < c_price:
            if (self._pending_valley is None
                    or b_price < self._pending_valley[1]):
                self._pending_valley = (b_bar, b_price)
                self._confirm_counter_valley = 0

        # Confirmation: if a pending peak exists and current price is
        # below it, increment confirmation counter.
        if self._pending_peak is not None:
            if c_price < self._pending_peak[1]:
                self._confirm_counter_peak += 1
                if self._confirm_counter_peak >= cfg.confirmation_bars:
                    self._crown(LocalExtreme(
                        bar_index=self._pending_peak[0],
                        price=self._pending_peak[1],
                        kind="peak",
                        confirmed_at_bar=c_bar,
                    ))
                    self._pending_peak = None
                    self._confirm_counter_peak = 0
            else:
                # Price exceeded pending peak — replace it.
                if c_price > self._pending_peak[1]:
                    self._pending_peak = (c_bar, c_price)
                    self._confirm_counter_peak = 0

        if self._pending_valley is not None:
            if c_price > self._pending_valley[1]:
                self._confirm_counter_valley += 1
                if self._confirm_counter_valley >= cfg.confirmation_bars:
                    self._crown(LocalExtreme(
                        bar_index=self._pending_valley[0],
                        price=self._pending_valley[1],
                        kind="valley",
                        confirmed_at_bar=c_bar,
                    ))
                    self._pending_valley = None
                    self._confirm_counter_valley = 0
            else:
                if c_price < self._pending_valley[1]:
                    self._pending_valley = (c_bar, c_price)
                    self._confirm_counter_valley = 0

    def _crown(self, extreme: LocalExtreme) -> None:
        cfg = self.cfg
        target_deque = (self._peaks if extreme.kind == "peak"
                        else self._valleys)
        # Spacing check
        if target_deque and (extreme.bar_index
                              - target_deque[-1].bar_index
                              < cfg.min_peak_spacing_bars):
            return
        target_deque.append(extreme)

    # ── Queries ─────────────────────────────────────────────────────

    def recent_peaks(self, n: int = 5) -> List[LocalExtreme]:
        return list(self._peaks)[-n:]

    def recent_valleys(self, n: int = 5) -> List[LocalExtreme]:
        return list(self._valleys)[-n:]

    def last_peak(self) -> Optional[LocalExtreme]:
        return self._peaks[-1] if self._peaks else None

    def last_valley(self) -> Optional[LocalExtreme]:
        return self._valleys[-1] if self._valleys else None

    def latest_price(self) -> Optional[float]:
        return self._history[-1][1] if self._history else None

    def latest_bar(self) -> Optional[int]:
        return self._history[-1][0] if self._history else None

    # ── Forecasting ────────────────────────────────────────────────

    def forecast_revisit(self, target_price: float) -> RevisitForecast:
        """Estimate whether the price is likely to revisit (or reach)
        the given target, and how soon."""
        cfg = self.cfg
        notes: List[str] = []
        if not self._history:
            return _empty_forecast(target_price, notes=["no history"])

        latest = self._history[-1]
        recent = [(b, p) for (b, p) in self._history
                   if latest[0] - b <= cfg.revisit_lookback_bars]
        if len(recent) < 5:
            return _empty_forecast(target_price,
                                     notes=["insufficient history"])

        # Compute trajectory of last N peaks / valleys.
        high_traj = self._trajectory(self._peaks, cfg.trajectory_window_peaks)
        low_traj = self._trajectory(self._valleys, cfg.trajectory_window_peaks)

        # Count revisits to the band around the target.
        band = abs(target_price) * cfg.revisit_band_pct
        revisits = sum(1 for (_, p) in recent if abs(p - target_price) <= band)

        # Peak frequency
        peaks_in_window = [pk for pk in self._peaks
                            if latest[0] - pk.bar_index
                            <= cfg.revisit_lookback_bars]
        n_peaks = len(peaks_in_window)
        if n_peaks >= 2:
            avg_peak_gap = (
                (peaks_in_window[-1].bar_index - peaks_in_window[0].bar_index)
                / max(1, n_peaks - 1))
        else:
            avg_peak_gap = float(cfg.revisit_lookback_bars)

        # Base probability from revisit frequency.
        base_prob = min(0.95, revisits / max(5, len(recent) // 4))

        # Adjust by trajectory
        if target_price >= latest[1]:
            # Need price to rise to reach target.
            if high_traj == "rising":
                base_prob = min(0.95, base_prob + 0.30)
            elif high_traj == "falling":
                base_prob *= 0.40
            # If the recent peak is already above target, revisit is likely.
            last_peak = self.last_peak()
            if last_peak is not None and last_peak.price >= target_price:
                base_prob = max(base_prob, 0.50)
        else:
            # Target is below current — easy to reach if we just need a dip.
            if low_traj == "falling":
                base_prob = min(0.95, base_prob + 0.30)

        # Expected bars to revisit ≈ avg_peak_gap, adjusted by base_prob.
        expected_bars = avg_peak_gap if base_prob >= 0.30 else avg_peak_gap * 2

        # Confidence scales with sample size.
        confidence = min(1.0, math.sqrt(len(recent) / 60.0))

        last_peak = self.last_peak()
        bars_since = (latest[0] - last_peak.bar_index
                       if last_peak is not None else None)

        return RevisitForecast(
            target_price=target_price,
            revisit_probability=max(0.0, min(1.0, base_prob)),
            expected_bars_to_revisit=expected_bars,
            confidence=confidence,
            high_trajectory=high_traj,
            low_trajectory=low_traj,
            recent_peak_price=(last_peak.price if last_peak else None),
            recent_peak_bar=(last_peak.bar_index if last_peak else None),
            bars_since_last_peak=bars_since,
            peak_count_in_window=n_peaks,
            revisits_to_band_in_window=revisits,
            notes=notes,
        )

    def _trajectory(self, deque_obj: Deque[LocalExtreme],
                      window: int) -> str:
        """Classify trajectory of recent peaks/valleys."""
        recent = list(deque_obj)[-window:]
        if len(recent) < 3:
            return "insufficient"
        prices = [e.price for e in recent]
        # Linear-fit slope
        n = len(prices)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(prices) / n
        num = sum((xs[i] - mean_x) * (prices[i] - mean_y)
                   for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n))
        if den == 0:
            return "flat"
        slope = num / den
        # Normalize by mean price
        slope_pct = slope / max(1e-6, mean_y)
        if slope_pct > 0.005:
            return "rising"
        if slope_pct < -0.005:
            return "falling"
        return "flat"


def _empty_forecast(target: float, notes: List[str]) -> RevisitForecast:
    return RevisitForecast(
        target_price=target,
        revisit_probability=0.50,    # neutral prior
        expected_bars_to_revisit=12.0,
        confidence=0.0,
        high_trajectory="insufficient",
        low_trajectory="insufficient",
        recent_peak_price=None,
        recent_peak_bar=None,
        bars_since_last_peak=None,
        peak_count_in_window=0,
        revisits_to_band_in_window=0,
        notes=list(notes),
    )
