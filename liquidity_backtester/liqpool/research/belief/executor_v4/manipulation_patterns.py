"""Manipulation pattern catalogue — the demonic-market reader.

The founder's mandate (2026-06-19): *"It should understand the
manipulation. It should thrive in this demonic market. We get rewarded
for reading the manipulation."*

This module is a catalogue of ten pattern detectors. Each runs against
the rolling flow_memory + substrate trajectories and emits a
``PatternMatch`` whenever its signature fires. The aggregator queries
``ManipulationPatternBoard.active_patterns(...)`` once per tick.

Catalogue (founder's spec):
  1. **stop_hunt_short**   — wick above HOD → fast revert below
  2. **stop_hunt_long**    — wick below LOD → fast revert above
  3. **liquidity_vacuum**  — sudden spread blowout + bid disappear
  4. **pin_near_expiry**   — coiling within tight band of strike
  5. **accumulation**      — sustained absorbing, narrow range, CE acc up
  6. **distribution**      — sustained selling masked by churn, PE acc up
  7. **squeeze_setup**     — narrowing range + decreasing realized vol
  8. **fake_breakout**     — break out → fail to follow within 5 bars
  9. **iceberg_buy**       — repeated absorption at same level
 10. **mm_inventory_flip** — rail signed-z reverses cleanly with no spot move

Each detector is INTENTIONALLY conservative — false positives erode
trust. The scoring uses sample sizes from the flow_memory event tape;
patterns with weak evidence return low confidence (<0.50) and the
critic / aggregator filter them out automatically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .flow_memory import FlowEvent, FlowMemory


# Pattern names — exported for the MM-mind and aggregator.
PATTERN_STOP_HUNT_SHORT = "stop_hunt_short"
PATTERN_STOP_HUNT_LONG = "stop_hunt_long"
PATTERN_LIQUIDITY_VACUUM = "liquidity_vacuum"
PATTERN_PIN_NEAR_EXPIRY = "pin_near_expiry"
PATTERN_ACCUMULATION = "accumulation"
PATTERN_DISTRIBUTION = "distribution"
PATTERN_SQUEEZE_SETUP = "squeeze_setup"
PATTERN_FAKE_BREAKOUT = "fake_breakout"
PATTERN_ICEBERG_BUY = "iceberg_buy"
PATTERN_MM_INVENTORY_FLIP = "mm_inventory_flip"

# Implied MM intent labels (consumed by market_maker_mind.py).
INTENT_HUNTING_STOPS = "hunting_stops"
INTENT_PINNING = "pinning_near_expiry"
INTENT_ACCUMULATING = "accumulating"
INTENT_DISTRIBUTING = "distributing"
INTENT_STEPPING_BACK = "stepping_back"
INTENT_FAKING_DIRECTION = "faking_direction"
INTENT_NEUTRAL = "neutral_inventory"


@dataclass
class PatternMatch:
    """One detected manipulation pattern."""
    pattern_name: str
    confidence: float                  # 0..1
    started_at_bar: int
    expected_resolution_bars: int      # how many bars until pattern resolves
    implied_mm_intent: str             # one of INTENT_*
    suggested_trade: str               # e.g. "fade_the_wick" / "wait"
    suggested_avoid: str               # what NOT to do
    evidence: List[str]                # human-readable reasons
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "confidence": round(self.confidence, 3),
            "started_at_bar": self.started_at_bar,
            "expected_resolution_bars": self.expected_resolution_bars,
            "implied_mm_intent": self.implied_mm_intent,
            "suggested_trade": self.suggested_trade,
            "suggested_avoid": self.suggested_avoid,
            "evidence": list(self.evidence),
            "notes": list(self.notes),
        }


@dataclass
class ManipulationBoardConfig:
    """Knobs for the detector board."""
    # Stop-hunt
    stop_hunt_spot_bps: float = 6.0        # min wick size in bps
    stop_hunt_revert_bars: int = 6          # must revert within
    # Liquidity vacuum
    vacuum_friend_drop: float = 0.25        # avg_friendliness drop
    # Pin (near expiry)
    pin_band_pct: float = 0.0015            # 0.15% of spot
    pin_window_bars: int = 15
    pin_min_bars: int = 8
    # Accumulation / Distribution
    acc_def_floor: float = 0.40             # acceptance fraction
    acc_window: int = 15
    # Squeeze
    squeeze_window: int = 25
    squeeze_range_shrink: float = 0.55      # latest range ≤ this × prior range
    # Fake breakout
    fake_breakout_window: int = 8
    fake_breakout_size_bps: float = 8.0
    # Iceberg buy
    iceberg_window: int = 20
    iceberg_min_occurrences: int = 3
    # MM inventory flip
    mm_flip_window: int = 8
    mm_flip_intent_swing: float = 1.8
    mm_flip_spot_max_bps: float = 4.0


class ManipulationBoard:
    """The detector board — runs all detectors against flow_memory.

    Stateless (apart from cfg): each call to ``scan(...)`` returns the
    currently active pattern matches as a list. Recency / decay are
    handled implicitly: each detector only looks at the most recent
    ``window`` events.
    """

    def __init__(self, cfg: Optional[ManipulationBoardConfig] = None) -> None:
        self.cfg = cfg or ManipulationBoardConfig()

    def scan(self, *,
              flow_memory: FlowMemory,
              rich_context: Any,
              snapshot: Dict[str, Any],
              near_expiry: bool = False,
              ) -> List[PatternMatch]:
        """Run every detector and return the firing pattern matches."""
        events = list(flow_memory.events)
        if not events:
            return []
        matches: List[PatternMatch] = []
        for detector in (
            self._detect_stop_hunt_short,
            self._detect_stop_hunt_long,
            self._detect_liquidity_vacuum,
            self._detect_pin_near_expiry,
            self._detect_accumulation,
            self._detect_distribution,
            self._detect_squeeze_setup,
            self._detect_fake_breakout,
            self._detect_iceberg_buy,
            self._detect_mm_inventory_flip,
        ):
            try:
                m = detector(events, rich_context, snapshot, near_expiry)
            except Exception:
                m = None
            if m is not None and m.confidence >= 0.30:
                matches.append(m)
        return matches

    # ── individual detectors ────────────────────────────────────────────

    def _detect_stop_hunt_short(self, events: List[FlowEvent],
                                  rich: Any, snapshot: Dict[str, Any],
                                  near_expiry: bool) -> Optional[PatternMatch]:
        """Wick UP above the recent high, then a fast revert BELOW the
        starting spot within ``stop_hunt_revert_bars`` events."""
        cfg = self.cfg
        if len(events) < 8:
            return None
        recent = events[-30:]
        # Walk every event; for each, check that revert happens within window.
        for i, ev in enumerate(recent):
            if ev.spot_pct_change < cfg.stop_hunt_spot_bps:
                continue
            prior_window = recent[max(0, i - 8):i]
            if not prior_window:
                continue
            prior_high = max(p.spot for p in prior_window)
            if ev.spot < prior_high:
                continue
            start_spot = ev.spot - ev.spot_pct_change * ev.spot / 10000.0
            end = min(len(recent), i + 1 + cfg.stop_hunt_revert_bars)
            for j in range(i + 1, end):
                if recent[j].spot < start_spot:
                    conf = min(1.0, 0.55
                                + 0.05 * (ev.spot_pct_change
                                          / cfg.stop_hunt_spot_bps))
                    return PatternMatch(
                        pattern_name=PATTERN_STOP_HUNT_SHORT,
                        confidence=conf,
                        started_at_bar=ev.bar_index,
                        expected_resolution_bars=cfg.stop_hunt_revert_bars,
                        implied_mm_intent=INTENT_HUNTING_STOPS,
                        suggested_trade="wait_for_stabilize_then_fade_or_join_revert",
                        suggested_avoid="chasing the up-spike",
                        evidence=[
                            f"+{ev.spot_pct_change:.1f}bps spike at "
                            f"bar {ev.bar_index} above prior high",
                            f"reverted below start spot within "
                            f"{j - i} bars",
                        ],
                    )
        return None

    def _detect_stop_hunt_long(self, events: List[FlowEvent],
                                 rich: Any, snapshot: Dict[str, Any],
                                 near_expiry: bool) -> Optional[PatternMatch]:
        cfg = self.cfg
        if len(events) < 8:
            return None
        recent = events[-30:]
        for i, ev in enumerate(recent):
            if ev.spot_pct_change > -cfg.stop_hunt_spot_bps:
                continue
            prior_window = recent[max(0, i - 8):i]
            if not prior_window:
                continue
            prior_low = min(p.spot for p in prior_window)
            if ev.spot > prior_low:
                continue
            start_spot = ev.spot - ev.spot_pct_change * ev.spot / 10000.0
            end = min(len(recent), i + 1 + cfg.stop_hunt_revert_bars)
            for j in range(i + 1, end):
                if recent[j].spot > start_spot:
                    conf = min(1.0, 0.55
                                + 0.05 * (abs(ev.spot_pct_change)
                                          / cfg.stop_hunt_spot_bps))
                    return PatternMatch(
                        pattern_name=PATTERN_STOP_HUNT_LONG,
                        confidence=conf,
                        started_at_bar=ev.bar_index,
                        expected_resolution_bars=cfg.stop_hunt_revert_bars,
                        implied_mm_intent=INTENT_HUNTING_STOPS,
                        suggested_trade="wait_for_stabilize_then_join_revert",
                        suggested_avoid="chasing the down-spike",
                        evidence=[
                            f"{ev.spot_pct_change:.1f}bps spike at bar "
                            f"{ev.bar_index} below prior low",
                            f"reverted above start spot within {j - i} bars",
                        ],
                    )
        return None

    def _detect_liquidity_vacuum(self, events: List[FlowEvent],
                                   rich: Any, snapshot: Dict[str, Any],
                                   near_expiry: bool) -> Optional[PatternMatch]:
        """Sudden drop in avg_friendliness + drop in clean_mark_fraction."""
        cfg = self.cfg
        if len(events) < 6:
            return None
        recent = events[-6:]
        baseline = sum(e.avg_friendliness for e in recent[:3]) / 3
        latest = sum(e.avg_friendliness for e in recent[-3:]) / 3
        drop = baseline - latest
        clean_drop = (recent[0].clean_mark_fraction
                       - recent[-1].clean_mark_fraction)
        if drop < cfg.vacuum_friend_drop and clean_drop < 0.10:
            return None
        conf = min(1.0, 0.55 + drop)
        return PatternMatch(
            pattern_name=PATTERN_LIQUIDITY_VACUUM,
            confidence=conf,
            started_at_bar=recent[-1].bar_index,
            expected_resolution_bars=6,
            implied_mm_intent=INTENT_STEPPING_BACK,
            suggested_trade="wait_or_use_limit_orders",
            suggested_avoid="market orders, aggressive sizing",
            evidence=[
                f"avg_friendliness dropped {drop:+.2f} (baseline "
                f"{baseline:.2f} → latest {latest:.2f})",
                f"clean_mark_fraction dropped {clean_drop:+.2f}",
            ],
        )

    def _detect_pin_near_expiry(self, events: List[FlowEvent],
                                  rich: Any, snapshot: Dict[str, Any],
                                  near_expiry: bool) -> Optional[PatternMatch]:
        """Coiling within a tight band of a round-number strike during
        the last hour of expiry."""
        cfg = self.cfg
        if not near_expiry:
            return None
        if len(events) < cfg.pin_window_bars:
            return None
        window = events[-cfg.pin_window_bars:]
        spots = [e.spot for e in window]
        spot_range = max(spots) - min(spots)
        mid = (max(spots) + min(spots)) / 2
        band = cfg.pin_band_pct * mid
        if spot_range > band:
            return None
        # Find nearest round 50 strike.
        nearest_strike = round(mid / 50) * 50
        if abs(mid - nearest_strike) > band:
            return None
        n_in_band = sum(1 for s in spots if abs(s - nearest_strike) <= band)
        if n_in_band < cfg.pin_min_bars:
            return None
        conf = min(1.0, 0.50 + 0.40 * (n_in_band / len(window)))
        return PatternMatch(
            pattern_name=PATTERN_PIN_NEAR_EXPIRY,
            confidence=conf,
            started_at_bar=window[0].bar_index,
            expected_resolution_bars=cfg.pin_window_bars,
            implied_mm_intent=INTENT_PINNING,
            suggested_trade="butterfly_or_iron_condor_at_strike",
            suggested_avoid="directional buy of ATM CE/PE",
            evidence=[
                f"spot coiling in ₹{spot_range:.1f} band near ₹{nearest_strike:.0f}",
                f"{n_in_band}/{len(window)} bars within band",
            ],
        )

    def _detect_accumulation(self, events: List[FlowEvent],
                              rich: Any, snapshot: Dict[str, Any],
                              near_expiry: bool) -> Optional[PatternMatch]:
        """Sustained CE acceptance + narrow spot range — bullish absorption."""
        cfg = self.cfg
        if len(events) < cfg.acc_window:
            return None
        window = events[-cfg.acc_window:]
        avg_ce_def = sum(e.ce_defended_fraction for e in window) / len(window)
        if avg_ce_def < cfg.acc_def_floor:
            return None
        spots = [e.spot for e in window]
        spot_range = max(spots) - min(spots)
        spot_range_pct = spot_range / max(1.0, spots[-1]) * 10000  # bps
        if spot_range_pct > 30.0:   # not narrow enough
            return None
        conf = min(1.0, 0.55 + avg_ce_def / 2.0)
        return PatternMatch(
            pattern_name=PATTERN_ACCUMULATION,
            confidence=conf,
            started_at_bar=window[0].bar_index,
            expected_resolution_bars=20,
            implied_mm_intent=INTENT_ACCUMULATING,
            suggested_trade="patient_long_bias",
            suggested_avoid="shorting into absorption",
            evidence=[
                f"CE defended avg {avg_ce_def:.0%} over {len(window)} bars",
                f"spot range {spot_range_pct:.1f}bps — narrow",
            ],
        )

    def _detect_distribution(self, events: List[FlowEvent],
                              rich: Any, snapshot: Dict[str, Any],
                              near_expiry: bool) -> Optional[PatternMatch]:
        cfg = self.cfg
        if len(events) < cfg.acc_window:
            return None
        window = events[-cfg.acc_window:]
        avg_pe_def = sum(e.pe_defended_fraction for e in window) / len(window)
        if avg_pe_def < cfg.acc_def_floor:
            return None
        spots = [e.spot for e in window]
        spot_range_pct = (max(spots) - min(spots)) / max(1.0, spots[-1]) * 10000
        if spot_range_pct > 30.0:
            return None
        conf = min(1.0, 0.55 + avg_pe_def / 2.0)
        return PatternMatch(
            pattern_name=PATTERN_DISTRIBUTION,
            confidence=conf,
            started_at_bar=window[0].bar_index,
            expected_resolution_bars=20,
            implied_mm_intent=INTENT_DISTRIBUTING,
            suggested_trade="patient_short_bias",
            suggested_avoid="longing into distribution",
            evidence=[
                f"PE defended avg {avg_pe_def:.0%} over {len(window)} bars",
                f"spot range {spot_range_pct:.1f}bps — narrow",
            ],
        )

    def _detect_squeeze_setup(self, events: List[FlowEvent],
                                rich: Any, snapshot: Dict[str, Any],
                                near_expiry: bool) -> Optional[PatternMatch]:
        cfg = self.cfg
        if len(events) < cfg.squeeze_window:
            return None
        window = events[-cfg.squeeze_window:]
        half = len(window) // 2
        early = window[:half]; late = window[half:]
        early_spots = [e.spot for e in early]
        late_spots = [e.spot for e in late]
        early_range = max(early_spots) - min(early_spots)
        late_range = max(late_spots) - min(late_spots)
        if early_range <= 0:
            return None
        ratio = late_range / early_range
        if ratio > cfg.squeeze_range_shrink:
            return None
        conf = min(1.0, 0.50 + (cfg.squeeze_range_shrink - ratio))
        return PatternMatch(
            pattern_name=PATTERN_SQUEEZE_SETUP,
            confidence=conf,
            started_at_bar=window[0].bar_index,
            expected_resolution_bars=cfg.squeeze_window,
            implied_mm_intent=INTENT_NEUTRAL,
            suggested_trade="straddle_or_wait_for_break_direction",
            suggested_avoid="iron_condor",
            evidence=[
                f"range shrunk {ratio:.2f}x (early {early_range:.1f} → "
                f"late {late_range:.1f})",
            ],
        )

    def _detect_fake_breakout(self, events: List[FlowEvent],
                                rich: Any, snapshot: Dict[str, Any],
                                near_expiry: bool) -> Optional[PatternMatch]:
        """Spot makes a >X bps move, then returns through start within W bars."""
        cfg = self.cfg
        if len(events) < 5:
            return None
        recent = events[-(cfg.fake_breakout_window + 5):]
        for i, ev in enumerate(recent):
            if abs(ev.spot_pct_change) < cfg.fake_breakout_size_bps:
                continue
            # Need at least one prior event for "breakout above prior level"
            # sanity.
            if i == 0:
                continue
            direction = 1 if ev.spot_pct_change > 0 else -1
            start_spot = ev.spot - ev.spot_pct_change * ev.spot / 10000.0
            end = min(len(recent), i + 1 + cfg.fake_breakout_window)
            for j in range(i + 1, end):
                if direction > 0 and recent[j].spot <= start_spot:
                    return self._fake_breakout_match(ev, j - i, direction)
                if direction < 0 and recent[j].spot >= start_spot:
                    return self._fake_breakout_match(ev, j - i, direction)
        return None

    def _fake_breakout_match(self, ev: FlowEvent, bars_to_revert: int,
                              direction: int) -> PatternMatch:
        conf = min(1.0, 0.55 + 0.05 * abs(ev.spot_pct_change) / 5.0)
        return PatternMatch(
            pattern_name=PATTERN_FAKE_BREAKOUT,
            confidence=conf,
            started_at_bar=ev.bar_index,
            expected_resolution_bars=bars_to_revert + 5,
            implied_mm_intent=INTENT_FAKING_DIRECTION,
            suggested_trade="fade_the_breakout",
            suggested_avoid=f"chasing the {'up' if direction > 0 else 'down'}-break",
            evidence=[
                f"{ev.spot_pct_change:+.1f}bps move at bar {ev.bar_index}",
                f"failed within {bars_to_revert} bars",
            ],
        )

    def _detect_iceberg_buy(self, events: List[FlowEvent],
                             rich: Any, snapshot: Dict[str, Any],
                             near_expiry: bool) -> Optional[PatternMatch]:
        """Repeated absorption at similar spot level."""
        cfg = self.cfg
        if len(events) < cfg.iceberg_window:
            return None
        window = events[-cfg.iceberg_window:]
        # Spots clustered near a level + CE defended on each visit.
        spots = [e.spot for e in window]
        median_spot = sorted(spots)[len(spots) // 2]
        band = median_spot * 0.0008  # 8bps band
        absorbed_visits = 0
        for e in window:
            if abs(e.spot - median_spot) <= band and e.ce_defended_fraction >= 0.30:
                absorbed_visits += 1
        if absorbed_visits < cfg.iceberg_min_occurrences:
            return None
        conf = min(1.0, 0.50 + 0.10 * absorbed_visits)
        return PatternMatch(
            pattern_name=PATTERN_ICEBERG_BUY,
            confidence=conf,
            started_at_bar=window[0].bar_index,
            expected_resolution_bars=20,
            implied_mm_intent=INTENT_ACCUMULATING,
            suggested_trade="patient_long_bias_above_iceberg",
            suggested_avoid="shorting at the iceberg level",
            evidence=[
                f"{absorbed_visits} absorption visits at ₹{median_spot:.1f}",
            ],
        )

    def _detect_mm_inventory_flip(self, events: List[FlowEvent],
                                    rich: Any, snapshot: Dict[str, Any],
                                    near_expiry: bool) -> Optional[PatternMatch]:
        """Rail signed-z reverses cleanly with no significant spot move —
        signature of MM rebalancing without participating in direction."""
        cfg = self.cfg
        if len(events) < cfg.mm_flip_window:
            return None
        window = events[-cfg.mm_flip_window:]
        net_change = window[-1].net_intent_z - window[0].net_intent_z
        if abs(net_change) < cfg.mm_flip_intent_swing:
            return None
        spot_change_bps = ((window[-1].spot - window[0].spot)
                           / max(0.01, window[0].spot) * 10000.0)
        if abs(spot_change_bps) > cfg.mm_flip_spot_max_bps:
            return None
        conf = min(1.0, 0.50 + 0.10 * abs(net_change))
        return PatternMatch(
            pattern_name=PATTERN_MM_INVENTORY_FLIP,
            confidence=conf,
            started_at_bar=window[0].bar_index,
            expected_resolution_bars=cfg.mm_flip_window,
            implied_mm_intent=INTENT_NEUTRAL,
            suggested_trade="wait_for_direction_confirmation",
            suggested_avoid="reading the rail flip as directional signal",
            evidence=[
                f"net_intent_z shifted {net_change:+.2f}σ",
                f"spot moved only {spot_change_bps:+.1f}bps",
            ],
        )
