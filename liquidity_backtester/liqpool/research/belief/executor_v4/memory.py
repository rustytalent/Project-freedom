"""Multi-timeframe memory.

Maintains rolling rollups of the snapshot stream at four resolutions:
  * L1  — last 60 ticks (1-min, the immediate)
  * L5  — 5-min rollups (medium)
  * L15 — 15-min rollups (macro)
  * L60 — 60-min rollups (session-level)

Each level computes:
  * dominant thesis composite state in the window
  * dominant battlefield verdict
  * thesis bull/bear net change
  * count of trap signals
  * count of continuation_trigger signals
  * regime stability (variance of dominant state changes)

Multi-timeframe alignment check used by the aggregator: an entry on L1
must be CONFIRMED by at least one of {L5, L15} matching directionally,
or it's downgraded to half size. Lone-L1 entries are the founder's
"retail trade the latest tick" pattern — refused.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple


class TimeframeLevel(str, Enum):
    L1 = "L1"      # immediate (last 60 ticks)
    L5 = "L5"      # 5-min rollup
    L15 = "L15"    # 15-min rollup
    L60 = "L60"    # 60-min rollup


@dataclass(frozen=True)
class TimeframeView:
    """One timeframe level's current view."""
    level: str
    n_samples: int
    dominant_thesis: str
    dominant_battlefield: str
    bull_score_net_change: float
    bear_score_net_change: float
    n_strong_traps: int
    n_continuation_triggers: int
    n_regime_breaks: int
    direction_bias: int                # +1/-1/0 majority vote
    direction_confidence: float        # 0..1
    regime_stability: float            # 0..1
    average_friendliness: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["notes"] = list(self.notes)
        return d


@dataclass
class _TimeframeBucket:
    """One bucket of snapshots at a given resolution."""
    snapshots: Deque[Dict[str, Any]] = field(default_factory=deque)
    cap: int = 60

    def push(self, snap: Dict[str, Any]) -> None:
        self.snapshots.append(snap)
        while len(self.snapshots) > self.cap:
            self.snapshots.popleft()


class MultiTimeframeMemory:
    """Multi-resolution memory of the snapshot stream."""

    def __init__(self,
                 l1_cap: int = 60,
                 l5_cap: int = 60,
                 l15_cap: int = 16,
                 l60_cap: int = 8,
                 l5_subsample: int = 5,
                 l15_subsample: int = 15,
                 l60_subsample: int = 60,
                 ) -> None:
        self.l1 = _TimeframeBucket(cap=l1_cap)
        self.l5 = _TimeframeBucket(cap=l5_cap)
        self.l15 = _TimeframeBucket(cap=l15_cap)
        self.l60 = _TimeframeBucket(cap=l60_cap)
        self._l5_subsample = max(1, l5_subsample)
        self._l15_subsample = max(1, l15_subsample)
        self._l60_subsample = max(1, l60_subsample)
        self._tick = 0

    def reset(self) -> None:
        self.l1.snapshots.clear()
        self.l5.snapshots.clear()
        self.l15.snapshots.clear()
        self.l60.snapshots.clear()
        self._tick = 0

    def observe(self, snapshot: Dict[str, Any]) -> None:
        """Push a snapshot through all four levels."""
        self._tick += 1
        self.l1.push(snapshot)
        if self._tick % self._l5_subsample == 0:
            self.l5.push(snapshot)
        if self._tick % self._l15_subsample == 0:
            self.l15.push(snapshot)
        if self._tick % self._l60_subsample == 0:
            self.l60.push(snapshot)

    def view(self, level: TimeframeLevel) -> TimeframeView:
        bucket = {
            TimeframeLevel.L1: self.l1,
            TimeframeLevel.L5: self.l5,
            TimeframeLevel.L15: self.l15,
            TimeframeLevel.L60: self.l60,
        }[level]
        return _build_view(level.value, list(bucket.snapshots))

    def all_views(self) -> Dict[str, TimeframeView]:
        return {
            level.value: self.view(level)
            for level in TimeframeLevel
        }

    def alignment(self, proposed_direction: int) -> Dict[str, Any]:
        """Multi-timeframe alignment check.

        Returns a dict with per-level direction match flags and a composite
        alignment score in [0,1].
        """
        views = self.all_views()
        l1_match = views["L1"].direction_bias == proposed_direction
        l5_match = views["L5"].direction_bias == proposed_direction
        l15_match = views["L15"].direction_bias == proposed_direction
        l60_match = views["L60"].direction_bias == proposed_direction

        # Founder's rule: L1 alone is retail; require at least one of {L5,L15}.
        confirmation_count = int(l5_match) + int(l15_match)
        alignment_ok = bool(l1_match and (confirmation_count >= 1 or l60_match))
        # Composite alignment score weighted toward longer timeframes.
        score = (0.20 * int(l1_match) + 0.35 * int(l5_match)
                 + 0.30 * int(l15_match) + 0.15 * int(l60_match))

        return {
            "l1_match": l1_match,
            "l5_match": l5_match,
            "l15_match": l15_match,
            "l60_match": l60_match,
            "confirmation_count": confirmation_count,
            "alignment_ok": alignment_ok,
            "alignment_score": round(score, 3),
            "views": {k: v.to_dict() for k, v in views.items()},
        }


def _build_view(level: str, snaps: List[Dict[str, Any]]) -> TimeframeView:
    if not snaps:
        return TimeframeView(
            level=level, n_samples=0,
            dominant_thesis="NEUTRAL",
            dominant_battlefield="quiet",
            bull_score_net_change=0.0, bear_score_net_change=0.0,
            n_strong_traps=0, n_continuation_triggers=0, n_regime_breaks=0,
            direction_bias=0, direction_confidence=0.0,
            regime_stability=1.0, average_friendliness=1.0,
        )

    thesis_states = []
    battlefield_states = []
    decision_dirs = []
    iv_dirs = []
    bf_dirs = []
    bull_scores = []
    bear_scores = []
    friend_total = 0.0
    friend_n = 0
    strong_traps = 0
    continuation_triggers = 0
    regime_breaks = 0

    for s in snaps:
        thesis = _map(s.get("thesis"))
        bf = _map(s.get("battlefield"))
        decision = _map(s.get("decision"))
        iv = _map(s.get("iv_state"))
        bull_state = _map(s.get("bull_state"))
        bear_state = _map(s.get("bear_state"))
        thesis_states.append(str(thesis.get("composite_state") or "NEUTRAL"))
        battlefield_states.append(str(bf.get("verdict") or "quiet"))
        decision_dirs.append(int(_num(decision.get("direction"), 0)))
        iv_dirs.append(int(_num(iv.get("direction"), 0)))
        bf_dirs.append(int(_num(bf.get("direction"), 0)))
        bull_scores.append(_num(thesis.get("bull_thesis_score")))
        bear_scores.append(_num(thesis.get("bear_thesis_score")))
        # Count event flags
        action = str(decision.get("action") or "")
        if action.startswith("EXIT"):
            regime_breaks += 1
        if int(_num(bull_state.get("state_index"), 0)) == 6:
            continuation_triggers += 1
        if int(_num(bear_state.get("state_index"), 0)) == 6:
            continuation_triggers += 1
        # Strong traps surfaced via thesis or decision
        if "STRONG_BULL_TRAP" in str(decision.get("invalidation_rule") or "") \
                or "STRONG_BEAR_TRAP" in str(decision.get("invalidation_rule") or ""):
            strong_traps += 1
        # Friendliness
        for raw in s.get("slot_readings") or []:
            slot = _map(raw)
            f = _num(slot.get("friendliness"), math.nan)
            if math.isfinite(f):
                friend_total += f
                friend_n += 1

    # Dominant labels via majority vote.
    thesis_counts = Counter(thesis_states)
    dom_thesis = thesis_counts.most_common(1)[0][0] if thesis_counts else "NEUTRAL"
    bf_counts = Counter(battlefield_states)
    dom_bf = bf_counts.most_common(1)[0][0] if bf_counts else "quiet"

    # Direction bias — majority of (decision, iv, battlefield).
    dir_combined = decision_dirs + iv_dirs + bf_dirs
    dir_counter = Counter(d for d in dir_combined if d != 0)
    if dir_counter:
        bias, votes = dir_counter.most_common(1)[0]
        total_votes = sum(dir_counter.values())
        confidence = votes / max(1, total_votes)
    else:
        bias = 0
        confidence = 0.0

    bull_net = (bull_scores[-1] - bull_scores[0]) if bull_scores else 0.0
    bear_net = (bear_scores[-1] - bear_scores[0]) if bear_scores else 0.0
    average_friend = friend_total / friend_n if friend_n else 1.0

    # Regime stability — how often dominant thesis flipped.
    flips = sum(1 for a, b in zip(thesis_states, thesis_states[1:]) if a != b)
    stability = max(0.0, 1.0 - flips / max(1, len(thesis_states) - 1))

    notes: List[str] = []
    if strong_traps:
        notes.append(f"{strong_traps} strong trap(s) in window")
    if regime_breaks:
        notes.append(f"{regime_breaks} exit/regime-break event(s) in window")
    if stability < 0.4:
        notes.append("regime unstable (frequent thesis flips)")

    return TimeframeView(
        level=level,
        n_samples=len(snaps),
        dominant_thesis=dom_thesis,
        dominant_battlefield=dom_bf,
        bull_score_net_change=round(bull_net, 1),
        bear_score_net_change=round(bear_net, 1),
        n_strong_traps=strong_traps,
        n_continuation_triggers=continuation_triggers,
        n_regime_breaks=regime_breaks,
        direction_bias=int(bias),
        direction_confidence=round(confidence, 3),
        regime_stability=round(stability, 3),
        average_friendliness=round(average_friend, 3),
        notes=notes,
    )


def _map(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return default
