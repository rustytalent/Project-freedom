r"""Forward Bayesian projection — transparent conditional probabilities.

The scenario web tracks *which* pathways are alive; projection answers
*how often this kind of context has resolved each way in the recent
past*. It's a closed-form, transparent estimator — NOT a black-box ML
model.

The estimator builds conditional empirical distributions over outcomes:

  P(continuation | current_state)
  P(reversal_in_N_bars | current_state)
  E[premium_move | current_state] + std + percentiles
  P(target_hit_first | current_state, target_R, stop_R)
  P(stop_hit_first  | current_state, target_R, stop_R)

These conditionals are estimated from a rolling tape of past
``ProjectionRecord``\s — every time the manager updates an open
position, it appends one record describing the state at observation
time and (after closure) the realized outcome. The estimator queries
matching past records using a similarity score over a small set of
state coordinates.

Sprint-2 design choice: the estimator returns explicit "sample_count"
and "confidence" alongside every probability — when sample size is
small (n<20) the aggregator falls back to Sprint-1 heuristics.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class ProjectionConfig:
    """Knobs for the projection layer."""
    history_cap: int = 600
    min_samples_for_confidence: int = 20
    similarity_threshold: float = 0.55     # min match score to include a record
    # Weight per coordinate when computing similarity.
    weight_thesis_state: float = 0.30
    weight_iv_state: float = 0.20
    weight_battlefield: float = 0.20
    weight_winding: float = 0.10
    weight_direction: float = 0.10
    weight_regime_stability: float = 0.10


@dataclass(frozen=True)
class ProjectionRecord:
    """One realized observation used as a prior sample for projection."""
    ts: Any
    bar_index: int
    thesis_state: str
    iv_state: str
    battlefield_verdict: str
    winding_zone: str
    direction: int
    regime_stability: float
    bull_score: float
    bear_score: float
    net_intent_z: float
    # Realized outcome (filled in on close).
    realized_premium_change_pct: float = 0.0
    realized_r: float = 0.0
    bars_to_resolution: int = 0
    closed_via_target: bool = False
    closed_via_stop: bool = False
    closed_via_neither: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": str(self.ts),
            "bar_index": self.bar_index,
            "thesis_state": self.thesis_state,
            "iv_state": self.iv_state,
            "battlefield_verdict": self.battlefield_verdict,
            "winding_zone": self.winding_zone,
            "direction": self.direction,
            "regime_stability": round(self.regime_stability, 3),
            "bull_score": round(self.bull_score, 1),
            "bear_score": round(self.bear_score, 1),
            "net_intent_z": round(self.net_intent_z, 3),
            "realized_premium_change_pct": round(self.realized_premium_change_pct, 4),
            "realized_r": round(self.realized_r, 3),
            "bars_to_resolution": self.bars_to_resolution,
            "closed_via_target": self.closed_via_target,
            "closed_via_stop": self.closed_via_stop,
            "closed_via_neither": self.closed_via_neither,
        }


@dataclass
class ProjectionDistribution:
    """Empirical conditional distribution of the realized R outcome."""
    n_samples: int
    confidence: float                 # 0..1, scales with sample_count
    mean_r: float
    std_r: float
    p05_r: float
    p25_r: float
    p50_r: float
    p75_r: float
    p95_r: float
    p_target_hit_first: float
    p_stop_hit_first: float
    p_neither: float
    p_continuation: float             # realized R same sign as direction
    p_reversal: float                  # realized R opposite sign
    median_bars_to_resolution: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "confidence": round(self.confidence, 3),
            "mean_r": round(self.mean_r, 3),
            "std_r": round(self.std_r, 3),
            "p05_r": round(self.p05_r, 3),
            "p25_r": round(self.p25_r, 3),
            "p50_r": round(self.p50_r, 3),
            "p75_r": round(self.p75_r, 3),
            "p95_r": round(self.p95_r, 3),
            "p_target_hit_first": round(self.p_target_hit_first, 3),
            "p_stop_hit_first": round(self.p_stop_hit_first, 3),
            "p_neither": round(self.p_neither, 3),
            "p_continuation": round(self.p_continuation, 3),
            "p_reversal": round(self.p_reversal, 3),
            "median_bars_to_resolution": round(self.median_bars_to_resolution, 1),
            "notes": list(self.notes),
        }


class ForwardProjection:
    """Closed-form forward projection estimator.

    Maintains a rolling tape of ``ProjectionRecord`` samples. At query
    time, returns a conditional empirical distribution over realized R
    matching the current state context.
    """

    def __init__(self, cfg: Optional[ProjectionConfig] = None) -> None:
        self.cfg = cfg or ProjectionConfig()
        self.tape: Deque[ProjectionRecord] = deque(maxlen=self.cfg.history_cap)

    def reset(self) -> None:
        self.tape.clear()

    def record(self, record: ProjectionRecord) -> None:
        """Append a closed-out observation to the tape."""
        self.tape.append(record)

    def project(self, *,
                 thesis_state: str,
                 iv_state: str,
                 battlefield_verdict: str,
                 winding_zone: str,
                 direction: int,
                 regime_stability: float,
                 target_r: float = 1.5,
                 stop_r: float = 1.0,
                 ) -> ProjectionDistribution:
        """Empirical conditional distribution of realized R given context.

        ``target_r`` and ``stop_r`` are the R levels that define a target
        vs stop hit. The estimator returns P(target hit first | context)
        from the realized outcomes of similar past records.
        """
        cfg = self.cfg
        matches: List[Tuple[float, ProjectionRecord]] = []
        for rec in self.tape:
            sim = self._similarity(rec,
                                    thesis_state=thesis_state,
                                    iv_state=iv_state,
                                    battlefield_verdict=battlefield_verdict,
                                    winding_zone=winding_zone,
                                    direction=direction,
                                    regime_stability=regime_stability)
            if sim >= cfg.similarity_threshold:
                matches.append((sim, rec))

        if not matches:
            return _empty_distribution(notes=[
                "no matching historical records — projection unavailable"
            ])

        # Weight by similarity. Compute empirical stats over realized_r.
        rs: List[float] = [rec.realized_r for _, rec in matches]
        weights: List[float] = [sim for sim, _ in matches]
        bars: List[int] = [rec.bars_to_resolution for _, rec in matches]
        # Target/stop hit counts.
        target_hits = sum(1 for _, rec in matches if rec.closed_via_target)
        stop_hits = sum(1 for _, rec in matches if rec.closed_via_stop)
        neither = sum(1 for _, rec in matches if rec.closed_via_neither)
        n = len(matches)
        p_target = target_hits / n
        p_stop = stop_hits / n
        p_neither = neither / n
        # Continuation = realized_r has same sign as direction (positive R for
        # long, negative R would be against — but R is signed wrt direction so
        # positive R always means moved with us).
        n_cont = sum(1 for r in rs if r > 0.0)
        n_rev = sum(1 for r in rs if r < 0.0)
        p_cont = n_cont / n
        p_rev = n_rev / n

        sorted_rs = sorted(rs)

        def pct(p: float) -> float:
            if not sorted_rs:
                return 0.0
            idx = int(round((len(sorted_rs) - 1) * p))
            return sorted_rs[idx]

        mean_r = sum(r * w for r, w in zip(rs, weights)) / sum(weights)
        std_r = statistics.pstdev(rs) if len(rs) >= 2 else 0.0
        # Confidence scales with sqrt(n) capped at min_samples_for_confidence.
        confidence = min(1.0, math.sqrt(n / cfg.min_samples_for_confidence))

        notes: List[str] = []
        if n < cfg.min_samples_for_confidence:
            notes.append(
                f"only {n} similar historical records (need "
                f"{cfg.min_samples_for_confidence}+ for full confidence)"
            )
        return ProjectionDistribution(
            n_samples=n,
            confidence=confidence,
            mean_r=mean_r,
            std_r=std_r,
            p05_r=pct(0.05),
            p25_r=pct(0.25),
            p50_r=pct(0.50),
            p75_r=pct(0.75),
            p95_r=pct(0.95),
            p_target_hit_first=p_target,
            p_stop_hit_first=p_stop,
            p_neither=p_neither,
            p_continuation=p_cont,
            p_reversal=p_rev,
            median_bars_to_resolution=statistics.median(bars) if bars else 0.0,
            notes=notes,
        )

    def _similarity(self, rec: ProjectionRecord, *,
                     thesis_state: str, iv_state: str,
                     battlefield_verdict: str, winding_zone: str,
                     direction: int, regime_stability: float) -> float:
        cfg = self.cfg
        score = 0.0
        if rec.thesis_state == thesis_state:
            score += cfg.weight_thesis_state
        elif _thesis_family(rec.thesis_state) == _thesis_family(thesis_state):
            score += cfg.weight_thesis_state * 0.5
        if rec.iv_state == iv_state:
            score += cfg.weight_iv_state
        if rec.battlefield_verdict == battlefield_verdict:
            score += cfg.weight_battlefield
        if rec.winding_zone == winding_zone:
            score += cfg.weight_winding
        if rec.direction == direction:
            score += cfg.weight_direction
        # Regime stability — bucketed by 0.20 windows.
        if abs(rec.regime_stability - regime_stability) <= 0.20:
            score += cfg.weight_regime_stability
        return score

    def summary(self) -> Dict[str, Any]:
        return {
            "tape_size": len(self.tape),
            "n_target_hits": sum(1 for r in self.tape if r.closed_via_target),
            "n_stop_hits": sum(1 for r in self.tape if r.closed_via_stop),
            "n_neither": sum(1 for r in self.tape if r.closed_via_neither),
            "mean_realized_r": (
                statistics.mean(r.realized_r for r in self.tape)
                if self.tape else 0.0
            ),
        }


def _empty_distribution(notes: List[str]) -> ProjectionDistribution:
    return ProjectionDistribution(
        n_samples=0, confidence=0.0, mean_r=0.0, std_r=0.0,
        p05_r=0.0, p25_r=0.0, p50_r=0.0, p75_r=0.0, p95_r=0.0,
        p_target_hit_first=0.0, p_stop_hit_first=0.0, p_neither=0.0,
        p_continuation=0.0, p_reversal=0.0,
        median_bars_to_resolution=0.0,
        notes=list(notes),
    )


def _thesis_family(state: str) -> str:
    """Coarse thesis family for fallback similarity scoring."""
    if "BULL" in state:
        return "BULL"
    if "BEAR" in state:
        return "BEAR"
    return "NEUTRAL"
