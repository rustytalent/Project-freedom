"""Weight evolution memory + analyzer — a new dimension for the executor.

The founder's insight (2026-06-22): "If we save every weight change, the
delta in changes can conclude many things — trend, velocity,
adaptability, warnings, new dimensions for looking at the market.
Historical reader (past 1–3 hours) has many applications."

He's right. Every time the live calibrator nudges (or refuses to nudge)
the aggregator weights, that event itself carries information:

  * Which signal-component is the market currently rewarding?
  * How FAST is the system re-learning? (adaptability index)
  * Are the weights stable (regime is being read correctly) or thrashing
    (regime is unfamiliar / hostile)?
  * If the validation loss is climbing snapshot-after-snapshot, the
    system is losing predictive power → operator warning.
  * Coordinated multi-weight drift = evidence of regime SHIFT.

This module persists every weight snapshot, then provides:

  * ``WeightEvolutionMemory.append(snap)`` — sliding window + disk persist
  * ``WeightEvolutionAnalyzer.analyze(memory)`` — produces the analysis:
        trend per weight, velocity, adaptability index, warnings,
        regime-shift evidence
  * ``WeightEvolutionAnalyzer.query_at(memory, seconds_ago)`` — the
    historical reader for "what did the system think 90 minutes ago?"
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


_DEFAULT_PERSIST_DIR = Path("/var/lib/sentinel/executor_v4_state")


# ── Snapshot ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class WeightSnapshot:
    """One observed weight state at a moment in time."""
    ts: float                           # unix timestamp (seconds)
    bar_index: int                       # engine's bar at the time
    weights: Dict[str, float]           # the 5 aggregator weights
    n_samples_used: int
    train_loss: float
    val_loss: float
    accepted: bool                       # was this update applied?
    overfit_ratio: float
    source: str                          # "live_calibrator" / "manual" / etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "bar_index": self.bar_index,
            "weights": dict(self.weights),
            "n_samples_used": self.n_samples_used,
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "accepted": self.accepted,
            "overfit_ratio": round(self.overfit_ratio, 4),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WeightSnapshot":
        return cls(
            ts=float(d.get("ts", 0.0)),
            bar_index=int(d.get("bar_index", 0)),
            weights=dict(d.get("weights") or {}),
            n_samples_used=int(d.get("n_samples_used", 0)),
            train_loss=float(d.get("train_loss", 0.0)),
            val_loss=float(d.get("val_loss", 0.0)),
            accepted=bool(d.get("accepted", False)),
            overfit_ratio=float(d.get("overfit_ratio", 1.0)),
            source=str(d.get("source", "")),
        )


# ── Memory ────────────────────────────────────────────────────────


@dataclass
class WeightEvolutionMemoryConfig:
    """Knobs."""
    window_seconds: float = 3 * 3600.0   # 3-hour sliding window
    max_snapshots: int = 4096
    persist_path: Optional[Path] = None  # None = no disk persist
    flush_every_n_appends: int = 4


class WeightEvolutionMemory:
    """Sliding-window memory of weight snapshots with disk persistence."""

    def __init__(self,
                  cfg: Optional[WeightEvolutionMemoryConfig] = None) -> None:
        self.cfg = cfg or WeightEvolutionMemoryConfig()
        self._snaps: Deque[WeightSnapshot] = deque(
            maxlen=self.cfg.max_snapshots)
        self._appends_since_flush = 0
        self._ensure_persist_dir()
        self._restore_from_disk()

    # ── Public API ─────────────────────────────────────────────────

    def append(self, snap: WeightSnapshot) -> None:
        self._snaps.append(snap)
        self._evict_old(now=snap.ts)
        self._appends_since_flush += 1
        if (self.cfg.persist_path is not None
                and self._appends_since_flush
                >= self.cfg.flush_every_n_appends):
            self._flush()

    def __len__(self) -> int:
        return len(self._snaps)

    def all(self) -> List[WeightSnapshot]:
        return list(self._snaps)

    def latest(self) -> Optional[WeightSnapshot]:
        return self._snaps[-1] if self._snaps else None

    def window(self, *, seconds: float) -> List[WeightSnapshot]:
        if not self._snaps:
            return []
        end_ts = self._snaps[-1].ts
        cutoff = end_ts - seconds
        return [s for s in self._snaps if s.ts >= cutoff]

    def reset(self) -> None:
        self._snaps.clear()
        self._appends_since_flush = 0
        if self.cfg.persist_path is not None:
            try:
                self.cfg.persist_path.unlink(missing_ok=True)
            except Exception:
                pass

    # ── Internal: disk persistence ─────────────────────────────────

    def _ensure_persist_dir(self) -> None:
        if self.cfg.persist_path is None:
            return
        try:
            self.cfg.persist_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _restore_from_disk(self) -> None:
        if self.cfg.persist_path is None:
            return
        if not self.cfg.persist_path.exists():
            return
        try:
            with open(self.cfg.persist_path) as f:
                rows = [json.loads(line)
                         for line in f if line.strip()]
            for row in rows:
                try:
                    self._snaps.append(WeightSnapshot.from_dict(row))
                except Exception:
                    continue
            if self._snaps:
                self._evict_old(now=self._snaps[-1].ts)
        except Exception:
            pass

    def _flush(self) -> None:
        if self.cfg.persist_path is None:
            return
        try:
            with open(self.cfg.persist_path, "w") as f:
                for s in self._snaps:
                    f.write(json.dumps(s.to_dict()) + "\n")
            self._appends_since_flush = 0
        except Exception:
            pass

    def _evict_old(self, *, now: float) -> None:
        cutoff = now - self.cfg.window_seconds
        while self._snaps and self._snaps[0].ts < cutoff:
            self._snaps.popleft()


# ── Analysis output ───────────────────────────────────────────────


@dataclass
class WeightEvolutionAnalysis:
    """The reader's output. The founder's 'new dimension' surfaced."""
    n_snapshots: int
    n_accepted: int
    n_rejected: int
    adaptability_index: float           # accepted / (accepted+rejected) in window
    trend_per_weight: Dict[str, str]     # weight → 'rising' / 'falling' / 'stable'
    velocity_per_weight: Dict[str, float]  # signed change rate per minute
    most_drifting_weight: Optional[str]
    coordinated_drift_score: float       # 0..1 — multi-weight coordinated motion
    val_loss_trend: str                   # 'improving' / 'degrading' / 'stable'
    warnings: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_snapshots": self.n_snapshots,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "adaptability_index": round(self.adaptability_index, 4),
            "trend_per_weight": dict(self.trend_per_weight),
            "velocity_per_weight": {
                k: round(v, 6) for k, v in self.velocity_per_weight.items()
            },
            "most_drifting_weight": self.most_drifting_weight,
            "coordinated_drift_score": round(self.coordinated_drift_score, 4),
            "val_loss_trend": self.val_loss_trend,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


# ── Analyzer ──────────────────────────────────────────────────────


class WeightEvolutionAnalyzer:
    """Extracts insight from a WeightEvolutionMemory."""

    TREND_RISING = "rising"
    TREND_FALLING = "falling"
    TREND_STABLE = "stable"

    LOSS_IMPROVING = "improving"
    LOSS_DEGRADING = "degrading"
    LOSS_STABLE = "stable"

    # Velocity threshold (per minute) — beyond this, weight is "drifting".
    VELOCITY_DRIFT_THRESHOLD = 0.0005

    # Coordinated motion threshold: this many weights moving same direction
    # = "the whole model is repointing" = regime shift evidence.
    COORD_MIN_AGREE = 3

    def analyze(self, memory: WeightEvolutionMemory,
                  *, seconds: Optional[float] = None
                  ) -> WeightEvolutionAnalysis:
        """Analyze the memory; ``seconds`` = lookback window (default = full)."""
        snaps = (memory.window(seconds=seconds)
                  if seconds is not None else memory.all())
        if not snaps:
            return _empty_analysis()

        n_accepted = sum(1 for s in snaps if s.accepted)
        n_rejected = len(snaps) - n_accepted
        adaptability = n_accepted / max(1, len(snaps))

        # Velocity per weight = (last - first) / minutes elapsed.
        velocities: Dict[str, float] = {}
        trends: Dict[str, str] = {}
        first = snaps[0]
        last = snaps[-1]
        minutes = max((last.ts - first.ts) / 60.0, 1e-6)
        for w in first.weights.keys():
            change = last.weights.get(w, 0.0) - first.weights.get(w, 0.0)
            v = change / minutes
            velocities[w] = v
            if abs(v) < self.VELOCITY_DRIFT_THRESHOLD:
                trends[w] = self.TREND_STABLE
            elif v > 0:
                trends[w] = self.TREND_RISING
            else:
                trends[w] = self.TREND_FALLING

        most_drifting = (max(velocities.keys(),
                              key=lambda w: abs(velocities[w]))
                          if velocities else None)

        # Coordinated drift — count weights moving in the same direction
        # significantly. If >= COORD_MIN_AGREE of them are in the same
        # direction, that's a model-wide repointing → regime shift.
        n_rising = sum(1 for t in trends.values() if t == self.TREND_RISING)
        n_falling = sum(1 for t in trends.values() if t == self.TREND_FALLING)
        coord_count = max(n_rising, n_falling)
        coord_score = (coord_count / max(1, len(trends))
                        if coord_count >= self.COORD_MIN_AGREE else 0.0)

        # Val loss trend — linear fit on val_loss across the window.
        val_trend = self._loss_trend([s.val_loss for s in snaps])

        warnings: List[str] = []
        notes: List[str] = []
        if val_trend == self.LOSS_DEGRADING:
            warnings.append(
                "validation loss is climbing across the window — "
                "the model is losing predictive power; consider pausing "
                "the live calibrator and reviewing")
        if n_rejected > 2 * max(1, n_accepted) and len(snaps) >= 6:
            warnings.append(
                f"high rejection ratio ({n_rejected} rejected vs "
                f"{n_accepted} accepted) — market is too noisy for the "
                "calibrator; updates are overfitting the train sample")
        if coord_score >= 0.6:
            warnings.append(
                f"coordinated drift across {coord_count} weights — "
                "evidence of a regime SHIFT; review whether current "
                "open positions still match the new regime")
        if most_drifting and abs(velocities[most_drifting]) > 0.005:
            notes.append(
                f"'{most_drifting}' is the dominant drifter "
                f"({velocities[most_drifting]:+.4f}/min) — the market "
                "is increasingly rewarding/punishing that component")
        if adaptability < 0.20 and len(snaps) >= 6:
            warnings.append(
                f"low adaptability ({adaptability:.0%}) — the system is "
                "rejecting most updates; either the market is stable "
                "(fine) or the calibrator is over-cautious (review)")

        return WeightEvolutionAnalysis(
            n_snapshots=len(snaps),
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            adaptability_index=adaptability,
            trend_per_weight=trends,
            velocity_per_weight=velocities,
            most_drifting_weight=most_drifting,
            coordinated_drift_score=coord_score,
            val_loss_trend=val_trend,
            warnings=warnings,
            notes=notes,
        )

    def query_at(self, memory: WeightEvolutionMemory,
                   *, seconds_ago: float
                   ) -> Optional[WeightSnapshot]:
        """Historical reader: the founder's 'what did the system think 1-3
        hours ago?' query."""
        snaps = memory.all()
        if not snaps:
            return None
        target_ts = snaps[-1].ts - seconds_ago
        # Return the snapshot CLOSEST to target_ts.
        best = None
        best_gap = float("inf")
        for s in snaps:
            gap = abs(s.ts - target_ts)
            if gap < best_gap:
                best_gap = gap
                best = s
        return best

    # ── Internals ──────────────────────────────────────────────────

    def _loss_trend(self, losses: List[float]) -> str:
        if len(losses) < 3:
            return self.LOSS_STABLE
        # Linear fit slope sign.
        n = len(losses)
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(losses) / n
        num = sum((xs[i] - mx) * (losses[i] - my) for i in range(n))
        den = sum((xs[i] - mx) ** 2 for i in range(n))
        if den < 1e-9:
            return self.LOSS_STABLE
        slope = num / den
        normalised = slope / max(1e-6, abs(my))
        if normalised > 0.02:
            return self.LOSS_DEGRADING
        if normalised < -0.02:
            return self.LOSS_IMPROVING
        return self.LOSS_STABLE


# ── Helpers ───────────────────────────────────────────────────────


def _empty_analysis() -> WeightEvolutionAnalysis:
    return WeightEvolutionAnalysis(
        n_snapshots=0, n_accepted=0, n_rejected=0,
        adaptability_index=0.0,
        trend_per_weight={}, velocity_per_weight={},
        most_drifting_weight=None,
        coordinated_drift_score=0.0,
        val_loss_trend=WeightEvolutionAnalyzer.LOSS_STABLE,
        warnings=["no snapshots in window"],
    )
