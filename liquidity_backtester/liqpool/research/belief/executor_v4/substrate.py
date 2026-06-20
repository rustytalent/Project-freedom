"""Information substrate — exposing rich module internals.

The condensed BeliefSnapshot was designed for human display. This module
augments it with the full high-dimensional context the v4 executor needs:

  * per-slot derivatives — rate of change of dod_z, acceptance run-length,
    epicenter migration
  * rail-level trajectories — net_intent_z, rail signed-z and abs-z over time
  * thesis score velocity and acceleration
  * state-machine transition history per machine
  * regime stability index — how consistent the regime has been
  * 22-slot heatmap flattened for downstream consumers

This is the "information bus" all downstream layers consume.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class SubstrateConfig:
    """Knobs for the substrate."""
    history_bars: int = 240
    velocity_window: int = 5         # bars over which to compute first derivative
    accel_window: int = 10           # bars over which to compute second derivative
    epicenter_window: int = 10
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.history_bars < 30:
            raise ValueError("history_bars must be >= 30")


@dataclass
class _Trajectory:
    """Rolling deque of scalar samples with derivative helpers."""
    samples: Deque[float] = field(default_factory=deque)
    timestamps: Deque[Any] = field(default_factory=deque)
    cap: int = 240

    def push(self, value: float, ts: Any) -> None:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return
        self.samples.append(float(value))
        self.timestamps.append(ts)
        while len(self.samples) > self.cap:
            self.samples.popleft()
            self.timestamps.popleft()

    def last(self) -> Optional[float]:
        return self.samples[-1] if self.samples else None

    def mean(self, window: int) -> Optional[float]:
        if not self.samples:
            return None
        n = min(window, len(self.samples))
        recent = list(self.samples)[-n:]
        return sum(recent) / n

    def velocity(self, window: int) -> float:
        """First derivative — current minus mean of (window-back, window)."""
        if len(self.samples) < window + 1:
            return 0.0
        current = self.samples[-1]
        prior = self.samples[-(window + 1)]
        return (current - prior) / window

    def acceleration(self, window: int) -> float:
        """Second derivative — velocity now minus velocity (window) ago."""
        if len(self.samples) < 2 * window + 1:
            return 0.0
        v_now = self.velocity(window)
        # velocity at (window) bars ago = (samples[-window-1] - samples[-2*window-1]) / window
        v_prior = (self.samples[-(window + 1)] - self.samples[-(2 * window + 1)]) / window
        return (v_now - v_prior) / window

    def percentile(self, pct: float) -> Optional[float]:
        if not self.samples:
            return None
        sorted_s = sorted(self.samples)
        idx = int(round((len(sorted_s) - 1) * pct))
        return sorted_s[idx]


@dataclass
class RichContext:
    """The augmented context for one bar — what the downstream layers see.

    Carries everything the v4 stack needs to think about a single tick:
      * the original snapshot dict
      * per-rail derivatives (velocity + accel of mean signed-z)
      * thesis velocity + accel
      * net-intent velocity + accel
      * per-slot acceptance run-lengths
      * epicenter migration tracking
      * regime stability index in [0,1] — 1 = stable, 0 = chaos
      * dispersion trajectory (the 22-slot heatmap evenness)
      * flat heatmap vector
    """
    snapshot: Dict[str, Any]
    ce_signed_z_velocity: float
    ce_signed_z_acceleration: float
    pe_signed_z_velocity: float
    pe_signed_z_acceleration: float
    net_intent_velocity: float
    net_intent_acceleration: float
    thesis_bull_velocity: float
    thesis_bear_velocity: float
    thesis_velocity_dominant_side: str   # "bull" / "bear" / "flat"
    acceptance_run_lengths: Dict[str, int]   # per slot label → bars in current state
    epicenter_label: str
    epicenter_level: int
    epicenter_migration_distance: float   # |level shift| over recent window
    dispersion_velocity: float            # CE dispersion rate of change
    regime_stability_index: float         # 0..1
    heatmap_flat: List[float]             # 22 entries, slot-ordered
    bar_index: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["acceptance_run_lengths"] = dict(self.acceptance_run_lengths)
        d["heatmap_flat"] = list(self.heatmap_flat)
        d["notes"] = list(self.notes)
        d.pop("snapshot", None)   # snapshot is large; carry separately
        return d


class SubstrateState:
    """Per-instance state for incremental enrichment.

    Maintains rolling trajectories of every signal that has a meaningful
    derivative, plus per-slot acceptance state for run-length tracking.
    Call observe() once per tick.
    """

    def __init__(self, cfg: Optional[SubstrateConfig] = None) -> None:
        self.cfg = cfg or SubstrateConfig()
        cap = self.cfg.history_bars
        self.ce_signed_z = _Trajectory(cap=cap)
        self.pe_signed_z = _Trajectory(cap=cap)
        self.ce_abs_z = _Trajectory(cap=cap)
        self.pe_abs_z = _Trajectory(cap=cap)
        self.net_intent = _Trajectory(cap=cap)
        self.thesis_bull = _Trajectory(cap=cap)
        self.thesis_bear = _Trajectory(cap=cap)
        self.ce_dispersion = _Trajectory(cap=cap)
        self.pe_dispersion = _Trajectory(cap=cap)
        self.spot = _Trajectory(cap=cap)
        # Per-slot acceptance runs.
        self.accept_state: Dict[str, str] = {}
        self.accept_run: Dict[str, int] = {}
        # Epicenter migration.
        self.recent_epicenters: Deque[Tuple[str, int]] = deque(maxlen=self.cfg.epicenter_window)
        self.bar_index = 0

    def reset(self) -> None:
        self.__init__(self.cfg)

    def observe(self, snapshot: Dict[str, Any]) -> RichContext:
        """Ingest one snapshot dict and emit the augmented context."""
        cfg = self.cfg
        self.bar_index += 1
        ts = snapshot.get("ts")
        spot = _num(snapshot.get("spot"))
        thesis = _map(snapshot.get("thesis"))
        bf = _map(snapshot.get("battlefield"))
        ce_rail = _map(bf.get("ce_rail"))
        pe_rail = _map(bf.get("pe_rail"))

        ce_signed = _num(ce_rail.get("weighted_mean_signed_z"))
        pe_signed = _num(pe_rail.get("weighted_mean_signed_z"))
        ce_abs = _num(ce_rail.get("weighted_mean_abs_z"))
        pe_abs = _num(pe_rail.get("weighted_mean_abs_z"))
        ce_disp = _num(ce_rail.get("dispersion_score"))
        pe_disp = _num(pe_rail.get("dispersion_score"))
        bull = _num(thesis.get("bull_thesis_score"))
        bear = _num(thesis.get("bear_thesis_score"))
        # net_intent_z lives on the snapshot's iv_state if not exposed directly
        # — fall back to ce_signed − pe_signed which is its definition
        net_intent_z = _num(_map(snapshot.get("iv_state")).get("net_intent_z"))
        if net_intent_z == 0.0:
            net_intent_z = ce_signed - pe_signed

        self.ce_signed_z.push(ce_signed, ts)
        self.pe_signed_z.push(pe_signed, ts)
        self.ce_abs_z.push(ce_abs, ts)
        self.pe_abs_z.push(pe_abs, ts)
        self.ce_dispersion.push(ce_disp, ts)
        self.pe_dispersion.push(pe_disp, ts)
        self.net_intent.push(net_intent_z, ts)
        self.thesis_bull.push(bull, ts)
        self.thesis_bear.push(bear, ts)
        self.spot.push(spot, ts)

        # Per-slot acceptance run-length tracking.
        slot_readings = list(snapshot.get("slot_readings") or [])
        acceptance_run_lengths: Dict[str, int] = {}
        heatmap: List[float] = []
        for raw in slot_readings:
            slot = _map(raw)
            label = str(slot.get("label") or slot.get("moneyness_label") or "")
            acc = str(slot.get("acceptance") or "normal")
            dod_z = _num(slot.get("dod_z"))
            heatmap.append(dod_z)
            prev = self.accept_state.get(label, "normal")
            if acc == prev:
                self.accept_run[label] = self.accept_run.get(label, 0) + 1
            else:
                self.accept_run[label] = 1
                self.accept_state[label] = acc
            acceptance_run_lengths[label] = self.accept_run[label]
        # pad heatmap to 22
        heatmap = heatmap[:22] + [0.0] * max(0, 22 - len(heatmap))

        # Epicenter migration.
        ce_epi_label = str(ce_rail.get("epicenter_label") or "")
        ce_epi_level = int(_num(ce_rail.get("epicenter_level"), 0))
        if ce_epi_label:
            self.recent_epicenters.append((ce_epi_label, ce_epi_level))
        if len(self.recent_epicenters) >= 2:
            levels = [lev for _, lev in self.recent_epicenters]
            mig_dist = float(max(levels) - min(levels))
        else:
            mig_dist = 0.0

        # Derivatives.
        ce_v = self.ce_signed_z.velocity(cfg.velocity_window)
        ce_a = self.ce_signed_z.acceleration(cfg.accel_window)
        pe_v = self.pe_signed_z.velocity(cfg.velocity_window)
        pe_a = self.pe_signed_z.acceleration(cfg.accel_window)
        net_v = self.net_intent.velocity(cfg.velocity_window)
        net_a = self.net_intent.acceleration(cfg.accel_window)
        bull_v = self.thesis_bull.velocity(cfg.velocity_window)
        bear_v = self.thesis_bear.velocity(cfg.velocity_window)
        disp_v = self.ce_dispersion.velocity(cfg.velocity_window)

        # Dominant velocity side
        if abs(bull_v) > abs(bear_v) and abs(bull_v) > 0.5:
            dom = "bull"
        elif abs(bear_v) > abs(bull_v) and abs(bear_v) > 0.5:
            dom = "bear"
        else:
            dom = "flat"

        # Regime stability index — bounded composite.
        # High = both rails not whipping, low dispersion velocity, thesis not erratic.
        whip = (abs(ce_v) + abs(pe_v)) / 2.0
        whip_norm = max(0.0, 1.0 - whip / 2.0)
        disp_norm = max(0.0, 1.0 - abs(disp_v) / 0.5)
        thesis_stability = max(0.0, 1.0 - (abs(bull_v) + abs(bear_v)) / 30.0)
        regime_stability = max(0.0, min(1.0,
                                         0.4 * whip_norm + 0.3 * disp_norm
                                         + 0.3 * thesis_stability))

        notes: List[str] = []
        if mig_dist >= 3:
            notes.append(f"epicenter migrated by {mig_dist:.0f} strikes recently")
        if regime_stability < 0.35:
            notes.append(f"regime unstable (stability index {regime_stability:.2f})")
        if disp_v > 0.10:
            notes.append("CE rail dispersion rising — single-strike pressure forming")

        return RichContext(
            snapshot=snapshot,
            ce_signed_z_velocity=round(ce_v, 4),
            ce_signed_z_acceleration=round(ce_a, 4),
            pe_signed_z_velocity=round(pe_v, 4),
            pe_signed_z_acceleration=round(pe_a, 4),
            net_intent_velocity=round(net_v, 4),
            net_intent_acceleration=round(net_a, 4),
            thesis_bull_velocity=round(bull_v, 3),
            thesis_bear_velocity=round(bear_v, 3),
            thesis_velocity_dominant_side=dom,
            acceptance_run_lengths=acceptance_run_lengths,
            epicenter_label=ce_epi_label,
            epicenter_level=ce_epi_level,
            epicenter_migration_distance=mig_dist,
            dispersion_velocity=round(disp_v, 4),
            regime_stability_index=round(regime_stability, 3),
            heatmap_flat=heatmap,
            bar_index=self.bar_index,
            notes=notes,
        )


def augment_snapshot(snapshot: Dict[str, Any],
                     state: SubstrateState) -> RichContext:
    """Convenience: augment a snapshot using the given persistent state."""
    return state.observe(snapshot)


# ─────────────────────────────────────────────────────────────────
# Local helpers
# ─────────────────────────────────────────────────────────────────

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
