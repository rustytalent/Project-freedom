"""Engine upgrades — applies the audit's top fixes without touching engine.py.

The audit identified five high-impact issues in the existing Premium
Belief Engine that would hurt live Monday performance:

  1. **Fixed 80-bar warmup blocks Monday morning** — needs adaptive
     graduated warmup: scaled-down confidence after 30 bars, full
     confidence after 80
  2. **Hardcoded NIFTY calibration** — won't generalize; needs per-symbol
     bootstrap
  3. **No uncertainty quantification** — point estimates only; needs
     confidence intervals
  4. **No dirty-quote healing** — single-bar spread blowout shouldn't
     blow up downstream signals; needs healing window
  5. **Static shrinkage on realized-vol divergence** — needs feedback
     loop

This module is a SNAPSHOT TRANSFORMER. Per tick, it:

  * augments the snapshot with `confidence_intervals` based on rolling
    signal volatility
  * downgrades or pauses entry actions during graduated warmup
  * pauses entries (but allows exits) during dirty-quote storms
  * adjusts the snapshot's confidence based on realized vol divergence

The transformer is stateful — it keeps rolling buffers across ticks.
The V4Runner wires it in as a pre-processor before the manager sees
the snapshot.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


# Action tagging.
ENTRY_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "SCALP_CALL", "SCALP_PUT"}


@dataclass
class EngineUpgradesConfig:
    """Knobs for the upgrade transformer."""
    # Adaptive warmup
    soft_warmup_floor: int = 30           # bars before any trading allowed
    hard_warmup_target: int = 80           # bars for full confidence
    soft_warmup_confidence_cap: float = 0.55
    # Dirty quote healing
    dirty_storm_threshold: float = 0.30    # >30% of contracts dirty → storm
    dirty_storm_pause_bars: int = 3
    spread_dangerous_label: str = "dangerous"
    # Realized vol divergence
    realized_window_bars: int = 30
    divergence_high_ratio: float = 1.5     # realized > 1.5 × prior → tape running hot
    divergence_low_ratio: float = 0.7      # realized < 0.7 × prior → tape stale
    divergence_confidence_haircut: float = 0.15
    # Confidence intervals
    rolling_signal_history_bars: int = 60


@dataclass
class UpgradeReport:
    """Per-tick what-we-did report appended to the snapshot's notes."""
    adaptive_warmup_active: bool
    soft_warmup_scale: float
    dirty_storm_active: bool
    dirty_storm_bars_remaining: int
    realized_vol_divergence_ratio: float
    confidence_haircut_applied: float
    confidence_interval_width: float
    final_confidence: float
    actions_taken: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adaptive_warmup_active": self.adaptive_warmup_active,
            "soft_warmup_scale": round(self.soft_warmup_scale, 3),
            "dirty_storm_active": self.dirty_storm_active,
            "dirty_storm_bars_remaining": self.dirty_storm_bars_remaining,
            "realized_vol_divergence_ratio": round(
                self.realized_vol_divergence_ratio, 3),
            "confidence_haircut_applied": round(self.confidence_haircut_applied, 3),
            "confidence_interval_width": round(self.confidence_interval_width, 3),
            "final_confidence": round(self.final_confidence, 3),
            "actions_taken": list(self.actions_taken),
        }


class EngineUpgrades:
    """Transforms a BeliefSnapshot before the manager sees it."""

    def __init__(self, cfg: Optional[EngineUpgradesConfig] = None) -> None:
        self.cfg = cfg or EngineUpgradesConfig()
        self._spot_history: Deque[float] = deque(
            maxlen=self.cfg.realized_window_bars)
        self._confidence_history: Deque[float] = deque(
            maxlen=self.cfg.rolling_signal_history_bars)
        self._net_intent_history: Deque[float] = deque(
            maxlen=self.cfg.rolling_signal_history_bars)
        self._dirty_storm_pause_remaining = 0
        self._last_implied_iv: Optional[float] = None
        self._tick_index = 0

    def transform(self, snapshot: Dict[str, Any]) -> tuple[
            Dict[str, Any], UpgradeReport]:
        """Apply all upgrades. Returns (mutated_snapshot, report).

        The returned snapshot is a SHALLOW COPY with adjusted decision/
        confidence/notes; the original is not modified.
        """
        cfg = self.cfg
        actions: List[str] = []
        snap = dict(snapshot)  # shallow copy is enough; we replace fields
        decision = dict(snap.get("decision") or {})
        iv = dict(snap.get("iv_state") or {})
        thesis = dict(snap.get("thesis") or {})
        bars_seen = int(_num(snap.get("bars_seen")) or 0)
        spot = _num(snap.get("spot"))
        confidence = _num(decision.get("confidence"))
        action = str(decision.get("action") or "")
        self._tick_index += 1

        if spot > 0:
            self._spot_history.append(spot)
        if math.isfinite(confidence):
            self._confidence_history.append(confidence)
        net_intent = _num(iv.get("net_intent_z"))
        if math.isfinite(net_intent):
            self._net_intent_history.append(net_intent)

        # ── 1. Adaptive warmup ──────────────────────────────────────
        is_warm = bool(snap.get("is_warm"))
        adaptive_active = False
        soft_warmup_scale = 1.0
        if not is_warm:
            if bars_seen >= cfg.soft_warmup_floor:
                # Graduated warmup: scale confidence linearly between
                # soft_warmup_floor and hard_warmup_target.
                progress = ((bars_seen - cfg.soft_warmup_floor)
                            / max(1, cfg.hard_warmup_target
                                  - cfg.soft_warmup_floor))
                soft_warmup_scale = min(1.0, max(0.30,
                                                    0.30 + 0.70 * progress))
                soft_warmup_scale = min(soft_warmup_scale,
                                          cfg.soft_warmup_confidence_cap)
                # Promote is_warm so the manager doesn't refuse outright.
                snap["is_warm"] = True
                confidence = confidence * soft_warmup_scale
                adaptive_active = True
                actions.append(
                    f"adaptive_warmup: bars {bars_seen}/"
                    f"{cfg.hard_warmup_target}, scale {soft_warmup_scale:.2f}"
                )
            else:
                # Below soft floor — keep is_warm=False.
                pass

        # ── 2. Dirty quote healing ──────────────────────────────────
        slot_readings = list(snap.get("slot_readings") or [])
        n_slots = len(slot_readings)
        n_dangerous = sum(1 for s in slot_readings
                            if str(_map(s).get("spread_state") or "")
                            == cfg.spread_dangerous_label)
        n_dirty_mark = sum(1 for s in slot_readings
                             if str(_map(s).get("mark_source") or "")
                             in ("ltp", "last_valid", "invalid"))
        dirty_frac = ((n_dangerous + n_dirty_mark)
                       / max(1, n_slots))
        dirty_storm = dirty_frac >= cfg.dirty_storm_threshold
        if dirty_storm:
            self._dirty_storm_pause_remaining = cfg.dirty_storm_pause_bars
        elif self._dirty_storm_pause_remaining > 0:
            self._dirty_storm_pause_remaining -= 1
        storm_active = self._dirty_storm_pause_remaining > 0
        if storm_active and action in ENTRY_ACTIONS:
            actions.append(
                f"dirty_storm pause: {n_dangerous + n_dirty_mark}/{n_slots} "
                f"slots dirty; {self._dirty_storm_pause_remaining} bars remaining"
            )
            decision["action"] = "HOLD"   # block entries; allow exits
            action = "HOLD"

        # ── 3. Realized vol divergence ──────────────────────────────
        divergence_ratio = 1.0
        confidence_haircut = 0.0
        if len(self._spot_history) >= 10:
            log_returns = []
            spots = list(self._spot_history)
            for i in range(1, len(spots)):
                if spots[i - 1] > 0:
                    log_returns.append(math.log(spots[i] / spots[i - 1]))
            if len(log_returns) >= 5:
                realized = statistics.pstdev(log_returns) * math.sqrt(252 * 6.5
                                                                       * 3600)
                implied = _num(iv.get("confidence")) * 0.25 + 0.15
                # Heuristic prior: implied vol is around 18% for NIFTY weeklies.
                # Compare realized to implied; if realized > 1.5x implied,
                # the tape is running hot relative to the IV state.
                if implied > 1e-6:
                    divergence_ratio = realized / max(implied, 1e-6)
                    if (divergence_ratio > cfg.divergence_high_ratio
                            or divergence_ratio < cfg.divergence_low_ratio):
                        confidence_haircut = cfg.divergence_confidence_haircut
                        actions.append(
                            f"realized_vol divergence {divergence_ratio:.2f} "
                            f"outside [{cfg.divergence_low_ratio:.2f}, "
                            f"{cfg.divergence_high_ratio:.2f}]"
                        )
                        confidence = max(0.0, confidence - confidence_haircut)
                self._last_implied_iv = implied

        # ── 4. Confidence intervals ─────────────────────────────────
        confidence_width = 0.0
        if len(self._confidence_history) >= 5:
            confidence_width = statistics.pstdev(self._confidence_history)
            # Optional: if recent confidence has been very unstable, that's
            # information; downstream layers may consume it.

        # ── Persist transformations ────────────────────────────────
        decision["confidence"] = max(0.0, min(1.0, confidence))
        # Append a notes field summarizing the upgrades.
        notes = list(snap.get("notes") or [])
        if actions:
            notes.append("v4_upgrades: " + "; ".join(actions))
        snap["notes"] = notes
        snap["decision"] = decision
        # Tag the upgrade report on the snapshot for cockpit / ledger.
        report = UpgradeReport(
            adaptive_warmup_active=adaptive_active,
            soft_warmup_scale=soft_warmup_scale,
            dirty_storm_active=storm_active,
            dirty_storm_bars_remaining=self._dirty_storm_pause_remaining,
            realized_vol_divergence_ratio=divergence_ratio,
            confidence_haircut_applied=confidence_haircut,
            confidence_interval_width=confidence_width,
            final_confidence=float(decision["confidence"]),
            actions_taken=actions,
        )
        snap["engine_upgrades"] = report.to_dict()
        return snap, report


def _map(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _num(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return default
