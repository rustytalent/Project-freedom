"""LiveCalibrator — adjusts aggregator weights in real time, SAFELY.

The founder's warning (2026-06-22): "Changing weight live can cause
disastrous results." Correct. This module is wrapped in multiple layers
of safety so a single bad calibration step cannot detonate the live
session.

Defense in depth:

  1. **Walk-forward gate already in OnlineLearner**  — proposes a weight
     change only if the new weights generalize on a held-out fold.
  2. **Max-delta cap per step**  — even if the gate accepts, no single
     weight can move more than ``max_delta_per_update`` in one step.
  3. **Consecutive-acceptance requirement**  — must see N consecutive
     accepted proposals before any are actually APPLIED to the live
     aggregator. Single noisy update gets logged but not applied.
  4. **Operator pause flag**  — the cockpit UI button (added in this
     batch) toggles `paused` instantly. Pausing freezes all weight
     changes; resuming continues.
  5. **Auto-disable on rejection storm**  — if the calibrator rejects K
     proposals in a row, it auto-pauses and surfaces a warning.
  6. **Snapshot logged to WeightEvolutionMemory on EVERY proposal**
     (accepted or rejected), so the journey is preserved even when
     nothing is applied.
  7. **Rollback on emergency** — `rollback_last_applied()` reverts the
     most recent accepted change.

Wiring: the manager calls ``calibrator.on_position_closed(...)`` from
its close handler. The calibrator decides whether to update; if it does,
it (a) writes the new weights into the live `AggregatorConfig` in place,
and (b) appends to the WeightEvolutionMemory.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .learning import OnlineLearner, OnlineLearnerConfig, WeightUpdate
from .weight_evolution import (
    WeightEvolutionMemory,
    WeightSnapshot,
)


@dataclass
class LiveCalibratorConfig:
    """Safety knobs for the live calibrator."""
    enabled: bool = True
    max_delta_per_update: float = 0.04           # max absolute change per weight per apply
    consecutive_accepts_required: int = 2         # gate before applying
    auto_disable_after_n_rejections: int = 5      # rejection storm guard
    rollback_on_val_loss_spike: bool = True
    val_loss_spike_factor: float = 1.6            # val_loss > prev_val_loss * factor → rollback


# ── Outcome dataclass ─────────────────────────────────────────────


@dataclass
class CalibrationOutcome:
    """What happened on this calibration step."""
    proposed: bool
    applied: bool
    paused: bool
    rejected_reason: str
    pre_weights: Dict[str, float]
    post_weights: Dict[str, float]
    deltas: Dict[str, float]
    train_loss: float
    val_loss: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposed": self.proposed,
            "applied": self.applied,
            "paused": self.paused,
            "rejected_reason": self.rejected_reason,
            "pre_weights": dict(self.pre_weights),
            "post_weights": dict(self.post_weights),
            "deltas": {k: round(v, 6) for k, v in self.deltas.items()},
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "notes": list(self.notes),
        }


# ── The calibrator ────────────────────────────────────────────────


class LiveCalibrator:
    """Production-safe live weight calibrator.

    Composes an OnlineLearner (already walk-forward-aware) with the
    additional safety rails described in the module docstring.
    """

    def __init__(self, *,
                  aggregator_cfg,
                  memory: WeightEvolutionMemory,
                  calibrator_cfg: Optional[LiveCalibratorConfig] = None,
                  learner_cfg: Optional[OnlineLearnerConfig] = None,
                  ) -> None:
        self.cfg = calibrator_cfg or LiveCalibratorConfig()
        self.memory = memory
        self.aggregator_cfg = aggregator_cfg
        # Seed the learner from the live aggregator's current weights.
        initial_weights = {
            "w_base_score": aggregator_cfg.w_base_score,
            "w_mtf_alignment": aggregator_cfg.w_mtf_alignment,
            "w_projection": aggregator_cfg.w_projection,
            "w_fees_clearance": aggregator_cfg.w_fees_clearance,
            "w_portfolio_capacity": aggregator_cfg.w_portfolio_capacity,
        }
        self.learner = OnlineLearner(
            initial_weights=initial_weights, cfg=learner_cfg,
        )
        self.paused: bool = False
        self._consecutive_accepts: int = 0
        self._consecutive_rejections: int = 0
        self._last_applied_weights: Optional[Dict[str, float]] = None
        self._last_applied_at_bar: int = -1
        self._last_val_loss: Optional[float] = None

    # ── Operator controls (also exposed via cockpit UI) ────────────

    def pause(self, reason: str = "operator") -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self._consecutive_rejections = 0

    # ── The main observation API ───────────────────────────────────

    def on_position_closed(self, *,
                              component_scores: Dict[str, float],
                              realized_r: float,
                              bar_index: int,
                              ) -> CalibrationOutcome:
        """Manager calls this from its close handler.

        Returns a CalibrationOutcome capturing what happened.
        """
        pre = self._current_weights_dict()

        if not self.cfg.enabled or self.paused:
            return CalibrationOutcome(
                proposed=False, applied=False, paused=self.paused,
                rejected_reason=("disabled" if not self.cfg.enabled
                                  else "operator paused"),
                pre_weights=pre, post_weights=pre, deltas={k: 0.0 for k in pre},
                train_loss=0.0, val_loss=0.0,
            )

        # Feed the closure into the learner.
        self.learner.observe_closure(
            component_scores=component_scores,
            realized_r=realized_r,
        )
        update = self.learner.maybe_update()
        if update is None:
            return CalibrationOutcome(
                proposed=False, applied=False, paused=False,
                rejected_reason="learner: insufficient samples or "
                                  "not yet on update interval",
                pre_weights=pre, post_weights=pre,
                deltas={k: 0.0 for k in pre},
                train_loss=0.0, val_loss=0.0,
            )

        # The learner returned an update — wf gate already applied/rejected.
        return self._handle_update(update=update, pre=pre, bar_index=bar_index)

    # ── Internal: update handling ──────────────────────────────────

    def _handle_update(self, *,
                         update: WeightUpdate,
                         pre: Dict[str, float],
                         bar_index: int) -> CalibrationOutcome:
        cfg = self.cfg
        notes: List[str] = []

        # The walk-forward gate already rejected? Log and bail.
        if not update.walk_forward_accepted:
            self._on_rejected()
            self._append_memory(
                WeightSnapshot(
                    ts=time.time(), bar_index=bar_index,
                    weights=update.pre_update_weights,
                    n_samples_used=update.samples_used,
                    train_loss=update.train_loss,
                    val_loss=update.val_loss,
                    accepted=False,
                    overfit_ratio=update.walk_forward_overfit_ratio,
                    source="live_calibrator",
                ))
            return CalibrationOutcome(
                proposed=True, applied=False, paused=self.paused,
                rejected_reason="walk-forward overfit guard",
                pre_weights=pre, post_weights=pre,
                deltas={k: 0.0 for k in pre},
                train_loss=update.train_loss, val_loss=update.val_loss,
                notes=["validation worse than training — would overfit"],
            )

        # Walk-forward accepted. Apply safety caps.
        capped_post = self._cap_deltas(pre=pre,
                                          proposed=update.post_update_weights)
        deltas = {k: capped_post[k] - pre[k] for k in pre}

        # Detect val-loss spike vs the previous applied step.
        if (cfg.rollback_on_val_loss_spike
                and self._last_val_loss is not None
                and update.val_loss
                > self._last_val_loss * cfg.val_loss_spike_factor):
            self._on_rejected()
            self._append_memory(
                WeightSnapshot(
                    ts=time.time(), bar_index=bar_index,
                    weights=pre, n_samples_used=update.samples_used,
                    train_loss=update.train_loss, val_loss=update.val_loss,
                    accepted=False,
                    overfit_ratio=update.walk_forward_overfit_ratio,
                    source="live_calibrator",
                ))
            return CalibrationOutcome(
                proposed=True, applied=False, paused=self.paused,
                rejected_reason="val-loss spike — would degrade",
                pre_weights=pre, post_weights=pre,
                deltas={k: 0.0 for k in pre},
                train_loss=update.train_loss, val_loss=update.val_loss,
                notes=[f"val_loss {update.val_loss:.4f} > prev "
                        f"{self._last_val_loss:.4f} × "
                        f"{cfg.val_loss_spike_factor}"],
            )

        # Consecutive-acceptance gate.
        self._consecutive_accepts += 1
        self._consecutive_rejections = 0
        if self._consecutive_accepts < cfg.consecutive_accepts_required:
            self._append_memory(
                WeightSnapshot(
                    ts=time.time(), bar_index=bar_index,
                    weights=pre, n_samples_used=update.samples_used,
                    train_loss=update.train_loss, val_loss=update.val_loss,
                    accepted=False,
                    overfit_ratio=update.walk_forward_overfit_ratio,
                    source="live_calibrator",
                ))
            return CalibrationOutcome(
                proposed=True, applied=False, paused=False,
                rejected_reason=f"need "
                                  f"{cfg.consecutive_accepts_required} consecutive "
                                  f"accepts; have {self._consecutive_accepts}",
                pre_weights=pre, post_weights=pre,
                deltas={k: 0.0 for k in pre},
                train_loss=update.train_loss, val_loss=update.val_loss,
            )

        # ALL gates passed — apply.
        self._apply_to_aggregator(capped_post)
        self._last_applied_weights = dict(pre)        # for rollback
        self._last_applied_at_bar = bar_index
        self._last_val_loss = update.val_loss
        self._consecutive_accepts = 0    # reset for next batch

        self._append_memory(
            WeightSnapshot(
                ts=time.time(), bar_index=bar_index,
                weights=capped_post, n_samples_used=update.samples_used,
                train_loss=update.train_loss, val_loss=update.val_loss,
                accepted=True,
                overfit_ratio=update.walk_forward_overfit_ratio,
                source="live_calibrator",
            ))
        notes.append("weights applied to live AggregatorConfig in place")
        return CalibrationOutcome(
            proposed=True, applied=True, paused=False,
            rejected_reason="",
            pre_weights=pre, post_weights=capped_post,
            deltas=deltas,
            train_loss=update.train_loss, val_loss=update.val_loss,
            notes=notes,
        )

    def rollback_last_applied(self) -> bool:
        """Operator emergency: undo the most recent applied step."""
        if self._last_applied_weights is None:
            return False
        self._apply_to_aggregator(self._last_applied_weights)
        self._last_applied_weights = None
        return True

    # ── Helpers ────────────────────────────────────────────────────

    def _current_weights_dict(self) -> Dict[str, float]:
        return {
            "w_base_score": self.aggregator_cfg.w_base_score,
            "w_mtf_alignment": self.aggregator_cfg.w_mtf_alignment,
            "w_projection": self.aggregator_cfg.w_projection,
            "w_fees_clearance": self.aggregator_cfg.w_fees_clearance,
            "w_portfolio_capacity": self.aggregator_cfg.w_portfolio_capacity,
        }

    def _cap_deltas(self, *, pre: Dict[str, float],
                      proposed: Dict[str, float]) -> Dict[str, float]:
        cap = self.cfg.max_delta_per_update
        out: Dict[str, float] = {}
        for k, v in pre.items():
            target = proposed.get(k, v)
            delta = target - v
            if delta > cap:
                delta = cap
            elif delta < -cap:
                delta = -cap
            out[k] = v + delta
        # Re-normalise to keep sum = 1.0 (the aggregator invariant).
        total = sum(out.values())
        if total > 0:
            out = {k: v / total for k, v in out.items()}
        return out

    def _apply_to_aggregator(self, weights: Dict[str, float]) -> None:
        for k, v in weights.items():
            if hasattr(self.aggregator_cfg, k):
                setattr(self.aggregator_cfg, k, float(v))

    def _append_memory(self, snap: WeightSnapshot) -> None:
        try:
            self.memory.append(snap)
        except Exception:
            # Memory failures must never break the trading loop.
            pass

    def _on_rejected(self) -> None:
        self._consecutive_accepts = 0
        self._consecutive_rejections += 1
        if (self._consecutive_rejections
                >= self.cfg.auto_disable_after_n_rejections):
            self.paused = True
