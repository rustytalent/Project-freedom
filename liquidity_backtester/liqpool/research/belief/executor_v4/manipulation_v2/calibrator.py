"""OutcomeCalibrator — wraps the SelfCalibrator with engine-level helpers.

This is the surface the manager uses. We:

  * Snapshot the per-detector posteriors at entry time (so we know
    which detectors fired when the position was opened).
  * On position close, attribute the realised R back to those snapshots
    and feed the SelfCalibrator.
  * Surface a cockpit-ready summary of per-detector weights + accuracy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from .detectors.base import DetectorPosterior, SelfCalibrator


@dataclass
class _EntrySnapshot:
    position_id: str
    bar_index: int
    intent_to_dict: Dict[str, Any]
    per_detector: Dict[str, DetectorPosterior]


class OutcomeCalibrator:
    """Engine-level calibration glue."""

    def __init__(self, *, detector_names: List[str],
                  learning_rate: float = 0.04,
                  min_fires_before_learning: int = 10,
                  ) -> None:
        self._self_calib = SelfCalibrator(
            detector_names=list(detector_names),
            learning_rate=learning_rate,
            min_fires_before_learning=min_fires_before_learning,
        )
        # Bounded snapshot buffer keyed by position_id.
        self._snapshots: Dict[str, _EntrySnapshot] = {}
        self._recent_close_log: Deque[Dict[str, Any]] = deque(maxlen=32)

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._self_calib.weights)

    def snapshot_at_entry(self, *,
                              position_id: str,
                              bar_index: int,
                              intent_dict: Dict[str, Any],
                              per_detector: Dict[str, DetectorPosterior],
                              ) -> None:
        # Cap stored snapshots so we never grow unbounded if positions
        # leak from the manager. 256 simultaneous open positions is
        # comfortably bigger than any sane portfolio.
        if len(self._snapshots) > 256:
            # Drop oldest by bar_index.
            oldest = min(self._snapshots.values(),
                          key=lambda s: s.bar_index)
            self._snapshots.pop(oldest.position_id, None)
        self._snapshots[position_id] = _EntrySnapshot(
            position_id=position_id,
            bar_index=int(bar_index),
            intent_to_dict=dict(intent_dict or {}),
            per_detector=dict(per_detector or {}),
        )

    def attribute_close(self, *,
                            position_id: str,
                            realised_r: float,
                            ) -> Optional[Dict[str, Any]]:
        snap = self._snapshots.pop(position_id, None)
        if snap is None:
            return None
        self._self_calib.record_outcome(
            per_detector_posteriors=snap.per_detector,
            realised_r=realised_r,
        )
        log_entry = {
            "position_id": position_id,
            "realised_r": float(realised_r),
            "intent_at_entry": snap.intent_to_dict,
        }
        self._recent_close_log.append(log_entry)
        return log_entry

    def summary(self) -> Dict[str, Any]:
        out = self._self_calib.summary()
        out["snapshots_held"] = len(self._snapshots)
        out["recent_attributions"] = list(self._recent_close_log)[-8:]
        return out
