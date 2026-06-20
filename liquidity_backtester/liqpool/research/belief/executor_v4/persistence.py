"""Persistence layer — survive container restarts mid-day.

The founder's Monday session can be interrupted by anything: container
restart, network blip, manual stop/start. This module snapshots the
manager's critical state to disk after every tick and reloads it on
startup, so the executor wakes up with all open positions, ledger
history, and daily P&L intact.

What's persisted:
  * Daily P&L + cumulative fees
  * Open positions (hypothesis dict + state)
  * Closed positions (ledger outcomes)
  * Projection tape
  * Per-position counterfactual plans

Format: a single JSON document per session day, atomically replaced
on each write (write-to-tempfile + rename). The format is intentionally
plain JSON (not pickle) so the founder can inspect or repair it by
hand.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class PersistenceConfig:
    """Knobs for the persistence layer."""
    state_dir: Path = Path("/var/lib/sentinel/executor_v4_state")
    enabled: bool = True
    write_every_tick: bool = True
    write_only_on_change: bool = True


@dataclass
class ManagerStateSnapshot:
    """Compact serialization of the manager's persistable state."""
    session_date: str               # IST date string
    last_bar_index: int
    last_ts: str
    daily_pnl_rupees: float
    cumulative_fees_rupees: float
    cooldown_until_bar: int
    open_positions: List[Dict[str, Any]]
    closed_positions: List[Dict[str, Any]]
    projection_tape: List[Dict[str, Any]]
    counterfactual_plans: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManagerStateSnapshot":
        return cls(
            session_date=str(d.get("session_date", "")),
            last_bar_index=int(d.get("last_bar_index", 0)),
            last_ts=str(d.get("last_ts", "")),
            daily_pnl_rupees=float(d.get("daily_pnl_rupees", 0.0)),
            cumulative_fees_rupees=float(d.get("cumulative_fees_rupees", 0.0)),
            cooldown_until_bar=int(d.get("cooldown_until_bar", -1)),
            open_positions=list(d.get("open_positions") or []),
            closed_positions=list(d.get("closed_positions") or []),
            projection_tape=list(d.get("projection_tape") or []),
            counterfactual_plans=dict(d.get("counterfactual_plans") or {}),
            metadata=dict(d.get("metadata") or {}),
        )


class ManagerPersistence:
    """Reads/writes ManagerStateSnapshot atomically."""

    def __init__(self, cfg: Optional[PersistenceConfig] = None) -> None:
        self.cfg = cfg or PersistenceConfig()
        self._last_hash: Optional[int] = None
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)

    def path_for_today(self) -> Path:
        d = datetime.now(IST).date().isoformat()
        return self.cfg.state_dir / f"manager_state_{d}.json"

    def load_today(self) -> Optional[ManagerStateSnapshot]:
        """Load today's snapshot if it exists, else return None."""
        if not self.cfg.enabled:
            return None
        path = self.path_for_today()
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                d = json.load(f)
        except Exception:
            return None
        return ManagerStateSnapshot.from_dict(d)

    def write(self, snap: ManagerStateSnapshot) -> bool:
        """Atomically write the snapshot. Returns True if a write occurred."""
        if not self.cfg.enabled:
            return False
        body = json.dumps(snap.to_dict(), default=str, sort_keys=True)
        h = hash(body)
        if self.cfg.write_only_on_change and h == self._last_hash:
            return False
        path = self.path_for_today()
        # Write to tempfile then rename atomically.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(path.parent),
            prefix=path.stem + ".",
            suffix=".tmp", delete=False,
        ) as tmp:
            tmp.write(body)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
        self._last_hash = h
        return True


def capture_state(manager) -> ManagerStateSnapshot:
    """Build a ManagerStateSnapshot from a PortfolioManager instance.

    Pure function — does NOT modify the manager. Re-runs each tick.
    """
    open_positions_data: List[Dict[str, Any]] = []
    for pid, state in manager._open_states.items():
        h = state.hypothesis
        open_positions_data.append({
            "position_id": pid,
            "hypothesis": h.to_dict(),
            "entry_bar": state.entry_bar,
            "last_bar": state.last_bar,
            "best_r": state.best_r,
            "worst_r": state.worst_r,
            "high_water_confidence": state.high_water_confidence,
            "low_water_confidence": state.low_water_confidence,
            "last_premium": state.last_premium,
        })
    closed_positions_data: List[Dict[str, Any]] = []
    for ledger in manager.ledger_store.closed_positions():
        closed_positions_data.append({
            "hypothesis": ledger.hypothesis.to_dict(),
            "outcome": ledger.outcome.to_dict() if ledger.outcome else None,
            "n_bar_records": len(ledger.bar_records),
        })
    projection_tape = [r.to_dict() for r in manager.projection.tape]
    cf_plans = {pid: plan.to_dict()
                 for pid, plan in manager._counterfactual_plans.items()}

    return ManagerStateSnapshot(
        session_date=datetime.now(IST).date().isoformat(),
        last_bar_index=manager.substrate_state.bar_index,
        last_ts=datetime.now(IST).isoformat(timespec="seconds"),
        daily_pnl_rupees=manager.daily_pnl_rupees,
        cumulative_fees_rupees=manager.cumulative_fees_rupees,
        cooldown_until_bar=manager.cooldown_until_bar,
        open_positions=open_positions_data,
        closed_positions=closed_positions_data,
        projection_tape=projection_tape,
        counterfactual_plans=cf_plans,
        metadata={},
    )


def restore_pnl_only(manager, snap: ManagerStateSnapshot) -> None:
    """Restore the safe-subset of state that doesn't require rebuilding
    PositionHypothesis / Ledger objects.

    Restores: daily_pnl, cumulative_fees, cooldown_until_bar, projection
    tape. Does NOT restore open positions or closed ledger — those are
    high-fidelity dataclasses and the caller should rebuild them via
    a hot-fix path (a Sprint+ task; today's safe path is "restart-warm
    P&L counters and let the engine warm up cleanly").
    """
    manager.daily_pnl_rupees = snap.daily_pnl_rupees
    manager.cumulative_fees_rupees = snap.cumulative_fees_rupees
    manager.cooldown_until_bar = max(-1, snap.cooldown_until_bar - 5)
