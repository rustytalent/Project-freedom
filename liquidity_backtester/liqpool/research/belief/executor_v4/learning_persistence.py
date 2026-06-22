"""Multi-day persistence for the live calibrator (founder 2026-06-22).

The original LiveCalibrator + OnlineLearner pair is session-local: every
container restart wipes ``OnlineLearner.observations`` and re-seeds
weights from the prior ``AggregatorConfig``. This means today's session
starts knowing exactly as much as yesterday's session ended NOT knowing.

This module fixes that. It writes a per-IST-date file containing:

  1. The learner's current weights and observation buffer.
  2. Every closed-position outcome augmented with a *regime tag*
     (dominant family + chop/manipulation/tail masses + horizon-weighted
     consensus at close time).

On startup ``CalibrationStateStore.bootstrap(...)`` reads the last N
days, replays the observations chronologically through the learner, and
applies the resulting weights to the live ``AggregatorConfig`` once.

The companion ``RegimeWinHistory`` aggregates yesterday's outcomes by
scenario family so today's confidence floor can be nudged by how each
regime did historically — looser when yesterday was kind to today's
regime, tighter when it wasn't.

Wire-points (manager.py):

  * Manager init → instantiate ``CalibrationStateStore``; if a prior
    file exists, call ``bootstrap(...)``. Result goes into
    ``portfolio_summary['calibrator_bootstrap']``.
  * On every position close, manager calls ``store.record_closure(...)``
    with the closure's component scores + realized R + the live
    web_snap dict so we can tag with the current regime.
  * Each tick the manager surfaces ``regime_history.today_summary(...)``
    in ``portfolio_summary['regime_win_history']``.

The store is intentionally crash-safe: writes are tempfile-then-rename,
flush+fsync, dated per IST. If anything goes wrong we degrade silently
so we never break the trading loop.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


IST = timezone(timedelta(hours=5, minutes=30))


def _today_ist() -> str:
    return datetime.now(IST).date().isoformat()


def _now_iso() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


# ── Regime tag dataclass ──────────────────────────────────────────


def regime_tag_from_web_snapshot(web_snap_dict: Optional[Dict[str, Any]],
                                    ) -> Dict[str, Any]:
    """Extract the bits of the web snapshot the regime-history layer
    cares about. Pure function; safe to call with None."""
    w = web_snap_dict or {}
    fam_mass = w.get("family_mass") or {}
    # The dominant family is the one with the highest mass. We avoid
    # picking 'directional' when it's near-zero by treating ties as
    # 'unknown' for cleanliness.
    if fam_mass:
        items = sorted(fam_mass.items(), key=lambda kv: kv[1], reverse=True)
        top_fam, top_mass = items[0]
        if top_mass < 1e-3:
            dominant = "unknown"
        else:
            dominant = str(top_fam)
    else:
        dominant = "unknown"
    return {
        "dominant_family": dominant,
        "chop_mass": float(w.get("chop_mass", 0.0)),
        "manipulation_mass": float(w.get("manipulation_mass", 0.0)),
        "tail_mass": float(w.get("tail_mass", 0.0)),
        "directional_consensus_horizon_weighted": float(
            w.get("directional_consensus_horizon_weighted", 0.0)),
        "directional_consensus_raw": float(
            w.get("directional_consensus", 0.0)),
    }


# ── On-disk schema ────────────────────────────────────────────────


@dataclass
class _DailyRecord:
    """One day's calibrator persistence file."""
    session_date: str
    last_updated_at: str
    weights_at_close: Dict[str, float]
    n_observations: int
    n_closures_recorded: int
    closures: List[Dict[str, Any]]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_date": self.session_date,
            "last_updated_at": self.last_updated_at,
            "weights_at_close": dict(self.weights_at_close),
            "n_observations": self.n_observations,
            "n_closures_recorded": self.n_closures_recorded,
            "closures": list(self.closures),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_DailyRecord":
        return cls(
            session_date=str(d.get("session_date", "")),
            last_updated_at=str(d.get("last_updated_at", "")),
            weights_at_close=dict(d.get("weights_at_close") or {}),
            n_observations=int(d.get("n_observations", 0)),
            n_closures_recorded=int(d.get("n_closures_recorded", 0)),
            closures=list(d.get("closures") or []),
            notes=list(d.get("notes") or []),
        )


@dataclass
class BootstrapResult:
    """What the calibrator inherited from prior days on startup."""
    ran: bool
    days_loaded: int
    observations_replayed: int
    updates_applied: int
    seeded_weights: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ran": self.ran,
            "days_loaded": self.days_loaded,
            "observations_replayed": self.observations_replayed,
            "updates_applied": self.updates_applied,
            "seeded_weights": dict(self.seeded_weights),
            "notes": list(self.notes),
        }


# ── The store ─────────────────────────────────────────────────────


class CalibrationStateStore:
    """Reads/writes per-IST-date calibrator state to disk.

    State directory layout::

        <state_dir>/calibrator_state_<YYYY-MM-DD>.json

    Files older than ``retain_days`` are not deleted (the operator may
    want to keep them); they're just not loaded by ``bootstrap``.
    """

    FILENAME_PREFIX = "calibrator_state_"
    FILENAME_SUFFIX = ".json"

    def __init__(self, *, state_dir: Path,
                  retain_days: int = 30,
                  bootstrap_days: int = 5) -> None:
        self.state_dir = Path(state_dir)
        self.retain_days = int(retain_days)
        self.bootstrap_days = int(bootstrap_days)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._today: Optional[_DailyRecord] = None

    # ── Path helpers ───────────────────────────────────────────────

    def path_for(self, date_str: str) -> Path:
        return self.state_dir / f"{self.FILENAME_PREFIX}{date_str}{self.FILENAME_SUFFIX}"

    def path_for_today(self) -> Path:
        return self.path_for(_today_ist())

    def prior_dates(self, n_days: int) -> List[str]:
        """Return the prior n_days date strings (newest first, excluding today)."""
        today = datetime.now(IST).date()
        out: List[str] = []
        for i in range(1, n_days + 1):
            d = today - timedelta(days=i)
            out.append(d.isoformat())
        return out

    # ── Read API ───────────────────────────────────────────────────

    def load_day(self, date_str: str) -> Optional[_DailyRecord]:
        path = self.path_for(date_str)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return _DailyRecord.from_dict(json.load(f))
        except Exception:
            return None

    def load_prior(self, n_days: int) -> List[_DailyRecord]:
        """Load up to n_days of prior records, oldest first.

        Tolerates gaps (weekends, missed days). Caller decides whether to
        replay or weight them — we just hand back what's on disk.
        """
        records: List[_DailyRecord] = []
        for date_str in reversed(self.prior_dates(n_days)):
            r = self.load_day(date_str)
            if r is not None:
                records.append(r)
        return records

    # ── Write API ──────────────────────────────────────────────────

    def _ensure_today(self) -> _DailyRecord:
        if self._today is not None:
            return self._today
        existing = self.load_day(_today_ist())
        if existing is not None:
            self._today = existing
        else:
            self._today = _DailyRecord(
                session_date=_today_ist(),
                last_updated_at=_now_iso(),
                weights_at_close={},
                n_observations=0,
                n_closures_recorded=0,
                closures=[],
            )
        return self._today

    def record_closure(self, *,
                          component_scores: Dict[str, float],
                          realized_r: float,
                          regime_tag: Dict[str, Any],
                          bar_index: int,
                          strategy: str = "",
                          ) -> None:
        """Append one closure to today's record and flush to disk."""
        rec = self._ensure_today()
        rec.closures.append({
            "ts": _now_iso(),
            "bar_index": int(bar_index),
            "strategy": str(strategy),
            "component_scores": {k: float(v) for k, v in
                                  (component_scores or {}).items()},
            "realized_r": float(realized_r),
            "regime_tag": dict(regime_tag or {}),
        })
        rec.n_closures_recorded = len(rec.closures)
        rec.last_updated_at = _now_iso()
        self._flush()

    def record_weights(self, *, weights: Dict[str, float],
                         n_observations: int) -> None:
        rec = self._ensure_today()
        rec.weights_at_close = {k: float(v) for k, v in weights.items()}
        rec.n_observations = int(n_observations)
        rec.last_updated_at = _now_iso()
        self._flush()

    def _flush(self) -> None:
        if self._today is None:
            return
        path = self.path_for_today()
        body = json.dumps(self._today.to_dict(), default=str, sort_keys=True)
        try:
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
        except Exception:
            # Persistence failures must never crash the trading loop.
            pass

    # ── Bootstrap ──────────────────────────────────────────────────

    def bootstrap(self, *, learner, aggregator_cfg,
                    days: Optional[int] = None,
                    force_update: bool = True,
                    ) -> BootstrapResult:
        """Replay prior days' closures through the learner.

        ``learner`` is the live OnlineLearner instance. We feed every
        prior closure through ``observe_closure`` (chronologically) and
        then force a ``maybe_update`` so the seeded weights land on the
        aggregator before today's trading starts.

        If ``aggregator_cfg`` is provided AND the learner accepted any
        updates, we push the learner weights into the aggregator.
        """
        n_days = self.bootstrap_days if days is None else int(days)
        records = self.load_prior(n_days)
        notes: List[str] = []

        if not records:
            notes.append(
                f"no prior calibrator state in last {n_days} days — "
                f"starting cold (this is normal for the first session)")
            return BootstrapResult(
                ran=False, days_loaded=0, observations_replayed=0,
                updates_applied=0, seeded_weights={}, notes=notes,
            )

        replayed = 0
        for rec in records:
            for closure in rec.closures:
                cs = closure.get("component_scores") or {}
                r = float(closure.get("realized_r", 0.0))
                try:
                    learner.observe_closure(
                        component_scores=cs, realized_r=r,
                    )
                    replayed += 1
                except Exception:
                    continue

        # Force-update: lower the update_every gate temporarily so we
        # actually apply the accumulated history (otherwise we'd be
        # waiting for the very next live close before the historical
        # gradient lands). We restore the gate after.
        updates_applied = 0
        if force_update and replayed >= max(1, learner.cfg.min_samples_per_update):
            orig_every = learner.cfg.update_every_n_closures
            learner.cfg.update_every_n_closures = 1
            try:
                update = learner.maybe_update()
                if update is not None and update.walk_forward_accepted:
                    updates_applied = 1
                    notes.append(
                        f"bootstrap update accepted: train_loss="
                        f"{update.train_loss:.4f} val_loss="
                        f"{update.val_loss:.4f}")
                elif update is not None:
                    notes.append(
                        f"bootstrap update REJECTED by walk-forward gate "
                        f"(overfit ratio {update.walk_forward_overfit_ratio:.2f}) — "
                        f"keeping prior weights")
                else:
                    notes.append(
                        "bootstrap: not enough samples to seed update")
            finally:
                learner.cfg.update_every_n_closures = orig_every

        # Apply learned weights to live aggregator (in-place).
        seeded_weights = dict(learner.weights)
        if updates_applied and aggregator_cfg is not None:
            for k, v in learner.weights.items():
                if hasattr(aggregator_cfg, k):
                    setattr(aggregator_cfg, k, float(v))
            notes.append("seeded weights pushed into live AggregatorConfig")

        notes.append(
            f"loaded {len(records)} day(s); replayed {replayed} closures")

        return BootstrapResult(
            ran=True,
            days_loaded=len(records),
            observations_replayed=replayed,
            updates_applied=updates_applied,
            seeded_weights=seeded_weights,
            notes=notes,
        )


# ── Regime-aware confidence adjustment ────────────────────────────


@dataclass
class _FamilyStats:
    n_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_realized_r: float = 0.0
    total_realized_rupees: float = 0.0

    def add(self, realized_r: float, realized_rupees: float,
              win_threshold_r: float) -> None:
        self.n_trades += 1
        if realized_r >= win_threshold_r:
            self.wins += 1
        else:
            self.losses += 1
        self.total_realized_r += realized_r
        self.total_realized_rupees += realized_rupees

    def summary(self) -> Dict[str, Any]:
        n = max(1, self.n_trades)
        return {
            "n_trades": self.n_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.wins / n, 3),
            "avg_realized_r": round(self.total_realized_r / n, 3),
            "total_realized_rupees": round(self.total_realized_rupees, 2),
        }


class RegimeWinHistory:
    """Aggregates prior-day closures by regime family and produces a
    confidence-floor adjustment for today.

    Philosophy: if yesterday's chop-dominant regime was an ER bath, and
    today's web is also chop-dominant, we should be *more* selective. If
    yesterday's tail-dominant regime printed money, we should be a touch
    *less* selective when today's web looks tail-dominant.

    The adjustment is a small additive nudge to the entry-confidence
    floor (positive = tighter floor; negative = looser). Bounded
    +/- ``max_adjustment`` so a single bad day can't fully shut us off.
    """

    def __init__(self, *, win_threshold_r: float = 0.20,
                  max_adjustment: float = 0.08) -> None:
        self.win_threshold_r = float(win_threshold_r)
        self.max_adjustment = float(max_adjustment)
        self._by_family: Dict[str, _FamilyStats] = {}
        self._days_loaded: int = 0

    # ── Population ────────────────────────────────────────────────

    def load_from_records(self, records: List[_DailyRecord]) -> None:
        self._by_family.clear()
        self._days_loaded = len(records)
        for rec in records:
            for closure in rec.closures:
                tag = closure.get("regime_tag") or {}
                fam = str(tag.get("dominant_family") or "unknown")
                r = float(closure.get("realized_r", 0.0))
                rupees = float(closure.get("realized_rupees", 0.0))
                stats = self._by_family.setdefault(fam, _FamilyStats())
                stats.add(r, rupees, self.win_threshold_r)

    def add_closure(self, *, regime_tag: Dict[str, Any],
                       realized_r: float,
                       realized_rupees: float = 0.0) -> None:
        """Live-closure feedback — keeps today's running totals current."""
        fam = str((regime_tag or {}).get("dominant_family") or "unknown")
        stats = self._by_family.setdefault(fam, _FamilyStats())
        stats.add(realized_r, realized_rupees, self.win_threshold_r)

    # ── Read API ──────────────────────────────────────────────────

    def confidence_adjustment_for(self, dominant_family: str) -> float:
        """Return the additive nudge for the current dominant family.

        Negative avgR + non-trivial sample size → tighter floor (positive).
        Positive avgR → looser floor (negative).
        Zero when we have no signal.
        """
        stats = self._by_family.get(str(dominant_family))
        if stats is None or stats.n_trades < 3:
            return 0.0
        avg_r = stats.total_realized_r / max(1, stats.n_trades)
        # Map avgR ∈ [-1, +1] → adjustment ∈ [+max, -max]. Sign-flipped.
        nudge = -avg_r * self.max_adjustment
        return max(-self.max_adjustment, min(self.max_adjustment, nudge))

    def today_summary(self, *, today_dominant_family: str = ""
                         ) -> Dict[str, Any]:
        by_family = {k: v.summary() for k, v in self._by_family.items()}
        notes: List[str] = []
        if not self._by_family:
            notes.append(
                "no regime history yet — calibrator runs unmodulated today")
        adjustment = (self.confidence_adjustment_for(today_dominant_family)
                      if today_dominant_family else 0.0)
        if adjustment != 0.0:
            sign = "tighter" if adjustment > 0 else "looser"
            notes.append(
                f"today's regime ({today_dominant_family}) → confidence "
                f"floor nudged {sign} by {abs(adjustment):.2f}")
        return {
            "days_loaded": self._days_loaded,
            "by_family": by_family,
            "today_dominant_family": today_dominant_family,
            "today_confidence_adjustment": round(adjustment, 4),
            "notes": notes,
        }
