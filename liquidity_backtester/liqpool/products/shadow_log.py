"""Shadow log — the counterfactual flywheel.

Every gate in the engine that DECLINES an action is a paired training
row we throw away today: the SKIP decision in the options executor,
the avoidance flag on a stock the basket actually rewarded, the pool
below ``min_target_to_cost_ratio``, the macro-gate-closed day where
the basket trended, the Q < threshold pool that was technically a
candidate. Each one has the answer attached to it (what would the
trade have done?), and each one is data for a future model:

  * SKIP regret estimator (Stream M.4)
  * Detector trust router (Stream M.5)
  * Drift-imminent model (Stream M.6)
  * Bucket-aging model (Stream M.8)

The collector is one append-only Parquet stream, partitioned by
(trading_date_ist, event_kind). The interface is a single
``ShadowLogger.record(event_kind, payload)`` call site; integration
points throughout the engine emit events through it. A separate
nightly resolver (``analysis/replay_shadow_counterfactuals.py``)
joins each shadow row to its counterfactual outcome the next session.

Schema versioning is additive-only — same discipline as outcome_log.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Event kinds — the named categories every gate writes under.
# Adding a new kind is a string-literal change; new readers/writers
# can ignore unknown kinds without breaking.
# ---------------------------------------------------------------------------

EVENT_KIND_SKIP_OPTIONS = "skip_options_executor"
EVENT_KIND_AVOIDANCE_FLAG = "avoidance_flag"
EVENT_KIND_BELOW_TARGET_TO_COST = "below_target_to_cost_ratio"
EVENT_KIND_MACRO_GATE_CLOSED = "macro_gate_closed"
EVENT_KIND_Q_BELOW_THRESHOLD = "q_below_threshold"
EVENT_KIND_DRIFT_FLAG_FIRED = "drift_flag_fired"
EVENT_KIND_ALPHA_SUPPRESSED = "alpha_suppressed_by_meta"
EVENT_KIND_POOL_MERGED = "pool_merged_in_builder"

KNOWN_EVENT_KINDS = frozenset({
    EVENT_KIND_SKIP_OPTIONS,
    EVENT_KIND_AVOIDANCE_FLAG,
    EVENT_KIND_BELOW_TARGET_TO_COST,
    EVENT_KIND_MACRO_GATE_CLOSED,
    EVENT_KIND_Q_BELOW_THRESHOLD,
    EVENT_KIND_DRIFT_FLAG_FIRED,
    EVENT_KIND_ALPHA_SUPPRESSED,
    EVENT_KIND_POOL_MERGED,
})


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class ShadowEvent:
    """One declined-decision row.

    The schema mirrors PredictionRecord's spirit: brand-stable IDs for
    later joining, the IST trading date as the partition key, a free-
    form context dict that lets each gate carry whatever it needs.

    The ``decision_context`` payload is intentionally open-ended so
    different gates can carry different shapes without a schema
    explosion — but every payload SHOULD include enough features to
    replay the counterfactual outcome. The resolver only needs the
    symbol + side + entry-reference price + a horizon; everything
    else is for the eventual model.
    """
    shadow_id: str
    event_kind: str
    trading_date_ist: str
    written_at_utc: str
    symbol: Optional[str]
    # The actual decision payload. Free-form JSON. Stored as a string
    # in Parquet to avoid nested-column schema explosion.
    decision_context: Dict[str, Any]
    # Tags to enable bucketed analysis without parsing the JSON.
    regime_tags: List[str] = field(default_factory=list)


@dataclass
class ShadowResolution:
    """The counterfactual outcome for one shadow event.

    Written by the nightly replay job. Keyed by ``shadow_id`` so the
    join is unambiguous. ``counterfactual_outcome`` is a JSON blob
    whose shape depends on event_kind — e.g. for a SKIP we record
    what the trade's net_r would have been; for an avoidance flag
    we record whether the basket actually had a winner.
    """
    shadow_id: str
    resolved_at_utc: str
    trading_date_ist_resolved: str
    counterfactual_outcome: Dict[str, Any]


# ---------------------------------------------------------------------------
# Deterministic IDs — let the same gate write the same row twice
# without producing duplicates. Pattern mirrors make_prediction_id().
# ---------------------------------------------------------------------------

def make_shadow_id(trading_date_ist: str, event_kind: str,
                   symbol: Optional[str], detail_token: str) -> str:
    """Stable across runs. The detail_token is whatever the caller
    chooses to disambiguate (a pool_id, a strike, a level). The
    same (date, kind, symbol, detail) always hashes to the same ID,
    so retrying a run is idempotent.
    """
    s = f"SHADOW_{trading_date_ist}_{event_kind}_{symbol or '_'}_{detail_token}"
    # Use a UUID5 over a stable namespace so the ID is deterministic
    # but still URL-safe and collision-resistant.
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class ShadowLogger:
    """Append-only writer with the same atomicity discipline as the
    outcome log: in-memory accumulation; atomic Parquet write on
    ``commit()``; idempotent across runs via shadow_id deduplication.

    Storage layout::

        <root>/
          shadow_events/
            trading_date_ist=2026-06-09/event_kind=skip_options_executor/events.parquet
          shadow_resolutions/
            trading_date_ist=2026-06-10/resolutions.parquet

    Resolutions are partitioned by the date they were RESOLVED on, not
    the date of the original event — most resolutions land the next
    trading day, but some (longer-horizon counterfactuals) land later.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._events: List[ShadowEvent] = []
        self._resolutions: List[ShadowResolution] = []

    # -- write API --------------------------------------------------

    def record(
        self,
        event_kind: str,
        trading_date_ist: str,
        symbol: Optional[str],
        detail_token: str,
        decision_context: Dict[str, Any],
        regime_tags: Optional[List[str]] = None,
        written_at_utc: Optional[str] = None,
    ) -> str:
        """Buffer one shadow event. Returns the shadow_id for the
        caller to remember if they need to correlate later.

        Unknown event_kind is allowed (additive schema) but a warning
        is the appropriate signal — for now we accept silently so
        new gates can land without coordinated writer updates."""
        if not isinstance(decision_context, dict):
            raise TypeError(
                f"decision_context must be a dict; got {type(decision_context)}"
            )
        shadow_id = make_shadow_id(
            trading_date_ist, event_kind, symbol, detail_token
        )
        ts = written_at_utc or pd.Timestamp.now("UTC").strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self._events.append(ShadowEvent(
            shadow_id=shadow_id,
            event_kind=event_kind,
            trading_date_ist=trading_date_ist,
            written_at_utc=ts,
            symbol=symbol,
            decision_context=decision_context,
            regime_tags=list(regime_tags or []),
        ))
        return shadow_id

    def record_resolution(
        self,
        shadow_id: str,
        trading_date_ist_resolved: str,
        counterfactual_outcome: Dict[str, Any],
        resolved_at_utc: Optional[str] = None,
    ) -> None:
        """Buffer the counterfactual outcome for one shadow event.
        Called by the nightly replay job (or, for cheap immediately-
        resolvable counterfactuals, inline by the caller)."""
        if not isinstance(counterfactual_outcome, dict):
            raise TypeError(
                f"counterfactual_outcome must be a dict; "
                f"got {type(counterfactual_outcome)}"
            )
        ts = resolved_at_utc or pd.Timestamp.now("UTC").strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self._resolutions.append(ShadowResolution(
            shadow_id=shadow_id,
            resolved_at_utc=ts,
            trading_date_ist_resolved=trading_date_ist_resolved,
            counterfactual_outcome=counterfactual_outcome,
        ))

    # -- commit -----------------------------------------------------

    def commit(self) -> Dict[str, int]:
        """Flush buffered events and resolutions to disk. Returns a
        small summary {events_written, resolutions_written} for
        operators / tests. Atomic: each partition file is written to
        a .tmp sibling and renamed into place."""
        n_events = self._flush_events()
        n_resolutions = self._flush_resolutions()
        return {
            "events_written": n_events,
            "resolutions_written": n_resolutions,
        }

    # -- internals --------------------------------------------------

    def _flush_events(self) -> int:
        if not self._events:
            return 0
        df = pd.DataFrame([self._event_to_dict(e) for e in self._events])
        # Partition by (trading_date_ist, event_kind) — this keeps
        # per-kind queries fast and lets new kinds land without
        # touching existing partition files.
        n = 0
        for (date, kind), part in df.groupby(["trading_date_ist", "event_kind"]):
            dest_dir = (
                self.root / "shadow_events"
                / f"trading_date_ist={date}"
                / f"event_kind={kind}"
            )
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / "events.parquet"
            merged = self._dedupe_merge(dest, part, key="shadow_id")
            self._atomic_write_parquet(merged, dest)
            n += len(part)
        self._events.clear()
        return n

    def _flush_resolutions(self) -> int:
        if not self._resolutions:
            return 0
        df = pd.DataFrame([self._resolution_to_dict(r) for r in self._resolutions])
        n = 0
        for date, part in df.groupby("trading_date_ist_resolved"):
            dest_dir = (
                self.root / "shadow_resolutions"
                / f"trading_date_ist_resolved={date}"
            )
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / "resolutions.parquet"
            merged = self._dedupe_merge(dest, part, key="shadow_id")
            self._atomic_write_parquet(merged, dest)
            n += len(part)
        self._resolutions.clear()
        return n

    @staticmethod
    def _event_to_dict(e: ShadowEvent) -> Dict[str, Any]:
        d = asdict(e)
        # decision_context is a dict; flatten to JSON string for
        # Parquet column stability. Same for regime_tags.
        d["decision_context"] = json.dumps(e.decision_context, default=str)
        d["regime_tags"] = json.dumps(e.regime_tags)
        return d

    @staticmethod
    def _resolution_to_dict(r: ShadowResolution) -> Dict[str, Any]:
        d = asdict(r)
        d["counterfactual_outcome"] = json.dumps(
            r.counterfactual_outcome, default=str
        )
        return d

    @staticmethod
    def _dedupe_merge(dest: Path, new: pd.DataFrame, key: str) -> pd.DataFrame:
        """If a partition already exists, merge keep-last on the
        dedupe key so repeated runs are idempotent."""
        if dest.exists():
            try:
                existing = pd.read_parquet(dest)
                merged = pd.concat([existing, new], ignore_index=True)
                return merged.drop_duplicates([key], keep="last")
            except Exception:
                # Corrupt partition: trust the new write rather than
                # block the pipeline. Operator can re-run with a fresh
                # root if needed.
                return new
        return new

    @staticmethod
    def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
        tmp = dest.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(dest)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_shadow_events(
    root: str | Path,
    event_kind: Optional[str] = None,
    trading_date_ist: Optional[str] = None,
) -> pd.DataFrame:
    """Load shadow events optionally filtered by kind and/or date.

    Returns an empty frame (not None) when nothing matches, so
    consumers can `len()` / merge / groupby without branching.
    decision_context and regime_tags come back as JSON strings;
    the caller decodes if needed.
    """
    root = Path(root) / "shadow_events"
    if not root.exists():
        return pd.DataFrame()
    parts = []
    for date_dir in sorted(root.glob("trading_date_ist=*")):
        if trading_date_ist is not None:
            if date_dir.name != f"trading_date_ist={trading_date_ist}":
                continue
        for kind_dir in sorted(date_dir.glob("event_kind=*")):
            if event_kind is not None:
                if kind_dir.name != f"event_kind={event_kind}":
                    continue
            f = kind_dir / "events.parquet"
            if f.exists():
                try:
                    parts.append(pd.read_parquet(f))
                except Exception:
                    continue
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def read_shadow_resolutions(
    root: str | Path,
    trading_date_ist_resolved: Optional[str] = None,
) -> pd.DataFrame:
    root = Path(root) / "shadow_resolutions"
    if not root.exists():
        return pd.DataFrame()
    parts = []
    for date_dir in sorted(root.glob("trading_date_ist_resolved=*")):
        if trading_date_ist_resolved is not None:
            target = f"trading_date_ist_resolved={trading_date_ist_resolved}"
            if date_dir.name != target:
                continue
        f = date_dir / "resolutions.parquet"
        if f.exists():
            try:
                parts.append(pd.read_parquet(f))
            except Exception:
                continue
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def join_events_to_resolutions(
    root: str | Path,
    event_kind: Optional[str] = None,
) -> pd.DataFrame:
    """Left join: every event, with its resolution attached when one
    exists. Unresolved rows surface as NaN on the resolution columns
    so trainers can stratify on coverage just like the outcome log
    does post-Stream-I."""
    events = read_shadow_events(root, event_kind=event_kind)
    if events.empty:
        return events
    resos = read_shadow_resolutions(root)
    if resos.empty:
        return events
    return events.merge(
        resos, on="shadow_id", how="left",
        suffixes=("", "_resolution"),
    )
