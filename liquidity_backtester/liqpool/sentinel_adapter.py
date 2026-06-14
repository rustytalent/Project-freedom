"""Sentinel ledger adapter — read Sentinel's exported shadow ledger
into the same DataFrame shape the flywheel's nightly trainer already
consumes.

The flow Codex called out and the founder asked for:

  DAY    : Sentinel runs live; every decision lands in
            <journal>/ledger_<session>.jsonl
  NIGHT  : ``ledger_export`` exports to <export>/sentinel_live/
            session=<date>/decision_events.{jsonl,parquet}
  NIGHT  : ``scripts/train_flywheel.py`` feeds the flywheel hub —
            it joins its own shadow_log + outcome_log with this
            adapter's output so EVERY live decision becomes
            training data for tomorrow's models.

This module is the *narrow* piece that does the conversion. It reads
JSONL (either Sentinel's raw ledger OR the exported DecisionEvent
form), filters by a date range, and produces a DataFrame in the same
column shape the existing ``join_events_to_resolutions`` returns —
so ``hub.fit_from_artifacts(shadow_joined=...)`` is a one-line
addition, not a rewrite.

Empty/missing inputs short-circuit to None so the flywheel can stay
"safe-unfit" the way every other organ already does.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from liqpool.contracts.events import (
    DecisionEvent, from_sentinel_ledger_row,
)

try:
    import pandas as pd
    _HAS_PD = True
except ImportError:                              # pragma: no cover
    _HAS_PD = False


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Tolerant JSONL reader; one malformed line doesn't kill the file."""
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def read_sentinel_ledger_dir(root: Path,
                              since: Optional[str] = None,
                              until: Optional[str] = None
                              ) -> List[DecisionEvent]:
    """Walk a Sentinel journal directory of ``ledger_<session>.jsonl``
    files and return DecisionEvents. ``since`` / ``until`` are
    ``YYYY-MM-DD`` strings (inclusive) for partition pruning."""
    root = Path(root)
    if not root.exists():
        return []
    events: List[DecisionEvent] = []
    seen: Dict[str, Dict[str, Any]] = {}             # last-write-wins per event_id
    for f in sorted(root.glob("ledger_*.jsonl")):
        # filename: ledger_YYYY-MM-DD.jsonl
        try:
            session = f.stem.split("_", 1)[1]
        except IndexError:
            continue
        if since and session < since:
            continue
        if until and session > until:
            continue
        for row in _iter_jsonl(f):
            eid = row.get("event_id")
            if not eid:
                continue
            seen[eid] = row
    for row in seen.values():
        try:
            events.append(from_sentinel_ledger_row(row))
        except Exception:
            continue
    return events


def read_decision_events_dir(root: Path,
                              since: Optional[str] = None,
                              until: Optional[str] = None
                              ) -> List[DecisionEvent]:
    """Walk an *exported* directory laid out as
    ``<root>/sentinel_live/session=<date>/decision_events.jsonl``
    (the shape ``sentinel.ledger_export`` produces). Returns
    DecisionEvents already in the canonical shape — no conversion
    needed."""
    root = Path(root)
    base = root / "sentinel_live"
    if not base.exists():
        base = root            # caller may pass the layer directly
    if not base.exists():
        return []
    events: List[DecisionEvent] = []
    for partition in sorted(base.glob("session=*")):
        date = partition.name.split("=", 1)[-1]
        if since and date < since:
            continue
        if until and date > until:
            continue
        for f in partition.glob("decision_events.jsonl"):
            for row in _iter_jsonl(f):
                try:
                    events.append(DecisionEvent.from_row(row))
                except Exception:
                    continue
    return events


def events_to_shadow_frame(events: List[DecisionEvent]):
    """Project DecisionEvents into the SAME column shape the
    flywheel's ``join_events_to_resolutions`` returns, so the night
    trainer treats them as one bigger pile.

    The output frame has these columns (subset; extra fields the hub
    ignores are harmless):

      shadow_id  event_kind  trading_date_ist  written_at_utc  symbol
      decision_context (dict)  regime_tags (list)
      resolved_at_utc  trading_date_ist_resolved  counterfactual_outcome (dict)

    Sentinel's ``trust_tier`` lands in ``regime_tags`` so per-tier
    bucketing comes for free.
    """
    if not _HAS_PD:                                  # pragma: no cover
        raise RuntimeError("pandas required for events_to_shadow_frame")
    rows: List[Dict[str, Any]] = []
    for ev in events:
        # Reuse the canonical extras for decision_context; carry the
        # journey/judgment numbers inline so the regret organ has them.
        decision_context = {
            "scientist": ev.scientist,
            "trust_tier": ev.trust_tier,
            "instrument": ev.instrument,
            "option_type": ev.option_type,
            "strike": ev.strike,
            "moneyness_key": ev.moneyness_key,
            "entry_premium": ev.entry_premium,
            "confidence": ev.confidence,
            "reason_codes": list(ev.reason_codes),
            "invalidation_conditions": list(ev.invalidation_conditions),
            "suggested_target": ev.suggested_target,
            "suggested_stop": ev.suggested_stop,
            **(ev.extras or {}),
        }
        outcome = {
            "mfe": ev.mfe,
            "mae": ev.mae,
            "time_to_target_min": ev.time_to_target_min,
            "time_to_stop_min": ev.time_to_stop_min,
            "realized_outcome_t60": ev.realized_outcome_t60,
            "was_entry_good": ev.was_entry_good,
            "was_target_too_far": ev.was_target_too_far,
            "was_stop_too_tight": ev.was_stop_too_tight,
            "journey_complete": ev.journey_complete,
        }
        rows.append({
            "shadow_id": ev.event_id,
            "event_kind": f"sentinel_{ev.event_type.lower()}",
            "trading_date_ist": ev.session_date,
            "written_at_utc": ev.ts_utc,
            "symbol": ev.instrument or None,
            "decision_context": decision_context,
            "regime_tags": [ev.trust_tier, ev.scientist] if ev.scientist else [ev.trust_tier],
            "resolved_at_utc": ev.ts_utc if ev.journey_complete else None,
            "trading_date_ist_resolved": ev.session_date if ev.journey_complete else None,
            "counterfactual_outcome": outcome if ev.journey_complete else None,
        })
    return pd.DataFrame(rows)


def load_sentinel_as_shadow_frame(
    journal_dir: Optional[Path],
    export_dir: Optional[Path] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
):
    """One-call convenience for ``scripts/train_flywheel.py``.

    Reads either the raw Sentinel journal (``journal_dir``) or its
    exported DecisionEvents (``export_dir``), or both — last-write-wins
    per event_id when an event_id is in both. Returns ``None`` if no
    rows found, so the trainer's existing "no data → safe unfit"
    path stays valid."""
    journal_dir = Path(journal_dir) if journal_dir else None
    export_dir = Path(export_dir) if export_dir else None
    bucket: Dict[str, DecisionEvent] = {}
    if journal_dir is not None and journal_dir.exists():
        for ev in read_sentinel_ledger_dir(journal_dir, since, until):
            bucket[ev.event_id] = ev
    if export_dir is not None and export_dir.exists():
        for ev in read_decision_events_dir(export_dir, since, until):
            bucket[ev.event_id] = ev   # last-write wins per event_id
    if not bucket:
        return None
    return events_to_shadow_frame(list(bucket.values()))
