"""Live-to-research adapter — export the Sentinel shadow ledger into the
liquidity_backtester's shadow-log partition schema.

Codex's Problem S4 in the unified report: the live moat — every actual,
virtual, rejected, alternative, and counterfactual decision logged with
five-block schema — must feed back into the research engine's flywheel.
Without an adapter, today's live decisions never become tomorrow's
training data.

Output: one ``DecisionEvent`` row per ledger event, normalised to the
shape the liquidity_backtester's ShadowLogger expects. Schema is
intentionally a *superset* of the research side's columns so a join is
trivial.

Run as a CLI (typically a nightly cron):

    python -m sentinel.ledger_export \\
        --journal ~/.sentinel \\
        --session 2026-06-13 \\
        --out ~/liqpool_shadow/sentinel_live/

Tier: TRUSTED. Decision events are append-only and idempotent by
event_id, so re-running the export is safe.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_decl import IOSpec, declare
from .shadow_ledger import read_session


# Mapping Sentinel event kinds -> the shared DecisionEvent type vocabulary
# that the research repo's shadow log can read.
KIND_MAP: Dict[str, str] = {
    "actual": "EXECUTED",
    "virtual": "VIRTUAL_DECISION",
    "rejected": "REJECTED_CANDIDATE",
    "alternative": "ALTERNATIVE",
    "model_suggestion": "MODEL_PREDICTION",
    "counterfactual": "COUNTERFACTUAL",
}


@dataclass
class DecisionEvent:
    """The cross-codebase contract row. Every field that's None is
    omitted on serialise so the research side sees clean columns."""
    event_id: str
    event_type: str                  # mapped via KIND_MAP
    source: str = "sentinel"
    trust_tier: str = "SHADOW"        # SHADOW/LOGGED/TRUSTED/EXECUTION
    session_date: str = ""
    ts_utc: str = ""
    scientist: str = ""
    instrument: str = ""
    option_type: Optional[str] = None
    strike: Optional[float] = None
    moneyness_key: Optional[str] = None
    underlying_price: Optional[float] = None
    entry_premium: Optional[float] = None
    suggested_target: Optional[float] = None
    suggested_stop: Optional[float] = None
    confidence: Optional[float] = None
    reason_codes: List[str] = field(default_factory=list)
    mfe: Optional[float] = None
    mae: Optional[float] = None
    time_to_target_min: Optional[float] = None
    time_to_stop_min: Optional[float] = None
    realized_outcome_t60: Optional[float] = None
    was_entry_good: Optional[bool] = None
    was_target_too_far: Optional[bool] = None
    was_stop_too_tight: Optional[bool] = None
    journey_complete: bool = False

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != []}


def to_decision_event(row: Dict[str, Any]) -> DecisionEvent:
    """Convert one Sentinel ledger row into a DecisionEvent."""
    ident = row.get("identity") or {}
    hyp = row.get("hypothesis") or {}
    journey = row.get("journey") or {}
    judg = row.get("judgment") or {}
    return DecisionEvent(
        event_id=row.get("event_id", ""),
        event_type=KIND_MAP.get(row.get("kind", ""), "UNKNOWN"),
        trust_tier=row.get("trust_tier", "SHADOW"),
        session_date=row.get("session", ""),
        ts_utc=row.get("ts_utc", ""),
        scientist=row.get("scientist", ""),
        instrument=ident.get("instrument", ""),
        option_type=ident.get("option_type"),
        strike=ident.get("strike"),
        moneyness_key=ident.get("moneyness_key"),
        underlying_price=ident.get("underlying_price"),
        entry_premium=ident.get("premium"),
        suggested_target=hyp.get("suggested_target") if hyp else None,
        suggested_stop=hyp.get("suggested_stop") if hyp else None,
        confidence=hyp.get("confidence") if hyp else None,
        reason_codes=list(hyp.get("reason_codes") or []) if hyp else [],
        mfe=journey.get("mfe"),
        mae=journey.get("mae"),
        time_to_target_min=journey.get("time_to_target_min"),
        time_to_stop_min=journey.get("time_to_stop_min"),
        realized_outcome_t60=(journey.get("outcomes") or {}).get("t+60"),
        was_entry_good=judg.get("was_entry_good"),
        was_target_too_far=judg.get("was_target_too_far"),
        was_stop_too_tight=judg.get("was_stop_too_tight"),
        journey_complete=bool(journey.get("complete")),
    )


def export_session(journal: Path, session: str, out_dir: Path) -> Path:
    """Read the Sentinel ledger for ``session`` and write a JSONL file
    of DecisionEvents under ``out_dir/sentinel_live/session=<...>/``.

    Returns the output JSONL path. Empty session -> the file is still
    created (zero rows) so the downstream consumer can distinguish
    'session ran with no events' from 'session not exported yet'.
    """
    rows = read_session(Path(journal), session)
    out_dir = Path(out_dir) / f"sentinel_live/session={session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "decision_events.jsonl"
    with open(path, "w") as f:
        for row in rows:
            ev = to_decision_event(row)
            f.write(json.dumps(ev.to_row()) + "\n")
    return path


def export_session_to_parquet(journal: Path, session: str,
                                out_dir: Path) -> Optional[Path]:
    """Same as ``export_session`` but writes parquet (preferred for the
    research-side flywheel). Returns None if pandas/pyarrow unavailable."""
    rows = read_session(Path(journal), session)
    try:
        import pandas as pd
    except Exception:
        return None
    out_dir = Path(out_dir) / f"sentinel_live/session={session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "decision_events.parquet"
    df = pd.DataFrame([to_decision_event(r).to_row() for r in rows])
    df.to_parquet(path, index=False)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--journal", type=Path, default=Path.home() / ".sentinel")
    p.add_argument("--session", default=datetime.utcnow().strftime("%Y-%m-%d"))
    p.add_argument("--out", type=Path,
                   default=Path.home() / "liqpool_shadow")
    p.add_argument("--parquet", action="store_true",
                   help="write parquet (default JSONL)")
    args = p.parse_args()
    if args.parquet:
        out = export_session_to_parquet(args.journal, args.session, args.out)
        if out is None:
            print("pandas not installed; falling back to JSONL", file=sys.stderr)
            out = export_session(args.journal, args.session, args.out)
    else:
        out = export_session(args.journal, args.session, args.out)
    print(f"wrote {out}")
    return 0


declare(IOSpec(
    module="sentinel.ledger_export",
    purpose="nightly live-to-research adapter — Sentinel shadow ledger "
            "JSONL -> DecisionEvent rows in the liquidity_backtester's "
            "shadow-log partition schema, idempotent by event_id",
    inputs=["file:<journal>/ledger_<session>.jsonl (via shadow_ledger.read_session)"],
    outputs=["file:<out>/sentinel_live/session=<...>/decision_events.{jsonl,parquet}"],
    consumes_from=["sentinel.shadow_ledger"],
    produces_for=["liquidity_backtester ShadowLogger / FlywheelHub (external)"],
    tier="TRUSTED",
))


if __name__ == "__main__":
    sys.exit(main())
