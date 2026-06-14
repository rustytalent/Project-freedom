"""DecisionEvent — the canonical export shape both codebases speak.

Sentinel's ``ledger_export.DecisionEvent`` and liqpool's
``shadow_log.ShadowEvent`` are two flavours of the same idea: 'a
decision was made, here is what we knew and what happened.' Until
this module they agreed by convention. Now they agree by import.

The canonical row carries enough to:

  * be PARTITIONED by trading_date_ist + event_type
  * be JOINED on event_id when an outcome lands later
  * be RANKED on confidence + trust_tier
  * carry the journey block when the event has resolved

Fields are JSON-friendly primitives so the same record round-trips
through JSONL on disk, parquet in the warehouse, and dicts on the
wire.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

CONTRACT_VERSION = "1.0"


# Canonical event types every consumer can rely on.
EVENT_TYPES = frozenset({
    "EXECUTED",           # a real fill — operator opened a position
    "VIRTUAL_DECISION",   # a scientist's simulated trade
    "REJECTED_CANDIDATE", # a candidate the spine declined
    "ALTERNATIVE",        # 'what if the +1 OTM strike instead'
    "MODEL_PREDICTION",   # a model output published live
    "COUNTERFACTUAL",     # 'what if we waited 5 min'
    "AVOIDANCE",          # a gate that refused to act (research side)
    "SKIP",               # explicit skip with reason (research side)
})


# Sentinel's six event kinds → unified event types.
SENTINEL_KIND_TO_EVENT_TYPE: Dict[str, str] = {
    "actual": "EXECUTED",
    "virtual": "VIRTUAL_DECISION",
    "rejected": "REJECTED_CANDIDATE",
    "alternative": "ALTERNATIVE",
    "model_suggestion": "MODEL_PREDICTION",
    "counterfactual": "COUNTERFACTUAL",
}

# Liqpool's shadow_log event_kind → unified event types.
KIND_TO_EVENT_TYPE: Dict[str, str] = {
    "skip_options_executor": "SKIP",
    "avoidance_flag": "AVOIDANCE",
    "below_target_to_cost_ratio": "SKIP",
    "macro_gate_closed": "AVOIDANCE",
    "q_below_threshold": "SKIP",
    "drift_flag_fired": "AVOIDANCE",
    "alpha_suppressed_by_meta": "REJECTED_CANDIDATE",
    "pool_merged_in_builder": "ALTERNATIVE",
}


@dataclass
class DecisionEvent:
    """One canonical decision row.

    Default values are chosen so a partial event still round-trips
    through JSON without losing intent. ``contract_version`` lets a
    future breaking change be routed without silently corrupting
    downstream readers."""
    # identity
    event_id: str
    event_type: str                          # one of EVENT_TYPES
    source: str = ""                         # "sentinel" | "liqpool" | ...
    contract_version: str = CONTRACT_VERSION
    trust_tier: str = "SHADOW"               # SHADOW / LOGGED / TRUSTED / EXECUTION
    session_date: str = ""                   # trading_date_ist (YYYY-MM-DD)
    ts_utc: str = ""

    # producer
    scientist: str = ""                      # the source organ / model name

    # instrument
    instrument: str = ""
    option_type: Optional[str] = None
    strike: Optional[float] = None
    moneyness_key: Optional[str] = None
    underlying_price: Optional[float] = None
    entry_premium: Optional[float] = None

    # hypothesis
    suggested_target: Optional[float] = None
    suggested_stop: Optional[float] = None
    confidence: Optional[float] = None
    reason_codes: List[str] = field(default_factory=list)
    invalidation_conditions: List[str] = field(default_factory=list)

    # journey (filled in when outcomes land)
    mfe: Optional[float] = None
    mae: Optional[float] = None
    time_to_target_min: Optional[float] = None
    time_to_stop_min: Optional[float] = None
    realized_outcome_t60: Optional[float] = None
    was_entry_good: Optional[bool] = None
    was_target_too_far: Optional[bool] = None
    was_stop_too_tight: Optional[bool] = None
    journey_complete: bool = False

    # free-form additional context (regime tags, asset_class, etc.)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_row(self, drop_empty: bool = True) -> Dict[str, Any]:
        """Dict for JSONL/parquet. ``drop_empty`` keeps the wire format
        skinny by omitting None / empty-list / empty-dict fields."""
        d = asdict(self)
        if drop_empty:
            d = {k: v for k, v in d.items()
                 if v is not None and v != [] and v != {} and v != ""}
            # always-keep core identity fields
            for k in ("event_id", "event_type", "contract_version", "source"):
                if k not in d:
                    d[k] = getattr(self, k)
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "DecisionEvent":
        """Tolerant deserialiser — any unknown key lands in ``extras``;
        any missing key uses its default. Numeric coercion is gentle
        (None / NaN passes through)."""
        known = {f for f in cls.__dataclass_fields__}
        extras = {k: v for k, v in row.items() if k not in known}
        kw = {k: v for k, v in row.items() if k in known}
        if extras:
            merged = dict(kw.get("extras") or {})
            merged.update(extras)
            kw["extras"] = merged
        return cls(**kw)


def from_sentinel_ledger_row(row: Dict[str, Any]) -> DecisionEvent:
    """Convert one row from Sentinel's `shadow_ledger.read_session()`
    output (5-block LedgerEvent shape) into the canonical
    DecisionEvent."""
    ident = row.get("identity") or {}
    hyp = row.get("hypothesis") or {}
    journey = row.get("journey") or {}
    judg = row.get("judgment") or {}
    return DecisionEvent(
        event_id=str(row.get("event_id", "")),
        event_type=SENTINEL_KIND_TO_EVENT_TYPE.get(row.get("kind", ""), "VIRTUAL_DECISION"),
        source="sentinel",
        trust_tier=str(row.get("trust_tier", "SHADOW")),
        session_date=str(row.get("session", "")),
        ts_utc=str(row.get("ts_utc", "")),
        scientist=str(row.get("scientist", "")),
        instrument=str(ident.get("instrument", "")),
        option_type=ident.get("option_type") or None,
        strike=ident.get("strike"),
        moneyness_key=ident.get("moneyness_key"),
        underlying_price=ident.get("underlying_price"),
        entry_premium=ident.get("premium"),
        suggested_target=hyp.get("suggested_target") if hyp else None,
        suggested_stop=hyp.get("suggested_stop") if hyp else None,
        confidence=hyp.get("confidence") if hyp else None,
        reason_codes=list(hyp.get("reason_codes") or []) if hyp else [],
        invalidation_conditions=list(hyp.get("invalidation_conditions") or []) if hyp else [],
        mfe=journey.get("mfe"),
        mae=journey.get("mae"),
        time_to_target_min=journey.get("time_to_target_min"),
        time_to_stop_min=journey.get("time_to_stop_min"),
        realized_outcome_t60=(journey.get("outcomes") or {}).get("t+60"),
        was_entry_good=judg.get("was_entry_good"),
        was_target_too_far=judg.get("was_target_too_far"),
        was_stop_too_tight=judg.get("was_stop_too_tight"),
        journey_complete=bool(journey.get("complete")),
        extras={"context": row.get("context") or {}},
    )


def from_shadow_event_row(row: Dict[str, Any]) -> DecisionEvent:
    """Convert one row from liqpool's `shadow_log.ShadowEvent` into the
    canonical DecisionEvent."""
    dc = row.get("decision_context") or {}
    if isinstance(dc, str):
        import json
        try:
            dc = json.loads(dc)
        except Exception:
            dc = {}
    return DecisionEvent(
        event_id=str(row.get("shadow_id", "")),
        event_type=KIND_TO_EVENT_TYPE.get(row.get("event_kind", ""), "AVOIDANCE"),
        source="liqpool",
        trust_tier="SHADOW",                 # shadow_log records are research-side noise by definition
        session_date=str(row.get("trading_date_ist", "")),
        ts_utc=str(row.get("written_at_utc", "")),
        scientist=str(row.get("event_kind", "")),
        instrument=str(row.get("symbol") or ""),
        confidence=dc.get("confidence"),
        reason_codes=list(row.get("regime_tags") or []),
        extras={"decision_context": dc},
    )
