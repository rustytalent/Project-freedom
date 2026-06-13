"""Shadow Ledger — the substrate, the product, the laboratory record.

Not OHLCV. Every real, virtual, rejected, alternative, model-suggestion,
and counterfactual decision is recorded with five blocks (identity,
context, hypothesis, journey, judgment) per the architecture spec.

Design:
  * append-only JSONL per session (crash-safe, GB-scale friendly);
    a parquet compaction helper folds a finished session for analytics.
  * the JOURNEY block is filled incrementally — an event is born with
    identity+context+hypothesis, then the journey resolver stamps
    T+1/3/5/.../60 outcomes as live data arrives, then the curator
    stamps the JUDGMENT block once the journey completes.
  * moneyness_key (sentinel.moneyness) is the transferable identity so
    the ledger survives expiry rollover.

Every record answers the founder's demand: *why this trade, why now,
why it beats the market, what invalidates it.* A hypothesis with no
reason_codes is rejected at write time — the laboratory does not log
trades it cannot explain.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Event kinds — the founder's full taxonomy.
KIND_ACTUAL = "actual"               # a real order the operator placed
KIND_VIRTUAL = "virtual"             # a scientist's simulated trade
KIND_REJECTED = "rejected"           # a candidate the gate declined
KIND_ALTERNATIVE = "alternative"     # "what if +1 OTM instead"
KIND_MODEL_SUGGESTION = "model_suggestion"
KIND_COUNTERFACTUAL = "counterfactual"   # "what if we waited 5 min"

ALL_KINDS = frozenset({
    KIND_ACTUAL, KIND_VIRTUAL, KIND_REJECTED, KIND_ALTERNATIVE,
    KIND_MODEL_SUGGESTION, KIND_COUNTERFACTUAL,
})

# Journey checkpoints, minutes after the event.
JOURNEY_CHECKPOINTS = (1, 3, 5, 10, 15, 30, 60)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# The five blocks
# ---------------------------------------------------------------------------

@dataclass
class Identity:
    instrument: str                 # tradingsymbol
    option_type: str                # CE | PE
    strike: float
    expiry: Optional[str]
    moneyness_key: str              # MoneynessKey.as_str() — the transferable id
    underlying_price: float = 0.0
    premium: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: float = 0.0
    iv: Optional[float] = None
    oi: float = 0.0
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


@dataclass
class Context:
    trend: Optional[str] = None
    liquidity: Optional[str] = None
    volatility: Optional[str] = None
    opening_range: Optional[str] = None
    time_of_day: Optional[str] = None
    support_resistance: Optional[str] = None
    model_signal: Optional[float] = None
    reaction_model: Optional[float] = None
    option_chain_state: Optional[str] = None
    # Equity-weightage divergence (wave 3) lands here when bridged.
    equity_weight_divergence: Optional[float] = None


@dataclass
class Hypothesis:
    expected_underlying_move: float          # points, signed
    expected_premium: float
    expected_horizon_min: float
    suggested_entry: float
    suggested_stop: float
    suggested_target: float
    confidence: float
    reason_codes: List[str] = field(default_factory=list)
    invalidation_conditions: List[str] = field(default_factory=list)


@dataclass
class Journey:
    # premium at each checkpoint; None until that checkpoint resolves
    outcomes: Dict[str, Optional[float]] = field(
        default_factory=lambda: {f"t+{m}": None for m in JOURNEY_CHECKPOINTS})
    mfe: Optional[float] = None              # max favourable excursion (premium)
    mae: Optional[float] = None              # max adverse excursion
    time_to_target_min: Optional[float] = None
    time_to_stop_min: Optional[float] = None
    premium_decay: Optional[float] = None
    slippage_estimate: Optional[float] = None
    complete: bool = False


@dataclass
class Judgment:
    was_entry_good: Optional[bool] = None
    was_exit_good: Optional[bool] = None
    was_stop_too_tight: Optional[bool] = None
    was_target_too_far: Optional[bool] = None
    was_confidence_calibrated: Optional[bool] = None
    did_selection_outperform: Optional[bool] = None
    itm_vs_otm: Optional[str] = None         # which won
    atm_vs_otm: Optional[str] = None
    ce_pe_as_expected: Optional[bool] = None
    notes: str = ""


@dataclass
class LedgerEvent:
    event_id: str
    kind: str
    ts_utc: str
    session: str                    # IST trading date
    identity: Identity
    context: Context
    hypothesis: Optional[Hypothesis]
    journey: Journey = field(default_factory=Journey)
    judgment: Judgment = field(default_factory=Judgment)
    scientist: str = ""             # which candidate generator emitted it

    def to_row(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id, "kind": self.kind,
            "ts_utc": self.ts_utc, "session": self.session,
            "scientist": self.scientist,
            "identity": asdict(self.identity),
            "context": asdict(self.context),
            "hypothesis": asdict(self.hypothesis) if self.hypothesis else None,
            "journey": asdict(self.journey),
            "judgment": asdict(self.judgment),
        }


def new_event(kind: str, session: str, identity: Identity,
              context: Context, hypothesis: Optional[Hypothesis],
              scientist: str = "") -> LedgerEvent:
    if kind not in ALL_KINDS:
        raise ValueError(f"unknown event kind: {kind}")
    # The laboratory does not log trades it cannot explain: any event
    # that carries a hypothesis MUST carry at least one reason code.
    if hypothesis is not None and not hypothesis.reason_codes:
        raise ValueError(
            "hypothesis without reason_codes rejected — every trade must "
            "state why it should beat the market")
    return LedgerEvent(
        event_id=f"EV_{uuid.uuid4().hex[:16]}",
        kind=kind, ts_utc=_utc(), session=session,
        identity=identity, context=context, hypothesis=hypothesis,
        scientist=scientist,
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class ShadowLedger:
    """Append-only JSONL writer, one file per session, thread-safe.

    Events are written once at birth (identity+context+hypothesis) and
    again whenever their journey/judgment is updated — the reader keeps
    the last write per event_id (same idempotence pattern the research
    repo's outcome log uses)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, session: str) -> Path:
        return self.root / f"ledger_{session}.jsonl"

    def write(self, ev: LedgerEvent) -> str:
        with self._lock:
            with open(self._path(ev.session), "a") as f:
                f.write(json.dumps(ev.to_row()) + "\n")
        return ev.event_id

    def update(self, ev: LedgerEvent) -> None:
        """Re-append the event after a journey/judgment stamp. Cheap;
        the reader collapses to last-write-wins per event_id."""
        self.write(ev)

    # -- journey resolution -------------------------------------------------

    @staticmethod
    def stamp_journey(ev: LedgerEvent, minutes_elapsed: float,
                      current_premium: float) -> None:
        """Record the premium at whichever checkpoint we just passed and
        keep MFE/MAE running. Idempotent per checkpoint."""
        j = ev.journey
        entry = ev.identity.premium
        # MFE/MAE relative to entry (premium terms; sign-agnostic store).
        if j.mfe is None or current_premium > j.mfe:
            j.mfe = current_premium
        if j.mae is None or current_premium < j.mae:
            j.mae = current_premium
        for m in JOURNEY_CHECKPOINTS:
            key = f"t+{m}"
            if minutes_elapsed >= m and j.outcomes.get(key) is None:
                j.outcomes[key] = round(current_premium, 2)
        if ev.hypothesis is not None and entry:
            tgt, stp = ev.hypothesis.suggested_target, ev.hypothesis.suggested_stop
            if j.time_to_target_min is None and tgt and current_premium >= tgt:
                j.time_to_target_min = minutes_elapsed
            if j.time_to_stop_min is None and stp and current_premium <= stp:
                j.time_to_stop_min = minutes_elapsed
        if minutes_elapsed >= JOURNEY_CHECKPOINTS[-1]:
            j.premium_decay = round(current_premium - entry, 2)
            j.complete = True


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_session(root: Path, session: str) -> List[Dict[str, Any]]:
    """Last-write-wins per event_id for one session."""
    p = Path(root) / f"ledger_{session}.jsonl"
    if not p.exists():
        return []
    latest: Dict[str, Dict[str, Any]] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            latest[row["event_id"]] = row
        except Exception:
            continue
    return list(latest.values())


def compact_to_parquet(root: Path, session: str) -> Optional[Path]:
    """Fold a finished session's last-write rows into a flat parquet for
    analytics / training. Best-effort: returns None if pandas/pyarrow
    unavailable or the session is empty."""
    rows = read_session(root, session)
    if not rows:
        return None
    try:
        import pandas as pd
    except Exception:
        return None
    flat = []
    for r in rows:
        base = {"event_id": r["event_id"], "kind": r["kind"],
                "ts_utc": r["ts_utc"], "session": r["session"],
                "scientist": r.get("scientist", "")}
        for block in ("identity", "context", "judgment"):
            for k, v in (r.get(block) or {}).items():
                base[f"{block}.{k}"] = v
        j = r.get("journey") or {}
        for k, v in (j.get("outcomes") or {}).items():
            base[f"journey.{k}"] = v
        for k in ("mfe", "mae", "time_to_target_min", "time_to_stop_min",
                  "premium_decay", "slippage_estimate", "complete"):
            base[f"journey.{k}"] = j.get(k)
        flat.append(base)
    out = Path(root) / f"ledger_{session}.parquet"
    pd.DataFrame(flat).to_parquet(out, index=False)
    return out


from .io_decl import IOSpec, declare
declare(IOSpec(
    module="sentinel.shadow_ledger",
    purpose="the substrate: identity+context+hypothesis+journey+judgment per decision",
    inputs=["LedgerEvent from scientists / actual trades / counterfactuals"],
    outputs=["file:<journal>/ledger_<session>.jsonl", "ledger_<session>.parquet"],
    consumes_from=["sentinel.scientists", "sentinel.greeks", "sentinel.moneyness"],
    produces_for=["sentinel.curator", "sentinel.calibration", "sentinel.reports"],
    tier="SHADOW",
))
