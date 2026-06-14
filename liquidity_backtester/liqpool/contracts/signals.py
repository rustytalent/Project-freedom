"""ModelSignal — one live research output, shared shape.

Sentinel's ``live_publisher.ModelSignal`` and liqpool's
``arsenal.AlphaSignal`` are NOT the same thing (the latter is
backtest-domain geometry); but the LIVE signal — what a research
model says, right now, about an asset — should be one type.

Frozen + dict-roundtrippable so it crosses process boundaries
unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelSignal:
    """One live research output. Fields chosen to match Sentinel's
    existing ModelSignal exactly so they're interchangeable."""
    ts_ist: str                                  # HH:MM:SS in IST
    asset: str                                   # NIFTY / RELIANCE / ATM_CE / ...
    model: str                                   # reaction_model / proximity_model / ...
    signal: str                                  # human-readable verdict
    confidence: float                            # 0..1
    trust_tier: str = "SHADOW"                   # SHADOW / LOGGED / TRUSTED / EXECUTION
    zone: Optional[Tuple[float, float]] = None
    risk: str = ""
    reason_codes: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    # which side produced this — useful when both Sentinel and
    # liqpool are publishing to the same bus
    source: str = ""                             # "sentinel" | "liqpool"

    @property
    def key(self) -> str:
        return f"{self.asset}|{self.model}"

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.zone is not None:
            d["zone"] = list(self.zone)
        # drop empties for a leaner wire format
        return {k: v for k, v in d.items()
                if v not in (None, [], {}, "")}

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ModelSignal":
        kw = dict(row)
        if isinstance(kw.get("zone"), list) and len(kw["zone"]) == 2:
            kw["zone"] = tuple(kw["zone"])
        # tolerate unknown fields by routing them into extras
        known = {f for f in cls.__dataclass_fields__}
        extras = {k: v for k, v in kw.items() if k not in known}
        kw = {k: v for k, v in kw.items() if k in known}
        if extras:
            merged = dict(kw.get("extras") or {})
            merged.update(extras)
            kw["extras"] = merged
        # defaults for missing optionals
        kw.setdefault("zone", None)
        kw.setdefault("risk", "")
        kw.setdefault("reason_codes", [])
        kw.setdefault("extras", {})
        kw.setdefault("source", "")
        kw.setdefault("trust_tier", "SHADOW")
        return cls(**kw)
