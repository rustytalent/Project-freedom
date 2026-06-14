"""ResearchContextPack — pre-market briefing handed to Sentinel.

This is the SLIM export shape of liqpool's `BriefDocument` that Sentinel
consumes via its ``liqpool_bridge``. Keeping it slim means the live
cockpit doesn't have to know the full daily-brief schema — only the
fields it puts on screen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ZoneLevel:
    label: str                       # PDH / VAL / VAH / OR_HIGH / ...
    price: float
    side: str                        # "support" | "resistance" | "magnet"
    horizon_minutes: int = 60
    reach_probability: float = 0.0


@dataclass
class ResearchContextPack:
    session_date: str = ""
    source_run_id: str = ""
    direction_probability: Optional[float] = None
    reaction_probability: Optional[float] = None
    proximity_h12: Optional[float] = None
    proximity_h36: Optional[float] = None
    proximity_h60: Optional[float] = None
    sector_regime: Optional[str] = None
    avoid_flags: List[str] = field(default_factory=list)
    model_health: Dict[str, Any] = field(default_factory=dict)
    key_zones: List[ZoneLevel] = field(default_factory=list)
    notes: str = ""

    @property
    def is_empty(self) -> bool:
        return (self.direction_probability is None
                and self.reaction_probability is None
                and not self.key_zones)

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["key_zones"] = [asdict(z) for z in self.key_zones]
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ResearchContextPack":
        zones = []
        for z in row.get("key_zones") or []:
            if isinstance(z, dict):
                zones.append(ZoneLevel(
                    label=str(z.get("label", "?")),
                    price=float(z.get("price") or 0),
                    side=str(z.get("side", "magnet")),
                    horizon_minutes=int(z.get("horizon_minutes", 60)),
                    reach_probability=float(z.get("reach_probability", 0.0)),
                ))
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in row.items()
              if k in known and k != "key_zones"}
        kw["key_zones"] = zones
        return cls(**kw)
