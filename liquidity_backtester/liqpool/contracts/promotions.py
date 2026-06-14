"""TrustPromotionRecord — graduation decisions, shared shape."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrustPromotionRecord:
    source: str                              # producer name (scientist / detector / ...)
    from_tier: str
    to_tier: str
    decision: str                            # promoted / demoted / held
    ts_utc: str = ""
    evidence_window: str = ""                # e.g. "2026-06-01..2026-06-13"
    n: Optional[int] = None
    hit_rate: Optional[float] = None
    expectancy: Optional[float] = None
    drawdown: Optional[float] = None
    calibration_error: Optional[float] = None
    leakage_status: Optional[str] = None
    note: str = ""

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "TrustPromotionRecord":
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in row.items() if k in known}
        return cls(**kw)
