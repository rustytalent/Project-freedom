"""ModificationBudget — 4-bucket Zerodha mod cap allocation.

Zerodha caps order modifications at 25 per order. The founder + ChatGPT
agreed on a 4-bucket allocation:

  Early correction      3–5  mods  — fix bad first quote after entry
  Normal adaptive       8–10 mods  — improve/defend target during life
  Endgame chase         5–7  mods  — when exit is near, push for fill
  Emergency reserve     4–5  mods  — thesis broken / fast market / rejection

The budget tracks usage per bucket. The modification gate (in gate.py)
queries the budget before approving any spend. If the requested bucket
is depleted, the gate refuses — except KILL-mode emergency exits, which
always go through the emergency_reserve bucket.

Founder's golden rule:
  "Never let the system spend all 25 on normal repricing. The bot
   should not modify 25 times randomly. It should spend maybe 8–15
   serious modifications per order in a full trade. More than that
   means the engine is confused."
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Bucket constants ──────────────────────────────────────────────


class ModificationBucket:
    EARLY_CORRECTION = "early_correction"
    NORMAL_ADAPTIVE = "normal_adaptive"
    ENDGAME_CHASE = "endgame_chase"
    EMERGENCY_RESERVE = "emergency_reserve"


_ALL_BUCKETS = (
    ModificationBucket.EARLY_CORRECTION,
    ModificationBucket.NORMAL_ADAPTIVE,
    ModificationBucket.ENDGAME_CHASE,
    ModificationBucket.EMERGENCY_RESERVE,
)


# ── Config + request ──────────────────────────────────────────────


@dataclass
class ModificationBudgetConfig:
    """Founder-confirmed bucket sizes."""
    total_cap: int = 25
    early_correction_cap: int = 4
    normal_adaptive_cap: int = 10
    endgame_chase_cap: int = 6
    emergency_reserve_cap: int = 5

    def __post_init__(self) -> None:
        total = (self.early_correction_cap + self.normal_adaptive_cap
                 + self.endgame_chase_cap + self.emergency_reserve_cap)
        if total != self.total_cap:
            raise ValueError(f"bucket sum {total} != total_cap {self.total_cap}")


@dataclass(frozen=True)
class ModificationRequest:
    """One request to spend a mod from a specific bucket."""
    bucket: str
    reason: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# ── Budget ────────────────────────────────────────────────────────


@dataclass
class ModificationBudget:
    """Per-position mod budget with 4-bucket accounting."""
    cfg: ModificationBudgetConfig = field(
        default_factory=ModificationBudgetConfig)
    used: Dict[str, int] = field(default_factory=dict)
    history: List[ModificationRequest] = field(default_factory=list)
    rejected_history: List[ModificationRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.used:
            self.used = {b: 0 for b in _ALL_BUCKETS}

    # ── Queries ────────────────────────────────────────────────────

    @property
    def total_used(self) -> int:
        return sum(self.used.values())

    @property
    def total_remaining(self) -> int:
        return max(0, self.cfg.total_cap - self.total_used)

    def bucket_cap(self, bucket: str) -> int:
        return {
            ModificationBucket.EARLY_CORRECTION: self.cfg.early_correction_cap,
            ModificationBucket.NORMAL_ADAPTIVE: self.cfg.normal_adaptive_cap,
            ModificationBucket.ENDGAME_CHASE: self.cfg.endgame_chase_cap,
            ModificationBucket.EMERGENCY_RESERVE: self.cfg.emergency_reserve_cap,
        }.get(bucket, 0)

    def bucket_remaining(self, bucket: str) -> int:
        return max(0, self.bucket_cap(bucket) - self.used.get(bucket, 0))

    def budget_pressure(self) -> float:
        """Returns a scaled 0..1 measure of how depleted the overall
        budget is — used by the gate to raise the modification threshold
        as the budget shrinks."""
        return (self.total_used / max(1, self.cfg.total_cap)) ** 2

    # ── Spend / reject ─────────────────────────────────────────────

    def can_spend(self, bucket: str) -> bool:
        """Whether we have headroom in the requested bucket."""
        return self.bucket_remaining(bucket) > 0

    def try_spend(self, bucket: str, reason: str,
                    *, fallback_to_emergency_on_kill: bool = False
                    ) -> Optional[ModificationRequest]:
        """Spend one modification from the requested bucket.

        If the requested bucket is depleted but ``fallback_to_emergency_on_kill``
        is True (used by KILL mode), falls back to the emergency reserve.
        Returns the granted ModificationRequest, or None if refused.
        """
        if self.can_spend(bucket):
            self.used[bucket] = self.used.get(bucket, 0) + 1
            req = ModificationRequest(bucket=bucket, reason=reason)
            self.history.append(req)
            return req
        # Fallback: KILL mode can dip into emergency reserve.
        if (fallback_to_emergency_on_kill
                and bucket != ModificationBucket.EMERGENCY_RESERVE
                and self.can_spend(ModificationBucket.EMERGENCY_RESERVE)):
            self.used[ModificationBucket.EMERGENCY_RESERVE] = (
                self.used.get(ModificationBucket.EMERGENCY_RESERVE, 0) + 1)
            req = ModificationRequest(
                bucket=ModificationBucket.EMERGENCY_RESERVE,
                reason=f"KILL fallback for {reason}",
            )
            self.history.append(req)
            return req
        # Refused.
        rej = ModificationRequest(bucket=bucket,
                                     reason=f"REJECTED: {reason}")
        self.rejected_history.append(rej)
        return None

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cap": self.cfg.total_cap,
            "total_used": self.total_used,
            "total_remaining": self.total_remaining,
            "budget_pressure": round(self.budget_pressure(), 3),
            "buckets": {
                b: {
                    "cap": self.bucket_cap(b),
                    "used": self.used.get(b, 0),
                    "remaining": self.bucket_remaining(b),
                } for b in _ALL_BUCKETS
            },
            "history_count": len(self.history),
            "rejected_count": len(self.rejected_history),
            "last_history": [r.to_dict() for r in self.history[-5:]],
        }
