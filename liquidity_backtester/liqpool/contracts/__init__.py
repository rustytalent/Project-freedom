"""Cross-codebase contracts — the shared event spine.

Codex's report identified this as the highest-priority architectural
gap: both halves of the project carry their own dataclasses that
'agree by convention' for things like DecisionEvent and ModelSignal.
This package is the single source of truth so both ``liqpool`` and
``sentinel`` import from one place.

Schema discipline:
  * **Additive-only** changes within a version: new fields land with
    sensible defaults so old readers keep working.
  * Bump ``CONTRACT_VERSION`` on a breaking change. Every record
    carries the version inline so consumers can route.
  * Every dataclass is frozen + dict-roundtrippable (``to_row()`` /
    ``from_row()``) — JSONL is the wire format both sides use.

What's in here:

  ModelSignal           — one live research output (asset + model +
                          signal + confidence + zone + invalidation)
  DecisionEvent         — one decision row in the canonical export
                          format (flat, queryable, partition-friendly)
  TrustPromotionRecord  — one graduation decision (source + tiers
                          + curator evidence)
  ResearchContextPack   — the pre-market handshake the daily brief
                          ships to Sentinel
  RunManifest           — training/inference run metadata
"""
from __future__ import annotations

from .events import (
    CONTRACT_VERSION, DecisionEvent, EVENT_TYPES, KIND_TO_EVENT_TYPE,
    SENTINEL_KIND_TO_EVENT_TYPE,
)
from .manifest import RunManifest
from .promotions import TrustPromotionRecord
from .research_context import ResearchContextPack, ZoneLevel
from .signals import ModelSignal

__all__ = [
    "CONTRACT_VERSION",
    "ModelSignal",
    "DecisionEvent",
    "TrustPromotionRecord",
    "ResearchContextPack",
    "ZoneLevel",
    "RunManifest",
    "EVENT_TYPES",
    "KIND_TO_EVENT_TYPE",
    "SENTINEL_KIND_TO_EVENT_TYPE",
]
