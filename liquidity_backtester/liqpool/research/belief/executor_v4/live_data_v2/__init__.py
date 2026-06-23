"""LiveDataV2 — full microstructure pipeline from Kite quotes.

Founder 2026-06-22 follow-up: "we already have L2 + OI from Kite; we
just throw 80% of it away. Use it." This package owns the path that
captures and propagates everything Kite gives us.

Public API:

    from .enriched_quote import EnrichedQuote, DepthLevel
    from .enricher import enrich_kite_quote
    from .oi_history import OIHistoryTracker
    from .cancel_rate_tracker import CancelRateTracker
    from .microstructure_features import build_microstructure_features

The pipeline:

    raw Kite quote payload
        ↓ enrich_kite_quote()
    EnrichedQuote (5 depth levels each side, OI, buy/sell pressure)
        ↓ OIHistoryTracker.observe()
        ↓ CancelRateTracker.observe()
    microstructure_features dict per slot
        ↓
    slot_reading dict (manager sees this)

Why a dedicated pipeline: the legacy ``kite_quote_to_belief_quote``
collapses everything to top-of-book bid/ask. The microstructure
detectors need the unflattened picture — without it iceberg/layering/
OI-velocity detection is impossible.
"""
from .enriched_quote import (
    DepthLevel,
    EnrichedQuote,
    extract_pressure_ratio,
)
from .enricher import enrich_kite_quote, slot_reading_from_enriched
from .oi_history import OIHistoryTracker, OIVelocityReading
from .cancel_rate_tracker import CancelRateTracker, CancelRateReading
from .microstructure_features import (
    SlotMicrostructureFeatures,
    build_microstructure_features,
)

__all__ = [
    "DepthLevel", "EnrichedQuote", "extract_pressure_ratio",
    "enrich_kite_quote", "slot_reading_from_enriched",
    "OIHistoryTracker", "OIVelocityReading",
    "CancelRateTracker", "CancelRateReading",
    "SlotMicrostructureFeatures", "build_microstructure_features",
]
