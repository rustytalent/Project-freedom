"""Framework-agnostic serving layer for the opaque analytics feed.

This sits between an HTTP adapter (``service/app.py``) and the scoring transforms
in :mod:`liqpool.scoring`. It is deliberately free of any web-framework import so
the request contract can be unit-tested directly, and so the HTTP layer stays a
thin shell.

A :class:`Scorer` turns a ``(symbol, date)`` request into a cross-section of
:class:`~liqpool.scoring.InternalLevel` objects. Those never leave the process —
:func:`handle_levels_request` runs them through the opaque pipeline and returns
only allow-listed public records.

Scorers:

  * :class:`StubScorer` — deterministic synthetic levels. Used by tests and for
    local smoke-testing the HTTP layer without any raw data.
  * :class:`~liqpool.ingest.RawFeedScorer` (re-exported) — ingests the raw
    output files a predict run already writes and converts them, touching no
    model code. This is the production scorer.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from liqpool.ingest import RawFeedScorer  # noqa: F401  (re-exported for callers)
from liqpool.scoring import (
    FEED_ANALYTICS_TYPE,
    FEED_VERSION,
    BlendWeights,
    InternalLevel,
    compliance_notice,
    score_levels,
)


class Scorer(Protocol):
    def internal_levels(self, symbol: str, date: str,
                        universe: str | None = None) -> Sequence[InternalLevel]:
        ...


def handle_levels_request(scorer: Scorer, *, symbol: str, date: str,
                          customer_id: str, universe: str | None = None,
                          weights: BlendWeights = BlendWeights(),
                          customer_tier: str = "licensed") -> dict:
    """Produce the public feed envelope for one instrument/day/customer.

    The returned dict contains only opaque, allow-listed records plus a
    non-recommendatory disclaimer. No internal estimate is exposed.
    """
    if not symbol:
        raise ValueError("symbol is required")
    if not customer_id:
        raise ValueError("customer_id is required")
    levels = list(scorer.internal_levels(symbol, date, universe))
    observations = score_levels(levels, customer_id=customer_id, day=date,
                                weights=weights,
                                customer_tier=customer_tier)
    as_of = levels[0].as_of if levels else date
    return {
        "analytics_type": FEED_ANALYTICS_TYPE,
        "instrument": symbol,
        "as_of": str(as_of),
        "feed_version": FEED_VERSION,
        "observation_count": len(observations),
        "observations": observations,
        "compliance_notice": compliance_notice(),
    }


class StubScorer:
    """Deterministic synthetic scorer for tests and local smoke-testing.

    Produces a fixed set of levels per symbol with a spread of internal signals,
    so the opaque pipeline and HTTP contract can be exercised without a trained
    model. The numbers are arbitrary fixtures, not real analytics.
    """

    def __init__(self, n_above: int = 5, n_below: int = 5):
        self.n_above = n_above
        self.n_below = n_below

    def internal_levels(self, symbol: str, date: str,
                        universe: str | None = None) -> list[InternalLevel]:
        levels: list[InternalLevel] = []
        base = 100.0 + (sum(ord(c) for c in symbol) % 50)
        for i in range(self.n_above):
            frac = (i + 1) / (self.n_above + 1)
            mid = base * (1.0 + 0.01 * (i + 1))
            levels.append(InternalLevel(
                symbol=symbol, side="above",
                level_low=mid * 0.999, level_high=mid * 1.001, level_mid=mid,
                p_touch=0.30 + 0.6 * frac,
                p_up=0.45 + 0.4 * frac,
                q=0.40 + 0.5 * frac,
                as_of=f"{date}T15:30:00+05:30",
            ))
        for i in range(self.n_below):
            frac = (i + 1) / (self.n_below + 1)
            mid = base * (1.0 - 0.01 * (i + 1))
            levels.append(InternalLevel(
                symbol=symbol, side="below",
                level_low=mid * 0.999, level_high=mid * 1.001, level_mid=mid,
                p_touch=0.30 + 0.6 * frac,
                p_up=0.55 - 0.4 * frac,
                q=0.40 + 0.5 * frac,
                as_of=f"{date}T15:30:00+05:30",
            ))
        return levels
