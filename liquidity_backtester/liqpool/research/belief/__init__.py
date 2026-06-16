"""Premium Belief Engine — a live option-battlefield interpreter.

This sub-package is the founder's architecture (2026-06-16). It is NOT a
candle/wick indicator. It reads the live option quote stream across the
ATM ±N strike battlefield (typically 22 contracts: ATM ±5 × CE/PE) and
answers one question:

    Given the current underlying move, the time of day, each strike's
    moneyness identity, the spread/liquidity state, the IV/skew pressure,
    and the recent premium memory — are the contracts behaving NORMALLY,
    or revealing abnormal belief?

Normal traders watch where the underlying went. This engine watches
whether the option battlefield ACCEPTED that move.

Architecture (built in phases, shadow-mode first):

    Premium Belief Engine
    ├── Data Quality Layer        ── mark_price.py        [Phase 2] ✅
    ├── Moneyness Identity Layer   ── moneyness.py          [Phase 3] ✅
    ├── Fair Response Model        ── (fair_response.py)    [Phase 4]
    ├── Residual / Dev-of-Dev      ── (residual.py)         [Phase 4]
    ├── Spread & Liquidity Layer   ── (spread.py)           [Phase 4]
    ├── Multi-Strike Battlefield   ── (battlefield.py)      [Phase 5]
    ├── IV / Skew Pressure Layer   ── (iv_state.py)         [Phase 5]
    ├── Thesis Memory Layer        ── (thesis_memory.py)    [Phase 6]
    ├── Winding Zone Detector      ── (winding.py)          [Phase 7]
    ├── State Machines             ── (state_machines.py)   [Phase 7]
    └── Decision Layer             ── (decision.py / engine) [Phase 8]

Honest scope (the founder's own verdict, preserved):
  * As a bad-trade avoider: very strong potential.
  * As a fake-pullback holder: very strong potential.
  * As an exit improver: strong potential.
  * As a standalone scalper: possible, only after serious shadow testing.
  * As an every-situation-proof engine: impossible (news/IV shocks exist).
  * As a serious trading engine: worth building.

See ROADMAP.md for the full build order and design notes.
"""
from __future__ import annotations

from .fair_response import (
    FairResponseConfig,
    estimate_effective_delta,
    fair_response_frame,
)
from .mark_price import (
    MarkPrice,
    MarkPriceConfig,
    MarkPriceTracker,
    Quote,
    compute_mark,
)
from .moneyness import (
    MoneynessSlot,
    build_chain_slots,
    classify_moneyness,
    detect_identity_anomaly,
    expected_abs_delta,
    moneyness_behavior,
)
from .residual import (
    ResidualConfig,
    deviation_of_deviation,
    slot_residual_frame,
)
from .spread import (
    CLEAN,
    DANGEROUS,
    IMPROVING,
    WIDENING,
    SpreadConfig,
    is_execution_friendly,
    spread_friendliness_frame,
)

__all__ = [
    "CLEAN",
    "DANGEROUS",
    "FairResponseConfig",
    "IMPROVING",
    "MarkPrice",
    "MarkPriceConfig",
    "MarkPriceTracker",
    "MoneynessSlot",
    "Quote",
    "ResidualConfig",
    "SpreadConfig",
    "WIDENING",
    "build_chain_slots",
    "classify_moneyness",
    "compute_mark",
    "detect_identity_anomaly",
    "deviation_of_deviation",
    "estimate_effective_delta",
    "expected_abs_delta",
    "fair_response_frame",
    "is_execution_friendly",
    "moneyness_behavior",
    "slot_residual_frame",
    "spread_friendliness_frame",
]
