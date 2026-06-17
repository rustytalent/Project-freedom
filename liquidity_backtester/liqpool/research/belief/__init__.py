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

from .battlefield import (
    BattlefieldConfig,
    BattlefieldSnapshot,
    FIELD_BEARISH_AGREEMENT,
    FIELD_BULLISH_AGREEMENT,
    FIELD_QUIET,
    FIELD_SINGLE_DISTORTION,
    FIELD_VOL_CONTRACTION,
    FIELD_VOL_EXPANSION,
    RAIL_MIXED,
    RAIL_QUIET,
    RAIL_STRONG_BEAR,
    RAIL_STRONG_BULL,
    RailSummary,
    SlotReading,
    battlefield_snapshot,
    make_readings,
)
from .fair_response import (
    FairResponseConfig,
    estimate_effective_delta,
    fair_response_frame,
)
from .iv_state import (
    IV_COMMON_SHOCK,
    IV_DIRECTIONAL_BEAR,
    IV_DIRECTIONAL_BULL,
    IV_DIRTY_DATA,
    IV_LIQUIDITY_DISTORTION,
    IV_NEUTRAL,
    IV_VOL_CONTRACTION,
    IVState,
    IVStateConfig,
    classify_iv_state,
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
from .state_machines import (
    BearContinuationMachine,
    BullContinuationMachine,
    LiquiditySweepMachine,
    StateUpdate,
)
from .thesis_memory import (
    BEAR_ENTRY,
    BULL_ENTRY,
    EXIT_BEAR,
    EXIT_BULL,
    HOLD_BEAR,
    HOLD_BULL,
    NEUTRAL_THESIS,
    NO_TRADE_DANGER,
    ThesisMemory,
    ThesisMemoryConfig,
    ThesisSnapshot,
)
from .winding import (
    BEAR_TRAP_WINDING,
    BEARISH_WINDING_DOWN,
    BULL_TRAP_WINDING,
    BULLISH_WINDING_UP,
    NO_WINDING,
    SCALP_CE,
    SCALP_NONE,
    SCALP_PE,
    WindingConfig,
    WindingDetector,
    WindingZone,
)

__all__ = [
    "BEARISH_WINDING_DOWN",
    "BEAR_ENTRY",
    "BEAR_TRAP_WINDING",
    "BULLISH_WINDING_UP",
    "BULL_ENTRY",
    "BULL_TRAP_WINDING",
    "BattlefieldConfig",
    "BattlefieldSnapshot",
    "BearContinuationMachine",
    "BullContinuationMachine",
    "CLEAN",
    "DANGEROUS",
    "EXIT_BEAR",
    "EXIT_BULL",
    "FIELD_BEARISH_AGREEMENT",
    "FIELD_BULLISH_AGREEMENT",
    "FIELD_QUIET",
    "FIELD_SINGLE_DISTORTION",
    "FIELD_VOL_CONTRACTION",
    "FIELD_VOL_EXPANSION",
    "FairResponseConfig",
    "HOLD_BEAR",
    "HOLD_BULL",
    "IMPROVING",
    "IVState",
    "IVStateConfig",
    "IV_COMMON_SHOCK",
    "IV_DIRECTIONAL_BEAR",
    "IV_DIRECTIONAL_BULL",
    "IV_DIRTY_DATA",
    "IV_LIQUIDITY_DISTORTION",
    "IV_NEUTRAL",
    "IV_VOL_CONTRACTION",
    "LiquiditySweepMachine",
    "MarkPrice",
    "MarkPriceConfig",
    "MarkPriceTracker",
    "MoneynessSlot",
    "NEUTRAL_THESIS",
    "NO_TRADE_DANGER",
    "NO_WINDING",
    "Quote",
    "RAIL_MIXED",
    "RAIL_QUIET",
    "RAIL_STRONG_BEAR",
    "RAIL_STRONG_BULL",
    "RailSummary",
    "ResidualConfig",
    "SCALP_CE",
    "SCALP_NONE",
    "SCALP_PE",
    "SlotReading",
    "SpreadConfig",
    "StateUpdate",
    "ThesisMemory",
    "ThesisMemoryConfig",
    "ThesisSnapshot",
    "WIDENING",
    "WindingConfig",
    "WindingDetector",
    "WindingZone",
    "battlefield_snapshot",
    "build_chain_slots",
    "classify_iv_state",
    "classify_moneyness",
    "compute_mark",
    "detect_identity_anomaly",
    "deviation_of_deviation",
    "estimate_effective_delta",
    "expected_abs_delta",
    "fair_response_frame",
    "is_execution_friendly",
    "make_readings",
    "moneyness_behavior",
    "slot_residual_frame",
    "spread_friendliness_frame",
]
