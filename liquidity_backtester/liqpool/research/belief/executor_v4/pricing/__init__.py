"""Pricing layer — Black-Scholes + IV surface + strategy Greeks.

This is the state-of-the-art math foundation that replaces the toy
linear-decay Greeks model used in Sprint 4. The strategy library,
risk layer, and hedge proposer all consume these.
"""
from __future__ import annotations

from .black_scholes import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_RISK_FREE_RATE,
    OptionGreeks,
    bs_call_price,
    bs_put_price,
    call_greeks,
    greeks_for_side,
    implied_volatility,
    norm_cdf,
    norm_pdf,
    put_greeks,
)
from .iv_surface import (
    IVSurface,
    IVSurfaceConfig,
    IVSurfaceFitter,
)
from .strategy_greeks import (
    PerLegGreeks,
    StrategyGreeksProfile,
    compute_strategy_greeks,
)

__all__ = [
    "DEFAULT_DIVIDEND_YIELD",
    "DEFAULT_RISK_FREE_RATE",
    "IVSurface",
    "IVSurfaceConfig",
    "IVSurfaceFitter",
    "OptionGreeks",
    "PerLegGreeks",
    "StrategyGreeksProfile",
    "bs_call_price",
    "bs_put_price",
    "call_greeks",
    "compute_strategy_greeks",
    "greeks_for_side",
    "implied_volatility",
    "norm_cdf",
    "norm_pdf",
    "put_greeks",
]
