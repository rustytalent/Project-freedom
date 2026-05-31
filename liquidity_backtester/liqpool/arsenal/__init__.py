"""Alpha arsenal — framework for hosting multiple independent signal generators.

Public surface:

  Alpha, AlphaSignal               — base abstractions in arsenal.base
  AlphaRegistry, default_registry  — alpha lookup
  ArsenalEvaluator, EvaluatorConfig— shared MIS+cost pipeline
  NullResult, time_shuffle_null,
  sign_flip_null                   — permutation null tests

Bundled alphas (instantiated by default_registry):
  LiquidityPoolReachAlpha    — the existing pool strategy as an alpha
  MeanReversionAlpha         — z-score mean reversion (independent of pools)
  MomentumAlpha              — multi-bar momentum continuation

Add a new alpha by subclassing :class:`Alpha`, implementing
:meth:`Alpha.candidates`, and registering it with an
:class:`AlphaRegistry`. The :class:`ArsenalEvaluator` handles execution,
costs, and aggregation; subclasses focus on signal generation alone.
"""
from __future__ import annotations

from .base import Alpha, AlphaSignal, signals_to_frame
from .evaluator import (
    TRADE_COLUMNS,
    ArsenalEvaluator,
    EvaluatorConfig,
)
from .meta import MetaANDAlpha, MetaORAlpha, MetaWeightedAlpha
from .null_tests import (
    NullResult,
    collect_signals_per_asset,
    sign_flip_null,
    time_shuffle_null,
)
from .registry import AlphaRegistry, default_registry

__all__ = [
    "Alpha",
    "AlphaSignal",
    "AlphaRegistry",
    "ArsenalEvaluator",
    "EvaluatorConfig",
    "MetaANDAlpha",
    "MetaORAlpha",
    "MetaWeightedAlpha",
    "NullResult",
    "TRADE_COLUMNS",
    "collect_signals_per_asset",
    "default_registry",
    "sign_flip_null",
    "signals_to_frame",
    "time_shuffle_null",
]
