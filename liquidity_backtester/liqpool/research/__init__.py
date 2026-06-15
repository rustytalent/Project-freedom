"""Research — the alpha-discovery sub-package.

The Profitability Doctrine identified that the missing link in the
codebase is not infrastructure but EDGE. We have a beautiful kitchen
and no recipes that have been proven to make food.

This package is where recipes (hypotheses) get tested against the
warehouse honestly. The discipline:

  1. Every hypothesis is a CALLABLE that, given a bar series,
     emits {entries, exits}.
  2. The harness simulates trades using real costs (sentinel.india_tax
     STT + GST + brokerage + slippage) and walk-forward windows
     (purged + embargoed, same as the quality model).
  3. The harness returns a HypothesisReport with Sharpe, Sortino,
     max_dd, expectancy_R, hit_rate, and per-regime breakdowns.
  4. Decision rule: keep hypotheses with OOS Sharpe ≥ 1.0 after
     costs; kill the rest. Two-thirds will die. That's normal.

Public surface:
  HypothesisSpec     — what to test
  HypothesisReport   — what came back
  HypothesisHarness  — the runner
  HYPOTHESIS_LIBRARY — the registry of candidate strategies
"""
from .harness import (
    BacktestTrade, HypothesisHarness, HypothesisReport, HypothesisSpec,
    simulate_trades, trade_stats,
)
from .library import HYPOTHESIS_LIBRARY, register_hypothesis

__all__ = [
    "BacktestTrade",
    "HypothesisHarness",
    "HypothesisReport",
    "HypothesisSpec",
    "simulate_trades",
    "trade_stats",
    "HYPOTHESIS_LIBRARY",
    "register_hypothesis",
]
