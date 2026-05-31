"""Null tests for alpha validation.

The question every alpha must answer: "is the realized R distribution
distinguishable from what you'd get by trading at random times with the
same per-trade cost structure?" If the answer is no, the alpha has no
edge — even if its raw mean R looks positive.

Two null tests here:

  Time-shuffle null:
    Take the SAME alpha's signals but RANDOMIZE the decision timestamps
    within each asset (preserving the per-asset signal count). Re-execute,
    measure mean R. Repeat many times. Compare actual mean R to the null
    distribution; one-sided p-value = fraction of null trials >= actual.

  Sign-flip null:
    Take the alpha's signals but RANDOMLY FLIP a fraction of sides
    (long <-> short). The alpha's direction call gets corrupted; if its
    edge survives, the signal-emission timing alone carries info; if it
    collapses, direction is the edge.

These are deliberately CHEAP nulls (no model retraining) that test
specific causal hypotheses. A proper CPCV / DSR validation lives in
``liqpool.validation`` and is a downstream check, not a quick gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .base import Alpha, AlphaSignal
from .evaluator import ArsenalEvaluator, EvaluatorConfig


@dataclass
class NullResult:
    """One null-test outcome for one alpha."""
    alpha_name: str
    test_name: str
    actual_mean_R: float
    null_mean_R_mean: float
    null_mean_R_std: float
    null_p_value: float           # one-sided: P(null_mean_R >= actual_mean_R)
    n_trials: int
    actual_n_trades: int


def _shuffled_signals_random_times(
    signals: Sequence[AlphaSignal],
    df_lengths_by_symbol: dict,
    rng: np.random.Generator,
) -> List[AlphaSignal]:
    """Randomize each signal's decision_idx within its asset's bar range.

    Preserves: signal count per asset, side, stop/target/horizon, confidence.
    Randomizes: when the signal fires (timing-null).

    Returns a NEW list; original signals are not mutated.
    """
    shuffled: List[AlphaSignal] = []
    for sig in signals:
        n_bars = df_lengths_by_symbol.get(sig.symbol)
        if not n_bars or n_bars <= 1:
            continue
        new_idx = int(rng.integers(low=0, high=max(1, n_bars - 1)))
        shuffled.append(AlphaSignal(
            alpha_name=sig.alpha_name,
            symbol=sig.symbol,
            decision_at=sig.decision_at,         # the timestamp is bookkeeping;
            decision_idx=new_idx,                # the evaluator uses decision_idx.
            side=sig.side,
            entry_reference=sig.entry_reference,
            stop_atr=sig.stop_atr,
            target_atr=sig.target_atr,
            horizon_bars=sig.horizon_bars,
            confidence=sig.confidence,
            state=sig.state,
        ))
    return shuffled


def _shuffled_signals_sign_flip(
    signals: Sequence[AlphaSignal],
    flip_fraction: float,
    rng: np.random.Generator,
) -> List[AlphaSignal]:
    """Flip a random fraction of signals' sides (long <-> short)."""
    shuffled: List[AlphaSignal] = []
    for sig in signals:
        if rng.random() < flip_fraction:
            new_side = "short" if sig.side == "long" else "long"
        else:
            new_side = sig.side
        shuffled.append(AlphaSignal(
            alpha_name=sig.alpha_name,
            symbol=sig.symbol,
            decision_at=sig.decision_at,
            decision_idx=sig.decision_idx,
            side=new_side,
            entry_reference=sig.entry_reference,
            stop_atr=sig.stop_atr,
            target_atr=sig.target_atr,
            horizon_bars=sig.horizon_bars,
            confidence=sig.confidence,
            state=sig.state,
        ))
    return shuffled


# ---------------------------------------------------------------------------
# Top-level harness
# ---------------------------------------------------------------------------

def collect_signals_per_asset(alpha: Alpha, report,
                              atr_period: int = 14,
                              extras: Optional[dict] = None,
                              ) -> List[AlphaSignal]:
    """Run the alpha across all assets in the bundle, return the unified
    signal list. Used as the "actual" reference before null-shuffling."""
    from liqpool.indicators import atr
    from liqpool.sectors import sector_of
    extras = extras or {}
    signals: List[AlphaSignal] = []
    for symbol, ad in report.assets.items():
        if ad.base_df is None or ad.base_df.empty:
            continue
        atr_series = atr(ad.base_df, atr_period).bfill()
        extra = {"asset_data": ad, "sector": sector_of(symbol), **extras}
        signals.extend(alpha.candidates(
            symbol=symbol, df_base=ad.base_df,
            atr_series=atr_series, extra=extra,
        ))
    return signals


def _signal_executor(alpha: Alpha, signals: Sequence[AlphaSignal],
                     report, config: EvaluatorConfig) -> pd.DataFrame:
    """Run the given signals through the evaluator pipeline.

    We bypass the registry/Alpha.candidates() loop and inject pre-built
    signals via a temporary one-shot alpha that just returns them.
    """
    from .base import Alpha as _A

    by_symbol: dict = {}
    for s in signals:
        by_symbol.setdefault(s.symbol, []).append(s)

    class _StaticAlpha(_A):
        @property
        def name(self) -> str:
            return alpha.name
        def candidates(self, *, symbol, df_base, atr_series, extra=None):
            return list(by_symbol.get(symbol, []))

    ev = ArsenalEvaluator([_StaticAlpha()], config=config)
    return ev.run(report)


def time_shuffle_null(alpha: Alpha, report,
                      config: Optional[EvaluatorConfig] = None,
                      n_trials: int = 100, seed: int = 17,
                      atr_period: int = 14,
                      extras: Optional[dict] = None,
                      ) -> NullResult:
    """Permutation null: scramble signal decision indices within each
    asset's bar range, re-execute, measure mean R per trial. Returns the
    one-sided p-value vs the actual alpha's mean R."""
    config = config or EvaluatorConfig()
    extras = extras or {}
    actual_signals = collect_signals_per_asset(alpha, report, atr_period, extras)
    if not actual_signals:
        return NullResult(
            alpha_name=alpha.name, test_name="time_shuffle",
            actual_mean_R=0.0, null_mean_R_mean=0.0, null_mean_R_std=0.0,
            null_p_value=float("nan"), n_trials=0, actual_n_trades=0,
        )
    actual_trades = _signal_executor(alpha, actual_signals, report, config)
    actual_mean = float(actual_trades["net_r"].mean()) if len(actual_trades) else 0.0
    actual_n = int(len(actual_trades))

    df_lengths = {sym: len(ad.base_df) for sym, ad in report.assets.items()
                  if ad.base_df is not None}

    rng = np.random.default_rng(seed)
    null_means: List[float] = []
    for _ in range(n_trials):
        sh = _shuffled_signals_random_times(actual_signals, df_lengths, rng)
        if not sh:
            continue
        trades = _signal_executor(alpha, sh, report, config)
        if trades.empty:
            null_means.append(0.0)
        else:
            null_means.append(float(trades["net_r"].mean()))

    null_arr = np.asarray(null_means, dtype=float)
    null_mean = float(null_arr.mean()) if len(null_arr) else 0.0
    null_std = float(null_arr.std(ddof=1)) if len(null_arr) > 1 else 0.0
    if len(null_arr):
        p = float((null_arr >= actual_mean).mean())
    else:
        p = float("nan")
    return NullResult(
        alpha_name=alpha.name, test_name="time_shuffle",
        actual_mean_R=actual_mean, null_mean_R_mean=null_mean,
        null_mean_R_std=null_std, null_p_value=p,
        n_trials=int(len(null_arr)), actual_n_trades=actual_n,
    )


def sign_flip_null(alpha: Alpha, report,
                   config: Optional[EvaluatorConfig] = None,
                   n_trials: int = 50, flip_fraction: float = 0.50,
                   seed: int = 23, atr_period: int = 14,
                   extras: Optional[dict] = None,
                   ) -> NullResult:
    """Sign-flip null: flip a fraction of signals' sides at random,
    re-execute, measure mean R per trial. Tests whether the alpha's
    direction prediction is the source of edge."""
    config = config or EvaluatorConfig()
    extras = extras or {}
    actual_signals = collect_signals_per_asset(alpha, report, atr_period, extras)
    if not actual_signals:
        return NullResult(
            alpha_name=alpha.name, test_name="sign_flip",
            actual_mean_R=0.0, null_mean_R_mean=0.0, null_mean_R_std=0.0,
            null_p_value=float("nan"), n_trials=0, actual_n_trades=0,
        )
    actual_trades = _signal_executor(alpha, actual_signals, report, config)
    actual_mean = float(actual_trades["net_r"].mean()) if len(actual_trades) else 0.0
    actual_n = int(len(actual_trades))

    rng = np.random.default_rng(seed)
    null_means: List[float] = []
    for _ in range(n_trials):
        sh = _shuffled_signals_sign_flip(actual_signals, flip_fraction, rng)
        trades = _signal_executor(alpha, sh, report, config)
        if trades.empty:
            null_means.append(0.0)
        else:
            null_means.append(float(trades["net_r"].mean()))

    null_arr = np.asarray(null_means, dtype=float)
    null_mean = float(null_arr.mean()) if len(null_arr) else 0.0
    null_std = float(null_arr.std(ddof=1)) if len(null_arr) > 1 else 0.0
    if len(null_arr):
        p = float((null_arr >= actual_mean).mean())
    else:
        p = float("nan")
    return NullResult(
        alpha_name=alpha.name, test_name="sign_flip",
        actual_mean_R=actual_mean, null_mean_R_mean=null_mean,
        null_mean_R_std=null_std, null_p_value=p,
        n_trials=int(len(null_arr)), actual_n_trades=actual_n,
    )
