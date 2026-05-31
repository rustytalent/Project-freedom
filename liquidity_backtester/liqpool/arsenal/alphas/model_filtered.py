"""Model-filtered alphas: every trained model in the bundle, exposed as an
Alpha that GATES the pool-reach base population by that model's prediction.

This is the user-requested unlock: each existing trained model
(quality / direction / R1 force regressor / proximity / reaction) is
already a predictor. With a threshold rule it becomes a signal generator.
This module wraps the most useful ones as Alpha subclasses so the arsenal
evaluator can compare them head-to-head.

Design choice: instead of building "new" alphas that emit from scratch,
each wrapped alpha takes the SAME pool-touched population as
:class:`LiquidityPoolReachAlpha` and FILTERS it by its model's prediction.
This makes the arsenal's per-alpha summary directly interpretable as
"what does filter X add on top of pool_reach?"

Trade-off: this excludes pre-touch signals from these wrapped alphas. If
the model is meant to fire BEFORE a touch (Track A direction model on
snapshots), the proper backtest is the dedicated sweep in
``analysis/run_phase4_track_a_pretouch_sweep.py``. The wrapped alphas
here intentionally stay on the post-touch population so they're
comparable to the rest of the arsenal.

Each wrapped alpha exposes ``threshold`` as a constructor argument so the
user can grid-search thresholds with multiple registry entries:

    reg.register(QualityFilteredPoolAlpha(min_q=0.55, name="quality_top_45pct"))
    reg.register(QualityFilteredPoolAlpha(min_q=0.50, name="quality_top_50pct"))
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import Alpha, AlphaSignal
from .pool_reach import LiquidityPoolReachAlpha, _headline_for_pool, _tf_bucket


# ---------------------------------------------------------------------------
# Common helpers — re-used by all wrapped alphas
# ---------------------------------------------------------------------------

def _iter_touched_pools(df_base: pd.DataFrame, ad) -> List[tuple]:
    """Yield (pool, result, decision_idx) for every OOS pool that was
    touched in-bundle. Decision_idx is the touch bar; the evaluator enters
    on the next bar's open."""
    out: List[tuple] = []
    pools = ad.walkforward.oos_pools
    results = ad.walkforward.oos_results
    for pool, result in zip(pools, results):
        if result.touched_at is None:
            continue
        idx = int(df_base.index.searchsorted(
            pd.Timestamp(result.touched_at), side="left"
        ))
        if idx <= 0 or idx >= len(df_base):
            continue
        out.append((pool, result, idx))
    return out


def _base_signal_for_pool(alpha_name: str, symbol: str,
                          df_base: pd.DataFrame, pool, result,
                          decision_idx: int,
                          stop_atr: float = 0.5, target_atr: float = 2.0,
                          horizon_bars: int = 40,
                          confidence: float = 0.5,
                          state: Optional[Dict[str, Any]] = None) -> AlphaSignal:
    """Construct the canonical pool-reach signal at the given touch bar,
    parametrized for the wrapping alpha."""
    side = "long" if pool.side == "low" else "short"
    entry_ref = float(pool.price_high if pool.side == "low"
                      else pool.price_low)
    s = {
        "pool_idx": int(result.pool_idx),
        "pool_score": float(pool.score),
        "factor": _headline_for_pool(pool),
        "tf_count": int(len(set(pool.tfs))),
    }
    if state:
        s.update(state)
    return AlphaSignal(
        alpha_name=alpha_name, symbol=symbol,
        decision_at=df_base.index[decision_idx], decision_idx=decision_idx,
        side=side, entry_reference=entry_ref,
        stop_atr=stop_atr, target_atr=target_atr, horizon_bars=horizon_bars,
        confidence=confidence, state=s,
    )


# ---------------------------------------------------------------------------
# QualityFilteredPoolAlpha — wraps the Q model (SectorMoERespectModel)
# ---------------------------------------------------------------------------

class QualityFilteredPoolAlpha(Alpha):
    """Pool-reach base, gated by the quality model's predicted P(respect).

    Only emit a signal when the Q model predicts the pool will be respected
    above ``min_q``. Default ``min_q=0.50`` is permissive (most pools); set
    higher (e.g. 0.55) for a more selective gate.

    Requires ``report.unified_ml`` and ``report.unified_featurizer`` in the
    bundle. Falls back gracefully (emits nothing) if either is missing.
    """

    def __init__(self, min_q: float = 0.50,
                 name: str = "quality_filtered_pool",
                 horizon_bars: int = 40) -> None:
        self._name = name
        self.min_q = float(min_q)
        self._horizon = int(horizon_bars)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return (
            f"Pool-reach base, gated by quality model. Emits only when the "
            f"SectorMoE Q model predicts P(respect) >= {self.min_q:.2f}. "
            f"Default barriers 0.5/2.0/{self._horizon} bars."
        )

    def candidates(self, *, symbol: str, df_base: pd.DataFrame,
                   atr_series: pd.Series,
                   extra: Optional[Dict[str, Any]] = None,
                   ) -> List[AlphaSignal]:
        extra = extra or {}
        ad = extra.get("asset_data")
        report = extra.get("report")
        if ad is None or report is None:
            return []
        ml = getattr(report, "unified_ml", None)
        feat = getattr(report, "unified_featurizer", None)
        if ml is None or feat is None:
            return []

        touched = _iter_touched_pools(df_base, ad)
        if not touched:
            return []
        # Predict Q for all touched pools in one batch (faster).
        pools = [p for p, _, _ in touched]
        try:
            X = feat.transform_batch(pools)
            q_preds = ml.predict(X, pools=pools)
        except Exception:
            return []
        q_preds = np.asarray(q_preds, dtype=float)

        signals: List[AlphaSignal] = []
        for (pool, result, idx), q in zip(touched, q_preds):
            if not np.isfinite(q) or q < self.min_q:
                continue
            signals.append(_base_signal_for_pool(
                self._name, symbol, df_base, pool, result, idx,
                horizon_bars=self._horizon,
                confidence=float(q),
                state={"q_pred": float(q)},
            ))
        return signals

    def regime_tags(self, signal, df_base, atr_series):
        tags = super().regime_tags(signal, df_base, atr_series)
        tags["factor"] = str(signal.state.get("factor", "OTHER"))
        tags["tf_bucket"] = _tf_bucket(int(signal.state.get("tf_count", 0)))
        # Q quartile so the report can show "top-25% Q outperforms?".
        q = float(signal.state.get("q_pred", 0.5))
        tags["q_bucket"] = ("q_top_25" if q >= 0.55 else
                            "q_50_to_55" if q >= 0.50 else
                            "q_below_50")
        return tags


# ---------------------------------------------------------------------------
# DirectionConfirmedPoolAlpha — wraps the direction model
# ---------------------------------------------------------------------------

class DirectionConfirmedPoolAlpha(Alpha):
    """Pool-reach base, gated by the direction model agreeing with the
    pool side.

    Long signals (pool below price) require P(up) >= ``min_direction``;
    short signals (pool above price) require P(up) <= 1 - ``min_direction``.
    Default 0.55 = mild alignment requirement.

    Requires ``report.unified_direction`` and ``report.unified_featurizer``
    in the bundle. Falls back gracefully (emits nothing) if missing.
    """

    def __init__(self, min_direction: float = 0.55,
                 name: str = "direction_confirmed_pool",
                 horizon_bars: int = 40) -> None:
        if not 0.50 < min_direction <= 0.95:
            raise ValueError(
                f"min_direction must be in (0.50, 0.95], got {min_direction}"
            )
        self._name = name
        self.min_direction = float(min_direction)
        self._horizon = int(horizon_bars)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return (
            f"Pool-reach base, gated by direction model alignment. Long "
            f"signals require P(up) >= {self.min_direction:.2f}; short "
            f"signals require P(up) <= {1 - self.min_direction:.2f}."
        )

    def candidates(self, *, symbol: str, df_base: pd.DataFrame,
                   atr_series: pd.Series,
                   extra: Optional[Dict[str, Any]] = None,
                   ) -> List[AlphaSignal]:
        extra = extra or {}
        ad = extra.get("asset_data")
        report = extra.get("report")
        if ad is None or report is None:
            return []
        direction_model = getattr(report, "unified_direction", None)
        if direction_model is None:
            return []

        # Build a StateFeaturizer for this asset to query the direction model
        # at touch bars. Lazy import to avoid module-load cost when this alpha
        # isn't registered.
        from liqpool.timing import StateFeaturizer
        try:
            sf = StateFeaturizer(df_base)
        except Exception:
            return []

        touched = _iter_touched_pools(df_base, ad)
        signals: List[AlphaSignal] = []
        for pool, result, idx in touched:
            # State at the touch bar's close (decision time).
            try:
                state = sf.features_at(idx, active_pools=[])
                p_up = float(direction_model.predict_state(state))
            except Exception:
                continue
            if not np.isfinite(p_up):
                continue
            side = "long" if pool.side == "low" else "short"
            if side == "long" and p_up < self.min_direction:
                continue
            if side == "short" and (1.0 - p_up) < self.min_direction:
                continue
            confidence = p_up if side == "long" else (1.0 - p_up)
            signals.append(_base_signal_for_pool(
                self._name, symbol, df_base, pool, result, idx,
                horizon_bars=self._horizon,
                confidence=float(confidence),
                state={"p_up": float(p_up),
                       "direction_alignment": float(confidence)},
            ))
        return signals

    def regime_tags(self, signal, df_base, atr_series):
        tags = super().regime_tags(signal, df_base, atr_series)
        align = float(signal.state.get("direction_alignment", 0.5))
        tags["direction_alignment_bucket"] = (
            "strong_align" if align >= 0.70 else
            "moderate_align" if align >= 0.60 else "mild_align"
        )
        return tags


# ---------------------------------------------------------------------------
# PolicyReturnAlpha — wraps the R1 force regressor (per-mode)
# ---------------------------------------------------------------------------

class PolicyReturnAlpha(Alpha):
    """Pool-reach base, gated by the R1 PolicyReturnModel's predicted R.

    For each touched pool, asks the per-mode return regressor "what R does
    it predict for this trade?" Only emit when predicted R exceeds
    ``min_predicted_r``. Default 0.0 = positive-expectancy filter.

    Requires ``report.policy_return_model`` (added by the R1 build). The
    regressor must have a model trained for ``execution_mode``.
    """

    def __init__(self, execution_mode: str = "touch_confirmed",
                 min_predicted_r: float = 0.0,
                 name: Optional[str] = None,
                 horizon_bars: int = 40) -> None:
        self.execution_mode = str(execution_mode)
        self.min_predicted_r = float(min_predicted_r)
        self._horizon = int(horizon_bars)
        self._name = name or f"policy_return_{execution_mode}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return (
            f"Pool-reach base, gated by R1 force regressor for mode "
            f"'{self.execution_mode}'. Only emits when predicted realized R "
            f"exceeds {self.min_predicted_r:+.2f}."
        )

    def candidates(self, *, symbol: str, df_base: pd.DataFrame,
                   atr_series: pd.Series,
                   extra: Optional[Dict[str, Any]] = None,
                   ) -> List[AlphaSignal]:
        extra = extra or {}
        ad = extra.get("asset_data")
        report = extra.get("report")
        if ad is None or report is None:
            return []
        suite = getattr(report, "policy_return_model", None)
        if suite is None or self.execution_mode not in getattr(suite, "models", {}):
            return []
        model = suite.models[self.execution_mode]

        touched = _iter_touched_pools(df_base, ad)
        if not touched:
            return []

        # Build a labels-style frame the model can consume. The R1 regressor's
        # ``predict_frame`` is robust to extra columns — it picks what it needs.
        from liqpool.sectors import sector_of
        sec = sector_of(symbol)
        rows = []
        for pool, result, idx in touched:
            side_label = "buy" if pool.side == "low" else "sell"
            direction_sign = 1 if pool.side == "low" else -1
            rows.append({
                "mode": self.execution_mode,
                "policy_target_trade_generated": 1,
                "policy_target_return_r": 0.0,           # required column, value ignored
                "symbol": symbol, "sector": sec,
                "side": side_label, "direction_sign": direction_sign,
                "pool_low": float(pool.price_low),
                "pool_high": float(pool.price_high),
                "pool_mid": float(pool.mid),
                "available_at": str(pool.available_at),
                "formed_at": str(pool.formed_at),
                "score": float(pool.score),
                "tf_count": int(len(set(pool.tfs))),
                "factor": _headline_for_pool(pool),
                "direction": "UP" if direction_sign > 0 else "DOWN",
            })
        frame = pd.DataFrame(rows)
        try:
            predicted = model.predict_frame(frame)
        except Exception:
            return []
        predicted = np.asarray(predicted, dtype=float)

        signals: List[AlphaSignal] = []
        for (pool, result, idx), pr in zip(touched, predicted):
            if not np.isfinite(pr) or pr < self.min_predicted_r:
                continue
            signals.append(_base_signal_for_pool(
                self._name, symbol, df_base, pool, result, idx,
                horizon_bars=self._horizon,
                confidence=float(min(1.0, max(0.0, 0.5 + pr * 0.10))),
                state={"predicted_r": float(pr),
                       "execution_mode": self.execution_mode},
            ))
        return signals

    def regime_tags(self, signal, df_base, atr_series):
        tags = super().regime_tags(signal, df_base, atr_series)
        pr = float(signal.state.get("predicted_r", 0.0))
        tags["predicted_r_bucket"] = (
            "high_predicted_r" if pr >= 0.30 else
            "moderate_predicted_r" if pr >= 0.10 else "low_predicted_r"
        )
        return tags
