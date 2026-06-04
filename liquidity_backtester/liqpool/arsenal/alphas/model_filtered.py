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

from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        "pool_mid": float(pool.mid),
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


# ---------------------------------------------------------------------------
# ProximityFilteredPoolAlpha — pre-touch journey-to-destination
# ---------------------------------------------------------------------------

class ProximityFilteredPoolAlpha(Alpha):
    """Pre-touch journey alpha: ride to the predicted pool touch.

    For every Nth bar with at least one active OOS pool in range, ask the
    proximity model "is this pool likely to be touched within H bars?". When
    yes (probability >= ``min_p_touch``), emit a signal NOW (pre-touch) that
    targets the journey to the pool:

      * side  = long  if pool is ABOVE price (we ride the rise to supply),
                short if pool is BELOW price (we ride the fall to demand).
      * entry = current close (decision_at), evaluator fills next-bar open.
      * target_atr = current distance to pool's NEAR boundary, capped at
                     ``max_target_atr`` so a 50-ATR-away pool can't produce a
                     50-ATR target.
      * stop_atr = configurable (default 1.0 — wider than the V1 touched
                   alpha because the signal entry is far from any structure).
      * horizon = the proximity model's shortest horizon (matches its
                  prediction window — typically 12 bars under MIS).

    Why this is different from the V1 ``pool_reach`` alpha: V1 trades the
    POST-touch rejection. This alpha trades the PRE-touch journey TO the
    pool. The two are orthogonal — a pool that reliably gets touched can be
    monetised even when the touch itself doesn't reject. This is the user's
    "journey-to-destination" thesis as an Alpha.

    Dedup: emits ONE signal per pool (the earliest qualifying bar) to avoid
    flooding the evaluator with sequential same-pool signals as the model's
    confidence ramps up.

    Requires ``report.unified_proximity`` (one model per horizon),
    ``report.unified_ml`` (for pool quality predictions used as a proximity
    input feature), and ``report.unified_featurizer``. Falls back gracefully
    (returns []) when any is missing.
    """

    DEFAULT_STOP_ATR: float = 1.0
    DEFAULT_MAX_TARGET_ATR: float = 4.0
    DEFAULT_MAX_DIST_ATR: float = 5.0
    DEFAULT_MIN_DIST_ATR: float = 0.10
    DEFAULT_SAMPLE_EVERY: int = 12
    DEFAULT_MIN_P_TOUCH: float = 0.60
    DEFAULT_MIN_SCORE: float = 0.60

    def __init__(self, min_p_touch: float = DEFAULT_MIN_P_TOUCH,
                 min_score: Optional[float] = None,
                 distance_band: Optional[Tuple[float, float]] = None,
                 allowed_sessions: Optional[Sequence[str]] = None,
                 use_soft_score: bool = False,
                 use_direction_score: bool = False,
                 use_sector_rotation: bool = False,
                 use_vol_regime: bool = False,
                 use_multi_horizon: bool = False,
                 require_opening_breakout: bool = False,
                 max_dist_atr: float = DEFAULT_MAX_DIST_ATR,
                 max_target_atr: float = DEFAULT_MAX_TARGET_ATR,
                 stop_atr: float = DEFAULT_STOP_ATR,
                 sample_every: int = DEFAULT_SAMPLE_EVERY,
                 horizon_bars: Optional[int] = None,
                 name: str = "proximity_journey") -> None:
        if not 0.0 < min_p_touch < 1.0:
            raise ValueError(
                f"min_p_touch must be in (0,1), got {min_p_touch}")
        if max_dist_atr <= 0:
            raise ValueError(f"max_dist_atr must be > 0, got {max_dist_atr}")
        if max_target_atr <= 0:
            raise ValueError(f"max_target_atr must be > 0, got {max_target_atr}")
        if stop_atr <= 0:
            raise ValueError(f"stop_atr must be > 0, got {stop_atr}")
        if sample_every < 1:
            raise ValueError(f"sample_every must be >= 1, got {sample_every}")
        self._name = name
        self.min_p_touch = float(min_p_touch)
        self.min_score = (float(min_score) if min_score is not None else None)
        self.distance_band = distance_band
        self.allowed_sessions = tuple(allowed_sessions or ())
        self.use_soft_score = bool(use_soft_score)
        self.use_direction_score = bool(use_direction_score)
        self.use_sector_rotation = bool(use_sector_rotation)
        self.use_vol_regime = bool(use_vol_regime)
        self.use_multi_horizon = bool(use_multi_horizon)
        self.require_opening_breakout = bool(require_opening_breakout)
        self.max_dist_atr = float(max_dist_atr)
        self.max_target_atr = float(max_target_atr)
        self.stop_atr = float(stop_atr)
        self.sample_every = int(sample_every)
        self._horizon_override = (int(horizon_bars)
                                  if horizon_bars is not None else None)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return (
            f"Pre-touch journey to a predicted pool touch. Emits when "
            f"proximity model P(touch <= H) >= {self.min_p_touch:.2f}"
            f"{' and soft score clears threshold' if self.use_soft_score else ''}. "
            f"Target = distance to pool (cap {self.max_target_atr:.1f} ATR), "
            f"stop {self.stop_atr:.2f} ATR, horizon = proximity model's "
            f"primary horizon. One signal per pool."
        )

    def _resolve_horizon(self, prox_models: Dict[int, Any]) -> int:
        if self._horizon_override is not None:
            return self._horizon_override
        return int(min(prox_models.keys()))

    def _opening_range(self, df_base: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        from liqpool.execution_backtest import _ist_minute_of_day
        minutes = pd.Series([_ist_minute_of_day(ts) for ts in df_base.index],
                            index=df_base.index)
        day_key = pd.Series([(ts + pd.Timedelta(hours=5, minutes=30)).date()
                             for ts in df_base.index], index=df_base.index)
        open_start = 9 * 60 + 15
        open_end = 10 * 60 + 15
        opening_high = pd.Series(np.nan, index=df_base.index, dtype=float)
        opening_low = pd.Series(np.nan, index=df_base.index, dtype=float)

        # Causal opening range: before 10:15, a decision only sees the
        # high/low accumulated so far; after 10:15, the completed range
        # is frozen for the rest of that IST session.
        for _, idxs in day_key.groupby(day_key).groups.items():
            day_idx = pd.Index(idxs)
            day_minutes = minutes.loc[day_idx]
            or_mask = (day_minutes >= open_start) & (day_minutes < open_end)
            or_idx = day_minutes.index[or_mask.to_numpy()]
            if len(or_idx) == 0:
                continue
            opening_high.loc[or_idx] = df_base.loc[or_idx, "high"].expanding().max().values
            opening_low.loc[or_idx] = df_base.loc[or_idx, "low"].expanding().min().values
            opening_high.loc[day_idx] = opening_high.loc[day_idx].ffill()
            opening_low.loc[day_idx] = opening_low.loc[day_idx].ffill()
        return opening_high, opening_low

    def _distance_score(self, dist_atr: float) -> float:
        if self.distance_band is not None:
            lo, hi = self.distance_band
            if lo <= dist_atr < hi:
                return 1.0
            return 0.0
        if 5.0 <= dist_atr < 8.0:
            return 1.0
        if 3.0 <= dist_atr < 10.0:
            return 0.80
        if 1.0 <= dist_atr < 12.0:
            return 0.55
        return 0.25

    @staticmethod
    def _truncate_asof(df: pd.DataFrame, asof_ts: pd.Timestamp) -> pd.DataFrame:
        """Return rows known by asof_ts, tolerating tz-aware/naive mismatch."""
        ts = pd.Timestamp(asof_ts)
        try:
            return df.loc[df.index <= ts]
        except TypeError:
            ts_cmp = ts.tz_localize(None) if ts.tzinfo is not None else ts
            idx = df.index
            if getattr(idx, "tz", None) is not None:
                idx_cmp = idx.tz_localize(None)
            else:
                idx_cmp = idx
            return df.loc[idx_cmp <= ts_cmp]

    @staticmethod
    def _ist_date_key(ts_like: Any) -> str:
        ts = pd.Timestamp(ts_like)
        if ts.tzinfo is not None:
            return str(ts.tz_convert("Asia/Kolkata").date())
        return str((ts + pd.Timedelta(hours=5, minutes=30)).date())

    def _sector_score(self, sector: str, trade_side: str,
                      report: Any, asof_ts: Optional[pd.Timestamp] = None
                      ) -> Tuple[float, str]:
        if not self.use_sector_rotation:
            return 0.50, "neutral"
        cache_attr = "_arsenal_sector_rotation_cache_by_date" if asof_ts is not None else "_arsenal_sector_rotation_cache"
        cache = getattr(report, cache_attr, None)
        if cache is None:
            cache = {}
            try:
                setattr(report, cache_attr, cache)
            except Exception:
                pass
        cache_key = self._ist_date_key(asof_ts) if asof_ts is not None else "full_history"
        if cache_key not in cache:
            try:
                from liqpool.sectors import compute_sector_metrics, detect_rotation
                asset_dfs = {}
                for sym, ad in report.assets.items():
                    df = getattr(ad, "base_df", None)
                    if df is None or df.empty:
                        continue
                    if asof_ts is not None:
                        df = self._truncate_asof(df, pd.Timestamp(asof_ts))
                    if len(df) >= 20:
                        asset_dfs[sym] = df
                rotation = detect_rotation(compute_sector_metrics(asset_dfs))
                cache[cache_key] = {
                    "rotation_in": set(rotation.get("rotation_in", []) or []),
                    "rotation_out": set(rotation.get("rotation_out", []) or []),
                }
            except Exception:
                cache[cache_key] = {"rotation_in": set(), "rotation_out": set()}
        rotation = cache.get(cache_key, {"rotation_in": set(), "rotation_out": set()})
        rotation_in = rotation.get("rotation_in", set())
        rotation_out = rotation.get("rotation_out", set())
        if trade_side == "long":
            if sector in rotation_in:
                return 1.0, "with_rotation_in"
            if sector in rotation_out:
                return 0.15, "against_rotation_out"
        else:
            if sector in rotation_out:
                return 1.0, "with_rotation_out"
            if sector in rotation_in:
                return 0.15, "against_rotation_in"
        return 0.55, "neutral"

    def _vol_score(self, state: Dict[str, float]) -> Tuple[float, str]:
        if not self.use_vol_regime:
            return 0.50, "not_used"
        vol = float(state.get("vol_ratio", 1.0) or 1.0)
        if 0.85 <= vol <= 1.35:
            return 1.0, "normal_vol"
        if 0.65 <= vol < 0.85:
            return 0.65, "low_vol"
        if 1.35 < vol <= 1.80:
            return 0.65, "high_vol"
        return 0.30, "extreme_vol"

    def _soft_score(self, *,
                    p_touch: float,
                    p_touch_by_h: Dict[int, float],
                    p_direction_to_pool: Optional[float],
                    dist_atr: float,
                    sector: str,
                    trade_side: str,
                    state: Dict[str, float],
                    report: Any,
                    asof_ts: Optional[pd.Timestamp] = None,
                    ) -> Tuple[float, Dict[str, Any]]:
        prox_score = float(p_touch)
        if self.use_multi_horizon and p_touch_by_h:
            vals = [float(v) for _, v in sorted(p_touch_by_h.items()) if np.isfinite(v)]
            if vals:
                urgent = vals[0]
                developing = max(vals)
                # Urgent + developing shapes both matter. A high longer-horizon
                # probability with weaker near-horizon probability is a slower
                # journey, not an automatic reject.
                prox_score = 0.60 * urgent + 0.40 * developing

        direction_score = 0.50
        if self.use_direction_score and p_direction_to_pool is not None:
            direction_score = float(np.clip(p_direction_to_pool, 0.0, 1.0))
        distance_score = self._distance_score(dist_atr)
        sector_score, sector_tag = self._sector_score(
            sector, trade_side, report, asof_ts=asof_ts
        )
        vol_score, vol_tag = self._vol_score(state)

        weights = {
            "proximity": 0.45,
            "direction": 0.20 if self.use_direction_score else 0.0,
            "distance": 0.15,
            "sector": 0.10 if self.use_sector_rotation else 0.0,
            "vol": 0.10 if self.use_vol_regime else 0.0,
        }
        used = sum(weights.values())
        # If optional dimensions are off, redistribute to proximity/distance.
        if used < 0.999:
            weights["proximity"] += 1.0 - used
        score = (
            weights["proximity"] * prox_score
            + weights["direction"] * direction_score
            + weights["distance"] * distance_score
            + weights["sector"] * sector_score
            + weights["vol"] * vol_score
        )
        components = {
            "score": float(score),
            "score_proximity": float(prox_score),
            "score_direction": float(direction_score),
            "score_distance": float(distance_score),
            "score_sector": float(sector_score),
            "score_vol": float(vol_score),
            "sector_rotation_tag": sector_tag,
            "vol_regime_tag": vol_tag,
        }
        return float(score), components

    def candidates(self, *, symbol: str, df_base: pd.DataFrame,
                   atr_series: pd.Series,
                   extra: Optional[Dict[str, Any]] = None,
                   ) -> List[AlphaSignal]:
        extra = extra or {}
        ad = extra.get("asset_data")
        report = extra.get("report")
        if ad is None or report is None:
            return []
        prox = getattr(report, "unified_proximity", None)
        if not prox:
            return []
        ml = getattr(report, "unified_ml", None)
        feat = getattr(report, "unified_featurizer", None)
        if ml is None or feat is None:
            return []

        primary_h = self._resolve_horizon(prox)
        if primary_h not in prox:
            return []
        model = prox[primary_h]
        direction_model = getattr(report, "unified_direction", None)
        sector = str(extra.get("sector", "OTHER"))

        pools = ad.walkforward.oos_pools
        results = ad.walkforward.oos_results
        if not pools:
            return []
        try:
            X = feat.transform_batch(pools)
            q_preds = np.asarray(ml.predict(X, pools=pools), dtype=float)
        except Exception:
            return []
        # Defaults for any failed Q prediction.
        q_preds = np.where(np.isfinite(q_preds), q_preds, 0.5)

        from liqpool.timing import StateFeaturizer
        try:
            sf = StateFeaturizer(df_base)
        except Exception:
            return []

        atr_vals = (atr_series.values if atr_series is not None
                    else sf.atr_14.values)
        idx_values = df_base.index
        n = len(df_base)
        # Need at least horizon bars of runway after decision to be honest.
        last_decision = max(0, n - primary_h - 1)
        first_decision = 80                        # warm-up matches snapshot generation
        opening_high = opening_low = None
        if self.require_opening_breakout:
            opening_high, opening_low = self._opening_range(df_base)

        emitted: set = set()
        close_arr = df_base["close"].values
        signals: List[AlphaSignal] = []
        for j in range(first_decision, last_decision + 1, self.sample_every):
            ts = idx_values[j]
            close_T = float(close_arr[j])
            a_T = max(float(atr_vals[j]), 1e-9)
            live_pools = []
            for pi, pool in enumerate(pools):
                result = results[pi]
                if pool.available_at > ts:
                    continue
                if (getattr(result, "touched_at", None) is not None
                        and result.touched_at <= ts):
                    continue
                if (getattr(result, "broken_at", None) is not None
                        and result.broken_at <= ts):
                    continue
                live_pools.append(pool)
            try:
                state = sf.features_at(j, active_pools=live_pools)
            except Exception:
                continue
            session = "unknown"
            try:
                from liqpool.regime import nse_session
                session = nse_session(pd.Timestamp(ts))
            except Exception:
                pass
            if self.allowed_sessions and session not in self.allowed_sessions:
                continue
            p_up = None
            if direction_model is not None and self.use_direction_score:
                try:
                    p_up = float(direction_model.predict_state(state))
                except Exception:
                    p_up = None

            for pi, pool in enumerate(pools):
                if pi in emitted:
                    continue
                if pool.available_at > ts:
                    continue
                result = results[pi]
                if (getattr(result, "touched_at", None) is not None
                        and result.touched_at <= ts):
                    continue
                if (getattr(result, "broken_at", None) is not None
                        and result.broken_at <= ts):
                    continue

                if pool.price_low > close_T:
                    dist_atr = (pool.price_low - close_T) / a_T
                    pool_side = "above"
                    trade_side = "long"
                    journey_target = (pool.price_low - close_T) / a_T
                    if self.require_opening_breakout and opening_high is not None:
                        if not np.isfinite(opening_high.iloc[j]) or close_T < float(opening_high.iloc[j]):
                            continue
                elif pool.price_high < close_T:
                    dist_atr = (close_T - pool.price_high) / a_T
                    pool_side = "below"
                    trade_side = "short"
                    journey_target = (close_T - pool.price_high) / a_T
                    if self.require_opening_breakout and opening_low is not None:
                        if not np.isfinite(opening_low.iloc[j]) or close_T > float(opening_low.iloc[j]):
                            continue
                else:
                    continue                       # already inside the zone
                if dist_atr < self.DEFAULT_MIN_DIST_ATR:
                    continue                       # too close — no journey to ride
                if dist_atr > self.max_dist_atr:
                    continue
                if self.distance_band is not None:
                    lo, hi = self.distance_band
                    if not (lo <= dist_atr < hi):
                        continue

                try:
                    p_touch_by_h = {}
                    for h, pm in prox.items():
                        if h != primary_h and not self.use_multi_horizon:
                            continue
                        p_touch_by_h[int(h)] = float(pm.predict_one(
                            pool, dist_atr, pool_side, state,
                            float(q_preds[pi]), atr_val=a_T,
                        ))
                    p_touch = float(p_touch_by_h.get(primary_h, model.predict_one(
                        pool, dist_atr, pool_side, state,
                        float(q_preds[pi]), atr_val=a_T,
                    )))
                except Exception:
                    continue
                if not np.isfinite(p_touch) or p_touch < self.min_p_touch:
                    continue
                p_direction_to_pool = None
                if p_up is not None:
                    p_direction_to_pool = float(p_up if trade_side == "long" else 1.0 - p_up)
                score = p_touch
                score_state: Dict[str, Any] = {
                    "score": float(score),
                    "score_proximity": float(p_touch),
                    "score_direction": float(p_direction_to_pool if p_direction_to_pool is not None else 0.5),
                    "score_distance": float(self._distance_score(dist_atr)),
                    "score_sector": 0.5,
                    "score_vol": 0.5,
                    "sector_rotation_tag": "not_used",
                    "vol_regime_tag": "not_used",
                }
                if self.use_soft_score:
                    score, score_state = self._soft_score(
                        p_touch=p_touch,
                        p_touch_by_h=p_touch_by_h,
                        p_direction_to_pool=p_direction_to_pool,
                        dist_atr=float(dist_atr),
                        sector=sector,
                        trade_side=trade_side,
                        state=state,
                        report=report,
                        asof_ts=pd.Timestamp(ts),
                    )
                    threshold = self.min_score if self.min_score is not None else self.DEFAULT_MIN_SCORE
                    if score < threshold:
                        continue

                target_atr = min(float(journey_target), self.max_target_atr)
                if target_atr <= 0:
                    continue
                signals.append(AlphaSignal(
                    alpha_name=self._name, symbol=symbol,
                    decision_at=ts, decision_idx=j,
                    side=trade_side,
                    entry_reference=close_T,
                    stop_atr=self.stop_atr,
                    target_atr=target_atr,
                    horizon_bars=int(primary_h),
                    confidence=float(np.clip(score, 0.0, 1.0)),
                    state={
                        "pool_idx": int(pi),
                        "pool_score": float(pool.score),
                        "pool_mid": float(pool.mid),
                        "q_pred": float(q_preds[pi]),
                        "factor": _headline_for_pool(pool),
                        "tf_count": int(len(set(pool.tfs))),
                        "p_touch": float(p_touch),
                        "p_touch_by_horizon": dict(sorted(p_touch_by_h.items())),
                        "p_direction_to_pool": (
                            float(p_direction_to_pool)
                            if p_direction_to_pool is not None else None
                        ),
                        "dist_atr_at_decision": float(dist_atr),
                        "side_to_pool": pool_side,
                        "proximity_horizon": int(primary_h),
                        "session": session,
                        **score_state,
                    },
                ))
                emitted.add(pi)
        return signals

    def regime_tags(self, signal, df_base, atr_series):
        tags = super().regime_tags(signal, df_base, atr_series)
        tags["factor"] = str(signal.state.get("factor", "OTHER"))
        tags["tf_bucket"] = _tf_bucket(int(signal.state.get("tf_count", 0)))
        p = float(signal.state.get("p_touch", 0.5))
        tags["p_touch_bucket"] = (
            "p_high" if p >= 0.80 else
            "p_moderate" if p >= 0.65 else "p_marginal"
        )
        d = float(signal.state.get("dist_atr_at_decision", 0.0))
        tags["dist_bucket"] = (
            "near_0_1_atr" if d < 1.0 else
            "mid_1_3_atr" if d < 3.0 else "far_3_plus_atr"
        )
        tags["sector_rotation"] = str(signal.state.get("sector_rotation_tag", "not_used"))
        tags["vol_regime"] = str(signal.state.get("vol_regime_tag", "not_used"))
        score = float(signal.state.get("score", signal.confidence))
        tags["score_bucket"] = (
            "score_high" if score >= 0.80 else
            "score_mid" if score >= 0.65 else "score_low"
        )
        return tags


class DistanceBandJourneyAlpha(ProximityFilteredPoolAlpha):
    """Pre-touch journey alpha restricted to a specific distance band."""

    def __init__(self, distance_band: Tuple[float, float] = (5.0, 8.0),
                 name: str = "distance_5_8_journey",
                 **kwargs: Any) -> None:
        super().__init__(
            name=name,
            distance_band=distance_band,
            max_dist_atr=max(float(distance_band[1]), kwargs.pop("max_dist_atr", 8.0)),
            use_soft_score=True,
            use_multi_horizon=True,
            min_p_touch=kwargs.pop("min_p_touch", 0.50),
            min_score=kwargs.pop("min_score", 0.58),
            **kwargs,
        )


class ProximityDirectionSoftAlpha(ProximityFilteredPoolAlpha):
    """Pre-touch journey alpha using direction as a soft score component."""

    def __init__(self, name: str = "proximity_direction_soft",
                 **kwargs: Any) -> None:
        super().__init__(
            name=name,
            use_soft_score=True,
            use_direction_score=True,
            use_multi_horizon=True,
            min_p_touch=kwargs.pop("min_p_touch", 0.45),
            min_score=kwargs.pop("min_score", 0.60),
            max_dist_atr=kwargs.pop("max_dist_atr", 10.0),
            **kwargs,
        )


class OpeningRangeToPoolAlpha(ProximityFilteredPoolAlpha):
    """Pre-touch journey alpha after opening-range break toward a pool."""

    def __init__(self, name: str = "opening_range_to_pool",
                 **kwargs: Any) -> None:
        super().__init__(
            name=name,
            allowed_sessions=kwargs.pop("allowed_sessions", ("morning", "midday")),
            require_opening_breakout=True,
            use_soft_score=True,
            use_direction_score=True,
            use_multi_horizon=True,
            min_p_touch=kwargs.pop("min_p_touch", 0.45),
            min_score=kwargs.pop("min_score", 0.62),
            max_dist_atr=kwargs.pop("max_dist_atr", 8.0),
            sample_every=kwargs.pop("sample_every", 6),
            **kwargs,
        )


class SectorRotationJourneyAlpha(ProximityFilteredPoolAlpha):
    """Pre-touch journey alpha with sector rotation and volatility context."""

    def __init__(self, name: str = "sector_rotation_journey",
                 **kwargs: Any) -> None:
        super().__init__(
            name=name,
            use_soft_score=True,
            use_direction_score=True,
            use_sector_rotation=True,
            use_vol_regime=True,
            use_multi_horizon=True,
            min_p_touch=kwargs.pop("min_p_touch", 0.45),
            min_score=kwargs.pop("min_score", 0.62),
            max_dist_atr=kwargs.pop("max_dist_atr", 10.0),
            **kwargs,
        )
