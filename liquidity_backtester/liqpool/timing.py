"""Track 3: direction and timing/proximity layer with multi-horizon support.

Two LightGBM models trained on time-series snapshots:

  DirectionModel:   P(up | state) over a single primary horizon (default 78 bars = 1 NSE session).
                    Target = (max_up_atr >= max_dn_atr) over horizon.

  ProximityModel:   P(pool touched within H bars | state, pool). Walkforward trains THREE of
                    these at H = 78 (1 day), 156 (2 days), 312 (4 days). Each next-pool entry
                    in the run output shows all three so the user can see "when".

A Snapshot stores enough information (cumulative future max-high / min-low arrays and per-pool
bars-to-touch) to derive labels for ANY horizon <= the max_horizon used at generation time.
That lets us train multiple horizon-specific models from one snapshot pass.

All snapshot generation is causal — features at T only use bars with timestamp <= T, and labels
are derived from bars in (T, T+H].
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from .pools import Pool
from .tester import PoolResult
from .regime import compute_regime_series, nse_session, SESSION_LABELS
from .indicators import atr


def distance_bucket(distance_atr: float) -> str:
    if distance_atr < 1.0:
        return "0-1 ATR"
    if distance_atr < 3.0:
        return "1-3 ATR"
    if distance_atr < 5.0:
        return "3-5 ATR"
    if distance_atr < 10.0:
        return "5-10 ATR"
    return "10+ ATR"


def _bucket_binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    pred = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    actual = np.asarray(y, dtype=int)
    auc = float(roc_auc_score(actual, pred)) if len(set(actual)) > 1 else None
    return {
        "n": int(len(actual)),
        "base_rate": float(actual.mean()) if len(actual) else 0.0,
        "auc": auc,
        "brier": float(brier_score_loss(actual, pred)) if len(actual) else 0.0,
        "logloss": float(log_loss(actual, pred)) if len(actual) else 0.0,
        "calibration_error": float(pred.mean() - actual.mean()) if len(actual) else 0.0,
    }


# ---------------------------------------------------------------------------
# State features
# ---------------------------------------------------------------------------

_STATE_NUMERIC = [
    "ret_1", "ret_6", "ret_24", "ret_78",
    "mom_12_atr", "zscore_close_50", "range_6_atr",
    "adx_14", "vol_ratio",
    "n_above", "n_below",
    "nearest_above_atr", "nearest_below_atr",
    "pull_above", "pull_below", "pull_ratio",
    "minutes_since_session_open",
]
_STATE_SESSION = [f"st_session_{s}" for s in SESSION_LABELS]
STATE_FEATURE_NAMES = _STATE_NUMERIC + _STATE_SESSION


class StateFeaturizer:
    """Pre-computes per-bar series; provides cheap snapshot-time feature lookup."""

    def __init__(self, df_base: pd.DataFrame):
        self.df = df_base
        self.atr_14 = atr(df_base, 14).bfill()
        self.atr_60 = atr(df_base, 60).bfill()
        self.regime = compute_regime_series(df_base)
        self.close = df_base["close"].values
        self.high = df_base["high"].values
        self.low = df_base["low"].values
        self.idx = df_base.index
        self.roll_mean_50 = df_base["close"].rolling(50, min_periods=10).mean().bfill().values
        self.roll_std_50 = df_base["close"].rolling(50, min_periods=10).std().bfill().values

    def features_at(self, j: int, active_pools: List[Pool]) -> Dict[str, float]:
        n = len(self.idx)
        if j < 1 or j >= n:
            return {k: 0.0 for k in STATE_FEATURE_NAMES}

        c = self.close
        ts = self.idx[j]
        a = max(float(self.atr_14.iloc[j]), 1e-9)

        def log_ret(k):
            i0 = max(0, j - k)
            if c[i0] <= 0:
                return 0.0
            return float(np.log(c[j] / c[i0]))

        # Momentum over last 12 bars
        i0 = max(0, j - 11)
        oa = (c[i0:j + 1] - c[max(0, i0 - 1):j])[:j - i0 + 1]
        mom_12 = float(np.sum(oa) / a) if len(oa) else 0.0

        mean50 = self.roll_mean_50[j]
        std50 = self.roll_std_50[j]
        z = float((c[j] - mean50) / std50) if std50 > 0 else 0.0

        i0 = max(0, j - 5)
        rng6 = float((self.high[i0:j + 1].max() - self.low[i0:j + 1].min()) / a)

        adx_14 = float(self.regime["adx_14"].iloc[j])
        vol_r = float(self.regime["vol_ratio"].iloc[j])
        session = nse_session(ts)
        ist = ts + pd.Timedelta(hours=5, minutes=30)
        ist_min = ist.hour * 60 + ist.minute
        m_since_open = max(0, ist_min - (9 * 60 + 15))

        close_T = c[j]
        n_above = n_below = 0
        nearest_above_d = float("inf")
        nearest_below_d = float("inf")
        pull_above = pull_below = 0.0
        for p in active_pools:
            if p.price_low > close_T:
                d = (p.mid - close_T) / a
                n_above += 1
                nearest_above_d = min(nearest_above_d, d)
                pull_above += 1.0 / max(d, 1.0)
            elif p.price_high < close_T:
                d = (close_T - p.mid) / a
                n_below += 1
                nearest_below_d = min(nearest_below_d, d)
                pull_below += 1.0 / max(d, 1.0)
        nearest_above_d = nearest_above_d if nearest_above_d != float("inf") else 100.0
        nearest_below_d = nearest_below_d if nearest_below_d != float("inf") else 100.0
        pull_ratio = (pull_above - pull_below) / max(pull_above + pull_below, 1e-9)

        feats: Dict[str, float] = {
            "ret_1": log_ret(1),
            "ret_6": log_ret(6),
            "ret_24": log_ret(24),
            "ret_78": log_ret(78),
            "mom_12_atr": mom_12,
            "zscore_close_50": z,
            "range_6_atr": rng6,
            "adx_14": adx_14,
            "vol_ratio": vol_r,
            "n_above": float(n_above),
            "n_below": float(n_below),
            "nearest_above_atr": float(nearest_above_d),
            "nearest_below_atr": float(nearest_below_d),
            "pull_above": float(pull_above),
            "pull_below": float(pull_below),
            "pull_ratio": float(pull_ratio),
            "minutes_since_session_open": float(m_since_open),
        }
        for s in SESSION_LABELS:
            feats[f"st_session_{s}"] = 1.0 if session == s else 0.0
        return feats


# ---------------------------------------------------------------------------
# Multi-horizon Snapshot
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    bar_idx: int
    ts: pd.Timestamp
    close: float
    atr_val: float
    state: Dict[str, float]
    # Cumulative running max-high / min-low over future bars (T+1 .. T+n_future_bars).
    # future_max_high[i-1] = max(high[j+1..j+i]) — i.e. max-high seen so far i bars into the future.
    future_max_high: List[float] = field(default_factory=list)
    future_min_low: List[float] = field(default_factory=list)
    # Per-pool: (pool_idx, bars_to_touch_or_None, distance_atr_at_snapshot, "above"/"below")
    pool_touches: List[Tuple[int, Optional[int], float, str]] = field(default_factory=list)
    # How many future bars we actually observed (could be < max_horizon near end of data).
    n_future_bars: int = 0

    def direction_label(self, horizon: int) -> Optional[int]:
        """1 if up-excursion exceeds down-excursion over `horizon` future bars, 0 if down wins.
        None if not enough data OR the move is an exact tie (e.g. a flat window) — labelling a
        tie as 'up' (the old behaviour) injected a systematic upward bias from flat snapshots."""
        h = min(horizon, self.n_future_bars)
        if h < 5:
            return None
        max_h = self.future_max_high[h - 1]
        min_l = self.future_min_low[h - 1]
        max_up = max_h - self.close
        max_dn = self.close - min_l
        if max_up == max_dn:
            return None
        return 1 if max_up > max_dn else 0

    def max_up_atr(self, horizon: int) -> float:
        h = min(horizon, self.n_future_bars)
        if h == 0:
            return 0.0
        return (self.future_max_high[h - 1] - self.close) / max(self.atr_val, 1e-9)

    def max_dn_atr(self, horizon: int) -> float:
        h = min(horizon, self.n_future_bars)
        if h == 0:
            return 0.0
        return (self.close - self.future_min_low[h - 1]) / max(self.atr_val, 1e-9)

    def pool_touch_labels(self, horizon: int) -> List[Tuple[int, int, float, str]]:
        """Returns (pool_idx, touched_within_horizon, distance_atr, side) for each pool that was
        active at the snapshot. Caller is responsible for filtering by self.n_future_bars >=
        horizon if it needs a definitive 0-label."""
        return [(pi,
                 1 if (bt is not None and bt <= horizon) else 0,
                 d, s)
                for (pi, bt, d, s) in self.pool_touches]


def generate_snapshots(df_base: pd.DataFrame, pools: List[Pool], results: List[PoolResult],
                       featurizer: StateFeaturizer,
                       window_start: pd.Timestamp, window_end: pd.Timestamp,
                       sample_every: int, max_horizon: int,
                       clip_future_to_window: bool = False) -> List[Snapshot]:
    """Build snapshots at every `sample_every` bar in [window_start, window_end].

    `window_end` bounds WHERE we sample snapshots (the last decision bar). By default a
    snapshot's future trajectory is allowed to extend past `window_end`, up to `max_horizon`
    bars or the data end. This is the correct behaviour for OOS *evaluation* (the future
    outcome is genuinely observable after the decision) and for single-bar causality probes.

    Pass `clip_future_to_window=True` when generating TRAINING snapshots in a walk-forward:
    it caps each snapshot's future trajectory at `window_end` so a train-window snapshot near
    the boundary cannot derive its label from bars that fall inside the OOS test window
    (which would leak test-period price action into training and inflate OOS metrics). Models
    filter `n_future_bars >= horizon` at fit time, so boundary snapshots are purged rather
    than trained on truncated labels.
    """
    idx = df_base.index
    n = len(idx)
    h_arr = df_base["high"].values
    l_arr = df_base["low"].values
    c_arr = df_base["close"].values

    j_start = int(np.searchsorted(idx.values, np.datetime64(window_start), side="left"))
    we_pos = int(np.searchsorted(idx.values, np.datetime64(window_end), side="right")) - 1
    j_end = min(we_pos, n - 5)

    snaps: List[Snapshot] = []
    for j in range(max(j_start, 80), j_end + 1, sample_every):
        T = idx[j]
        close_T = c_arr[j]
        a_T = max(float(featurizer.atr_14.iloc[j]), 1e-9)

        n_future = min(max_horizon, n - 1 - j)
        if clip_future_to_window:
            n_future = min(n_future, we_pos - j)
        if n_future < 5:
            continue

        # Cumulative running max-high / min-low for the future window
        f_max_high: List[float] = []
        f_min_low: List[float] = []
        cur_max = h_arr[j + 1]
        cur_min = l_arr[j + 1]
        f_max_high.append(cur_max)
        f_min_low.append(cur_min)
        for i in range(2, n_future + 1):
            cur_max = max(cur_max, float(h_arr[j + i]))
            cur_min = min(cur_min, float(l_arr[j + i]))
            f_max_high.append(cur_max)
            f_min_low.append(cur_min)

        # Active pools at T (available_at <= T, not yet touched/broken)
        active_with_idx: List[Tuple[int, Pool]] = []
        for pi, p in enumerate(pools):
            if p.available_at > T:
                continue
            r = results[pi]
            if r.touched_at is not None and r.touched_at <= T:
                continue
            if r.broken_at is not None and r.broken_at <= T:
                continue
            active_with_idx.append((pi, p))
        active_pools = [p for _, p in active_with_idx]
        state = featurizer.features_at(j, active_pools)

        # Per-pool bars-to-touch (within max_horizon)
        pool_touches: List[Tuple[int, Optional[int], float, str]] = []
        for pi, p in active_with_idx:
            ta = results[pi].touched_at
            bars_to_touch: Optional[int] = None
            if ta is not None and ta > T:
                ta_idx = int(idx.searchsorted(ta, side="left"))
                if j < ta_idx <= j + n_future:
                    bars_to_touch = ta_idx - j
            if p.price_low > close_T:
                dist = (p.mid - close_T) / a_T
                side = "above"
            elif p.price_high < close_T:
                dist = (close_T - p.mid) / a_T
                side = "below"
            else:
                continue   # already inside the zone
            pool_touches.append((pi, bars_to_touch, float(dist), side))

        snaps.append(Snapshot(
            bar_idx=j, ts=T, close=float(close_T), atr_val=a_T,
            state=state,
            future_max_high=f_max_high, future_min_low=f_min_low,
            pool_touches=pool_touches, n_future_bars=int(n_future),
        ))
    return snaps


# ---------------------------------------------------------------------------
# Direction model — single horizon
# ---------------------------------------------------------------------------

@dataclass
class DirectionModel:
    horizon: int = 78                                # 1 NSE trading session
    feature_names: List[str] = field(default_factory=list)
    train_n: int = 0
    val_n: int = 0
    val_brier: float = 0.0
    val_logloss: float = 0.0
    val_auc: float = 0.0
    base_rate: float = 0.0

    def fit(self, snapshots: List[Snapshot], val_frac: float = 0.25, seed: int = 19
            ) -> "DirectionModel":
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        usable = [(s.state, s.direction_label(self.horizon)) for s in snapshots
                  if s.n_future_bars >= self.horizon and s.direction_label(self.horizon) is not None]
        if len(usable) < 30:
            raise ValueError(f"DirectionModel needs >= 30 valid snapshots at horizon "
                              f"{self.horizon}, got {len(usable)}")
        X_df = pd.DataFrame([u[0] for u in usable], columns=STATE_FEATURE_NAMES).fillna(0.0)
        y = np.array([u[1] for u in usable], dtype=int)
        self.feature_names = list(X_df.columns)
        self.base_rate = float(y.mean())

        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        n_vp = max(1, int(round(len(pos) * val_frac)))
        n_vn = max(1, int(round(len(neg) * val_frac)))
        val_idx = np.concatenate([pos[:n_vp], neg[:n_vn]])
        tr_idx = np.concatenate([pos[n_vp:], neg[n_vn:]])
        rng.shuffle(val_idx); rng.shuffle(tr_idx)

        X_tr, y_tr = X_df.iloc[tr_idx].values, y[tr_idx]
        X_v, y_v = X_df.iloc[val_idx].values, y[val_idx]

        params = dict(objective="binary", metric="binary_logloss",
                       learning_rate=0.05, num_leaves=15, max_depth=4,
                       min_data_in_leaf=8, feature_fraction=0.85,
                       bagging_fraction=0.85, bagging_freq=5,
                       lambda_l2=2.0, verbose=-1, seed=seed)
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        dval = lgb.Dataset(X_v, label=y_v, reference=dtr, feature_name=self.feature_names)
        self._gbm = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dval],
                               valid_names=["val"],
                               callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                                          lgb.log_evaluation(0)])
        val_raw = self._gbm.predict(X_v, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(val_raw, y_v)
        val_cal = self._iso.transform(val_raw)
        self.train_n = int(len(tr_idx)); self.val_n = int(len(val_idx))
        self.val_brier = float(brier_score_loss(y_v, val_cal))
        self.val_logloss = float(log_loss(y_v, np.clip(val_cal, 1e-6, 1 - 1e-6)))
        self.val_auc = float(roc_auc_score(y_v, val_cal)) if len(set(y_v)) > 1 else 0.0
        return self

    def fit_frame(self, frame: pd.DataFrame, val_frac: float = 0.25,
                  seed: int = 19) -> "DirectionModel":
        """Fit from persisted direction feature rows instead of in-memory Snapshot objects."""
        if frame is None or frame.empty:
            raise ValueError("DirectionModel needs non-empty direction feature frame")
        if "direction_label" not in frame.columns:
            raise ValueError("direction feature frame missing direction_label")
        X_df = frame.reindex(columns=STATE_FEATURE_NAMES).fillna(0.0)
        y = frame["direction_label"].astype(int).to_numpy()
        if len(y) < 30:
            raise ValueError(f"DirectionModel needs >= 30 valid rows, got {len(y)}")
        if len(np.unique(y)) < 2:
            raise ValueError("DirectionModel needs both classes")
        self.feature_names = list(X_df.columns)
        self.base_rate = float(y.mean())

        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        n_vp = max(1, int(round(len(pos) * val_frac)))
        n_vn = max(1, int(round(len(neg) * val_frac)))
        val_idx = np.concatenate([pos[:n_vp], neg[:n_vn]])
        tr_idx = np.concatenate([pos[n_vp:], neg[n_vn:]])
        rng.shuffle(val_idx); rng.shuffle(tr_idx)

        X_tr, y_tr = X_df.iloc[tr_idx].values, y[tr_idx]
        X_v, y_v = X_df.iloc[val_idx].values, y[val_idx]
        params = dict(objective="binary", metric="binary_logloss",
                       learning_rate=0.05, num_leaves=15, max_depth=4,
                       min_data_in_leaf=8, feature_fraction=0.85,
                       bagging_fraction=0.85, bagging_freq=5,
                       lambda_l2=2.0, verbose=-1, seed=seed)
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        dval = lgb.Dataset(X_v, label=y_v, reference=dtr, feature_name=self.feature_names)
        self._gbm = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dval],
                               valid_names=["val"],
                               callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                                          lgb.log_evaluation(0)])
        val_raw = self._gbm.predict(X_v, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(val_raw, y_v)
        val_cal = self._iso.transform(val_raw)
        self.train_n = int(len(tr_idx)); self.val_n = int(len(val_idx))
        self.val_brier = float(brier_score_loss(y_v, val_cal))
        self.val_logloss = float(log_loss(y_v, np.clip(val_cal, 1e-6, 1 - 1e-6)))
        self.val_auc = float(roc_auc_score(y_v, val_cal)) if len(set(y_v)) > 1 else 0.0
        return self

    def predict_state(self, state: Dict[str, float]) -> float:
        X = pd.DataFrame([state], columns=self.feature_names).fillna(0.0).values
        raw = self._gbm.predict(X, num_iteration=self._gbm.best_iteration)
        cal = float(self._iso.transform(raw)[0])
        # Clip to [0.05, 0.95]. Isotonic on small validation sets pushes some predictions to
        # exactly 0 or 1, producing nonsensical "P(up) = 100%" outputs. Clipping caps the
        # confidence at a realistic level — top-quartile-confidence accuracy on this model
        # is ~79%, so claiming 100% certainty was always overconfident.
        return float(np.clip(cal, 0.05, 0.95))

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.feature_names).fillna(0.0).values
        raw = self._gbm.predict(X, num_iteration=self._gbm.best_iteration)
        return np.clip(self._iso.transform(raw), 0.05, 0.95)

    def feature_importance(self, top_k: int = 10) -> List[Tuple[str, int]]:
        if not hasattr(self, "_gbm"):
            return []
        gains = self._gbm.feature_importance(importance_type="gain")
        return sorted(zip(self.feature_names, gains), key=lambda x: -x[1])[:top_k]

    def explain_state(self, state: Dict[str, float], top_k: int = 5) -> Dict:
        """Per-prediction feature contribution explanation. See PoolRespectModel.explain_prediction."""
        if not hasattr(self, "_gbm"):
            return {}
        X = pd.DataFrame([state], columns=self.feature_names).fillna(0.0).values
        contribs = self._gbm.predict(X, pred_contrib=True,
                                       num_iteration=self._gbm.best_iteration)[0]
        base = float(contribs[-1])
        feat = list(zip(self.feature_names, [float(v) for v in contribs[:-1]]))
        feat.sort(key=lambda kv: -abs(kv[1]))
        return {
            "base_logit": base,
            "raw_prediction_logit": float(contribs.sum()),
            "top_features": feat[:top_k],
        }


# ---------------------------------------------------------------------------
# Proximity model — horizon-specific
# ---------------------------------------------------------------------------

# pool_quality (P_respect) is intentionally NOT a proximity model feature: it is an
# in-sample prediction for train pools (the Q model trained on them), which leaks label
# information into P_touch training/eval. It is also conceptually weak — whether price will
# TOUCH a pool shouldn't depend on its post-touch respect quality, and distance dominates
# importance anyway. The column is still emitted by _pool_features_for_snapshot for the
# Q×T joint sanity-check diagnostic, just not consumed as a model input.
_POOL_FEATURE_NAMES = ["distance_atr", "side_above",
                       "pool_score", "pool_width_atr",
                       "pool_n_tfs", "pool_n_contributors",
                       "pool_age_at_avail_bars"]


def _pool_features_for_snapshot(pool: Pool, dist_atr: float, side: str,
                                 quality_pred: float, base_period_seconds: float = 300.0,
                                 atr_val: float = 1.0) -> Dict[str, float]:
    earliest = min((c.ts for c in pool.contributors), default=pool.formed_at)
    age_bars = (pool.available_at - earliest).total_seconds() / base_period_seconds \
               if pool.available_at >= earliest else 0.0
    return {
        "distance_atr": float(dist_atr),
        "side_above": 1.0 if side == "above" else 0.0,
        "pool_quality": float(quality_pred),
        "pool_score": float(pool.score),
        # ATR-normalised so width is comparable across assets of different price levels
        # (a raw price width confounds the feature with the stock's price scale).
        "pool_width_atr": float(pool.width / max(atr_val, 1e-9)),
        "pool_n_tfs": float(len(set(pool.tfs))),
        "pool_n_contributors": float(len(pool.contributors)),
        "pool_age_at_avail_bars": float(age_bars),
    }


@dataclass
class ProximityModel:
    horizon: int = 78
    base_period_seconds: float = 300.0      # inferred and overridden by walkforward at fit time
    feature_names: List[str] = field(default_factory=list)
    train_n: int = 0
    val_n: int = 0
    val_brier: float = 0.0
    val_logloss: float = 0.0
    val_auc: float = 0.0
    val_decile_lift: float = 0.0
    base_rate: float = 0.0

    def fit(self, snapshots: List[Snapshot], pools: List[Pool],
            quality_preds: np.ndarray,
            val_frac: float = 0.25, seed: int = 21) -> "ProximityModel":
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        rows, labels = [], []
        for s in snapshots:
            if s.n_future_bars < self.horizon:
                continue
            for (pi, touched, dist, side) in s.pool_touch_labels(self.horizon):
                pool = pools[pi]
                pool_feat = _pool_features_for_snapshot(
                    pool, dist, side, float(quality_preds[pi]),
                    base_period_seconds=self.base_period_seconds, atr_val=s.atr_val,
                )
                rows.append({**s.state, **pool_feat})
                labels.append(touched)
        if len(rows) < 100:
            raise ValueError(f"ProximityModel at horizon {self.horizon} needs >= 100 rows, "
                              f"got {len(rows)}")

        feature_cols = STATE_FEATURE_NAMES + _POOL_FEATURE_NAMES
        X_df = pd.DataFrame(rows, columns=feature_cols).fillna(0.0)
        y = np.array(labels, dtype=int)
        self.feature_names = list(X_df.columns)
        self.base_rate = float(y.mean())

        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        n_vp = max(1, int(round(len(pos) * val_frac)))
        n_vn = max(1, int(round(len(neg) * val_frac)))
        val_idx = np.concatenate([pos[:n_vp], neg[:n_vn]])
        tr_idx = np.concatenate([pos[n_vp:], neg[n_vn:]])
        rng.shuffle(val_idx); rng.shuffle(tr_idx)
        X_tr, y_tr = X_df.iloc[tr_idx].values, y[tr_idx]
        X_v, y_v = X_df.iloc[val_idx].values, y[val_idx]

        params = dict(objective="binary", metric="binary_logloss",
                       learning_rate=0.05, num_leaves=20, max_depth=5,
                       min_data_in_leaf=20, feature_fraction=0.85,
                       bagging_fraction=0.85, bagging_freq=5,
                       lambda_l2=2.0, verbose=-1, seed=seed)
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        dval = lgb.Dataset(X_v, label=y_v, reference=dtr, feature_name=self.feature_names)
        self._gbm = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dval],
                               valid_names=["val"],
                               callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False),
                                          lgb.log_evaluation(0)])
        val_raw = self._gbm.predict(X_v, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(val_raw, y_v)
        val_cal = self._iso.transform(val_raw)
        self.train_n = int(len(tr_idx)); self.val_n = int(len(val_idx))
        self.val_brier = float(brier_score_loss(y_v, val_cal))
        self.val_logloss = float(log_loss(y_v, np.clip(val_cal, 1e-6, 1 - 1e-6)))
        self.val_auc = float(roc_auc_score(y_v, val_cal)) if len(set(y_v)) > 1 else 0.0
        # Decile lift on validation
        if len(val_cal) >= 20:
            order = np.argsort(val_cal)
            nv = len(val_cal)
            bot = order[: nv // 10]; top = order[-nv // 10:]
            br = float(y_v[bot].mean()) if len(bot) else 0.0
            tr = float(y_v[top].mean()) if len(top) else 0.0
            self.val_decile_lift = (tr / br) if br > 0 else float("inf")
        return self

    def fit_frame(self, frame: pd.DataFrame, val_frac: float = 0.25,
                  seed: int = 21) -> "ProximityModel":
        """Fit from persisted proximity rows.

        This is the Mac-safe path used by the feature store: each row already contains state
        features, pool features, and the horizon-specific touch label.
        """
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        if frame is None or frame.empty:
            raise ValueError(f"ProximityModel at horizon {self.horizon} needs non-empty frame")
        if "touch_label" not in frame.columns:
            raise ValueError("proximity feature frame missing touch_label")
        feature_cols = STATE_FEATURE_NAMES + _POOL_FEATURE_NAMES
        X_df = frame.reindex(columns=feature_cols).fillna(0.0)
        y = frame["touch_label"].astype(int).to_numpy()
        if len(y) < 100:
            raise ValueError(f"ProximityModel at horizon {self.horizon} needs >= 100 rows, "
                              f"got {len(y)}")
        if len(np.unique(y)) < 2:
            raise ValueError(f"ProximityModel at horizon {self.horizon} needs both classes")
        self.feature_names = list(X_df.columns)
        self.base_rate = float(y.mean())

        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        n_vp = max(1, int(round(len(pos) * val_frac)))
        n_vn = max(1, int(round(len(neg) * val_frac)))
        val_idx = np.concatenate([pos[:n_vp], neg[:n_vn]])
        tr_idx = np.concatenate([pos[n_vp:], neg[n_vn:]])
        rng.shuffle(val_idx); rng.shuffle(tr_idx)
        X_tr, y_tr = X_df.iloc[tr_idx].values, y[tr_idx]
        X_v, y_v = X_df.iloc[val_idx].values, y[val_idx]

        params = dict(objective="binary", metric="binary_logloss",
                       learning_rate=0.05, num_leaves=20, max_depth=5,
                       min_data_in_leaf=20, feature_fraction=0.85,
                       bagging_fraction=0.85, bagging_freq=5,
                       lambda_l2=2.0, verbose=-1, seed=seed)
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        dval = lgb.Dataset(X_v, label=y_v, reference=dtr, feature_name=self.feature_names)
        self._gbm = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dval],
                               valid_names=["val"],
                               callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False),
                                          lgb.log_evaluation(0)])
        val_raw = self._gbm.predict(X_v, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(val_raw, y_v)
        val_cal = self._iso.transform(val_raw)
        self.train_n = int(len(tr_idx)); self.val_n = int(len(val_idx))
        self.val_brier = float(brier_score_loss(y_v, val_cal))
        self.val_logloss = float(log_loss(y_v, np.clip(val_cal, 1e-6, 1 - 1e-6)))
        self.val_auc = float(roc_auc_score(y_v, val_cal)) if len(set(y_v)) > 1 else 0.0
        if len(val_cal) >= 20:
            order = np.argsort(val_cal)
            nv = len(val_cal)
            bot = order[: nv // 10]; top = order[-nv // 10:]
            br = float(y_v[bot].mean()) if len(bot) else 0.0
            tr = float(y_v[top].mean()) if len(top) else 0.0
            self.val_decile_lift = (tr / br) if br > 0 else float("inf")
        return self

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame.reindex(columns=self.feature_names).fillna(0.0).values
        raw = self._gbm.predict(X, num_iteration=self._gbm.best_iteration)
        return np.clip(self._iso.transform(raw), 0.0, 1.0)

    def predict_one(self, pool: Pool, dist_atr: float, side: str,
                    state: Dict[str, float], quality_pred: float,
                    atr_val: float = 1.0) -> float:
        pool_feat = _pool_features_for_snapshot(
            pool, dist_atr, side, quality_pred,
            base_period_seconds=self.base_period_seconds, atr_val=atr_val,
        )
        merged = {**state, **pool_feat}
        X = pd.DataFrame([merged], columns=self.feature_names).fillna(0.0).values
        raw = self._gbm.predict(X, num_iteration=self._gbm.best_iteration)
        return float(np.clip(self._iso.transform(raw)[0], 0.0, 1.0))

    def feature_importance(self, top_k: int = 10) -> List[Tuple[str, int]]:
        if not hasattr(self, "_gbm"):
            return []
        gains = self._gbm.feature_importance(importance_type="gain")
        return sorted(zip(self.feature_names, gains), key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------------------------
# Evaluation (multi-horizon proximity, plus joint score sanity check)
# ---------------------------------------------------------------------------

@dataclass
class HorizonProxStats:
    horizon: int
    n: int
    auc: float
    brier: float
    decile_lift: float
    base_rate: float
    logloss: float = 0.0
    distance_bucket_metrics: List[Dict] = field(default_factory=list)


@dataclass
class TimingReport:
    direction_horizon: int = 0
    direction_n: int = 0
    direction_brier: float = 0.0
    direction_logloss: float = 0.0
    direction_auc: float = 0.0
    direction_top_quartile_acc: float = 0.0

    proximity_per_horizon: List[HorizonProxStats] = field(default_factory=list)

    # Joint sanity check using primary (shortest) proximity horizon
    primary_prox_horizon: int = 0
    n_joint_evaluated: int = 0
    joint_top_quartile_respect: float = 0.0
    joint_bottom_quartile_respect: float = 0.0
    joint_lift: float = 0.0


def evaluate_timing(snapshots: List[Snapshot], pools: List[Pool], results: List[PoolResult],
                     direction_model: DirectionModel,
                     proximity_models: Dict[int, ProximityModel],
                     quality_preds: np.ndarray) -> TimingReport:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    rpt = TimingReport()

    # Direction OOS evaluation at the direction model's horizon
    if direction_model is not None and snapshots:
        usable = [s for s in snapshots
                  if s.n_future_bars >= direction_model.horizon
                  and s.direction_label(direction_model.horizon) is not None]
        if usable:
            X = pd.DataFrame([s.state for s in usable], columns=STATE_FEATURE_NAMES).fillna(0.0)
            y = np.array([s.direction_label(direction_model.horizon) for s in usable], dtype=int)
            p = direction_model.predict_batch(X)
            rpt.direction_horizon = direction_model.horizon
            rpt.direction_n = int(len(y))
            rpt.direction_brier = float(brier_score_loss(y, p))
            rpt.direction_logloss = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
            if len(set(y)) > 1:
                rpt.direction_auc = float(roc_auc_score(y, p))
            conf = np.abs(p - 0.5)
            if len(p):
                thr = float(np.quantile(conf, 0.75))
                m = conf >= thr
                if m.sum() > 0:
                    pred_dir = (p[m] >= 0.5).astype(int)
                    rpt.direction_top_quartile_acc = float((pred_dir == y[m]).mean())

    # Proximity OOS evaluation per horizon
    primary_horizon = None
    primary_p_touch = None
    primary_pi_array = None
    primary_touched_array = None
    for h in sorted(proximity_models.keys()):
        pm = proximity_models[h]
        rows, labels, pis, touched_lst, distances = [], [], [], [], []
        for s in snapshots:
            if s.n_future_bars < h:
                continue
            for (pi, touched, dist, side) in s.pool_touch_labels(h):
                pool = pools[pi]
                pf = _pool_features_for_snapshot(
                    pool, dist, side, float(quality_preds[pi]),
                    base_period_seconds=pm.base_period_seconds, atr_val=s.atr_val,
                )
                rows.append({**s.state, **pf})
                labels.append(touched)
                pis.append(pi)
                touched_lst.append(touched)
                distances.append(float(dist))
        if not rows:
            continue
        X = pd.DataFrame(rows, columns=pm.feature_names).fillna(0.0)
        y = np.array(labels, dtype=int)
        raw = pm._gbm.predict(X.values, num_iteration=pm._gbm.best_iteration)
        p = np.clip(pm._iso.transform(raw), 0.0, 1.0)
        if len(set(y)) > 1:
            auc = float(roc_auc_score(y, p))
        else:
            auc = 0.0
        br = float(brier_score_loss(y, p))
        ll = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
        bucket_rows = []
        dist_arr = np.asarray(distances, dtype=float)
        for bucket in ("0-1 ATR", "1-3 ATR", "3-5 ATR", "5-10 ATR", "10+ ATR"):
            mask = np.array([distance_bucket(d) == bucket for d in dist_arr], dtype=bool)
            if not mask.any():
                continue
            row = _bucket_binary_metrics(y[mask], p[mask])
            row["bucket"] = bucket
            bucket_rows.append(row)
        # Decile lift
        decile_lift = 0.0
        if len(p) >= 20:
            order = np.argsort(p)
            nv = len(p)
            bot = order[: nv // 10]; top = order[-nv // 10:]
            bot_rate = float(y[bot].mean()) if len(bot) else 0.0
            top_rate = float(y[top].mean()) if len(top) else 0.0
            decile_lift = (top_rate / bot_rate) if bot_rate > 0 else float("inf")
        rpt.proximity_per_horizon.append(HorizonProxStats(
            horizon=h, n=int(len(y)), auc=auc, brier=br,
            decile_lift=decile_lift, base_rate=float(y.mean()),
            logloss=ll, distance_bucket_metrics=bucket_rows,
        ))
        if primary_horizon is None:
            primary_horizon = h
            primary_p_touch = p
            primary_pi_array = pis
            primary_touched_array = touched_lst

    # Joint sanity check at the primary (shortest) proximity horizon
    if primary_horizon is not None:
        rpt.primary_prox_horizon = primary_horizon
        joint = []
        for k, (pi, touched) in enumerate(zip(primary_pi_array, primary_touched_array)):
            if touched != 1:
                continue
            r = results[pi]
            if r.is_respect:
                label = 1
            elif r.is_break:
                label = 0
            else:
                continue
            score = float(quality_preds[pi]) * float(primary_p_touch[k])
            joint.append((score, label))
        if len(joint) >= 20:
            joint.sort(key=lambda x: x[0])
            m = len(joint)
            bot = joint[: m // 4]
            top = joint[-m // 4:]
            rpt.joint_bottom_quartile_respect = float(np.mean([r for _, r in bot]))
            rpt.joint_top_quartile_respect = float(np.mean([r for _, r in top]))
            rpt.joint_lift = (rpt.joint_top_quartile_respect /
                              rpt.joint_bottom_quartile_respect
                              if rpt.joint_bottom_quartile_respect > 0 else float("inf"))
            rpt.n_joint_evaluated = m

    return rpt


def evaluate_timing_frames(direction_frame: pd.DataFrame,
                           proximity_frames: Dict[int, pd.DataFrame],
                           direction_model: Optional[DirectionModel],
                           proximity_models: Dict[int, ProximityModel]) -> TimingReport:
    """Evaluate timing models from persisted feature-store rows."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    rpt = TimingReport()
    if direction_model is not None and direction_frame is not None and not direction_frame.empty:
        if "direction_label" in direction_frame.columns:
            X = direction_frame.reindex(columns=direction_model.feature_names).fillna(0.0)
            y = direction_frame["direction_label"].astype(int).to_numpy()
            if len(y):
                p = direction_model.predict_batch(X)
                rpt.direction_horizon = direction_model.horizon
                rpt.direction_n = int(len(y))
                rpt.direction_brier = float(brier_score_loss(y, p))
                rpt.direction_logloss = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
                if len(set(y)) > 1:
                    rpt.direction_auc = float(roc_auc_score(y, p))
                conf = np.abs(p - 0.5)
                thr = float(np.quantile(conf, 0.75)) if len(conf) else 1.0
                m = conf >= thr
                if m.sum() > 0:
                    rpt.direction_top_quartile_acc = float(((p[m] >= 0.5).astype(int) == y[m]).mean())

    primary_horizon = None
    primary_scores = None
    primary_respect = None
    for h in sorted(proximity_models.keys()):
        frame = proximity_frames.get(h)
        if frame is None or frame.empty or "touch_label" not in frame.columns:
            continue
        pm = proximity_models[h]
        y = frame["touch_label"].astype(int).to_numpy()
        p = pm.predict_frame(frame)
        auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else 0.0
        bucket_rows = []
        dist_arr = frame["distance_atr"].astype(float).to_numpy() if "distance_atr" in frame else np.zeros(len(y))
        for bucket in ("0-1 ATR", "1-3 ATR", "3-5 ATR", "5-10 ATR", "10+ ATR"):
            mask = np.array([distance_bucket(d) == bucket for d in dist_arr], dtype=bool)
            if not mask.any():
                continue
            row = _bucket_binary_metrics(y[mask], p[mask])
            row["bucket"] = bucket
            bucket_rows.append(row)
        decile_lift = 0.0
        if len(p) >= 20:
            order = np.argsort(p)
            nv = len(p)
            bot = order[: nv // 10]; top = order[-nv // 10:]
            bot_rate = float(y[bot].mean()) if len(bot) else 0.0
            top_rate = float(y[top].mean()) if len(top) else 0.0
            decile_lift = (top_rate / bot_rate) if bot_rate > 0 else float("inf")
        rpt.proximity_per_horizon.append(HorizonProxStats(
            horizon=h,
            n=int(len(y)),
            auc=auc,
            brier=float(brier_score_loss(y, p)),
            decile_lift=decile_lift,
            base_rate=float(y.mean()),
            logloss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            distance_bucket_metrics=bucket_rows,
        ))
        if primary_horizon is None:
            primary_horizon = h
            q = frame["pool_quality"].astype(float).to_numpy() if "pool_quality" in frame else np.ones(len(p))
            primary_scores = q * p
            primary_respect = frame["respect_label"].to_numpy() if "respect_label" in frame else None

    if primary_horizon is not None:
        rpt.primary_prox_horizon = int(primary_horizon)
    if primary_scores is not None and primary_respect is not None:
        valid = ~pd.isna(primary_respect)
        if valid.sum() >= 20:
            items = sorted(zip(primary_scores[valid], primary_respect[valid]), key=lambda x: x[0])
            m = len(items)
            bot = items[: m // 4]
            top = items[-m // 4:]
            rpt.joint_bottom_quartile_respect = float(np.mean([r for _, r in bot]))
            rpt.joint_top_quartile_respect = float(np.mean([r for _, r in top]))
            rpt.joint_lift = (
                rpt.joint_top_quartile_respect / rpt.joint_bottom_quartile_respect
                if rpt.joint_bottom_quartile_respect > 0 else float("inf")
            )
            rpt.n_joint_evaluated = int(m)
    return rpt


def print_timing_report(rpt: TimingReport, file=None) -> None:
    print("\n=== Track 3: Direction + Timing Models (OOS) ===", file=file)
    print(f"\n[direction model]   horizon = {rpt.direction_horizon} bars  "
          f"(~{rpt.direction_horizon * 5 / 60:.1f}h on 5m base)", file=file)
    print(f"  OOS samples:                    {rpt.direction_n}", file=file)
    print(f"  Brier:                          {rpt.direction_brier:.4f}", file=file)
    print(f"  Log loss:                       {rpt.direction_logloss:.4f}", file=file)
    print(f"  AUC-ROC:                        {rpt.direction_auc:.3f}", file=file)
    print(f"  Top-quartile-confidence acc:    {rpt.direction_top_quartile_acc:.1%}  "
          f"(accuracy when model is most certain)", file=file)

    if rpt.proximity_per_horizon:
        print(f"\n[proximity models per horizon]   target = pool touched within H bars",
              file=file)
        print(f"  {'horizon':<8} {'~hours':<8} {'n_oos':>7} {'AUC':>6} {'Brier':>7} "
              f"{'base_rate':>10} {'decile_lift':>12}", file=file)
        for s in rpt.proximity_per_horizon:
            hours = s.horizon * 5 / 60.0
            lift_str = "inf" if s.decile_lift == float("inf") else f"{s.decile_lift:.1f}x"
            print(f"  {s.horizon:<8} {hours:<7.1f}h {s.n:>7} {s.auc:>6.3f} {s.brier:>7.4f} "
                  f"{s.base_rate:>9.1%} {lift_str:>12}", file=file)
            if s.distance_bucket_metrics:
                print(f"    {'bucket':<9} {'n':>6} {'base':>8} {'AUC':>6} {'Brier':>8} "
                      f"{'logloss':>8} {'cal_err':>8}", file=file)
                for row in s.distance_bucket_metrics:
                    auc_s = f"{row['auc']:.3f}" if row["auc"] is not None else "n/a"
                    print(f"    {row['bucket']:<9} {row['n']:>6} {row['base_rate']:>7.1%} "
                          f"{auc_s:>6} {row['brier']:>8.4f} {row['logloss']:>8.4f} "
                          f"{row['calibration_error']:>+7.1%}", file=file)

    print(f"\n[joint score sanity check]  Q × T_h{rpt.primary_prox_horizon} on touched pools",
          file=file)
    print(f"  n decisive:             {rpt.n_joint_evaluated}", file=file)
    print(f"  Top quartile respect:   {rpt.joint_top_quartile_respect:.1%}", file=file)
    print(f"  Bottom quartile:        {rpt.joint_bottom_quartile_respect:.1%}", file=file)
    print(f"  Joint lift:             {rpt.joint_lift:.2f}x", file=file)
    if rpt.joint_lift < 1.2:
        print(f"  → joint score does NOT rank touched pools well. Use T as filter "
              f"(tradeable today), Q for ranking among tradeable.", file=file)
