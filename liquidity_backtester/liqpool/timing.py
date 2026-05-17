"""Track 3: direction and timing/proximity layer.

Track 2 predicts P(respect | touched) — pool quality. It says nothing about WHICH pool will be
touched first or which DIRECTION price will move next. Track 3 closes those gaps with two
independent LightGBM models trained on time-series snapshots:

  DirectionModel:  at time T, predict P(up | state at T) over next horizon bars.
                   Target = (max_high - close) > (close - min_low) over the horizon.
                   Features = momentum, ADX/vol regime, VWAP deviation, session, aggregate pool
                              pull (Q × 1/distance summed over each side).

  ProximityModel:  at (time T, pool P), predict P(touched within horizon bars | state).
                   Target = pool's touched_at falls in (T, T+horizon].
                   Features = state features + per-pool features (distance, quality, age, TFs).

Joint trading score for a pool: Q × T (quality × touch-probability).
A pool that's likely to be touched AND likely to respect when touched is the actionable signal.

All snapshot generation is causal — features at T only use bars with timestamp <= T, and labels
are determined from future bars but those future bars are never used as inputs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterable
import numpy as np
import pandas as pd

from .pools import Pool
from .tester import PoolResult
from .regime import compute_regime_series, lookup_regime, SESSION_LABELS, nse_session
from .indicators import atr


# ---------------------------------------------------------------------------
# Causal state features at a single bar
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
    """Pre-computes anything that can be precomputed per-bar; cheap lookup per snapshot."""

    def __init__(self, df_base: pd.DataFrame):
        self.df = df_base
        self.atr_14 = atr(df_base, 14).bfill()
        self.atr_60 = atr(df_base, 60).bfill()
        self.regime = compute_regime_series(df_base)
        self.close = df_base["close"].values
        self.high = df_base["high"].values
        self.low = df_base["low"].values
        self.idx = df_base.index
        # Rolling mean of close for z-score
        self.roll_mean_50 = df_base["close"].rolling(50, min_periods=10).mean().bfill().values
        self.roll_std_50 = df_base["close"].rolling(50, min_periods=10).std().bfill().values

    def features_at(self, j: int, active_pools: List[Pool]) -> Dict[str, float]:
        """Build the state feature dict at bar j. `active_pools` are pools with available_at <=
        ts[j] AND not yet touched/broken before ts[j] (caller's responsibility)."""
        n = len(self.idx)
        if j < 1 or j >= n:
            return {k: 0.0 for k in STATE_FEATURE_NAMES}

        c = self.close
        ts = self.idx[j]
        a = float(self.atr_14.iloc[j])
        a = max(a, 1e-9)

        def log_ret(k):
            i0 = max(0, j - k)
            if c[i0] <= 0:
                return 0.0
            return float(np.log(c[j] / c[i0]))

        # Momentum: sum of body / ATR over last 12 bars
        i0 = max(0, j - 11)
        oa = (c[i0:j + 1] - c[max(0, i0 - 1):j])[:j - i0 + 1]
        mom_12 = float(np.sum(oa) / a) if len(oa) else 0.0

        # Z-score of close vs 50-bar mean
        mean50 = self.roll_mean_50[j]
        std50 = self.roll_std_50[j]
        z = float((c[j] - mean50) / std50) if std50 > 0 else 0.0

        # Range over last 6 bars / ATR
        i0 = max(0, j - 5)
        rng6 = float((self.high[i0:j + 1].max() - self.low[i0:j + 1].min()) / a)

        # Regime
        adx_14 = float(self.regime["adx_14"].iloc[j])
        vol_r = float(self.regime["vol_ratio"].iloc[j])
        session = nse_session(ts)
        ist_min = (ts + pd.Timedelta(hours=5, minutes=30)).hour * 60 + \
                  (ts + pd.Timedelta(hours=5, minutes=30)).minute
        m_since_open = max(0, ist_min - (9 * 60 + 15))

        # Pool environment
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
                pull_above += 1.0 / max(d, 1.0)  # quality multiplied in later if model provided
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
# Snapshot generation
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    bar_idx: int
    ts: pd.Timestamp
    close: float
    atr: float
    state: Dict[str, float]
    direction_label: int          # 1 = up wins, 0 = down wins in horizon
    max_up_atr: float
    max_dn_atr: float
    # (pool_index_in_pools_list, touched_in_window, distance_atr_at_snapshot, side)
    pool_touches: List[Tuple[int, int, float, str]] = field(default_factory=list)


def generate_snapshots(df_base: pd.DataFrame, pools: List[Pool], results: List[PoolResult],
                        featurizer: StateFeaturizer,
                        window_start: pd.Timestamp, window_end: pd.Timestamp,
                        sample_every: int, horizon: int) -> List[Snapshot]:
    """Walk through df_base at `sample_every` bars; build a Snapshot at each sample point. Only
    sample times within [window_start, window_end - horizon * base_period] are used so every
    snapshot has a full forward window inside the requested window."""
    idx = df_base.index
    n = len(idx)
    h_arr = df_base["high"].values
    l_arr = df_base["low"].values
    c_arr = df_base["close"].values

    # Convert window bounds to bar indices.
    j_start = int(np.searchsorted(idx.values, np.datetime64(window_start), side="left"))
    j_end = int(np.searchsorted(idx.values, np.datetime64(window_end), side="right")) - 1
    j_end = min(j_end, n - horizon - 1)

    snaps: List[Snapshot] = []
    for j in range(max(j_start, 80), j_end + 1, sample_every):  # need >=80 bars warmup
        T = idx[j]

        # Active pools at T: available_at <= T AND not yet touched/broken at T
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

        # Forward window for labels
        j_end_h = j + horizon
        close_T = c_arr[j]
        a_T = float(featurizer.atr_14.iloc[j])
        a_T = max(a_T, 1e-9)
        h_win = h_arr[j + 1 : j_end_h + 1]
        l_win = l_arr[j + 1 : j_end_h + 1]
        if len(h_win) == 0:
            continue
        max_up = float(h_win.max()) - close_T
        max_dn = close_T - float(l_win.min())
        dir_label = 1 if max_up >= max_dn else 0
        max_up_atr = max_up / a_T
        max_dn_atr = max_dn / a_T

        # Per-pool touch labels: touched_at falls in (T, ts[j_end_h]]
        T_end = idx[min(j_end_h, n - 1)]
        pool_touches: List[Tuple[int, int, float, str]] = []
        for pi, p in active_with_idx:
            ta = results[pi].touched_at
            touched_in_window = 1 if (ta is not None and T < ta <= T_end) else 0
            if p.price_low > close_T:
                dist = (p.mid - close_T) / a_T
                side = "above"
            elif p.price_high < close_T:
                dist = (close_T - p.mid) / a_T
                side = "below"
            else:
                # Currently inside the pool — skip (price has already reached it)
                continue
            pool_touches.append((pi, touched_in_window, float(dist), side))

        snaps.append(Snapshot(
            bar_idx=j, ts=T, close=float(close_T), atr=a_T,
            state=state, direction_label=int(dir_label),
            max_up_atr=float(max_up_atr), max_dn_atr=float(max_dn_atr),
            pool_touches=pool_touches,
        ))
    return snaps


# ---------------------------------------------------------------------------
# Direction model
# ---------------------------------------------------------------------------

@dataclass
class DirectionModel:
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

        if len(snapshots) < 30:
            raise ValueError(f"need >= 30 snapshots, got {len(snapshots)}")

        X_df = pd.DataFrame([s.state for s in snapshots], columns=STATE_FEATURE_NAMES).fillna(0.0)
        y = np.array([s.direction_label for s in snapshots], dtype=int)
        self.feature_names = list(X_df.columns)
        self.base_rate = float(y.mean())

        rng = np.random.default_rng(seed)
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
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
        return float(self._iso.transform(raw)[0])

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.feature_names).fillna(0.0).values
        raw = self._gbm.predict(X, num_iteration=self._gbm.best_iteration)
        return np.clip(self._iso.transform(raw), 0.0, 1.0)

    def feature_importance(self, top_k: int = 10) -> List[Tuple[str, int]]:
        if not hasattr(self, "_gbm"):
            return []
        gains = self._gbm.feature_importance(importance_type="gain")
        return sorted(zip(self.feature_names, gains), key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------------------------
# Proximity model
# ---------------------------------------------------------------------------

_POOL_FEATURE_NAMES = ["distance_atr", "side_above",
                       "pool_quality", "pool_score", "pool_width_atr",
                       "pool_n_tfs", "pool_n_contributors",
                       "pool_age_at_avail_bars"]


def _pool_features_for_snapshot(pool: Pool, dist_atr: float, side: str,
                                 quality_pred: float, base_period_seconds: float = 300.0
                                 ) -> Dict[str, float]:
    earliest = min((c.ts for c in pool.contributors), default=pool.formed_at)
    age_bars = (pool.available_at - earliest).total_seconds() / base_period_seconds \
               if pool.available_at >= earliest else 0.0
    # width_atr requires ATR median, approximated by ratio to first contributor's half_width? Use
    # raw width (the proximity model can scale it). Better: width in INR / current atr (caller
    # passes ATR? simpler: use width in price units, model learns scale relative to other feats).
    return {
        "distance_atr": float(dist_atr),
        "side_above": 1.0 if side == "above" else 0.0,
        "pool_quality": float(quality_pred),
        "pool_score": float(pool.score),
        "pool_width_atr": float(pool.width),       # raw price width; comparable across pools
        "pool_n_tfs": float(len(set(pool.tfs))),
        "pool_n_contributors": float(len(pool.contributors)),
        "pool_age_at_avail_bars": float(age_bars),
    }


@dataclass
class ProximityModel:
    feature_names: List[str] = field(default_factory=list)
    train_n: int = 0
    val_n: int = 0
    val_brier: float = 0.0
    val_logloss: float = 0.0
    val_auc: float = 0.0
    base_rate: float = 0.0

    def fit(self, snapshots: List[Snapshot], pools: List[Pool],
            quality_preds: np.ndarray,
            val_frac: float = 0.25, seed: int = 21) -> "ProximityModel":
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        rows = []
        labels = []
        for s in snapshots:
            for (pi, touched, dist, side) in s.pool_touches:
                pool = pools[pi]
                pool_feat = _pool_features_for_snapshot(pool, dist, side, float(quality_preds[pi]))
                merged = {**s.state, **pool_feat}
                rows.append(merged)
                labels.append(touched)
        if len(rows) < 100:
            raise ValueError(f"need >= 100 (snapshot, pool) rows, got {len(rows)}")

        feature_cols = STATE_FEATURE_NAMES + _POOL_FEATURE_NAMES
        X_df = pd.DataFrame(rows, columns=feature_cols).fillna(0.0)
        y = np.array(labels, dtype=int)
        self.feature_names = list(X_df.columns)
        self.base_rate = float(y.mean())

        rng = np.random.default_rng(seed)
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
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
        return self

    def predict_one(self, pool: Pool, dist_atr: float, side: str,
                    state: Dict[str, float], quality_pred: float) -> float:
        pool_feat = _pool_features_for_snapshot(pool, dist_atr, side, quality_pred)
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
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class TimingReport:
    direction_n: int = 0
    direction_brier: float = 0.0
    direction_logloss: float = 0.0
    direction_auc: float = 0.0
    direction_top_quartile_acc: float = 0.0   # acc when |pred-0.5| is highest 25%
    proximity_n: int = 0
    proximity_brier: float = 0.0
    proximity_logloss: float = 0.0
    proximity_auc: float = 0.0
    proximity_decile_lift: float = 0.0
    # Combined ranking quality (using joint quality × touch_prob)
    joint_top_quartile_respect: float = 0.0
    joint_bottom_quartile_respect: float = 0.0
    joint_lift: float = 0.0
    n_joint_evaluated: int = 0


def evaluate_timing(snapshots: List[Snapshot], pools: List[Pool], results: List[PoolResult],
                     direction_model: DirectionModel, proximity_model: ProximityModel,
                     quality_preds: np.ndarray) -> TimingReport:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    rpt = TimingReport()

    # Direction eval
    if direction_model is not None and snapshots:
        X = pd.DataFrame([s.state for s in snapshots], columns=STATE_FEATURE_NAMES).fillna(0.0)
        y = np.array([s.direction_label for s in snapshots], dtype=int)
        p = direction_model.predict_batch(X)
        rpt.direction_n = int(len(y))
        rpt.direction_brier = float(brier_score_loss(y, p))
        rpt.direction_logloss = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
        if len(set(y)) > 1:
            rpt.direction_auc = float(roc_auc_score(y, p))
        # Top-quartile confidence accuracy
        conf = np.abs(p - 0.5)
        if len(p):
            thr = float(np.quantile(conf, 0.75))
            conf_mask = conf >= thr
            if conf_mask.sum() > 0:
                pred_dir = (p[conf_mask] >= 0.5).astype(int)
                rpt.direction_top_quartile_acc = float((pred_dir == y[conf_mask]).mean())

    # Proximity eval
    if proximity_model is not None and snapshots:
        rows = []
        labels = []
        joint_scores = []
        actual_touched = []
        actual_outcome_respected = []
        for s in snapshots:
            for (pi, touched, dist, side) in s.pool_touches:
                pool = pools[pi]
                pf = _pool_features_for_snapshot(pool, dist, side, float(quality_preds[pi]))
                rows.append({**s.state, **pf})
                labels.append(touched)
                joint_scores.append(float(quality_preds[pi]) * float(touched))  # placeholder; replace below
        if rows:
            X = pd.DataFrame(rows, columns=proximity_model.feature_names).fillna(0.0)
            y = np.array(labels, dtype=int)
            raw = proximity_model._gbm.predict(
                X.values, num_iteration=proximity_model._gbm.best_iteration
            )
            p_touch = np.clip(proximity_model._iso.transform(raw), 0.0, 1.0)
            rpt.proximity_n = int(len(y))
            rpt.proximity_brier = float(brier_score_loss(y, p_touch))
            rpt.proximity_logloss = float(log_loss(y, np.clip(p_touch, 1e-6, 1 - 1e-6)))
            if len(set(y)) > 1:
                rpt.proximity_auc = float(roc_auc_score(y, p_touch))
            # Decile lift on touch prediction
            if len(p_touch) >= 20:
                order = np.argsort(p_touch)
                n = len(p_touch)
                bottom = order[: n // 10]
                top = order[-n // 10:]
                bot_rate = float(y[bottom].mean()) if len(bottom) else 0.0
                top_rate = float(y[top].mean()) if len(top) else 0.0
                rpt.proximity_decile_lift = (top_rate / bot_rate) if bot_rate > 0 else float("inf")

            # Joint score evaluation: rank pools by (quality × touch_prob) and check that the
            # top-quartile-scored pools that ACTUALLY got touched were also more likely to be
            # respected than the bottom-quartile-scored touched pools. This is the headline
            # "is the combined signal tradeable" check.
            flat = [(pi, touched) for s in snapshots for (pi, touched, _, _) in s.pool_touches]
            joint = []
            for k, (pi, touched) in enumerate(flat):
                if touched != 1:
                    continue
                r = results[pi]
                if r.is_respect:
                    label = 1
                elif r.is_break:
                    label = 0
                else:
                    continue  # ambiguous outcome — skip
                score = float(quality_preds[pi]) * float(p_touch[k])
                joint.append((score, label))
            if len(joint) >= 20:
                joint_sorted = sorted(joint, key=lambda x: x[0])
                m = len(joint_sorted)
                bot = joint_sorted[: m // 4]
                top = joint_sorted[-m // 4:]
                bot_resp = float(np.mean([r for _, r in bot]))
                top_resp = float(np.mean([r for _, r in top]))
                rpt.joint_bottom_quartile_respect = bot_resp
                rpt.joint_top_quartile_respect = top_resp
                rpt.joint_lift = (top_resp / bot_resp) if bot_resp > 0 else float("inf")
                rpt.n_joint_evaluated = m

    return rpt


def print_timing_report(rpt: TimingReport, file=None) -> None:
    print("\n=== Track 3: Direction + Timing Models (OOS) ===", file=file)
    print(f"\n[direction model]   target = (max_up >= max_dn) in next H bars", file=file)
    print(f"  OOS samples:            {rpt.direction_n}", file=file)
    print(f"  Brier:                  {rpt.direction_brier:.4f}", file=file)
    print(f"  Log loss:               {rpt.direction_logloss:.4f}", file=file)
    print(f"  AUC-ROC:                {rpt.direction_auc:.3f}  (0.5=random, 0.55+=useful)", file=file)
    print(f"  Top-quartile-confidence accuracy:  {rpt.direction_top_quartile_acc:.1%}  "
          f"(when model is sure)", file=file)

    print(f"\n[proximity model]   target = pool touched within H bars from snapshot", file=file)
    print(f"  OOS (snapshot,pool) rows:  {rpt.proximity_n}", file=file)
    print(f"  Brier:                  {rpt.proximity_brier:.4f}", file=file)
    print(f"  Log loss:               {rpt.proximity_logloss:.4f}", file=file)
    print(f"  AUC-ROC:                {rpt.proximity_auc:.3f}", file=file)
    print(f"  Decile lift:            {rpt.proximity_decile_lift:.2f}x", file=file)

    print(f"\n[joint score ranking]  quality × touch_prob, scored on actually-touched pools",
          file=file)
    print(f"  decisive pools evaluated:   {rpt.n_joint_evaluated}", file=file)
    print(f"  Top quartile respect rate:  {rpt.joint_top_quartile_respect:.1%}", file=file)
    print(f"  Bottom quartile respect:    {rpt.joint_bottom_quartile_respect:.1%}", file=file)
    print(f"  Joint lift:                 {rpt.joint_lift:.2f}x", file=file)
