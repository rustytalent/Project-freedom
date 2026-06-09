"""LightGBM model + isotonic calibration for per-pool P(respect).

Training is done on OOS (pool → outcome) pairs gathered from walk-forward folds, where every
training pool is one whose outcome was determined by a config that did NOT see its label during
training. So this is honest OOS calibration.

Design notes:
  - Trees are kept SMALL (max_depth=4, min_data_in_leaf=8, strong L2) because the training set
    is typically a few hundred pools. Default LightGBM would overfit immediately.
  - We hold out a slice for early stopping AND for isotonic calibration. The calibrator is fit
    on the held-out validation predictions vs labels, so it learns the model's miscalibration.
  - The bucket-level isotonic shrinkage layer (per-bucket recalibration) is applied AFTER the
    global isotonic. It pulls predictions toward each bucket's empirical OOS rate, with weight
    proportional to bucket sample size — small buckets are mostly trusted to the global model,
    large buckets influence more.

Target = "is_decisive AND is_respect" (so respected_strong + swept_and_reclaimed = 1, broken_strong
and broken_weak = 0). Weak respects and ambiguous outcomes are EXCLUDED from training (they're
noise). This focuses the model on learning what produces decisive positive outcomes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Sequence
from collections import defaultdict
import numpy as np
import pandas as pd

from .pools import Pool
from .tester import PoolResult
from .stratified import _tf_bucket, _headline_factor
from .distance_calibration import DistanceCalibrator


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

# A pool is "trainable" if its outcome is decisive (clean win or clean loss). We exclude
# untouched/ambiguous outcomes because they aren't real evidence of pool quality.
_TRAIN_OUTCOMES = ("respected_strong", "swept_and_reclaimed", "broken_strong", "broken_weak")
_POSITIVE_OUTCOMES = ("respected_strong", "swept_and_reclaimed")


def trainable_mask(results: Sequence[PoolResult]) -> np.ndarray:
    return np.array([r.outcome in _TRAIN_OUTCOMES for r in results], dtype=bool)


def labels(results: Sequence[PoolResult]) -> np.ndarray:
    return np.array([1 if r.outcome in _POSITIVE_OUTCOMES else 0 for r in results], dtype=int)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class BucketCalib:
    """Per-bucket recalibration: pulls global predictions toward bucket's empirical rate."""
    n_oos: int
    empirical_rate: float
    pull_weight: float        # how much to pull (0=trust global, 1=use empirical)


@dataclass
class PurgedFoldStats:
    """Audit trail for purged/embargoed quality-model validation folds."""
    fold: int
    validation_start: str
    validation_end: str
    original_train_size: int
    purged_train_size: int
    purged_rows_removed: int
    embargoed_rows_removed: int
    validation_size: int


def label_end_time(pool: Pool, result: PoolResult) -> pd.Timestamp:
    """Best available end of the pool's evaluation window for purging overlap checks.

    Picks the latest of {available_at, touched_at, broken_at} that is
    non-null. ``available_at`` is always set by the pool builder so
    this is normally well-defined. If somehow all three are None (a
    corrupt result), we raise rather than silently returning a
    zero-width window — that would create unembargoable label
    intervals for adjacent pools, exactly the boundary the purge is
    supposed to enforce.
    """
    candidates = [pool.available_at, result.touched_at, result.broken_at]
    valid = [ts for ts in candidates if ts is not None]
    if not valid:
        raise ValueError(
            f"label_end_time: no end candidate for pool "
            f"{getattr(pool, 'pool_id', '?')} - "
            f"available_at={pool.available_at}, "
            f"touched_at={result.touched_at}, broken_at={result.broken_at}"
        )
    return max(valid)


def _lgb_params(seed: int, regularization_preset: str) -> Dict:
    if regularization_preset == "conservative_finml":
        return dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.035,
            # Conservative financial-ML preset: shallow, bagged, column-subsampled, and heavily
            # regularized to reduce small-sample memorisation in the quality/reaction model.
            num_leaves=12,
            max_depth=4,
            min_data_in_leaf=35,
            feature_fraction=0.60,
            bagging_fraction=0.70,
            bagging_freq=5,
            lambda_l2=12.0,
            lambda_l1=2.0,
            min_gain_to_split=0.01,
            verbose=-1,
            seed=seed,
        )
    return dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=8,
        max_depth=3,
        min_data_in_leaf=15,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=5,
        lambda_l2=5.0,
        lambda_l1=0.5,
        verbose=-1,
        seed=seed,
    )


def _purged_embargoed_splits(start_times: Sequence[pd.Timestamp],
                             end_times: Sequence[pd.Timestamp],
                             y: np.ndarray,
                             embargo_bars: int,
                             base_period_seconds: float,
                             n_folds: int = 4) -> Tuple[List[Tuple[np.ndarray, np.ndarray]],
                                                        List[PurgedFoldStats]]:
    n = len(start_times)
    if n < 30:
        return [], []
    starts = pd.to_datetime(list(start_times))
    ends = pd.to_datetime(list(end_times))
    order = np.argsort(starts.values)
    chunks = [c for c in np.array_split(order, max(2, min(n_folds, n // 10))) if len(c) > 0]
    embargo_td = pd.Timedelta(seconds=float(base_period_seconds) * max(0, embargo_bars))
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    stats: List[PurgedFoldStats] = []

    for fold, val_idx in enumerate(chunks):
        val_start = starts[val_idx].min()
        val_end = starts[val_idx].max()
        non_val = np.setdiff1d(np.arange(n), val_idx, assume_unique=False)
        # Purge any sample whose own label/evaluation window intersects the validation window.
        overlaps_validation = (starts[non_val] <= val_end) & (ends[non_val] >= val_start)
        # Embargo samples that begin immediately after validation, where shared market state can
        # make near-duplicate pools look independent.
        embargoed = (starts[non_val] > val_end) & (starts[non_val] <= val_end + embargo_td)
        train_idx = non_val[~(overlaps_validation | embargoed)]
        st = PurgedFoldStats(
            fold=fold,
            validation_start=str(val_start),
            validation_end=str(val_end),
            original_train_size=int(len(non_val)),
            purged_train_size=int(len(train_idx)),
            purged_rows_removed=int(overlaps_validation.sum()),
            embargoed_rows_removed=int(embargoed.sum()),
            validation_size=int(len(val_idx)),
        )
        stats.append(st)
        if len(train_idx) >= 20 and len(val_idx) >= 5:
            if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[val_idx])) == 2:
                splits.append((train_idx, val_idx))
    return splits, stats


def _stratified_halves(idx: np.ndarray, y: np.ndarray, rng: np.random.Generator
                       ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Split `idx` into two disjoint, class-stratified halves: (early_stop, holdout, ok).

    The first half is used for early stopping + isotonic calibration (tuning); the second is a
    pure holdout for honest metric reporting. If a clean split keeping both classes on each side
    isn't possible (too few of either class), returns (idx, idx, False) so the caller falls back
    to the pre-holdout behaviour (calibrate and report on the same rows)."""
    idx = np.asarray(idx)
    yv = y[idx]
    pos = idx[yv == 1]
    neg = idx[yv == 0]
    if len(pos) < 2 or len(neg) < 2:
        return idx, idx, False
    rng.shuffle(pos)
    rng.shuffle(neg)
    ep = max(1, len(pos) // 2)
    en = max(1, len(neg) // 2)
    es = np.concatenate([pos[:ep], neg[:en]])
    holdout = np.concatenate([pos[ep:], neg[en:]])
    if (len(holdout) == 0 or len(np.unique(y[holdout])) < 2
            or len(np.unique(y[es])) < 2):
        return idx, idx, False
    rng.shuffle(es)
    rng.shuffle(holdout)
    return es, holdout, True


@dataclass
class PoolRespectModel:
    """LightGBM + isotonic calibration + bucket shrinkage. predict() returns calibrated P(respect)."""
    feature_names: List[str] = field(default_factory=list)
    bucket_calib: Dict[Tuple[str, str], BucketCalib] = field(default_factory=dict)
    train_n: int = 0
    val_n: int = 0
    val_brier: float = 0.0
    val_logloss: float = 0.0
    val_auc: float = 0.0
    train_brier: float = 0.0
    train_logloss: float = 0.0
    train_auc: float = 0.0
    base_rate: float = 0.0
    has_calibration_holdout: bool = True
    validation_method: str = "random_stratified"
    regularization_preset: str = "default"
    hyperparameters: Dict = field(default_factory=dict)
    validation_fold_stats: List[PurgedFoldStats] = field(default_factory=list)
    validation_indices: List[int] = field(default_factory=list)
    # The underlying objects are stashed as attributes after fit().

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            buckets: Optional[List[Tuple[str, str]]] = None,
            val_frac: float = 0.25, seed: int = 17,
            sample_start_times: Optional[Sequence[pd.Timestamp]] = None,
            sample_end_times: Optional[Sequence[pd.Timestamp]] = None,
            embargo_bars: int = 78,
            base_period_seconds: float = 300.0,
            validation_method: str = "random_stratified",
            regularization_preset: str = "default",
            sample_weight: Optional[np.ndarray] = None) -> "PoolRespectModel":
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        rng = np.random.default_rng(seed)
        n = len(X)
        if n < 20:
            raise ValueError(f"need at least 20 trainable pools, got {n}")

        self.feature_names = list(X.columns)
        self.base_rate = float(y.mean())
        self.regularization_preset = regularization_preset
        self.validation_fold_stats = []

        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        if (validation_method == "purged_embargoed_walk_forward"
                and sample_start_times is not None and sample_end_times is not None):
            splits, self.validation_fold_stats = _purged_embargoed_splits(
                sample_start_times, sample_end_times, y,
                embargo_bars=embargo_bars,
                base_period_seconds=base_period_seconds,
            )

        if splits:
            # Fit on the latest usable chronological validation split. Earlier fold stats remain
            # logged so the run can show how much purging/embargoing affected sample availability.
            train_idx, val_idx = splits[-1]
            self.validation_method = "purged_embargoed_walk_forward"
        else:
            # Backward-compatible fallback: stratified random split for callers that have not yet
            # supplied sample label windows.
            self.validation_method = "random_stratified"
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            n_val_pos = max(1, int(round(len(pos_idx) * val_frac)))
            n_val_neg = max(1, int(round(len(neg_idx) * val_frac)))
            val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
            train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])
            rng.shuffle(val_idx); rng.shuffle(train_idx)

        X_tr, y_tr = X.iloc[train_idx].values, y[train_idx]
        # Split the validation fold so the booster's best_iteration AND the isotonic map are fit
        # on one half (early-stop/calibration), and ALL reported val_* metrics come from the other
        # (a pure holdout). Fitting isotonic on the early-stopping rows and then scoring those same
        # rows is optimistic; the holdout makes the reported calibration metrics honest.
        es_idx, holdout_idx, has_holdout = _stratified_halves(val_idx, y, rng)
        self.has_calibration_holdout = bool(has_holdout)

        X_es, y_es = X.iloc[es_idx].values, y[es_idx]
        X_hold, y_hold = X.iloc[holdout_idx].values, y[holdout_idx]

        params = _lgb_params(seed, regularization_preset)
        self.hyperparameters = dict(params)
        # Optional per-sample weight (e.g. time-decay). Skip when not
        # provided or wrong-length so the back-compat path is identical
        # to pre-feature behaviour.
        w_tr = None
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if len(sample_weight) == len(X):
                w_tr = sample_weight[train_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                              feature_name=self.feature_names)
        des = lgb.Dataset(X_es, label=y_es, reference=dtrain, feature_name=self.feature_names)

        self._gbm = lgb.train(
            params, dtrain, num_boost_round=400, valid_sets=[des], valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                       lgb.log_evaluation(0)],
        )

        # Isotonic calibration on the early-stop set (a monotone map — a tuning step).
        es_raw = self._gbm.predict(X_es, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(es_raw, y_es)

        # Honest calibrated metrics on the holdout rows (never used for booster fit, early
        # stopping, or isotonic). If no clean holdout existed, these fall back to the es rows.
        hold_calib = self._iso.transform(
            self._gbm.predict(X_hold, num_iteration=self._gbm.best_iteration))
        train_calib = self._iso.transform(
            self._gbm.predict(X_tr, num_iteration=self._gbm.best_iteration))

        self.train_n = int(len(train_idx))
        self.val_n = int(len(holdout_idx))
        self.validation_indices = [int(i) for i in holdout_idx]
        self.train_brier = float(brier_score_loss(y_tr, train_calib))
        self.train_logloss = float(log_loss(y_tr, np.clip(train_calib, 1e-6, 1 - 1e-6)))
        self.train_auc = float(roc_auc_score(y_tr, train_calib)) if len(set(y_tr)) > 1 else 0.0
        self.val_brier = float(brier_score_loss(y_hold, hold_calib))
        self.val_logloss = float(log_loss(y_hold, np.clip(hold_calib, 1e-6, 1 - 1e-6)))
        # AUC needs both classes present. The stratified split guarantees this, but the
        # purged/embargoed walk-forward split does not, so guard it (mirrors train_auc above).
        self.val_auc = float(roc_auc_score(y_hold, hold_calib)) if len(set(y_hold)) > 1 else 0.0

        # Bucket recalibration is fit by the caller (walkforward) after predict on the full OOS set
        # since it needs (bucket, predicted, actual) — see fit_bucket_calib below.
        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._gbm.predict(X[self.feature_names].values,
                                num_iteration=self._gbm.best_iteration)
        return self._iso.transform(raw)

    def fit_distance_calib(self, p_predicted: np.ndarray, distance_atr: np.ndarray,
                           y_true: np.ndarray,
                           min_bucket_n: int = 300) -> Optional[DistanceCalibrator]:
        """Stage-C: per-distance-bucket isotonic on top of (global iso × bucket shrinkage).

        ``p_predicted`` is the model's already-calibrated output (post-isotonic;
        callers usually pass the value of ``predict(...)`` on the OOS rows so the
        layer composes correctly). Buckets below ``min_bucket_n`` fall back to
        identity, so a sparse bucket can never make calibration worse.

        Not auto-applied in ``predict``: that path has no ``distance_atr``. The
        joint Q × proximity evaluation site supplies distance and should call
        :meth:`apply_distance_calibration` explicitly.
        """
        calib = DistanceCalibrator(min_bucket_n=min_bucket_n).fit(
            raw_p=np.asarray(p_predicted, dtype=float),
            distance_atr=np.asarray(distance_atr, dtype=float),
            y_true=np.asarray(y_true, dtype=int),
        )
        self._distance_calib = calib
        return calib

    def apply_distance_calibration(self, p: np.ndarray,
                                   distance_atr: np.ndarray) -> np.ndarray:
        """Apply the Stage-C layer if fitted, else return ``p`` unchanged."""
        calib = getattr(self, "_distance_calib", None)
        if calib is None or not calib.is_fitted:
            return np.asarray(p, dtype=float)
        return np.clip(calib.transform(p, distance_atr), 0.02, 0.98)

    def fit_bucket_calib(self, pools: List[Pool], y_true: np.ndarray, p_predicted: np.ndarray,
                         min_bucket_n: int = 10,
                         shrinkage_max: float = 1.0) -> None:
        """Build per-bucket pull weights from observed (pred, actual) on the OOS set.

        For each bucket: empirical_rate = mean(y_true_in_bucket). The "pull" applied at predict
        time is a sample-size-weighted shrinkage toward this rate:
            pull_weight = min(shrinkage_max, n / (n + 20))
            (Wilson-style shrinkage; 20 = global-prior strength)
        Adjustment at predict time:
            calibrated = (1 - pull) * global_prediction + pull * empirical_rate
        Small buckets (n<min_bucket_n) get no per-bucket adjustment.

        ``shrinkage_max`` caps the maximum pull weight. With ``shrinkage_max=1.0``
        (legacy default for backward compatibility) the pull is unbounded, so
        large buckets (e.g. 4+TF EQHL with n>=4000) get pull≈1.0 which
        collapses ALL predictions in that bucket to the bucket empirical mean.
        That's the "Q compression" symptom documented in
        ``reports/phase4_workstream0_q_audit.md``: regardless of how confident
        the underlying gbm was, the post-shrinkage Q range collapsed to ~23-56%.
        Setting ``shrinkage_max`` to e.g. 0.30 preserves the model's
        within-bucket variance while still applying meaningful per-bucket bias
        correction.
        """
        groups: Dict[Tuple[str, str], List[Tuple[float, int]]] = defaultdict(list)
        for p, pred, y in zip(pools, p_predicted, y_true):
            key = (_tf_bucket(len(set(p.tfs))), _headline_factor(p))
            groups[key].append((float(pred), int(y)))
        self.bucket_calib.clear()
        cap = float(max(0.0, min(1.0, shrinkage_max)))
        for key, items in groups.items():
            if len(items) < min_bucket_n:
                continue
            arr_y = np.array([y for _, y in items])
            emp = float(arr_y.mean())
            n = len(items)
            pull = min(cap, n / (n + 20.0))
            self.bucket_calib[key] = BucketCalib(n_oos=n, empirical_rate=emp, pull_weight=pull)

    def predict(self, X: pd.DataFrame, pools: Optional[List[Pool]] = None) -> np.ndarray:
        """Returns final calibrated P(respect). If pools given AND bucket_calib has been fit,
        applies bucket-level shrinkage on top of global calibration."""
        base = self.predict_raw(X)
        if pools is None or not self.bucket_calib:
            return np.clip(base, 0.02, 0.98)
        adj = base.copy()
        for i, p in enumerate(pools):
            key = (_tf_bucket(len(set(p.tfs))), _headline_factor(p))
            bc = self.bucket_calib.get(key)
            if bc is not None:
                adj[i] = (1.0 - bc.pull_weight) * base[i] + bc.pull_weight * bc.empirical_rate
        return np.clip(adj, 0.02, 0.98)

    def feature_importance(self, top_k: int = 15) -> List[Tuple[str, int]]:
        if not hasattr(self, "_gbm"):
            return []
        gains = self._gbm.feature_importance(importance_type="gain")
        pairs = sorted(zip(self.feature_names, gains), key=lambda x: -x[1])
        return pairs[:top_k]

    def explain_prediction(self, X: pd.DataFrame, top_k: int = 5) -> List[Dict]:
        """Per-row feature contribution explanations (SHAP-like via LightGBM's pred_contrib).

        For each row in X, returns a dict:
            {
              "base_logit": <model prior in logit space>,
              "raw_prediction_logit": <pre-isotonic prediction>,
              "top_features": [(feature_name, logit_contribution), ...]
            }
        Contributions are in LOGIT space (before sigmoid + isotonic). Positive = pushed
        prediction up; negative = pushed it down. Sum of all contributions ≈ raw_prediction_logit.
        Top features are sorted by |contribution| so both pro and con drivers surface."""
        if not hasattr(self, "_gbm"):
            return []
        X_arr = X[self.feature_names].values
        contribs = self._gbm.predict(X_arr, pred_contrib=True,
                                       num_iteration=self._gbm.best_iteration)
        # contribs shape: (n, n_features + 1). Last column is the base value.
        out = []
        for row in contribs:
            base = float(row[-1])
            feat_contribs = list(zip(self.feature_names, [float(v) for v in row[:-1]]))
            feat_contribs.sort(key=lambda kv: -abs(kv[1]))
            out.append({
                "base_logit": base,
                "raw_prediction_logit": float(row.sum()),
                "top_features": feat_contribs[:top_k],
            })
        return out


@dataclass
class SectorMoERespectModel:
    """Global PoolRespectModel plus sector experts with hard routing + soft blend.

    This is intentionally conservative: the global model remains the fallback/stabilizer, while
    sector experts are only trained when a sector has enough decisive examples and both classes.
    predict() keeps the same shape as PoolRespectModel.predict(), so downstream ranking, timing,
    and live execution can consume it without a separate code path.
    """
    global_model: Optional[PoolRespectModel] = None
    sector_models: Dict[str, PoolRespectModel] = field(default_factory=dict)
    sector_stats: Dict[str, Dict] = field(default_factory=dict)
    sector_weights: Dict[str, float] = field(default_factory=dict)
    expert_weight: float = 0.70          # cap only; per-sector weights are dynamic.
    global_weight: float = 0.30
    min_sector_train_n: int = 60
    min_sector_class_n: int = 8
    min_sector_oos_n: int = 20
    feature_names: List[str] = field(default_factory=list)
    train_n: int = 0
    val_n: int = 0
    val_brier: float = 0.0
    val_logloss: float = 0.0
    val_auc: float = 0.0
    train_brier: float = 0.0
    train_logloss: float = 0.0
    train_auc: float = 0.0
    base_rate: float = 0.0
    validation_method: str = "random_stratified"
    regularization_preset: str = "default"
    hyperparameters: Dict = field(default_factory=dict)
    validation_fold_stats: List[PurgedFoldStats] = field(default_factory=list)

    @property
    def bucket_calib(self) -> Dict[Tuple[str, str], BucketCalib]:
        return self.global_model.bucket_calib if self.global_model is not None else {}

    def fit_distance_calib(self, p_predicted: np.ndarray,
                           distance_atr: np.ndarray, y_true: np.ndarray,
                           min_bucket_n: int = 300) -> Optional[DistanceCalibrator]:
        """Delegates Stage-C distance calibration to the global model."""
        if self.global_model is None:
            return None
        return self.global_model.fit_distance_calib(
            p_predicted, distance_atr, y_true, min_bucket_n=min_bucket_n)

    def apply_distance_calibration(self, p: np.ndarray,
                                   distance_atr: np.ndarray) -> np.ndarray:
        if self.global_model is None:
            return np.asarray(p, dtype=float)
        return self.global_model.apply_distance_calibration(p, distance_atr)

    def _copy_global_metrics(self) -> None:
        if self.global_model is None:
            return
        self.feature_names = list(self.global_model.feature_names)
        self.train_n = self.global_model.train_n
        self.val_n = self.global_model.val_n
        self.val_brier = self.global_model.val_brier
        self.val_logloss = self.global_model.val_logloss
        self.val_auc = self.global_model.val_auc
        self.train_brier = self.global_model.train_brier
        self.train_logloss = self.global_model.train_logloss
        self.train_auc = self.global_model.train_auc
        self.base_rate = self.global_model.base_rate
        self.validation_method = self.global_model.validation_method
        self.regularization_preset = self.global_model.regularization_preset
        self.hyperparameters = dict(self.global_model.hyperparameters)
        self.validation_fold_stats = list(self.global_model.validation_fold_stats)

    @staticmethod
    def _sector_for_pool(pool: Pool) -> str:
        from .sectors import sector_of
        return sector_of(pool.asset)

    def _fit_bucket_if_possible(self, model: PoolRespectModel, X_oos: Optional[pd.DataFrame],
                                oos_pools: Optional[List[Pool]],
                                oos_results: Optional[List[PoolResult]],
                                min_bucket_n: int = 10,
                                shrinkage_max: float = 1.0) -> int:
        if X_oos is None or not oos_pools or not oos_results:
            return 0
        decisive_mask = trainable_mask(oos_results)
        if decisive_mask.sum() < self.min_sector_oos_n:
            return int(decisive_mask.sum())
        y_oos_dec = labels([r for r, keep in zip(oos_results, decisive_mask) if keep])
        pools_dec = [p for p, keep in zip(oos_pools, decisive_mask) if keep]
        p_raw = model.predict_raw(X_oos)
        p_raw_dec = p_raw[decisive_mask]
        model.fit_bucket_calib(pools_dec, y_oos_dec, p_raw_dec,
                                min_bucket_n=min_bucket_n,
                                shrinkage_max=shrinkage_max)
        return int(decisive_mask.sum())

    def fit(self, X: pd.DataFrame, y: np.ndarray, train_pools: List[Pool],
            X_oos: Optional[pd.DataFrame] = None,
            oos_pools: Optional[List[Pool]] = None,
            oos_results: Optional[List[PoolResult]] = None,
            val_frac: float = 0.25, seed: int = 17,
            min_sector_train_n: Optional[int] = None,
            min_sector_class_n: Optional[int] = None,
            min_sector_oos_n: Optional[int] = None,
            expert_weight: Optional[float] = None,
            sample_start_times: Optional[Sequence[pd.Timestamp]] = None,
            sample_end_times: Optional[Sequence[pd.Timestamp]] = None,
            embargo_bars: int = 78,
            base_period_seconds: float = 300.0,
            validation_method: str = "purged_embargoed_walk_forward",
            regularization_preset: str = "default",
            bucket_shrinkage_max: float = 1.0,
            train_sector_experts: bool = True,
            sample_decay_halflife_days: Optional[float] = None,
            ) -> "SectorMoERespectModel":
        """Fit the global Q model and (optionally) per-sector experts.

        ``train_sector_experts`` controls whether the per-sector expert
        loop runs at all. The pipeline default (set in Config) is False
        because empirically, 4 of 5 sectors get dynamic MoE weight 0%
        every retrain — the experts get trained and then thrown away.
        Skipping the per-sector loop saves ~40% of Q training time with
        no measurable impact on blended Q AUC. Setting True restores
        the legacy MoE behaviour for research toggling. When False, all
        sector_weights are 0.0 and the global model carries all weight.
        """
        if len(X) != len(y) or len(X) != len(train_pools):
            raise ValueError("X, y, and train_pools must have matching lengths")
        if expert_weight is not None:
            self.expert_weight = float(expert_weight)
            self.global_weight = 1.0 - self.expert_weight
        if min_sector_train_n is not None:
            self.min_sector_train_n = int(min_sector_train_n)
        if min_sector_class_n is not None:
            self.min_sector_class_n = int(min_sector_class_n)
        if min_sector_oos_n is not None:
            self.min_sector_oos_n = int(min_sector_oos_n)

        # Time-decay sample weights: recent samples weighted higher.
        # When the kwarg is None or <= 0, time_decay_weights returns
        # uniform 1.0s — identical to the no-decay baseline.
        from .sample_weights import time_decay_weights
        sample_w = None
        if (sample_decay_halflife_days is not None
                and sample_decay_halflife_days > 0
                and sample_start_times is not None):
            sample_w = time_decay_weights(
                sample_times=sample_start_times,
                half_life_days=float(sample_decay_halflife_days),
            )
        self.global_model = PoolRespectModel().fit(
            X, y, val_frac=val_frac, seed=seed,
            sample_start_times=sample_start_times,
            sample_end_times=sample_end_times,
            embargo_bars=embargo_bars,
            base_period_seconds=base_period_seconds,
            validation_method=validation_method,
            regularization_preset=regularization_preset,
            sample_weight=sample_w,
        )
        self._fit_bucket_if_possible(self.global_model, X_oos, oos_pools, oos_results,
                                     min_bucket_n=10,
                                     shrinkage_max=bucket_shrinkage_max)
        self._copy_global_metrics()

        sectors = sorted({self._sector_for_pool(p) for p in train_pools})
        self.sector_models.clear()
        self.sector_stats.clear()
        self.sector_weights.clear()

        if not train_sector_experts:
            # Mark every sector as 'experts_disabled' so the report still
            # carries the per-sector accounting (downstream callers
            # iterate sector_stats and would crash on a missing key).
            for sector in sectors:
                idx = [i for i, p in enumerate(train_pools)
                        if self._sector_for_pool(p) == sector]
                y_sector = y[idx]
                self.sector_stats[sector] = {
                    "train_n": int(len(idx)),
                    "pos_n": int((y_sector == 1).sum()),
                    "neg_n": int((y_sector == 0).sum()),
                    "status": "experts_disabled",
                    "reason": "train_sector_experts=False",
                    "oos_decisive_n": 0,
                    "val_auc": None,
                    "expert_logloss": None,
                    "global_logloss": None,
                    "sector_weight": 0.0,
                }
                self.sector_weights[sector] = 0.0
            return self

        for sector in sectors:
            idx = [i for i, p in enumerate(train_pools) if self._sector_for_pool(p) == sector]
            y_sector = y[idx]
            pos_n = int((y_sector == 1).sum())
            neg_n = int((y_sector == 0).sum())
            stat = {
                "train_n": int(len(idx)),
                "pos_n": pos_n,
                "neg_n": neg_n,
                "status": "skipped",
                "reason": "",
                "oos_decisive_n": 0,
                "val_auc": None,
                "expert_logloss": None,
                "global_logloss": None,
                "sector_weight": 0.0,
            }

            if len(idx) < self.min_sector_train_n:
                stat["reason"] = f"train_n<{self.min_sector_train_n}"
                self.sector_stats[sector] = stat
                continue
            if pos_n < self.min_sector_class_n or neg_n < self.min_sector_class_n:
                stat["reason"] = f"class_n<{self.min_sector_class_n}"
                self.sector_stats[sector] = stat
                continue

            try:
                sector_starts = ([sample_start_times[i] for i in idx]
                                 if sample_start_times is not None else None)
                sector_ends = ([sample_end_times[i] for i in idx]
                               if sample_end_times is not None else None)
                expert = PoolRespectModel().fit(
                    X.iloc[idx], y_sector,
                    val_frac=val_frac, seed=seed + len(idx),
                    sample_start_times=sector_starts,
                    sample_end_times=sector_ends,
                    embargo_bars=embargo_bars,
                    base_period_seconds=base_period_seconds,
                    validation_method=validation_method,
                    regularization_preset=regularization_preset,
                )

                sector_oos_n = 0
                if X_oos is not None and oos_pools and oos_results:
                    oos_idx = [i for i, p in enumerate(oos_pools)
                               if self._sector_for_pool(p) == sector]
                    if oos_idx:
                        sector_oos_n = self._fit_bucket_if_possible(
                            expert, X_oos.iloc[oos_idx],
                            [oos_pools[i] for i in oos_idx],
                            [oos_results[i] for i in oos_idx],
                            min_bucket_n=8,
                            shrinkage_max=bucket_shrinkage_max,
                        )

                global_loss = None
                expert_loss = float(expert.val_logloss)
                sector_weight = 0.0
                shrink_reason = ""
                if expert.validation_indices:
                    from sklearn.metrics import log_loss
                    val_local = np.array(expert.validation_indices, dtype=int)
                    val_global = [idx[int(i)] for i in val_local]
                    y_val_sector = y[val_global]
                    global_pred = self.global_model.predict(X.iloc[val_global],
                                                            pools=[train_pools[i] for i in val_global])
                    global_loss = float(log_loss(y_val_sector,
                                                 np.clip(global_pred, 1e-6, 1 - 1e-6)))
                    improvement = global_loss - expert_loss
                    if improvement <= 0.005:
                        shrink_reason = "expert did not beat global enough"
                    elif expert.val_auc < 0.53:
                        shrink_reason = "expert AUC too weak"
                    else:
                        # Dynamic MoE shrinkage: trust a sector expert only in proportion to
                        # sample size, validation-loss improvement over global, and expert AUC.
                        sample_factor = min(1.0, max(0.0, (len(idx) - self.min_sector_train_n)
                                                     / max(1.0, 200.0 - self.min_sector_train_n)))
                        loss_factor = min(1.0, max(0.0, improvement / 0.05))
                        auc_factor = min(1.0, max(0.0, (expert.val_auc - 0.50) / 0.15))
                        sector_weight = min(self.expert_weight,
                                            self.expert_weight * sample_factor
                                            * loss_factor * auc_factor)
                        if sector_weight < 0.05:
                            sector_weight = 0.0
                            shrink_reason = "weight shrunk below usable floor"
                        else:
                            shrink_reason = "dynamic shrinkage accepted"
                else:
                    shrink_reason = "no expert validation fold"

                self.sector_models[sector] = expert
                self.sector_weights[sector] = float(sector_weight)
                stat.update({
                    "status": "trained",
                    "reason": shrink_reason,
                    "oos_decisive_n": int(sector_oos_n),
                    "val_auc": float(expert.val_auc),
                    "base_rate": float(expert.base_rate),
                    "expert_logloss": expert_loss,
                    "global_logloss": global_loss,
                    "sector_weight": float(sector_weight),
                })
            except Exception as exc:
                stat["reason"] = str(exc)
            self.sector_stats[sector] = stat

        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_model is None:
            raise ValueError("SectorMoERespectModel is not fitted")
        return self.global_model.predict_raw(X)

    def predict(self, X: pd.DataFrame, pools: Optional[List[Pool]] = None) -> np.ndarray:
        if self.global_model is None:
            raise ValueError("SectorMoERespectModel is not fitted")
        global_pred = self.global_model.predict(X, pools=pools)
        if pools is None or not self.sector_models:
            return global_pred

        final = global_pred.copy()
        for sector, expert in self.sector_models.items():
            idx = [i for i, p in enumerate(pools) if self._sector_for_pool(p) == sector]
            if not idx:
                continue
            expert_pred = expert.predict(X.iloc[idx], pools=[pools[i] for i in idx])
            w = float(self.sector_weights.get(sector, 0.0))
            final[idx] = w * expert_pred + (1.0 - w) * global_pred[idx]
        return np.clip(final, 0.02, 0.98)

    def predict_components(self, X: pd.DataFrame, pools: List[Pool]) -> pd.DataFrame:
        """Return global, sector-expert, and blended Q for audit/explanation.

        `sector_q` is NaN when no trained expert exists for that pool's sector. In that case the
        blended prediction is the global fallback and `gate_weight` is 0.0. This keeps live code
        and audit tables explicit about when the MoE is really routing to a sector expert.
        """
        if self.global_model is None:
            raise ValueError("SectorMoERespectModel is not fitted")
        if len(X) != len(pools):
            raise ValueError("X and pools must have matching lengths")

        global_pred = self.global_model.predict(X, pools=pools)
        sector_pred = np.full(len(pools), np.nan, dtype=float)
        gate_weight = np.zeros(len(pools), dtype=float)
        fallback_reason = []
        for p in pools:
            sector = self._sector_for_pool(p)
            stat = self.sector_stats.get(sector, {})
            fallback_reason.append(stat.get("reason") or "no trained sector expert")

        for sector, expert in self.sector_models.items():
            idx = [i for i, p in enumerate(pools) if self._sector_for_pool(p) == sector]
            if not idx:
                continue
            pred = expert.predict(X.iloc[idx], pools=[pools[i] for i in idx])
            w = float(self.sector_weights.get(sector, 0.0))
            sector_pred[idx] = pred
            gate_weight[idx] = w
            for i in idx:
                fallback_reason[i] = "" if w > 0 else self.sector_stats.get(sector, {}).get(
                    "reason", "sector expert fully shrunk to global")

        blended = np.where(
            np.isnan(sector_pred),
            global_pred,
            gate_weight * sector_pred + (1.0 - gate_weight) * global_pred,
        )
        sectors = [self._sector_for_pool(p) for p in pools]
        assets = [p.asset for p in pools]
        return pd.DataFrame({
            "asset": assets,
            "sector": sectors,
            "global_q": np.clip(global_pred, 0.02, 0.98),
            "sector_q": sector_pred,
            "gate_weight": gate_weight,
            "blended_q": np.clip(blended, 0.02, 0.98),
            "fallback_reason": fallback_reason,
        })

    def feature_importance(self, top_k: int = 15) -> List[Tuple[str, int]]:
        if self.global_model is None:
            return []
        return self.global_model.feature_importance(top_k)

    def explain_prediction(self, X: pd.DataFrame, top_k: int = 5) -> List[Dict]:
        if self.global_model is None:
            return []
        return self.global_model.explain_prediction(X, top_k=top_k)

    def expert_summary(self) -> List[Tuple[str, Dict]]:
        return sorted(self.sector_stats.items(), key=lambda kv: kv[0])
