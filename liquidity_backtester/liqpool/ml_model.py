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
class PoolRespectModel:
    """LightGBM + isotonic calibration + bucket shrinkage. predict() returns calibrated P(respect)."""
    feature_names: List[str] = field(default_factory=list)
    bucket_calib: Dict[Tuple[str, str], BucketCalib] = field(default_factory=dict)
    train_n: int = 0
    val_n: int = 0
    val_brier: float = 0.0
    val_logloss: float = 0.0
    val_auc: float = 0.0
    base_rate: float = 0.0
    # The underlying objects are stashed as attributes after fit().

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            buckets: Optional[List[Tuple[str, str]]] = None,
            val_frac: float = 0.25, seed: int = 17) -> "PoolRespectModel":
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        rng = np.random.default_rng(seed)
        n = len(X)
        if n < 20:
            raise ValueError(f"need at least 20 trainable pools, got {n}")

        self.feature_names = list(X.columns)
        self.base_rate = float(y.mean())
        # Stratified random split. We want both train and val to contain positives + negatives.
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
        X_val, y_val = X.iloc[val_idx].values, y[val_idx]

        params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.05,
            # Tighter than initial defaults — with only ~500 train pools, max_depth=4 with
            # min_data_in_leaf=8 overfit (train AUC 0.83 vs OOS 0.56). Tighter trees and stronger
            # L2 reduce the gap; accept a small drop in best-case AUC for better generalization.
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
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=self.feature_names)

        self._gbm = lgb.train(
            params, dtrain, num_boost_round=400, valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                       lgb.log_evaluation(0)],
        )

        # Isotonic calibration on validation predictions
        val_raw = self._gbm.predict(X_val, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(val_raw, y_val)
        val_calib = self._iso.transform(val_raw)

        self.train_n = int(len(train_idx))
        self.val_n = int(len(val_idx))
        self.val_brier = float(brier_score_loss(y_val, val_calib))
        self.val_logloss = float(log_loss(y_val, np.clip(val_calib, 1e-6, 1 - 1e-6)))
        # AUC needs both classes present (which our stratified split guarantees by construction).
        self.val_auc = float(roc_auc_score(y_val, val_calib))

        # Bucket recalibration is fit by the caller (walkforward) after predict on the full OOS set
        # since it needs (bucket, predicted, actual) — see fit_bucket_calib below.
        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._gbm.predict(X[self.feature_names].values,
                                num_iteration=self._gbm.best_iteration)
        return self._iso.transform(raw)

    def fit_bucket_calib(self, pools: List[Pool], y_true: np.ndarray, p_predicted: np.ndarray,
                         min_bucket_n: int = 10) -> None:
        """Build per-bucket pull weights from observed (pred, actual) on the OOS set.

        For each bucket: empirical_rate = mean(y_true_in_bucket). The "pull" applied at predict
        time is a sample-size-weighted shrinkage toward this rate:
            pull_weight = n / (n + 20)   (Wilson-style shrinkage; 20 = global-prior strength)
        Adjustment at predict time:
            calibrated = (1 - pull) * global_prediction + pull * empirical_rate
        Small buckets (n<min_bucket_n) get no per-bucket adjustment.
        """
        groups: Dict[Tuple[str, str], List[Tuple[float, int]]] = defaultdict(list)
        for p, pred, y in zip(pools, p_predicted, y_true):
            key = (_tf_bucket(len(set(p.tfs))), _headline_factor(p))
            groups[key].append((float(pred), int(y)))
        self.bucket_calib.clear()
        for key, items in groups.items():
            if len(items) < min_bucket_n:
                continue
            arr_y = np.array([y for _, y in items])
            emp = float(arr_y.mean())
            n = len(items)
            pull = n / (n + 20.0)
            self.bucket_calib[key] = BucketCalib(n_oos=n, empirical_rate=emp, pull_weight=pull)

    def predict(self, X: pd.DataFrame, pools: Optional[List[Pool]] = None) -> np.ndarray:
        """Returns final calibrated P(respect). If pools given AND bucket_calib has been fit,
        applies bucket-level shrinkage on top of global calibration."""
        base = self.predict_raw(X)
        if pools is None or not self.bucket_calib:
            return np.clip(base, 0.0, 1.0)
        adj = base.copy()
        for i, p in enumerate(pools):
            key = (_tf_bucket(len(set(p.tfs))), _headline_factor(p))
            bc = self.bucket_calib.get(key)
            if bc is not None:
                adj[i] = (1.0 - bc.pull_weight) * base[i] + bc.pull_weight * bc.empirical_rate
        return np.clip(adj, 0.0, 1.0)

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
