"""Phase 3 post-touch reaction models.

The quality model is a pre-touch prior and the proximity model is a reachability
model.  This module starts the separate post-touch layer: after price has reached
a pool and a small confirmation window has printed, estimate whether the event
looks like a strict reaction, a reclaim success, or a break continuation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


TARGET_COLUMNS = {
    "strict_reaction": "strict_respect_label",
    "reclaim_success": "reclaim_success_label",
    "break_continuation": "break_continuation_label",
}


def _tf_count_from_row(row: pd.Series) -> float:
    tfs = row.get("tfs")
    if isinstance(tfs, str) and tfs:
        return float(len([x for x in tfs.split("+") if x]))
    bucket = str(row.get("tf_bucket", ""))
    if bucket.endswith("+"):
        return float(bucket[:-1] or 4)
    try:
        return float(bucket)
    except Exception:
        return 0.0


def reaction_feature_frame(events: pd.DataFrame,
                           feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Build a numeric model matrix from post-touch event rows.

    The intentionally excluded columns are labels/outcomes and full-horizon MAE/MFE fields.
    The `pt_*` columns are confirmation-window features produced after touch.
    """
    if events is None or events.empty:
        return pd.DataFrame(columns=feature_names or [])

    df = events.copy()

    def _num(col: str) -> pd.Series:
        # df.get(col, default) returns a SCALAR default when the column is missing, and
        # pd.to_numeric(scalar).fillna(...) raises. Always work with an index-aligned Series.
        s = df[col] if col in df.columns else pd.Series(0.0, index=df.index)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    out = pd.DataFrame(index=df.index)
    out["score"] = _num("score")
    out["width"] = _num("width")
    out["n_contributors"] = _num("n_contributors")
    out["bars_to_touch_log1p"] = np.log1p(_num("bars_to_touch").clip(lower=0.0))
    out["side_high"] = (df.get("side", "") == "high").astype(float)
    out["tf_count"] = df.apply(_tf_count_from_row, axis=1)

    for col in sorted(c for c in df.columns if str(c).startswith("pt_")):
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    cats = pd.DataFrame(index=df.index)
    for col in ("sector", "headline_factor", "tf_bucket"):
        if col in df.columns:
            dummies = pd.get_dummies(df[col].fillna("UNKNOWN").astype(str),
                                     prefix=col, dtype=float)
            cats = pd.concat([cats, dummies], axis=1)
    out = pd.concat([out, cats], axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if feature_names is not None:
        out = out.reindex(columns=feature_names, fill_value=0.0)
    return out


@dataclass
class ReactionTargetMetrics:
    target: str
    train_n: int = 0
    val_n: int = 0
    oos_n: int = 0
    base_rate: float = 0.0
    val_brier: float = 0.0
    val_logloss: float = 0.0
    val_auc: Optional[float] = None
    oos_brier: float = 0.0
    oos_logloss: float = 0.0
    oos_auc: Optional[float] = None
    oos_top_decile_rate: float = 0.0
    oos_mean_prediction: float = 0.0
    has_calibration_holdout: bool = True

    def to_dict(self) -> Dict:
        return {
            "target": self.target,
            "train_n": self.train_n,
            "val_n": self.val_n,
            "oos_n": self.oos_n,
            "base_rate": self.base_rate,
            "has_calibration_holdout": self.has_calibration_holdout,
            "val_brier": self.val_brier,
            "val_logloss": self.val_logloss,
            "val_auc": self.val_auc,
            "oos_brier": self.oos_brier,
            "oos_logloss": self.oos_logloss,
            "oos_auc": self.oos_auc,
            "oos_top_decile_rate": self.oos_top_decile_rate,
            "oos_mean_prediction": self.oos_mean_prediction,
        }


def _binary_metrics(target: str, y: np.ndarray, p: np.ndarray) -> Dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    if len(y) == 0:
        return {
            "n": 0, "brier": 0.0, "logloss": 0.0, "auc": None,
            "base_rate": 0.0, "mean_prediction": 0.0, "top_decile_rate": 0.0,
        }
    pred = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    actual = np.asarray(y, dtype=int)
    auc = float(roc_auc_score(actual, pred)) if len(np.unique(actual)) == 2 else None
    top_n = max(1, int(np.ceil(len(actual) * 0.10)))
    top = np.argsort(-pred)[:top_n]
    return {
        "target": target,
        "n": int(len(actual)),
        "brier": float(brier_score_loss(actual, pred)),
        "logloss": float(log_loss(actual, pred)),
        "auc": auc,
        "base_rate": float(actual.mean()),
        "mean_prediction": float(pred.mean()),
        "top_decile_rate": float(actual[top].mean()),
    }


def _stratified_split(y: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_vp = max(1, int(round(len(pos) * val_frac)))
    n_vn = max(1, int(round(len(neg) * val_frac)))
    val_idx = np.concatenate([pos[:n_vp], neg[:n_vn]])
    tr_idx = np.concatenate([pos[n_vp:], neg[n_vn:]])
    rng.shuffle(val_idx)
    rng.shuffle(tr_idx)
    return tr_idx, val_idx


def _stratified_halves(idx: np.ndarray, y: np.ndarray, seed: int
                       ) -> tuple[np.ndarray, np.ndarray, bool]:
    """Split `idx` into class-stratified (early_stop, holdout, ok) halves. Returns
    (idx, idx, False) when a clean both-classes split isn't possible (caller then calibrates
    and reports on the same rows)."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    yv = y[idx]
    pos, neg = idx[yv == 1], idx[yv == 0]
    if len(pos) < 2 or len(neg) < 2:
        return idx, idx, False
    rng.shuffle(pos)
    rng.shuffle(neg)
    ep, en = max(1, len(pos) // 2), max(1, len(neg) // 2)
    es = np.concatenate([pos[:ep], neg[:en]])
    hold = np.concatenate([pos[ep:], neg[en:]])
    if len(hold) == 0 or len(np.unique(y[hold])) < 2 or len(np.unique(y[es])) < 2:
        return idx, idx, False
    rng.shuffle(es)
    rng.shuffle(hold)
    return es, hold, True


@dataclass
class ReactionBinaryModel:
    target: str
    label_col: str
    feature_names: List[str] = field(default_factory=list)
    metrics: ReactionTargetMetrics = field(init=False)

    def __post_init__(self) -> None:
        self.metrics = ReactionTargetMetrics(target=self.target)

    def fit(self, train_events: pd.DataFrame, val_frac: float = 0.25,
            seed: int = 31, min_rows: int = 100,
            min_class_n: int = 20) -> "ReactionBinaryModel":
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression

        frame = train_events[train_events[self.label_col].notna()].copy()
        if len(frame) < min_rows:
            raise ValueError(f"{self.target} needs >= {min_rows} labelled rows, got {len(frame)}")
        y = frame[self.label_col].astype(int).to_numpy()
        pos_n = int((y == 1).sum())
        neg_n = int((y == 0).sum())
        if pos_n < min_class_n or neg_n < min_class_n:
            raise ValueError(
                f"{self.target} needs both classes >= {min_class_n}, got pos={pos_n} neg={neg_n}"
            )

        X = reaction_feature_frame(frame)
        self.feature_names = list(X.columns)
        tr_idx, val_idx = _stratified_split(y, val_frac=val_frac, seed=seed)
        # Hold out half of the validation fold: early stopping + isotonic are fit on the es half;
        # reported metrics come from the disjoint holdout half (honest calibration metrics).
        es_idx, hold_idx, has_holdout = _stratified_halves(val_idx, y, seed=seed + 1)
        X_tr, y_tr = X.iloc[tr_idx].values, y[tr_idx]
        X_es, y_es = X.iloc[es_idx].values, y[es_idx]
        X_hold, y_hold = X.iloc[hold_idx].values, y[hold_idx]

        params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.04,
            num_leaves=15,
            max_depth=4,
            min_data_in_leaf=40,
            feature_fraction=0.75,
            bagging_fraction=0.80,
            bagging_freq=5,
            lambda_l2=10.0,
            lambda_l1=1.0,
            min_gain_to_split=0.01,
            verbose=-1,
            seed=seed,
        )
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        des = lgb.Dataset(X_es, label=y_es, reference=dtr,
                          feature_name=self.feature_names)
        self._gbm = lgb.train(
            params, dtr, num_boost_round=350, valid_sets=[des],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=35, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        raw_es = self._gbm.predict(X_es, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(raw_es, y_es)
        hold_pred = self._iso.transform(
            self._gbm.predict(X_hold, num_iteration=self._gbm.best_iteration))
        mt = _binary_metrics(self.target, y_hold, hold_pred)
        self.metrics.train_n = int(len(tr_idx))
        self.metrics.val_n = int(len(hold_idx))
        self.metrics.base_rate = float(y.mean())
        self.metrics.has_calibration_holdout = bool(has_holdout)
        self.metrics.val_brier = mt["brier"]
        self.metrics.val_logloss = mt["logloss"]
        self.metrics.val_auc = mt["auc"]
        return self

    def predict_frame(self, events: pd.DataFrame) -> np.ndarray:
        X = reaction_feature_frame(events, self.feature_names)
        raw = self._gbm.predict(X.values, num_iteration=self._gbm.best_iteration)
        return np.clip(self._iso.transform(raw), 0.02, 0.98)

    def evaluate(self, events: pd.DataFrame) -> ReactionTargetMetrics:
        frame = events[events[self.label_col].notna()].copy()
        if frame.empty:
            return self.metrics
        y = frame[self.label_col].astype(int).to_numpy()
        p = self.predict_frame(frame)
        mt = _binary_metrics(self.target, y, p)
        self.metrics.oos_n = mt["n"]
        self.metrics.oos_brier = mt["brier"]
        self.metrics.oos_logloss = mt["logloss"]
        self.metrics.oos_auc = mt["auc"]
        self.metrics.oos_top_decile_rate = mt["top_decile_rate"]
        self.metrics.oos_mean_prediction = mt["mean_prediction"]
        return self.metrics

    def feature_importance(self, top_k: int = 30) -> List[Dict]:
        if not hasattr(self, "_gbm"):
            return []
        gains = self._gbm.feature_importance(importance_type="gain")
        pairs = sorted(zip(self.feature_names, gains), key=lambda kv: -kv[1])[:top_k]
        return [
            {"target": self.target, "feature": name, "importance_gain": float(gain)}
            for name, gain in pairs
        ]


@dataclass
class ReactionModelSuite:
    models: Dict[str, ReactionBinaryModel] = field(default_factory=dict)
    metrics: Dict[str, ReactionTargetMetrics] = field(default_factory=dict)
    calibration_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def fit(self, train_events: pd.DataFrame, oos_events: Optional[pd.DataFrame] = None,
            seed: int = 31) -> "ReactionModelSuite":
        self.models.clear()
        self.metrics.clear()
        cal_rows = []

        for offset, (target, label_col) in enumerate(TARGET_COLUMNS.items()):
            if label_col not in train_events.columns:
                continue
            model = ReactionBinaryModel(target=target, label_col=label_col)
            try:
                model.fit(train_events, seed=seed + offset)
                if oos_events is not None and not oos_events.empty and label_col in oos_events:
                    model.evaluate(oos_events)
                    cal_rows.extend(self._calibration_rows(model, oos_events))
                self.models[target] = model
                self.metrics[target] = model.metrics
            except ValueError as exc:
                self.metrics[target] = ReactionTargetMetrics(target=target)
                self.metrics[target].base_rate = float("nan")
                self.metrics[target].oos_mean_prediction = float("nan")
                self.metrics[target].val_logloss = float("nan")
                self.metrics[target].oos_logloss = float("nan")
                # Keep the reason in the table without adding another dataclass field.
                self.metrics[target].val_auc = None
                print(f"[reaction_model] {target} skipped: {exc}")

        self.calibration_table = pd.DataFrame(cal_rows)
        return self

    def _calibration_rows(self, model: ReactionBinaryModel,
                          events: pd.DataFrame, bins: int = 5) -> List[Dict]:
        frame = events[events[model.label_col].notna()].copy()
        if len(frame) < bins:
            return []
        y = frame[model.label_col].astype(int).to_numpy()
        p = model.predict_frame(frame)
        try:
            bucket = pd.qcut(p, q=min(bins, len(np.unique(p))), duplicates="drop")
        except ValueError:
            bucket = pd.cut(p, bins=min(bins, max(1, len(p))))
        rows = []
        tmp = pd.DataFrame({"prediction": p, "actual": y, "bucket": bucket})
        for b, bdf in tmp.groupby("bucket", observed=False):
            rows.append({
                "target": model.target,
                "bucket": str(b),
                "n": int(len(bdf)),
                "mean_prediction": float(bdf["prediction"].mean()),
                "actual_rate": float(bdf["actual"].mean()),
                "calibration_error": float(bdf["prediction"].mean() - bdf["actual"].mean()),
            })
        return rows

    def predict_event_frame(self, events: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=events.index)
        for target, model in self.models.items():
            out[f"p_{target}"] = model.predict_frame(events)
        if "p_strict_reaction" in out:
            out["p_reaction_model"] = out["p_strict_reaction"]
        elif "p_break_continuation" in out:
            out["p_reaction_model"] = 1.0 - out["p_break_continuation"]
        return out

    def report_frame(self) -> pd.DataFrame:
        return pd.DataFrame([mt.to_dict() for mt in self.metrics.values()])

    def feature_importance_frame(self, top_k: int = 30) -> pd.DataFrame:
        rows: List[Dict] = []
        for model in self.models.values():
            rows.extend(model.feature_importance(top_k=top_k))
        return pd.DataFrame(rows)
