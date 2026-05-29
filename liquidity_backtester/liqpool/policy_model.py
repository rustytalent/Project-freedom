"""Phase 3D models for executable policy outcomes.

Policy labels answer a stricter question than pool taxonomies: "What happened
after this explicit execution policy generated a trade, net of costs?"  This
module trains small, research-only classifiers that estimate the probability a
policy trade wins.  Live gates do not consume these predictions yet; the first
job is to measure whether executable outcomes contain stable OOS structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def policy_feature_frame(labels: pd.DataFrame,
                         feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Build a causal-ish numeric matrix from policy label rows.

    Excludes realized outcomes, MAE/MFE, bars-to-touch, entry/exit prices, and
    barrier fields. The point is to avoid giving the policy model future
    information while still letting it learn from pool structure and timing.
    """
    if labels is None or labels.empty:
        return pd.DataFrame(columns=feature_names or [])

    df = labels.copy()
    idx = df.index

    def _series(col: str, default):
        # df.get(col, default) yields a SCALAR when the column is missing, which breaks the
        # downstream .astype/.dt/.fillna calls. Always return an index-aligned Series.
        return df[col] if col in df.columns else pd.Series(default, index=idx)

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(_series(col, 0.0), errors="coerce").fillna(0.0)

    out = pd.DataFrame(index=idx)
    out["score"] = _num("score")
    out["tf_count"] = _num("tf_count")
    out["side_high"] = (_series("side", "") == "sell").astype(float)
    out["direction_sign"] = _num("direction_sign")
    low = _num("pool_low")
    high = _num("pool_high")
    mid = _num("pool_mid")
    width = (high - low).clip(lower=0.0)
    out["pool_width"] = width
    out["pool_width_pct"] = width / mid.replace(0.0, np.nan).abs()
    out["pool_width_pct"] = out["pool_width_pct"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    available = pd.to_datetime(_series("available_at", pd.NaT), errors="coerce")
    formed = pd.to_datetime(_series("formed_at", pd.NaT), errors="coerce")
    lag_hours = (available - formed).dt.total_seconds() / 3600.0
    out["formation_lag_hours"] = lag_hours.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["available_hour"] = available.dt.hour.fillna(0).astype(float)
    out["available_dayofweek"] = available.dt.dayofweek.fillna(0).astype(float)

    cats = pd.DataFrame(index=df.index)
    for col in ("mode", "symbol", "sector", "factor", "direction"):
        if col in df.columns:
            dummies = pd.get_dummies(
                df[col].fillna("UNKNOWN").astype(str),
                prefix=col,
                dtype=float,
            )
            cats = pd.concat([cats, dummies], axis=1)
    out = pd.concat([out, cats], axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if feature_names is not None:
        out = out.reindex(columns=feature_names, fill_value=0.0)
    return out


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    if len(y) == 0:
        return {"n": 0, "brier": 0.0, "logloss": 0.0, "auc": None,
                "base_rate": 0.0, "mean_prediction": 0.0}
    actual = np.asarray(y, dtype=int)
    pred = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    auc = float(roc_auc_score(actual, pred)) if len(np.unique(actual)) == 2 else None
    return {
        "n": int(len(actual)),
        "brier": float(brier_score_loss(actual, pred)),
        "logloss": float(log_loss(actual, pred)),
        "auc": auc,
        "base_rate": float(actual.mean()),
        "mean_prediction": float(pred.mean()),
    }


def _chronological_split(frame: pd.DataFrame, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    ordered = frame.copy()
    ordered["_available_sort"] = pd.to_datetime(
        ordered.get("available_at"), errors="coerce",
    )
    ordered = ordered.sort_values(["_available_sort", "symbol", "pool_idx"], na_position="last")
    idx = ordered.index.to_numpy()
    n_val = max(1, int(round(len(idx) * val_frac)))
    n_val = min(n_val, max(1, len(idx) - 1))
    train_idx = idx[:-n_val]
    val_idx = idx[-n_val:]
    return train_idx, val_idx


@dataclass
class PolicyModeMetrics:
    mode: str
    status: str = "skipped"
    reason: str = ""
    train_n: int = 0
    val_n: int = 0
    oos_n: int = 0
    train_base_win: float = 0.0
    val_base_win: float = 0.0
    oos_base_win: float = 0.0
    has_calibration_holdout: bool = True
    val_brier: float = 0.0
    val_logloss: float = 0.0
    val_auc: Optional[float] = None
    oos_brier: float = 0.0
    oos_logloss: float = 0.0
    oos_auc: Optional[float] = None
    oos_mean_prediction: float = 0.0
    oos_mean_return_r: float = 0.0
    oos_top_decile_win: float = 0.0
    oos_top_decile_return_r: float = 0.0
    oos_top_decile_n: int = 0

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class PolicyOutcomeModel:
    mode: str
    feature_names: List[str] = field(default_factory=list)
    metrics: PolicyModeMetrics = field(init=False)

    def __post_init__(self) -> None:
        self.metrics = PolicyModeMetrics(mode=self.mode)

    def _trade_frame(self, labels: pd.DataFrame) -> pd.DataFrame:
        if labels is None or labels.empty:
            return pd.DataFrame()
        df = labels[
            (labels["mode"] == self.mode)
            & (labels["policy_target_trade_generated"] == 1)
            & labels["policy_target_win"].notna()
        ].copy()
        return df

    def fit(self, train_labels: pd.DataFrame, val_frac: float = 0.25,
            seed: int = 41, min_trades: int = 80,
            min_class_n: int = 15) -> "PolicyOutcomeModel":
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression

        frame = self._trade_frame(train_labels)
        if len(frame) < min_trades:
            raise ValueError(f"needs >= {min_trades} generated trades, got {len(frame)}")
        y_all = frame["policy_target_win"].astype(int)
        pos_n = int((y_all == 1).sum())
        neg_n = int((y_all == 0).sum())
        if pos_n < min_class_n or neg_n < min_class_n:
            raise ValueError(
                f"needs both classes >= {min_class_n}, got wins={pos_n} losses={neg_n}"
            )

        tr_idx, val_idx = _chronological_split(frame, val_frac=val_frac)
        y_tr = frame.loc[tr_idx, "policy_target_win"].astype(int).to_numpy()
        y_val = frame.loc[val_idx, "policy_target_win"].astype(int).to_numpy()
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            raise ValueError("chronological train/validation split lost a class")

        X = policy_feature_frame(frame)
        self.feature_names = list(X.columns)
        X_tr = X.loc[tr_idx].values

        # Split the chronological validation tail in time: the earlier half is used for early
        # stopping + isotonic calibration, the LATER (most recent) half is a pure holdout that
        # all reported val_* metrics come from. Fitting isotonic on the early-stopping rows and
        # then scoring those same rows is optimistic; the time-ordered holdout fixes that and
        # also mirrors live use (you face the most recent data). Falls back if a half loses a class.
        cut = len(val_idx) // 2
        es_idx, hold_idx = val_idx[:cut], val_idx[cut:]
        y_es = frame.loc[es_idx, "policy_target_win"].astype(int).to_numpy() if len(es_idx) else np.array([])
        y_hold = frame.loc[hold_idx, "policy_target_win"].astype(int).to_numpy() if len(hold_idx) else np.array([])
        has_holdout = (len(es_idx) > 0 and len(hold_idx) > 0
                       and len(np.unique(y_es)) == 2 and len(np.unique(y_hold)) == 2)
        if not has_holdout:
            es_idx, hold_idx = val_idx, val_idx
            y_es = y_hold = y_val
        X_es = X.loc[es_idx].values
        X_hold = X.loc[hold_idx].values

        params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.035,
            num_leaves=12,
            max_depth=4,
            min_data_in_leaf=40,
            feature_fraction=0.70,
            bagging_fraction=0.80,
            bagging_freq=5,
            lambda_l2=12.0,
            lambda_l1=1.0,
            min_gain_to_split=0.01,
            verbose=-1,
            seed=seed,
        )
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        des = lgb.Dataset(X_es, label=y_es, reference=dtr,
                          feature_name=self.feature_names)
        self._gbm = lgb.train(
            params, dtr, num_boost_round=300, valid_sets=[des],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        raw_es = self._gbm.predict(X_es, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(raw_es, y_es)
        hold_pred = self._iso.transform(
            self._gbm.predict(X_hold, num_iteration=self._gbm.best_iteration))
        mt = _binary_metrics(y_hold, hold_pred)
        self.metrics.status = "trained"
        self.metrics.train_n = int(len(tr_idx))
        self.metrics.val_n = int(len(hold_idx))
        self.metrics.has_calibration_holdout = bool(has_holdout)
        self.metrics.train_base_win = float(y_tr.mean())
        self.metrics.val_base_win = float(y_hold.mean())
        self.metrics.val_brier = mt["brier"]
        self.metrics.val_logloss = mt["logloss"]
        self.metrics.val_auc = mt["auc"]
        return self

    def predict_frame(self, labels: pd.DataFrame) -> np.ndarray:
        X = policy_feature_frame(labels, self.feature_names)
        raw = self._gbm.predict(X.values, num_iteration=self._gbm.best_iteration)
        return np.clip(self._iso.transform(raw), 0.02, 0.98)

    def evaluate(self, labels: pd.DataFrame) -> PolicyModeMetrics:
        frame = self._trade_frame(labels)
        if frame.empty:
            return self.metrics
        y = frame["policy_target_win"].astype(int).to_numpy()
        p = self.predict_frame(frame)
        mt = _binary_metrics(y, p)
        frame = frame.assign(policy_model_p_win=p)
        top_n = max(1, int(np.ceil(len(frame) * 0.10)))
        top = frame.sort_values("policy_model_p_win", ascending=False).head(top_n)

        self.metrics.oos_n = mt["n"]
        self.metrics.oos_base_win = mt["base_rate"]
        self.metrics.oos_brier = mt["brier"]
        self.metrics.oos_logloss = mt["logloss"]
        self.metrics.oos_auc = mt["auc"]
        self.metrics.oos_mean_prediction = mt["mean_prediction"]
        self.metrics.oos_mean_return_r = float(frame["policy_target_return_r"].mean())
        self.metrics.oos_top_decile_n = int(len(top))
        self.metrics.oos_top_decile_win = float(top["policy_target_win"].mean())
        self.metrics.oos_top_decile_return_r = float(top["policy_target_return_r"].mean())
        return self.metrics

    def predictions(self, labels: pd.DataFrame) -> pd.DataFrame:
        frame = self._trade_frame(labels)
        if frame.empty:
            return pd.DataFrame()
        pred = self.predict_frame(frame)
        keep = [
            "mode", "split", "symbol", "sector", "pool_idx", "direction",
            "factor", "policy_target_win", "policy_target_return_r",
        ]
        out = frame[[c for c in keep if c in frame.columns]].copy()
        out["policy_model_p_win"] = pred
        return out.sort_values("policy_model_p_win", ascending=False)

    def feature_importance(self, top_k: int = 30) -> List[Dict]:
        if not hasattr(self, "_gbm"):
            return []
        gains = self._gbm.feature_importance(importance_type="gain")
        pairs = sorted(zip(self.feature_names, gains), key=lambda kv: -kv[1])[:top_k]
        return [
            {"mode": self.mode, "feature": name, "importance_gain": float(gain)}
            for name, gain in pairs
        ]


@dataclass
class PolicyOutcomeModelSuite:
    models: Dict[str, PolicyOutcomeModel] = field(default_factory=dict)
    metrics: Dict[str, PolicyModeMetrics] = field(default_factory=dict)
    calibration_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    prediction_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def fit(self, train_labels: pd.DataFrame, oos_labels: pd.DataFrame,
            seed: int = 41, min_trades: int = 80) -> "PolicyOutcomeModelSuite":
        self.models.clear()
        self.metrics.clear()
        cal_rows: List[Dict] = []
        pred_rows: List[pd.DataFrame] = []
        modes = sorted(set(train_labels.get("mode", pd.Series(dtype=str)).dropna().astype(str)))
        for offset, mode in enumerate(modes):
            model = PolicyOutcomeModel(mode=mode)
            try:
                model.fit(train_labels, seed=seed + offset, min_trades=min_trades)
                model.evaluate(oos_labels)
                self.models[mode] = model
                self.metrics[mode] = model.metrics
                cal_rows.extend(self._calibration_rows(model, oos_labels))
                pred = model.predictions(oos_labels)
                if not pred.empty:
                    pred_rows.append(pred.head(200))
            except ValueError as exc:
                mt = PolicyModeMetrics(mode=mode, status="skipped", reason=str(exc))
                self.metrics[mode] = mt
                print(f"[policy_model] {mode} skipped: {exc}")
        self.calibration_table = pd.DataFrame(cal_rows)
        self.prediction_table = (
            pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
        )
        return self

    def _calibration_rows(self, model: PolicyOutcomeModel,
                          labels: pd.DataFrame, bins: int = 5) -> List[Dict]:
        frame = model._trade_frame(labels)
        if len(frame) < bins:
            return []
        y = frame["policy_target_win"].astype(int).to_numpy()
        p = model.predict_frame(frame)
        try:
            bucket = pd.qcut(p, q=min(bins, len(np.unique(p))), duplicates="drop")
        except ValueError:
            bucket = pd.cut(p, bins=min(bins, max(1, len(p))))
        rows = []
        tmp = pd.DataFrame({
            "prediction": p,
            "actual": y,
            "return_r": pd.to_numeric(
                frame["policy_target_return_r"], errors="coerce",
            ).fillna(0.0).to_numpy(),
            "bucket": bucket,
        })
        for b, bdf in tmp.groupby("bucket", observed=False):
            rows.append({
                "mode": model.mode,
                "bucket": str(b),
                "n": int(len(bdf)),
                "mean_prediction": float(bdf["prediction"].mean()),
                "actual_win_rate": float(bdf["actual"].mean()),
                "mean_return_r": float(bdf["return_r"].mean()),
                "calibration_error": float(bdf["prediction"].mean() - bdf["actual"].mean()),
            })
        return rows

    def report_frame(self) -> pd.DataFrame:
        return pd.DataFrame([mt.to_dict() for mt in self.metrics.values()])

    def feature_importance_frame(self, top_k: int = 30) -> pd.DataFrame:
        rows: List[Dict] = []
        for model in self.models.values():
            rows.extend(model.feature_importance(top_k=top_k))
        return pd.DataFrame(rows)
