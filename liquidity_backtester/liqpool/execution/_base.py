"""Shared LightGBM-regressor wrapper for the Stream K execution models.

Three reasons for the wrapper rather than direct LightGBM calls:

  1. Each model has the same fit / predict surface; pulling the
     boilerplate into one place keeps the three model files focused
     on their feature engineering.
  2. We want a deterministic NO-OP fallback when the model wasn't
     fit (or the data was too thin to train). Without it, V3 would
     crash or silently emit nonsense the first time it runs.
  3. The wrapper makes save/load to JSON + parquet cheap so the
     trained models can ship alongside the model bundle.

Design constraints:
  * No state at construction time beyond the config. Fit returns
    self.
  * predict() always returns finite floats; if untrained or input
    is malformed, returns the configured `fallback`.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class LGBMConfig:
    """Hyperparameters shared across the execution models."""
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    num_leaves: int = 15
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    seed: int = 42
    early_stopping_rounds: int = 30
    verbose: int = -1


class LGBMRegressorWrapper:
    """Thin LightGBM regressor with a deterministic no-op fallback."""

    def __init__(
        self,
        feature_names: Sequence[str],
        fallback: float,
        config: Optional[LGBMConfig] = None,
        objective: str = "regression",
    ) -> None:
        self.feature_names: List[str] = list(feature_names)
        self.fallback: float = float(fallback)
        self.config: LGBMConfig = config or LGBMConfig()
        self.objective: str = objective
        self._booster = None  # set by fit()
        self._trained_n: int = 0
        self._oos_metric: Optional[float] = None

    # -- fit --------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        val_frac: float = 0.2,
        weight: Optional[np.ndarray] = None,
    ) -> "LGBMRegressorWrapper":
        """Train with a chronological holdout (last `val_frac` rows
        used for early stopping). LightGBM is imported lazily so
        codepaths that never train don't pay the import cost."""
        import lightgbm as lgb

        y_arr = np.asarray(y, dtype=float)
        # Require finite TARGETS only. NaN features are fine —
        # LightGBM handles missing values natively, and several
        # callers (regret model especially) have wide feature sets
        # where most context keys are absent per event kind. Rows
        # that are ENTIRELY NaN carry no signal and are dropped.
        mask = np.isfinite(y_arr) & X.notna().any(axis=1).to_numpy()
        X, y_arr = X.loc[mask], y_arr[mask]
        if weight is not None:
            weight = np.asarray(weight, dtype=float)[mask]
        n = len(X)
        if n < max(50, self.config.min_data_in_leaf * 2):
            # Not enough data — leave booster as None so predict()
            # returns the fallback. Document the gap on the wrapper.
            self._trained_n = n
            return self
        # Chronological split: assume the caller passed X already sorted
        # by time so the last val_frac rows are the holdout.
        n_val = max(20, int(n * val_frac))
        n_train = n - n_val
        if n_train < 50:
            self._trained_n = n
            return self
        Xtr, ytr = X.iloc[:n_train], y_arr[:n_train]
        Xva, yva = X.iloc[n_train:], y_arr[n_train:]
        wtr = weight[:n_train] if weight is not None else None
        wva = weight[n_train:] if weight is not None else None

        params: Dict[str, Any] = {
            "objective": self.objective,
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "num_leaves": self.config.num_leaves,
            "min_data_in_leaf": self.config.min_data_in_leaf,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "bagging_freq": self.config.bagging_freq,
            "seed": self.config.seed,
            "verbose": self.config.verbose,
        }
        dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
        dva = lgb.Dataset(Xva, label=yva, weight=wva, reference=dtr)
        self._booster = lgb.train(
            params, dtr,
            num_boost_round=self.config.n_estimators,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(
                stopping_rounds=self.config.early_stopping_rounds,
                verbose=False,
            )],
        )
        self._trained_n = n
        try:
            pred_val = self._booster.predict(Xva)
            self._oos_metric = float(np.sqrt(np.mean((pred_val - yva) ** 2)))
        except Exception:
            self._oos_metric = None
        return self

    # -- predict ----------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Always returns finite floats. Untrained or malformed input
        falls back to ``self.fallback`` (broadcast to the right shape).
        """
        if self._booster is None:
            return np.full(len(X), self.fallback)
        # Reorder columns to the trained feature names; missing
        # columns get filled with NaN and LightGBM handles that.
        cols = [c for c in self.feature_names if c in X.columns]
        if not cols:
            return np.full(len(X), self.fallback)
        sub = X[cols].copy()
        # Add any missing trained columns as NaN.
        for c in self.feature_names:
            if c not in sub.columns:
                sub[c] = np.nan
        sub = sub[self.feature_names]
        try:
            raw = np.asarray(self._booster.predict(sub), dtype=float)
        except Exception:
            return np.full(len(X), self.fallback)
        # Guard against NaN/Inf leaking out.
        raw = np.where(np.isfinite(raw), raw, self.fallback)
        return raw

    def predict_one(self, row: Dict[str, Any]) -> float:
        """Convenience wrapper for single-row inference (the V3 hot path)."""
        df = pd.DataFrame([row])
        return float(self.predict(df)[0])

    # -- persistence -------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist as (LightGBM model text + JSON metadata sidecar)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._booster is None:
            # Save a marker so load() can detect "trained-but-empty".
            path.with_suffix(".meta.json").write_text(json.dumps({
                "feature_names": self.feature_names,
                "fallback": self.fallback,
                "objective": self.objective,
                "trained_n": self._trained_n,
                "oos_metric": self._oos_metric,
                "booster_present": False,
            }))
            return
        self._booster.save_model(str(path))
        path.with_suffix(".meta.json").write_text(json.dumps({
            "feature_names": self.feature_names,
            "fallback": self.fallback,
            "objective": self.objective,
            "trained_n": self._trained_n,
            "oos_metric": self._oos_metric,
            "booster_present": True,
        }))

    @classmethod
    def load(cls, path: Path) -> "LGBMRegressorWrapper":
        import lightgbm as lgb
        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        obj = cls(
            feature_names=meta["feature_names"],
            fallback=meta["fallback"],
            objective=meta.get("objective", "regression"),
        )
        obj._trained_n = int(meta.get("trained_n", 0))
        obj._oos_metric = meta.get("oos_metric")
        if meta.get("booster_present"):
            obj._booster = lgb.Booster(model_file=str(path))
        return obj

    # -- introspection ----------------------------------------------

    @property
    def trained_n(self) -> int:
        return self._trained_n

    @property
    def oos_metric(self) -> Optional[float]:
        return self._oos_metric

    @property
    def is_fitted(self) -> bool:
        return self._booster is not None
