"""Options Expected-Return Model suite (Stream D.3).

One LightGBM Huber regressor per ``(side × tenor × ToD)`` bucket,
trained against ``realized_premium_atr_units_60min`` (the D.2 label).
Structural priors come from ``model_config.py`` — interaction
constraints encode the cascade DAG; monotone constraints pin known
directional relationships per side.

Reporting per head (after fit):

  - n_train, n_oos
  - mean_predicted_R_oos, mean_realized_R_oos
  - spearman(predicted_R, realized_R)_oos
  - top_decile_mean_realized_R_oos
  - top_decile_realized_hit_rate (frac with realized > 0)
  - calibration_table: 10 deciles × (mean_predicted, mean_realized, n)
  - feature_importance_top_20
  - sub-slice calibration error by IV-decile and DTE-bucket

The model file is the input to Gate 1 of the methodology:
  >= one head has Spearman > 0.10 AND top_decile_realized_R > 0
  AND calibration error <= 0.15 ATR across VIX deciles.

If Gate 1 fails the entire options vertical pauses; we re-evaluate
features (D.1) before any commercial step. The trainer reports the
gate as a structured boolean so the brief can show "pending model
edge" without leaking which head failed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .model_config import (
    CATEGORICAL_FEATURE_COLUMNS,
    EARLY_STOPPING_ROUNDS,
    LIGHTGBM_PARAMS,
    MONEYNESS_BUCKET_LEVELS,
    NUMERIC_FEATURE_COLUMNS,
    NUM_BOOST_ROUND,
    build_constraints,
    derive_tenor,
    expand_categorical_features,
)


# ---------------------------------------------------------------------------
# Bucket identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BucketKey:
    """Identifies one head's slice of the OOS frame."""
    side: str
    tenor: str
    tod: str

    def as_label(self) -> str:
        return f"{self.side}/{self.tenor}/{self.tod}"


# ---------------------------------------------------------------------------
# Reporting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CalibrationRow:
    decile: int
    n: int
    mean_predicted: float
    mean_realized: float
    calibration_error: float


@dataclass
class OptionsHeadMetrics:
    bucket: BucketKey
    status: str = "skipped"
    reason: str = ""
    n_train: int = 0
    n_oos: int = 0
    mean_predicted_oos: float = 0.0
    mean_realized_oos: float = 0.0
    spearman_pred_vs_real: float = 0.0
    top_decile_mean_realized: float = 0.0
    top_decile_hit_rate: float = 0.0
    calibration_table: List[CalibrationRow] = field(default_factory=list)
    feature_importance_top_20: List[Tuple[str, float]] = field(
        default_factory=list)
    sub_slice_calibration_by_iv_decile: List[CalibrationRow] = field(
        default_factory=list)
    sub_slice_calibration_by_dte_bucket: Dict[str, float] = field(
        default_factory=dict)


@dataclass
class SuiteSummary:
    head_count: int = 0
    fit_count: int = 0
    skipped_count: int = 0
    gate1_pass: bool = False
    gate1_winning_buckets: List[str] = field(default_factory=list)
    per_head: Dict[str, OptionsHeadMetrics] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers — pure, testable
# ---------------------------------------------------------------------------

def _winsorize(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return series.clip(lower=lo, upper=hi)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation; tolerates NaN by dropping pairwise."""
    mask = x.notna() & y.notna()
    xs, ys = x[mask], y[mask]
    if len(xs) < 5:
        return 0.0
    rx = xs.rank()
    ry = ys.rank()
    if rx.std(ddof=0) == 0 or ry.std(ddof=0) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _decile_table(predicted: pd.Series, realized: pd.Series,
                  n_deciles: int = 10) -> List[CalibrationRow]:
    """Decile reliability table: bucket by predicted, report mean
    predicted vs mean realized per decile.
    """
    df = pd.DataFrame({"p": predicted, "r": realized}).dropna()
    if df.empty:
        return []
    # qcut handles ties by spreading; pin to n_deciles or fewer if the
    # distribution is too tight.
    try:
        df["decile"] = pd.qcut(df["p"], q=n_deciles, labels=False,
                                duplicates="drop")
    except ValueError:
        # Predicted is degenerate (all the same value). Single bucket.
        df["decile"] = 0
    rows: List[CalibrationRow] = []
    for d, g in df.groupby("decile"):
        rows.append(CalibrationRow(
            decile=int(d),
            n=int(len(g)),
            mean_predicted=float(g["p"].mean()),
            mean_realized=float(g["r"].mean()),
            calibration_error=float(g["p"].mean() - g["r"].mean()),
        ))
    return rows


def _build_feature_matrix(frame: pd.DataFrame,
                          ) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot expand categoricals and select the final feature
    matrix. The column order is deterministic — driven by
    :func:`expand_categorical_features`, not by the frame's column
    layout — so train and predict frames produce comparable vectors.
    """
    feature_names = expand_categorical_features(frame.columns)
    X = pd.DataFrame(index=frame.index)

    for col in feature_names:
        if col.startswith("moneyness_bucket__"):
            level = col[len("moneyness_bucket__"):]
            if "moneyness_bucket" in frame.columns:
                X[col] = (frame["moneyness_bucket"] == level).astype(int)
            else:
                X[col] = 0
        elif col == "weekly_expiry_flag_int":
            if "weekly_expiry_flag" in frame.columns:
                X[col] = frame["weekly_expiry_flag"].fillna(False).astype(int)
            else:
                X[col] = 0
        else:
            if col in frame.columns:
                X[col] = pd.to_numeric(frame[col], errors="coerce")
            else:
                X[col] = np.nan
    return X, feature_names


def _chronological_split(frame: pd.DataFrame, val_frac: float
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Split the frame by ``bar_ts`` so val is strictly later than train."""
    ordered = frame.sort_values("bar_ts")
    idx = ordered.index.to_numpy()
    n_val = max(1, int(round(len(idx) * val_frac)))
    n_val = min(n_val, max(1, len(idx) - 1))
    return idx[:-n_val], idx[-n_val:]


def _apply_embargo(train_idx: np.ndarray, val_idx: np.ndarray,
                   frame: pd.DataFrame, embargo_bars: int) -> np.ndarray:
    """Drop train rows whose bar_ts falls within `embargo_bars` of the
    earliest val bar. Prevents label window overlap between train and
    val for 60-min-horizon labels at 5-min cadence (embargo = 12 bars
    plus a 1-bar safety = 13 by default).
    """
    if len(val_idx) == 0:
        return train_idx
    val_start = frame.loc[val_idx, "bar_ts"].min()
    cutoff = val_start - pd.Timedelta(minutes=5 * embargo_bars)
    keep = frame.loc[train_idx, "bar_ts"] <= cutoff
    return train_idx[keep.values]


# ---------------------------------------------------------------------------
# OptionsExpectedReturnModel — one bucket
# ---------------------------------------------------------------------------

@dataclass
class OptionsExpectedReturnModel:
    bucket: BucketKey
    label_clip: Tuple[float, float] = (-3.0, +5.0)
    embargo_bars: int = 13
    val_frac: float = 0.25
    min_trades: int = 200
    seed: int = 41
    metrics: OptionsHeadMetrics = field(init=False)

    feature_names: List[str] = field(default_factory=list, init=False)
    _gbm: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.metrics = OptionsHeadMetrics(bucket=self.bucket)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, frame: pd.DataFrame) -> "OptionsExpectedReturnModel":
        """Fit one head on the bucket's slice. The input frame must be
        pre-filtered to this bucket — the SUITE does that routing.
        """
        if frame is None or frame.empty:
            self.metrics.status = "skipped"
            self.metrics.reason = "empty_frame"
            return self
        labelled = frame[frame["label_valid"] == True].copy()       # noqa: E712
        labelled = labelled[
            labelled["realized_premium_atr_units_60min"].notna()
        ]
        if len(labelled) < self.min_trades:
            self.metrics.status = "skipped"
            self.metrics.reason = (
                f"only {len(labelled)} valid trades "
                f"(< min_trades={self.min_trades})"
            )
            self.metrics.n_train = len(labelled)
            return self

        # Chronological split + embargo.
        tr_idx, val_idx = _chronological_split(labelled, self.val_frac)
        tr_idx = _apply_embargo(tr_idx, val_idx, labelled, self.embargo_bars)
        if len(tr_idx) < self.min_trades // 2 or len(val_idx) < 10:
            self.metrics.status = "skipped"
            self.metrics.reason = (
                f"insufficient after embargo: "
                f"train={len(tr_idx)} val={len(val_idx)}"
            )
            return self

        # Build features + winsorize the training target only.
        X, feature_names = _build_feature_matrix(labelled)
        self.feature_names = feature_names

        y_full = labelled["realized_premium_atr_units_60min"]
        y_train = _winsorize(y_full.loc[tr_idx],
                              self.label_clip[0], self.label_clip[1])
        y_val = y_full.loc[val_idx]      # reported unclipped

        interaction, monotone = build_constraints(
            feature_names, self.bucket.side)

        params = dict(LIGHTGBM_PARAMS)
        params["interaction_constraints"] = interaction
        params["monotone_constraints"] = monotone
        params["seed"] = self.seed

        import lightgbm as lgb            # lazy import; codebase convention
        dtr = lgb.Dataset(X.loc[tr_idx].values, label=y_train.values,
                           feature_name=feature_names)
        dval = lgb.Dataset(X.loc[val_idx].values, label=y_val.values,
                            reference=dtr, feature_name=feature_names)
        callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS,
                                        verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gbm = lgb.train(
                params,
                dtr,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[dval],
                callbacks=callbacks,
            )

        # ---- Metrics on the OOS slice (= val slice for this head) ---
        pred_val = pd.Series(
            self._gbm.predict(X.loc[val_idx].values),
            index=val_idx,
        )
        real_val = y_val

        self.metrics.status = "fit"
        self.metrics.n_train = int(len(tr_idx))
        self.metrics.n_oos = int(len(val_idx))
        self.metrics.mean_predicted_oos = float(pred_val.mean())
        self.metrics.mean_realized_oos = float(real_val.mean())
        self.metrics.spearman_pred_vs_real = _spearman(pred_val, real_val)

        top_decile_cutoff = pred_val.quantile(0.9)
        top_mask = pred_val >= top_decile_cutoff
        if top_mask.sum() > 0:
            top_real = real_val[top_mask]
            self.metrics.top_decile_mean_realized = float(top_real.mean())
            self.metrics.top_decile_hit_rate = float((top_real > 0).mean())

        self.metrics.calibration_table = _decile_table(pred_val, real_val)
        self._populate_subslice_calibration(
            X, val_idx, pred_val, real_val, labelled)
        self._populate_feature_importance(feature_names)

        return self

    def predict_frame(self, frame: pd.DataFrame) -> pd.Series:
        """Predict realized_premium_atr_units_60min for each row."""
        if self._gbm is None:
            return pd.Series(np.nan, index=frame.index)
        X, _ = _build_feature_matrix(frame)
        return pd.Series(self._gbm.predict(X.values), index=frame.index)

    # ------------------------------------------------------------------
    # Internal helpers for metrics
    # ------------------------------------------------------------------

    def _populate_subslice_calibration(self, X: pd.DataFrame,
                                        val_idx: np.ndarray,
                                        pred: pd.Series,
                                        real: pd.Series,
                                        labelled: pd.DataFrame) -> None:
        # IV-decile sub-slice (uses iv_percentile_60d when available).
        if "iv_percentile_60d" in labelled.columns:
            iv = labelled.loc[val_idx, "iv_percentile_60d"]
            df = pd.DataFrame({"iv": iv, "p": pred, "r": real}).dropna()
            if not df.empty:
                try:
                    df["d"] = pd.qcut(df["iv"], q=10, labels=False,
                                       duplicates="drop")
                except ValueError:
                    df["d"] = 0
                rows: List[CalibrationRow] = []
                for d, g in df.groupby("d"):
                    rows.append(CalibrationRow(
                        decile=int(d), n=int(len(g)),
                        mean_predicted=float(g["p"].mean()),
                        mean_realized=float(g["r"].mean()),
                        calibration_error=float(g["p"].mean() - g["r"].mean()),
                    ))
                self.metrics.sub_slice_calibration_by_iv_decile = rows

        # DTE-bucket sub-slice (1-3 / 4-7 / 8-14).
        if "dte_trading_days" in labelled.columns:
            dte = labelled.loc[val_idx, "dte_trading_days"]
            df = pd.DataFrame({"dte": dte, "p": pred, "r": real}).dropna()
            if not df.empty:
                buckets = pd.cut(df["dte"], bins=[-0.1, 3.5, 7.5, 14.5,
                                                   1e9],
                                 labels=["1-3", "4-7", "8-14", "15+"])
                df["b"] = buckets
                err_by_bucket: Dict[str, float] = {}
                for b, g in df.groupby("b", observed=False):
                    if len(g) == 0:
                        continue
                    err_by_bucket[str(b)] = float(
                        g["p"].mean() - g["r"].mean())
                self.metrics.sub_slice_calibration_by_dte_bucket = err_by_bucket

    def _populate_feature_importance(self, feature_names: List[str]) -> None:
        if self._gbm is None:
            return
        gains = self._gbm.feature_importance(importance_type="gain")
        pairs = sorted(zip(feature_names, gains),
                       key=lambda kv: kv[1], reverse=True)
        self.metrics.feature_importance_top_20 = [
            (name, float(g)) for name, g in pairs[:20]
        ]


# ---------------------------------------------------------------------------
# Suite — routes the frame into buckets, fits one head per bucket
# ---------------------------------------------------------------------------

@dataclass
class OptionsExpectedReturnModelSuite:
    label_clip: Tuple[float, float] = (-3.0, +5.0)
    embargo_bars: int = 13
    val_frac: float = 0.25
    min_trades: int = 200
    seed: int = 41
    heads: Dict[BucketKey, OptionsExpectedReturnModel] = field(
        default_factory=dict)

    def _bucket_label(self, frame: pd.DataFrame) -> pd.Series:
        """Per-row bucket key as (side, tenor, ToD)."""
        tenor = frame["dte_trading_days"].apply(derive_tenor)
        return list(zip(frame["side"], tenor, frame["tod_bucket"]))

    def fit(self, frame: pd.DataFrame) -> SuiteSummary:
        """Route the frame into buckets, fit each head, collect metrics."""
        summary = SuiteSummary()
        if frame is None or frame.empty:
            return summary
        bucket_labels = self._bucket_label(frame)
        frame = frame.copy()
        frame["_bucket"] = bucket_labels
        for (side, tenor, tod), slab in frame.groupby("_bucket",
                                                       sort=False):
            if not side or not tenor or not tod:
                continue
            key = BucketKey(side=side, tenor=tenor, tod=tod)
            head = OptionsExpectedReturnModel(
                bucket=key,
                label_clip=self.label_clip,
                embargo_bars=self.embargo_bars,
                val_frac=self.val_frac,
                min_trades=self.min_trades,
                seed=self.seed,
            )
            head.fit(slab.drop(columns="_bucket"))
            self.heads[key] = head
            summary.per_head[key.as_label()] = head.metrics
            summary.head_count += 1
            if head.metrics.status == "fit":
                summary.fit_count += 1
            else:
                summary.skipped_count += 1
        summary.gate1_pass, summary.gate1_winning_buckets = (
            self._gate1_status(summary))
        return summary

    def predict_frame(self, frame: pd.DataFrame) -> pd.Series:
        """Route each row to its head and predict. Rows whose bucket
        has no fit (e.g. tenor out of scope, head skipped) get NaN.
        """
        if frame is None or frame.empty:
            return pd.Series(dtype=float)
        bucket_labels = self._bucket_label(frame)
        out = pd.Series(np.nan, index=frame.index, dtype=float)
        for (side, tenor, tod), slab in frame.assign(
            _bucket=bucket_labels
        ).groupby("_bucket", sort=False):
            if not side or not tenor or not tod:
                continue
            key = BucketKey(side=side, tenor=tenor, tod=tod)
            head = self.heads.get(key)
            if head is None or head._gbm is None:
                continue
            out.loc[slab.index] = head.predict_frame(
                slab.drop(columns="_bucket")).values
        return out

    def predict_strikes(self, strike_inputs: List[Dict[str, Any]]
                        ) -> List[Dict[str, Any]]:
        """Brief-time scoring helper.

        ``strike_inputs`` is a list of dicts. Each dict must carry the
        routing columns (``side``, ``dte_trading_days``, ``tod_bucket``)
        and may carry any subset of the model's feature columns. Missing
        features are passed to LightGBM as NaN (LightGBM handles missing
        natively).

        Returns a list of the same shape with each dict augmented by:

          predicted_net_return_atr      — the model's R prediction in
                                          ATR units. NaN when the
                                          routing bucket has no fit.
          bucket_key                    — ``side/tenor/tod`` for audit.
          predict_status                — "ok" | "no_head" | "out_of_scope"

        The brief generator does NOT need to know the model's full
        feature set; it passes what it has, and the model uses it. v1
        production paths should aim for the full feature row to avoid
        feature-imputation drift; the brief renderer should surface a
        "thin features" hint when it detects many NaNs (deferred to
        the renderer-side change in D.4).
        """
        if not strike_inputs:
            return []
        frame = pd.DataFrame(strike_inputs)
        preds = self.predict_frame(frame)
        out: List[Dict[str, Any]] = []
        for i, row in enumerate(strike_inputs):
            tenor = derive_tenor(row.get("dte_trading_days"))
            side = row.get("side")
            tod = row.get("tod_bucket")
            if not tenor:
                status, key = "out_of_scope", None
            else:
                key = BucketKey(side=side, tenor=tenor, tod=tod)
                head = self.heads.get(key)
                status = "ok" if head is not None and head._gbm is not None \
                    else "no_head"
            augmented = dict(row)
            augmented["predicted_net_return_atr"] = (
                float(preds.iloc[i]) if pd.notna(preds.iloc[i]) else None
            )
            augmented["bucket_key"] = key.as_label() if key else None
            augmented["predict_status"] = status
            out.append(augmented)
        return out

    @staticmethod
    def _gate1_status(summary: SuiteSummary) -> Tuple[bool, List[str]]:
        """Gate 1: at least one head with
        Spearman > 0.10 AND top_decile_mean_realized > 0 AND
        calibration error <= 0.15 ATR across IV-decile sub-slices.
        """
        winners: List[str] = []
        for label, m in summary.per_head.items():
            if m.status != "fit":
                continue
            if m.spearman_pred_vs_real <= 0.10:
                continue
            if m.top_decile_mean_realized <= 0:
                continue
            if m.sub_slice_calibration_by_iv_decile:
                max_err = max(abs(r.calibration_error)
                              for r in m.sub_slice_calibration_by_iv_decile)
                if max_err > 0.15:
                    continue
            winners.append(label)
        return (len(winners) > 0, winners)
