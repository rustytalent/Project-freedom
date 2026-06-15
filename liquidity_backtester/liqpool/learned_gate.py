"""Phase 3B + 3C — learned per-pool dynamic gate for the sector MoE.

The static Phase 1 blend was a constant 70/30 sector/global. The current
``SectorMoERespectModel`` already moved beyond that by computing a *per-sector*
``sector_weight`` from sample size, validation-loss improvement, and AUC. But
that weight is still constant across every pool inside the sector — which
throws away the information that disagreement, distance, and pool structure
carry about *when* the sector expert should be trusted.

This module turns the Phase 3A OOS audit table into:

  * Phase 3B — a labelled training frame where each decisive OOS pool yields
    a target gate weight ``w*`` that, applied to that single pool, would have
    minimised the squared blended-prediction error. The target is well-defined
    only when ``|sector_q - global_q|`` is above a small floor — degenerate
    rows are dropped from training.

  * Phase 3C — a small Huber LightGBM regressor that predicts ``w ∈ [0.20,
    0.85]`` per pool. The clipping floor / ceiling are the handoff doc's
    safety rails: a runaway gate cannot route 100% into either head.

Discipline:
  * The gate is FIT-OR-FALLBACK. If the dataset is too small, the signal too
    weak, or the trained model fails to beat a constant baseline on a
    chronological holdout, ``fit`` returns ``None`` and the caller keeps
    using the static per-sector weight. No silent regression.
  * The gate does NOT touch ``SectorMoERespectModel`` directly. It consumes
    the audit columns ``global_q``, ``sector_q`` (NaN when no expert exists)
    and emits a per-pool weight + a re-blended ``learned_blended_q``. That
    keeps the upstream MoE stable and makes Phase 3E fallback a one-line
    swap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Weight clamps from the handoff doc. Below the floor the gate would silently
# disable the expert; above the ceiling it would route everything to a sector
# head we've already shrunk because it wasn't trustworthy on its own.
GATE_WEIGHT_FLOOR: float = 0.20
GATE_WEIGHT_CEILING: float = 0.85

# Disagreement floor below which target weight is undefined (the two heads agree
# so the blend is invariant to w). Set conservatively: 1 percentage point of
# probability is well below the noise of either head.
DISAGREEMENT_FLOOR: float = 0.01

# The list of audit + side-channel columns the feature builder requires. Kept
# here so the contract is explicit; missing inputs raise rather than silently
# fall back to zeros.
REQUIRED_AUDIT_COLUMNS: Tuple[str, ...] = (
    "asset", "sector", "global_q", "sector_q", "actual",
    "pool_score", "pool_width", "n_tfs", "distance_atr",
)

GATE_FEATURE_NAMES: Tuple[str, ...] = (
    "global_q",
    "sector_q_filled",
    "has_sector_expert",
    "disagreement_abs",
    "disagreement_signed",
    "sector_regime_score",
    "sector_regime_conviction",
    "sector_oos_strict",
    "asset_reliability",
    "pool_score",
    "pool_width",
    "n_tfs",
    "distance_atr",
)


@dataclass
class LearnedGateTrainStats:
    """Diagnostic record for one gate fit attempt.

    Stored on the model so the orchestrator can print why a gate was kept,
    shrunk, or rejected — the same discipline the per-sector MoE prints today.
    """
    status: str = "skipped"
    reason: str = ""
    n_total: int = 0
    n_decisive: int = 0
    n_with_expert: int = 0
    n_with_disagreement: int = 0
    n_train: int = 0
    n_val: int = 0
    val_mae: float = 0.0
    baseline_val_mae: float = 0.0
    mae_improvement: float = 0.0
    mean_target: float = 0.0
    val_pearson: Optional[float] = None
    # Single-shuffle permutation null: same model architecture trained on
    # shuffled targets. If the real gate's val MAE is not meaningfully below
    # the shuffled gate's MAE the "win" was chance and we reject.
    shuffle_val_mae: float = 0.0
    shuffle_margin: float = 0.0
    target_floor: float = GATE_WEIGHT_FLOOR
    target_ceiling: float = GATE_WEIGHT_CEILING
    disagreement_floor: float = DISAGREEMENT_FLOOR

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _winsorize(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), low, high)


def derive_target_weight(actual: np.ndarray,
                         global_q: np.ndarray,
                         sector_q: np.ndarray,
                         disagreement_floor: float = DISAGREEMENT_FLOOR,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pool optimal gate weight that would minimise squared blended error.

    For a single pool with outcome ``y ∈ {0,1}``, the blended prediction
    ``b(w) = w·s + (1-w)·g`` has zero squared error when ``w = (y - g)/(s - g)``.
    Below ``disagreement_floor`` the two heads agree so the target is
    undefined; those rows are masked out and the caller should drop them
    from the training frame.

    Returns (target, mask) — target is the clipped-to-[0,1] weight; mask is
    True for rows where the target is well-defined.
    """
    actual = np.asarray(actual, dtype=float)
    g = np.asarray(global_q, dtype=float)
    s = np.asarray(sector_q, dtype=float)
    diff = s - g
    finite = np.isfinite(actual) & np.isfinite(g) & np.isfinite(s)
    decisive = np.abs(diff) > disagreement_floor
    mask = finite & decisive
    target = np.full(len(actual), np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(mask, (actual - g) / diff, np.nan)
    target = np.clip(raw, 0.0, 1.0)
    return target, mask


def build_gate_feature_frame(audit_table: pd.DataFrame,
                             sector_regime_by_sector: Dict[str, Dict],
                             sector_oos_strict_by_sector: Dict[str, float],
                             asset_reliability_by_asset: Dict[str, float],
                             ) -> pd.DataFrame:
    """Phase 3B — assemble the feature matrix exactly as the gate consumes it.

    Side-channel inputs default to neutral values when a sector/asset is not
    listed: missing reliability → 0.5, missing regime → score=0/conviction=0.
    That keeps train and live frames structurally identical even when one of
    the inputs is partially populated."""
    missing = [c for c in REQUIRED_AUDIT_COLUMNS if c not in audit_table.columns]
    if missing:
        raise ValueError(f"audit_table missing required columns: {missing}")
    df = audit_table.copy()
    g = pd.to_numeric(df["global_q"], errors="coerce").to_numpy(dtype=float)
    s_raw = pd.to_numeric(df["sector_q"], errors="coerce").to_numpy(dtype=float)
    has_expert = np.isfinite(s_raw).astype(float)
    s_filled = np.where(np.isfinite(s_raw), s_raw, g)
    disagreement_signed = s_filled - g

    def _regime(sec: str, key: str) -> float:
        r = sector_regime_by_sector.get(sec) or {}
        return float(r.get(key, 0.0) or 0.0)

    def _sector_strict(sec: str) -> float:
        return float(sector_oos_strict_by_sector.get(sec, 0.5))

    def _reliability(asset: str) -> float:
        return float(asset_reliability_by_asset.get(asset, 0.5))

    out = pd.DataFrame({
        "global_q": g,
        "sector_q_filled": s_filled,
        "has_sector_expert": has_expert,
        "disagreement_abs": np.abs(disagreement_signed),
        "disagreement_signed": disagreement_signed,
        "sector_regime_score": df["sector"].map(lambda x: _regime(x, "score")).to_numpy(dtype=float),
        "sector_regime_conviction": df["sector"].map(lambda x: _regime(x, "conviction")).to_numpy(dtype=float),
        "sector_oos_strict": df["sector"].map(_sector_strict).to_numpy(dtype=float),
        "asset_reliability": df["asset"].map(_reliability).to_numpy(dtype=float),
        "pool_score": pd.to_numeric(df["pool_score"], errors="coerce").fillna(0.0).to_numpy(),
        "pool_width": pd.to_numeric(df["pool_width"], errors="coerce").fillna(0.0).to_numpy(),
        "n_tfs": pd.to_numeric(df["n_tfs"], errors="coerce").fillna(0.0).to_numpy(),
        "distance_atr": pd.to_numeric(df["distance_atr"], errors="coerce").fillna(0.0).to_numpy(),
    }, index=df.index)
    # Re-order to the canonical schema so train/predict frames are bit-identical.
    out = out[list(GATE_FEATURE_NAMES)]
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _chronological_or_random_split(frame: pd.DataFrame, val_frac: float, seed: int,
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Prefer chronological split when `available_at` is on the frame; fall
    back to a seeded random split otherwise. The chronological path matches
    what live use looks like (you predict on the most recent data)."""
    if "available_at" in frame.columns and frame["available_at"].notna().any():
        ordered = frame.copy()
        ordered["_sort"] = pd.to_datetime(ordered["available_at"], errors="coerce")
        ordered = ordered.sort_values(["_sort"], na_position="last", kind="mergesort")
        idx = ordered.index.to_numpy()
        n_val = max(1, int(round(len(idx) * val_frac)))
        n_val = min(n_val, max(1, len(idx) - 1))
        return idx[:-n_val], idx[-n_val:]
    rng = np.random.default_rng(seed)
    order = rng.permutation(frame.index.to_numpy())
    n_val = max(1, int(round(len(order) * val_frac)))
    n_val = min(n_val, max(1, len(order) - 1))
    return order[:-n_val], order[-n_val:]


@dataclass
class LearnedGateModel:
    """Per-pool dynamic gate. Predicts a weight ``w ∈ [floor, ceiling]`` that
    blends ``sector_q`` against ``global_q`` for one OOS pool.

    Created via the classmethod :meth:`fit_from_audit`; that returns ``None``
    when the data does not support a gate so callers can fall through to the
    static per-sector weight without a special case.
    """
    booster: Any = None
    feature_names: List[str] = field(default_factory=list)
    stats: LearnedGateTrainStats = field(default_factory=LearnedGateTrainStats)
    sector_regime_by_sector: Dict[str, Dict] = field(default_factory=dict)
    sector_oos_strict_by_sector: Dict[str, float] = field(default_factory=dict)
    asset_reliability_by_asset: Dict[str, float] = field(default_factory=dict)
    weight_floor: float = GATE_WEIGHT_FLOOR
    weight_ceiling: float = GATE_WEIGHT_CEILING

    @classmethod
    def fit_from_audit(cls,
                       audit_table: pd.DataFrame,
                       sector_regime_by_sector: Dict[str, Dict],
                       sector_oos_strict_by_sector: Dict[str, float],
                       asset_reliability_by_asset: Dict[str, float],
                       *,
                       seed: int = 73,
                       val_frac: float = 0.25,
                       min_decisive: int = 60,
                       min_disagreement_rows: int = 40,
                       min_mae_improvement: float = 0.008,
                       min_shuffle_margin: float = 0.010,
                       disagreement_floor: float = DISAGREEMENT_FLOOR,
                       weight_floor: float = GATE_WEIGHT_FLOOR,
                       weight_ceiling: float = GATE_WEIGHT_CEILING,
                       ) -> Optional["LearnedGateModel"]:
        """Fit a per-pool gate from a Phase 3A audit table.

        Returns ``None`` (not an empty model) when the data is insufficient,
        so the caller's branch is the natural ``if gate is None: keep static``.
        The :class:`LearnedGateTrainStats` shows up regardless on the returned
        model — when ``None`` is returned no diagnostic is preserved because
        the orchestrator builds its own message.
        """
        stats = LearnedGateTrainStats(
            target_floor=weight_floor,
            target_ceiling=weight_ceiling,
            disagreement_floor=disagreement_floor,
        )
        if audit_table is None or audit_table.empty:
            stats.status = "skipped"
            stats.reason = "audit table empty"
            return None

        decisive_mask = pd.to_numeric(audit_table["actual"], errors="coerce").notna()
        with_expert_mask = pd.to_numeric(audit_table["sector_q"], errors="coerce").notna()
        usable_mask = decisive_mask & with_expert_mask
        stats.n_total = int(len(audit_table))
        stats.n_decisive = int(decisive_mask.sum())
        stats.n_with_expert = int(with_expert_mask.sum())

        if int(usable_mask.sum()) < min_decisive:
            stats.status = "skipped"
            stats.reason = (
                f"decisive rows with a sector expert = {int(usable_mask.sum())} "
                f"< min_decisive={min_decisive}"
            )
            return None

        usable = audit_table.loc[usable_mask].copy()
        target, mask = derive_target_weight(
            usable["actual"].to_numpy(dtype=float),
            usable["global_q"].to_numpy(dtype=float),
            usable["sector_q"].to_numpy(dtype=float),
            disagreement_floor=disagreement_floor,
        )
        usable["_gate_target"] = target
        usable = usable.loc[mask].copy()
        stats.n_with_disagreement = int(len(usable))
        if stats.n_with_disagreement < min_disagreement_rows:
            stats.status = "skipped"
            stats.reason = (
                f"rows with |disagreement| > {disagreement_floor:.3f} = "
                f"{stats.n_with_disagreement} < min_disagreement_rows={min_disagreement_rows}"
            )
            return None

        features = build_gate_feature_frame(
            usable,
            sector_regime_by_sector=sector_regime_by_sector,
            sector_oos_strict_by_sector=sector_oos_strict_by_sector,
            asset_reliability_by_asset=asset_reliability_by_asset,
        )
        tr_idx, val_idx = _chronological_or_random_split(usable, val_frac=val_frac, seed=seed)
        y_all = usable["_gate_target"].to_numpy(dtype=float)
        y_tr = usable.loc[tr_idx, "_gate_target"].to_numpy(dtype=float)
        y_val = usable.loc[val_idx, "_gate_target"].to_numpy(dtype=float)
        X_tr = features.loc[tr_idx].to_numpy(dtype=float)
        X_val = features.loc[val_idx].to_numpy(dtype=float)
        stats.n_train = int(len(tr_idx))
        stats.n_val = int(len(val_idx))
        stats.mean_target = float(y_all.mean())

        # Baseline: constant prediction at the train-set mean. The trained gate
        # must beat this on the held-out tail, otherwise we keep the static
        # per-sector weight.
        train_mean = float(y_tr.mean())
        baseline_pred_val = np.full(len(y_val), train_mean, dtype=float)
        baseline_mae = float(np.mean(np.abs(y_val - baseline_pred_val)))
        stats.baseline_val_mae = baseline_mae

        try:
            import lightgbm as lgb
        except Exception as exc:
            stats.status = "skipped"
            stats.reason = f"lightgbm unavailable: {exc}"
            return None

        # Tight regularization is mandatory here — the OOS audit will rarely
        # have many thousands of rows, and the target derivation is noisy on
        # binary outcomes. Small leaves + aggressive L2 + low max_depth keep
        # the model honest. The "trained on pure noise" smoke test is what
        # this discipline is sized against.
        params = dict(
            objective="huber",
            alpha=0.9,
            metric="l1",
            learning_rate=0.03,
            num_leaves=8,
            max_depth=4,
            min_data_in_leaf=30,
            feature_fraction=0.70,
            bagging_fraction=0.80,
            bagging_freq=4,
            lambda_l2=10.0,
            lambda_l1=1.0,
            min_gain_to_split=0.02,
            verbose=-1,
            seed=int(seed),
        )

        def _fit_and_eval(y_train: np.ndarray, y_eval: np.ndarray,
                          seed_offset: int = 0) -> Tuple[Any, float]:
            local_params = dict(params)
            local_params["seed"] = int(seed) + int(seed_offset)
            dtr = lgb.Dataset(X_tr, label=y_train, feature_name=list(GATE_FEATURE_NAMES))
            dval = lgb.Dataset(X_val, label=y_eval, reference=dtr,
                               feature_name=list(GATE_FEATURE_NAMES))
            booster = lgb.train(
                local_params, dtr, num_boost_round=400, valid_sets=[dval],
                valid_names=["val"],
                callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False),
                           lgb.log_evaluation(0)],
            )
            pred = booster.predict(X_val, num_iteration=booster.best_iteration)
            pred_clipped = np.clip(pred, weight_floor, weight_ceiling)
            return booster, float(np.mean(np.abs(y_eval - pred_clipped)))

        booster, val_mae = _fit_and_eval(y_tr, y_val, seed_offset=0)
        stats.val_mae = val_mae
        stats.mae_improvement = baseline_mae - val_mae

        # Two-shuffle permutation null — refits the same architecture on
        # shuffled training targets to estimate "luck floor" val MAE. Mean
        # across shuffles smooths the single-shuffle variance that fluke-
        # passed pure noise in development. If the real gate's val MAE is
        # not below the mean shuffled MAE by ``min_shuffle_margin`` we treat
        # the win as statistical luck and reject.
        shuffle_maes: List[float] = []
        for k, off in enumerate((37, 113), start=1):
            rng = np.random.default_rng(seed * 13 + k)
            y_tr_shuf = rng.permutation(y_tr)
            _, shuf_mae = _fit_and_eval(y_tr_shuf, y_val, seed_offset=off)
            shuffle_maes.append(shuf_mae)
        shuffle_val_mae = float(np.mean(shuffle_maes))
        stats.shuffle_val_mae = shuffle_val_mae
        stats.shuffle_margin = shuffle_val_mae - val_mae

        val_pred_clipped = np.clip(
            booster.predict(X_val, num_iteration=booster.best_iteration),
            weight_floor, weight_ceiling,
        )
        if len(np.unique(val_pred_clipped)) > 1 and len(np.unique(y_val)) > 1:
            pearson = float(np.corrcoef(val_pred_clipped, y_val)[0, 1])
            stats.val_pearson = pearson if np.isfinite(pearson) else None

        # Layered acceptance. ALL three must hold; any failure means we keep
        # the static per-sector weight and the orchestrator records why.
        if stats.mae_improvement < min_mae_improvement:
            stats.status = "rejected"
            stats.reason = (
                f"val MAE improvement {stats.mae_improvement:+.4f} "
                f"< min_mae_improvement={min_mae_improvement:.4f}"
            )
            return None
        if stats.shuffle_margin < min_shuffle_margin:
            stats.status = "rejected"
            stats.reason = (
                f"shuffle-null margin {stats.shuffle_margin:+.4f} "
                f"< min_shuffle_margin={min_shuffle_margin:.4f} "
                f"(real {val_mae:.4f} vs shuffled {shuffle_val_mae:.4f}); "
                f"observed improvement is not distinguishable from chance"
            )
            return None

        stats.status = "trained"
        stats.reason = (
            f"val MAE {val_mae:.4f} vs baseline {baseline_mae:.4f} "
            f"(Δ={stats.mae_improvement:+.4f}); shuffle margin "
            f"{stats.shuffle_margin:+.4f}"
        )
        return cls(
            booster=booster,
            feature_names=list(GATE_FEATURE_NAMES),
            stats=stats,
            sector_regime_by_sector=dict(sector_regime_by_sector),
            sector_oos_strict_by_sector=dict(sector_oos_strict_by_sector),
            asset_reliability_by_asset=dict(asset_reliability_by_asset),
            weight_floor=weight_floor,
            weight_ceiling=weight_ceiling,
        )

    def predict_weights(self, audit_table: pd.DataFrame) -> np.ndarray:
        """Predict a clipped gate weight per row. Rows whose ``sector_q`` is
        NaN — i.e. no sector expert for that sector — get weight 0, which
        leaves their blended prediction equal to the global prediction.
        """
        if self.booster is None:
            raise RuntimeError("LearnedGateModel is not fitted")
        features = build_gate_feature_frame(
            audit_table,
            sector_regime_by_sector=self.sector_regime_by_sector,
            sector_oos_strict_by_sector=self.sector_oos_strict_by_sector,
            asset_reliability_by_asset=self.asset_reliability_by_asset,
        )
        raw = self.booster.predict(features.to_numpy(dtype=float),
                                   num_iteration=self.booster.best_iteration)
        clipped = np.clip(raw, self.weight_floor, self.weight_ceiling)
        # No expert -> route entirely to global. The per-row weight on the
        # expert side becomes 0 regardless of what the model said.
        has_expert = np.isfinite(
            pd.to_numeric(audit_table["sector_q"], errors="coerce").to_numpy(dtype=float)
        )
        return np.where(has_expert, clipped, 0.0)

    def blend_one(self, *, global_q: float, sector_q: Optional[float],
                  sector: str, asset: str, pool_score: float, pool_width: float,
                  n_tfs: int, distance_atr: float) -> Tuple[float, float]:
        """Predict a learned weight + re-blended q for a single live candidate.

        Returns ``(learned_gate_weight, learned_blended_q)``. When
        ``sector_q`` is ``None`` (no trained expert for that sector) the weight
        is forced to 0 and the blended q equals the global q — the same
        fallback the audit-table ``apply`` path uses.

        This is the Phase 3D wiring point: live runners take ``comp`` from
        ``SectorMoERespectModel.predict_components`` and call this to override
        the static blend with the learned dynamic one.
        """
        if self.booster is None:
            raise RuntimeError("LearnedGateModel is not fitted")
        if sector_q is None or not np.isfinite(sector_q):
            return 0.0, float(np.clip(global_q, 0.02, 0.98))
        row = pd.DataFrame([{
            "asset": asset, "sector": sector,
            "global_q": float(global_q), "sector_q": float(sector_q),
            "actual": np.nan, "pool_score": float(pool_score),
            "pool_width": float(pool_width), "n_tfs": float(n_tfs),
            "distance_atr": float(distance_atr),
        }])
        weights = self.predict_weights(row)
        w = float(weights[0])
        blended = w * float(sector_q) + (1 - w) * float(global_q)
        return w, float(np.clip(blended, 0.02, 0.98))

    def apply(self, audit_table: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the audit table augmented with the learned weight
        and the re-blended prediction. ``learned_blended_q`` falls back to
        ``global_q`` when no expert is available, mirroring the static MoE."""
        out = audit_table.copy()
        weights = self.predict_weights(audit_table)
        out["learned_gate_weight"] = weights
        global_q = pd.to_numeric(out["global_q"], errors="coerce").to_numpy(dtype=float)
        sector_q = pd.to_numeric(out["sector_q"], errors="coerce").to_numpy(dtype=float)
        blended = np.where(
            np.isfinite(sector_q),
            weights * sector_q + (1.0 - weights) * global_q,
            global_q,
        )
        out["learned_blended_q"] = np.clip(blended, 0.02, 0.98)
        return out

    def feature_importance(self, top_k: int = 13) -> List[Dict]:
        if self.booster is None:
            return []
        gains = self.booster.feature_importance(importance_type="gain")
        pairs = sorted(zip(self.feature_names, gains), key=lambda kv: -kv[1])[:top_k]
        return [{"feature": name, "importance_gain": float(gain)} for name, gain in pairs]
