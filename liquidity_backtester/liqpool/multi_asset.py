"""Track 4: multi-asset training.

The Q/direction/proximity models trained on a single stock (~500-700 pools, ~200-300 snapshots)
sit at the edge of their reliable sample size — OOS AUC oscillates 0.54–0.70 across runs. The
fix isn't more code: it's more data. Pool behaviour is structurally similar across liquid
large-caps (especially within a sector), so training on 5-6 stocks simultaneously gives the
models 5-10x the training data and should push the Q model OOS AUC into the 0.62–0.68 band.

This module orchestrates:
  1. Per-asset walk-forward (with train_models=False so no per-asset model fitting).
     - Per-asset detection params via the optimizer (different stocks have different vol).
     - Collects train_pools_for_ml + oos_pools per asset, tagged with `pool.asset = symbol`.
  2. UNIFIED model training on the combined cross-asset pool/snapshot sets:
     - PoolRespectModel on combined train pools, calibrated on combined OOS pools.
     - DirectionModel + ProximityModel on combined snapshots from per-asset snapshot generation.
     - StratifiedRespectModel on combined OOS pools.
  3. Per-asset final-config fit + per-asset current-state for prediction.
  4. Cross-asset ranked recommendation: every pool's Q × T_today × direction-bonus → sorted.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Callable, Union
import copy
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

from .config import Config
from .data import DataProvider, fetch, multi_timeframe
from .pools import build_pools, project_to_base, Pool
from .tester import test_pools, summarise, PoolResult
from .optimizer import optimize
from .walkforward import walk_forward, WalkForwardReport
from .featurize import MultiAssetFeaturizer
from .ml_model import (PoolRespectModel, SectorMoERespectModel,
                       trainable_mask, labels as ml_labels, label_end_time)
from .stratified import StratifiedRespectModel
from .timing import (StateFeaturizer, generate_snapshots, DirectionModel, ProximityModel,
                     evaluate_timing, evaluate_timing_frames, TimingReport)
from .feature_store import (FeatureStore, build_feature_store_for_report,
                            PROXIMITY_HORIZONS, DEFAULT_DIRECTION_HORIZON)
from .reaction_model import ReactionModelSuite
from .policy_model import PolicyOutcomeModelSuite
from .stats import wilson_score_interval, bootstrap_proportion_ci
from .indicators import atr
from .stratified import _headline_factor
from .sectors import (sector_of, compute_sector_metrics, sector_regime_signal,
                      per_sector_oos, per_asset_reliability)
from .learned_gate import LearnedGateModel


@dataclass
class AssetData:
    """Per-asset state collected during the multi-asset run."""
    symbol: str
    base_df: pd.DataFrame
    tf_data: Dict[str, pd.DataFrame]
    walkforward: WalkForwardReport               # train_models=False
    final_cfg: Config                             # fit on full history
    final_pools: List[Pool]                       # tagged with asset
    final_results: List[PoolResult]


@dataclass
class MultiAssetReport:
    assets: Dict[str, AssetData] = field(default_factory=dict)
    # Symbols that were dropped during fetch (either failed completely or fell back to
    # synthetic data — we don't want synthetic data polluting the unified ML training).
    skipped_symbols: List[str] = field(default_factory=list)
    skip_reasons: Dict[str, str] = field(default_factory=dict)
    # Combined OOS pool/result population across assets
    total_train_pools: int = 0
    total_oos_pools: int = 0
    total_oos_tested: int = 0
    pooled_oos_respect: float = 0.0
    pooled_oos_strict: float = 0.0
    pooled_oos_ci_wilson: Tuple[float, float] = (0.0, 0.0)
    pooled_oos_ci_bootstrap: Tuple[float, float] = (0.0, 0.0)
    mean_overfit_gap: float = 0.0
    # Unified models (trained on combined data)
    unified_featurizer: Optional[MultiAssetFeaturizer] = None
    unified_ml: Optional[Union[PoolRespectModel, SectorMoERespectModel]] = None
    unified_direction: Optional[DirectionModel] = None
    unified_proximity: Dict[int, ProximityModel] = field(default_factory=dict)
    unified_stratified: Optional[StratifiedRespectModel] = None
    unified_timing_report: Optional[TimingReport] = None
    unified_oos_audit: Optional["OOSPredictionAudit"] = None
    # Phase 3B/3C: per-pool dynamic gate learned from the OOS audit. ``None``
    # when the audit didn't carry enough signal to fit one — callers must keep
    # using the SectorMoE's static per-sector weight in that case.
    unified_learned_gate: Optional["LearnedGateModel"] = None
    learned_gate_metrics: Optional[QualityModelAuditMetrics] = None
    post_touch_reaction_metrics: Dict = field(default_factory=dict)
    reaction_model: Optional[ReactionModelSuite] = None
    reaction_model_report: List[Dict] = field(default_factory=list)
    reaction_model_calibration: List[Dict] = field(default_factory=list)
    reaction_feature_importance: List[Dict] = field(default_factory=list)
    policy_model: Optional[PolicyOutcomeModelSuite] = None
    policy_model_report: List[Dict] = field(default_factory=list)
    policy_model_calibration: List[Dict] = field(default_factory=list)
    policy_model_feature_importance: List[Dict] = field(default_factory=list)
    feature_store_stats: Dict = field(default_factory=dict)


@dataclass
class QualityModelAuditMetrics:
    """OOS prediction metrics for one quality-model output."""
    model: str
    n: int
    brier: float
    logloss: float
    auc: Optional[float]
    top_decile_hit_rate: float
    base_rate: float
    mean_prediction: float


@dataclass
class OOSPredictionAudit:
    """Phase 3A audit table plus aggregate and per-sector calibration summaries."""
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, QualityModelAuditMetrics] = field(default_factory=dict)
    sector_calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    distance_bucket_metrics: Dict[str, List[Dict]] = field(default_factory=dict)


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


def q_bucket(q: float) -> str:
    if q < 0.40:
        return "<40%"
    if q < 0.55:
        return "40-55%"
    if q < 0.70:
        return "55-70%"
    return "70%+"


def _post_touch_group_metrics(items: List[Tuple[Pool, PoolResult, Optional[float]]]) -> Dict:
    touched = [(p, r, q) for p, r, q in items if r.touched_at is not None]
    n = len(touched)
    if n == 0:
        return {
            "n": 0,
            "respected_strong_rate": 0.0,
            "swept_and_reclaimed_rate": 0.0,
            "broken_strong_rate": 0.0,
            "strict_respect_rate": 0.0,
            "avg_mae_after_touch": 0.0,
            "avg_mfe_after_touch": 0.0,
            "r_multiple_available": False,
        }
    strong = sum(1 for _, r, _ in touched if r.outcome == "respected_strong")
    swept = sum(1 for _, r, _ in touched if r.outcome == "swept_and_reclaimed")
    broken = sum(1 for _, r, _ in touched if r.outcome == "broken_strong")
    strict_den = strong + swept + broken
    return {
        "n": int(n),
        "respected_strong_rate": strong / n,
        "swept_and_reclaimed_rate": swept / n,
        "broken_strong_rate": broken / n,
        "strict_respect_rate": (strong + swept) / strict_den if strict_den else 0.0,
        "avg_mae_after_touch": float(np.mean([r.max_excursion_through for _, r, _ in touched])),
        "avg_mfe_after_touch": float(np.mean([r.reaction_atr for _, r, _ in touched])),
        "r_multiple_available": False,
    }


def build_post_touch_reaction_metrics(pools: List[Pool], results: List[PoolResult],
                                      q_values: Optional[Sequence[float]] = None) -> Dict:
    q_list: List[Optional[float]] = ([float(v) for v in q_values]
                                    if q_values is not None else [None] * len(pools))
    paired = list(zip(pools, results, q_list))
    out = {"overall": _post_touch_group_metrics(paired), "by": {}}

    groups: Dict[str, Dict[str, List[Tuple[Pool, PoolResult, Optional[float]]]]] = {
        "factor_type": {},
        "tf_count": {},
        "sector": {},
        "direction_alignment": {},
        "q_bucket": {},
    }
    for p, r, q in paired:
        groups["factor_type"].setdefault(_headline_factor(p), []).append((p, r, q))
        groups["tf_count"].setdefault(str(len(set(p.tfs))), []).append((p, r, q))
        groups["sector"].setdefault(sector_of(p.asset), []).append((p, r, q))
        # Historical OOS pools do not currently store the contemporaneous direction-model signal.
        # Keep the dimension explicit so live runs can fill it later without changing artifacts.
        groups["direction_alignment"].setdefault("DIR_NA", []).append((p, r, q))
        groups["q_bucket"].setdefault(q_bucket(q) if q is not None else "Q_NA", []).append(
            (p, r, q)
        )

    for group_name, group_items in groups.items():
        out["by"][group_name] = {
            name: _post_touch_group_metrics(items)
            for name, items in sorted(group_items.items(), key=lambda kv: kv[0])
        }
    return out


def _prediction_metrics(model_name: str, y_true: np.ndarray,
                        y_pred: np.ndarray) -> QualityModelAuditMetrics:
    from sklearn.metrics import roc_auc_score

    n = int(len(y_true))
    if n == 0:
        return QualityModelAuditMetrics(model_name, 0, float("nan"), float("nan"), None,
                                        float("nan"), float("nan"), float("nan"))

    pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, 1.0 - 1e-6)
    actual = np.asarray(y_true, dtype=float)
    brier = float(np.mean((pred - actual) ** 2))
    logloss = float(-np.mean(actual * np.log(pred) + (1.0 - actual) * np.log(1.0 - pred)))
    auc = float(roc_auc_score(actual, pred)) if len(np.unique(actual)) == 2 else None
    top_n = max(1, int(np.ceil(n * 0.10)))
    top_idx = np.argsort(-pred)[:top_n]
    return QualityModelAuditMetrics(
        model=model_name,
        n=n,
        brier=brier,
        logloss=logloss,
        auc=auc,
        top_decile_hit_rate=float(actual[top_idx].mean()),
        base_rate=float(actual.mean()),
        mean_prediction=float(pred.mean()),
    )


def build_oos_prediction_audit(model: Union[PoolRespectModel, SectorMoERespectModel],
                               featurizer: MultiAssetFeaturizer,
                               oos_pools: List[Pool],
                               oos_results: List[PoolResult],
                               asset_dfs: Optional[Dict[str, pd.DataFrame]] = None
                               ) -> OOSPredictionAudit:
    """Build the Phase 3A OOS table before any learned gate changes.

    The table keeps every OOS pool for traceability, but metrics use only decisive outcomes
    (`trainable_mask`) because weak/ambiguous touches are intentionally excluded from Q training.
    """
    if not oos_pools or not oos_results:
        return OOSPredictionAudit()
    X_oos = featurizer.transform_batch(oos_pools)
    if isinstance(model, SectorMoERespectModel):
        table = model.predict_components(X_oos, oos_pools)
    else:
        pred = model.predict(X_oos, pools=oos_pools)
        table = pd.DataFrame({
            "asset": [p.asset for p in oos_pools],
            "sector": ["OTHER"] * len(oos_pools),
            "global_q": pred,
            "sector_q": np.nan,
            "gate_weight": 0.0,
            "blended_q": pred,
            "fallback_reason": "",
        })

    trainable = trainable_mask(oos_results)
    actual = np.full(len(oos_results), np.nan, dtype=float)
    if trainable.any():
        actual[trainable] = ml_labels([r for r, keep in zip(oos_results, trainable) if keep])

    table.insert(0, "pool_index", np.arange(len(oos_pools)))
    table["actual"] = actual
    table["outcome"] = [r.outcome for r in oos_results]
    table["is_decisive"] = trainable
    table["pool_score"] = [float(p.score) for p in oos_pools]
    table["pool_mid"] = [float(p.mid) for p in oos_pools]
    table["pool_width"] = [float(p.width) for p in oos_pools]
    table["n_tfs"] = [int(len(set(p.tfs))) for p in oos_pools]
    distances = []
    if asset_dfs:
        atr_cache: Dict[str, pd.Series] = {}
        for p in oos_pools:
            df = asset_dfs.get(p.asset)
            if df is None or df.empty:
                distances.append(np.nan)
                continue
            if p.asset not in atr_cache:
                atr_cache[p.asset] = atr(df, 14).bfill()
            # Bar at-or-before available_at (side="right"-1), matching featurize.py — side="left"
            # would select the NEXT bar for a non-matching timestamp, a 1-bar forward peek.
            pos = int(df.index.searchsorted(p.available_at, side="right")) - 1
            pos = min(max(pos, 0), len(df) - 1)
            close = float(df["close"].iloc[pos])
            atr_val = max(float(atr_cache[p.asset].iloc[pos]), 1e-9)
            if p.price_low > close:
                dist = (p.mid - close) / atr_val
            elif p.price_high < close:
                dist = (close - p.mid) / atr_val
            else:
                dist = 0.0
            distances.append(float(max(0.0, dist)))
    else:
        distances = [np.nan] * len(oos_pools)
    table["distance_atr"] = distances
    table["distance_bucket"] = [
        distance_bucket(float(d)) if pd.notna(d) else "unknown" for d in distances
    ]

    metrics: Dict[str, QualityModelAuditMetrics] = {}
    metric_cols = {
        "global": "global_q",
        "sector_expert": "sector_q",
        "blended": "blended_q",
    }
    for name, col in metric_cols.items():
        valid = table["actual"].notna() & table[col].notna()
        metrics[name] = _prediction_metrics(
            name,
            table.loc[valid, "actual"].to_numpy(dtype=float),
            table.loc[valid, col].to_numpy(dtype=float),
        )

    cal_rows = []
    decisive = table[table["actual"].notna()]
    for sector, sec_df in decisive.groupby("sector"):
        for name, col in metric_cols.items():
            valid = sec_df[col].notna()
            if not valid.any():
                continue
            pred = sec_df.loc[valid, col].astype(float)
            act = sec_df.loc[valid, "actual"].astype(float)
            cal_rows.append({
                "model": name,
                "sector": sector,
                "n": int(len(act)),
                "actual_rate": float(act.mean()),
                "mean_prediction": float(pred.mean()),
                "calibration_error": float(pred.mean() - act.mean()),
            })
    sector_calib = pd.DataFrame(cal_rows).sort_values(
        ["model", "sector"], ignore_index=True,
    ) if cal_rows else pd.DataFrame()

    distance_metrics: Dict[str, List[Dict]] = {}
    decisive = table[table["actual"].notna() & table["distance_atr"].notna()]
    for name, col in metric_cols.items():
        rows = []
        for bucket, bdf in decisive.groupby("distance_bucket"):
            valid = bdf[col].notna()
            if not valid.any():
                continue
            mt = _prediction_metrics(
                name,
                bdf.loc[valid, "actual"].to_numpy(dtype=float),
                bdf.loc[valid, col].to_numpy(dtype=float),
            )
            rows.append({
                "bucket": bucket,
                "n": mt.n,
                "base_rate": mt.base_rate,
                "auc": mt.auc,
                "brier": mt.brier,
                "logloss": mt.logloss,
                "calibration_error": mt.mean_prediction - mt.base_rate,
            })
        distance_metrics[name] = sorted(rows, key=lambda r: [
            "0-1 ATR", "1-3 ATR", "3-5 ATR", "5-10 ATR", "10+ ATR", "unknown"
        ].index(r["bucket"]) if r["bucket"] in [
            "0-1 ATR", "1-3 ATR", "3-5 ATR", "5-10 ATR", "10+ ATR", "unknown"
        ] else 99)

    return OOSPredictionAudit(table=table, metrics=metrics,
                              sector_calibration=sector_calib,
                              distance_bucket_metrics=distance_metrics)


def _fit_and_apply_learned_gate(report: MultiAssetReport,
                                ml: SectorMoERespectModel,
                                oos_pools: List[Pool],
                                oos_results: List[PoolResult],
                                asset_dfs: Dict[str, pd.DataFrame],
                                seed: int = 200) -> None:
    """Try to fit a per-pool learned gate (Phase 3B+3C) from the audit table.

    Mutates ``report`` in place: on success attaches the gate to
    ``report.unified_learned_gate``, recomputes ``learned_blended_q`` per row,
    and records audit metrics for that column. On failure leaves the static
    blend untouched — this is the explicit Phase 3E safety rail.
    """
    audit = report.unified_oos_audit
    if audit is None or audit.table.empty:
        return
    # Side-channel inputs the gate needs at fit time. Kept here (not on the
    # audit table) so the schema of the audit is stable across runs.
    sector_metrics = compute_sector_metrics(asset_dfs)
    sectors_seen = sorted({sector_of(p.asset) for p in oos_pools})
    regime_by_sector = {sec: sector_regime_signal(sector_metrics, sec) for sec in sectors_seen}
    sector_oos_raw = per_sector_oos(oos_pools, oos_results)
    sector_strict = {sec: float(v.get("strict_respect", 0.5)) for sec, v in sector_oos_raw.items()}
    reliability_raw = per_asset_reliability(report)
    asset_reliability = {asset: float(v) for asset, v in reliability_raw.items()}

    gate = LearnedGateModel.fit_from_audit(
        audit.table,
        sector_regime_by_sector=regime_by_sector,
        sector_oos_strict_by_sector=sector_strict,
        asset_reliability_by_asset=asset_reliability,
        seed=seed,
    )
    if gate is None:
        # The orchestrator deliberately stays quiet on the skip case — the
        # next-chat handoff explicitly wants "fall back to fixed blend" with
        # no behaviour change for downstream consumers. The console block
        # emitted later prints whether the gate was tried + skipped.
        return

    enriched = gate.apply(audit.table)
    audit.table["learned_gate_weight"] = enriched["learned_gate_weight"].to_numpy(dtype=float)
    audit.table["learned_blended_q"] = enriched["learned_blended_q"].to_numpy(dtype=float)

    valid = audit.table["actual"].notna() & audit.table["learned_blended_q"].notna()
    if valid.any():
        metrics = _prediction_metrics(
            "learned_blended",
            audit.table.loc[valid, "actual"].to_numpy(dtype=float),
            audit.table.loc[valid, "learned_blended_q"].to_numpy(dtype=float),
        )
        audit.metrics["learned_blended"] = metrics
        report.learned_gate_metrics = metrics

    report.unified_learned_gate = gate


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_SYNTHETIC_SENTINEL = pd.Timestamp("2024-01-02 09:30:00")


def _safe_symbol_name(symbol: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in symbol)


def _asset_checkpoint_path(checkpoint_dir: Optional[str], symbol: str) -> Optional[Path]:
    if not checkpoint_dir:
        return None
    return Path(checkpoint_dir).expanduser() / f"{_safe_symbol_name(symbol)}.pkl"


def _save_asset_checkpoint(payload: Dict, checkpoint_dir: Optional[str]) -> None:
    path = _asset_checkpoint_path(checkpoint_dir, payload["symbol"])
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)


def _load_asset_checkpoint(checkpoint_dir: Optional[str], symbol: str) -> Optional[Dict]:
    path = _asset_checkpoint_path(checkpoint_dir, symbol)
    if path is None or not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _run_one_asset_payload(symbol: str, cfg: Config,
                           n_folds: int, train_frac: float,
                           iters_per_fold: int, final_iters: int,
                           min_train_days: int,
                           data_provider: Optional[DataProvider],
                           checkpoint_dir: Optional[str] = None) -> Dict:
    """Run one asset end-to-end. Designed to be safe for ProcessPoolExecutor."""
    # Avoid each worker asking LightGBM/BLAS for all cores. This keeps 3-4 asset workers
    # usable on a 16GB laptop instead of oversubscribing the machine.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        if data_provider is not None:
            tf = data_provider.load_timeframes(
                symbol, cfg.base_interval, cfg.higher_tfs,
                period=cfg.period, start=cfg.start, end=cfg.end,
            )
            base = tf["base"]
        else:
            base = fetch(symbol, cfg.base_interval, cfg.period,
                         start=cfg.start, end=cfg.end,
                         allow_synthetic=False)
            if len(base) > 0 and base.index[0] == _SYNTHETIC_SENTINEL:
                return {
                    "symbol": symbol,
                    "skip_reason": "yfinance returned no data; would have used synthetic",
                }
            tf = multi_timeframe(base, cfg.higher_tfs)

        per_asset_cfg = copy.deepcopy(cfg)
        per_asset_cfg.opt_iterations = iters_per_fold
        wf = walk_forward(tf, per_asset_cfg,
                          n_folds=n_folds, train_frac=train_frac,
                          iters_per_fold=iters_per_fold,
                          min_train_days=min_train_days,
                          train_models=False)

        for p in wf.train_pools_for_ml:
            p.asset = symbol
        for p in wf.oos_pools:
            p.asset = symbol

        final_cfg = copy.deepcopy(cfg)
        final_cfg.opt_iterations = final_iters
        best, _ = optimize(tf, final_cfg)
        pools = project_to_base(build_pools(tf, best), tf["base"].index)
        results = test_pools(tf["base"], pools, best)
        for p in pools:
            p.asset = symbol

        asset = AssetData(
            symbol=symbol, base_df=base, tf_data=tf,
            walkforward=wf, final_cfg=best,
            final_pools=pools, final_results=results,
        )
        payload = {
            "symbol": symbol,
            "skip_reason": None,
            "asset": asset,
            "train_pools": wf.train_pools_for_ml,
            "train_results": wf.train_results_for_ml,
            "oos_pools": wf.oos_pools,
            "oos_results": wf.oos_results,
            "raw_oos_outcomes": wf.raw_oos_outcomes,
            "oos_outcome_counts": wf.oos_outcome_counts,
            "overfit_gap": wf.mean_overfit_gap,
        }
        _save_asset_checkpoint(payload, checkpoint_dir)
        return payload
    except Exception as e:
        source = "parquet" if data_provider is not None else "fetch"
        return {"symbol": symbol, "skip_reason": f"{source} failed: {e}"}


def _merge_asset_payload(report: MultiAssetReport, payload: Dict,
                         all_train_pools: List[Pool],
                         all_train_results: List[PoolResult],
                         all_oos_pools: List[Pool],
                         all_oos_results: List[PoolResult],
                         all_oos_outcomes: List[int],
                         all_oos_outcome_counts: Dict[str, int],
                         overfit_gaps: List[float],
                         asset_dfs: Dict[str, pd.DataFrame]) -> None:
    symbol = payload["symbol"]
    asset = payload["asset"]
    report.assets[symbol] = asset
    asset_dfs[symbol] = asset.base_df
    all_train_pools.extend(payload["train_pools"])
    all_train_results.extend(payload["train_results"])
    all_oos_pools.extend(payload["oos_pools"])
    all_oos_results.extend(payload["oos_results"])
    all_oos_outcomes.extend(payload["raw_oos_outcomes"])
    for k, v in payload["oos_outcome_counts"].items():
        all_oos_outcome_counts[k] = all_oos_outcome_counts.get(k, 0) + v
    overfit_gaps.append(payload["overfit_gap"])

def run_multi_asset(symbols: List[str], cfg: Config,
                    n_folds: int = 5, train_frac: float = 0.6,
                    iters_per_fold: int = 60,
                    final_iters: int = 100,
                    min_train_days: int = 10,
                    data_provider: Optional[DataProvider] = None,
                    asset_workers: int = 1,
                    checkpoint_dir: Optional[str] = None,
                    resume: bool = False,
                    feature_store_dir: Optional[str] = None,
                    progress: Optional[Callable[[str, str], None]] = None,
                    ) -> MultiAssetReport:
    """Run walk-forward per asset (NO per-asset model training), then train unified models on
    the combined cross-asset pool / snapshot sets. Returns a `MultiAssetReport` with all unified
    models ready for `multi_asset_run.py` to consume for cross-asset ranking."""

    report = MultiAssetReport()
    all_train_pools: List[Pool] = []
    all_train_results: List[PoolResult] = []
    all_oos_pools: List[Pool] = []
    all_oos_results: List[PoolResult] = []
    all_oos_outcomes: List[int] = []
    all_oos_outcome_counts: Dict[str, int] = {}
    overfit_gaps: List[float] = []
    asset_dfs: Dict[str, pd.DataFrame] = {}

    pending: List[str] = []
    payload_by_symbol: Dict[str, Dict] = {}
    for symbol in symbols:
        if resume:
            payload = _load_asset_checkpoint(checkpoint_dir, symbol)
            if payload is not None:
                payload_by_symbol[symbol] = payload
                if progress:
                    progress(symbol, "loaded checkpoint")
                continue
        pending.append(symbol)

    asset_workers = max(1, int(asset_workers or 1))
    if pending and asset_workers > 1:
        if progress:
            progress("[assets]", f"parallel run ({asset_workers} workers, {len(pending)} pending)")
        with ProcessPoolExecutor(max_workers=asset_workers) as ex:
            futs = {
                ex.submit(_run_one_asset_payload, symbol, cfg, n_folds, train_frac,
                          iters_per_fold, final_iters, min_train_days,
                          data_provider, checkpoint_dir): symbol
                for symbol in pending
            }
            for fut in as_completed(futs):
                symbol = futs[fut]
                payload = fut.result()
                payload_by_symbol[symbol] = payload
                reason = payload.get("skip_reason")
                if progress:
                    progress(symbol, f"skipped: {reason}" if reason else "completed")
    else:
        for symbol in pending:
            if progress:
                progress(symbol, f"walk-forward + final fit")
            payload = _run_one_asset_payload(
                symbol, cfg, n_folds, train_frac, iters_per_fold,
                final_iters, min_train_days, data_provider, checkpoint_dir,
            )
            payload_by_symbol[symbol] = payload
            reason = payload.get("skip_reason")
            if progress:
                progress(symbol, f"skipped: {reason}" if reason else "completed")

    # Merge in the caller's symbol order for deterministic cross-asset training.
    for symbol in symbols:
        payload = payload_by_symbol.get(symbol)
        if payload is None:
            continue
        reason = payload.get("skip_reason")
        if reason:
            print(f"  [{symbol}] SKIPPED — {reason}")
            report.skipped_symbols.append(symbol)
            report.skip_reasons[symbol] = reason
            continue
        _merge_asset_payload(
            report, payload,
            all_train_pools, all_train_results,
            all_oos_pools, all_oos_results,
            all_oos_outcomes, all_oos_outcome_counts,
            overfit_gaps, asset_dfs,
        )

    if not report.assets:
        msg = ("no assets ran successfully. Check parquet files/data-source settings and "
               "symbol spelling." if data_provider is not None else
               "no assets ran successfully — all symbols either failed to fetch or fell back "
               "to synthetic data. Check yfinance connectivity and symbol spelling.")
        if report.skipped_symbols:
            msg += f"\nSkipped: " + ", ".join(
                f"{s} ({report.skip_reasons.get(s, 'unknown')})"
                for s in report.skipped_symbols
            )
        raise ValueError(msg)

    # ----------------------------------------------------------------------
    # Combined OOS stats
    # ----------------------------------------------------------------------
    report.total_train_pools = len(all_train_pools)
    report.total_oos_pools = len(all_oos_pools)
    report.total_oos_tested = len(all_oos_outcomes)
    if all_oos_outcomes:
        n_resp = sum(all_oos_outcomes)
        report.pooled_oos_respect = n_resp / len(all_oos_outcomes)
        report.pooled_oos_ci_wilson = wilson_score_interval(n_resp, len(all_oos_outcomes), ci=0.90)
        report.pooled_oos_ci_bootstrap = bootstrap_proportion_ci(all_oos_outcomes, ci=0.90,
                                                                  n_boot=2000)
    s_resp = (all_oos_outcome_counts.get("respected_strong", 0)
              + all_oos_outcome_counts.get("swept_and_reclaimed", 0))
    s_break = all_oos_outcome_counts.get("broken_strong", 0)
    report.pooled_oos_strict = (s_resp / (s_resp + s_break)) if (s_resp + s_break) else 0.0
    report.mean_overfit_gap = float(np.mean(overfit_gaps)) if overfit_gaps else 0.0

    # ----------------------------------------------------------------------
    # Unified Featurizer + ML model on combined pool sets
    # ----------------------------------------------------------------------
    if progress:
        progress("[unified]", "training cross-asset sector MoE quality model")
    multi_feat = MultiAssetFeaturizer(asset_dfs)
    report.unified_featurizer = multi_feat

    tr_mask = trainable_mask(all_train_results)
    if tr_mask.sum() >= 30:
        train_pool_keep = [p for p, k in zip(all_train_pools, tr_mask) if k]
        train_result_keep = [r for r, k in zip(all_train_results, tr_mask) if k]
        X_train = multi_feat.transform_batch(train_pool_keep)
        y_train = ml_labels(train_result_keep)
        try:
            X_oos_all = multi_feat.transform_batch(all_oos_pools) if all_oos_pools else None
            base_periods = []
            for df in asset_dfs.values():
                diffs = pd.Series(df.index).diff().dropna()
                if len(diffs):
                    sec = float(diffs.median().total_seconds())
                    if sec > 0:
                        base_periods.append(sec)
            base_period_seconds = float(np.median(base_periods)) if base_periods else 300.0
            ml = SectorMoERespectModel().fit(
                X_train, y_train, train_pool_keep,
                X_oos=X_oos_all, oos_pools=all_oos_pools, oos_results=all_oos_results,
                val_frac=0.25, seed=cfg.opt_seed + 100,
                min_sector_train_n=60, min_sector_class_n=8, min_sector_oos_n=20,
                expert_weight=0.70,
                sample_start_times=[p.available_at for p in train_pool_keep],
                sample_end_times=[label_end_time(p, r)
                                  for p, r in zip(train_pool_keep, train_result_keep)],
                embargo_bars=cfg.embargo_bars,
                base_period_seconds=base_period_seconds,
                validation_method=cfg.validation_method,
                regularization_preset=cfg.regularization_preset,
                bucket_shrinkage_max=getattr(cfg, "q_bucket_shrinkage_max", 1.0),
                train_sector_experts=getattr(cfg, "train_sector_experts", True),
                sample_decay_halflife_days=getattr(
                    cfg, "sample_decay_halflife_days", None),
            )
            report.unified_ml = ml
            report.unified_oos_audit = build_oos_prediction_audit(
                ml, multi_feat, all_oos_pools, all_oos_results, asset_dfs=asset_dfs,
            )
            if report.unified_oos_audit is not None and not report.unified_oos_audit.table.empty:
                # Phase 3B/3C — fit the per-pool learned gate from the OOS
                # audit. The gate is OPTIONAL and FALLBACK-SAFE: when
                # `fit_from_audit` returns None the static per-sector weight
                # stays in effect.
                try:
                    _fit_and_apply_learned_gate(
                        report, ml, all_oos_pools, all_oos_results, asset_dfs,
                        seed=cfg.opt_seed + 200,
                    )
                except Exception as gate_exc:
                    print(f"[multi_asset] learned gate fit skipped: {gate_exc}")

                report.post_touch_reaction_metrics = build_post_touch_reaction_metrics(
                    all_oos_pools, all_oos_results,
                    q_values=report.unified_oos_audit.table["blended_q"].to_numpy(dtype=float),
                )
        except Exception as e:
            print(f"[multi_asset] unified sector MoE training skipped: {e}")

    # ----------------------------------------------------------------------
    # Unified Stratified model on combined OOS
    # ----------------------------------------------------------------------
    if all_oos_pools:
        try:
            report.unified_stratified = StratifiedRespectModel.fit(
                all_oos_pools, all_oos_results, min_sample=8,
            )
        except Exception as e:
            print(f"[multi_asset] unified stratified fit skipped: {e}")

    # ----------------------------------------------------------------------
    # Unified Direction + Proximity models on combined snapshots
    # ----------------------------------------------------------------------
    if report.unified_ml is not None:
        if progress:
            progress("[unified]", "training cross-asset direction + proximity models")
        try:
            PROX_HORIZONS = list(PROXIMITY_HORIZONS)
            DIR_HORIZON = DEFAULT_DIRECTION_HORIZON
            MAX_HORIZON = max(PROX_HORIZONS + [DIR_HORIZON])
            sample_every = max(20, cfg.test_horizon_bars // 6)

            if feature_store_dir:
                if progress:
                    progress("[feature_store]", f"writing shards -> {feature_store_dir}")
                fs_stats = build_feature_store_for_report(
                    report, multi_feat, feature_store_dir,
                    horizons=PROX_HORIZONS,
                    direction_horizon=DIR_HORIZON,
                    sample_every=sample_every,
                    seed=cfg.opt_seed + 500,
                )
                report.feature_store_stats = fs_stats.to_dict()
                store = FeatureStore(feature_store_dir)

                try:
                    if progress:
                        progress("[reaction_model]", "training post-touch reaction suite")
                    reaction_train = store.scan("reaction_events/*/train.parquet")
                    reaction_oos = store.scan("reaction_events/*/oos.parquet")
                    if len(reaction_train) >= 100:
                        suite = ReactionModelSuite().fit(
                            reaction_train, reaction_oos,
                            seed=cfg.opt_seed + 700,
                        )
                        report.reaction_model = suite
                        report.reaction_model_report = suite.report_frame().to_dict(
                            orient="records",
                        )
                        report.reaction_model_calibration = (
                            suite.calibration_table.to_dict(orient="records")
                            if not suite.calibration_table.empty else []
                        )
                        fi = suite.feature_importance_frame(top_k=50)
                        report.reaction_feature_importance = (
                            fi.to_dict(orient="records") if not fi.empty else []
                        )
                    else:
                        print("[multi_asset] reaction model skipped: "
                              f"need >=100 post-touch train events, got {len(reaction_train)}")
                except Exception as e:
                    print(f"[multi_asset] reaction model skipped: {e}")

                direction_train = store.load_direction("train")
                direction_oos = store.load_direction("oos")
                if len(direction_train) >= 30:
                    try:
                        report.unified_direction = DirectionModel(horizon=DIR_HORIZON).fit_frame(
                            direction_train, seed=cfg.opt_seed + 200,
                        )
                    except ValueError as ve:
                        print(f"[multi_asset] feature-store direction skipped: {ve}")

                first_asset_df = next(iter(report.assets.values())).base_df
                _d = pd.Series(first_asset_df.index).diff().dropna()
                bps = float(_d.median().total_seconds()) if len(_d) else 300.0
                if bps <= 0:
                    bps = 300.0

                prox_oos_frames = {}
                for h in PROX_HORIZONS:
                    train_frame = store.load_proximity(h, "train")
                    oos_frame = store.load_proximity(h, "oos")
                    prox_oos_frames[h] = oos_frame
                    try:
                        pm = ProximityModel(horizon=h, base_period_seconds=bps).fit_frame(
                            train_frame, seed=cfg.opt_seed + 300 + h,
                        )
                        report.unified_proximity[h] = pm
                    except ValueError as ve:
                        print(f"[multi_asset] feature-store proximity h={h} skipped: {ve}")

                if report.unified_direction is not None and report.unified_proximity:
                    report.unified_timing_report = evaluate_timing_frames(
                        direction_oos, prox_oos_frames,
                        report.unified_direction, report.unified_proximity,
                    )
                return report

            # Build a GLOBAL pool list across all assets so one DirectionModel/ProximityModel can
            # train on combined snapshots. Snapshot.pool_touches stores pool indices, so we
            # generate per-asset and remap pi → global by adding the asset's offset.
            global_pools: List[Pool] = []
            global_results: List[PoolResult] = []
            global_quality: List[float] = []
            asset_offset: Dict[str, int] = {}
            all_train_snaps: List = []
            all_oos_snaps: List = []

            for symbol, ad in report.assets.items():
                wf = ad.walkforward
                a_pools = list(wf.train_pools_for_ml) + list(wf.oos_pools)
                a_results = list(wf.train_results_for_ml) + list(wf.oos_results)
                a_X = multi_feat.transform_batch(a_pools)
                a_quality = report.unified_ml.predict(a_X, pools=a_pools)

                offset = len(global_pools)
                asset_offset[symbol] = offset
                global_pools.extend(a_pools)
                global_results.extend(a_results)
                global_quality.extend(a_quality)

                state_feat = StateFeaturizer(ad.base_df)
                # Train snapshots once over the largest (final) expanding train window — nested
                # per-fold windows would duplicate early-history snapshots. Future is clipped at
                # the train-window end so no label leaks into OOS. OOS test windows are
                # non-overlapping, so we union them per fold.
                first_train_start = min(fr.train_start for fr in wf.folds)
                last_train_end = max(fr.train_end for fr in wf.folds)
                tr = generate_snapshots(ad.base_df, a_pools, a_results, state_feat,
                                         window_start=first_train_start,
                                         window_end=last_train_end,
                                         sample_every=sample_every,
                                         max_horizon=MAX_HORIZON,
                                         clip_future_to_window=True)  # no train→test label leak
                os_ = []
                for fr in wf.folds:
                    # OOS eval: do NOT clip — the full future horizon is genuinely observable.
                    os_.extend(generate_snapshots(ad.base_df, a_pools, a_results, state_feat,
                                                   window_start=fr.test_start,
                                                   window_end=fr.test_end,
                                                   sample_every=sample_every,
                                                   max_horizon=MAX_HORIZON))
                # Remap pi → global index so all snapshots share one pool indexing.
                for snap_collection in (tr, os_):
                    for snap in snap_collection:
                        snap.pool_touches = [(pi + offset, bt, d, s)
                                              for (pi, bt, d, s) in snap.pool_touches]
                all_train_snaps.extend(tr)
                all_oos_snaps.extend(os_)

            global_quality_arr = np.asarray(global_quality)

            if len(all_train_snaps) >= 30:
                try:
                    dir_m = DirectionModel(horizon=DIR_HORIZON).fit(
                        all_train_snaps, seed=cfg.opt_seed + 200,
                    )
                    report.unified_direction = dir_m
                except ValueError as ve:
                    print(f"[multi_asset] unified direction skipped: {ve}")

                # Use the first asset's base_period for proximity (they're all 5m on NSE)
                first_asset_df = next(iter(report.assets.values())).base_df
                _d = pd.Series(first_asset_df.index).diff().dropna()
                bps = float(_d.median().total_seconds()) if len(_d) else 300.0
                if bps <= 0:
                    bps = 300.0
                for h in PROX_HORIZONS:
                    try:
                        pm = ProximityModel(horizon=h, base_period_seconds=bps).fit(
                            all_train_snaps, global_pools, global_quality_arr,
                            seed=cfg.opt_seed + 300 + h,
                        )
                        report.unified_proximity[h] = pm
                    except ValueError as ve:
                        print(f"[multi_asset] unified proximity h={h} skipped: {ve}")

                if (report.unified_direction is not None and report.unified_proximity
                        and all_oos_snaps):
                    report.unified_timing_report = evaluate_timing(
                        all_oos_snaps, global_pools, global_results,
                        report.unified_direction, report.unified_proximity, global_quality_arr,
                    )
        except Exception as e:
            print(f"[multi_asset] unified timing models skipped: {e}")

    return report


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def print_multi_asset_summary(report: MultiAssetReport, file=None) -> None:
    print("\n================ MULTI-ASSET SUMMARY ================", file=file)
    print(f"  Assets:                 {', '.join(report.assets.keys())}", file=file)
    print(f"  Total train pools:      {report.total_train_pools}", file=file)
    print(f"  Total OOS pools:        {report.total_oos_pools}", file=file)
    print(f"  Total OOS tested:       {report.total_oos_tested}", file=file)
    print(f"  Pooled OOS broad:       {report.pooled_oos_respect:.1%}   "
          f"Wilson CI {report.pooled_oos_ci_wilson[0]:.1%}–"
          f"{report.pooled_oos_ci_wilson[1]:.1%}", file=file)
    print(f"  {'':<22}  "
          f"Boot   CI {report.pooled_oos_ci_bootstrap[0]:.1%}–"
          f"{report.pooled_oos_ci_bootstrap[1]:.1%}", file=file)
    print(f"  Pooled OOS strict:      {report.pooled_oos_strict:.1%}", file=file)
    print(f"  Mean per-asset overfit gap: {report.mean_overfit_gap:+.1%}", file=file)

    # -- Per-asset breakdown --
    print(f"\n[per-asset OOS]   (tighter slices, smaller samples per row)", file=file)
    print(f"  {'symbol':<14} {'oos_n':>6} {'tested':>7} {'respect':>8} {'strict':>8} "
          f"{'gap':>7}", file=file)
    for sym, ad in report.assets.items():
        wf = ad.walkforward
        outs = wf.raw_oos_outcomes
        n = len(outs)
        rate = (sum(outs) / n) if n else 0.0
        srh = sum(1 for r in wf.oos_results
                   if r.outcome in ("respected_strong", "swept_and_reclaimed"))
        sbk = sum(1 for r in wf.oos_results if r.outcome == "broken_strong")
        strict = srh / (srh + sbk) if (srh + sbk) else 0.0
        print(f"  {sym:<14} {wf.n_total_oos_pools:>6} {n:>7} {rate:>7.1%} {strict:>7.1%} "
              f"{wf.mean_overfit_gap:>+6.1%}", file=file)

    # -- Combined OOS outcome breakdown (all assets pooled) --
    counts: Dict[str, int] = {}
    for ad in report.assets.values():
        for k, v in ad.walkforward.oos_outcome_counts.items():
            counts[k] = counts.get(k, 0) + v
    if counts:
        total = sum(counts.values())
        print(f"\n[combined OOS outcome breakdown]   {total} eligible pools across assets",
              file=file)
        for k in ("respected_strong", "swept_and_reclaimed", "respected_weak",
                  "broken_weak", "broken_strong",
                  "touched_no_signal", "untouched"):
            v = counts.get(k, 0)
            pct = (v / total * 100) if total else 0.0
            print(f"  {k:<22} {v:>5}  ({pct:>4.1f}%)", file=file)

    # -- Unified ML model fit + calibration --
    if report.unified_ml is not None:
        m = report.unified_ml
        model_name = ("sector MoE quality model"
                      if isinstance(m, SectorMoERespectModel)
                      else "unified ML quality model")
        print(f"\n[{model_name}]   train n={m.train_n} val n={m.val_n}", file=file)
        print(f"  Validation method:       {getattr(m, 'validation_method', 'unknown')}", file=file)
        print(f"  Regularization preset:   {getattr(m, 'regularization_preset', 'default')}", file=file)
        print(f"  Train Brier / log-loss: {getattr(m, 'train_brier', 0.0):.4f} / "
              f"{getattr(m, 'train_logloss', 0.0):.4f}", file=file)
        print(f"  Val Brier / log-loss:   {m.val_brier:.4f} / {m.val_logloss:.4f}", file=file)
        print(f"  Fit gap (Brier):         "
              f"{getattr(m, 'val_brier', 0.0) - getattr(m, 'train_brier', 0.0):+.4f}",
              file=file)
        print(f"  Val AUC-ROC:            {m.val_auc:.3f}", file=file)
        print(f"  Base rate:              {m.base_rate:.1%}", file=file)
        if getattr(m, "validation_fold_stats", None):
            print(f"\n[purged/embargoed quality validation folds]", file=file)
            print(f"  {'fold':>4} {'orig_tr':>8} {'purged_tr':>9} {'purged':>7} "
                  f"{'embargo':>8} {'val':>6}", file=file)
            for st in m.validation_fold_stats:
                print(f"  {st.fold + 1:>4} {st.original_train_size:>8} "
                      f"{st.purged_train_size:>9} {st.purged_rows_removed:>7} "
                      f"{st.embargoed_rows_removed:>8} {st.validation_size:>6}", file=file)
        if isinstance(m, SectorMoERespectModel):
            print(f"  Sector expert cap:      {m.expert_weight:.0%} "
                  f"(actual weights are dynamically shrunk)", file=file)
            if m.sector_stats:
                print(f"\n[sector expert dynamic shrinkage]", file=file)
                print(f"  {'sector':<10} {'status':<8} {'train':>6} {'expert_ll':>9} "
                      f"{'global_ll':>9} {'auc':>6} {'weight':>7} {'reason':<28}",
                      file=file)
                for sector, st in m.expert_summary():
                    auc = st.get("val_auc")
                    auc_s = f"{auc:.3f}" if isinstance(auc, float) else "-"
                    ell = st.get("expert_logloss")
                    gll = st.get("global_logloss")
                    ell_s = f"{ell:.4f}" if isinstance(ell, float) else "-"
                    gll_s = f"{gll:.4f}" if isinstance(gll, float) else "-"
                    print(f"  {sector:<10} {st.get('status', '-'):<8} "
                          f"{st.get('train_n', 0):>6} {ell_s:>9} {gll_s:>9} "
                          f"{auc_s:>6} {st.get('sector_weight', 0.0):>6.1%} "
                          f"{st.get('reason', ''):<28}", file=file)
        print(f"\n[unified ML feature importance — top 15]", file=file)
        for name, gain in m.feature_importance(15):
            print(f"  {name:<34} {gain:>10.1f}", file=file)
        if m.bucket_calib:
            print(f"\n[unified bucket-level OOS shrinkage]", file=file)
            print(f"  {'TFs':<5} {'factor':<8} {'n_oos':>6} {'emp_rate':>9} {'pull':>6}",
                  file=file)
            for (tfb, fam), bc in sorted(m.bucket_calib.items(),
                                          key=lambda kv: -kv[1].pull_weight):
                print(f"  {tfb:<5} {fam:<8} {bc.n_oos:>6} {bc.empirical_rate:>8.1%} "
                      f"{bc.pull_weight:>5.2f}", file=file)

    # -- Phase 3A: honest OOS prediction audit for global / sector / blended Q --
    if report.unified_oos_audit is not None and report.unified_oos_audit.metrics:
        audit = report.unified_oos_audit
        print(f"\n[Phase 3A OOS prediction audit]", file=file)
        print(f"  {'model':<14} {'n':>6} {'brier':>8} {'logloss':>8} {'auc':>6} "
              f"{'top10_hit':>10} {'base':>7} {'mean_q':>7}", file=file)
        for name in ("global", "sector_expert", "blended", "learned_blended"):
            mt = audit.metrics.get(name)
            if mt is None:
                continue
            auc_s = f"{mt.auc:.3f}" if mt.auc is not None else "n/a"
            print(f"  {name:<14} {mt.n:>6} {mt.brier:>8.4f} {mt.logloss:>8.4f} "
                  f"{auc_s:>6} {mt.top_decile_hit_rate:>9.1%} "
                  f"{mt.base_rate:>6.1%} {mt.mean_prediction:>6.1%}", file=file)
        # Phase 3B/3C — show whether the learned dynamic gate was fit, and how
        # much it beat the static blend. Blank when the gate fell back.
        gate = report.unified_learned_gate
        if gate is not None and gate.stats is not None:
            st = gate.stats
            pearson_s = f"{st.val_pearson:+.3f}" if st.val_pearson is not None else "n/a"
            print(f"\n[Phase 3C learned dynamic gate]  val MAE vs constant baseline",
                  file=file)
            print(f"  trained n={st.n_train}, val n={st.n_val}, "
                  f"target_mean={st.mean_target:.3f}", file=file)
            print(f"  val_mae={st.val_mae:.4f}  baseline={st.baseline_val_mae:.4f}  "
                  f"Δ={st.mae_improvement:+.4f}  pearson={pearson_s}", file=file)
            print(f"  weight clip = [{st.target_floor:.2f}, {st.target_ceiling:.2f}], "
                  f"disagreement floor = {st.disagreement_floor:.3f}", file=file)
            for fi in gate.feature_importance(8):
                print(f"    {fi['feature']:<28} {fi['importance_gain']:>10.1f}",
                      file=file)
        elif report.unified_oos_audit is not None and not report.unified_oos_audit.table.empty:
            print(f"\n[Phase 3C learned dynamic gate]  not fit — "
                  f"static per-sector weight retained (Phase 3E fallback)", file=file)
        if not audit.sector_calibration.empty:
            print(f"\n[Phase 3A per-sector calibration]  mean_q - actual", file=file)
            print(f"  {'model':<14} {'sector':<10} {'n':>6} {'actual':>8} "
                  f"{'mean_q':>8} {'err':>8}", file=file)
            cal = audit.sector_calibration.copy()
            cal["abs_err"] = cal["calibration_error"].abs()
            for _, row in cal.sort_values(["model", "abs_err"], ascending=[True, False]).iterrows():
                print(f"  {row['model']:<14} {row['sector']:<10} {int(row['n']):>6} "
                      f"{row['actual_rate']:>7.1%} {row['mean_prediction']:>7.1%} "
                      f"{row['calibration_error']:>+7.1%}", file=file)
        if audit.distance_bucket_metrics:
            print(f"\n[quality distance-binned OOS metrics]  blended Q", file=file)
            print(f"  {'bucket':<9} {'n':>6} {'base':>8} {'AUC':>6} {'Brier':>8} "
                  f"{'logloss':>8} {'cal_err':>8}", file=file)
            for row in audit.distance_bucket_metrics.get("blended", []):
                auc_s = f"{row['auc']:.3f}" if row["auc"] is not None else "n/a"
                print(f"  {row['bucket']:<9} {row['n']:>6} {row['base_rate']:>7.1%} "
                      f"{auc_s:>6} {row['brier']:>8.4f} {row['logloss']:>8.4f} "
                      f"{row['calibration_error']:>+7.1%}", file=file)

    # -- Unified Stratified bucket model --
    if report.unified_stratified is not None:
        from .stratified import print_model as _print_strat
        _print_strat(report.unified_stratified, file=file)

    if report.post_touch_reaction_metrics:
        pt = report.post_touch_reaction_metrics
        overall = pt.get("overall", {})
        print(f"\n================ POST-TOUCH REACTION QUALITY ================",
              file=file)
        print(f"  touched n:              {overall.get('n', 0)}", file=file)
        print(f"  respected_strong:       {overall.get('respected_strong_rate', 0.0):.1%}",
              file=file)
        print(f"  swept_and_reclaimed:    {overall.get('swept_and_reclaimed_rate', 0.0):.1%}",
              file=file)
        print(f"  broken_strong:          {overall.get('broken_strong_rate', 0.0):.1%}",
              file=file)
        print(f"  strict respect:         {overall.get('strict_respect_rate', 0.0):.1%}",
              file=file)
        print(f"  avg MAE after touch:    {overall.get('avg_mae_after_touch', 0.0):.2f} ATR",
              file=file)
        print(f"  avg MFE after touch:    {overall.get('avg_mfe_after_touch', 0.0):.2f} ATR",
              file=file)
        for group in ("factor_type", "tf_count", "sector", "direction_alignment", "q_bucket"):
            rows = pt.get("by", {}).get(group, {})
            if not rows:
                continue
            print(f"\n[post-touch by {group}]", file=file)
            print(f"  {'bucket':<14} {'n':>6} {'strict':>8} {'strong':>8} "
                  f"{'swept':>8} {'broken':>8} {'MFE':>7} {'MAE':>7}", file=file)
            for name, row in sorted(rows.items(), key=lambda kv: -kv[1].get("n", 0)):
                print(f"  {name:<14} {row.get('n', 0):>6} "
                      f"{row.get('strict_respect_rate', 0.0):>7.1%} "
                      f"{row.get('respected_strong_rate', 0.0):>7.1%} "
                      f"{row.get('swept_and_reclaimed_rate', 0.0):>7.1%} "
                      f"{row.get('broken_strong_rate', 0.0):>7.1%} "
                      f"{row.get('avg_mfe_after_touch', 0.0):>6.2f} "
                      f"{row.get('avg_mae_after_touch', 0.0):>6.2f}", file=file)

    reaction_model_report = getattr(report, "reaction_model_report", [])
    if reaction_model_report:
        print(f"\n[Phase 3 reaction model suite]  target = post-touch confirmation outcome",
              file=file)
        print(f"  {'target':<20} {'train':>7} {'val':>7} {'oos':>7} "
              f"{'base':>7} {'val_auc':>8} {'oos_auc':>8} "
              f"{'oos_brier':>10} {'top10':>8}", file=file)
        for row in reaction_model_report:
            val_auc = row.get("val_auc")
            oos_auc = row.get("oos_auc")
            val_s = f"{val_auc:.3f}" if isinstance(val_auc, float) else "n/a"
            oos_s = f"{oos_auc:.3f}" if isinstance(oos_auc, float) else "n/a"
            print(f"  {row.get('target', ''):<20} {int(row.get('train_n', 0)):>7} "
                  f"{int(row.get('val_n', 0)):>7} {int(row.get('oos_n', 0)):>7} "
                  f"{row.get('base_rate', 0.0):>6.1%} {val_s:>8} {oos_s:>8} "
                  f"{row.get('oos_brier', 0.0):>10.4f} "
                  f"{row.get('oos_top_decile_rate', 0.0):>7.1%}", file=file)

    # -- Unified direction + proximity models --
    if report.unified_timing_report is not None:
        tr = report.unified_timing_report
        print(f"\n[unified direction model]  horizon={tr.direction_horizon}b", file=file)
        print(f"  OOS samples:            {tr.direction_n}", file=file)
        print(f"  Brier / log-loss:       {tr.direction_brier:.4f} / "
              f"{tr.direction_logloss:.4f}", file=file)
        print(f"  OOS AUC:                {tr.direction_auc:.3f}", file=file)
        print(f"  Top-quartile-confidence:{tr.direction_top_quartile_acc:.1%}", file=file)

        print(f"\n[unified proximity models]", file=file)
        for s in tr.proximity_per_horizon:
            lift_str = "inf" if s.decile_lift == float("inf") else f"{s.decile_lift:.1f}x"
            print(f"  h={s.horizon:<4} AUC={s.auc:.3f}  brier={s.brier:.4f}  "
                  f"base_rate={s.base_rate:.1%}  decile_lift={lift_str}  n={s.n}", file=file)
            if getattr(s, "distance_bucket_metrics", None):
                print(f"    {'bucket':<9} {'n':>6} {'base':>8} {'AUC':>6} {'Brier':>8} "
                      f"{'logloss':>8} {'cal_err':>8}", file=file)
                for row in s.distance_bucket_metrics:
                    auc_s = f"{row['auc']:.3f}" if row["auc"] is not None else "n/a"
                    print(f"    {row['bucket']:<9} {row['n']:>6} {row['base_rate']:>7.1%} "
                          f"{auc_s:>6} {row['brier']:>8.4f} {row['logloss']:>8.4f} "
                          f"{row['calibration_error']:>+7.1%}", file=file)

        # Feature importances
        if report.unified_direction is not None:
            print(f"\n[unified direction model — top 8 features]", file=file)
            for name, gain in report.unified_direction.feature_importance(8):
                print(f"  {name:<32} {gain:>10.1f}", file=file)
        if report.unified_proximity:
            shortest = min(report.unified_proximity.keys())
            pm = report.unified_proximity[shortest]
            print(f"\n[unified proximity h={shortest} — top 8 features]", file=file)
            for name, gain in pm.feature_importance(8):
                print(f"  {name:<32} {gain:>10.1f}", file=file)

    if report.feature_store_stats:
        fs = report.feature_store_stats
        print(f"\n[feature store]", file=file)
        print(f"  root:                 {fs.get('root')}", file=file)
        print(f"  quality rows:         {fs.get('quality_rows', 0)}", file=file)
        print(f"  post-touch rows:      {fs.get('post_touch_rows', 0)}", file=file)
        for key, n in sorted(fs.get("proximity_rows", {}).items()):
            cand = fs.get("proximity_candidates", {}).get(key, n)
            kept = fs.get("proximity_kept", {}).get(key, n)
            keep_rate = (kept / cand) if cand else 0.0
            print(f"  proximity {key:<10} {n:>8} rows kept ({keep_rate:.1%})", file=file)
