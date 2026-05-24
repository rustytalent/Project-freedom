"""Partitioned parquet feature store for memory-safe multi-asset training.

The first implementation keeps existing pandas detector/model code, but changes the heavy
cross-asset timing layer from "one giant in-memory Snapshot list" into persisted parquet shards.
Polars is used opportunistically for lazy scans when installed; pandas/pyarrow remains the
fallback so the existing environment keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math

import numpy as np
import pandas as pd

from .ml_model import labels as ml_labels, trainable_mask
from .pools import Pool
from .sectors import sector_of
from .stratified import _headline_factor, _tf_bucket
from .tester import PoolResult
from .timing import (
    STATE_FEATURE_NAMES,
    _POOL_FEATURE_NAMES,
    _pool_features_for_snapshot,
    generate_snapshots,
    StateFeaturizer,
)


PROXIMITY_HORIZONS = (78, 156, 312)
DEFAULT_DIRECTION_HORIZON = 78


@dataclass
class FeatureStoreStats:
    root: str
    symbols: List[str] = field(default_factory=list)
    bar_rows: int = 0
    pool_rows: int = 0
    quality_rows: int = 0
    post_touch_rows: int = 0
    direction_rows: Dict[str, int] = field(default_factory=dict)
    proximity_rows: Dict[str, int] = field(default_factory=dict)
    proximity_candidates: Dict[str, int] = field(default_factory=dict)
    proximity_kept: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class FeatureStore:
    """Small parquet writer/reader around a feature-store root directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def mkdir(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _write(self, df: pd.DataFrame, rel: str) -> int:
        if df is None or df.empty:
            return 0
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return int(len(df))

    def write_manifest(self, stats: FeatureStoreStats, extra: Optional[Dict] = None) -> None:
        self.mkdir()
        payload = stats.to_dict()
        if extra:
            payload.update(extra)
        (self.root / "manifest.json").write_text(json.dumps(payload, indent=2, default=str))

    def write_bars(self, symbol: str, tf_data: Dict[str, pd.DataFrame]) -> int:
        total = 0
        for tf_name, df in tf_data.items():
            if df is None or df.empty:
                continue
            key = "5m" if tf_name == "base" else str(tf_name).replace("min", "m")
            out = df.reset_index().rename(columns={"index": "ts"}).copy()
            if "ts" not in out.columns:
                out = out.rename(columns={out.columns[0]: "ts"})
            out["ts"] = pd.to_datetime(out["ts"])
            out["symbol"] = symbol
            out["timeframe"] = key
            for year, ydf in out.groupby(out["ts"].dt.year):
                rel = f"bars/timeframe={key}/symbol={symbol}/year={int(year)}/part.parquet"
                total += self._write(ydf, rel)
        return total

    def write_pools(self, symbol: str, split: str,
                    pools: Sequence[Pool], results: Sequence[PoolResult]) -> int:
        rows = []
        for i, (pool, result) in enumerate(zip(pools, results)):
            rows.append(pool_result_row(symbol, split, i, pool, result))
        return self._write(pd.DataFrame(rows), f"pools/symbol={symbol}/{split}.parquet")

    def write_quality(self, symbol: str, split: str, X: pd.DataFrame,
                      pools: Sequence[Pool], results: Sequence[PoolResult]) -> int:
        if X is None or X.empty:
            return 0
        rows = X.copy()
        rows.insert(0, "symbol", symbol)
        rows.insert(1, "split", split)
        rows.insert(2, "pool_idx", np.arange(len(rows)))
        rows["sector"] = sector_of(symbol)
        rows["outcome"] = [r.outcome for r in results]
        rows["trainable"] = trainable_mask(results)
        label_arr = np.full(len(rows), np.nan, dtype=float)
        if len(rows):
            mask = trainable_mask(results)
            if mask.any():
                label_arr[mask] = ml_labels([r for r, keep in zip(results, mask) if keep])
        rows["respect_label"] = label_arr
        rows["available_at"] = [str(p.available_at) for p in pools]
        return self._write(rows, f"quality/symbol={symbol}/{split}.parquet")

    def write_reaction_events(self, symbol: str, split: str,
                              pools: Sequence[Pool], results: Sequence[PoolResult]) -> int:
        rows = []
        for i, (pool, result) in enumerate(zip(pools, results)):
            if result.touched_at is None:
                continue
            rows.append(post_touch_event_row(symbol, split, i, pool, result))
        return self._write(
            pd.DataFrame(rows),
            f"reaction_events/symbol={symbol}/{split}.parquet",
        )

    def write_timing_shards(self, symbol: str, split: str, fold_id: int,
                            snapshots, pools: Sequence[Pool],
                            results: Sequence[PoolResult],
                            quality_preds: Sequence[float],
                            horizons: Sequence[int],
                            direction_horizon: int,
                            base_period_seconds: float,
                            rng: np.random.Generator) -> Tuple[Dict[int, int], int, Dict[int, int]]:
        direction_rows = direction_rows_from_snapshots(symbol, split, fold_id,
                                                       snapshots, direction_horizon)
        direction_n = self._write(
            pd.DataFrame(direction_rows),
            f"direction/symbol={symbol}/split={split}/fold={fold_id}.parquet",
        )

        kept: Dict[int, int] = {}
        candidates: Dict[int, int] = {}
        for horizon in horizons:
            prox_rows, n_candidates = proximity_rows_from_snapshots(
                symbol, split, fold_id, snapshots, pools, results,
                quality_preds, horizon, base_period_seconds, rng,
            )
            kept[horizon] = self._write(
                pd.DataFrame(prox_rows),
                f"proximity/horizon={int(horizon)}/symbol={symbol}/split={split}/fold={fold_id}.parquet",
            )
            candidates[horizon] = int(n_candidates)
        return kept, direction_n, candidates

    def scan(self, rel_glob: str) -> pd.DataFrame:
        pattern = str(self.root / rel_glob)
        try:
            import polars as pl  # type: ignore
            return pl.scan_parquet(pattern).collect().to_pandas()
        except Exception:
            paths = sorted(self.root.glob(rel_glob))
            if not paths:
                return pd.DataFrame()
            return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    def load_direction(self, split: str) -> pd.DataFrame:
        return self.scan(f"direction/*/split={split}/*.parquet")

    def load_proximity(self, horizon: int, split: str) -> pd.DataFrame:
        return self.scan(f"proximity/horizon={int(horizon)}/*/split={split}/*.parquet")


def pool_result_row(symbol: str, split: str, pool_idx: int,
                    pool: Pool, result: PoolResult) -> Dict:
    return {
        "symbol": symbol,
        "sector": sector_of(symbol),
        "split": split,
        "pool_idx": int(pool_idx),
        "side": pool.side,
        "price_low": float(pool.price_low),
        "price_high": float(pool.price_high),
        "mid": float(pool.mid),
        "width": float(pool.width),
        "score": float(pool.score),
        "formed_at": str(pool.formed_at),
        "available_at": str(pool.available_at),
        "tfs": "+".join(sorted(set(pool.tfs))),
        "tf_bucket": _tf_bucket(len(set(pool.tfs))),
        "headline_factor": _headline_factor(pool),
        "n_contributors": int(len(pool.contributors)),
        "outcome": result.outcome,
        "touched_at": str(result.touched_at) if result.touched_at is not None else None,
        "broken_at": str(result.broken_at) if result.broken_at is not None else None,
        "bars_to_touch": result.bars_to_touch,
        "bars_to_break": result.bars_to_break,
        "mae_after_touch_atr": float(result.max_excursion_through),
        "mfe_after_touch_atr": float(result.reaction_atr),
    }


def post_touch_label(result: PoolResult) -> str:
    if result.outcome == "respected_strong":
        return "HARD_REJECT"
    if result.outcome == "swept_and_reclaimed":
        return "SWEEP_RECLAIM"
    if result.outcome == "respected_weak":
        return "ABSORPTION"
    if result.outcome in ("broken_strong", "broken_weak"):
        return "FAIL_CONTINUE"
    if result.outcome == "touched_no_signal":
        return "NO_SIGNAL"
    return "LIQUIDITY_VACUUM"


def post_touch_event_row(symbol: str, split: str, pool_idx: int,
                         pool: Pool, result: PoolResult) -> Dict:
    row = pool_result_row(symbol, split, pool_idx, pool, result)
    row["reaction_label"] = post_touch_label(result)
    row["strict_respect_label"] = (
        1 if result.outcome in ("respected_strong", "swept_and_reclaimed")
        else 0 if result.outcome in ("broken_strong", "broken_weak") else np.nan
    )
    return row


def direction_rows_from_snapshots(symbol: str, split: str, fold_id: int,
                                  snapshots, direction_horizon: int) -> List[Dict]:
    rows: List[Dict] = []
    for snap in snapshots:
        label = snap.direction_label(direction_horizon)
        if snap.n_future_bars < direction_horizon or label is None:
            continue
        row = {
            "symbol": symbol,
            "sector": sector_of(symbol),
            "split": split,
            "fold": int(fold_id),
            "ts": str(snap.ts),
            "bar_idx": int(snap.bar_idx),
            "direction_label": int(label),
            "max_up_atr": float(snap.max_up_atr(direction_horizon)),
            "max_dn_atr": float(snap.max_dn_atr(direction_horizon)),
        }
        row.update({name: float(snap.state.get(name, 0.0)) for name in STATE_FEATURE_NAMES})
        rows.append(row)
    return rows


def _keep_proximity_row(touched: int, dist: float, rng: np.random.Generator) -> bool:
    if touched == 1:
        return True
    if dist < 3.0:
        return True
    if dist < 5.0:
        return bool(rng.random() < 0.50)
    if dist < 10.0:
        return bool(rng.random() < 0.20)
    return bool(rng.random() < 0.05)


def proximity_rows_from_snapshots(symbol: str, split: str, fold_id: int,
                                  snapshots, pools: Sequence[Pool],
                                  results: Sequence[PoolResult],
                                  quality_preds: Sequence[float],
                                  horizon: int,
                                  base_period_seconds: float,
                                  rng: np.random.Generator) -> Tuple[List[Dict], int]:
    rows: List[Dict] = []
    candidates = 0
    for snap in snapshots:
        if snap.n_future_bars < horizon:
            continue
        for pi, touched, dist, side in snap.pool_touch_labels(horizon):
            candidates += 1
            if not _keep_proximity_row(touched, dist, rng):
                continue
            pool = pools[pi]
            result = results[pi]
            pool_feat = _pool_features_for_snapshot(
                pool, dist, side, float(quality_preds[pi]),
                base_period_seconds=base_period_seconds,
            )
            row = {
                "symbol": symbol,
                "sector": sector_of(symbol),
                "split": split,
                "fold": int(fold_id),
                "ts": str(snap.ts),
                "bar_idx": int(snap.bar_idx),
                "pool_idx": int(pi),
                "horizon": int(horizon),
                "touch_label": int(touched),
                "respect_label": (
                    1 if result.is_respect else 0 if result.is_break else np.nan
                ),
                "outcome": result.outcome,
            }
            row.update({name: float(snap.state.get(name, 0.0)) for name in STATE_FEATURE_NAMES})
            row.update(pool_feat)
            rows.append(row)
    return rows, candidates


def build_feature_store_for_report(report, multi_feat,
                                   root: str | Path,
                                   horizons: Sequence[int] = PROXIMITY_HORIZONS,
                                   direction_horizon: int = DEFAULT_DIRECTION_HORIZON,
                                   sample_every: int = 20,
                                   seed: int = 11) -> FeatureStoreStats:
    """Persist bars, pools, quality rows, timing shards, and post-touch events for a report."""
    store = FeatureStore(root)
    store.mkdir()
    stats = FeatureStoreStats(root=str(store.root), symbols=list(report.assets.keys()))
    rng = np.random.default_rng(seed)

    for symbol, ad in report.assets.items():
        stats.bar_rows += store.write_bars(symbol, ad.tf_data)

        train_pools = list(ad.walkforward.train_pools_for_ml)
        train_results = list(ad.walkforward.train_results_for_ml)
        oos_pools = list(ad.walkforward.oos_pools)
        oos_results = list(ad.walkforward.oos_results)
        stats.pool_rows += store.write_pools(symbol, "train", train_pools, train_results)
        stats.pool_rows += store.write_pools(symbol, "oos", oos_pools, oos_results)
        stats.post_touch_rows += store.write_reaction_events(symbol, "train", train_pools, train_results)
        stats.post_touch_rows += store.write_reaction_events(symbol, "oos", oos_pools, oos_results)

        if train_pools:
            X_train = multi_feat.transform_batch(train_pools)
            stats.quality_rows += store.write_quality(symbol, "train", X_train, train_pools, train_results)
        if oos_pools:
            X_oos = multi_feat.transform_batch(oos_pools)
            stats.quality_rows += store.write_quality(symbol, "oos", X_oos, oos_pools, oos_results)

        if report.unified_ml is None:
            continue

        a_pools = train_pools + oos_pools
        a_results = train_results + oos_results
        if not a_pools:
            continue
        a_X = multi_feat.transform_batch(a_pools)
        a_quality = report.unified_ml.predict(a_X, pools=a_pools)
        state_feat = StateFeaturizer(ad.base_df)
        base_diffs = pd.Series(ad.base_df.index).diff().dropna()
        base_period_seconds = float(base_diffs.median().total_seconds()) if len(base_diffs) else 300.0
        if not math.isfinite(base_period_seconds) or base_period_seconds <= 0:
            base_period_seconds = 300.0

        for fold_id, fr in enumerate(ad.walkforward.folds):
            max_h = max(list(horizons) + [direction_horizon])
            tr = generate_snapshots(
                ad.base_df, a_pools, a_results, state_feat,
                window_start=fr.train_start, window_end=fr.train_end,
                sample_every=sample_every, max_horizon=max_h,
            )
            os_ = generate_snapshots(
                ad.base_df, a_pools, a_results, state_feat,
                window_start=fr.test_start, window_end=fr.test_end,
                sample_every=sample_every, max_horizon=max_h,
            )
            for split, snaps in (("train", tr), ("oos", os_)):
                prox_kept, dir_n, prox_candidates = store.write_timing_shards(
                    symbol, split, fold_id, snaps, a_pools, a_results, a_quality,
                    horizons, direction_horizon, base_period_seconds, rng,
                )
                stats.direction_rows[split] = stats.direction_rows.get(split, 0) + dir_n
                for horizon in horizons:
                    key = f"{split}_h{int(horizon)}"
                    stats.proximity_rows[key] = stats.proximity_rows.get(key, 0) + prox_kept[horizon]
                    stats.proximity_kept[key] = stats.proximity_kept.get(key, 0) + prox_kept[horizon]
                    stats.proximity_candidates[key] = (
                        stats.proximity_candidates.get(key, 0) + prox_candidates[horizon]
                    )

    store.write_manifest(stats, extra={
        "direction_horizon": int(direction_horizon),
        "proximity_horizons": [int(h) for h in horizons],
        "sample_every": int(sample_every),
        "proximity_policy": "keep positives and <3ATR; sample 3-5ATR at 50%, 5-10ATR at 20%, 10+ATR at 5%",
    })
    return stats


def load_feature_frame(root: str | Path, rel_glob: str) -> pd.DataFrame:
    return FeatureStore(root).scan(rel_glob)
