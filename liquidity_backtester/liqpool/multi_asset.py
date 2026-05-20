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
from typing import Dict, List, Optional, Tuple, Callable, Union
import copy
import numpy as np
import pandas as pd

from .config import Config
from .data import fetch, multi_timeframe
from .pools import build_pools, project_to_base, Pool
from .tester import test_pools, summarise, PoolResult
from .optimizer import optimize
from .walkforward import walk_forward, WalkForwardReport
from .featurize import MultiAssetFeaturizer
from .ml_model import (PoolRespectModel, SectorMoERespectModel,
                       trainable_mask, labels as ml_labels)
from .stratified import StratifiedRespectModel
from .timing import (StateFeaturizer, generate_snapshots, DirectionModel, ProximityModel,
                     evaluate_timing, TimingReport)
from .stats import wilson_score_interval, bootstrap_proportion_ci


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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_multi_asset(symbols: List[str], cfg: Config,
                    n_folds: int = 5, train_frac: float = 0.6,
                    iters_per_fold: int = 60,
                    final_iters: int = 100,
                    min_train_days: int = 10,
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

    # Synthetic-data sentinel: the _synthetic() fallback in data.py always builds an index
    # starting at 2024-01-02 09:30:00. If we see that timestamp at base.index[0], the asset
    # didn't have real data and was filled in with random-walk. Skip those — mixing synthetic
    # data into the unified training would pollute the model AND corrupt cross-asset sector
    # metrics (date-range mismatches give wrong correlation matrices).
    _SYNTHETIC_SENTINEL = pd.Timestamp("2024-01-02 09:30:00")

    for symbol in symbols:
        if progress:
            progress(symbol, "fetch")
        try:
            base = fetch(symbol, cfg.base_interval, cfg.period)
        except Exception as e:
            print(f"  [{symbol}] SKIPPED — fetch failed: {e}")
            report.skipped_symbols.append(symbol)
            report.skip_reasons[symbol] = f"fetch failed: {e}"
            continue
        if len(base) > 0 and base.index[0] == _SYNTHETIC_SENTINEL:
            print(f"  [{symbol}] SKIPPED — yfinance returned no data; "
                  f"refusing to train on synthetic fallback (would pollute basket)")
            report.skipped_symbols.append(symbol)
            report.skip_reasons[symbol] = "yfinance returned no data; would have used synthetic"
            continue
        tf = multi_timeframe(base, cfg.higher_tfs)
        asset_dfs[symbol] = base

        if progress:
            progress(symbol, f"walk-forward ({n_folds} folds × {iters_per_fold} iters)")
        per_asset_cfg = copy.deepcopy(cfg)
        per_asset_cfg.opt_iterations = iters_per_fold
        wf = walk_forward(tf, per_asset_cfg,
                           n_folds=n_folds, train_frac=train_frac,
                           iters_per_fold=iters_per_fold,
                           min_train_days=min_train_days,
                           train_models=False)

        # Tag every pool with its asset symbol BEFORE we combine across assets.
        for p in wf.train_pools_for_ml:
            p.asset = symbol
        for p in wf.oos_pools:
            p.asset = symbol

        all_train_pools.extend(wf.train_pools_for_ml)
        all_train_results.extend(wf.train_results_for_ml)
        all_oos_pools.extend(wf.oos_pools)
        all_oos_results.extend(wf.oos_results)
        all_oos_outcomes.extend(wf.raw_oos_outcomes)
        for k, v in wf.oos_outcome_counts.items():
            all_oos_outcome_counts[k] = all_oos_outcome_counts.get(k, 0) + v
        overfit_gaps.append(wf.mean_overfit_gap)

        # Final-config fit on full history (this is what's used to surface "today's pools").
        if progress:
            progress(symbol, f"final fit ({final_iters} iters)")
        final_cfg = copy.deepcopy(cfg)
        final_cfg.opt_iterations = final_iters
        best, _ = optimize(tf, final_cfg)
        pools = project_to_base(build_pools(tf, best), tf["base"].index)
        results = test_pools(tf["base"], pools, best)
        for p in pools:
            p.asset = symbol

        report.assets[symbol] = AssetData(
            symbol=symbol, base_df=base, tf_data=tf,
            walkforward=wf, final_cfg=best,
            final_pools=pools, final_results=results,
        )

    if not report.assets:
        msg = ("no assets ran successfully — all symbols either failed to fetch or fell back "
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
            ml = SectorMoERespectModel().fit(
                X_train, y_train, train_pool_keep,
                X_oos=X_oos_all, oos_pools=all_oos_pools, oos_results=all_oos_results,
                val_frac=0.25, seed=cfg.opt_seed + 100,
                min_sector_train_n=60, min_sector_class_n=8, min_sector_oos_n=20,
                expert_weight=0.70,
            )
            report.unified_ml = ml
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
            PROX_HORIZONS = [78, 156, 312]
            DIR_HORIZON = 78
            MAX_HORIZON = max(PROX_HORIZONS + [DIR_HORIZON])
            sample_every = max(20, cfg.test_horizon_bars // 6)

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
                for fr in wf.folds:
                    tr = generate_snapshots(ad.base_df, a_pools, a_results, state_feat,
                                             window_start=fr.train_start,
                                             window_end=fr.train_end,
                                             sample_every=sample_every,
                                             max_horizon=MAX_HORIZON)
                    os_ = generate_snapshots(ad.base_df, a_pools, a_results, state_feat,
                                              window_start=fr.test_start,
                                              window_end=fr.test_end,
                                              sample_every=sample_every,
                                              max_horizon=MAX_HORIZON)
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
    print(f"  Pooled OOS broad:       {report.pooled_oos_respect:.1%}   "
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
        print(f"  Val Brier / log-loss:   {m.val_brier:.4f} / {m.val_logloss:.4f}", file=file)
        print(f"  Val AUC-ROC:            {m.val_auc:.3f}", file=file)
        print(f"  Base rate:              {m.base_rate:.1%}", file=file)
        if isinstance(m, SectorMoERespectModel):
            print(f"  Blend:                  {m.expert_weight:.0%} sector expert / "
                  f"{m.global_weight:.0%} global fallback", file=file)
            if m.sector_stats:
                print(f"\n[sector experts]", file=file)
                print(f"  {'sector':<10} {'status':<8} {'train':>6} {'pos':>5} {'neg':>5} "
                      f"{'oos_dec':>7} {'auc':>6} {'reason':<18}", file=file)
                for sector, st in m.expert_summary():
                    auc = st.get("val_auc")
                    auc_s = f"{auc:.3f}" if isinstance(auc, float) else "-"
                    print(f"  {sector:<10} {st.get('status', '-'):<8} "
                          f"{st.get('train_n', 0):>6} {st.get('pos_n', 0):>5} "
                          f"{st.get('neg_n', 0):>5} {st.get('oos_decisive_n', 0):>7} "
                          f"{auc_s:>6} {st.get('reason', ''):<18}", file=file)
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

    # -- Unified Stratified bucket model --
    if report.unified_stratified is not None:
        from .stratified import print_model as _print_strat
        _print_strat(report.unified_stratified, file=file)

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
