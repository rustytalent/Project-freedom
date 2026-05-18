"""Walk-forward cross-validation for the liquidity-pool optimizer.

The single-window optimizer (fit-and-score on the same 60d window) tends to overfit. Walk-forward
splits history into chronological folds and reports the OUT-OF-SAMPLE respect rate with a
confidence interval — the honest answer to "does this config actually generalise?".

Design (no forward-bias leak):
  - Expanding training windows. Each fold's train ends at the start of its test slice; the next
    fold extends train forward by one test-slice width.
  - When scoring training pools we only count those whose forward horizon stays inside train
    (`available_at + horizon < train_end`). No future-from-test ever influences a training score.
  - When scoring OOS pools we require the same (`available_at >= test_start` and
    `available_at + horizon < test_end`). No bleed from one fold's test into another's train.

We aggregate raw OOS outcomes across all folds, then compute:
  - Pooled OOS respect rate (= total_respected / total_tested).
  - 90% CI via Wilson score interval (exact-ish binomial).
  - 90% CI via bootstrap (cross-check, distribution-free).
  - Mean overfit gap (train_respect - oos_respect per fold).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Callable, Dict, Optional
import copy
import math
import pandas as pd
import numpy as np

from .config import Config
from .pools import build_pools, project_to_base, Pool
from .tester import test_pools, summarise, PoolResult
from .optimizer import optimize
from .stats import wilson_score_interval, bootstrap_proportion_ci
from .stratified import StratifiedRespectModel
from .featurize import Featurizer
from .ml_model import PoolRespectModel, trainable_mask, labels as ml_labels
from .directional import evaluate as directional_evaluate, DirectionalReport
from .timing import (StateFeaturizer, generate_snapshots, DirectionModel, ProximityModel,
                     evaluate_timing, TimingReport)


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_n_pools: int
    train_n_tested: int
    train_respect: float
    test_n_pools: int
    test_n_tested: int
    test_respect: float
    test_respect_ci_low: float
    test_respect_ci_high: float
    test_break_rate: float
    test_untouched_rate: float
    overfit_gap: float
    best_weights: Dict[str, float]


@dataclass
class WalkForwardReport:
    folds: List[FoldResult] = field(default_factory=list)
    n_total_oos_pools: int = 0
    n_total_oos_tested: int = 0
    oos_respect_pooled: float = 0.0
    oos_respect_strict: float = 0.0
    oos_respect_median_fold: float = 0.0
    oos_respect_ci_wilson: Tuple[float, float] = (0.0, 0.0)
    oos_respect_ci_bootstrap: Tuple[float, float] = (0.0, 0.0)
    mean_overfit_gap: float = 0.0
    raw_oos_outcomes: List[int] = field(default_factory=list)
    oos_outcome_counts: Dict[str, int] = field(default_factory=dict)
    # OOS (pool, result) pairs accumulated across folds — used to fit the stratified model.
    # Pools come from each fold's best config (which was trained WITHOUT seeing that pool's
    # outcome), so this is a clean OOS calibration set.
    oos_pools: List[Pool] = field(default_factory=list)
    oos_results: List[PoolResult] = field(default_factory=list)
    stratified_model: Optional[StratifiedRespectModel] = None
    # Train-window pool/result pairs across folds; used to TRAIN the ML model (each pool's
    # outcome was determined by data inside its fold's train window only).
    train_pools_for_ml: List[Pool] = field(default_factory=list)
    train_results_for_ml: List[PoolResult] = field(default_factory=list)
    # ML model fit on train pools across folds, calibrated on OOS pools, then evaluated.
    ml_model: Optional[PoolRespectModel] = None
    ml_oos_predictions: Optional[List[float]] = None   # P(respect) for each oos_pools entry
    directional_report: Optional[DirectionalReport] = None
    # Track 3: direction + proximity models and their OOS evaluation.
    direction_model: Optional[DirectionModel] = None
    proximity_models: Dict[int, ProximityModel] = field(default_factory=dict)
    timing_report: Optional[TimingReport] = None


def _fold_windows(base_index: pd.DatetimeIndex, n_folds: int, train_frac: float,
                  min_train: pd.Timedelta
                  ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Expanding-window folds. Tests are consecutive non-overlapping slices at the end of history;
    each train is everything before its test. Returned in chronological order."""
    if len(base_index) == 0:
        return []
    total_span = base_index[-1] - base_index[0]
    first_test_start = base_index[0] + total_span * train_frac
    remaining = base_index[-1] - first_test_start
    if remaining <= pd.Timedelta(0) or n_folds < 1:
        return []
    test_span = remaining / n_folds

    windows = []
    for fold in range(n_folds):
        test_start = first_test_start + test_span * fold
        test_end = first_test_start + test_span * (fold + 1)
        train_start = base_index[0]
        train_end = test_start
        if (train_end - train_start) < min_train:
            continue
        windows.append((train_start, train_end, test_start, test_end))
    return windows


def _safe_end_ts(base_index: pd.DatetimeIndex, w_end_ts: pd.Timestamp,
                 horizon_bars: int) -> Optional[pd.Timestamp]:
    """Return the latest `available_at` value such that walking `horizon_bars` forward in BAR
    INDEX terms stays at or before `w_end_ts`. This is intentionally bar-aware: 150 5m bars on
    NSE intraday span ~2 calendar days because of overnight gaps, so a calendar-time cutoff like
    `w_end - bars × period` would leak overnight."""
    w_end_pos = int(np.searchsorted(base_index.values, np.datetime64(w_end_ts), side="right"))
    safe_pos = w_end_pos - horizon_bars - 1
    if safe_pos < 0:
        return None
    return base_index[safe_pos]


def _in_window(pool: Pool, w_start: pd.Timestamp,
               w_safe_end: Optional[pd.Timestamp]) -> bool:
    """Bar-aware window check: pool's available_at must lie in [w_start, w_safe_end].
    `w_safe_end` is precomputed via _safe_end_ts and accounts for non-uniform bar spacing."""
    if w_safe_end is None:
        return False
    return w_start <= pool.available_at <= w_safe_end


def walk_forward(tf_data: Dict[str, pd.DataFrame], cfg: Config,
                 n_folds: int = 5, train_frac: float = 0.6,
                 iters_per_fold: Optional[int] = None,
                 min_train_days: int = 10,
                 progress: Optional[Callable[[int, int, str], None]] = None,
                 ) -> WalkForwardReport:
    base = tf_data["base"]
    if len(base) < 200:
        raise ValueError(f"need >= 200 base bars for walk-forward, have {len(base)}")

    min_train = pd.Timedelta(days=min_train_days)

    windows = _fold_windows(base.index, n_folds, train_frac, min_train)
    if not windows:
        raise ValueError("no usable folds; try fewer folds, lower train_frac, or more data")

    iters = iters_per_fold or max(40, cfg.opt_iterations // max(len(windows), 1))
    report = WalkForwardReport()

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        if progress:
            progress(fold_idx, len(windows),
                     f"train {tr_s.date()} → {tr_e.date()}  |  test {te_s.date()} → {te_e.date()}")

        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.opt_iterations = iters
        fold_cfg.opt_seed = cfg.opt_seed + fold_idx

        # Precompute bar-aware end timestamps for this fold's train and test windows.
        tr_safe_end = _safe_end_ts(base.index, tr_e, cfg.test_horizon_bars)
        te_safe_end = _safe_end_ts(base.index, te_e, cfg.test_horizon_bars)

        # Training-only evaluator: score pools fully testable within train (no future-leak)
        def train_eval(pools, results, _tr_s=tr_s, _tr_safe=tr_safe_end):
            kept = [r for p, r in zip(pools, results) if _in_window(p, _tr_s, _tr_safe)]
            return summarise(kept)

        best_cfg, _ = optimize(tf_data, fold_cfg, evaluator=train_eval)

        # Evaluate best on OOS test slice with the same bar-aware no-leak rule.
        pools = project_to_base(build_pools(tf_data, best_cfg), tf_data["base"].index)
        results = test_pools(tf_data["base"], pools, best_cfg)
        train_kept = [r for p, r in zip(pools, results) if _in_window(p, tr_s, tr_safe_end)]
        oos_kept = [r for p, r in zip(pools, results) if _in_window(p, te_s, te_safe_end)]

        ts = summarise(train_kept)
        os_ = summarise(oos_kept)

        # Save OOS (pool, result) pairs for the stratified-probability model fit later.
        oos_pairs = [(p, r) for p, r in zip(pools, results)
                     if _in_window(p, te_s, te_safe_end)]
        report.oos_pools.extend(p for p, _ in oos_pairs)
        report.oos_results.extend(r for _, r in oos_pairs)

        # Also collect TRAIN-window pairs to use as ML training data later. These are pools whose
        # outcome was determined by data inside this fold's train window only (no leak into test).
        train_pairs = [(p, r) for p, r in zip(pools, results)
                        if _in_window(p, tr_s, tr_safe_end)]
        report.train_pools_for_ml.extend(p for p, _ in train_pairs)
        report.train_results_for_ml.extend(r for _, r in train_pairs)

        oos_outs = [1 if r.is_respect else 0 for r in oos_kept if r.is_tested]
        ci_lo, ci_hi = wilson_score_interval(sum(oos_outs), len(oos_outs), ci=0.90)
        report.raw_oos_outcomes.extend(oos_outs)
        for r in oos_kept:
            report.oos_outcome_counts[r.outcome] = report.oos_outcome_counts.get(r.outcome, 0) + 1

        report.folds.append(FoldResult(
            fold=fold_idx,
            train_start=tr_s, train_end=tr_e, test_start=te_s, test_end=te_e,
            train_n_pools=ts["n"], train_n_tested=ts["tested_n"],
            train_respect=ts["respect_rate"],
            test_n_pools=os_["n"], test_n_tested=os_["tested_n"],
            test_respect=os_["respect_rate"],
            test_respect_ci_low=ci_lo, test_respect_ci_high=ci_hi,
            test_break_rate=os_["break_rate"], test_untouched_rate=os_["untouched_rate"],
            overfit_gap=ts["respect_rate"] - os_["respect_rate"],
            best_weights=best_cfg.weights.as_dict(),
        ))

    if report.folds:
        outs = report.raw_oos_outcomes
        n_tested = len(outs)
        n_resp = sum(outs)
        report.n_total_oos_tested = n_tested
        report.n_total_oos_pools = sum(fr.test_n_pools for fr in report.folds)
        report.oos_respect_pooled = (n_resp / n_tested) if n_tested else 0.0
        # Strict rate = decisive respects (strong + swept_and_reclaimed) over all decisive outcomes
        s_resp = (report.oos_outcome_counts.get("respected_strong", 0)
                  + report.oos_outcome_counts.get("swept_and_reclaimed", 0))
        s_break = report.oos_outcome_counts.get("broken_strong", 0)
        report.oos_respect_strict = (s_resp / (s_resp + s_break)) if (s_resp + s_break) else 0.0
        fold_rates = [fr.test_respect for fr in report.folds if fr.test_n_tested > 0]
        report.oos_respect_median_fold = float(np.median(fold_rates)) if fold_rates else 0.0
        report.oos_respect_ci_wilson = wilson_score_interval(n_resp, n_tested, ci=0.90)
        report.oos_respect_ci_bootstrap = bootstrap_proportion_ci(outs, ci=0.90, n_boot=2000)
        report.mean_overfit_gap = float(np.mean([fr.overfit_gap for fr in report.folds]))

    # Fit the stratified per-pool probability model on the union of OOS pool sets.
    if report.oos_pools:
        report.stratified_model = StratifiedRespectModel.fit(
            report.oos_pools, report.oos_results, min_sample=8,
        )

    # Track 2: ML model. Train on per-fold train-window outcomes, calibrate on OOS, evaluate
    # directional metrics. Each step is a try because lightgbm/sklearn might not be installed
    # in slimmer environments.
    if report.train_pools_for_ml and report.oos_pools:
        try:
            base = tf_data["base"]
            feat = Featurizer(base)
            train_mask = trainable_mask(report.train_results_for_ml)
            if train_mask.sum() >= 20:
                X_train = feat.transform_batch(
                    [p for p, keep in zip(report.train_pools_for_ml, train_mask) if keep]
                )
                y_train = ml_labels(
                    [r for r, keep in zip(report.train_results_for_ml, train_mask) if keep]
                )
                model = PoolRespectModel().fit(X_train, y_train,
                                                val_frac=0.25, seed=cfg.opt_seed)
                # Predict on OOS pool set (every OOS pool, decisive or not — for ranking/touch
                # evaluation; only decisive ones are used for calibration metrics).
                X_oos = feat.transform_batch(report.oos_pools)
                # First a pass without bucket calibration so we can fit bucket calibration
                # on (predicted, actual) for decisive OOS pools only.
                p_raw = model.predict_raw(X_oos)
                oos_decisive_mask = trainable_mask(report.oos_results)
                if oos_decisive_mask.sum() >= 20:
                    y_oos_dec = ml_labels(
                        [r for r, m in zip(report.oos_results, oos_decisive_mask) if m]
                    )
                    pools_dec = [p for p, m in zip(report.oos_pools, oos_decisive_mask) if m]
                    p_raw_dec = p_raw[oos_decisive_mask]
                    model.fit_bucket_calib(pools_dec, y_oos_dec, p_raw_dec, min_bucket_n=10)

                # Final calibrated predictions on the FULL OOS set (with bucket shrinkage)
                p_final = model.predict(X_oos, pools=report.oos_pools)
                report.ml_model = model
                report.ml_oos_predictions = [float(v) for v in p_final]

                # Directional evaluation: needs the full pool/result/pred arrays and the base df.
                # `start_after` = first OOS test window start (so the time-loop doesn't sample
                # times where most OOS pools haven't become available yet).
                first_oos_start = min((fr.test_start for fr in report.folds), default=None)
                report.directional_report = directional_evaluate(
                    report.oos_pools, report.oos_results, p_final, base,
                    horizon_bars=max(50, cfg.test_horizon_bars // 3),
                    sample_every=max(20, cfg.test_horizon_bars // 5),
                    start_after=first_oos_start,
                )
        except Exception as e:
            print(f"[walkforward] ML training skipped: {e}")

    # Track 3: direction + proximity models on time-series snapshots.
    # Generate snapshots ONCE at the max horizon; derive per-horizon labels on demand. Train one
    # DirectionModel (primary horizon 78 bars = 1 NSE session) and THREE ProximityModels
    # (78, 156, 312 bars = today, 2 days, 4 days).
    if report.ml_model is not None and len(report.oos_pools) > 0:
        try:
            base = tf_data["base"]
            unified_pools = list(report.train_pools_for_ml) + list(report.oos_pools)
            unified_results = list(report.train_results_for_ml) + list(report.oos_results)
            feat_for_quality = Featurizer(base)
            X_uni = feat_for_quality.transform_batch(unified_pools)
            quality_preds = report.ml_model.predict(X_uni, pools=unified_pools)

            state_feat = StateFeaturizer(base)
            PROX_HORIZONS = [78, 156, 312]            # 1 day, 2 days, 4 days on NSE 5m
            DIR_HORIZON = 78                            # match "today" planning
            MAX_HORIZON = max(PROX_HORIZONS + [DIR_HORIZON])
            sample_every = max(20, cfg.test_horizon_bars // 6)
            # Base period in seconds inferred from the actual index — used to normalise pool age.
            _diffs = pd.Series(base.index).diff().dropna()
            base_period_seconds = float(_diffs.median().total_seconds()) if len(_diffs) else 300.0
            if base_period_seconds <= 0:
                base_period_seconds = 300.0

            train_snaps: List = []
            oos_snaps: List = []
            for fr in report.folds:
                tr = generate_snapshots(base, unified_pools, unified_results, state_feat,
                                         window_start=fr.train_start, window_end=fr.train_end,
                                         sample_every=sample_every, max_horizon=MAX_HORIZON)
                os_ = generate_snapshots(base, unified_pools, unified_results, state_feat,
                                          window_start=fr.test_start, window_end=fr.test_end,
                                          sample_every=sample_every, max_horizon=MAX_HORIZON)
                train_snaps.extend(tr)
                oos_snaps.extend(os_)

            if len(train_snaps) >= 30:
                try:
                    dir_m = DirectionModel(horizon=DIR_HORIZON).fit(
                        train_snaps, seed=cfg.opt_seed + 100,
                    )
                    report.direction_model = dir_m
                except ValueError as ve:
                    print(f"[walkforward] direction model skipped: {ve}")

                for h in PROX_HORIZONS:
                    try:
                        pm = ProximityModel(horizon=h,
                                              base_period_seconds=base_period_seconds).fit(
                            train_snaps, unified_pools, quality_preds,
                            seed=cfg.opt_seed + 200 + h,
                        )
                        report.proximity_models[h] = pm
                    except ValueError as ve:
                        print(f"[walkforward] proximity h={h} skipped: {ve}")

            if (report.direction_model is not None and report.proximity_models
                    and oos_snaps):
                report.timing_report = evaluate_timing(
                    oos_snaps, unified_pools, unified_results,
                    report.direction_model, report.proximity_models, quality_preds,
                )
        except Exception as e:
            print(f"[walkforward] Track 3 (direction/timing) skipped: {e}")

    return report


def print_report(report: WalkForwardReport, file=None) -> None:
    print("\n=== Walk-Forward Validation Report ===", file=file)
    if not report.folds:
        print("(no folds run)", file=file)
        return

    print(f"\n[folds]  expanding train window, consecutive OOS test slices "
          f"({len(report.folds)} folds):", file=file)
    print(f"  {'#':<3} {'train period':<25} {'test period':<25} "
          f"{'tr_n':>5} {'tr_resp':>8} {'oos_n':>5} {'oos_resp':>8} "
          f"{'90% CI':>15} {'gap':>7}", file=file)
    for fr in report.folds:
        ci = f"{fr.test_respect_ci_low*100:.0f}-{fr.test_respect_ci_high*100:.0f}%"
        train_lbl = f"{fr.train_start.date()} → {fr.train_end.date()}"
        test_lbl = f"{fr.test_start.date()} → {fr.test_end.date()}"
        print(f"  {fr.fold+1:<3} {train_lbl:<25} {test_lbl:<25} "
              f"{fr.train_n_tested:>5} {fr.train_respect:>7.1%} "
              f"{fr.test_n_tested:>5} {fr.test_respect:>7.1%} "
              f"{ci:>15} {fr.overfit_gap:>+6.1%}",
              file=file)

    print(f"\n[aggregated out-of-sample]", file=file)
    print(f"  OOS pools (eligible):    {report.n_total_oos_pools}", file=file)
    print(f"  OOS pools (tested):      {report.n_total_oos_tested}", file=file)
    print(f"  Respect (broad):         {report.oos_respect_pooled:.1%}  "
          f"  Wilson 90% CI: {report.oos_respect_ci_wilson[0]:.1%} – "
          f"{report.oos_respect_ci_wilson[1]:.1%}", file=file)
    print(f"  Respect (broad):         {report.oos_respect_pooled:.1%}  "
          f"  Boot   90% CI: {report.oos_respect_ci_bootstrap[0]:.1%} – "
          f"{report.oos_respect_ci_bootstrap[1]:.1%}", file=file)
    print(f"  Respect (strict):        {report.oos_respect_strict:.1%}  "
          f"(decisive outcomes only: respected_strong vs broken_strong)", file=file)
    print(f"  Median per-fold OOS:     {report.oos_respect_median_fold:.1%}", file=file)
    print(f"  Mean overfit gap:        {report.mean_overfit_gap:+.1%}  "
          f"(train_respect minus OOS — how much in-sample numbers are inflated)", file=file)

    # Outcome texture
    oc = report.oos_outcome_counts
    if oc:
        print(f"\n[OOS outcome breakdown]", file=file)
        for k in ("respected_strong", "swept_and_reclaimed", "respected_weak",
                  "broken_weak", "broken_strong",
                  "touched_no_signal", "untouched"):
            v = oc.get(k, 0)
            pct = (v / sum(oc.values()) * 100) if oc else 0.0
            print(f"  {k:<22} {v:>5}  ({pct:>4.1f}%)", file=file)

    # ML model fit summary + directional report
    if report.ml_model is not None:
        m = report.ml_model
        print(f"\n[ML model fit]  (LightGBM + isotonic + bucket shrinkage)", file=file)
        print(f"  train pools:            {m.train_n}", file=file)
        print(f"  validation pools:       {m.val_n}", file=file)
        print(f"  base rate:              {m.base_rate:.1%}", file=file)
        print(f"  val Brier / log-loss:   {m.val_brier:.4f}  /  {m.val_logloss:.4f}", file=file)
        print(f"  val AUC-ROC:            {m.val_auc:.3f}", file=file)
        if m.bucket_calib:
            print(f"\n[ML bucket-level OOS shrinkage]   "
                  f"(applied on top of global isotonic)", file=file)
            print(f"  {'TFs':<5} {'factor':<8} {'n_oos':>6} {'emp_rate':>9} {'pull':>6}",
                  file=file)
            for (tfb, fam), bc in sorted(m.bucket_calib.items(),
                                          key=lambda kv: -kv[1].pull_weight):
                print(f"  {tfb:<5} {fam:<8} {bc.n_oos:>6} {bc.empirical_rate:>8.1%} "
                      f"{bc.pull_weight:>5.2f}", file=file)
        if m.feature_names:
            print(f"\n[ML feature importance — top 10 by gain]", file=file)
            for name, gain in m.feature_importance(10):
                print(f"  {name:<32} {gain:>10.1f}", file=file)

    if report.directional_report is not None:
        from .directional import print_report as _print_dir
        _print_dir(report.directional_report, file=file)

    # Track 3 timing report + feature importances
    if report.timing_report is not None:
        from .timing import print_timing_report as _print_timing
        _print_timing(report.timing_report, file=file)
        if report.direction_model is not None and report.direction_model.feature_names:
            print(f"\n[direction model — top 8 features by gain]", file=file)
            for name, gain in report.direction_model.feature_importance(8):
                print(f"  {name:<32} {gain:>10.1f}", file=file)
        if report.proximity_models:
            shortest = min(report.proximity_models.keys())
            pm = report.proximity_models[shortest]
            if pm.feature_names:
                print(f"\n[proximity model (h={shortest}) — top 8 features by gain]", file=file)
                for name, gain in pm.feature_importance(8):
                    print(f"  {name:<32} {gain:>10.1f}", file=file)

    gap = report.mean_overfit_gap
    if gap > 0.15:
        verdict = "⚠ LARGE overfit gap. Don't trust in-sample numbers; the OOS estimate is the real one."
    elif gap > 0.07:
        verdict = "Moderate overfit gap. OOS estimate is the honest performance."
    elif gap > 0.0:
        verdict = "Small overfit gap — optimizer mostly generalises."
    else:
        verdict = "No overfit (or OOS happens to be easier). Be skeptical of any single fold."
    print(f"\n  → {verdict}", file=file)
