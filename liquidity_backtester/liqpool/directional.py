"""Directional / ranking quality evaluation.

Three flavours of question this module answers:

  1. Is the model well-calibrated?
        - calibration_table:  10-bin reliability table (predicted bin vs actual rate).
        - brier_score:        mean squared error of prediction vs label.
  2. Is the ranking useful?
        - auc_roc:            does sorting pools by predicted prob separate respects from breaks?
        - decile_lift:        top-decile respect rate vs bottom-decile.
  3. Does it predict where price actually goes (directional bias)?
        - touch_rank_correlation: among pools available at time T, is the highest-ranked one
          touched first?
        - directional_accuracy:   at time T, side with higher "expected pull" should match the
          direction of the next K-bar excursion.
        - hit_first_rate:         was the actually-first-touched pool among the top-N by predicted
          probability?

All time-based evaluators sample the base TF every `sample_every` bars and only use pools whose
`available_at` is <= the sample time AND that hadn't already been touched/broken by then. This
keeps every measurement honest in real time.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Sequence, Optional
import numpy as np
import pandas as pd

from .pools import Pool
from .tester import PoolResult
from .stratified import _tf_bucket, _headline_factor


@dataclass
class CalibrationBin:
    bin_lo: float
    bin_hi: float
    n: int
    mean_pred: float
    actual_rate: float


@dataclass
class DirectionalReport:
    n_pools_scored: int = 0
    brier_score: float = 0.0
    log_loss: float = 0.0
    auc_roc: float = 0.0
    decile_top_rate: float = 0.0
    decile_bottom_rate: float = 0.0
    decile_lift: float = 0.0
    calibration: List[CalibrationBin] = field(default_factory=list)
    bucket_calibration: List[dict] = field(default_factory=list)

    # Directional bias metrics
    n_sample_times: int = 0
    touch_rank_correlation: float = 0.0   # Spearman: predicted rank vs touch order
    directional_accuracy: float = 0.0
    hit_first_top1: float = 0.0
    hit_first_top3: float = 0.0


# ---------------------------------------------------------------------------
# Pool-level: calibration, ranking
# ---------------------------------------------------------------------------

def calibration_table(preds: np.ndarray, labels: np.ndarray, n_bins: int = 10
                       ) -> List[CalibrationBin]:
    """Reliability diagram bins. Equal-width [0, 1] bins."""
    out = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (preds >= lo) & (preds < hi) if i < n_bins - 1 else (preds >= lo) & (preds <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(CalibrationBin(
            bin_lo=float(lo), bin_hi=float(hi),
            n=n, mean_pred=float(preds[mask].mean()),
            actual_rate=float(labels[mask].mean()),
        ))
    return out


def decile_lift(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    if len(preds) < 20:
        return (float(labels.mean()) if len(labels) else 0.0, 0.0, 0.0)
    order = np.argsort(preds)
    n = len(preds)
    bottom = order[: n // 10]
    top = order[-n // 10:]
    bot_rate = float(labels[bottom].mean())
    top_rate = float(labels[top].mean())
    lift = (top_rate / bot_rate) if bot_rate > 0 else float("inf")
    return top_rate, bot_rate, lift


def per_bucket_calibration(pools: List[Pool], preds: np.ndarray, labels: np.ndarray,
                            min_n: int = 8) -> List[dict]:
    from collections import defaultdict
    groups: Dict[Tuple[str, str], List[Tuple[float, int]]] = defaultdict(list)
    for p, pred, y in zip(pools, preds, labels):
        key = (_tf_bucket(len(set(p.tfs))), _headline_factor(p))
        groups[key].append((float(pred), int(y)))
    out = []
    for (tfb, fam), items in groups.items():
        if len(items) < min_n:
            continue
        arr_p = np.array([p for p, _ in items])
        arr_y = np.array([y for _, y in items])
        out.append({
            "tf_bucket": tfb, "factor": fam, "n": int(len(items)),
            "mean_pred": float(arr_p.mean()),
            "actual_rate": float(arr_y.mean()),
            "bias": float(arr_p.mean() - arr_y.mean()),  # positive = over-confident
        })
    out.sort(key=lambda d: -abs(d["bias"]))
    return out


# ---------------------------------------------------------------------------
# Time-based: directional accuracy and touch-order ranking
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    # If either is constant, correlation is undefined; return 0.
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate_directional(pools: List[Pool], results: List[PoolResult], preds: np.ndarray,
                         df_base: pd.DataFrame,
                         horizon_bars: int = 50, sample_every: int = 50,
                         start_after: Optional[pd.Timestamp] = None
                         ) -> Tuple[float, float, float, float, int]:
    """At each sample bar T (with T > start_after if provided):
       Active pools = (available_at <= T) AND (touched_at > T OR untouched, OR touched_at None)
       AND (broken_at > T OR not broken).
       Above-current = pools whose price_low > close[T].
       Below-current = pools whose price_high < close[T].

       Per side: rank active pools by predicted prob; observe what got touched in (T, T+horizon].

       1) directional_accuracy: side with higher "expected pull" score should match the side of
          the larger excursion in (T, T+horizon].
            expected_pull(side) = sum_p prob(p) / max(distance_atr(p), 1)
       2) hit_first_top1 / hit_first_top3: the actually-first-touched pool in (T, T+horizon]
          (across either side) was the top-1 / top-3 by predicted prob among active pools.
       3) touch_rank_correlation: Spearman of (predicted_prob rank, touch-time rank) across
          pools touched in the window.
    """
    if len(df_base) == 0 or len(pools) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    idx = df_base.index
    h = df_base["high"].values
    l = df_base["low"].values
    c = df_base["close"].values

    # Per-pool absolute touch / break times in absolute terms.
    touch_at = [r.touched_at for r in results]
    broken_at = [r.broken_at for r in results]

    # ATR median for distance normalisation
    atr_proxy = float((df_base["high"] - df_base["low"]).rolling(14).mean().median())
    if atr_proxy <= 0:
        atr_proxy = float((df_base["high"] - df_base["low"]).median()) or 1.0

    sample_starts = list(range(0, len(idx) - horizon_bars, max(1, sample_every)))
    if start_after is not None:
        sample_starts = [j for j in sample_starts if idx[j] >= start_after]

    if not sample_starts:
        return 0.0, 0.0, 0.0, 0.0, 0

    dir_correct = 0
    dir_total = 0
    hit1 = 0; hit3 = 0; hit_total = 0
    rank_corrs: List[float] = []

    for j in sample_starts:
        T = idx[j]
        close_T = c[j]
        h_window = h[j + 1 : j + 1 + horizon_bars]
        l_window = l[j + 1 : j + 1 + horizon_bars]
        if len(h_window) == 0:
            continue
        max_up = float(h_window.max()) - close_T
        max_dn = close_T - float(l_window.min())

        active_ids: List[int] = []
        for i, p in enumerate(pools):
            if p.available_at > T:
                continue
            ta = touch_at[i]
            if ta is not None and ta <= T:
                continue  # already touched before sample time
            ba = broken_at[i]
            if ba is not None and ba <= T:
                continue
            active_ids.append(i)

        if len(active_ids) < 2:
            continue

        # Directional pull
        up_score = 0.0
        dn_score = 0.0
        above_ids: List[int] = []
        below_ids: List[int] = []
        for i in active_ids:
            p = pools[i]
            if p.price_low > close_T:
                dist = max((p.mid - close_T) / atr_proxy, 1.0)
                up_score += float(preds[i]) / dist
                above_ids.append(i)
            elif p.price_high < close_T:
                dist = max((close_T - p.mid) / atr_proxy, 1.0)
                dn_score += float(preds[i]) / dist
                below_ids.append(i)
        if up_score > 0 or dn_score > 0:
            predicted_dir = "up" if up_score >= dn_score else "down"
            actual_dir = "up" if max_up >= max_dn else "down"
            dir_correct += int(predicted_dir == actual_dir)
            dir_total += 1

        # Hit-first rate
        touched_in_window: List[Tuple[int, pd.Timestamp]] = []
        for i in active_ids:
            ta = touch_at[i]
            if ta is not None and T < ta <= idx[min(j + horizon_bars, len(idx) - 1)]:
                touched_in_window.append((i, ta))
        if touched_in_window:
            touched_in_window.sort(key=lambda x: x[1])
            first_idx = touched_in_window[0][0]
            ranked = sorted(active_ids, key=lambda i: -float(preds[i]))
            top1 = set(ranked[:1])
            top3 = set(ranked[:3])
            hit1 += int(first_idx in top1)
            hit3 += int(first_idx in top3)
            hit_total += 1

            # Touch-rank correlation: pred vs touch-time order, on touched-in-window pools.
            tw_ids = [tid for tid, _ in touched_in_window]
            tw_ranks_pred = np.array([float(preds[i]) for i in tw_ids])
            tw_ranks_time = np.array([(tt - T).total_seconds() for _, tt in touched_in_window])
            # Higher predicted prob should correspond to earlier touch → negative Spearman.
            corr = _spearman(tw_ranks_pred, -tw_ranks_time)
            rank_corrs.append(corr)

    dir_acc = (dir_correct / dir_total) if dir_total else 0.0
    hit1_rate = (hit1 / hit_total) if hit_total else 0.0
    hit3_rate = (hit3 / hit_total) if hit_total else 0.0
    rank_c = float(np.mean(rank_corrs)) if rank_corrs else 0.0
    return dir_acc, hit1_rate, hit3_rate, rank_c, dir_total


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------

def evaluate(pools: List[Pool], results: List[PoolResult], preds: np.ndarray,
             df_base: pd.DataFrame, horizon_bars: int = 50, sample_every: int = 50,
             start_after: Optional[pd.Timestamp] = None) -> DirectionalReport:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    rpt = DirectionalReport()
    # Only score pools with a decisive outcome (positive or negative); ignore ambiguous.
    from .ml_model import trainable_mask, labels as ml_labels
    mask = trainable_mask(results)
    if mask.sum() < 5:
        return rpt
    p_eval = preds[mask]
    y_eval = ml_labels([r for r, m in zip(results, mask) if m])
    pools_eval = [p for p, m in zip(pools, mask) if m]

    rpt.n_pools_scored = int(mask.sum())
    rpt.brier_score = float(brier_score_loss(y_eval, p_eval))
    rpt.log_loss = float(log_loss(y_eval, np.clip(p_eval, 1e-6, 1 - 1e-6)))
    if len(set(y_eval)) > 1:
        rpt.auc_roc = float(roc_auc_score(y_eval, p_eval))
    top_rate, bot_rate, lift = decile_lift(p_eval, y_eval)
    rpt.decile_top_rate = top_rate
    rpt.decile_bottom_rate = bot_rate
    rpt.decile_lift = lift
    rpt.calibration = calibration_table(p_eval, y_eval, n_bins=10)
    rpt.bucket_calibration = per_bucket_calibration(pools_eval, p_eval, y_eval, min_n=8)

    dir_acc, hit1, hit3, rank_c, n_samples = evaluate_directional(
        pools, results, preds, df_base,
        horizon_bars=horizon_bars, sample_every=sample_every, start_after=start_after,
    )
    rpt.directional_accuracy = dir_acc
    rpt.hit_first_top1 = hit1
    rpt.hit_first_top3 = hit3
    rpt.touch_rank_correlation = rank_c
    rpt.n_sample_times = n_samples
    return rpt


def print_report(rpt: DirectionalReport, file=None) -> None:
    print("\n=== ML Model OOS Evaluation ===", file=file)
    print(f"  pools scored:           {rpt.n_pools_scored}", file=file)
    print(f"  Brier score:            {rpt.brier_score:.4f}  (lower=better; 0 perfect, 0.25 random)",
          file=file)
    print(f"  Log loss:               {rpt.log_loss:.4f}  (lower=better)", file=file)
    print(f"  AUC-ROC:                {rpt.auc_roc:.3f}  (0.5=random, 1.0=perfect)", file=file)
    print(f"  Top decile respect:     {rpt.decile_top_rate:.1%}", file=file)
    print(f"  Bottom decile respect:  {rpt.decile_bottom_rate:.1%}", file=file)
    print(f"  Decile lift:            {rpt.decile_lift:.2f}x  (top/bottom)", file=file)

    if rpt.calibration:
        print("\n[calibration table]   bin → mean_pred vs actual_rate", file=file)
        print(f"  {'bin':<14} {'n':>5} {'mean_pred':>10} {'actual':>9}", file=file)
        for b in rpt.calibration:
            label = f"[{b.bin_lo:.2f},{b.bin_hi:.2f}]"
            print(f"  {label:<14} {b.n:>5} {b.mean_pred:>9.1%} {b.actual_rate:>8.1%}", file=file)

    if rpt.bucket_calibration:
        print(f"\n[per-bucket calibration]  bias = mean_pred - actual_rate", file=file)
        print(f"  {'TFs':<5} {'factor':<8} {'n':>5} {'mean_pred':>10} {'actual':>9} {'bias':>8}",
              file=file)
        for d in rpt.bucket_calibration:
            print(f"  {d['tf_bucket']:<5} {d['factor']:<8} {d['n']:>5} "
                  f"{d['mean_pred']:>9.1%} {d['actual_rate']:>8.1%} {d['bias']:>+7.1%}", file=file)

    print(f"\n=== Directional Bias Metrics  [LEGACY — superseded by Track 3 below] ===",
          file=file)
    print(f"  sample points:          {rpt.n_sample_times}", file=file)
    print(f"  directional accuracy:   {rpt.directional_accuracy:.1%}  "
          f"(side w/ higher predicted pull matched larger excursion)", file=file)
    print(f"  hit-first @ top-1:      {rpt.hit_first_top1:.1%}  "
          f"(first-touched pool was the #1 ranked one)", file=file)
    print(f"  hit-first @ top-3:      {rpt.hit_first_top3:.1%}", file=file)
    print(f"  touch-order rank corr:  {rpt.touch_rank_correlation:+.3f}  "
          f"(Spearman: higher predicted prob → earlier touch)", file=file)
