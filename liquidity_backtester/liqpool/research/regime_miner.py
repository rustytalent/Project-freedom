"""Regime-conditional + walk-forward hypothesis mining (playbook ideas #2 + #4).

The existing ``hypothesis_miner`` evaluates a candidate rule on the entire
frame in one pass. Two structural blind spots make that scoring optimistic
in ways that systematically convert in-sample "edge" into live losses:

  1. **No walk-forward.** A hypothesis is scored on the same data its
     thresholds were quantilised from. The Sharpe is therefore an
     in-sample fit. Walking forward — train thresholds on the first K
     bars, evaluate the rule's mask on the next K bars, repeat — is the
     discipline that converts "tested" into "tested honestly."

  2. **No regime conditioning.** A hypothesis can have edge only in
     trending regimes (or only in low-vol mornings, etc.) but be
     averaged out by all-day evaluation. Splitting the frame by regime
     and running the miner per-regime surfaces conditional edges
     directly; the playbook explicitly flags this as 🔴💎 — "uniquely
     possible because of OUR codebase" via ``liqpool.regime``.

The trick that makes per-regime mining actually work: feature quantiles
are computed *within the regime*. A condition like ``score > q70`` means
"top 30% of score within this regime", not "top 30% across the whole
year" — otherwise a low-vol regime where every row's score is below the
global median would never satisfy the condition and the miner would
report zero trades.

What this module is NOT:
  * A regime classifier. We consume an already-tagged ``regime_col`` —
    typically the output of ``liqpool.regime.compute_regime_series``
    plus a bucketing pass (e.g. ``vol_ratio > 1.2 → "high_vol"``).
  * A bet sizer. Like the other research modules, it answers
    "which hypothesis to deploy" not "how much capital it should get".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .hypothesis_miner import (
    Condition,
    HypothesisMetrics,
    HypothesisSpec,
    OPS,
    _empty_metrics,
    _feature_thresholds,
    _max_drawdown,
    _usable_features,
    evaluate_hypothesis,
)


# Minimum bars per fold for the walk-forward fitness to be statistically
# meaningful. Below this, the per-fold Sharpe is dominated by noise.
DEFAULT_MIN_BARS_PER_FOLD: int = 200

# Default number of expanding-window folds. Smaller = more out-of-sample
# data per fold (more reliable per-fold Sharpe) but fewer folds (noisier
# consistency score). 5 is the standard quant default.
DEFAULT_N_FOLDS: int = 5

# A regime is too small to mine if it has fewer rows than this. Below
# this, threshold quantiles collapse to a tiny support and the resulting
# hypotheses are overfit.
DEFAULT_MIN_ROWS_PER_REGIME: int = 200


@dataclass(frozen=True)
class FoldResult:
    """One walk-forward fold's outcome for a single hypothesis."""
    fold_index: int
    n_trades: int
    mean_r: float
    sharpe: float


@dataclass(frozen=True)
class WalkForwardMetrics:
    """Walk-forward fitness for a single hypothesis."""
    name: str
    n_folds: int
    folds: Tuple[FoldResult, ...]
    n_trades_total: int
    mean_oos_sharpe: float
    std_oos_sharpe: float
    consistency: float          # fraction of folds with positive Sharpe
    fitness: float              # mean OOS Sharpe × consistency × log1p(n_trades)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_folds": int(self.n_folds),
            "fold_sharpes": [float(f.sharpe) for f in self.folds],
            "fold_n_trades": [int(f.n_trades) for f in self.folds],
            "n_trades_total": int(self.n_trades_total),
            "mean_oos_sharpe": float(self.mean_oos_sharpe),
            "std_oos_sharpe": float(self.std_oos_sharpe),
            "consistency": float(self.consistency),
            "fitness": float(self.fitness),
        }


def _fold_slices(n_rows: int, n_folds: int,
                 min_bars_per_fold: int = DEFAULT_MIN_BARS_PER_FOLD,
                 ) -> List[Tuple[int, int]]:
    """Expanding-window walk-forward slice indices.

    Returns ``[(test_start, test_end), ...]`` for each fold. The train
    window is implicitly ``[0, test_start)``. We drop folds whose test
    window has fewer than ``min_bars_per_fold`` rows so a final stub
    fold can't dominate the consistency score.
    """
    if n_rows < 2 * min_bars_per_fold or n_folds < 1:
        return []
    # Reserve the first fold for training: start the first test fold at
    # ``n_rows // (n_folds + 1)`` so the train window is always at least
    # as big as one test fold.
    step = max(min_bars_per_fold, n_rows // (n_folds + 1))
    slices: List[Tuple[int, int]] = []
    start = step
    while start < n_rows and len(slices) < n_folds:
        end = min(n_rows, start + step)
        if end - start >= min_bars_per_fold:
            slices.append((start, end))
        start = end
    return slices


def walk_forward_metrics(frame: pd.DataFrame,
                         spec: HypothesisSpec,
                         *,
                         forward_return_col: str,
                         cost_r_col: Optional[str] = None,
                         n_folds: int = DEFAULT_N_FOLDS,
                         min_bars_per_fold: int = DEFAULT_MIN_BARS_PER_FOLD,
                         min_trades_per_fold: int = 5,
                         ) -> WalkForwardMetrics:
    """Score a hypothesis with expanding-window walk-forward.

    Each fold evaluates the rule's mask on rows ``[start, end)`` only —
    the spec is fixed (no per-fold re-fit). This is the right discipline
    for *rule* hypotheses: their thresholds were already chosen from the
    frame's overall quantiles, and walk-forward simply asks "does the
    rule keep working as time advances?"

    ``fitness`` combines mean OOS Sharpe (the magnitude), consistency
    (the fraction of folds with positive Sharpe), and a sample-size
    factor (``log1p(n_trades)/log1p(min_target)``) so a hypothesis with
    8 OOS trades doesn't beat one with 80 just by luck."""
    n_rows = int(len(frame))
    slices = _fold_slices(n_rows, n_folds, min_bars_per_fold)
    if not slices:
        return WalkForwardMetrics(
            name=spec.name, n_folds=0, folds=tuple(),
            n_trades_total=0, mean_oos_sharpe=0.0, std_oos_sharpe=0.0,
            consistency=0.0, fitness=0.0,
        )
    folds: List[FoldResult] = []
    n_trades_total = 0
    sharpes: List[float] = []
    for i, (start, end) in enumerate(slices):
        sub = frame.iloc[start:end]
        # Reuse the existing evaluator so the per-fold accounting is
        # identical to the single-shot evaluator — only the slice differs.
        m = evaluate_hypothesis(
            sub, spec,
            forward_return_col=forward_return_col,
            cost_r_col=cost_r_col,
            min_trades=min_trades_per_fold,
        )
        folds.append(FoldResult(
            fold_index=i,
            n_trades=int(m.trades),
            mean_r=float(m.mean_r),
            sharpe=float(m.sharpe),
        ))
        n_trades_total += int(m.trades)
        if m.trades >= min_trades_per_fold:
            sharpes.append(float(m.sharpe))
    if not sharpes:
        return WalkForwardMetrics(
            name=spec.name, n_folds=int(len(slices)), folds=tuple(folds),
            n_trades_total=n_trades_total, mean_oos_sharpe=0.0,
            std_oos_sharpe=0.0, consistency=0.0, fitness=0.0,
        )
    mean_sh = float(np.mean(sharpes))
    std_sh = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    consistency = float(np.mean([s > 0 for s in sharpes]))
    sample_factor = min(1.0, math.log1p(n_trades_total) / math.log1p(30 * len(slices)))
    fitness = mean_sh * consistency * sample_factor
    return WalkForwardMetrics(
        name=spec.name, n_folds=int(len(slices)), folds=tuple(folds),
        n_trades_total=n_trades_total, mean_oos_sharpe=mean_sh,
        std_oos_sharpe=std_sh, consistency=consistency, fitness=fitness,
    )


def rank_walk_forward_hypotheses(frame: pd.DataFrame,
                                  *,
                                  forward_return_col: str,
                                  cost_r_col: Optional[str] = None,
                                  specs: Optional[Sequence[HypothesisSpec]] = None,
                                  n: int = 500,
                                  seed: int = 7,
                                  feature_columns: Optional[Sequence[str]] = None,
                                  n_folds: int = DEFAULT_N_FOLDS,
                                  min_bars_per_fold: int = DEFAULT_MIN_BARS_PER_FOLD,
                                  min_trades_per_fold: int = 5,
                                  ) -> pd.DataFrame:
    """Sample (or accept) a set of specs, walk-forward each, return a
    fitness-sorted DataFrame.

    When ``specs`` is None we sample via the existing
    ``sample_random_hypotheses`` — which uses the frame's global
    quantiles for thresholds. The walk-forward then tells you which of
    those random specs has a *consistent* OOS edge."""
    from .hypothesis_miner import sample_random_hypotheses
    if specs is None:
        specs = sample_random_hypotheses(
            frame, n=n, seed=seed, feature_columns=feature_columns,
        )
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        wf = walk_forward_metrics(
            frame, spec,
            forward_return_col=forward_return_col,
            cost_r_col=cost_r_col,
            n_folds=n_folds,
            min_bars_per_fold=min_bars_per_fold,
            min_trades_per_fold=min_trades_per_fold,
        )
        row = wf.to_dict()
        row["side"] = spec.side
        row["hold_bars"] = int(spec.hold_bars)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["fitness", "mean_oos_sharpe", "n_trades_total"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# Regime-conditional mining
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeSegment:
    """One regime's slice of the source frame."""
    regime: str
    index: pd.Index
    n_rows: int


def segment_by_regime(frame: pd.DataFrame, regime_col: str,
                       *, min_rows: int = DEFAULT_MIN_ROWS_PER_REGIME,
                       ) -> List[RegimeSegment]:
    """Split the frame by ``regime_col``. Regimes with fewer than
    ``min_rows`` rows are dropped — under that floor the per-regime
    quantiles are noise and any mined edge is overfit."""
    if regime_col not in frame.columns:
        raise KeyError(f"missing regime column {regime_col!r}")
    out: List[RegimeSegment] = []
    for label, sub in frame.groupby(regime_col, sort=True):
        n = int(len(sub))
        if n < min_rows:
            continue
        out.append(RegimeSegment(regime=str(label), index=sub.index, n_rows=n))
    return out


@dataclass(frozen=True)
class RegimePerformance:
    """How one hypothesis performs in one regime."""
    regime: str
    n_rows: int
    n_trades: int
    mean_r: float
    sharpe: float
    win_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def evaluate_across_regimes(frame: pd.DataFrame,
                             spec: HypothesisSpec,
                             *,
                             regime_col: str,
                             forward_return_col: str,
                             cost_r_col: Optional[str] = None,
                             min_rows: int = DEFAULT_MIN_ROWS_PER_REGIME,
                             min_trades_per_regime: int = 5,
                             ) -> List[RegimePerformance]:
    """Score one hypothesis separately in each regime.

    The spec is held fixed across regimes — this answers "where does
    THIS rule work?" If you also want to *find* regime-specific rules
    use :func:`mine_per_regime` which samples specs from each regime's
    own quantile structure."""
    segments = segment_by_regime(frame, regime_col, min_rows=min_rows)
    out: List[RegimePerformance] = []
    for seg in segments:
        sub = frame.loc[seg.index]
        m = evaluate_hypothesis(
            sub, spec,
            forward_return_col=forward_return_col,
            cost_r_col=cost_r_col,
            min_trades=min_trades_per_regime,
        )
        out.append(RegimePerformance(
            regime=seg.regime,
            n_rows=int(seg.n_rows),
            n_trades=int(m.trades),
            mean_r=float(m.mean_r),
            sharpe=float(m.sharpe),
            win_rate=float(m.win_rate),
        ))
    return out


def regime_specialization_score(performances: Sequence[RegimePerformance],
                                *, min_trades_per_regime: int = 10,
                                ) -> float:
    """Score how concentrated a hypothesis's edge is in one regime.

    Returns a value in ``[0, 1]``: 0 means the hypothesis's positive-
    Sharpe edge is spread evenly across regimes (always-on), 1 means
    it concentrates in exactly one regime. Specialization is interesting
    because it makes the live regime classifier a tradeable gate: route
    a specialist only when its preferred regime is active.

    Calculation: clip per-regime Sharpe at zero (a regime with negative
    Sharpe contributes nothing to the positive edge), then compute
    ``1 - normalized_entropy`` over the resulting distribution. Regimes
    with too few trades are excluded — they can't reliably contribute
    or detract from the score."""
    pos = [max(0.0, p.sharpe) for p in performances
           if p.n_trades >= min_trades_per_regime]
    if not pos or sum(pos) <= 0.0:
        return 0.0
    if len(pos) == 1:
        return 1.0
    total = sum(pos)
    probs = [p / total for p in pos]
    entropy = -sum(q * math.log(q) for q in probs if q > 0)
    max_entropy = math.log(len(pos))
    if max_entropy <= 0:
        return 0.0
    return float(1.0 - entropy / max_entropy)


def _sample_specs_within_quantiles(frame: pd.DataFrame,
                                    *,
                                    n: int,
                                    seed: int,
                                    feature_columns: Optional[Sequence[str]],
                                    max_conditions: int = 3,
                                    quantiles: Sequence[float] = (0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85),
                                    name_prefix: str = "regime",
                                    ) -> List[HypothesisSpec]:
    """Like ``sample_random_hypotheses`` but parametrised by the caller
    so a per-regime miner can use the regime's own quantiles."""
    import random
    rng = random.Random(seed)
    features = _usable_features(frame, feature_columns)
    if not features:
        return []
    thresholds = _feature_thresholds(frame, features, quantiles)
    out: List[HypothesisSpec] = []
    for i in range(n):
        n_conds = rng.randint(1, max(1, max_conditions))
        picked = rng.sample(features, k=min(n_conds, len(features)))
        conds: List[Condition] = []
        for feature in picked:
            pool = thresholds.get(feature) or [0.0]
            conds.append(Condition(
                feature=feature,
                op=rng.choice(OPS),
                threshold=float(rng.choice(pool)),
            ))
        side = rng.choice(["long", "short"])
        out.append(HypothesisSpec(
            name=f"{name_prefix}_{seed}_{i:05d}",
            side=side,
            conditions=conds,
            hold_bars=rng.choice([3, 6, 12, 24, 36, 60]),
            stop_atr=rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]),
            target_atr=rng.choice([0.75, 1.0, 1.5, 2.0, 3.0]),
            metadata={"generator": "regime_conditional_v1", "seed": seed,
                      "name_prefix": name_prefix},
        ))
    return out


def mine_per_regime(frame: pd.DataFrame,
                     *,
                     regime_col: str,
                     forward_return_col: str,
                     cost_r_col: Optional[str] = None,
                     n_per_regime: int = 300,
                     seed: int = 7,
                     feature_columns: Optional[Sequence[str]] = None,
                     min_rows: int = DEFAULT_MIN_ROWS_PER_REGIME,
                     min_trades: int = 20,
                     max_conditions: int = 3,
                     ) -> pd.DataFrame:
    """Sample and score hypotheses *per regime*, using each regime's
    own feature quantiles for thresholds.

    Returns a DataFrame with one row per (regime, hypothesis) sorted by
    that regime's ``score``. ``regime_col`` should not be in
    ``feature_columns`` (otherwise a regime miner would re-discover
    the regime label as its own predictor). This is enforced
    automatically.
    """
    segments = segment_by_regime(frame, regime_col, min_rows=min_rows)
    if not segments:
        return pd.DataFrame()
    feature_candidates = (
        [c for c in (feature_columns or frame.columns)
         if c != regime_col and c != forward_return_col and c != cost_r_col]
    )
    rows: List[Dict[str, Any]] = []
    for offset, seg in enumerate(segments):
        sub = frame.loc[seg.index]
        specs = _sample_specs_within_quantiles(
            sub, n=n_per_regime, seed=seed + offset * 1000,
            feature_columns=feature_candidates,
            max_conditions=max_conditions,
            name_prefix=f"regime_{seg.regime}",
        )
        for spec in specs:
            m = evaluate_hypothesis(
                sub, spec,
                forward_return_col=forward_return_col,
                cost_r_col=cost_r_col,
                min_trades=min_trades,
            )
            d = m.to_dict()
            d["regime"] = seg.regime
            d["regime_n_rows"] = seg.n_rows
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["regime", "score", "mean_R", "trades"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
