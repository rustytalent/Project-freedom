"""Tests for regime-conditional + walk-forward hypothesis mining.

These pin the discipline that makes the module trustworthy:

  * Walk-forward folds are chronological, non-overlapping, sized above
    a minimum, and consistent across reruns.
  * Per-regime evaluation respects the regime split (a hypothesis that
    works only in regime A should report ~0 Sharpe in regime B).
  * Specialization score is 0 when edge is uniform and 1 when edge
    concentrates in one regime; in between it grades.
  * The miner uses per-regime quantiles for thresholds (the trick that
    makes regime A's "top 30% of x" mean something different than
    regime B's "top 30% of x" when their distributions differ).
  * Edge cases — empty frames, all-zero Sharpe, missing regime
    column — are handled without crashing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liqpool.research.hypothesis_miner import Condition, HypothesisSpec
from liqpool.research.regime_miner import (
    DEFAULT_MIN_BARS_PER_FOLD,
    DEFAULT_MIN_ROWS_PER_REGIME,
    FoldResult,
    RegimePerformance,
    RegimeSegment,
    WalkForwardMetrics,
    _fold_slices,
    evaluate_across_regimes,
    mine_per_regime,
    rank_walk_forward_hypotheses,
    regime_specialization_score,
    segment_by_regime,
    walk_forward_metrics,
)


def _frame_with_regime_specialist(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Build a frame where ``x > 0.5`` predicts positive forward return
    only in regime A; regime B is pure noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    regime = np.where(np.arange(n) % 2 == 0, "A", "B")
    fwd_r = np.where(
        regime == "A",
        np.where(x > 0.5, 0.8, -0.2) + rng.normal(0, 0.4, n),
        rng.normal(0, 0.5, n),
    )
    return pd.DataFrame({"x": x, "y": y, "regime": regime, "fwd_r": fwd_r})


def _spec_x_above_half_long() -> HypothesisSpec:
    return HypothesisSpec(
        name="x_above_half_long",
        side="long",
        conditions=[Condition(feature="x", op=">", threshold=0.5)],
        hold_bars=6,
    )


# ─────────────────────────────────────────────────────────────────
# Fold slicing
# ─────────────────────────────────────────────────────────────────

def test_fold_slices_are_chronological_and_non_overlapping():
    slices = _fold_slices(n_rows=2400, n_folds=5,
                          min_bars_per_fold=DEFAULT_MIN_BARS_PER_FOLD)
    assert len(slices) > 0
    prev_end = 0
    for start, end in slices:
        assert start >= prev_end, "folds must not overlap"
        assert end > start
        assert end - start >= DEFAULT_MIN_BARS_PER_FOLD
        prev_end = end


def test_fold_slices_empty_when_data_too_small():
    # Need >= 2 * min_bars_per_fold rows to get any folds.
    assert _fold_slices(n_rows=300, n_folds=5,
                         min_bars_per_fold=200) == []


# ─────────────────────────────────────────────────────────────────
# Walk-forward fitness
# ─────────────────────────────────────────────────────────────────

def test_walk_forward_metrics_surfaces_persistent_edge():
    frame = _frame_with_regime_specialist(n=2000, seed=11)
    wf = walk_forward_metrics(
        frame, _spec_x_above_half_long(),
        forward_return_col="fwd_r",
        n_folds=4, min_bars_per_fold=200, min_trades_per_fold=5,
    )
    # The truth lives in regime A which is every other row, so the rule
    # has positive OOS Sharpe in every fold.
    assert wf.consistency == pytest.approx(1.0)
    assert wf.mean_oos_sharpe > 0.30
    assert wf.fitness > 0


def test_walk_forward_metrics_punishes_no_edge():
    rng = np.random.default_rng(3)
    n = 1600
    frame = pd.DataFrame({
        "x": rng.uniform(0, 1, n),
        "fwd_r": rng.normal(0, 0.5, n),     # pure noise — no relationship to x
    })
    wf = walk_forward_metrics(
        frame, _spec_x_above_half_long(),
        forward_return_col="fwd_r",
        n_folds=4, min_bars_per_fold=200, min_trades_per_fold=5,
    )
    # Mean OOS Sharpe should be near zero and fitness tiny.
    assert abs(wf.mean_oos_sharpe) < 0.30
    assert wf.fitness < 0.25
    # Consistency on noise will hover around 0.5 (~half the folds show positive Sharpe by chance).
    assert 0.0 <= wf.consistency <= 1.0


def test_walk_forward_metrics_handles_too_small_frame():
    frame = pd.DataFrame({"x": np.linspace(0, 1, 50),
                          "fwd_r": np.linspace(0.1, -0.1, 50)})
    wf = walk_forward_metrics(
        frame, _spec_x_above_half_long(),
        forward_return_col="fwd_r",
        n_folds=4, min_bars_per_fold=200, min_trades_per_fold=5,
    )
    assert wf.n_folds == 0
    assert wf.mean_oos_sharpe == 0.0
    assert wf.fitness == 0.0


def test_walk_forward_metrics_deterministic_for_same_inputs():
    frame = _frame_with_regime_specialist(n=2000, seed=13)
    spec = _spec_x_above_half_long()
    wf1 = walk_forward_metrics(frame, spec, forward_return_col="fwd_r")
    wf2 = walk_forward_metrics(frame, spec, forward_return_col="fwd_r")
    assert wf1.mean_oos_sharpe == wf2.mean_oos_sharpe
    assert wf1.fitness == wf2.fitness
    assert [f.sharpe for f in wf1.folds] == [f.sharpe for f in wf2.folds]


def test_walk_forward_metrics_to_dict_is_json_serializable():
    frame = _frame_with_regime_specialist(n=1500, seed=17)
    wf = walk_forward_metrics(frame, _spec_x_above_half_long(),
                               forward_return_col="fwd_r")
    import json
    json.dumps(wf.to_dict())


def test_rank_walk_forward_hypotheses_sorts_by_fitness():
    frame = _frame_with_regime_specialist(n=2400, seed=19)
    ranked = rank_walk_forward_hypotheses(
        frame, forward_return_col="fwd_r",
        n=30, seed=19, n_folds=4, min_bars_per_fold=200,
        feature_columns=["x", "y"],
    )
    assert not ranked.empty
    # Sorted descending by fitness.
    assert (np.diff(ranked["fitness"].to_numpy()) <= 1e-9).all()


# ─────────────────────────────────────────────────────────────────
# Per-regime segmentation + evaluation
# ─────────────────────────────────────────────────────────────────

def test_segment_by_regime_drops_under_min_rows():
    frame = pd.DataFrame({"x": np.arange(500), "regime": ["big"] * 400 + ["tiny"] * 100})
    segs = segment_by_regime(frame, "regime", min_rows=200)
    names = sorted(s.regime for s in segs)
    assert names == ["big"]


def test_segment_by_regime_raises_on_missing_column():
    frame = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(KeyError, match="regime"):
        segment_by_regime(frame, "missing_regime", min_rows=1)


def test_evaluate_across_regimes_isolates_signal_regime():
    frame = _frame_with_regime_specialist(n=2000, seed=23)
    perfs = evaluate_across_regimes(
        frame, _spec_x_above_half_long(),
        regime_col="regime", forward_return_col="fwd_r",
        min_rows=200, min_trades_per_regime=5,
    )
    perfs_by_regime = {p.regime: p for p in perfs}
    assert perfs_by_regime["A"].sharpe > 0.5
    # B is pure noise so its rule-Sharpe should be near zero.
    assert abs(perfs_by_regime["B"].sharpe) < 0.30


# ─────────────────────────────────────────────────────────────────
# Specialization score
# ─────────────────────────────────────────────────────────────────

def test_specialization_score_max_when_one_regime_carries_edge():
    perfs = [
        RegimePerformance(regime="A", n_rows=500, n_trades=100,
                          mean_r=0.5, sharpe=1.5, win_rate=0.6),
        RegimePerformance(regime="B", n_rows=500, n_trades=100,
                          mean_r=0.0, sharpe=0.0, win_rate=0.5),
        RegimePerformance(regime="C", n_rows=500, n_trades=100,
                          mean_r=-0.1, sharpe=-0.2, win_rate=0.4),
    ]
    # Only A contributes positive edge -> specialization is 1.0.
    assert regime_specialization_score(perfs) == pytest.approx(1.0)


def test_specialization_score_zero_when_edge_uniform_across_regimes():
    perfs = [
        RegimePerformance(regime="A", n_rows=500, n_trades=100,
                          mean_r=0.3, sharpe=0.8, win_rate=0.55),
        RegimePerformance(regime="B", n_rows=500, n_trades=100,
                          mean_r=0.3, sharpe=0.8, win_rate=0.55),
        RegimePerformance(regime="C", n_rows=500, n_trades=100,
                          mean_r=0.3, sharpe=0.8, win_rate=0.55),
    ]
    assert regime_specialization_score(perfs) == pytest.approx(0.0, abs=1e-9)


def test_specialization_score_excludes_low_trade_regimes():
    """A regime with too few trades should not influence the
    specialization score, otherwise a single-trade lucky regime would
    look more concentrated than it actually is."""
    perfs = [
        RegimePerformance(regime="A", n_rows=500, n_trades=100,
                          mean_r=0.5, sharpe=1.5, win_rate=0.6),
        RegimePerformance(regime="B", n_rows=10, n_trades=2,
                          mean_r=2.0, sharpe=10.0, win_rate=1.0),
    ]
    # B has too few trades; A alone -> score 1.0 (not influenced by lucky B).
    assert regime_specialization_score(perfs, min_trades_per_regime=10) == pytest.approx(1.0)


def test_specialization_score_zero_when_no_positive_edge():
    perfs = [RegimePerformance(regime="A", n_rows=500, n_trades=100,
                                mean_r=-0.1, sharpe=-0.5, win_rate=0.4)]
    assert regime_specialization_score(perfs) == 0.0


# ─────────────────────────────────────────────────────────────────
# Per-regime mining
# ─────────────────────────────────────────────────────────────────

def test_mine_per_regime_surfaces_signal_regime_above_noise_regime():
    frame = _frame_with_regime_specialist(n=2400, seed=27)
    mined = mine_per_regime(
        frame, regime_col="regime", forward_return_col="fwd_r",
        n_per_regime=100, seed=27, min_rows=200, min_trades=20,
        feature_columns=["x", "y"],
    )
    assert not mined.empty
    assert set(mined["regime"].unique()) == {"A", "B"}
    top_A = mined[mined["regime"] == "A"]["score"].max()
    top_B = mined[mined["regime"] == "B"]["score"].max()
    assert top_A > top_B + 0.10, (
        f"A's top mined rule should beat B's by a wide margin "
        f"(A={top_A:.3f}, B={top_B:.3f})"
    )


def test_mine_per_regime_excludes_regime_column_from_features():
    """The regime label must never be in the candidate features —
    otherwise the miner re-discovers the regime as its own predictor."""
    frame = _frame_with_regime_specialist(n=1200, seed=31)
    mined = mine_per_regime(
        frame, regime_col="regime", forward_return_col="fwd_r",
        n_per_regime=50, seed=31, min_rows=200, min_trades=10,
        feature_columns=None,  # default: all columns except regime/fwd_r
    )
    # Sanity: condition features should never include "regime".
    for _, row in mined.iterrows():
        import json as _json
        spec = _json.loads(row["spec_json"])
        for cond in spec["conditions"]:
            assert cond["feature"] != "regime"


def test_mine_per_regime_returns_empty_on_undersized_regimes():
    frame = pd.DataFrame({"x": np.arange(100), "regime": ["X"] * 100,
                          "fwd_r": np.linspace(-0.1, 0.1, 100)})
    mined = mine_per_regime(
        frame, regime_col="regime", forward_return_col="fwd_r",
        n_per_regime=50, seed=37, min_rows=200, min_trades=10,
    )
    assert mined.empty


# ─────────────────────────────────────────────────────────────────
# Composition / integration
# ─────────────────────────────────────────────────────────────────

def test_walk_forward_and_evaluate_across_regimes_compose_for_specialist():
    """A specialist hypothesis must show BOTH:
      * stable positive OOS Sharpe under walk-forward, AND
      * a high specialization score across regimes.
    Both together is the discovery signature the operator is looking for."""
    frame = _frame_with_regime_specialist(n=2400, seed=41)
    spec = _spec_x_above_half_long()
    wf = walk_forward_metrics(frame, spec, forward_return_col="fwd_r",
                               n_folds=4, min_bars_per_fold=200)
    perfs = evaluate_across_regimes(frame, spec, regime_col="regime",
                                     forward_return_col="fwd_r", min_rows=200)
    spec_score = regime_specialization_score(perfs)
    assert wf.consistency >= 0.75
    assert wf.mean_oos_sharpe > 0.30
    assert spec_score > 0.30
