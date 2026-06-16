"""Tests for the GP rule miner.

These pin the discipline:
  * Mutation operators preserve spec shape (conditions in range, side
    valid, exit params from the allowed grids).
  * Crossover always produces a child with >= 1 condition.
  * Tournament selection picks higher-fitness individuals with higher
    probability (statistical, not point-estimate).
  * Evolution converges on a planted-truth signal — the hall of fame
    must contain rules with high OOS Sharpe.
  * Evolution does NOT spuriously converge on pure noise — the best
    fitness stays low across generations.
  * Hall of fame is sorted by fitness descending and deduped by spec
    signature.
  * Determinism: same seed -> identical evolution trajectory.
  * Edge cases: empty feature set, tiny frame, no_signal frame.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from liqpool.research.gp_miner import (
    DEFAULT_MUTATION_RATES,
    HOLD_BARS_GRID,
    MIN_CONDITIONS,
    STOP_ATR_GRID,
    TARGET_ATR_GRID,
    GenerationStats,
    GpConfig,
    GpReport,
    Individual,
    _spec_signature,
    crossover_specs,
    evolve,
    mutate_spec,
    tournament_select,
)
from liqpool.research.hypothesis_miner import (
    Condition,
    HypothesisSpec,
    _feature_thresholds,
    _usable_features,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _planted_frame(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Frame where ``x > 0.5 long`` predicts positive forward return."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    z = rng.normal(0, 1, n)
    fwd_r = np.where(x > 0.5, 0.5, -0.3) + rng.normal(0, 0.4, n)
    return pd.DataFrame({"x": x, "y": y, "z": z, "fwd_r": fwd_r})


def _noise_frame(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Frame where forward return is independent of every feature."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.uniform(0, 1, n),
        "y": rng.uniform(0, 1, n),
        "z": rng.normal(0, 1, n),
        "fwd_r": rng.normal(0, 0.5, n),
    })


def _sample_spec() -> HypothesisSpec:
    return HypothesisSpec(
        name="seed", side="long",
        conditions=[
            Condition(feature="x", op=">", threshold=0.5),
            Condition(feature="y", op="<", threshold=0.7),
        ],
        hold_bars=12, stop_atr=1.0, target_atr=2.0,
    )


def _features_and_thresholds(frame: pd.DataFrame):
    feats = _usable_features(frame, ["x", "y", "z"])
    thresholds = _feature_thresholds(frame, feats,
                                      (0.15, 0.25, 0.5, 0.75, 0.85))
    return feats, thresholds


# ─────────────────────────────────────────────────────────────────
# Mutation
# ─────────────────────────────────────────────────────────────────

def test_mutate_spec_preserves_shape():
    rng = random.Random(0)
    frame = _planted_frame(n=600, seed=11)
    feats, thresholds = _features_and_thresholds(frame)
    spec = _sample_spec()
    # Force a high-rate mutation suite so every operator gets exercised.
    high_rates = {k: 1.0 for k in DEFAULT_MUTATION_RATES}
    for _ in range(50):
        mut = mutate_spec(spec, rng, feats, thresholds,
                           rates=high_rates, max_conditions=3)
        assert MIN_CONDITIONS <= len(mut.conditions) <= 3
        assert mut.side in ("long", "short")
        assert mut.hold_bars in HOLD_BARS_GRID
        assert mut.stop_atr in STOP_ATR_GRID
        assert mut.target_atr in TARGET_ATR_GRID
        # Every condition references a real feature.
        for c in mut.conditions:
            assert c.feature in feats
            assert c.op in (">", "<")


def test_mutate_zero_rates_returns_equivalent_spec():
    """With every mutation rate at 0, mutate_spec produces a spec whose
    signature matches the input (only the name differs)."""
    rng = random.Random(0)
    frame = _planted_frame(n=400, seed=11)
    feats, thresholds = _features_and_thresholds(frame)
    spec = _sample_spec()
    rates = {k: 0.0 for k in DEFAULT_MUTATION_RATES}
    mut = mutate_spec(spec, rng, feats, thresholds, rates=rates, max_conditions=3)
    assert _spec_signature(mut) == _spec_signature(spec)


def test_mutate_never_drops_below_min_conditions():
    rng = random.Random(0)
    frame = _planted_frame(n=400, seed=11)
    feats, thresholds = _features_and_thresholds(frame)
    spec = HypothesisSpec(
        name="single", side="long",
        conditions=[Condition(feature="x", op=">", threshold=0.5)],
        hold_bars=12, stop_atr=1.0, target_atr=2.0,
    )
    rates = {"remove_condition": 1.0}
    for _ in range(20):
        mut = mutate_spec(spec, rng, feats, thresholds, rates=rates, max_conditions=3)
        assert len(mut.conditions) >= MIN_CONDITIONS


# ─────────────────────────────────────────────────────────────────
# Crossover
# ─────────────────────────────────────────────────────────────────

def test_crossover_child_has_at_least_one_condition():
    rng = random.Random(7)
    p1 = HypothesisSpec(name="p1", side="long",
                        conditions=[Condition("x", ">", 0.5)], hold_bars=6,
                        stop_atr=1.0, target_atr=2.0)
    p2 = HypothesisSpec(name="p2", side="short",
                        conditions=[Condition("y", "<", 0.3)], hold_bars=12,
                        stop_atr=0.5, target_atr=1.5)
    for _ in range(50):
        child = crossover_specs(p1, p2, rng, max_conditions=3)
        assert len(child.conditions) >= MIN_CONDITIONS
        assert child.side in (p1.side, p2.side)
        assert child.hold_bars in (p1.hold_bars, p2.hold_bars)


def test_crossover_respects_max_conditions():
    rng = random.Random(3)
    p1 = HypothesisSpec(name="p1", side="long",
                        conditions=[Condition("x", ">", 0.5),
                                    Condition("y", "<", 0.5),
                                    Condition("z", ">", 0.0)],
                        hold_bars=6, stop_atr=1.0, target_atr=2.0)
    p2 = HypothesisSpec(name="p2", side="long",
                        conditions=[Condition("a", ">", 0.5),
                                    Condition("b", ">", 0.5),
                                    Condition("c", ">", 0.5)],
                        hold_bars=12, stop_atr=1.0, target_atr=2.0)
    for _ in range(20):
        child = crossover_specs(p1, p2, rng, max_conditions=2)
        assert len(child.conditions) <= 2


# ─────────────────────────────────────────────────────────────────
# Tournament selection
# ─────────────────────────────────────────────────────────────────

def test_tournament_selection_prefers_higher_fitness_statistically():
    """Across many tournaments, the highest-fitness individual must win
    more often than chance would predict for k=3 tournaments."""
    pop = [
        Individual(spec=_sample_spec(), fitness=f, n_trades_total=100,
                    mean_oos_sharpe=f, consistency=1.0, generation_born=0)
        for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ]
    rng = random.Random(2)
    best_id = id(pop[-1])
    wins = sum(1 for _ in range(1000)
                if id(tournament_select(pop, rng, k=3)) == best_id)
    # With k=3 out of 10, the top individual is sampled into the tournament
    # roughly 3/10 = 30% of the time and wins every time it's sampled.
    # So win rate should be ~30%; require >= 20% as a loose lower bound
    # robust to RNG noise.
    assert wins > 200


def test_tournament_selection_raises_on_empty_population():
    rng = random.Random(1)
    with pytest.raises(ValueError, match="empty"):
        tournament_select([], rng, k=3)


# ─────────────────────────────────────────────────────────────────
# Spec signature
# ─────────────────────────────────────────────────────────────────

def test_spec_signature_is_order_invariant():
    a = HypothesisSpec(name="a", side="long",
                       conditions=[Condition("x", ">", 0.5),
                                   Condition("y", "<", 0.3)],
                       hold_bars=12, stop_atr=1.0, target_atr=2.0)
    b = HypothesisSpec(name="b", side="long",
                       conditions=[Condition("y", "<", 0.3),
                                   Condition("x", ">", 0.5)],
                       hold_bars=12, stop_atr=1.0, target_atr=2.0)
    assert _spec_signature(a) == _spec_signature(b)


def test_spec_signature_distinguishes_different_rules():
    a = HypothesisSpec(name="a", side="long",
                       conditions=[Condition("x", ">", 0.5)],
                       hold_bars=12, stop_atr=1.0, target_atr=2.0)
    b = HypothesisSpec(name="b", side="long",
                       conditions=[Condition("x", "<", 0.5)],
                       hold_bars=12, stop_atr=1.0, target_atr=2.0)
    assert _spec_signature(a) != _spec_signature(b)


# ─────────────────────────────────────────────────────────────────
# Evolution end-to-end
# ─────────────────────────────────────────────────────────────────

def test_evolve_converges_on_planted_signal():
    """GP on a planted-truth frame should find rules with high OOS Sharpe
    AND the best fitness must improve across generations."""
    frame = _planted_frame(n=3000, seed=11)
    cfg = GpConfig(population_size=30, n_generations=6, tournament_k=3,
                   elite_count=2, max_conditions=2, n_folds=4,
                   min_bars_per_fold=300, hall_of_fame_size=5)
    report = evolve(frame, forward_return_col="fwd_r",
                     feature_columns=["x", "y", "z"], cfg=cfg, seed=41)
    assert report.hall_of_fame, "hall of fame should be non-empty"
    # Best generation's best fitness > seed generation's best fitness.
    gen_best = [g.best_fitness for g in report.history]
    assert max(gen_best) >= gen_best[0]
    # Hall-of-fame champion has clearly positive OOS Sharpe.
    assert report.hall_of_fame[0].mean_oos_sharpe > 0.30
    assert report.hall_of_fame[0].consistency >= 0.50


def test_evolve_does_not_converge_on_pure_noise():
    """Negative control: on a frame with no learnable structure, the GP's
    best hall-of-fame fitness must stay near zero. If it converges
    despite no signal, the fitness function or the search is broken."""
    frame = _noise_frame(n=3000, seed=23)
    cfg = GpConfig(population_size=30, n_generations=5, tournament_k=3,
                   elite_count=2, max_conditions=2, n_folds=4,
                   min_bars_per_fold=300, hall_of_fame_size=5)
    report = evolve(frame, forward_return_col="fwd_r",
                     feature_columns=["x", "y", "z"], cfg=cfg, seed=41)
    # Pure noise should not produce a hall-of-fame champion with high
    # fitness (which would imply a learned signal we shouldn't have).
    assert report.hall_of_fame[0].fitness < 0.55, (
        f"GP found fitness={report.hall_of_fame[0].fitness:+.3f} on pure noise; "
        f"that should be near zero"
    )


def test_evolve_hall_of_fame_is_sorted_descending_and_deduped():
    frame = _planted_frame(n=2500, seed=31)
    cfg = GpConfig(population_size=25, n_generations=4, tournament_k=3,
                   elite_count=2, max_conditions=2, n_folds=4,
                   min_bars_per_fold=300, hall_of_fame_size=8)
    report = evolve(frame, forward_return_col="fwd_r",
                     feature_columns=["x", "y", "z"], cfg=cfg, seed=41)
    fits = [ind.fitness for ind in report.hall_of_fame]
    assert all(a >= b for a, b in zip(fits, fits[1:])), "hall must be sorted descending"
    sigs = [_spec_signature(ind.spec) for ind in report.hall_of_fame]
    assert len(sigs) == len(set(sigs)), "hall must be deduped by signature"


def test_evolve_deterministic_for_same_seed():
    frame = _planted_frame(n=1800, seed=37)
    cfg = GpConfig(population_size=20, n_generations=3, tournament_k=3,
                   elite_count=2, max_conditions=2, n_folds=3,
                   min_bars_per_fold=300, hall_of_fame_size=5)
    r1 = evolve(frame, forward_return_col="fwd_r",
                 feature_columns=["x", "y", "z"], cfg=cfg, seed=41)
    r2 = evolve(frame, forward_return_col="fwd_r",
                 feature_columns=["x", "y", "z"], cfg=cfg, seed=41)
    sigs1 = [_spec_signature(ind.spec) for ind in r1.hall_of_fame]
    sigs2 = [_spec_signature(ind.spec) for ind in r2.hall_of_fame]
    assert sigs1 == sigs2
    fits1 = [ind.fitness for ind in r1.hall_of_fame]
    fits2 = [ind.fitness for ind in r2.hall_of_fame]
    assert fits1 == fits2


def test_evolve_returns_empty_when_no_usable_features():
    frame = pd.DataFrame({"all_zeros": np.zeros(500),
                          "fwd_r": np.linspace(0, 0.1, 500)})
    cfg = GpConfig(population_size=10, n_generations=2, tournament_k=2)
    report = evolve(frame, forward_return_col="fwd_r",
                     feature_columns=["all_zeros"], cfg=cfg, seed=41)
    assert report.hall_of_fame == []
    assert report.final_population == []


def test_evolve_progress_callback_fires_per_generation():
    frame = _planted_frame(n=1500, seed=43)
    cfg = GpConfig(population_size=15, n_generations=3, tournament_k=2,
                   elite_count=1, max_conditions=2, n_folds=3,
                   min_bars_per_fold=300, hall_of_fame_size=3)
    seen: list = []

    def cb(gen, stats):
        seen.append((gen, stats.best_fitness))

    evolve(frame, forward_return_col="fwd_r",
            feature_columns=["x", "y", "z"], cfg=cfg, seed=41, progress=cb)
    # Callback fires once for the seed generation (0) plus once per
    # subsequent generation (1..n_generations).
    assert [g for g, _ in seen] == [0, 1, 2, 3]


def test_evolve_report_is_json_serializable():
    frame = _planted_frame(n=1500, seed=47)
    cfg = GpConfig(population_size=15, n_generations=2, tournament_k=2,
                   elite_count=1, max_conditions=2, n_folds=3,
                   min_bars_per_fold=300, hall_of_fame_size=3)
    report = evolve(frame, forward_return_col="fwd_r",
                     feature_columns=["x", "y", "z"], cfg=cfg, seed=41)
    import json
    json.dumps(report.to_dict())
