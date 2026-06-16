"""Genetic programming over the rule grammar (playbook idea #2).

Codex's ``hypothesis_miner.py`` samples rules at random and ranks them.
That works well as a coarse first pass but ignores a free signal — once
you know which rules score well, the *neighbours* of those rules in the
grammar space are disproportionately likely to also score well. A
genetic algorithm exploits exactly that: keep the high-fitness
individuals, mutate them, breed pairs of them, evaluate the next
generation, repeat.

The fitness function is the existing walk-forward fitness
(``regime_miner.walk_forward_metrics``). This matters: a random miner
that scored on in-sample fit would converge a GA toward overfit; using
walk-forward fitness as the GA target rewards *consistent OOS* edge,
which is what we actually want to deploy.

Mutation operators:

  * ``change_threshold`` — re-pick one condition's threshold from the
    feature's quantile pool.
  * ``change_op`` — flip a condition's comparison.
  * ``change_feature`` — replace a condition entirely.
  * ``add_condition`` — extend the AND, up to ``max_conditions``.
  * ``remove_condition`` — drop a condition, never below 1.
  * ``change_side`` — flip long ↔ short.
  * ``change_hold_bars`` — sample a different hold horizon.
  * ``change_stop`` / ``change_target`` — re-sample the exit grid.

Crossover takes the union of two parents' conditions and keeps a
random subset; exit params come from one parent at random. Tournament
selection (size 3) keeps selection pressure bounded so the population
doesn't collapse onto one winner in two generations.

What this module is NOT:
  * A real-time learner. It is a research tool to be run offline against
    a frame plus a forward-return column. Outputs go to the harness +
    paper-trade gauntlet like any other hypothesis.
  * A guarantee of edge. The hall-of-fame champion must still pass
    ``synthetic_validation``, ``kill_switch``, and live paper before
    it sees real capital. The GA is upstream of those gates.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .hypothesis_miner import (
    Condition,
    HypothesisSpec,
    OPS,
    _feature_thresholds,
    _usable_features,
)
from .regime_miner import walk_forward_metrics


# Grid of allowed exit horizons. Same shape the random miner uses so a
# GP individual stays in-distribution with the random seed population.
HOLD_BARS_GRID: Tuple[int, ...] = (3, 6, 12, 24, 36, 60)
STOP_ATR_GRID: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
TARGET_ATR_GRID: Tuple[float, ...] = (0.75, 1.0, 1.5, 2.0, 3.0)

# Default mutation rates per individual per generation. Each rate is
# the probability that THAT operator fires at all; if it fires, it
# picks one condition (or exit param) to act on.
DEFAULT_MUTATION_RATES: Dict[str, float] = {
    "change_threshold": 0.30,
    "change_op": 0.10,
    "change_feature": 0.15,
    "add_condition": 0.10,
    "remove_condition": 0.10,
    "change_side": 0.05,
    "change_hold_bars": 0.10,
    "change_stop": 0.05,
    "change_target": 0.05,
}

# A condition we can't reduce below this number. Keeping at least one
# condition keeps the spec interpretable and prevents the GA from
# wandering into "fire on every bar" trivial individuals.
MIN_CONDITIONS: int = 1


@dataclass
class GpConfig:
    """All knobs for one evolve() call.

    Defaults are chosen to be safe on a 2000-row synthetic frame; for
    real warehouse data with 50k+ rows, push ``population_size`` and
    ``n_generations`` up."""
    population_size: int = 80
    n_generations: int = 12
    tournament_k: int = 3
    elite_count: int = 4
    crossover_prob: float = 0.55
    mutation_rates: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MUTATION_RATES))
    max_conditions: int = 3
    quantiles: Tuple[float, ...] = (0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85)
    hall_of_fame_size: int = 10
    novelty_penalty: float = 0.05
    # Walk-forward inputs forwarded to walk_forward_metrics.
    n_folds: int = 4
    min_bars_per_fold: int = 200
    min_trades_per_fold: int = 5


@dataclass(frozen=True)
class Individual:
    """One member of the GA population."""
    spec: HypothesisSpec
    fitness: float
    n_trades_total: int
    mean_oos_sharpe: float
    consistency: float
    generation_born: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "fitness": float(self.fitness),
            "n_trades_total": int(self.n_trades_total),
            "mean_oos_sharpe": float(self.mean_oos_sharpe),
            "consistency": float(self.consistency),
            "generation_born": int(self.generation_born),
        }


@dataclass
class GenerationStats:
    """Per-generation summary so the operator can plot convergence."""
    generation: int
    best_fitness: float
    median_fitness: float
    mean_fitness: float
    diversity: float       # fraction of unique spec signatures

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class GpReport:
    """Final output of an evolve() run."""
    hall_of_fame: List[Individual] = field(default_factory=list)
    final_population: List[Individual] = field(default_factory=list)
    history: List[GenerationStats] = field(default_factory=list)
    config: Optional[GpConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hall_of_fame": [ind.to_dict() for ind in self.hall_of_fame],
            "final_population": [ind.to_dict() for ind in self.final_population],
            "history": [g.to_dict() for g in self.history],
            "config": self.config.__dict__ if self.config else None,
        }


# ─────────────────────────────────────────────────────────────────
# Operators
# ─────────────────────────────────────────────────────────────────

def _spec_signature(spec: HypothesisSpec) -> str:
    """Order-invariant signature used for diversity + duplicate detection."""
    cond_sig = tuple(sorted(
        (c.feature, c.op, round(c.threshold, 6)) for c in spec.conditions
    ))
    return repr((spec.side, cond_sig, int(spec.hold_bars),
                  round(spec.stop_atr, 4), round(spec.target_atr, 4)))


def _new_condition(rng: random.Random, features: Sequence[str],
                   thresholds: Dict[str, List[float]]) -> Condition:
    feature = rng.choice(list(features))
    pool = thresholds.get(feature) or [0.0]
    return Condition(feature=feature, op=rng.choice(OPS),
                      threshold=float(rng.choice(pool)))


def mutate_spec(spec: HypothesisSpec,
                rng: random.Random,
                features: Sequence[str],
                thresholds: Dict[str, List[float]],
                *,
                rates: Optional[Dict[str, float]] = None,
                max_conditions: int = 3,
                ) -> HypothesisSpec:
    """Apply mutation operators stochastically. Returns a new spec —
    the input spec is never modified in place. Operators that don't
    fire leave the corresponding attribute unchanged."""
    r = rates or DEFAULT_MUTATION_RATES
    conditions = list(spec.conditions)
    side = spec.side
    hold_bars = int(spec.hold_bars)
    stop_atr = float(spec.stop_atr)
    target_atr = float(spec.target_atr)

    if conditions and rng.random() < r.get("change_threshold", 0.0):
        i = rng.randrange(len(conditions))
        feature = conditions[i].feature
        pool = thresholds.get(feature) or [conditions[i].threshold]
        new_thresh = float(rng.choice(pool))
        conditions[i] = Condition(feature=feature, op=conditions[i].op,
                                   threshold=new_thresh)

    if conditions and rng.random() < r.get("change_op", 0.0):
        i = rng.randrange(len(conditions))
        new_op = ">" if conditions[i].op == "<" else "<"
        conditions[i] = Condition(feature=conditions[i].feature, op=new_op,
                                   threshold=conditions[i].threshold)

    if conditions and rng.random() < r.get("change_feature", 0.0):
        i = rng.randrange(len(conditions))
        conditions[i] = _new_condition(rng, features, thresholds)

    if len(conditions) < max_conditions and rng.random() < r.get("add_condition", 0.0):
        conditions.append(_new_condition(rng, features, thresholds))

    if len(conditions) > MIN_CONDITIONS and rng.random() < r.get("remove_condition", 0.0):
        del conditions[rng.randrange(len(conditions))]

    if rng.random() < r.get("change_side", 0.0):
        side = "short" if side == "long" else "long"

    if rng.random() < r.get("change_hold_bars", 0.0):
        hold_bars = int(rng.choice(HOLD_BARS_GRID))

    if rng.random() < r.get("change_stop", 0.0):
        stop_atr = float(rng.choice(STOP_ATR_GRID))

    if rng.random() < r.get("change_target", 0.0):
        target_atr = float(rng.choice(TARGET_ATR_GRID))

    return HypothesisSpec(
        name=spec.name + "_m",
        side=side,
        conditions=conditions,
        hold_bars=hold_bars,
        stop_atr=stop_atr,
        target_atr=target_atr,
        metadata={**spec.metadata, "ancestor": spec.name},
    )


def crossover_specs(parent1: HypothesisSpec,
                     parent2: HypothesisSpec,
                     rng: random.Random,
                     *, max_conditions: int = 3,
                     ) -> HypothesisSpec:
    """Uniform crossover over conditions + per-attribute parent pick.

    The child gets each condition from parent1 or parent2 (50/50);
    duplicates by ``(feature, op)`` are deduped keeping the most recent.
    Side and exit grid params are inherited from a randomly chosen
    parent each (independent draws — gives the GA more recombination
    diversity than copying all params from one parent)."""
    pool: List[Condition] = []
    for c in parent1.conditions + parent2.conditions:
        if rng.random() < 0.5:
            pool.append(c)
    # Dedupe by (feature, op) keeping last; keep at least one condition
    # so the rule actually fires on something.
    seen: Dict[Tuple[str, str], Condition] = {}
    for c in pool:
        seen[(c.feature, c.op)] = c
    conditions = list(seen.values())[:max_conditions]
    if not conditions:
        conditions = [rng.choice(parent1.conditions or parent2.conditions)]
    side = rng.choice([parent1.side, parent2.side])
    hold_bars = int(rng.choice([parent1.hold_bars, parent2.hold_bars]))
    stop_atr = float(rng.choice([parent1.stop_atr, parent2.stop_atr]))
    target_atr = float(rng.choice([parent1.target_atr, parent2.target_atr]))
    return HypothesisSpec(
        name=f"x_{parent1.name}_{parent2.name}",
        side=side,
        conditions=conditions,
        hold_bars=hold_bars,
        stop_atr=stop_atr,
        target_atr=target_atr,
        metadata={"parents": [parent1.name, parent2.name]},
    )


def tournament_select(population: Sequence[Individual],
                      rng: random.Random, *, k: int = 3) -> Individual:
    """Pick ``k`` random individuals, return the highest-fitness one."""
    if not population:
        raise ValueError("tournament_select called on empty population")
    sampled = rng.sample(list(population), min(k, len(population)))
    return max(sampled, key=lambda ind: ind.fitness)


# ─────────────────────────────────────────────────────────────────
# The main loop
# ─────────────────────────────────────────────────────────────────

def _make_population(frame: pd.DataFrame,
                     *,
                     n: int,
                     seed: int,
                     features: Sequence[str],
                     thresholds: Dict[str, List[float]],
                     max_conditions: int) -> List[HypothesisSpec]:
    rng = random.Random(seed)
    out: List[HypothesisSpec] = []
    for i in range(n):
        n_conds = rng.randint(1, max(1, max_conditions))
        conds = [_new_condition(rng, features, thresholds) for _ in range(n_conds)]
        out.append(HypothesisSpec(
            name=f"gp_seed_{seed}_{i:05d}",
            side=rng.choice(["long", "short"]),
            conditions=conds,
            hold_bars=int(rng.choice(HOLD_BARS_GRID)),
            stop_atr=float(rng.choice(STOP_ATR_GRID)),
            target_atr=float(rng.choice(TARGET_ATR_GRID)),
            metadata={"generator": "gp_seed_v1", "seed": seed},
        ))
    return out


def _evaluate(frame: pd.DataFrame, spec: HypothesisSpec,
              *, forward_return_col: str, cost_r_col: Optional[str],
              cfg: GpConfig, generation: int) -> Individual:
    wf = walk_forward_metrics(
        frame, spec,
        forward_return_col=forward_return_col, cost_r_col=cost_r_col,
        n_folds=cfg.n_folds,
        min_bars_per_fold=cfg.min_bars_per_fold,
        min_trades_per_fold=cfg.min_trades_per_fold,
    )
    return Individual(
        spec=spec, fitness=float(wf.fitness),
        n_trades_total=int(wf.n_trades_total),
        mean_oos_sharpe=float(wf.mean_oos_sharpe),
        consistency=float(wf.consistency),
        generation_born=int(generation),
    )


def _apply_novelty_penalty(individuals: List[Individual],
                            penalty: float) -> List[Individual]:
    """Penalize duplicate-signature individuals so the population
    doesn't collapse onto one rule. Each duplicate's fitness is
    reduced by ``penalty * (count - 1) / max_count``."""
    if penalty <= 0.0 or not individuals:
        return individuals
    counts: Dict[str, int] = {}
    for ind in individuals:
        sig = _spec_signature(ind.spec)
        counts[sig] = counts.get(sig, 0) + 1
    max_count = max(counts.values())
    if max_count <= 1:
        return individuals
    out: List[Individual] = []
    for ind in individuals:
        sig = _spec_signature(ind.spec)
        dup_factor = (counts[sig] - 1) / max(1, max_count - 1)
        adjusted = ind.fitness - penalty * dup_factor
        out.append(Individual(
            spec=ind.spec, fitness=float(adjusted),
            n_trades_total=ind.n_trades_total,
            mean_oos_sharpe=ind.mean_oos_sharpe,
            consistency=ind.consistency,
            generation_born=ind.generation_born,
        ))
    return out


def _update_hall_of_fame(hall: List[Individual],
                          population: Sequence[Individual],
                          size: int) -> List[Individual]:
    """Maintain top-``size`` individuals across all generations.

    Deduplicates by spec signature so two near-identical winners don't
    crowd out unique candidates."""
    combined = list(hall) + list(population)
    by_sig: Dict[str, Individual] = {}
    for ind in combined:
        sig = _spec_signature(ind.spec)
        existing = by_sig.get(sig)
        if existing is None or ind.fitness > existing.fitness:
            by_sig[sig] = ind
    ranked = sorted(by_sig.values(), key=lambda i: -i.fitness)
    return ranked[:size]


def evolve(frame: pd.DataFrame,
           *,
           forward_return_col: str,
           cost_r_col: Optional[str] = None,
           feature_columns: Optional[Sequence[str]] = None,
           cfg: Optional[GpConfig] = None,
           seed: int = 41,
           progress: Optional[Callable[[int, GenerationStats], None]] = None,
           ) -> GpReport:
    """Run a GP search for high-fitness rule hypotheses on ``frame``.

    Returns a ``GpReport`` whose ``hall_of_fame`` is the top
    ``cfg.hall_of_fame_size`` individuals across all generations. The
    operator's job: take that list to ``synthetic_validation`` (to
    catch overfit), then to paper trading.
    """
    cfg = cfg or GpConfig()
    rng = random.Random(int(seed))
    features = _usable_features(frame, feature_columns)
    if not features:
        return GpReport(config=cfg)
    thresholds = _feature_thresholds(frame, features, cfg.quantiles)

    seeds = _make_population(
        frame, n=cfg.population_size, seed=int(seed),
        features=features, thresholds=thresholds,
        max_conditions=cfg.max_conditions,
    )
    population: List[Individual] = []
    for spec in seeds:
        population.append(_evaluate(
            frame, spec,
            forward_return_col=forward_return_col,
            cost_r_col=cost_r_col, cfg=cfg, generation=0,
        ))
    population = _apply_novelty_penalty(population, cfg.novelty_penalty)
    hall = _update_hall_of_fame([], population, cfg.hall_of_fame_size)
    history: List[GenerationStats] = []
    history.append(_stats(population, generation=0))
    if progress:
        progress(0, history[-1])

    for gen in range(1, cfg.n_generations + 1):
        # Elites carry over unchanged.
        population_sorted = sorted(population, key=lambda i: -i.fitness)
        next_specs: List[HypothesisSpec] = [
            ind.spec for ind in population_sorted[:cfg.elite_count]
        ]
        while len(next_specs) < cfg.population_size:
            p1 = tournament_select(population, rng, k=cfg.tournament_k)
            if rng.random() < cfg.crossover_prob:
                p2 = tournament_select(population, rng, k=cfg.tournament_k)
                child_spec = crossover_specs(
                    p1.spec, p2.spec, rng, max_conditions=cfg.max_conditions,
                )
            else:
                child_spec = p1.spec
            child_spec = mutate_spec(
                child_spec, rng, features, thresholds,
                rates=cfg.mutation_rates, max_conditions=cfg.max_conditions,
            )
            # Rename so the GA tree stays traceable.
            child_spec = HypothesisSpec(
                name=f"gp_g{gen}_{len(next_specs):05d}",
                side=child_spec.side,
                conditions=child_spec.conditions,
                hold_bars=child_spec.hold_bars,
                stop_atr=child_spec.stop_atr,
                target_atr=child_spec.target_atr,
                metadata=child_spec.metadata,
            )
            next_specs.append(child_spec)

        population = [
            _evaluate(
                frame, spec,
                forward_return_col=forward_return_col,
                cost_r_col=cost_r_col, cfg=cfg, generation=gen,
            ) for spec in next_specs
        ]
        population = _apply_novelty_penalty(population, cfg.novelty_penalty)
        hall = _update_hall_of_fame(hall, population, cfg.hall_of_fame_size)
        history.append(_stats(population, generation=gen))
        if progress:
            progress(gen, history[-1])

    return GpReport(
        hall_of_fame=hall,
        final_population=sorted(population, key=lambda i: -i.fitness),
        history=history,
        config=cfg,
    )


def _stats(population: Sequence[Individual], *, generation: int) -> GenerationStats:
    fits = [ind.fitness for ind in population]
    sigs = {_spec_signature(ind.spec) for ind in population}
    diversity = len(sigs) / max(1, len(population))
    return GenerationStats(
        generation=int(generation),
        best_fitness=float(max(fits)) if fits else 0.0,
        median_fitness=float(np.median(fits)) if fits else 0.0,
        mean_fitness=float(np.mean(fits)) if fits else 0.0,
        diversity=float(diversity),
    )
