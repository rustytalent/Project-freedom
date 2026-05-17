"""Iterative search over (FactorWeights, DetectionParams) to maximise pool quality.

Two-phase loop:
  1. Random exploration over a wide search space.
  2. Local refinement around the top-K configs (gaussian perturbation in a shrinking radius).

Objective:   J = respect_rate * log(1 + tested_n) - lambda_complexity * weight_l2
Higher tested_n is good (we want many usable pools), but only if respect rate stays high.
The l2 penalty prevents one factor from dominating with absurd weights."""
from __future__ import annotations
from dataclasses import replace, asdict
from typing import Callable, Dict, List, Tuple
import copy
import math
import random
import numpy as np
import pandas as pd

from .config import Config, FactorWeights, DetectionParams
from .pools import build_pools, project_to_base
from .tester import test_pools, summarise


def _sample_weights(rng: random.Random) -> FactorWeights:
    return FactorWeights(
        eqhl=rng.uniform(0.5, 2.5),
        prev_day=rng.uniform(0.4, 2.0),
        prev_week=rng.uniform(0.5, 2.5),
        prev_month=rng.uniform(0.5, 3.0),
        fvg=rng.uniform(0.2, 1.8),
        order_block=rng.uniform(0.3, 2.0),
        in_candle_imbalance=rng.uniform(0.1, 1.5),
        volume_node=rng.uniform(0.2, 1.8),
        orb_extreme=rng.uniform(0.2, 1.5),
        multi_tf_overlap=rng.uniform(1.2, 2.5),
    )


def _sample_detect(base: DetectionParams, rng: random.Random) -> DetectionParams:
    return replace(
        base,
        swing_left=rng.choice([2, 3, 4, 5]),
        swing_right=rng.choice([2, 3, 4, 5]),
        eqhl_tol_atr=rng.uniform(0.05, 0.35),
        eqhl_min_touches=rng.choice([2, 2, 3]),
        fvg_min_atr=rng.uniform(0.10, 0.6),
        ob_displacement_atr=rng.uniform(1.0, 2.5),
        wick_dominance=rng.uniform(0.45, 0.7),
        body_max_ratio=rng.uniform(0.2, 0.45),
        merge_atr=rng.uniform(0.10, 0.35),
        pool_halfwidth_atr=rng.uniform(0.05, 0.20),
    )


def _perturb_weights(w: FactorWeights, sigma: float, rng: random.Random) -> FactorWeights:
    d = w.as_dict()
    for k in d:
        d[k] = max(0.05, d[k] * (1 + rng.gauss(0, sigma)))
    return FactorWeights(**d)


def _perturb_detect(p: DetectionParams, sigma: float, rng: random.Random) -> DetectionParams:
    def jitter(x, lo, hi):
        return float(min(hi, max(lo, x * (1 + rng.gauss(0, sigma)))))
    return replace(
        p,
        eqhl_tol_atr=jitter(p.eqhl_tol_atr, 0.03, 0.5),
        fvg_min_atr=jitter(p.fvg_min_atr, 0.05, 0.8),
        ob_displacement_atr=jitter(p.ob_displacement_atr, 0.8, 3.0),
        wick_dominance=jitter(p.wick_dominance, 0.4, 0.8),
        body_max_ratio=jitter(p.body_max_ratio, 0.15, 0.55),
        merge_atr=jitter(p.merge_atr, 0.05, 0.4),
        pool_halfwidth_atr=jitter(p.pool_halfwidth_atr, 0.03, 0.25),
    )


def _objective(stats: dict, weights: FactorWeights, lam: float = 0.02) -> float:
    respect = stats.get("respect_rate", 0.0)
    tested = stats.get("tested_n", 0)
    j = respect * math.log1p(tested)
    w2 = sum(v * v for v in weights.as_dict().values())
    return j - lam * w2 / 100.0


def optimize(tf_data: Dict[str, pd.DataFrame], cfg: Config,
             progress: Callable[[int, dict], None] | None = None,
             evaluator: Callable[[list, list], dict] | None = None,
             ) -> Tuple[Config, List[dict]]:
    """Returns (best_cfg, trial_log).

    The base TF dataframe is reused across all trials, so cost per trial is just pool build + test.

    `evaluator` lets the caller score the trial on a subset of pools (e.g. only those formed in a
    walk-forward training window). It receives the full (pools, results) from a trial and must
    return a `summarise`-style dict containing at least `respect_rate` and `tested_n`. When None,
    we use the standard `summarise(results)` over all pools.
    """
    rng = random.Random(cfg.opt_seed)
    np.random.seed(cfg.opt_seed)

    if evaluator is None:
        evaluator = lambda pools, results: summarise(results)

    trial_log: List[dict] = []
    best: Tuple[float, Config] = (-1e18, cfg)

    n_explore = max(1, int(cfg.opt_iterations * cfg.opt_explore_frac))

    for t in range(cfg.opt_iterations):
        if t < n_explore:
            trial = copy.deepcopy(cfg)
            trial.weights = _sample_weights(rng)
            trial.detect = _sample_detect(cfg.detect, rng)
        else:
            sigma = max(0.05, 0.4 * (1 - (t - n_explore) / max(1, cfg.opt_iterations - n_explore)))
            trial = copy.deepcopy(best[1])
            trial.weights = _perturb_weights(best[1].weights, sigma, rng)
            trial.detect = _perturb_detect(best[1].detect, sigma, rng)

        try:
            pools = build_pools(tf_data, trial)
            pools = project_to_base(pools, tf_data["base"].index)
            results = test_pools(tf_data["base"], pools, trial)
            stats = evaluator(pools, results)
        except Exception as e:
            stats = {"error": str(e), "respect_rate": 0.0, "tested_n": 0, "n": 0,
                     "break_rate": 0.0, "untouched_rate": 1.0}

        j = _objective(stats, trial.weights)
        rec = {
            "trial": t,
            "phase": "explore" if t < n_explore else "refine",
            "objective": j,
            **stats,
            "weights": trial.weights.as_dict(),
            "detect": asdict(trial.detect),
        }
        trial_log.append(rec)
        if j > best[0]:
            best = (j, copy.deepcopy(trial))
        if progress:
            progress(t, rec)

    return best[1], trial_log
