"""End-to-end regime-conditional discovery on synthetic data.

Two of the most useful disciplines from the playbook (#2 walk-forward
fitness, #4 regime-conditional mining) compose into one short workflow:

    1. Build a feature frame tagged with a ``regime`` column.
    2. Run ``mine_per_regime`` — sample candidate rules within each
       regime's own quantile structure, score them on that regime's
       forward returns. Surfaces conditional edges directly.
    3. Pick the top candidate per regime and verify it with
       ``walk_forward_metrics`` — does it survive an honest expanding-
       window OOS test?
    4. Use ``regime_specialization_score`` to confirm the survivor is
       genuinely a specialist (carries edge in one regime, not all).

Usage (from ``liquidity_backtester/`` root):

    PYTHONPATH=. python examples/run_regime_discovery.py

The script generates its own synthetic frame so you can reproduce the
findings without a parquet warehouse. The truth embedded in the data:
in regime A, ``x > 0.5`` predicts positive forward returns. Regime B is
pure noise. The miner should find the A-specialist rule and reject
anything from regime B.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from liqpool.research import (
    GpConfig,
    evaluate_across_regimes,
    evolve,
    mine_per_regime,
    rank_walk_forward_hypotheses,
    regime_specialization_score,
    walk_forward_metrics,
)
from liqpool.research.hypothesis_miner import Condition, HypothesisSpec


def _build_frame(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    z = rng.normal(0, 1, n)
    regime = np.where(np.arange(n) % 2 == 0, "A", "B")
    fwd_r = np.where(
        regime == "A",
        np.where(x > 0.5, 0.8, -0.2) + rng.normal(0, 0.4, n),
        rng.normal(0, 0.5, n),
    )
    return pd.DataFrame({"x": x, "y": y, "z": z,
                          "regime": regime, "fwd_r": fwd_r})


def _spec_from_json(spec_json: str) -> HypothesisSpec:
    d = json.loads(spec_json)
    return HypothesisSpec(
        name=d["name"], side=d["side"],
        conditions=[Condition(feature=c["feature"], op=c["op"],
                              threshold=float(c["threshold"]))
                    for c in d["conditions"]],
        hold_bars=int(d["hold_bars"]),
        stop_atr=float(d["stop_atr"]),
        target_atr=float(d["target_atr"]),
        metadata=dict(d.get("metadata", {})),
    )


def main() -> None:
    frame = _build_frame(n=3000, seed=7)
    print(f"[setup]  n_rows={len(frame)}  regimes={sorted(frame['regime'].unique())}")

    print("\n[stage 1]  Per-regime mining — sample candidate rules within each "
          "regime's own quantiles")
    mined = mine_per_regime(
        frame, regime_col="regime", forward_return_col="fwd_r",
        n_per_regime=400, seed=11, min_rows=300, min_trades=20,
        feature_columns=["x", "y", "z"],
    )
    print("Top 3 mined hypotheses in each regime (by score):")
    for reg, g in mined.groupby("regime"):
        print(f"  --- regime={reg} ({len(g)} candidates) ---")
        for _, row in g.head(3).iterrows():
            print(f"    score={row['score']:+.3f}  mean_R={row['mean_R']:+.2f}  "
                  f"sharpe={row['sharpe']:+.2f}  trades={int(row['trades'])}  "
                  f"name={row['name']}")

    # Pick the top candidate from each regime.
    top_per_regime = {
        reg: g.iloc[0] for reg, g in mined.groupby("regime")
    }

    print("\n[stage 2]  Walk-forward fitness for each regime's top candidate")
    survivors = []
    for reg, row in top_per_regime.items():
        spec = _spec_from_json(row["spec_json"])
        wf = walk_forward_metrics(
            frame, spec, forward_return_col="fwd_r",
            n_folds=5, min_bars_per_fold=300, min_trades_per_fold=5,
        )
        marker = "✓" if wf.fitness > 0.20 else "✗"
        print(f"  {marker}  regime={reg}  {spec.name}")
        print(f"      mean OOS Sharpe={wf.mean_oos_sharpe:+.2f}  "
              f"consistency={wf.consistency:.2f}  "
              f"fitness={wf.fitness:+.2f}  "
              f"per-fold={[round(f.sharpe, 2) for f in wf.folds]}")
        if wf.fitness > 0.20:
            survivors.append((reg, spec, wf))

    if not survivors:
        print("\nNo walk-forward survivors. Mining surfaced candidates but none "
              "had consistent OOS edge — exactly the discipline we want.")
        return

    print(f"\n[stage 3]  Regime specialization for the {len(survivors)} survivor(s)")
    for reg, spec, wf in survivors:
        perfs = evaluate_across_regimes(
            frame, spec, regime_col="regime",
            forward_return_col="fwd_r",
            min_rows=300, min_trades_per_regime=10,
        )
        spec_score = regime_specialization_score(perfs)
        verdict = ("specialist" if spec_score > 0.50 else
                   "always-on" if spec_score < 0.15 else
                   "mixed")
        print(f"  {spec.name}  (mined from regime={reg})")
        print(f"      specialization={spec_score:.2f}  ({verdict})")
        for p in perfs:
            print(f"        regime={p.regime}  n_trades={p.n_trades}  "
                  f"sharpe={p.sharpe:+.2f}  mean_R={p.mean_r:+.2f}")
        if spec_score > 0.50:
            print(f"      → trade-time gate: route this rule only when "
                  f"regime classifier reports regime={reg}")

    print("\n[stage 4]  Walk-forward leaderboard across 200 randomly sampled rules")
    leaderboard = rank_walk_forward_hypotheses(
        frame, forward_return_col="fwd_r", n=200, seed=23,
        feature_columns=["x", "y", "z"],
        n_folds=4, min_bars_per_fold=300, min_trades_per_fold=5,
    )
    print(f"Top 5 by fitness:")
    for _, row in leaderboard.head(5).iterrows():
        print(f"  fitness={row['fitness']:+.2f}  "
              f"OOS Sharpe={row['mean_oos_sharpe']:+.2f}  "
              f"consistency={row['consistency']:.2f}  "
              f"trades={int(row['n_trades_total'])}  "
              f"side={row['side']}  name={row['name']}")

    print("\n[stage 5]  GP evolution — refine the survivors via mutation + crossover")

    def progress(gen, stats):
        print(f"  gen {gen:>2}  best={stats.best_fitness:+.2f}  "
              f"median={stats.median_fitness:+.2f}  "
              f"diversity={stats.diversity:.2f}")

    gp_cfg = GpConfig(population_size=40, n_generations=6, tournament_k=3,
                       elite_count=3, max_conditions=2, n_folds=4,
                       min_bars_per_fold=300, hall_of_fame_size=5)
    gp_report = evolve(
        frame, forward_return_col="fwd_r",
        feature_columns=["x", "y", "z"],
        cfg=gp_cfg, seed=53, progress=progress,
    )
    print("\nGP hall of fame (top 5):")
    for ind in gp_report.hall_of_fame:
        conds = " AND ".join(
            f"{c.feature}{c.op}{c.threshold:.2f}" for c in ind.spec.conditions
        )
        print(f"  fitness={ind.fitness:+.2f}  OOS Sharpe={ind.mean_oos_sharpe:+.2f}  "
              f"consistency={ind.consistency:.2f}  trades={ind.n_trades_total}  "
              f"gen_born={ind.generation_born}  side={ind.spec.side}  rule=[{conds}]")


if __name__ == "__main__":
    main()
