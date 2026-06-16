"""End-to-end research committee — chain together the Tier 1 modules.

Usage (from ``liquidity_backtester/`` root):

    PYTHONPATH=. python examples/run_research_committee.py

The script does NOT run the harness on real data — its purpose is to
show how the three Tier 1 modules compose into one production decision
flow:

    synthetic_validation  →  kill_switch  →  thompson_allocate

Each module is intentionally narrow:

  * ``validate_hypothesis_against_noise`` says "could the in-sample
    Sharpe be a noise artifact?"
  * ``verdict_for_hypothesis`` says "has this strategy decayed enough
    to stop trading it?"
  * ``thompson_allocate`` says "of the survivors, how should I split
    the bankroll this week?"

When you have real ``HypothesisReport`` objects from the harness, swap
the synthetic histories below for ``HypothesisHistory.from_report(...)``
and ``HypothesisPosterior.from_report(...)``. Everything else is the
same — the discipline is the same whether the inputs are mock or live.
"""
from __future__ import annotations

import numpy as np

from liqpool.research import (
    HypothesisHistory,
    HypothesisPosterior,
    filter_alive,
    thompson_allocate,
    validate_hypothesis_against_noise,
    verdicts_for_basket,
)


def _overfit_runner(bars):
    """A runner that fakes positive R irrespective of input — the
    canonical noise-susceptible failure mode."""
    return [0.3, 0.2, 0.4, -0.1, 0.5, 0.2, 0.4, 0.3] * 3


def _silent_runner(bars):
    """A runner that returns no trades on synthetic data — what a real
    strategy with structural entry rules looks like when the bars don't
    contain the structure it needs. Passes the noise check cleanly."""
    return []


def main() -> None:
    rng = np.random.default_rng(101)

    # ── Stage 0: per-hypothesis live trade histories (synthetic for demo) ──
    candidates = {
        "h1_thursday_theta_scalp": {
            "history": HypothesisHistory(
                "h1_thursday_theta_scalp", original_sharpe=1.20,
                r_multiples=rng.normal(0.70, 0.60, 140),
            ),
            "noise_runner": _silent_runner,
        },
        "h2_opening_range_fade": {
            "history": HypothesisHistory(
                "h2_opening_range_fade", original_sharpe=0.80,
                r_multiples=np.concatenate([
                    rng.normal(0.50, 0.65, 100),
                    rng.normal(-0.05, 0.60, 50),
                ]),
            ),
            "noise_runner": _silent_runner,
        },
        "h5_buy_and_hold_baseline": {
            "history": HypothesisHistory(
                "h5_buy_and_hold_baseline", original_sharpe=0.40,
                r_multiples=rng.normal(0.25, 0.62, 120),
            ),
            "noise_runner": _silent_runner,
        },
        "mined_42_overfit_candidate": {
            "history": HypothesisHistory(
                "mined_42_overfit_candidate", original_sharpe=1.80,
                r_multiples=rng.normal(1.10, 0.61, 60),
            ),
            "noise_runner": _overfit_runner,
        },
    }

    print("\n[stage 1] Synthetic noise check — does the hypothesis invent edge?")
    survivors_after_noise = []
    for name, c in candidates.items():
        report = validate_hypothesis_against_noise(
            c["noise_runner"], name=name,
            n_realizations=10, n_bars=400, seed=7,
        )
        kept = report.status != "noise_susceptible"
        marker = "✓ kept" if kept else "✗ DROP (noise)"
        print(f"  {marker:<14} {name:<32} status={report.status}")
        if kept:
            survivors_after_noise.append(name)

    print("\n[stage 2] Decay kill switch — has the live edge collapsed?")
    histories = [candidates[n]["history"] for n in survivors_after_noise]
    verdicts = verdicts_for_basket(histories, window_trades=30)
    for name, v in verdicts.items():
        status_marker = {"alive": "✓ alive", "watching": "△ watching",
                         "killed": "✗ killed",
                         "insufficient_data": "? sparse"}[v.status]
        print(f"  {status_marker:<14} {name:<32} rolling={v.rolling_sharpe:+.2f}  "
              f"ratio={v.ratio:+.2f}")
    alive_names = filter_alive(verdicts, include_watching=False)
    if not alive_names:
        print("\nNo alive hypotheses — entire bankroll stays in cash this week.")
        return

    print("\n[stage 3] Thompson allocation across surviving hypotheses")
    posteriors = [
        HypothesisPosterior.from_stats(
            name=n,
            n_trades=int(candidates[n]["history"].r_multiples.size),
            mean_sharpe=float(candidates[n]["history"].original_sharpe),
        ) for n in alive_names
    ]
    report = thompson_allocate(posteriors, n_draws=4000, seed=11)
    for name in sorted(report.weights, key=lambda n: -report.weights[n]):
        w = report.weights[name]
        ws = report.win_share[name]
        print(f"  weight={w:>5.1%}   win_share={ws:>5.1%}   {name}")
    if report.cash_share > 0.01:
        print(f"  cash  ={report.cash_share:>5.1%}")
    if report.note != "ok":
        print(f"  note: {report.note}")


if __name__ == "__main__":
    main()
