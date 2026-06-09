"""Stream J — statistical honesty tests.

Pins for the three independent honesty layers:

  * Holm-Bonferroni step-down — strictly more powerful than plain
    Bonferroni at the same family-wise error rate.
  * Probabilistic Sharpe Ratio — Bailey & LdP 2012.
  * Deflated Sharpe Ratio — Bailey & LdP 2014, deflates by the
    search budget.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from liqpool.arsenal.honesty import (
    deflated_sharpe,
    expected_max_sharpe,
    holm_bonferroni,
    probabilistic_sharpe,
)


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------

def test_holm_marks_only_smallest_p_significant_when_just_one_passes():
    """6 tests, alpha=0.05. The threshold for the smallest p is
    0.05/6 = 0.00833. Test 1 (p=0.001) passes; test 2 (p=0.02) does not
    even at the next threshold 0.05/5 = 0.01 — so it fails. All later
    tests fail by the step-down rule."""
    results = holm_bonferroni([
        ("a", 0.001),
        ("b", 0.02),
        ("c", 0.03),
        ("d", 0.04),
        ("e", 0.045),
        ("f", 0.049),
    ], alpha=0.05)
    sig = {r.name: r.significant for r in results}
    assert sig["a"] is True
    for name in ("b", "c", "d", "e", "f"):
        assert sig[name] is False


def test_holm_is_strictly_more_powerful_than_plain_bonferroni():
    """Plain Bonferroni threshold = 0.05/3 = 0.01667. Holm gives the
    smallest p the same bar but the next p 0.05/2 = 0.025. So a
    p=0.02 second test passes Holm but FAILS plain Bonferroni."""
    results = holm_bonferroni([
        ("a", 0.001),
        ("b", 0.02),
        ("c", 0.04),
    ], alpha=0.05)
    by_name = {r.name: r for r in results}
    # Step-down rule: a passes; b is checked at 0.05/2=0.025 -> passes
    # because 0.02 <= 0.025. c is then checked at 0.05/1=0.05 -> passes
    # because 0.04 <= 0.05.
    assert by_name["a"].significant
    assert by_name["b"].significant
    assert by_name["c"].significant


def test_holm_returns_in_input_order():
    results = holm_bonferroni([
        ("zeta", 0.001),
        ("alpha", 0.04),
        ("beta", 0.02),
    ], alpha=0.05)
    # Order matches input, not sorted by p.
    assert [r.name for r in results] == ["zeta", "alpha", "beta"]


def test_holm_handles_empty_input():
    assert holm_bonferroni([]) == []


# ---------------------------------------------------------------------------
# Probabilistic Sharpe
# ---------------------------------------------------------------------------

def test_psr_high_when_sharpe_clearly_positive():
    """A strategy with high per-trade Sharpe over many trades should
    produce PSR close to 1.0."""
    rng = np.random.default_rng(0)
    # Mean 0.10, std 0.20 -> per-trade Sharpe 0.5; n=400 trades.
    r = rng.normal(0.10, 0.20, 400)
    psr = probabilistic_sharpe(r)
    assert psr > 0.99


def test_psr_uniform_under_null_not_strongly_significant():
    """Under H0 (true mean = 0) the PSR is uniformly distributed on
    (0, 1) over samples — any single realisation can be anywhere.
    The contract is: we should NOT be claiming strong evidence of
    edge. So we pin "not in the rejection region" instead of pinning
    a tight central band."""
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.2, 400)
    psr = probabilistic_sharpe(r)
    assert psr < 0.95  # not "this is a clear winner"
    # And by symmetry, very-low PSR is also fine — that's a clear loser.


def test_psr_returns_nan_on_too_few_observations():
    assert math.isnan(probabilistic_sharpe(np.array([0.1, 0.2])))


def test_psr_returns_nan_on_zero_variance():
    # constant returns -> std=0 -> nan
    assert math.isnan(probabilistic_sharpe(np.full(50, 0.05)))


# ---------------------------------------------------------------------------
# Deflated Sharpe + expected max
# ---------------------------------------------------------------------------

def test_expected_max_sharpe_grows_with_n_trials():
    """Bailey 2014: E[max trials Sharpe under null] grows with N
    (gentle, sub-linear). Pin the monotonicity."""
    sd = 0.5
    e1 = expected_max_sharpe(sd, n_trials=1)
    e10 = expected_max_sharpe(sd, n_trials=10)
    e100 = expected_max_sharpe(sd, n_trials=100)
    assert e1 < e10 < e100


def test_dsr_lower_than_psr_when_search_was_wide():
    """Same strategy, but if we searched 100 alphas to find it, the
    DSR is materially lower than the PSR — that's the whole point of
    deflation."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.05, 0.20, 300)  # decent Sharpe
    psr = probabilistic_sharpe(r)
    # Pretend we tried 100 alphas with a real spread of Sharpes.
    trial_sharpes = list(rng.normal(0.0, 0.3, 100))
    dsr = deflated_sharpe(r, trial_sharpes)
    assert dsr < psr


def test_dsr_falls_back_to_psr_when_only_one_trial():
    rng = np.random.default_rng(3)
    r = rng.normal(0.05, 0.20, 300)
    psr = probabilistic_sharpe(r)
    dsr = deflated_sharpe(r, [0.25])  # only one trial sharpe
    assert math.isclose(dsr, psr, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Integration into per_alpha_summary
# ---------------------------------------------------------------------------

def test_per_alpha_summary_emits_psr_and_dsr_columns():
    import pandas as pd

    from liqpool.arsenal.evaluator import ArsenalEvaluator

    rng = np.random.default_rng(4)
    # Two alphas with clearly different distributions.
    trades_a = pd.DataFrame({
        "alpha_name": ["good"] * 200,
        "net_r": rng.normal(0.10, 0.20, 200),
        "net_pnl": rng.normal(50.0, 100.0, 200),
        "cost_inr": np.full(200, 15.0),
    })
    trades_b = pd.DataFrame({
        "alpha_name": ["weak"] * 200,
        "net_r": rng.normal(0.0, 0.20, 200),
        "net_pnl": rng.normal(0.0, 100.0, 200),
        "cost_inr": np.full(200, 15.0),
    })
    trades = pd.concat([trades_a, trades_b], ignore_index=True)
    summary = ArsenalEvaluator.per_alpha_summary(trades)
    assert {"sharpe", "psr", "dsr"}.issubset(set(summary.columns))
    # The "good" alpha should have a higher PSR than the "weak" one.
    good_psr = float(summary.loc[summary["alpha_name"] == "good", "psr"].iloc[0])
    weak_psr = float(summary.loc[summary["alpha_name"] == "weak", "psr"].iloc[0])
    assert good_psr > weak_psr
    # DSR <= PSR for each row by construction (deflation by trials).
    for _, row in summary.iterrows():
        if math.isfinite(row["psr"]) and math.isfinite(row["dsr"]):
            assert row["dsr"] <= row["psr"] + 1e-9
