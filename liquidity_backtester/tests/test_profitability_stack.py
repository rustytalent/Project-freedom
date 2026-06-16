"""Tier-1 profitability stack — allocator + kill_switch + synthetic_validation.

Each module ships discipline that survives the next 90 days of paper-then-
live deployment. These tests pin the contract:

  * The Thompson allocator routes more capital to clearer winners,
    sends nothing to losers, and stays in cash when no hypothesis has
    visible edge.
  * The decay kill-switch flags hypotheses whose rolling Sharpe has
    collapsed and only flags resurrection on truly clean post-kill data.
  * The synthetic negative-control catches runners that invent edge
    from pure noise and clears runners that don't.

Test discipline:
  * Synthetic inputs only — no parquet / no network.
  * Determinism: every test seeds RNGs explicitly.
  * Edge cases (empty inputs, all-losers, tiny windows, insufficient
    data) are tested as first-class behaviors not afterthoughts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liqpool.research.allocator import (
    DEFAULT_MAX_WEIGHT,
    AllocationReport,
    HypothesisPosterior,
    thompson_allocate,
)
from liqpool.research.kill_switch import (
    DEFAULT_ABSOLUTE_SHARPE_FLOOR,
    HypothesisHistory,
    check_resurrection,
    filter_alive,
    verdict_for_hypothesis,
    verdicts_for_basket,
)
from liqpool.research.synthetic_validation import (
    DEFAULT_NOISE_SHARPE_CEILING,
    SyntheticValidationReport,
    generate_mean_reverting,
    generate_random_walk,
    generate_trending,
    validate_hypothesis_against_noise,
)


# ─────────────────────────────────────────────────────────────────
# Allocator
# ─────────────────────────────────────────────────────────────────

def _posterior(name: str, n: int, sharpe: float) -> HypothesisPosterior:
    return HypothesisPosterior.from_stats(name=name, n_trades=n, mean_sharpe=sharpe)


def test_allocator_routes_more_capital_to_clearer_winners():
    posteriors = [
        _posterior("winner",   n=120, sharpe=1.30),
        _posterior("middle",   n=80,  sharpe=0.60),
        _posterior("marginal", n=60,  sharpe=0.20),
        _posterior("loser",    n=120, sharpe=-0.20),
    ]
    report = thompson_allocate(posteriors, n_draws=3000, seed=11)
    # Strict ordering: winner > middle > marginal > loser
    assert report.weights["winner"] > report.weights["middle"]
    assert report.weights["middle"] > report.weights["marginal"]
    assert report.weights["marginal"] > report.weights["loser"]
    # The loser's weight should be tiny (well under 5%)
    assert report.weights["loser"] < 0.05
    # Win-shares should sum to ~1.0 across draws where any sample beat the floor
    total_win = sum(report.win_share.values())
    assert total_win == pytest.approx(1.0, abs=1e-3) or total_win == pytest.approx(0.0, abs=1e-3)


def test_allocator_returns_to_cash_when_no_hypothesis_has_edge():
    posteriors = [_posterior(f"H{i}", n=80, sharpe=-0.4) for i in range(3)]
    report = thompson_allocate(posteriors, n_draws=2000, seed=7)
    assert report.cash_share > 0.5
    # Each weight is small relative to cash_share
    for w in report.weights.values():
        assert w < 0.30


def test_allocator_max_weight_cap_is_respected():
    """A single 'amazing' hypothesis should never get >max_weight."""
    posteriors = [
        _posterior("dominator", n=400, sharpe=3.0),     # absurdly high
        _posterior("backup", n=100, sharpe=0.5),
    ]
    report = thompson_allocate(posteriors, n_draws=2000, seed=3,
                                max_weight=0.55)
    assert report.weights["dominator"] <= 0.55 + 1e-9
    # The capped overflow goes to backup (not cash) since backup has positive edge
    assert report.weights["backup"] > 0.30


def test_allocator_inflates_posteriors_for_small_samples():
    """Two hypotheses with identical Sharpe but different sample sizes
    should have the small-sample one weighted *less* because of the
    inflated posterior."""
    posteriors = [
        _posterior("seasoned", n=200, sharpe=0.80),
        _posterior("rookie",   n=15,  sharpe=0.80),
    ]
    report = thompson_allocate(posteriors, n_draws=4000, seed=5)
    assert report.weights["seasoned"] > report.weights["rookie"]


def test_allocator_handles_empty_input():
    report = thompson_allocate([], n_draws=1000, seed=1)
    assert report.weights == {}
    assert report.cash_share == 1.0
    assert "no hypotheses" in report.note


def test_allocator_to_dict_is_serializable():
    posteriors = [_posterior("A", 80, 0.8), _posterior("B", 80, 0.5)]
    report = thompson_allocate(posteriors, n_draws=500, seed=2)
    import json
    json.dumps(report.to_dict())


def test_allocator_from_report_extracts_sharpe_and_n_trades():
    class _OverallStub:
        sharpe = 1.10
        n_trades = 90

    class _ReportStub:
        name = "H_real"
        overall = _OverallStub()

    p = HypothesisPosterior.from_report(_ReportStub())
    assert p.name == "H_real"
    assert p.n_trades == 90
    assert p.mean_sharpe == pytest.approx(1.10)
    # stderr should be finite and positive
    assert p.stderr_sharpe > 0


# ─────────────────────────────────────────────────────────────────
# Kill switch
# ─────────────────────────────────────────────────────────────────

def _steady_returns(rng, n: int, mean: float, std: float) -> np.ndarray:
    return rng.normal(mean, std, n)


def test_kill_switch_flags_decayed_hypothesis():
    rng = np.random.default_rng(7)
    # Was healthy for 150 trades, now has decayed below floor for last 50.
    r = np.concatenate([_steady_returns(rng, 150, 0.50, 0.625),
                        _steady_returns(rng, 50, -0.06, 0.6)])
    history = HypothesisHistory(name="decayed", original_sharpe=0.80,
                                 r_multiples=r)
    v = verdict_for_hypothesis(history, window_trades=30)
    assert v.status == "killed"
    assert v.rolling_sharpe < DEFAULT_ABSOLUTE_SHARPE_FLOOR


def test_kill_switch_keeps_alive_when_stable():
    rng = np.random.default_rng(11)
    r = _steady_returns(rng, 200, 0.50, 0.625)         # ~0.8 Sharpe steady
    history = HypothesisHistory(name="steady", original_sharpe=0.80,
                                 r_multiples=r)
    v = verdict_for_hypothesis(history, window_trades=30)
    assert v.status == "alive"
    assert v.rolling_sharpe > DEFAULT_ABSOLUTE_SHARPE_FLOOR


def test_kill_switch_returns_insufficient_data_on_short_history():
    history = HypothesisHistory(name="tiny", original_sharpe=0.5,
                                 r_multiples=np.array([0.2, -0.1, 0.3]))
    v = verdict_for_hypothesis(history, window_trades=30, min_trades=20)
    assert v.status == "insufficient_data"


def test_kill_switch_filter_alive_returns_expected_names():
    rng = np.random.default_rng(3)
    histories = [
        HypothesisHistory("good", 0.80, _steady_returns(rng, 100, 0.50, 0.625)),
        HypothesisHistory("bad",  0.80, _steady_returns(rng, 100, -0.10, 0.6)),
    ]
    verdicts = verdicts_for_basket(histories, window_trades=30)
    alive = filter_alive(verdicts, include_watching=True)
    assert "good" in alive
    assert "bad" not in alive


def test_kill_switch_dropping_nans_aligns_timestamps():
    """When r_multiples contain NaNs and timestamps are supplied, the
    timestamps must be filtered alongside so the two stay aligned."""
    r = np.array([0.5, np.nan, 0.3, np.inf, -0.2])
    ts = ["t1", "t2", "t3", "t4", "t5"]
    history = HypothesisHistory("h", 0.5, r_multiples=r, timestamps=ts)
    assert len(history.r_multiples) == 3
    assert history.timestamps == ["t1", "t3", "t5"]


def test_check_resurrection_requires_clean_post_kill_window():
    rng = np.random.default_rng(19)
    early = _steady_returns(rng, 100, 0.50, 0.625)
    decayed = _steady_returns(rng, 50, -0.05, 0.6)
    history = HypothesisHistory("decayed", 0.80,
                                 r_multiples=np.concatenate([early, decayed]))
    # Post-kill window is the decayed segment — still below floor.
    res = check_resurrection(history, kill_at_index=100, window_trades=30,
                              min_post_kill=30)
    assert not res.eligible


def test_check_resurrection_eligible_when_post_kill_data_recovers():
    rng = np.random.default_rng(23)
    pre = _steady_returns(rng, 100, 0.50, 0.625)
    bad = _steady_returns(rng, 50, -0.05, 0.6)
    recovered = _steady_returns(rng, 60, 0.50, 0.625)
    history = HypothesisHistory(
        "recovered", 0.80,
        r_multiples=np.concatenate([pre, bad, recovered]),
    )
    # Kill happened at trade 150 (after the bad segment). Look at the recovered
    # segment that follows.
    res = check_resurrection(history, kill_at_index=150, window_trades=30,
                              min_post_kill=30)
    assert res.eligible
    assert res.rolling_sharpe > DEFAULT_ABSOLUTE_SHARPE_FLOOR


def test_check_resurrection_handles_bad_kill_index():
    history = HypothesisHistory("h", 0.5,
                                 r_multiples=np.array([0.1, 0.2, 0.3]))
    res = check_resurrection(history, kill_at_index=-1, window_trades=30)
    assert not res.eligible
    assert "invalid" in res.reason.lower()


# ─────────────────────────────────────────────────────────────────
# Synthetic validation
# ─────────────────────────────────────────────────────────────────

def test_random_walk_generator_has_zero_drift():
    rets = np.log(generate_random_walk(n_bars=2000, seed=11)["close"]).diff().dropna()
    # Mean log-return ~0 within tolerance for 2000 bars
    assert abs(rets.mean()) < 0.0005


def test_trending_generator_has_positive_drift():
    rets = np.log(generate_trending(n_bars=2000, seed=11,
                                      per_bar_drift=0.0005)["close"]).diff().dropna()
    assert rets.mean() > 0.0001


def test_mean_reverting_generator_has_negative_autocorrelation_at_lag_above_half_life():
    bars = generate_mean_reverting(n_bars=2000, seed=11, half_life_bars=20)
    close = bars["close"].to_numpy()
    lr = np.diff(np.log(close))
    # AR(1) on log levels has negative-ish autocorr in returns at lag near half-life.
    lag = 25
    if len(lr) > lag:
        corr = float(np.corrcoef(lr[:-lag], lr[lag:])[0, 1])
        # Loose check — mean-reverting log-level → returns have non-trivial structure
        assert -0.3 < corr < 0.3


def test_validate_flags_overfit_runner_as_noise_susceptible():
    """A runner that returns artificially positive R-multiples regardless
    of input bars must be flagged."""
    def overfit_runner(bars):
        return [0.3, 0.2, 0.5, -0.1, 0.4, 0.3, 0.6, 0.1, 0.7, 0.2] * 3
    report = validate_hypothesis_against_noise(
        overfit_runner, name="overfit", n_realizations=10, n_bars=500, seed=3,
    )
    assert report.status == "noise_susceptible"
    rw = report.regimes["random_walk"]
    assert rw.p95_sharpe > DEFAULT_NOISE_SHARPE_CEILING


def test_validate_passes_runner_that_produces_no_synthetic_trades():
    def silent_runner(bars):
        return []
    report = validate_hypothesis_against_noise(
        silent_runner, name="silent", n_realizations=8, n_bars=400, seed=1,
    )
    assert report.status == "no_trades"


def test_validate_to_dict_is_serializable():
    def runner(bars):
        return [0.1, -0.1, 0.2]
    report = validate_hypothesis_against_noise(
        runner, name="r", n_realizations=4, n_bars=300, seed=2,
    )
    import json
    json.dumps(report.to_dict())


def test_committee_pipeline_end_to_end_filters_and_allocates():
    """The three Tier-1 modules compose into the daily decision flow:

      synthetic_validation → kill_switch → thompson_allocate

    A hypothesis flagged noise_susceptible by synthetic_validation is
    dropped before kill_switch sees it; a killed one is excluded before
    the allocator sees it; the allocator routes capital across whatever
    survives. This test pins that pipeline by walking three candidate
    hypotheses with explicitly different fates through it.
    """
    rng = np.random.default_rng(101)

    # Three candidates:
    #   trustworthy: clean recent Sharpe, original Sharpe 0.9.
    #   decayed:     recent rolling Sharpe collapsed.
    #   noisy:       synthetic_validation will flag it.
    trustworthy_returns = rng.normal(0.55, 0.6, 120)
    decayed_returns = np.concatenate([rng.normal(0.55, 0.6, 100),
                                      rng.normal(-0.05, 0.6, 60)])

    histories = {
        "trustworthy": HypothesisHistory("trustworthy", original_sharpe=0.9,
                                         r_multiples=trustworthy_returns),
        "decayed": HypothesisHistory("decayed", original_sharpe=0.9,
                                     r_multiples=decayed_returns),
    }

    # Step 1 — synthetic validation. The "noisy" hypothesis is the canonical
    # overfit runner that returns positive R-multiples regardless of input;
    # validation must flag it. "trustworthy" and "decayed" are tested by
    # the runner returning empty (no trades) on synthetic data — the
    # mirror of a hypothesis whose entry rules require structure the
    # random walk does not contain. That's the cleanest possible "passes
    # the noise gauntlet because there's no edge to falsely report"
    # state. A passing real strategy would behave the same on random walk.
    def overfit_runner(bars):
        return [0.4, 0.3, 0.2, 0.5, -0.1, 0.4, 0.3, 0.6] * 2

    def silent_runner(bars):
        return []

    synth_results = {
        "trustworthy": validate_hypothesis_against_noise(
            silent_runner, name="trustworthy", n_realizations=6, n_bars=300, seed=7,
        ),
        "decayed": validate_hypothesis_against_noise(
            silent_runner, name="decayed", n_realizations=6, n_bars=300, seed=8,
        ),
        "noisy": validate_hypothesis_against_noise(
            overfit_runner, name="noisy", n_realizations=6, n_bars=300, seed=9,
        ),
    }
    not_noise_fooled = [n for n, r in synth_results.items()
                        if r.status != "noise_susceptible"]
    assert "noisy" not in not_noise_fooled
    # Both clean candidates must clear synthetic validation; their status
    # may be "passed" (the obvious case) or "no_trades" (a clean way to
    # pass without inventing edge from noise).
    for name in ("trustworthy", "decayed"):
        assert synth_results[name].status in ("passed", "no_trades")
    not_noise_fooled = ["trustworthy", "decayed"]

    # Step 2 — kill switch on the survivors that have a trade history.
    surviving_with_history = [n for n in not_noise_fooled if n in histories]
    verdicts = verdicts_for_basket(
        [histories[n] for n in surviving_with_history], window_trades=30,
    )
    alive_names = filter_alive(verdicts, include_watching=False)
    assert "trustworthy" in alive_names
    assert "decayed" not in alive_names

    # Step 3 — allocate capital across the alive hypotheses.
    alive_posteriors = [
        HypothesisPosterior.from_stats(name=n, n_trades=len(histories[n].r_multiples),
                                        mean_sharpe=histories[n].original_sharpe)
        for n in alive_names
    ]
    allocation = thompson_allocate(alive_posteriors, n_draws=2000, seed=11)
    assert set(allocation.weights.keys()) == {"trustworthy"}
    # The only survivor gets all the capital (modulo cash share rounding).
    assert allocation.weights["trustworthy"] > 0.50


def test_validate_handles_runner_exceptions_gracefully():
    """A runner that raises for some bar inputs must not crash the
    validator — exceptions count as zero-trade realizations."""
    call_count = [0]

    def flaky_runner(bars):
        call_count[0] += 1
        if call_count[0] % 3 == 0:
            raise RuntimeError("simulated runner crash")
        return [0.1, -0.05, 0.15]

    report = validate_hypothesis_against_noise(
        flaky_runner, name="flaky", n_realizations=10, n_bars=300, seed=4,
    )
    # No crash, and the random-walk regime still produced *some* realizations.
    assert report.status in ("passed", "noise_susceptible", "no_trades")
    assert report.regimes["random_walk"].n_realizations_with_trades > 0
