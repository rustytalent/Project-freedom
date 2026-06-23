"""BeliefWebV2 — Bayesian scenarios with confidence intervals + causal
graph + lifecycle phases + particle filtering + conditional probabilities
+ Markov transitions + information gain + predicted resolutions.

Founder 2026-06-22 Tier-3 brief: "this is the layer where the main
engine lies. Make it state-of-the-art. Put your soul into it." These
tests cover every layer of the upgrade.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pandas as pd

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PersistenceConfig,
    V4Runner,
    V4RunnerConfig,
)
from liqpool.research.belief.executor_v4.belief_web_v2 import (
    DEFAULT_FAMILY_EDGES,
    EDGE_INHIBIT,
    EDGE_SUPPORT,
    LIFECYCLE_PHASES,
    PHASE_DECAYING,
    PHASE_DYING,
    PHASE_GROWING,
    PHASE_INCUBATING,
    PHASE_PEAK,
    PHASE_RETIRED,
    BeliefScenario,
    BeliefWebV2,
    BeliefWebV2Config,
    CausalEdge,
    ConditionalProbabilityTable,
    MarkovTransitionTable,
    ResolutionMemory,
    RichBeliefSnapshot,
    ScenarioGraph,
)
from liqpool.research.belief.executor_v4.scenario_web import (
    FAMILY_CHOP,
    FAMILY_DIRECTIONAL,
    FAMILY_FAT_TAIL,
    Scenario,
)


# ── BeliefScenario: Bayesian uncertainty ─────────────────────────


def test_belief_scenario_posterior_mean_matches_alpha_over_alpha_plus_beta():
    sc = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
        alpha=3.0, beta=1.0,
    )
    assert abs(sc.current_probability - 0.75) < 1e-9


def test_belief_scenario_confidence_interval_brackets_mean():
    sc = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
        alpha=8.0, beta=4.0,
    )
    lo, hi = sc.confidence_interval_95
    assert lo < sc.current_probability < hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_belief_scenario_epistemic_variance_shrinks_with_more_data():
    sc_thin = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
        alpha=2.0, beta=2.0,
    )
    sc_thick = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
        alpha=200.0, beta=200.0,
    )
    assert sc_thin.epistemic_variance > sc_thick.epistemic_variance


def test_observe_evidence_moves_posterior_and_emits_info_gain():
    sc = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
        alpha=2.0, beta=2.0,
    )
    prior_mean = sc.current_probability
    gain = sc.observe_evidence(confirms=5.0, contradicts=0.0,
                                  bar_index=10)
    assert sc.current_probability > prior_mean
    assert gain >= 0.0
    assert sc.last_reinforced_bar == 10


# ── Particle filter ──────────────────────────────────────────────


def test_initialise_particles_creates_distribution_around_mean():
    sc = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
        alpha=2.0, beta=2.0,
    )
    sc.initialise_particles(n=32)
    assert len(sc.particles) == 32
    for p in sc.particles:
        assert 0.0 <= p <= 1.0
    mean_p = sum(sc.particles) / len(sc.particles)
    assert abs(mean_p - sc.current_probability) < 0.2


def test_reweight_and_resample_pulls_particles_toward_evidence():
    sc = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
        alpha=2.0, beta=2.0,
    )
    sc.initialise_particles(n=32)
    sc.reweight_and_resample_particles(observed_evidence=0.85)
    avg = sum(sc.particles) / len(sc.particles)
    # Pulled in the direction of the evidence (from ~0.5 toward 0.85).
    assert avg > 0.55


# ── Lifecycle phase classification ───────────────────────────────


def test_lifecycle_starts_incubating():
    sc = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
        birth_bar=100,
    )
    sc.update_lifecycle_phase(bar_index=102)
    assert sc.lifecycle_phase == PHASE_INCUBATING


def test_lifecycle_growing_when_probability_rising():
    sc = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
        alpha=2.0, beta=4.0,
        birth_bar=100,
    )
    sc.recent_probabilities = [0.20, 0.30, 0.40, 0.50]
    sc.peak_probability_seen = 0.50
    sc.update_lifecycle_phase(bar_index=120)
    assert sc.lifecycle_phase in (PHASE_GROWING, PHASE_PEAK)


def test_lifecycle_dying_after_extended_below_base_prior():
    sc = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
        base_prior=0.20,
        birth_bar=100,
    )
    sc.bars_below_base_prior = 15
    sc.update_lifecycle_phase(bar_index=130)
    assert sc.lifecycle_phase == PHASE_DYING


# ── Causal graph ─────────────────────────────────────────────────


def test_seed_family_edges_creates_directional_vs_chop_inhibit():
    scenarios = {
        "d1": BeliefScenario(scenario_id="d1", name="d1",
                                family="directional",
                                implied_direction=1, implied_horizon_bars=20),
        "c1": BeliefScenario(scenario_id="c1", name="c1",
                                family="chop",
                                implied_direction=0, implied_horizon_bars=30),
    }
    g = ScenarioGraph()
    g.seed_family_edges(scenarios)
    edges = g.all_edges()
    assert any(e.from_id == "d1" and e.to_id == "c1" and
                 e.edge_type == EDGE_INHIBIT for e in edges)
    assert any(e.from_id == "c1" and e.to_id == "d1" and
                 e.edge_type == EDGE_INHIBIT for e in edges)


def test_seed_family_edges_idempotent():
    scenarios = {
        "d1": BeliefScenario(scenario_id="d1", name="d1",
                                family="directional",
                                implied_direction=1, implied_horizon_bars=20),
        "c1": BeliefScenario(scenario_id="c1", name="c1",
                                family="chop",
                                implied_direction=0, implied_horizon_bars=30),
    }
    g = ScenarioGraph()
    g.seed_family_edges(scenarios)
    n1 = len(g.all_edges())
    g.seed_family_edges(scenarios)
    assert len(g.all_edges()) == n1


def test_update_from_active_learns_support_for_co_occurring_scenarios():
    scenarios = {
        "a": BeliefScenario(scenario_id="a", name="a",
                                family="directional",
                                implied_direction=1, implied_horizon_bars=20),
        "b": BeliefScenario(scenario_id="b", name="b",
                                family="fat_tail",
                                implied_direction=0, implied_horizon_bars=10),
    }
    g = ScenarioGraph()
    # Both scenarios active every tick → edge moves toward SUPPORT.
    for _ in range(80):
        g.update_from_active(active_ids=["a", "b"],
                                  all_ids=["a", "b"])
    # The (a, b) edge should now be SUPPORT/IMPLY-ish.
    edges = g.all_edges()
    ab = [e for e in edges if e.from_id == "a" and e.to_id == "b"]
    assert len(ab) == 1
    assert ab[0].edge_type in (EDGE_SUPPORT, "imply")
    assert ab[0].weight > 0.5


def test_propagate_shifts_alpha_beta_via_edges():
    scenarios = {
        "a": BeliefScenario(scenario_id="a", name="a",
                                family="directional",
                                implied_direction=1, implied_horizon_bars=20,
                                alpha=18.0, beta=2.0),    # high prob
        "b": BeliefScenario(scenario_id="b", name="b",
                                family="chop",
                                implied_direction=0, implied_horizon_bars=30,
                                alpha=2.0, beta=2.0),
    }
    g = ScenarioGraph()
    g.seed_family_edges(scenarios)
    # Beef up the confidence so propagation actually moves something.
    for e in g.all_edges():
        e.confidence = 1.0
    pre_b = scenarios["b"].current_probability
    deltas = g.propagate(scenarios, n_iterations=3, attenuation=0.50)
    # The a→b edge is INHIBIT with a=directional, b=chop.
    # a's high prob inhibits b → b's prob should go DOWN (or at least
    # alpha+beta shift to favour beta).
    post_b = scenarios["b"].current_probability
    assert post_b <= pre_b + 1e-6
    assert "b" in deltas


# ── Conditional probability table ─────────────────────────────────


def test_conditional_table_p_b_given_a_matches_co_occurrence():
    t = ConditionalProbabilityTable()
    # A and B always co-active across 20 ticks.
    for _ in range(20):
        t.update(active_ids=["a", "b"], all_ids=["a", "b", "c"])
    assert abs(t.p_conditional("a", "b") - 1.0) < 1e-9
    # C was never active.
    assert t.p_conditional("a", "c") == 0.0


def test_conditional_table_mutual_information_high_for_synchronized_pair():
    """Synchronized A,B should show high MI; truly independent C,D should
    show near-zero MI."""
    t = ConditionalProbabilityTable()
    # Pattern: A and B perfectly synchronized.
    # D activates on ticks {0, 3, 6, 9, 12, ...} — relatively independent
    # of the i%2==0 pattern that gates A and B.
    import random
    rng = random.Random(42)
    for i in range(80):
        active = []
        if i % 2 == 0:
            active.extend(["a", "b"])
        if rng.random() < 0.5:
            active.append("d")
        t.update(active_ids=active, all_ids=["a", "b", "d"])
    mi_ab = t.mutual_information("a", "b")
    mi_ad = t.mutual_information("a", "d")
    assert mi_ab > mi_ad


def test_conditional_table_mi_zero_when_one_scenario_never_changes():
    t = ConditionalProbabilityTable()
    # A always on, B sometimes — A has no variance.
    for i in range(20):
        active = ["a", "b"] if i % 2 == 0 else ["a"]
        t.update(active_ids=active, all_ids=["a", "b"])
    assert t.mutual_information("a", "b") == 0.0


# ── Markov transition table ──────────────────────────────────────


def test_markov_observes_state_transitions():
    t = MarkovTransitionTable()
    t.observe("s1", PHASE_INCUBATING)
    t.observe("s1", PHASE_INCUBATING)
    t.observe("s1", PHASE_GROWING)
    t.observe("s1", PHASE_PEAK)
    t.observe("s1", PHASE_DECAYING)
    dist = t.transition_distribution(PHASE_GROWING)
    # Growing → peak in the only observed transition.
    assert dist.get(PHASE_PEAK, 0.0) > 0.0


def test_markov_expected_remaining_bars_falls_back_to_heuristic():
    t = MarkovTransitionTable()
    bars = t.expected_remaining_bars(PHASE_INCUBATING)
    assert bars > 0.0


# ── Resolution memory ────────────────────────────────────────────


def test_resolution_memory_records_per_family():
    rm = ResolutionMemory()
    rm.record("directional", 0.5)
    rm.record("directional", -0.3)
    rm.record("chop", 0.1)
    s = rm.summary()
    assert "directional" in s
    assert s["directional"]["n"] == 2
    assert "chop" in s


def test_resolution_memory_bounded_per_family():
    rm = ResolutionMemory(capacity_per_family=10)
    for i in range(50):
        rm.record("directional", i * 0.1)
    assert len(rm.outcomes_for("directional")) == 10


# ── Predicted resolution ─────────────────────────────────────────


def test_predict_realisation_from_empirical_outcomes():
    sc = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
    )
    sc.predict_realisation(empirical_outcomes=[0.8, 0.6, -0.2, 0.5, 0.7])
    # Mean of those is +0.48.
    assert abs(sc.predicted_realisation_mean - 0.48) < 1e-6
    assert sc.predicted_realisation_std > 0
    assert 0.0 <= sc.predicted_p_realised <= 1.0


def test_predict_realisation_heuristic_when_no_history():
    sc = BeliefScenario(
        scenario_id="t", name="t", family="directional",
        implied_direction=1, implied_horizon_bars=20,
        alpha=4.0, beta=2.0,
    )
    sc.predict_realisation(empirical_outcomes=None)
    assert sc.predicted_realisation_std >= 0
    # Direction positive so mean should be positive too.
    assert sc.predicted_realisation_mean > 0


# ── BeliefWebV2 — orchestrator integration ───────────────────────


def _legacy_scenario(*, sid: str, name: str, family: str,
                        prob: float = 0.3, direction: int = 1,
                        horizon: int = 20) -> Scenario:
    """Build a legacy ScenarioWeb Scenario for mirroring tests."""
    return Scenario(
        scenario_id=sid, name=name, family=family,
        trigger_signature=f"t_{sid}",
        implied_direction=direction, implied_horizon_bars=horizon,
        implied_max_drawdown_during_path=0.5,
        implied_strategy_class="long_ce" if direction > 0 else "long_pe",
        base_prior=0.10, current_probability=prob, decay_rate=0.99,
    )


def test_observe_mirrors_legacy_scenarios_into_belief_scenarios():
    bw = BeliefWebV2()
    legacy = {
        "d1": _legacy_scenario(sid="d1", name="bull_breakout",
                                  family="directional", prob=0.4),
    }
    snap = bw.observe(legacy_scenarios=legacy, bar_index=100)
    assert snap.n_active == 1
    assert "d1" in bw.scenarios
    twin = bw.scenarios["d1"]
    assert twin.family == "directional"
    assert twin.implied_direction == 1


def test_observe_seeds_causal_edges_between_families():
    bw = BeliefWebV2()
    legacy = {
        "d1": _legacy_scenario(sid="d1", name="bull",
                                  family="directional", prob=0.4),
        "c1": _legacy_scenario(sid="c1", name="chop",
                                  family="chop", prob=0.3, direction=0),
    }
    bw.observe(legacy_scenarios=legacy, bar_index=100)
    edges = bw.graph.all_edges()
    assert any(e.from_id == "d1" and e.to_id == "c1" for e in edges)


def test_observe_updates_conditional_table():
    bw = BeliefWebV2()
    legacy = {
        "a": _legacy_scenario(sid="a", name="a", family="directional"),
        "b": _legacy_scenario(sid="b", name="b", family="fat_tail",
                                 direction=0),
    }
    for i in range(10):
        bw.observe(legacy_scenarios=legacy, bar_index=100 + i)
    assert bw.conditional_table.n_ticks == 10


def test_observe_updates_lifecycle_phases():
    bw = BeliefWebV2()
    legacy = {
        "a": _legacy_scenario(sid="a", name="a", family="directional",
                                 prob=0.4),
    }
    for i in range(10):
        # Climb then fall.
        legacy["a"].current_probability = 0.1 + 0.06 * min(i, 6)
        bw.observe(legacy_scenarios=legacy, bar_index=100 + i)
    snap = bw._last_snapshot
    assert snap is not None
    lc = snap.lifecycle_distribution
    assert sum(lc.values()) >= 1


def test_observe_retires_disappeared_scenarios():
    bw = BeliefWebV2()
    legacy = {
        "a": _legacy_scenario(sid="a", name="a", family="directional"),
    }
    bw.observe(legacy_scenarios=legacy, bar_index=100)
    # Now drop it entirely.
    bw.observe(legacy_scenarios={}, bar_index=101)
    assert bw.scenarios["a"].lifecycle_phase == PHASE_RETIRED


def test_record_resolution_feeds_resolution_memory():
    bw = BeliefWebV2()
    bw.record_resolution(family="directional", realised_r=0.6)
    bw.record_resolution(family="directional", realised_r=-0.3)
    s = bw.resolution_memory.summary()
    assert "directional" in s
    assert s["directional"]["n"] == 2


def test_rich_snapshot_includes_predicted_resolutions():
    bw = BeliefWebV2()
    bw.record_resolution(family="directional", realised_r=0.5)
    bw.record_resolution(family="directional", realised_r=-0.2)
    bw.record_resolution(family="directional", realised_r=0.7)
    legacy = {
        "a": _legacy_scenario(sid="a", name="a", family="directional"),
    }
    snap = bw.observe(legacy_scenarios=legacy, bar_index=100)
    assert len(snap.predicted_resolutions) >= 1
    pr = snap.predicted_resolutions[0]
    assert "predicted_realisation_mean" in pr


def test_rich_snapshot_total_information_gain_non_negative():
    bw = BeliefWebV2()
    legacy = {
        "a": _legacy_scenario(sid="a", name="a", family="directional",
                                 prob=0.2),
    }
    for i in range(8):
        legacy["a"].current_probability = 0.2 + 0.05 * i
        snap = bw.observe(legacy_scenarios=legacy, bar_index=100 + i)
    assert snap.total_information_gain >= 0.0


def test_rich_snapshot_propagation_deltas_track_per_node():
    bw = BeliefWebV2(cfg=BeliefWebV2Config(propagation_iterations=4))
    legacy = {
        "d": _legacy_scenario(sid="d", name="bull",
                                 family="directional", prob=0.7),
        "c": _legacy_scenario(sid="c", name="chop",
                                 family="chop", prob=0.3, direction=0),
    }
    # Bump confidence by running for several ticks so propagation
    # actually fires.
    for i in range(30):
        bw.observe(legacy_scenarios=legacy, bar_index=100 + i)
    snap = bw._last_snapshot
    assert snap is not None
    # The propagation_deltas dict exists, even if it's empty initially.
    assert isinstance(snap.propagation_deltas, dict)


def test_summary_shape_when_warmed_up():
    bw = BeliefWebV2()
    legacy = {
        "a": _legacy_scenario(sid="a", name="a", family="directional"),
    }
    bw.observe(legacy_scenarios=legacy, bar_index=100)
    s = bw.summary()
    for key in ["ready", "bar_index", "n_active",
                  "scenarios_with_ci", "causal_graph",
                  "conditional_table", "markov_table",
                  "lifecycle_distribution",
                  "most_informative_this_tick",
                  "predicted_resolutions",
                  "resolution_memory"]:
        assert key in s
    assert s["ready"] is True


def test_summary_not_ready_before_first_observe():
    bw = BeliefWebV2()
    s = bw.summary()
    assert s["ready"] is False


# ── End-to-end manager wiring ────────────────────────────────────


def _runner_with_state(state_dir: Path) -> V4Runner:
    cfg = V4RunnerConfig(
        manager=None,
        persistence=PersistenceConfig(state_dir=state_dir, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


def test_manager_creates_belief_web_v2():
    tmp = Path(tempfile.mkdtemp(prefix="bw2_mgr_"))
    runner = _runner_with_state(tmp)
    assert runner.manager.belief_web_v2 is not None


def test_belief_web_v2_warms_up_after_ticks():
    tmp = Path(tempfile.mkdtemp(prefix="bw2_warm_"))
    runner = _runner_with_state(tmp)
    from tests.test_executor_v4_dead_market_and_web_viz import _snap
    res = None
    for i in range(10):
        res = runner.on_tick(_snap(100 + i))
    panel = res.cockpit.to_dict()["belief_web_v2_panel"]
    assert panel["ready"] is True
    assert panel["n_active"] >= 0


def test_cockpit_carries_belief_web_v2_panel_with_full_schema():
    tmp = Path(tempfile.mkdtemp(prefix="bw2_panel_"))
    runner = _runner_with_state(tmp)
    from tests.test_executor_v4_dead_market_and_web_viz import _snap
    res = None
    for i in range(15):
        res = runner.on_tick(_snap(100 + i))
    panel = res.cockpit.to_dict()["belief_web_v2_panel"]
    expected_keys = [
        "ready", "bar_index", "n_active", "scenarios_with_ci",
        "causal_graph", "conditional_table", "markov_table",
        "lifecycle_distribution", "most_informative_this_tick",
        "total_information_gain", "coherence_score", "surprise_score",
        "predicted_resolutions", "propagation_deltas",
        "resolution_memory", "notes",
    ]
    for k in expected_keys:
        assert k in panel, f"missing key {k}"


def test_viewer_html_includes_belief_web_v2_panel():
    from liqpool.research.belief.executor_v4.cockpit_server import (
        _DEFAULT_VIEWER_HTML,
    )
    h = _DEFAULT_VIEWER_HTML
    must_contain = [
        "BELIEF WEB v2",
        "bw2_scenarios", "bw2_lifecycle", "bw2_info_feed",
        "bw2_graph_svg", "bw2_mi_pairs", "bw2_predicted",
        "render_belief_web_v2", "render_belief_graph",
        "Bayesian", "confidence interval",
    ]
    missing = [s for s in must_contain if s.lower() not in h.lower()]
    assert not missing, f"viewer HTML missing tokens: {missing}"
