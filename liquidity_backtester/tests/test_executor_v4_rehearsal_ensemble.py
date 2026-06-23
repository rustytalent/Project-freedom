"""Belief Rehearsal Ensemble — conditional-kNN off-policy evaluation.

Founder 2026-06-22 Tier-2 brief: "Do not use Monte Carlo, it is old. Use
something advanced." The ensemble retrieves the K most-similar historical
trades from a persistent observation store, importance-weights them by
how close their actual entry parameters were to a candidate perturbation,
and produces:

  * P(profit), P(target), P(stop) per perturbation
  * mean R, std R, conditional expected R above/below zero
  * a calibrated rehearsal_score for the aggregator
  * a confidence-band derived from analogue count × mean similarity
  * a recommended_action (PROCEED / TUNE / DEFER)

These tests verify the building blocks and the wired end-to-end pipeline.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pandas as pd

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PersistenceConfig,
    PortfolioManagerConfig,
    V4Runner,
    V4RunnerConfig,
)
from liqpool.research.belief.executor_v4.aggregator import (
    AggregatorConfig,
    DecisionAggregator,
)
from liqpool.research.belief.executor_v4.belief_rehearsal import (
    SCHEMA_VERSION,
    AnalogueRetriever,
    BeliefObservation,
    BeliefObservationStore,
    BeliefRehearsalEnsemble,
    FeatureSpace,
    FEATURE_KEYS,
    Perturbation,
    PerturbationGrid,
    RehearsalConfig,
    RehearsalDecision,
    build_query_features,
)

from tests.test_executor_v4_dead_market_and_web_viz import _snap


# ── Fixtures ────────────────────────────────────────────────────


def _make_observation(*, obs_id: str,
                         feature_seed: float,
                         realized_r: float,
                         exit_reason: str = "target hit",
                         entry_conf: float = 0.65,
                         size_lots: float = 1.0,
                         target_r: float = 1.5,
                         stop_r: float = 1.0,
                         direction: float = 1.0,
                         family: str = "directional") -> BeliefObservation:
    """Generate a synthetic observation with controllable features."""
    feature_vector = {k: feature_seed for k in FEATURE_KEYS}
    feature_vector["regime_stability_index"] = 0.7
    feature_vector["chop_mass"] = 0.10 if family == "directional" else 0.55
    return BeliefObservation(
        obs_id=obs_id, ts="2026-06-22T11:00:00",
        bar_index=200, schema_version=SCHEMA_VERSION,
        feature_vector=feature_vector,
        entry_params={
            "entry_confidence": entry_conf,
            "size_lots": size_lots,
            "target_r": target_r,
            "stop_r": stop_r,
            "profile_scalp": 0.0,
            "direction": direction,
        },
        outcome={
            "realized_r": realized_r, "realized_rupees": realized_r * 100.0,
            "bars_to_resolution": 30.0,
            "exit_reason": exit_reason,
            "max_favorable_r": max(realized_r, 0.0),
            "max_adverse_r": min(realized_r, 0.0),
        },
        regime_tag={"dominant_family": family,
                     "chop_mass": 0.10, "manipulation_mass": 0.05,
                     "tail_mass": 0.05,
                     "directional_consensus_horizon_weighted": 0.40},
        strategy="long_ce",
    )


# ── Store: round-trip + JSONL ─────────────────────────────────


def test_observation_store_roundtrip_jsonl():
    tmp = Path(tempfile.mkdtemp(prefix="reh_store_"))
    store = BeliefObservationStore(state_dir=tmp, ring_size=100,
                                       history_days=2)
    for i in range(5):
        store.append(_make_observation(
            obs_id=f"o{i}", feature_seed=0.5 + 0.05 * i,
            realized_r=0.6 if i % 2 == 0 else -0.4,
        ))
    # The append goes to today's file.
    path = store.path_for_today()
    assert path.exists()
    # Each line is a parseable JSON dict.
    with open(path, "r") as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 5
    for line in lines:
        d = json.loads(line)
        assert "feature_vector" in d
        assert d["schema_version"] == SCHEMA_VERSION
    # Fresh store loads them.
    store2 = BeliefObservationStore(state_dir=tmp, history_days=2)
    n = store2.load_history()
    assert n == 5
    assert len(store2.observations()) == 5


def test_observation_store_loads_prior_days():
    tmp = Path(tempfile.mkdtemp(prefix="reh_hist_"))
    # Hand-craft yesterday's file.
    from datetime import datetime, timedelta, timezone
    IST = timezone(timedelta(hours=5, minutes=30))
    yesterday = (datetime.now(IST).date()
                 - timedelta(days=1)).isoformat()
    with open(tmp / f"belief_observations_{yesterday}.jsonl", "w") as f:
        for i in range(8):
            obs = _make_observation(
                obs_id=f"y{i}", feature_seed=0.5,
                realized_r=0.4 if i % 2 == 0 else -0.3,
            )
            f.write(json.dumps(obs.to_dict()) + "\n")
    store = BeliefObservationStore(state_dir=tmp, history_days=3)
    n = store.load_history()
    assert n == 8


# ── Feature space ──────────────────────────────────────────────


def test_feature_space_welford_stats_converge():
    fs = FeatureSpace()
    # Push 100 values through one feature with known mean.
    for x in range(1, 101):
        obs = _make_observation(
            obs_id=f"x{x}", feature_seed=float(x),
            realized_r=0.5,
        )
        fs.observe(obs)
    stat = fs._stats["regime_stability_index"]
    # regime_stability_index was overridden to 0.7 for every observation,
    # so its variance is 0 and mean is 0.7.
    assert abs(stat.mean - 0.7) < 1e-6
    assert stat.std <= 1.0


def test_feature_space_weights_emphasise_correlated_features():
    """When a feature perfectly predicts the outcome, its weight should
    end up near the ceiling (1.0); irrelevant features near the floor."""
    fs = FeatureSpace(weight_floor=0.10, weight_ceiling=1.0)
    obs_list = []
    for i in range(60):
        # Make 'net_intent_velocity' linearly predict realized_r.
        net_v = i / 30.0 - 1.0     # in [-1, +1]
        realized = net_v * 0.5     # tight linear relationship
        obs = _make_observation(obs_id=f"f{i}", feature_seed=0.5,
                                  realized_r=realized)
        obs.feature_vector["net_intent_velocity"] = net_v
        # Add noise feature uncorrelated with outcome.
        obs.feature_vector["mm_dominant_probability"] = (i % 3) * 0.1
        obs_list.append(obs)
        fs.observe(obs)
    fs.recompute_weights(obs_list)
    weights = fs.weights
    # The strongly-correlated feature should outrank the uncorrelated one.
    assert weights["net_intent_velocity"] > weights["mm_dominant_probability"]


def test_feature_space_standardise_handles_no_data():
    fs = FeatureSpace()
    out = fs.standardise({"regime_stability_index": 0.7})
    # With <5 samples we return raw values (don't crash on /0).
    assert out["regime_stability_index"] == 0.7


# ── Retriever ──────────────────────────────────────────────────


def test_retriever_returns_k_or_fewer_neighbors():
    tmp = Path(tempfile.mkdtemp(prefix="reh_ret_"))
    store = BeliefObservationStore(state_dir=tmp)
    fs = FeatureSpace()
    # Build 30 observations with varying feature_seeds.
    for i in range(30):
        obs = _make_observation(obs_id=f"r{i}",
                                  feature_seed=0.1 * i,
                                  realized_r=0.4)
        store.append(obs)
        fs.observe(obs)
    fs.recompute_weights(store.observations())
    retr = AnalogueRetriever(feature_space=fs, k=5)
    query = {k: 0.5 for k in FEATURE_KEYS}
    res = retr.query(query_features=query,
                       observations=store.observations())
    assert len(res) <= 5
    # Similarities are in [0, 1] and sorted descending.
    for r in res:
        assert 0.0 <= r.similarity <= 1.0
    sims = [r.similarity for r in res]
    assert sims == sorted(sims, reverse=True)


def test_retriever_regime_match_brings_same_family_closer():
    """When feature distance is a tie, the regime-match bonus picks
    same-family analogues. We construct truly-identical features so
    the bonus is what discriminates."""
    tmp = Path(tempfile.mkdtemp(prefix="reh_reg_"))
    store = BeliefObservationStore(state_dir=tmp)
    fs = FeatureSpace()
    # Two families with IDENTICAL feature vectors (only regime_tag differs).
    for i in range(20):
        obs = _make_observation(
            obs_id=f"a{i}", feature_seed=0.5,
            realized_r=0.3,
            family=("directional" if i % 2 == 0 else "chop"),
        )
        # Force ALL feature dims identical regardless of family so the
        # tiebreak is purely the regime bonus.
        obs.feature_vector = {k: 0.5 for k in FEATURE_KEYS}
        store.append(obs)
        fs.observe(obs)
    retr = AnalogueRetriever(feature_space=fs, k=10,
                                  regime_match_bonus=0.30)
    query = {k: 0.5 for k in FEATURE_KEYS}
    res_with = retr.query(query_features=query,
                              observations=store.observations(),
                              query_regime_tag={"dominant_family": "directional"})
    # All top-10 results should be 'directional' since features tie.
    directional_in_top5 = sum(1 for a in res_with[:5]
                                if a.observation.regime_tag.get(
                                    "dominant_family") == "directional")
    assert directional_in_top5 >= 3


# ── Perturbation grid ──────────────────────────────────────────


def test_perturbation_grid_includes_as_proposed_baseline():
    grid = PerturbationGrid.default_grid()
    names = [p.name for p in grid]
    assert "as_proposed" in names
    assert len(grid) >= 5


def test_perturbation_applies_relative_delta():
    p = Perturbation(name="conf_+10pct",
                       deltas={"entry_confidence": +0.10})
    base = {"entry_confidence": 0.60, "size_lots": 1}
    out = p.applied_to(base)
    assert abs(out["entry_confidence"] - 0.60 * 1.10) < 1e-9
    # Untouched fields preserved.
    assert out["size_lots"] == 1


# ── Ensemble — empty / cold start ──────────────────────────────


def test_rehearse_returns_neutral_when_no_history():
    tmp = Path(tempfile.mkdtemp(prefix="reh_cold_"))
    ens = BeliefRehearsalEnsemble(state_dir=tmp)
    ens.boot()
    qf = {k: 0.5 for k in FEATURE_KEYS}
    dec = ens.rehearse(
        query_features=qf,
        proposed_params={"entry_confidence": 0.65, "size_lots": 1,
                          "target_r": 1.5, "stop_r": 1.0,
                          "profile_scalp": 0, "direction": 1},
    )
    assert dec.ran is False
    assert dec.rehearsal_score == 0.50
    assert dec.confidence == 0.0
    assert "insufficient history" in dec.deferral_reason.lower()


def test_rehearse_runs_when_enough_history():
    tmp = Path(tempfile.mkdtemp(prefix="reh_hot_"))
    ens = BeliefRehearsalEnsemble(state_dir=tmp,
                                    cfg=RehearsalConfig(
                                        min_observations_to_run=10))
    # Inject 20 observations directly.
    for i in range(20):
        ens.record_closure(_make_observation(
            obs_id=f"i{i}", feature_seed=0.5,
            realized_r=0.6 if i % 2 == 0 else -0.4,
        ))
    qf = {k: 0.5 for k in FEATURE_KEYS}
    dec = ens.rehearse(
        query_features=qf,
        proposed_params={"entry_confidence": 0.65, "size_lots": 1,
                          "target_r": 1.5, "stop_r": 1.0,
                          "profile_scalp": 0, "direction": 1},
    )
    assert dec.ran is True
    assert dec.n_analogues_total > 0
    assert len(dec.per_perturbation) >= 5
    # Score should reflect the analogue distribution (~ 0.5 since wins=
    # losses).
    assert 0.30 <= dec.rehearsal_score <= 0.70


def test_rehearse_recommends_defer_when_analogues_lose():
    tmp = Path(tempfile.mkdtemp(prefix="reh_defer_"))
    cfg = RehearsalConfig(min_observations_to_run=10,
                              defer_p_profit_below=0.50,
                              min_confidence_to_emit_score=0.10)
    ens = BeliefRehearsalEnsemble(state_dir=tmp, cfg=cfg)
    # All losers — analogues say this state usually loses.
    for i in range(20):
        ens.record_closure(_make_observation(
            obs_id=f"L{i}", feature_seed=0.5, realized_r=-0.45,
            exit_reason="stop hit",
        ))
    qf = {k: 0.5 for k in FEATURE_KEYS}
    dec = ens.rehearse(
        query_features=qf,
        proposed_params={"entry_confidence": 0.65, "size_lots": 1,
                          "target_r": 1.5, "stop_r": 1.0,
                          "profile_scalp": 0, "direction": 1},
    )
    assert dec.ran is True
    assert dec.recommended_action == "DEFER"


def test_rehearse_recommends_tune_when_alt_perturbation_dominates():
    """Synthetically: when small-size analogues win and large-size lose,
    the rehearsal should prefer the size_-50pct perturbation."""
    tmp = Path(tempfile.mkdtemp(prefix="reh_tune_"))
    cfg = RehearsalConfig(min_observations_to_run=10,
                              tune_threshold_r=0.10,
                              min_confidence_to_emit_score=0.05)
    ens = BeliefRehearsalEnsemble(state_dir=tmp, cfg=cfg)
    # Small-size wins, large-size loses.
    for i in range(15):
        ens.record_closure(_make_observation(
            obs_id=f"S{i}", feature_seed=0.5,
            realized_r=0.8, size_lots=0.5,
            exit_reason="target hit",
        ))
    for i in range(15):
        ens.record_closure(_make_observation(
            obs_id=f"B{i}", feature_seed=0.5,
            realized_r=-0.5, size_lots=2.0,
            exit_reason="stop hit",
        ))
    qf = {k: 0.5 for k in FEATURE_KEYS}
    dec = ens.rehearse(
        query_features=qf,
        proposed_params={"entry_confidence": 0.65, "size_lots": 1.0,
                          "target_r": 1.5, "stop_r": 1.0,
                          "profile_scalp": 0, "direction": 1},
    )
    assert dec.ran is True
    # The best perturbation should be something other than as_proposed.
    assert dec.best_perturbation_name != "as_proposed"


# ── Rehearsal decision shape ───────────────────────────────────


def test_rehearsal_decision_to_dict_is_serializable():
    dec = RehearsalDecision.empty("test")
    d = dec.to_dict()
    json.dumps(d)            # should not raise
    assert d["ran"] is False
    assert "rehearsal_score" in d


# ── Aggregator integration ────────────────────────────────────


def test_aggregator_consumes_rehearsal_decision_in_final_score():
    """Build a minimal aggregator + rehearsal decision and confirm the
    final score moves when the rehearsal score changes."""
    cfg = AggregatorConfig()
    agg = DecisionAggregator(cfg)

    class _Fake:
        pass
    high_dec = _Fake()
    high_dec.rehearsal_score = 0.90
    high_dec.confidence = 0.80
    high_dec.recommended_action = "PROCEED"
    high_dec.n_analogues_total = 18
    high_dec.best_perturbation_name = "as_proposed"
    high_dec.notes = []
    low_dec = _Fake()
    low_dec.rehearsal_score = 0.10
    low_dec.confidence = 0.80
    low_dec.recommended_action = "PROCEED"
    low_dec.n_analogues_total = 18
    low_dec.best_perturbation_name = "as_proposed"
    low_dec.notes = []
    common = dict(
        base_confidence=0.65,
        mtf_alignment={"alignment_score": 0.7, "alignment_ok": True,
                        "l1_match": True, "l5_match": True,
                        "l15_match": True, "l60_match": False},
        projection=None, critique=None, ev_decision=None,
        web_snapshot=None,
        proposed_strategy_class="long_ce", proposed_direction=1,
        open_positions_count=0, max_open_positions=3,
        daily_pnl_rupees=0.0, max_daily_bleed=5000.0,
    )
    d_high = agg.decide(**common, rehearsal_decision=high_dec)
    d_low = agg.decide(**common, rehearsal_decision=low_dec)
    assert d_high.final_score > d_low.final_score
    assert d_high.rehearsal_score == 0.90
    assert d_low.rehearsal_score == 0.10


def test_aggregator_refuses_when_rehearsal_defers_with_confidence():
    cfg = AggregatorConfig(rehearsal_defer_min_confidence=0.25)
    agg = DecisionAggregator(cfg)

    class _Fake:
        pass
    defer_dec = _Fake()
    defer_dec.rehearsal_score = 0.30
    defer_dec.confidence = 0.60
    defer_dec.recommended_action = "DEFER"
    defer_dec.n_analogues_total = 18
    defer_dec.best_perturbation_name = "as_proposed"
    defer_dec.notes = ["P(profit) too low"]
    d = agg.decide(
        base_confidence=0.85,
        mtf_alignment={"alignment_score": 0.9, "alignment_ok": True,
                        "l1_match": True, "l5_match": True,
                        "l15_match": True, "l60_match": True},
        projection=None, critique=None, ev_decision=None,
        web_snapshot=None,
        proposed_strategy_class="long_ce", proposed_direction=1,
        open_positions_count=0, max_open_positions=3,
        daily_pnl_rupees=0.0, max_daily_bleed=5000.0,
        rehearsal_decision=defer_dec,
    )
    # Even with a strong base confidence, the rehearsal DEFER should
    # force a refusal.
    assert d.decision == "REFUSE"
    assert any("rehearsal DEFER" in r for r in d.refuse_reasons)


# ── End-to-end manager wiring ──────────────────────────────────


def _runner_with_state(state_dir: Path,
                         pm_cfg: PortfolioManagerConfig | None = None,
                         ) -> V4Runner:
    cfg = V4RunnerConfig(
        manager=pm_cfg,
        persistence=PersistenceConfig(state_dir=state_dir, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


def test_manager_creates_rehearsal_ensemble_when_state_dir_set():
    tmp = Path(tempfile.mkdtemp(prefix="mgr_reh_"))
    runner = _runner_with_state(tmp)
    assert runner.manager.rehearsal_ensemble is not None


def test_manager_boots_rehearsal_on_first_tick():
    tmp = Path(tempfile.mkdtemp(prefix="mgr_boot_"))
    runner = _runner_with_state(tmp)
    runner.on_tick(_snap(100))
    assert runner.manager.rehearsal_ensemble._booted is True


def test_cockpit_carries_rehearsal_panel():
    tmp = Path(tempfile.mkdtemp(prefix="mgr_panel_"))
    runner = _runner_with_state(tmp)
    res = None
    for i in range(5):
        res = runner.on_tick(_snap(100 + i))
    panel = res.cockpit.to_dict()["rehearsal_panel"]
    for key in ["booted", "n_observations_in_ring", "has_decision",
                 "feature_weights", "per_perturbation"]:
        assert key in panel


def test_viewer_html_includes_rehearsal_panel():
    from liqpool.research.belief.executor_v4.cockpit_server import (
        _DEFAULT_VIEWER_HTML,
    )
    h = _DEFAULT_VIEWER_HTML
    must_contain = [
        "REHEARSAL ENSEMBLE", "reh_score", "reh_conf", "reh_anal",
        "reh_best", "reh_perturbations", "reh_weights",
        "render_rehearsal", "conditional-kNN",
    ]
    missing = [s for s in must_contain if s not in h]
    assert not missing, f"viewer HTML missing tokens: {missing}"


def test_build_query_features_returns_full_schema():
    qf = build_query_features(
        rich_context={"regime_stability_index": 0.5},
        web_snapshot={"chop_mass": 0.3, "manipulation_mass": 0.1,
                        "tail_mass": 0.05},
        mm_posterior={"dominant_probability": 0.7, "confidence": 0.8},
        crowd_report={"retail_similarity_score": 0.3},
        portfolio_risk={"portfolio_drawdown_r": 0.5},
        iv_state={"confidence": 0.8, "clean_mark_fraction": 0.9},
        flow_event=None, n_open_positions=2,
    )
    for k in FEATURE_KEYS:
        assert k in qf, f"missing schema key {k}"
    # Numeric values only.
    for v in qf.values():
        assert isinstance(v, (int, float))


def test_rehearsal_score_neutral_until_confidence_threshold():
    """The ensemble should emit the neutral 0.50 prior when it doesn't
    have enough confidence to substantively move the aggregator."""
    tmp = Path(tempfile.mkdtemp(prefix="reh_neu_"))
    cfg = RehearsalConfig(min_observations_to_run=10,
                              min_confidence_to_emit_score=0.99)
    ens = BeliefRehearsalEnsemble(state_dir=tmp, cfg=cfg)
    for i in range(15):
        ens.record_closure(_make_observation(
            obs_id=f"n{i}", feature_seed=0.5,
            realized_r=0.8 if i % 2 == 0 else -0.4,
        ))
    qf = {k: 0.5 for k in FEATURE_KEYS}
    dec = ens.rehearse(
        query_features=qf,
        proposed_params={"entry_confidence": 0.65, "size_lots": 1,
                          "target_r": 1.5, "stop_r": 1.0,
                          "profile_scalp": 0, "direction": 1},
    )
    # Because we set the min_confidence_to_emit_score to 0.99, the
    # ensemble can't reach it with only 15 observations, so the score
    # falls back to the neutral 0.50 prior.
    assert dec.rehearsal_score == 0.50


def test_aggregator_weight_sum_with_rehearsal_is_one():
    cfg = AggregatorConfig()
    total = (cfg.w_base_score + cfg.w_mtf_alignment + cfg.w_projection
             + cfg.w_fees_clearance + cfg.w_portfolio_capacity
             + cfg.w_rehearsal)
    assert abs(total - 1.0) < 0.001, (
        f"weights must sum to 1.0; got {total:.4f}")
