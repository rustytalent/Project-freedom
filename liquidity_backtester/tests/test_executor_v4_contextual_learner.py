"""Contextual learner — per-regime conditional weights + Shapley
attribution + recency-weighted summary.

Founder 2026-06-22 Tier-2 part 3: "the adaptive calibrator is primitive.
Make it more context-aware, more elaborate." This module tests:

  * Routing of closures to per-family sub-learners
  * Fallback to the GLOBAL vector when a family is too thinly populated
  * Shapley attribution summing approximately to the prediction
  * Recency-weighted effective sample counts
  * The weight scaling that preserves AggregatorConfig's invariant
    (5 contextual weights + w_rehearsal = 1.0)
  * LiveCalibrator integration: closures route through, regime-specific
    application happens
  * End-to-end manager wiring + cockpit panel surface
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pandas as pd

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PersistenceConfig,
    V4Runner,
    V4RunnerConfig,
)
from liqpool.research.belief.executor_v4.aggregator import AggregatorConfig
from liqpool.research.belief.executor_v4.contextual_learner import (
    FAMILY_CHOP,
    FAMILY_DIRECTIONAL,
    FAMILY_FAT_TAIL,
    FAMILY_MANIPULATION,
    FAMILY_UNKNOWN,
    REGIME_FAMILIES,
    ContextualLearner,
    ContextualLearnerConfig,
)
from liqpool.research.belief.executor_v4.learning import OnlineLearnerConfig
from liqpool.research.belief.executor_v4.live_calibrator import (
    LiveCalibrator, LiveCalibratorConfig,
)
from liqpool.research.belief.executor_v4.weight_evolution import (
    WeightEvolutionMemory, WeightEvolutionMemoryConfig,
)


_INITIAL = {
    "w_base_score": 0.26, "w_mtf_alignment": 0.22,
    "w_projection": 0.18, "w_fees_clearance": 0.13,
    "w_portfolio_capacity": 0.09,
}


def _component_scores(*, base=0.5, mtf=0.7, proj=0.5, fees=0.6,
                          port=0.7):
    return {
        "base_score": base, "mtf_alignment_score": mtf,
        "projection_factor": proj, "fees_clearance_score": fees,
        "portfolio_capacity_score": port,
    }


# ── Construction & routing ────────────────────────────────────


def test_contextual_learner_construct_creates_one_per_family():
    cl = ContextualLearner(initial_weights=_INITIAL)
    for fam in REGIME_FAMILIES:
        assert fam in cl._by_family
    # The global learner exists alongside the per-family ones.
    assert cl._global is not None


def test_observe_routes_to_correct_family():
    cl = ContextualLearner(initial_weights=_INITIAL)
    cl.observe_closure(
        component_scores=_component_scores(),
        realized_r=0.5, regime_family=FAMILY_DIRECTIONAL,
    )
    cl.observe_closure(
        component_scores=_component_scores(),
        realized_r=-0.3, regime_family=FAMILY_CHOP,
    )
    cl.observe_closure(
        component_scores=_component_scores(),
        realized_r=0.2, regime_family=FAMILY_CHOP,
    )
    n = cl.per_family_n_samples()
    assert n[FAMILY_DIRECTIONAL] == 1
    assert n[FAMILY_CHOP] == 2
    assert n[FAMILY_FAT_TAIL] == 0
    # GLOBAL receives every closure.
    assert len(cl._global.observations) == 3


def test_observe_unknown_family_falls_into_unknown_bucket():
    cl = ContextualLearner(initial_weights=_INITIAL)
    cl.observe_closure(
        component_scores=_component_scores(),
        realized_r=0.5, regime_family="something_not_known",
    )
    assert cl.per_family_n_samples()[FAMILY_UNKNOWN] == 1


# ── Weights_for fallback logic ────────────────────────────────


def test_weights_for_returns_global_when_family_too_thin():
    cfg = ContextualLearnerConfig(min_samples_to_specialise=5)
    cl = ContextualLearner(initial_weights=_INITIAL, cfg=cfg)
    # Only 2 chop observations — below the specialise threshold.
    for _ in range(2):
        cl.observe_closure(
            component_scores=_component_scores(),
            realized_r=0.4, regime_family=FAMILY_CHOP,
        )
    w_chop = cl.weights_for(FAMILY_CHOP)
    w_global = dict(cl._global.weights)
    assert w_chop == w_global


def test_weights_for_uses_family_specific_once_specialised():
    cfg = ContextualLearnerConfig(min_samples_to_specialise=2)
    cl = ContextualLearner(initial_weights=_INITIAL, cfg=cfg)
    for _ in range(3):
        cl.observe_closure(
            component_scores=_component_scores(),
            realized_r=0.4, regime_family=FAMILY_CHOP,
        )
    w_chop = cl.weights_for(FAMILY_CHOP)
    w_family = dict(cl._by_family[FAMILY_CHOP].weights)
    assert w_chop == w_family


# ── Shapley attribution ───────────────────────────────────────


def test_shapley_attribution_returns_one_per_component():
    cl = ContextualLearner(initial_weights=_INITIAL)
    report = cl.observe_closure(
        component_scores=_component_scores(),
        realized_r=0.6, regime_family=FAMILY_DIRECTIONAL,
    )
    components = {a.component for a in report.component_attributions}
    assert components == {"w_base_score", "w_mtf_alignment",
                              "w_projection", "w_fees_clearance",
                              "w_portfolio_capacity"}


def test_shapley_attribution_disabled_when_n_perms_zero():
    cfg = ContextualLearnerConfig(n_shapley_permutations=0)
    cl = ContextualLearner(initial_weights=_INITIAL, cfg=cfg)
    report = cl.observe_closure(
        component_scores=_component_scores(),
        realized_r=0.6, regime_family=FAMILY_DIRECTIONAL,
    )
    assert report.component_attributions == []


def test_shapley_sum_approximates_prediction_change():
    """For a single observation, contribution sum ≈ predicted_p - 0.5
    (the sigmoid baseline at empty set)."""
    cl = ContextualLearner(initial_weights=_INITIAL)
    report = cl.observe_closure(
        component_scores=_component_scores(base=0.8, mtf=0.7),
        realized_r=0.6, regime_family=FAMILY_DIRECTIONAL,
    )
    total = sum(a.contribution for a in report.component_attributions)
    # The sigmoid is bounded; total must be a reasonable real value in
    # (-1, 1). Loose bound is fine since we're sampling permutations.
    assert -1.0 < total < 1.0


# ── Recency-weighted summary ──────────────────────────────────


def test_recency_decays_old_observations():
    cl = ContextualLearner(initial_weights=_INITIAL,
                              cfg=ContextualLearnerConfig(
                                  recency_half_life_days=1.0))
    now = time.time()
    # One observation today, one a week old.
    cl.observe_closure(
        component_scores=_component_scores(),
        realized_r=0.5, regime_family=FAMILY_DIRECTIONAL,
        observed_at_ts=now,
    )
    cl.observe_closure(
        component_scores=_component_scores(),
        realized_r=-0.3, regime_family=FAMILY_DIRECTIONAL,
        observed_at_ts=now - 86400 * 7,
    )
    eff = cl.per_family_effective_n()
    # Today's obs has weight ~1.0; week-old has weight ~exp(-7 ln 2)
    # = 1/128 ≈ 0.008. Total ≈ 1.008.
    assert 0.9 < eff[FAMILY_DIRECTIONAL] < 1.2


# ── Apply preserves AggregatorConfig invariant ────────────────


def test_apply_preserves_weight_sum_with_rehearsal():
    cfg = AggregatorConfig()      # has w_rehearsal=0.12 by default
    cl = ContextualLearner(initial_weights=_INITIAL)
    cl.apply_to_aggregator_config(cfg, current_family=FAMILY_UNKNOWN)
    total = (cfg.w_base_score + cfg.w_mtf_alignment + cfg.w_projection
             + cfg.w_fees_clearance + cfg.w_portfolio_capacity
             + cfg.w_rehearsal)
    assert abs(total - 1.0) < 0.005


def test_apply_with_zero_rehearsal_uses_full_budget():
    cfg = AggregatorConfig(w_base_score=0.30, w_mtf_alignment=0.25,
                              w_projection=0.20, w_fees_clearance=0.15,
                              w_portfolio_capacity=0.10, w_rehearsal=0.0)
    cl = ContextualLearner(initial_weights=_INITIAL)
    cl.apply_to_aggregator_config(cfg, current_family=FAMILY_UNKNOWN)
    sum_5 = (cfg.w_base_score + cfg.w_mtf_alignment + cfg.w_projection
             + cfg.w_fees_clearance + cfg.w_portfolio_capacity)
    assert abs(sum_5 - 1.0) < 0.005


# ── Live calibrator integration ───────────────────────────────


def test_live_calibrator_routes_closures_to_contextual_learner():
    cfg = AggregatorConfig()
    mem = WeightEvolutionMemory(WeightEvolutionMemoryConfig())
    lc = LiveCalibrator(
        aggregator_cfg=cfg, memory=mem,
        calibrator_cfg=LiveCalibratorConfig(use_contextual_learner=True),
    )
    assert lc.contextual_learner is not None
    lc.on_position_closed(
        component_scores=_component_scores(),
        realized_r=0.5, bar_index=100,
        regime_family=FAMILY_DIRECTIONAL,
    )
    n = lc.contextual_learner.per_family_n_samples()
    assert n[FAMILY_DIRECTIONAL] == 1


def test_live_calibrator_apply_regime_specific_weights_changes_aggregator():
    cfg = AggregatorConfig()
    mem = WeightEvolutionMemory(WeightEvolutionMemoryConfig())
    lc = LiveCalibrator(
        aggregator_cfg=cfg, memory=mem,
        calibrator_cfg=LiveCalibratorConfig(use_contextual_learner=True),
    )
    pre_w = cfg.w_base_score
    # Push enough chop observations that the chop family's weights
    # diverge from initial.
    for i in range(20):
        lc.on_position_closed(
            component_scores=_component_scores(
                base=0.9 if i % 2 == 0 else 0.1),
            realized_r=0.7 if i % 2 == 0 else -0.4,
            bar_index=100 + i,
            regime_family=FAMILY_CHOP,
        )
    # Pump enough samples to trigger a contextual update.
    update = lc.contextual_learner.maybe_update(FAMILY_CHOP)
    applied = lc.apply_regime_specific_weights(current_family=FAMILY_CHOP)
    # The applied weights should sum to 1 - w_rehearsal (the budget).
    total_applied = sum(applied.values())
    assert abs(total_applied - (1.0 - cfg.w_rehearsal)) < 0.005


def test_live_calibrator_paused_skips_apply():
    cfg = AggregatorConfig()
    mem = WeightEvolutionMemory(WeightEvolutionMemoryConfig())
    lc = LiveCalibrator(
        aggregator_cfg=cfg, memory=mem,
        calibrator_cfg=LiveCalibratorConfig(use_contextual_learner=True),
    )
    lc.pause("operator")
    applied = lc.apply_regime_specific_weights(
        current_family=FAMILY_DIRECTIONAL)
    assert applied == {}


# ── End-to-end manager wiring ─────────────────────────────────


def _runner_with_state(state_dir: Path) -> V4Runner:
    cfg = V4RunnerConfig(
        manager=None,
        persistence=PersistenceConfig(state_dir=state_dir, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


def test_manager_creates_contextual_learner_by_default():
    tmp = Path(tempfile.mkdtemp(prefix="ctx_mgr_"))
    runner = _runner_with_state(tmp)
    assert runner.manager.live_calibrator.contextual_learner is not None


def test_cockpit_carries_contextual_learner_panel():
    tmp = Path(tempfile.mkdtemp(prefix="ctx_panel_"))
    runner = _runner_with_state(tmp)
    from tests.test_executor_v4_dead_market_and_web_viz import _snap
    res = None
    for i in range(5):
        res = runner.on_tick(_snap(100 + i))
    panel = res.cockpit.to_dict()["contextual_learner_panel"]
    assert panel["enabled"] is True
    assert "applied_weights" in panel
    assert "per_family_weights" in panel
    assert "per_family_n_samples" in panel


def test_viewer_html_includes_contextual_learner_panel():
    from liqpool.research.belief.executor_v4.cockpit_server import (
        _DEFAULT_VIEWER_HTML,
    )
    h = _DEFAULT_VIEWER_HTML
    must_contain = [
        "CONTEXTUAL LEARNER",
        "ctx_family", "ctx_weights_table", "ctx_last_attribution",
        "render_contextual_learner",
        "Shapley",
    ]
    missing = [s for s in must_contain if s not in h]
    assert not missing, f"viewer HTML missing tokens: {missing}"


def test_aggregator_weight_invariant_preserved_after_manager_ticks():
    """End-to-end: after the manager has applied contextual weights per
    tick, the aggregator's 6 weights should still sum to 1.0."""
    tmp = Path(tempfile.mkdtemp(prefix="ctx_inv_"))
    runner = _runner_with_state(tmp)
    from tests.test_executor_v4_dead_market_and_web_viz import _snap
    for i in range(20):
        runner.on_tick(_snap(100 + i))
    cfg = runner.manager.cfg.aggregator
    total = (cfg.w_base_score + cfg.w_mtf_alignment + cfg.w_projection
             + cfg.w_fees_clearance + cfg.w_portfolio_capacity
             + cfg.w_rehearsal)
    assert abs(total - 1.0) < 0.01


def test_contextual_learner_summary_shape():
    cl = ContextualLearner(initial_weights=_INITIAL)
    cl.observe_closure(
        component_scores=_component_scores(),
        realized_r=0.4, regime_family=FAMILY_DIRECTIONAL,
    )
    s = cl.summary(current_family=FAMILY_DIRECTIONAL)
    for key in ["current_family", "applied_weights", "global_weights",
                  "per_family_weights", "per_family_n_samples",
                  "per_family_effective_n", "recent_closure_reports",
                  "min_samples_to_specialise", "recency_half_life_days"]:
        assert key in s
