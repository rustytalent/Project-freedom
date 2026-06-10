"""Magic-number closure — registry/code agreement pins.

The tuning registry (liqpool/tuning.py) documents every numeric
threshold with rationale + validation status. These tests enforce
that the registry can never lie: each registry value is pinned to
the ACTUAL default in the code. Change a code default without
updating the registry (or vice versa) and CI fails with a message
naming both sides.

Also pins the registry's own hygiene: every entry has a non-empty
rationale and a known status; every unvalidated/bootstrap entry
with a sweep_range is reachable by the sensitivity harness.
"""
from __future__ import annotations

import pytest

from liqpool.tuning import REGISTRY, by_status, get, sweepable


class _Bare:
    """Object with NO detector attributes — getattr falls back to the
    inline defaults, which is exactly what we want to pin."""


# ---------------------------------------------------------------------------
# Detector fallbacks
# ---------------------------------------------------------------------------

def test_sweep_detector_defaults_match_registry():
    from liqpool.detectors.sweep import _sweep_params
    cfg = _sweep_params(_Bare())
    assert cfg["min_atr"] == get("sweep_min_atr").value
    assert cfg["reclaim_window"] == get("sweep_reclaim_window").value
    assert cfg["strength_cap"] == get("sweep_strength_cap").value
    assert cfg["stop_run_close"] == get("stop_run_close_atr").value


def test_imbalance_detector_defaults_match_registry():
    from liqpool.detectors.imbalance import _imbalance_params
    cfg = _imbalance_params(_Bare())
    assert cfg["min_run"] == get("imbalance_min_run").value
    assert cfg["max_run"] == get("imbalance_max_run").value
    assert cfg["min_atr"] == get("imbalance_min_atr").value


def test_volume_detector_defaults_match_registry():
    from liqpool.detectors.volume import _volume_params
    cfg = _volume_params(_Bare())
    assert cfg["vw_window"] == get("vw_swing_window").value
    assert cfg["vw_multiplier"] == get("vw_swing_multiplier").value
    assert cfg["cd_lookback"] == get("cum_delta_lookback").value


# ---------------------------------------------------------------------------
# Options executor / CCV constants
# ---------------------------------------------------------------------------

def test_executor_constants_match_registry():
    from liqpool.options import executor as ex
    assert ex.FORCE_ENTRY_THRESHOLD == get("force_entry_threshold").value
    assert ex.WAIT_THRESHOLD == get("wait_threshold").value
    assert ex.VIX_KILL_SPIKE == get("vix_kill_spike").value


def test_ccv_constants_match_registry():
    from liqpool.options import ccv
    assert ccv.MACRO_GATE_THRESHOLD == get("macro_gate_threshold").value


# ---------------------------------------------------------------------------
# Execution / arsenal / ml constants
# ---------------------------------------------------------------------------

def test_execution_v3_impact_matches_registry():
    from liqpool.execution_simulator_v3 import ExecutionV3Config
    assert ExecutionV3Config().impact_coefficient_bps == \
        get("impact_coefficient_bps").value


def test_arsenal_cost_ratio_matches_registry():
    from liqpool.arsenal.evaluator import EvaluatorConfig
    assert EvaluatorConfig().min_target_to_cost_ratio == \
        get("arsenal_min_target_to_cost_ratio").value


def test_empirical_bayes_fallback_matches_registry():
    """The legacy prior strength survives only as the empirical-Bayes
    fallback; pin that the fallback agrees with the registry."""
    import inspect
    from liqpool.calibration.empirical_bayes import fit_beta_binomial_prior
    sig = inspect.signature(fit_beta_binomial_prior)
    assert sig.parameters["fallback_prior_strength"].default == \
        get("bucket_shrinkage_prior_strength").value


# ---------------------------------------------------------------------------
# Registry hygiene
# ---------------------------------------------------------------------------

VALID_STATUSES = {"bootstrap", "unvalidated", "validated", "data-derived"}


def test_every_entry_has_rationale_and_valid_status():
    for name, t in REGISTRY.items():
        assert len(t.rationale.strip()) > 20, (
            f"{name}: rationale must explain WHY, not just exist"
        )
        assert t.status in VALID_STATUSES, f"{name}: bad status {t.status}"
        assert t.used_in.strip(), f"{name}: must name its use site"


def test_unvalidated_entries_are_sweepable():
    """Anything marked unvalidated must carry a sweep range —
    otherwise the harness can never promote it and the status is a
    dead end."""
    for t in by_status("unvalidated"):
        assert t.sweep_range is not None, (
            f"{t.name}: unvalidated but no sweep_range; either give it "
            f"a range or reclassify (bootstrap = accepted-as-is rail)"
        )


def test_sweep_grid_includes_current_default():
    from analysis.run_param_sensitivity import build_grid
    for t in sweepable():
        grid = build_grid(t.name)
        assert any(abs(g - t.value) < 1e-9 for g in grid), (
            f"{t.name}: sweep grid must include the current default so "
            f"the report shows where the engine sits on the curve"
        )


def test_curve_classifier_verdicts():
    from analysis.run_param_sensitivity import classify_curve
    flat = {0.1: 100.0, 0.2: 101.0, 0.3: 99.5, 0.4: 100.5}
    assert classify_curve(flat) == "plateau"
    rising = {0.1: 50.0, 0.2: 80.0, 0.3: 110.0, 0.4: 140.0}
    assert classify_curve(rising) == "monotone"
    cliff = {0.1: 100.0, 0.2: 99.0, 0.3: 10.0, 0.4: 9.0}
    assert classify_curve(cliff) == "cliff"
