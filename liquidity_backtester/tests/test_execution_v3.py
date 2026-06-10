"""Stream K — execution_simulator_v3 tests.

Pins:
  * Near-zero price guard rejects sizing before V2's billion-share bug
  * Same-bar tie-breaker is configurable, not hard-coded
  * Sqrt-impact adds bps only above the notional threshold
  * Untrained learned models fall back to V2 baselines (no regression)
  * Trained slippage model takes over from baseline
  * Fill / survival prediction respects fit/unfit state
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from liqpool.execution.fill_model import FillProbabilityModel
from liqpool.execution.slippage_model import SlippageRealisationModel
from liqpool.execution.survival_model import TimeToEventModel
from liqpool.execution_simulator_v3 import (
    ExecutionV3Config,
    check_sizing_v3,
    predict_bars_to_stop_v3,
    predict_bars_to_target_v3,
    predict_fill_probability_v3,
    predict_slippage_bps_v3,
    resolve_same_bar_tiebreak_v3,
)


# ---------------------------------------------------------------------------
# Near-zero price guard
# ---------------------------------------------------------------------------

def test_sizing_rejected_for_sub_paisa_reference_price():
    """The V2 bug: notional 1_00_000 / 1e-9 = 100 billion shares.
    V3 catches this before the multiplication happens."""
    cfg = ExecutionV3Config()
    d = check_sizing_v3(notional_inr=100_000, reference_price=0.005, cfg=cfg)
    assert not d.accepted
    assert d.quantity == 0
    assert "below floor" in d.rejection_reason


def test_sizing_rejected_for_non_finite_price():
    cfg = ExecutionV3Config()
    d = check_sizing_v3(notional_inr=100_000, reference_price=float("nan"), cfg=cfg)
    assert not d.accepted


def test_sizing_accepted_for_sensible_inputs():
    cfg = ExecutionV3Config()
    d = check_sizing_v3(notional_inr=100_000, reference_price=1000.0, cfg=cfg)
    assert d.accepted
    assert d.quantity == 100
    assert d.notional_used_inr == 100_000.0


def test_sizing_rejected_when_rounding_to_zero():
    cfg = ExecutionV3Config()
    # ₹500 notional / ₹1000 price -> 0.5 -> rounds to 0
    d = check_sizing_v3(notional_inr=500, reference_price=1000.0, cfg=cfg)
    assert not d.accepted
    assert "rounds to zero" in d.rejection_reason


# ---------------------------------------------------------------------------
# Same-bar tie-breaker
# ---------------------------------------------------------------------------

def test_same_bar_tiebreaker_defaults_conservative_v2():
    cfg = ExecutionV3Config()
    assert resolve_same_bar_tiebreak_v3(cfg, True, True) == "stop"


def test_same_bar_tiebreaker_configurable_to_target():
    cfg = ExecutionV3Config(conservative_same_bar_resolution=False)
    assert resolve_same_bar_tiebreak_v3(cfg, True, True) == "target"


def test_same_bar_tiebreaker_passes_through_single_hits():
    cfg = ExecutionV3Config()
    assert resolve_same_bar_tiebreak_v3(cfg, True, False) == "stop"
    assert resolve_same_bar_tiebreak_v3(cfg, False, True) == "target"
    assert resolve_same_bar_tiebreak_v3(cfg, False, False) == "neither"


# ---------------------------------------------------------------------------
# Sqrt-impact
# ---------------------------------------------------------------------------

def test_impact_zero_below_threshold():
    cfg = ExecutionV3Config()
    state = {"size_notional": 100_000.0}  # at threshold (200k default)
    s = predict_slippage_bps_v3(state, cfg, baseline_bps=2.0)
    # Below threshold -> no impact, same as baseline.
    assert math.isclose(s, 2.0, rel_tol=1e-9)


def test_impact_grows_with_sqrt_of_notional():
    cfg = ExecutionV3Config(
        impact_threshold_inr=100_000,
        impact_coefficient_bps=1.5,
        impact_typical_depth_inr=100_000,
    )
    # 4x typical depth -> sqrt(4) = 2 -> 1.5 * 2 = 3 bps impact.
    state = {"size_notional": 400_000.0}
    s = predict_slippage_bps_v3(state, cfg, baseline_bps=2.0)
    assert math.isclose(s, 2.0 + 3.0, rel_tol=1e-6)


def test_impact_disabled_when_coefficient_zero():
    cfg = ExecutionV3Config(impact_coefficient_bps=0.0)
    state = {"size_notional": 10_000_000.0}
    s = predict_slippage_bps_v3(state, cfg, baseline_bps=2.0)
    assert math.isclose(s, 2.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Fallback paths when models aren't trained
# ---------------------------------------------------------------------------

def test_slippage_uses_baseline_when_model_not_fit():
    cfg = ExecutionV3Config(slippage_model=SlippageRealisationModel())
    state = {"size_notional": 50_000.0, "session_phase": "mid"}
    s = predict_slippage_bps_v3(state, cfg, baseline_bps=3.7)
    # No impact (below default threshold), no learned model -> baseline.
    assert math.isclose(s, 3.7, rel_tol=1e-9)


def test_fill_probability_uses_baseline_when_model_not_fit():
    cfg = ExecutionV3Config(fill_model=FillProbabilityModel())
    p = predict_fill_probability_v3({}, cfg, baseline_prob=0.75)
    assert math.isclose(p, 0.75, rel_tol=1e-9)


def test_survival_uses_baseline_when_model_not_fit():
    cfg = ExecutionV3Config(survival_model=TimeToEventModel())
    t = predict_bars_to_target_v3({}, cfg, baseline_bars=42.0)
    assert math.isclose(t, 42.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Trained model takes over
# ---------------------------------------------------------------------------

def _build_slippage_training_frame(n: int = 500) -> pd.DataFrame:
    """Synthetic training frame with a clear vol -> slippage signal."""
    rng = np.random.default_rng(0)
    vol = rng.uniform(0.001, 0.02, n)        # 0.1% to 2%
    distance = rng.uniform(0.0, 6.0, n)
    size = rng.uniform(20_000, 300_000, n)
    side_long = rng.integers(0, 2, n).astype(float)
    phase = rng.integers(0, 3, n)
    session_open = (phase == 0).astype(float)
    session_close = (phase == 1).astype(float)
    mode_market = rng.integers(0, 2, n).astype(float)
    mode_limit = 1.0 - mode_market
    # Target: 2 bps base + 10 bps per pct vol + 1 / (1 + distance)
    target = (
        2.0
        + 10.0 * 100.0 * vol  # vol is fractional; scale to pct
        + 1.0 / (1.0 + distance)
        + 1.5 * session_open
        + 0.5 * session_close
        + rng.normal(0.0, 0.5, n)
    )
    return pd.DataFrame({
        "vol_frac": vol,
        "distance_atr": distance,
        "size_notional": size,
        "side_long": side_long,
        "session_open": session_open,
        "session_close": session_close,
        "mode_market": mode_market,
        "mode_limit": mode_limit,
        "slippage_bps": target,
    })


def test_trained_slippage_model_takes_over_from_baseline():
    """When fit, the learned model overrides the baseline param.
    Pin: a high-vol state should predict materially higher slippage
    than a low-vol state under the same other features."""
    trades_long = _build_slippage_training_frame(n=500)
    model = SlippageRealisationModel().fit(trades_long)
    cfg = ExecutionV3Config(slippage_model=model)
    high = predict_slippage_bps_v3(
        {"vol_frac": 0.02, "distance_atr": 1.0,
         "size_notional": 100_000, "side_long": 1,
         "session_open": 0, "session_close": 0,
         "mode_market": 1, "mode_limit": 0},
        cfg, baseline_bps=999.0,  # baseline ignored
    )
    low = predict_slippage_bps_v3(
        {"vol_frac": 0.001, "distance_atr": 1.0,
         "size_notional": 100_000, "side_long": 1,
         "session_open": 0, "session_close": 0,
         "mode_market": 1, "mode_limit": 0},
        cfg, baseline_bps=999.0,
    )
    assert high > low + 1.0  # learned signal: vol up -> slippage up


def test_trained_slippage_model_clamps_to_predict_band():
    """Even a stale or thinly-trained model can't break V3 — the
    public predictor clamps into [FLOOR, CEIL] bps."""
    trades_long = _build_slippage_training_frame(n=200)
    model = SlippageRealisationModel().fit(trades_long)
    cfg = ExecutionV3Config(slippage_model=model)
    # Pathological inputs (NaN everywhere) -> fallback in the
    # wrapper -> clamped by SlippageRealisationModel.predict_bps.
    nan_state = {k: float("nan") for k in [
        "vol_frac", "distance_atr", "size_notional", "side_long",
        "session_open", "session_close", "mode_market", "mode_limit",
    ]}
    s = predict_slippage_bps_v3(nan_state, cfg, baseline_bps=0.0)
    assert 0.5 <= s <= 50.0


# ---------------------------------------------------------------------------
# Survival head fit/predict
# ---------------------------------------------------------------------------

def test_survival_model_fits_target_and_stop_separately():
    """Build a synthetic trades frame where 'target' trades resolve
    fast and 'stop' trades resolve slow; verify the heads learn
    distinct expected values."""
    rng = np.random.default_rng(1)
    n = 400
    rows = []
    for _ in range(n):
        if rng.random() < 0.5:
            rows.append({
                "outcome": "target_hit",
                "bars_held": int(rng.normal(15, 4)),
                "distance_atr": rng.uniform(0.5, 3.0),
                "vol_frac": rng.uniform(0.005, 0.015),
                "side": "long",
                "session_phase": "mid",
                "tf_count": 2, "score": 1.2,
            })
        else:
            rows.append({
                "outcome": "stop_hit",
                "bars_held": int(rng.normal(60, 10)),
                "distance_atr": rng.uniform(0.5, 3.0),
                "vol_frac": rng.uniform(0.005, 0.015),
                "side": "long",
                "session_phase": "mid",
                "tf_count": 2, "score": 1.2,
            })
    trades = pd.DataFrame(rows)
    model = TimeToEventModel().fit(trades)
    cfg = ExecutionV3Config(survival_model=model)
    state = {
        "distance_atr": 1.5, "vol_frac": 0.01,
        "side_long": 1.0, "session_open": 0.0,
        "session_close": 0.0, "tf_count": 2, "score": 1.2,
    }
    e_target = predict_bars_to_target_v3(state, cfg, baseline_bars=99.0)
    e_stop = predict_bars_to_stop_v3(state, cfg, baseline_bars=99.0)
    # The two heads must have separated.
    assert e_target < e_stop


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

def test_slippage_model_save_load_round_trip(tmp_path):
    trades_long = _build_slippage_training_frame(n=200)
    model = SlippageRealisationModel().fit(trades_long)
    path = tmp_path / "slippage.txt"
    model.save(path)
    loaded = SlippageRealisationModel.load(path)
    assert loaded.is_fitted
    # Predictions match within a tiny tolerance (LightGBM model save
    # is lossless for floats but JSON metadata could pick up rounding).
    state = {
        "vol_frac": 0.01, "distance_atr": 1.0, "size_notional": 100_000,
        "side_long": 1, "session_open": 0, "session_close": 0,
        "mode_market": 1, "mode_limit": 0,
    }
    assert math.isclose(
        model.predict_bps(state), loaded.predict_bps(state),
        rel_tol=1e-6,
    )


def test_unfit_model_save_load_keeps_unfit_state(tmp_path):
    """A model that was never fit should round-trip as an unfit
    model (fallback predictions only)."""
    model = SlippageRealisationModel()
    path = tmp_path / "empty.txt"
    model.save(path)
    loaded = SlippageRealisationModel.load(path)
    assert not loaded.is_fitted
