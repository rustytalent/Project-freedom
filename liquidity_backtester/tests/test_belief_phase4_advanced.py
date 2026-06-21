"""Tests for the advanced (opt-in) Phase-4 deviation-of-deviation features.

Founder audit (2026-06-22): the original Phase-4 used a single fixed
band window with no time-of-day awareness or multi-horizon view, and
the acceptance classifier was a hard binary with no confidence carry.

These tests cover the additive upgrades:
  * multi-horizon dod_z (tactical / current / session) + horizon_agreement
  * time-of-day band scaling
  * continuous acceptance_confidence alongside the discrete label
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liqpool.research.belief.moneyness import MoneynessSlot
from liqpool.research.belief.residual import ResidualConfig, slot_residual_frame


@pytest.fixture
def synthetic_slot_df() -> pd.DataFrame:
    """Generate a synthetic per-slot frame long enough for multi-horizon."""
    n = 320
    rng = np.random.default_rng(7)
    spot = 23500.0 + np.cumsum(rng.normal(0, 1.5, size=n))
    # Premium roughly delta-weighted on spot, with extra noise.
    mark = 100.0 + 0.5 * (spot - spot[0]) + rng.normal(0, 1.2, size=n)
    ts = pd.date_range("2026-06-23 09:15", periods=n, freq="30s",
                       tz="Asia/Kolkata")
    df = pd.DataFrame({"spot": spot, "mark": mark}, index=ts)
    return df


_SLOT = MoneynessSlot(
    option_type="CE", strike=23500.0, level=0, label="CE_ATM",
    expected_abs_delta=0.50, expected_signed_delta=0.50,
    behavior="gamma_atm", is_future_like=False,
)


# ── Default behavior: unchanged shape and columns ────────────────


def test_default_config_preserves_original_columns(synthetic_slot_df):
    out = slot_residual_frame(synthetic_slot_df, _SLOT)
    # The legacy columns the engine + battlefield depend on must all
    # be present and have the original names.
    for col in ("dod_z", "is_abnormal", "acceptance",
                "resid_median", "resid_sigma", "band_lower", "band_upper"):
        assert col in out.columns, f"missing legacy column: {col}"
    # Advanced columns must NOT appear in default mode.
    for col in ("dod_z_tactical", "dod_z_current", "dod_z_session",
                "horizon_agreement", "acceptance_confidence",
                "tod_scale_applied"):
        assert col not in out.columns, f"unexpected advanced column: {col}"


# ── Multi-horizon view ───────────────────────────────────────────


def test_multi_horizon_adds_three_horizons_and_agreement(synthetic_slot_df):
    cfg = ResidualConfig(enable_multi_horizon=True,
                          horizon_tactical=30, horizon_current=60,
                          horizon_session=240)
    out = slot_residual_frame(synthetic_slot_df, _SLOT, resid_cfg=cfg)
    for col in ("dod_z_tactical", "dod_z_current",
                "dod_z_session", "horizon_agreement"):
        assert col in out.columns
    # The "current" horizon equals the primary band_window (default 60)
    # so values should be numerically very close where both have warmed up.
    warm = out.dropna(subset=["dod_z_current"]).iloc[120:]
    assert ((warm["dod_z"] - warm["dod_z_current"]).abs() < 1e-6).all()
    # Agreement must be a fraction in [0, 1].
    agreement = out["horizon_agreement"].dropna()
    assert ((agreement >= 0.0) & (agreement <= 1.0)).all()


def test_multi_horizon_agreement_at_fresh_anomaly():
    """A fresh sharp anomaly should make tactical AND current bands fire
    in the same direction, raising horizon_agreement above the quiet
    baseline. (A SUSTAINED anomaly gets absorbed by the rolling band —
    that's correct behavior; the band re-anchors. Testing the FRESH
    transition is the meaningful signal.)"""
    n = 280
    rng = np.random.default_rng(11)
    spot = 23500.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    mark = 100.0 + 0.5 * (spot - spot[0]) + rng.normal(0, 0.4, size=n)
    # Sharp 5-bar premium pump at bar 240 — too short for any band to
    # fully re-anchor, but long enough for tactical to catch it first.
    mark[240:245] += 12.0
    ts = pd.date_range("2026-06-23 09:15", periods=n, freq="30s",
                       tz="Asia/Kolkata")
    df = pd.DataFrame({"spot": spot, "mark": mark}, index=ts)
    cfg = ResidualConfig(enable_multi_horizon=True)
    out = slot_residual_frame(df, _SLOT, resid_cfg=cfg)
    # Quiet baseline (bars 200-235): low agreement expected.
    quiet = out["horizon_agreement"].iloc[200:235].mean()
    # Fresh anomaly window (bars 240-245): higher agreement expected.
    fresh = out["horizon_agreement"].iloc[240:248].max()
    assert fresh >= quiet, f"fresh {fresh:.2f} should be >= quiet {quiet:.2f}"


def test_multi_horizon_rejects_bad_config():
    with pytest.raises(ValueError):
        ResidualConfig(enable_multi_horizon=True,
                        horizon_tactical=80, horizon_current=60,
                        horizon_session=200)


# ── Time-of-day band scaling ─────────────────────────────────────


def test_time_of_day_scaling_widens_band_when_scale_high(synthetic_slot_df):
    df = synthetic_slot_df
    # Build a per-bar scale: 1.0 normally, 2.5 in the second half.
    scale = pd.Series(1.0, index=df.index)
    scale.iloc[len(df) // 2:] = 2.5
    cfg = ResidualConfig(enable_time_of_day_scaling=True)
    wide = slot_residual_frame(df, _SLOT, resid_cfg=cfg,
                                  time_of_day_scale=scale)
    narrow = slot_residual_frame(df, _SLOT)
    # In the wider-band window, |dod_z| must be LOWER on average.
    half = len(df) // 2 + 60
    wide_z = wide["dod_z"].iloc[half:].abs().mean()
    narrow_z = narrow["dod_z"].iloc[half:].abs().mean()
    assert wide_z < narrow_z + 1e-9


def test_time_of_day_scale_clipped(synthetic_slot_df):
    df = synthetic_slot_df
    # Out-of-range scales: must be clipped to [floor, cap].
    scale = pd.Series(99.0, index=df.index)
    cfg = ResidualConfig(enable_time_of_day_scaling=True,
                          time_of_day_scale_floor=0.5,
                          time_of_day_scale_cap=3.0)
    out = slot_residual_frame(df, _SLOT, resid_cfg=cfg,
                                time_of_day_scale=scale)
    assert "tod_scale_applied" in out.columns
    applied = out["tod_scale_applied"].dropna()
    assert (applied <= 3.0).all()
    assert (applied >= 0.5).all()


# ── Acceptance confidence (continuous) ───────────────────────────


def test_acceptance_confidence_in_unit_interval(synthetic_slot_df):
    cfg = ResidualConfig(enable_acceptance_confidence=True)
    out = slot_residual_frame(synthetic_slot_df, _SLOT, resid_cfg=cfg)
    assert "acceptance_confidence" in out.columns
    conf = out["acceptance_confidence"].dropna()
    assert ((conf >= 0.0) & (conf <= 1.0)).all()


def test_acceptance_confidence_spikes_at_fresh_anomaly():
    """The confidence rises immediately after a fresh anomaly arrives,
    then decays as the rolling band re-anchors (which IS correct behavior:
    a sustained level stops being abnormal once the band catches up)."""
    n = 200
    rng = np.random.default_rng(19)
    spot = 23500.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    mark = 100.0 + 0.5 * (spot - spot[0]) + rng.normal(0, 0.4, size=n)
    # Sharp 5-bar premium shock at bar 130.
    mark[130:135] += 8.0
    ts = pd.date_range("2026-06-23 09:15", periods=n, freq="30s",
                       tz="Asia/Kolkata")
    df = pd.DataFrame({"spot": spot, "mark": mark}, index=ts)
    cfg = ResidualConfig(enable_acceptance_confidence=True,
                          acceptance_confidence_window=8)
    out = slot_residual_frame(df, _SLOT, resid_cfg=cfg)
    quiet = out["acceptance_confidence"].iloc[100:128].mean()
    fresh = out["acceptance_confidence"].iloc[130:140].max()
    assert fresh > quiet, f"fresh {fresh:.2f} should exceed quiet {quiet:.2f}"
