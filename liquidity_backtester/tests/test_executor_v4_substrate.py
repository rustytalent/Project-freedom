"""Tests for executor_v4.substrate — rich-context extraction."""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4.substrate import (
    SubstrateConfig,
    SubstrateState,
    augment_snapshot,
)


def _mk_snap(*, spot: float = 23000.0, bars: int = 100,
              ce_signed: float = 1.2, pe_signed: float = -1.0,
              ce_disp: float = 0.05, pe_disp: float = 0.05,
              ce_epi_label: str = "CE_ATM", ce_epi_level: int = 0,
              bull_score: float = 60.0, bear_score: float = 10.0,
              thesis_state: str = "HOLD_BULL",
              slot_acceptance: dict | None = None,
              ts: pd.Timestamp | None = None) -> dict:
    if ts is None:
        ts = pd.Timestamp("2026-06-19 10:00")
    slot_acceptance = slot_acceptance or {}
    slot_readings = []
    for level in range(-5, 6):
        for typ in ("CE", "PE"):
            label = f"{typ}_{'ATM' if level == 0 else f'OTM{level}' if level > 0 else f'ITM{abs(level)}'}"
            slot_readings.append({
                "strike": 23000.0 + 50 * level,
                "option_type": typ,
                "label": label,
                "moneyness_label": label,
                "level": level,
                "behavior": "gamma_atm" if abs(level) <= 1 else "convex_otm",
                "dod_z": 0.5 if (typ == "CE" and level >= 0) else -0.3,
                "is_abnormal": False,
                "acceptance": slot_acceptance.get(label, "normal"),
                "friendliness": 0.92,
                "spread_state": "clean",
                "mark_source": "microprice",
                "mark_price": 100.0,
            })
    return {
        "ts": ts, "spot": spot, "bars_seen": bars, "is_warm": True,
        "thesis": {"composite_state": thesis_state,
                   "bull_thesis_score": bull_score,
                   "bear_thesis_score": bear_score,
                   "no_trade_score": 5.0},
        "iv_state": {"state": "directional_bull", "direction": 1,
                     "confidence": 0.85, "clean_mark_fraction": 0.98,
                     "net_intent_z": ce_signed - pe_signed},
        "battlefield": {
            "verdict": "bullish_agreement", "direction": 1, "confidence": 0.8,
            "ce_rail": {"weighted_mean_signed_z": ce_signed,
                        "weighted_mean_abs_z": abs(ce_signed),
                        "dispersion_score": ce_disp,
                        "epicenter_label": ce_epi_label,
                        "epicenter_level": ce_epi_level},
            "pe_rail": {"weighted_mean_signed_z": pe_signed,
                        "weighted_mean_abs_z": abs(pe_signed),
                        "dispersion_score": pe_disp,
                        "epicenter_label": "PE_ATM",
                        "epicenter_level": 0},
        },
        "decision": {"action": "HOLD", "confidence": 0.8,
                     "direction": 1, "trade_allowed": True},
        "winding": {"zone": "NO_WINDING"},
        "bull_state": {"state_index": 2},
        "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


def test_substrate_first_observation_returns_zero_derivatives():
    state = SubstrateState()
    rich = state.observe(_mk_snap())
    assert rich.ce_signed_z_velocity == 0.0
    assert rich.ce_signed_z_acceleration == 0.0
    assert len(rich.heatmap_flat) == 22


def test_substrate_computes_velocity_after_window_bars():
    state = SubstrateState(SubstrateConfig(velocity_window=5))
    # Push 6 bars with ce_signed climbing 0.5 → 0.6 → 0.7 → 0.8 → 0.9 → 1.0
    for i, ce in enumerate([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        rich = state.observe(_mk_snap(ce_signed=ce, bars=100 + i))
    # velocity over last 5 bars = (1.0 - 0.5) / 5 = 0.1 per bar
    assert rich.ce_signed_z_velocity == 0.1


def test_substrate_acceptance_run_length_tracks_state_changes():
    state = SubstrateState()
    # Three bars all defended on CE_ATM
    for i in range(3):
        rich = state.observe(_mk_snap(slot_acceptance={"CE_ATM": "defended"},
                                       bars=100 + i))
    assert rich.acceptance_run_lengths["CE_ATM"] == 3
    # Bar 4: CE_ATM flips to normal → run resets
    rich = state.observe(_mk_snap(slot_acceptance={}, bars=103))
    assert rich.acceptance_run_lengths["CE_ATM"] == 1


def test_substrate_epicenter_migration_tracking():
    state = SubstrateState()
    state.observe(_mk_snap(ce_epi_label="CE_ATM", ce_epi_level=0, bars=100))
    state.observe(_mk_snap(ce_epi_label="CE_OTM2", ce_epi_level=2, bars=101))
    rich = state.observe(_mk_snap(ce_epi_label="CE_OTM4", ce_epi_level=4, bars=102))
    assert rich.epicenter_migration_distance >= 4
    assert any("epicenter migrated" in n for n in rich.notes)


def test_substrate_regime_stability_falls_with_chop():
    """If signals whip violently (big swings on ce_signed and thesis flips),
    regime_stability_index should drop."""
    state = SubstrateState()
    chops_ce = [2.5, -2.5, 2.5, -2.5, 2.5, -2.5, 2.5, -2.5, 2.5, -2.5, 2.5, -2.5]
    chops_thesis = ["BULL_ENTRY", "BEAR_ENTRY"] * 6
    chops_bull = [80.0, 10.0, 80.0, 10.0, 80.0, 10.0, 80.0, 10.0, 80.0, 10.0, 80.0, 10.0]
    chops_bear = [10.0, 80.0, 10.0, 80.0, 10.0, 80.0, 10.0, 80.0, 10.0, 80.0, 10.0, 80.0]
    for i, (ce, ts, bs, brs) in enumerate(zip(chops_ce, chops_thesis,
                                                chops_bull, chops_bear)):
        rich = state.observe(_mk_snap(
            ce_signed=ce, thesis_state=ts,
            bull_score=bs, bear_score=brs,
            bars=100 + i,
        ))
    assert rich.regime_stability_index < 0.7, (
        f"Got stability {rich.regime_stability_index} — expected chop to drop it"
    )


def test_substrate_thesis_velocity_direction():
    state = SubstrateState()
    # Bull score rising fast
    for i, bs in enumerate([20, 30, 45, 60, 75, 90]):
        rich = state.observe(_mk_snap(bull_score=bs, bear_score=5,
                                        bars=100 + i))
    assert rich.thesis_velocity_dominant_side == "bull"
    assert rich.thesis_bull_velocity > 0


def test_substrate_to_dict_serializable():
    state = SubstrateState()
    rich = state.observe(_mk_snap())
    import json
    json.dumps(rich.to_dict(), default=str)


def test_substrate_reset_clears_history():
    state = SubstrateState()
    state.observe(_mk_snap(bars=100))
    state.observe(_mk_snap(bars=101))
    state.reset()
    rich = state.observe(_mk_snap(bars=200))
    assert rich.bar_index == 1


def test_augment_snapshot_convenience():
    state = SubstrateState()
    rich = augment_snapshot(_mk_snap(), state)
    assert rich.bar_index == 1
