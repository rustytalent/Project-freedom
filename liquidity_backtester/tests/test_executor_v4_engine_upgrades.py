"""Tests for executor_v4.engine_upgrades — audit fixes for Monday."""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4.engine_upgrades import (
    EngineUpgrades,
    EngineUpgradesConfig,
)


def _mk(*, bars: int = 50, is_warm: bool = False, action: str = "HOLD",
        confidence: float = 0.80, spot: float = 23000.0,
        n_dangerous: int = 0, n_dirty_marks: int = 0) -> dict:
    ts = pd.Timestamp("2026-06-19 10:00") + pd.Timedelta(seconds=bars)
    slot_readings = []
    for level in range(-5, 6):
        sstate = "dangerous" if n_dangerous > 0 else "clean"
        msrc = "ltp" if n_dirty_marks > 0 else "microprice"
        slot_readings.append({
            "strike": 23000.0 + 50 * level, "option_type": "CE",
            "level": level, "label": f"CE_{level}",
            "acceptance": "normal", "friendliness": 0.92,
            "spread_state": sstate, "mark_source": msrc,
            "is_abnormal": False, "dod_z": 0.5, "mark_price": 100.0,
        })
        if n_dangerous > 0: n_dangerous -= 1
        if n_dirty_marks > 0: n_dirty_marks -= 1
    return {
        "ts": ts, "spot": spot, "bars_seen": bars, "is_warm": is_warm,
        "thesis": {"composite_state": "HOLD_BULL",
                   "bull_thesis_score": 70.0, "bear_thesis_score": 10.0},
        "iv_state": {"state": "directional_bull", "confidence": 0.80,
                     "net_intent_z": 1.5, "clean_mark_fraction": 0.98},
        "battlefield": {"verdict": "bullish_agreement",
                        "ce_rail": {}, "pe_rail": {}},
        "decision": {"action": action, "confidence": confidence,
                     "direction": 1},
        "winding": {"zone": "NO_WINDING"},
        "slot_readings": slot_readings,
    }


def test_below_soft_floor_keeps_is_warm_false():
    eu = EngineUpgrades(EngineUpgradesConfig(soft_warmup_floor=30,
                                                hard_warmup_target=80))
    snap, report = eu.transform(_mk(bars=10))
    assert snap["is_warm"] is False
    assert not report.adaptive_warmup_active


def test_in_soft_window_promotes_warm_and_scales_confidence():
    eu = EngineUpgrades(EngineUpgradesConfig(soft_warmup_floor=30,
                                                hard_warmup_target=80,
                                                soft_warmup_confidence_cap=0.55))
    snap, report = eu.transform(_mk(bars=40, is_warm=False,
                                       confidence=0.85))
    assert snap["is_warm"] is True
    assert report.adaptive_warmup_active
    # Confidence should be scaled down.
    assert snap["decision"]["confidence"] < 0.85
    assert snap["decision"]["confidence"] <= 0.55   # cap


def test_above_hard_target_no_adaptive_action():
    eu = EngineUpgrades()
    snap, report = eu.transform(_mk(bars=100, is_warm=True,
                                       confidence=0.85))
    assert not report.adaptive_warmup_active


def test_dirty_storm_blocks_entry_action():
    eu = EngineUpgrades(EngineUpgradesConfig(dirty_storm_threshold=0.30))
    # 7 out of 11 slots dangerous = 64% → triggers storm.
    snap, report = eu.transform(_mk(bars=100, is_warm=True,
                                       action="ENTER_LONG",
                                       n_dangerous=7))
    assert report.dirty_storm_active
    assert snap["decision"]["action"] == "HOLD"


def test_dirty_storm_pause_lingers_after_clean_data():
    eu = EngineUpgrades(EngineUpgradesConfig(dirty_storm_threshold=0.30,
                                                dirty_storm_pause_bars=3))
    # Trigger storm
    eu.transform(_mk(bars=100, is_warm=True, action="HOLD",
                      n_dangerous=8))
    # Next bar: clean data, but pause should still be active.
    snap, report = eu.transform(_mk(bars=101, is_warm=True,
                                       action="ENTER_LONG", n_dangerous=0))
    assert report.dirty_storm_active
    assert snap["decision"]["action"] == "HOLD"


def test_dirty_storm_clears_after_pause_bars():
    eu = EngineUpgrades(EngineUpgradesConfig(dirty_storm_threshold=0.30,
                                                dirty_storm_pause_bars=2))
    eu.transform(_mk(bars=100, is_warm=True, action="HOLD",
                      n_dangerous=8))
    eu.transform(_mk(bars=101, is_warm=True, action="HOLD"))
    eu.transform(_mk(bars=102, is_warm=True, action="HOLD"))
    snap, report = eu.transform(_mk(bars=103, is_warm=True,
                                       action="ENTER_LONG"))
    assert not report.dirty_storm_active
    assert snap["decision"]["action"] == "ENTER_LONG"


def test_realized_vol_divergence_haircuts_confidence():
    eu = EngineUpgrades(EngineUpgradesConfig(
        divergence_high_ratio=1.5,
        divergence_confidence_haircut=0.15,
    ))
    # Build a spot history with HUGE moves to push realized vol high.
    spots = [23000.0]
    for i in range(15):
        # ±50 point swings on 1-second bars = enormous annualized vol.
        spots.append(spots[-1] + (50 if i % 2 == 0 else -50))
    for i, s in enumerate(spots):
        snap, report = eu.transform(_mk(bars=100 + i, is_warm=True,
                                           spot=s, confidence=0.80))
    assert report.realized_vol_divergence_ratio > 1.5


def test_engine_upgrades_attaches_report_to_snapshot():
    eu = EngineUpgrades()
    snap, _ = eu.transform(_mk(bars=50, is_warm=False))
    assert "engine_upgrades" in snap
    assert "actions_taken" in snap["engine_upgrades"]


def test_engine_upgrades_serializable_report():
    eu = EngineUpgrades()
    snap, report = eu.transform(_mk(bars=50, is_warm=False))
    import json
    json.dumps(report.to_dict())
