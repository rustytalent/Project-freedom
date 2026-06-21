"""Integration test: PortfolioManager drives the AdaptiveExitEngine."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PersistenceConfig,
    V4Runner,
    V4RunnerConfig,
)


def _mk(*, bars: int = 100, action: str = "HOLD", confidence: float = 0.82,
        spot: float = 23000.0, premium: float = 100.0) -> dict:
    ts = pd.Timestamp("2026-06-23 10:00") + pd.Timedelta(seconds=bars)
    slot_readings = []
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "CE", "level": level,
            "label": f"CE_{'ATM' if level == 0 else (f'OTM{level}' if level > 0 else f'ITM{abs(level)}')}",
            "acceptance": "normal", "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "is_abnormal": False, "dod_z": 1.5, "mark_price": premium,
        })
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "PE", "level": level,
            "label": f"PE_{'ATM' if level == 0 else (f'OTM{level}' if level > 0 else f'ITM{abs(level)}')}",
            "acceptance": "normal", "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "is_abnormal": False, "dod_z": -1.0, "mark_price": 100.0,
        })
    return {
        "ts": ts, "spot": spot, "bars_seen": bars, "is_warm": True,
        "thesis": {"composite_state": "HOLD_BULL",
                   "bull_thesis_score": 70.0, "bear_thesis_score": 10.0,
                   "no_trade_score": 5.0},
        "iv_state": {"state": "directional_bull", "direction": 1,
                     "confidence": 0.85,
                     "clean_mark_fraction": 0.98, "net_intent_z": 2.5},
        "battlefield": {
            "verdict": "bullish_agreement", "direction": 1, "confidence": 0.8,
            "ce_rail": {"weighted_mean_signed_z": 1.5,
                        "dispersion_score": 0.10,
                        "epicenter_label": "CE_ATM", "epicenter_level": 0},
            "pe_rail": {"weighted_mean_signed_z": -1.0,
                        "dispersion_score": 0.10,
                        "epicenter_label": "PE_ATM", "epicenter_level": 0},
        },
        "decision": {"action": action, "direction": 1,
                     "confidence": confidence, "trade_allowed": True,
                     "strike": {"side": "CE", "level": 0, "label": "CE_ATM"}},
        "winding": {"zone": "NO_WINDING"},
        "bull_state": {"state_index": 2},
        "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


def _new_runner() -> V4Runner:
    tmp = Path(tempfile.mkdtemp(prefix="v4_exit_test_"))
    cfg = V4RunnerConfig(
        persistence=PersistenceConfig(state_dir=tmp, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


def test_manager_registers_exit_engine_on_open():
    runner = _new_runner()
    for i in range(90):
        runner.on_tick(_mk(bars=100 + i, action="HOLD"))
    result = runner.on_tick(_mk(bars=200, action="ENTER_LONG"))
    assert result.intent.new_entry is not None
    pid = result.intent.new_entry["hypothesis"]["position_id"]
    assert pid in runner.manager.portfolio_exit.engines


def test_manager_drives_exit_engine_each_tick():
    runner = _new_runner()
    for i in range(90):
        runner.on_tick(_mk(bars=100 + i, action="HOLD"))
    runner.on_tick(_mk(bars=200, action="ENTER_LONG"))
    # Run several ticks with rising premium and check that exit engine
    # decisions appear in the portfolio_summary.
    last_result = None
    for i in range(1, 8):
        last_result = runner.on_tick(_mk(bars=200 + i, action="HOLD",
                                            premium=100.0 + i * 2))
    assert last_result is not None
    summary = last_result.intent.portfolio_summary
    assert "adaptive_exit" in summary
    if summary["adaptive_exit"] is not None:
        assert "per_position" in summary["adaptive_exit"]


def test_manager_unregisters_exit_engine_on_close():
    runner = _new_runner()
    for i in range(90):
        runner.on_tick(_mk(bars=100 + i, action="HOLD"))
    runner.on_tick(_mk(bars=200, action="ENTER_LONG"))
    n_before = len(runner.manager.portfolio_exit.engines)
    assert n_before == 1
    # Force exit
    runner.on_tick(_mk(bars=201, action="EXIT"))
    n_after = len(runner.manager.portfolio_exit.engines)
    assert n_after == 0


def test_exit_engine_decision_serializable_in_intent():
    runner = _new_runner()
    for i in range(90):
        runner.on_tick(_mk(bars=100 + i, action="HOLD"))
    runner.on_tick(_mk(bars=200, action="ENTER_LONG"))
    result = runner.on_tick(_mk(bars=201, action="HOLD"))
    import json
    json.dumps(result.intent.to_dict(), default=str)


def test_exit_engine_does_not_break_on_no_held_premium():
    """Edge case: snapshot has no slot_readings → exit engine should
    gracefully skip without breaking the loop."""
    runner = _new_runner()
    for i in range(90):
        runner.on_tick(_mk(bars=100 + i, action="HOLD"))
    runner.on_tick(_mk(bars=200, action="ENTER_LONG"))
    # Strip slot_readings — exit engine should handle it.
    snap = _mk(bars=201, action="HOLD")
    snap["slot_readings"] = []
    result = runner.on_tick(snap)
    # Should not raise; intent should still come through.
    assert result.intent is not None
