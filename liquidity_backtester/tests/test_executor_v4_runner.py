"""Tests for executor_v4.v4_runner — paper mode end-to-end + state persistence."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PersistenceConfig,
    V4Runner,
    V4RunnerConfig,
)
from liqpool.research.belief.executor_v4.broker.base import STATE_FILLED


def _mk(*, bars: int = 100, action: str = "HOLD", direction: int = 1,
        thesis: str = "HOLD_BULL", iv: str = "directional_bull",
        confidence: float = 0.82, spot: float = 23000.0) -> dict:
    ts = pd.Timestamp("2026-06-19 10:00") + pd.Timedelta(seconds=bars)
    slot_readings = []
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "CE", "level": level,
            "label": f"CE_{'ATM' if level == 0 else (f'OTM{level}' if level > 0 else f'ITM{abs(level)}')}",
            "acceptance": "normal", "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "is_abnormal": False, "dod_z": 1.5, "mark_price": 100.0,
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
        "thesis": {"composite_state": thesis,
                   "bull_thesis_score": 70.0, "bear_thesis_score": 10.0,
                   "no_trade_score": 5.0},
        "iv_state": {"state": iv, "direction": direction, "confidence": 0.85,
                     "clean_mark_fraction": 0.98, "net_intent_z": 2.5},
        "battlefield": {
            "verdict": "bullish_agreement", "direction": direction,
            "confidence": 0.8,
            "ce_rail": {"weighted_mean_signed_z": 1.5,
                        "dispersion_score": 0.10,
                        "epicenter_label": "CE_ATM", "epicenter_level": 0},
            "pe_rail": {"weighted_mean_signed_z": -1.0,
                        "dispersion_score": 0.10,
                        "epicenter_label": "PE_ATM", "epicenter_level": 0},
        },
        "decision": {"action": action, "direction": direction,
                     "confidence": confidence, "trade_allowed": True,
                     "strike": {"side": "CE", "level": 0,
                                 "label": "CE_ATM"}},
        "winding": {"zone": "NO_WINDING"},
        "bull_state": {"state_index": 2},
        "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


def _warmup_runner(runner: V4Runner, n: int = 90) -> None:
    for i in range(n):
        runner.on_tick(_mk(bars=100 + i, action="HOLD"))


def _new_tmp_runner() -> V4Runner:
    """Build a paper-mode runner with a tmp state dir."""
    tmp = Path(tempfile.mkdtemp(prefix="v4runner_test_"))
    cfg = V4RunnerConfig(
        persistence=PersistenceConfig(state_dir=tmp, enabled=True),
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


def test_runner_paper_mode_handles_hold_tick():
    runner = _new_tmp_runner()
    _warmup_runner(runner, 90)
    result = runner.on_tick(_mk(bars=200, action="HOLD"))
    assert result.intent.new_entry is None
    assert result.broker_results == []
    assert result.cockpit.action_card["kind"] == "HOLD"


def test_runner_paper_mode_places_entry_order():
    runner = _new_tmp_runner()
    _warmup_runner(runner, 90)
    result = runner.on_tick(_mk(bars=200, action="ENTER_LONG"))
    assert result.intent.new_entry is not None
    assert len(result.broker_results) == 1
    fill = result.broker_results[0]
    assert fill.state == STATE_FILLED
    assert fill.filled_quantity == 65   # 1 lot × 65 lot_size (default lots=1 here)


def test_runner_paper_mode_places_exit_order():
    runner = _new_tmp_runner()
    _warmup_runner(runner, 90)
    runner.on_tick(_mk(bars=200, action="ENTER_LONG"))
    result = runner.on_tick(_mk(bars=201, action="EXIT"))
    # Exit should produce a SELL.
    assert result.broker_results
    sell = result.broker_results[0]
    assert sell.state == STATE_FILLED


def test_runner_persists_state_to_disk():
    runner = _new_tmp_runner()
    _warmup_runner(runner, 10)
    runner.on_tick(_mk(bars=110, action="HOLD"))
    snap_path = runner.persistence.path_for_today()
    assert snap_path.exists()
    with open(snap_path) as f:
        data = json.load(f)
    assert "session_date" in data
    assert "daily_pnl_rupees" in data


def test_runner_restore_on_restart_keeps_daily_pnl():
    """Build runner, place a closing trade with realized P&L, snapshot,
    then build a new runner pointing at the same state dir — daily_pnl
    should be preserved."""
    tmp = Path(tempfile.mkdtemp(prefix="v4runner_test_"))
    cfg = V4RunnerConfig(
        persistence=PersistenceConfig(state_dir=tmp, enabled=True),
    )
    runner1 = V4Runner(cfg=cfg, broker=PaperBrokerAdapter())
    _warmup_runner(runner1, 90)
    runner1.on_tick(_mk(bars=200, action="ENTER_LONG"))
    runner1.on_tick(_mk(bars=201, action="EXIT"))
    pnl1 = runner1.manager.daily_pnl_rupees

    # Restart.
    runner2 = V4Runner(cfg=cfg, broker=PaperBrokerAdapter())
    assert runner2.manager.daily_pnl_rupees == pnl1


def test_runner_paper_kite_factory():
    """Smoke test: the .paper() factory builds a working runner."""
    runner = V4Runner.paper()
    # The runner should function without external state dir on default.
    assert isinstance(runner.broker, PaperBrokerAdapter)


def test_runner_cockpit_contains_all_panels():
    runner = _new_tmp_runner()
    _warmup_runner(runner, 90)
    result = runner.on_tick(_mk(bars=200, action="HOLD"))
    c = result.cockpit
    assert c.action_card
    assert c.web_panel is not None
    assert c.mm_panel is not None
    assert c.fat_tail_dial is not None
    assert c.risk_panel is not None
    assert "Tick @ bar" in c.explainer_text


def test_runner_jsonl_emit():
    tmp = Path(tempfile.mkdtemp(prefix="v4runner_test_"))
    jsonl = tmp / "out" / "cockpit.jsonl"
    cfg = V4RunnerConfig(
        persistence=PersistenceConfig(state_dir=tmp, enabled=True),
        write_cockpit_to_jsonl=jsonl,
    )
    runner = V4Runner(cfg=cfg, broker=PaperBrokerAdapter())
    _warmup_runner(runner, 5)
    runner.on_tick(_mk(bars=110, action="HOLD"))
    assert jsonl.exists()
    lines = jsonl.read_text().strip().split("\n")
    assert len(lines) >= 1
    for line in lines:
        json.loads(line)  # valid JSON


def test_runner_result_serializable():
    runner = _new_tmp_runner()
    _warmup_runner(runner, 90)
    result = runner.on_tick(_mk(bars=200, action="ENTER_LONG"))
    json.dumps(result.to_dict(), default=str)
