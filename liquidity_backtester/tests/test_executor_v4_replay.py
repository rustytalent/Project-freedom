"""Tests for executor_v4.replay — backtest harness over snapshot tapes."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd

from liqpool.research.belief.executor_v4 import (
    ReplayConfig,
    ReplayReport,
    replay_jsonl,
    replay_snapshots,
)


def _mk(*, bars: int = 100, action: str = "HOLD", direction: int = 1,
        confidence: float = 0.82, spot: float = 23000.0,
        thesis: str = "HOLD_BULL", iv: str = "directional_bull",
        bf: str = "bullish_agreement") -> Dict:
    ts = pd.Timestamp("2026-06-19 10:00") + pd.Timedelta(seconds=bars)
    slot_readings = []
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "CE", "level": level,
            "label": f"CE_{level}", "acceptance": "normal",
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": 1.5, "mark_price": 100.0,
        })
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "PE", "level": level,
            "label": f"PE_{level}", "acceptance": "normal",
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": -1.0, "mark_price": 100.0,
        })
    return {
        "ts": str(ts), "spot": spot, "bars_seen": bars, "is_warm": True,
        "thesis": {"composite_state": thesis,
                   "bull_thesis_score": 70.0, "bear_thesis_score": 10.0,
                   "no_trade_score": 5.0},
        "iv_state": {"state": iv, "direction": direction, "confidence": 0.85,
                     "clean_mark_fraction": 0.98, "net_intent_z": 2.5},
        "battlefield": {
            "verdict": bf, "direction": direction, "confidence": 0.8,
            "ce_rail": {"weighted_mean_signed_z": 1.5,
                        "dispersion_score": 0.10,
                        "epicenter_label": "CE_ATM", "epicenter_level": 0},
            "pe_rail": {"weighted_mean_signed_z": -1.0,
                        "dispersion_score": 0.10,
                        "epicenter_label": "PE_ATM", "epicenter_level": 0},
        },
        "decision": {"action": action, "direction": direction,
                     "confidence": confidence, "trade_allowed": True,
                     "strike": {"side": "CE" if direction > 0 else "PE",
                                 "level": 0, "label": "CE_ATM"}},
        "winding": {"zone": "NO_WINDING"},
        "bull_state": {"state_index": 2},
        "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


def _snap_stream(n_holds: int = 95, entry_at: int = 95, exit_at: int = 105):
    """Generate a snapshot stream: holds, then one entry, then holds, then exit."""
    out = []
    for i in range(n_holds):
        out.append(_mk(bars=100 + i, action="HOLD"))
    out.append(_mk(bars=100 + entry_at, action="ENTER_LONG"))
    for i in range(entry_at + 1, exit_at):
        out.append(_mk(bars=100 + i, action="HOLD"))
    out.append(_mk(bars=100 + exit_at, action="EXIT"))
    for i in range(exit_at + 1, exit_at + 5):
        out.append(_mk(bars=100 + i, action="HOLD"))
    return out


def test_replay_handles_empty_stream():
    report = replay_snapshots([])
    assert report.n_ticks == 0
    assert report.n_entries == 0


def test_replay_runs_simple_entry_exit_cycle():
    stream = _snap_stream()
    report = replay_snapshots(stream)
    assert report.n_ticks > 0
    # Should have at least the engine EXIT close.
    assert report.n_exits >= 1


def test_replay_report_serializable():
    stream = _snap_stream()
    report = replay_snapshots(stream)
    import json
    json.dumps(report.to_dict())


def test_replay_summary_string_renders():
    stream = _snap_stream()
    report = replay_snapshots(stream)
    text = report.to_summary_string()
    assert "Replay summary" in text


def test_replay_jsonl_round_trip():
    """Write snapshots to a JSONL file, then replay from it."""
    tmpdir = Path(tempfile.mkdtemp(prefix="v4_replay_test_"))
    tape_path = tmpdir / "tape.jsonl"
    with open(tape_path, "w") as f:
        for snap in _snap_stream():
            f.write(json.dumps(snap, default=str) + "\n")
    report = replay_jsonl(tape_path)
    assert report.n_ticks > 0


def test_replay_jsonl_handles_extras_envelope():
    """If snapshots are wrapped in {'extras': {'belief_snapshot': ...}},
    the replay should still find them."""
    tmpdir = Path(tempfile.mkdtemp(prefix="v4_replay_test_"))
    tape_path = tmpdir / "tape.jsonl"
    with open(tape_path, "w") as f:
        for snap in _snap_stream()[:5]:
            wrapped = {"model": "test", "extras": {"belief_snapshot": snap}}
            f.write(json.dumps(wrapped, default=str) + "\n")
    report = replay_jsonl(tape_path)
    assert report.n_ticks >= 1


def test_replay_writes_cockpit_jsonl_when_configured():
    tmpdir = Path(tempfile.mkdtemp(prefix="v4_replay_test_"))
    cockpit_out = tmpdir / "cockpit.jsonl"
    cfg = ReplayConfig(cockpit_out=cockpit_out)
    replay_snapshots(_snap_stream()[:20], cfg=cfg)
    assert cockpit_out.exists()
    lines = cockpit_out.read_text().strip().split("\n")
    assert len(lines) >= 5
    for line in lines:
        json.loads(line)


def test_replay_tracks_strategy_usage():
    stream = _snap_stream()
    report = replay_snapshots(stream)
    if report.n_entries > 0:
        assert sum(report.strategy_usage.values()) == report.n_entries
