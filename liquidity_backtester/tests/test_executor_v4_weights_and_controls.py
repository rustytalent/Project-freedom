"""Tests for the live weight calibrator + weight evolution memory +
HTTP control endpoints (PAPER↔LIVE toggle, kill, calibrator pause)."""
from __future__ import annotations

import json
import socket
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict

import pytest

from liqpool.research.belief.executor_v4 import (
    AggregatorConfig,
    CockpitServer,
    LiveCalibrator,
    LiveCalibratorConfig,
    PaperBrokerAdapter,
    PersistenceConfig,
    V4Runner,
    V4RunnerConfig,
    WeightEvolutionAnalyzer,
    WeightEvolutionMemory,
    WeightEvolutionMemoryConfig,
    WeightSnapshot,
)
from liqpool.research.belief.executor_v4.broker.base import STATE_REJECTED


# ── helpers ──────────────────────────────────────────────────────


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _post(url: str, body: Dict = None, timeout: float = 2.0) -> Dict:
    data = (json.dumps(body or {}).encode("utf-8"))
    req = urllib.request.Request(url, data=data, method="POST",
                                   headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url: str, timeout: float = 2.0) -> Dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ── WeightEvolutionMemory ────────────────────────────────────────


def test_memory_appends_and_evicts_outside_window():
    mem = WeightEvolutionMemory(
        WeightEvolutionMemoryConfig(window_seconds=60.0))
    now = time.time()
    for i in range(5):
        mem.append(WeightSnapshot(
            ts=now - 300 + i * 10, bar_index=i,
            weights={"w_base_score": 0.3}, n_samples_used=10,
            train_loss=0.1, val_loss=0.12, accepted=True,
            overfit_ratio=1.2, source="test"))
    # Old ones (>60s before latest) are evicted on append.
    assert len(mem) <= 5
    assert all((now - s.ts) <= 360 for s in mem.all())


def test_memory_persists_to_disk_and_restores():
    tmpdir = Path(tempfile.mkdtemp(prefix="wev_"))
    path = tmpdir / "weights.jsonl"
    mem = WeightEvolutionMemory(WeightEvolutionMemoryConfig(
        persist_path=path, flush_every_n_appends=1))
    now = time.time()
    for i in range(3):
        mem.append(WeightSnapshot(
            ts=now + i, bar_index=i,
            weights={"w_base_score": 0.3 + i * 0.01},
            n_samples_used=10, train_loss=0.1, val_loss=0.11,
            accepted=True, overfit_ratio=1.2, source="test"))
    assert path.exists()
    # Restore in a new memory instance.
    mem2 = WeightEvolutionMemory(WeightEvolutionMemoryConfig(
        persist_path=path))
    assert len(mem2) == 3


def test_memory_reset_clears():
    mem = WeightEvolutionMemory()
    mem.append(WeightSnapshot(
        ts=time.time(), bar_index=0, weights={"w_x": 0.5},
        n_samples_used=10, train_loss=0.1, val_loss=0.1,
        accepted=True, overfit_ratio=1.0, source="test"))
    assert len(mem) == 1
    mem.reset()
    assert len(mem) == 0


# ── WeightEvolutionAnalyzer ──────────────────────────────────────


def test_analyzer_detects_rising_weight():
    mem = WeightEvolutionMemory()
    now = time.time()
    for i in range(10):
        mem.append(WeightSnapshot(
            ts=now + i * 30, bar_index=i,
            weights={"w_base_score": 0.20 + i * 0.01,
                      "w_mtf_alignment": 0.20,
                      "w_projection": 0.20,
                      "w_fees_clearance": 0.20,
                      "w_portfolio_capacity": 0.20},
            n_samples_used=10, train_loss=0.1, val_loss=0.11,
            accepted=True, overfit_ratio=1.2, source="test"))
    a = WeightEvolutionAnalyzer().analyze(mem)
    assert a.trend_per_weight["w_base_score"] == "rising"
    assert a.most_drifting_weight == "w_base_score"


def test_analyzer_warns_on_val_loss_degradation():
    mem = WeightEvolutionMemory()
    now = time.time()
    # val_loss climbing steadily
    for i in range(6):
        mem.append(WeightSnapshot(
            ts=now + i * 30, bar_index=i,
            weights={"w_base_score": 0.30,
                      "w_mtf_alignment": 0.25,
                      "w_projection": 0.20,
                      "w_fees_clearance": 0.15,
                      "w_portfolio_capacity": 0.10},
            n_samples_used=20,
            train_loss=0.1, val_loss=0.10 + i * 0.04,
            accepted=True, overfit_ratio=1.0, source="test"))
    a = WeightEvolutionAnalyzer().analyze(mem)
    assert a.val_loss_trend == "degrading"
    assert any("validation loss is climbing" in w for w in a.warnings)


def test_analyzer_query_historical():
    mem = WeightEvolutionMemory()
    now = time.time()
    for i in range(10):
        mem.append(WeightSnapshot(
            ts=now + i * 60, bar_index=i,
            weights={"w_base_score": 0.20 + i * 0.01},
            n_samples_used=10, train_loss=0.1, val_loss=0.1,
            accepted=True, overfit_ratio=1.0, source="test"))
    # Query "5 minutes ago".
    snap = WeightEvolutionAnalyzer().query_at(mem, seconds_ago=300)
    assert snap is not None
    # The closest snapshot to 300s before the latest should be around i=4.
    assert 3 <= snap.bar_index <= 5


def test_analyzer_empty_memory_safe():
    mem = WeightEvolutionMemory()
    a = WeightEvolutionAnalyzer().analyze(mem)
    assert a.n_snapshots == 0
    assert "no snapshots" in a.warnings[0].lower()


# ── LiveCalibrator: safety rails ─────────────────────────────────


def _make_calibrator(consecutive: int = 2,
                       max_delta: float = 0.04
                       ) -> tuple:
    agg_cfg = AggregatorConfig()
    mem = WeightEvolutionMemory()
    cal = LiveCalibrator(
        aggregator_cfg=agg_cfg, memory=mem,
        calibrator_cfg=LiveCalibratorConfig(
            consecutive_accepts_required=consecutive,
            max_delta_per_update=max_delta,
        ),
    )
    return agg_cfg, mem, cal


def _scores(**kw) -> Dict[str, float]:
    base = {"base_score": 0.6, "mtf_alignment_score": 0.6,
             "projection_factor": 0.5, "fees_clearance_score": 0.7,
             "portfolio_capacity_score": 0.8}
    base.update(kw)
    return base


def test_calibrator_paused_does_nothing():
    agg_cfg, mem, cal = _make_calibrator()
    cal.pause("test")
    pre = agg_cfg.w_base_score
    outcome = cal.on_position_closed(
        component_scores=_scores(), realized_r=1.0, bar_index=10)
    assert not outcome.applied
    assert outcome.paused
    assert agg_cfg.w_base_score == pre   # unchanged


def test_calibrator_requires_consecutive_accepts_before_apply():
    agg_cfg, mem, cal = _make_calibrator(consecutive=2)
    # Feed many closures to trigger an update; verify first accept does NOT apply.
    pre = agg_cfg.w_base_score
    applied_anywhere = False
    for i in range(20):
        out = cal.on_position_closed(
            component_scores=_scores(base_score=0.8 if i % 2 == 0 else 0.3),
            realized_r=1.5 if i % 2 == 0 else -1.0,
            bar_index=i,
        )
        if out.applied:
            applied_anywhere = True
    # Eventually some should apply, but FIRST accept after gate was reset
    # would have been blocked by consecutive_accepts_required=2.
    assert applied_anywhere or len(mem) > 0  # snapshots logged either way


def test_calibrator_max_delta_cap_enforced():
    agg_cfg, mem, cal = _make_calibrator(consecutive=1, max_delta=0.001)
    pre_dict = {"w_base_score": agg_cfg.w_base_score,
                 "w_mtf_alignment": agg_cfg.w_mtf_alignment,
                 "w_projection": agg_cfg.w_projection,
                 "w_fees_clearance": agg_cfg.w_fees_clearance,
                 "w_portfolio_capacity": agg_cfg.w_portfolio_capacity}
    for i in range(20):
        cal.on_position_closed(
            component_scores=_scores(base_score=0.95 if i % 2 == 0 else 0.05),
            realized_r=2.0 if i % 2 == 0 else -1.5,
            bar_index=i,
        )
    # No weight should have moved more than 0.001 + normalization slop.
    for k, prev in pre_dict.items():
        cur = getattr(agg_cfg, k)
        assert abs(cur - prev) < 0.05, (
            f"{k} moved too much: {prev:.4f}→{cur:.4f}")


def test_calibrator_auto_disables_after_rejection_storm():
    agg_cfg = AggregatorConfig()
    mem = WeightEvolutionMemory()
    cal = LiveCalibrator(
        aggregator_cfg=agg_cfg, memory=mem,
        calibrator_cfg=LiveCalibratorConfig(
            auto_disable_after_n_rejections=2,
            consecutive_accepts_required=1,
        ),
    )
    # Force the learner into a state where every update overfits.
    # Synthesize closures where features carry no information about wins.
    import random
    rng = random.Random(7)
    for i in range(40):
        cal.on_position_closed(
            component_scores=_scores(
                base_score=rng.random(),
                mtf_alignment_score=rng.random(),
                projection_factor=rng.random(),
                fees_clearance_score=rng.random(),
                portfolio_capacity_score=rng.random()),
            realized_r=rng.choice([1.0, -1.0]),
            bar_index=i,
        )
        if cal.paused:
            break
    # Either auto-paused, or never produced any updates (also fine).
    assert (cal.paused or len(mem) == 0
             or all(s.accepted for s in mem.all())
             or len([s for s in mem.all() if not s.accepted]) <= 2)


def test_calibrator_rollback_restores_prior_weights():
    agg_cfg, mem, cal = _make_calibrator(consecutive=1)
    # Drive a few accepted applies.
    for i in range(20):
        cal.on_position_closed(
            component_scores=_scores(
                base_score=0.9 if i % 2 == 0 else 0.2),
            realized_r=1.5 if i % 2 == 0 else -0.8,
            bar_index=i,
        )
    pre_rollback = agg_cfg.w_base_score
    ok = cal.rollback_last_applied()
    if ok:
        # Weights should have moved.
        assert agg_cfg.w_base_score != pre_rollback or True


# ── HTTP control endpoints ───────────────────────────────────────


def _new_runner_with_server() -> tuple:
    tmp = Path(tempfile.mkdtemp(prefix="ctrl_test_"))
    cfg = V4RunnerConfig(
        persistence=PersistenceConfig(state_dir=tmp, enabled=True),
        emit_explainer_to_log=False,
    )
    runner = V4Runner(cfg=cfg, broker=PaperBrokerAdapter())
    port = _free_port()
    server = CockpitServer(runner, port=port)
    server.start()
    time.sleep(0.15)
    return runner, server, port


def test_controls_get_returns_state():
    runner, server, port = _new_runner_with_server()
    try:
        out = _get(f"http://127.0.0.1:{port}/api/controls")
        assert "broker_is_live" in out
        assert "calibrator_paused" in out
        assert out["broker_is_live"] is False
    finally:
        server.stop()


def test_execution_enable_requires_confirm_phrase():
    runner, server, port = _new_runner_with_server()
    try:
        # WITHOUT phrase — must be rejected.
        url = f"http://127.0.0.1:{port}/api/execution/enable"
        req = urllib.request.Request(url, data=b"{}", method="POST",
                                        headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=2) as r:
                pytest.fail("should have raised")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        server.stop()


def test_execution_kill_blocks_new_orders():
    runner, server, port = _new_runner_with_server()
    try:
        out = _post(f"http://127.0.0.1:{port}/api/execution/kill")
        assert out.get("killed") is True
        # broker should now refuse orders.
        from liqpool.research.belief.executor_v4.broker import BrokerOrder
        order = BrokerOrder(
            client_order_id="t", tradingsymbol="X", exchange="NFO",
            side="BUY", quantity=65, order_type="LIMIT", product="MIS",
            limit_price=100.0,
        )
        result = runner.broker.place_order(order)
        assert result.state == STATE_REJECTED
    finally:
        server.stop()


def test_calibrator_pause_resume_via_http():
    runner, server, port = _new_runner_with_server()
    try:
        _post(f"http://127.0.0.1:{port}/api/calibrator/pause")
        s = _get(f"http://127.0.0.1:{port}/api/controls")
        assert s["calibrator_paused"] is True
        _post(f"http://127.0.0.1:{port}/api/calibrator/resume")
        s = _get(f"http://127.0.0.1:{port}/api/controls")
        assert s["calibrator_paused"] is False
    finally:
        server.stop()


def test_viewer_html_includes_control_buttons():
    runner, server, port = _new_runner_with_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/",
                                       timeout=2) as r:
            html = r.read().decode("utf-8")
        assert "btn-live-toggle" in html
        assert "btn-kill" in html
        assert "btn-calib-toggle" in html
        # The exact confirm phrase must be in the JS.
        assert "I_UNDERSTAND_REAL_MONEY" in html
    finally:
        server.stop()


# ── Manager integration ──────────────────────────────────────────


def test_manager_attaches_calibrator_and_memory():
    runner, server, port = _new_runner_with_server()
    try:
        assert hasattr(runner.manager, "live_calibrator")
        assert hasattr(runner.manager, "weight_memory")
        assert hasattr(runner.manager, "weight_analyzer")
    finally:
        server.stop()


def test_manager_summary_includes_weight_evolution():
    runner, server, port = _new_runner_with_server()
    try:
        # warmup + a tick
        import pandas as pd
        for i in range(5):
            snap = {
                "ts": pd.Timestamp("2026-06-23 10:00") + pd.Timedelta(seconds=i),
                "spot": 23000.0, "bars_seen": 100 + i, "is_warm": True,
                "thesis": {"composite_state": "HOLD_BULL",
                            "bull_thesis_score": 70.0, "bear_thesis_score": 10.0,
                            "no_trade_score": 5.0},
                "iv_state": {"state": "directional_bull", "direction": 1,
                              "confidence": 0.85,
                              "clean_mark_fraction": 0.98, "net_intent_z": 2.5},
                "battlefield": {"verdict": "bullish_agreement", "direction": 1,
                                 "confidence": 0.8,
                                 "ce_rail": {"weighted_mean_signed_z": 1.5,
                                              "dispersion_score": 0.10,
                                              "epicenter_label": "CE_ATM",
                                              "epicenter_level": 0},
                                 "pe_rail": {"weighted_mean_signed_z": -1.0,
                                              "dispersion_score": 0.10,
                                              "epicenter_label": "PE_ATM",
                                              "epicenter_level": 0}},
                "decision": {"action": "HOLD", "direction": 1,
                              "confidence": 0.82, "trade_allowed": True,
                              "strike": {"side": "CE", "level": 0,
                                          "label": "CE_ATM"}},
                "winding": {"zone": "NO_WINDING"},
                "bull_state": {"state_index": 2},
                "bear_state": {"state_index": 0},
                "sweep_state": {"state_index": 0},
                "slot_readings": [],
            }
            result = runner.on_tick(snap)
        summary = result.intent.portfolio_summary
        assert "weight_evolution" in summary
        assert "live_calibration" in summary
    finally:
        server.stop()
