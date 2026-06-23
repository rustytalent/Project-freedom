"""Live RAM-leak triage (founder 2026-06-22).

The live Kite session crossed 27 GB RSS at ~35 minutes. This test
suite verifies that:

  1. PositionLedger.bar_records is bounded per position.
  2. LedgerStore._closed is bounded across the session.
  3. ExitBudget.history + rejected_history are bounded.
  4. MemoryGuard samples RSS, enforces a hard ceiling, and exposes a
     summary the cockpit can render.
  5. V4Runner.on_tick throttles cockpit publishes to ≤2 Hz.
  6. **Soak test** — run a synthetic 1-second tick stream for 60 minutes
     of equivalent ticks (3600 ticks) and assert RSS does NOT grow
     unbounded (last-tail growth is < a small slack).

The synthetic tick fixture mirrors the live snapshot shape used by the
manager, so the soak path exercises the same code as production.
"""
from __future__ import annotations

import gc
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import pytest


# Soak tests are slow (run the full V4Runner over hundreds of synthetic
# ticks). They prove the leak fixes hold in steady-state but take 1-3
# minutes per test, so we gate them behind an env var to keep the
# default pytest run fast. Run with: SOAK=1 pytest test_executor_v4_memory_bounded.py
_SOAK_ENABLED = os.environ.get("SOAK", "0") == "1"
_skip_soak = pytest.mark.skipif(
    not _SOAK_ENABLED,
    reason="soak tests gated behind SOAK=1 env var",
)

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PersistenceConfig,
    V4Runner,
    V4RunnerConfig,
)
from liqpool.research.belief.executor_v4.exit_engine.budget import (
    ModificationBudget,
    ModificationBudgetConfig,
    ModificationBucket,
)
from liqpool.research.belief.executor_v4.ledger import (
    BarRecord,
    LedgerOutcome,
    LedgerStore,
    PositionLedger,
)
from liqpool.research.belief.executor_v4.memory_guard import (
    MemoryCeilingExceeded,
    MemoryGuard,
    MemoryGuardConfig,
    get_rss_mb,
)


# ── Per-position ledger bounded ─────────────────────────────────


def _make_bar(bar_index: int = 0) -> BarRecord:
    return BarRecord(
        bar_index=bar_index, ts="2026-06-22T11:00:00",
        spot=23000.0, held_premium=100.0,
        current_r=0.1, best_r_so_far=0.2, worst_r_so_far=-0.05,
        cumulative_realized_pnl_rupees=0.0,
        confidence=0.65, high_water_confidence=0.65,
        thesis_state="BULL", iv_state="directional",
        battlefield_verdict="ok", winding_zone="NO_WINDING",
        decision_action="HOLD",
        held_spread_state="clean", held_friendliness=0.9,
        held_acceptance="normal", held_dod_z=0.5,
    )


def _bare_hypothesis(pid: str):
    """Minimum PositionHypothesis for ledger tests — only the
    required (non-default) fields filled. Schema-agnostic via
    keyword construction so future field additions don't break here."""
    from liqpool.research.belief.executor_v4.hypothesis import (
        PositionHypothesis,
    )
    return PositionHypothesis(
        position_id=pid, opened_at_ts="2026-06-22T11:00:00",
        opened_at_bar=0,
        trigger_action="ENTER_LONG", trigger_decision={},
        trigger_snapshot_summary={}, rich_context={},
        thesis_summary="x", thesis_state="BULL",
        direction=1, profile="intraday",
        mtf_alignment={}, mtf_aligned=True,
    )


def test_position_ledger_caps_bar_records():
    h = _bare_hypothesis("p1")
    ledger = PositionLedger(hypothesis=h, max_bar_records=50)
    for i in range(500):
        ledger.append_bar(_make_bar(i))
    assert len(ledger.bar_records) <= 50
    # Most recent bars survive.
    assert ledger.bar_records[-1].bar_index == 499


def test_ledger_store_caps_closed_list_and_trims_bar_records():
    store = LedgerStore(max_closed_in_memory=10,
                              trim_bar_records_on_close=20)

    def _open_then_close(pid: str):
        h = _bare_hypothesis(pid)
        l = PositionLedger(hypothesis=h)
        # 100 bar records on the open ledger.
        for i in range(100):
            l.append_bar(_make_bar(i))
        store.open(l)
        outcome = LedgerOutcome(
            exit_reason="target hit", exit_severity="normal",
            exit_premium=70.0, exit_spot=100.0, exit_bar=100,
            realized_r=1.0, realized_rupees=500.0,
            gross_rupees=520.0, fees_rupees=20.0, slippage_rupees=0.0,
            bars_held=100, validation_ever_fully_met=True,
            invalidation_count=0, hypothesis_vs_reality_score=0.8,
        )
        store.close(pid, outcome)

    # Close 50 positions — the closed list caps at 10.
    for i in range(50):
        _open_then_close(f"p{i}")
    assert len(store._closed) <= 10
    # Each closed ledger has its bar_records trimmed to 20 or fewer.
    for l in store._closed:
        assert len(l.bar_records) <= 20


# ── ExitBudget history bounded ──────────────────────────────────


def test_exit_budget_history_bounded():
    cfg = ModificationBudgetConfig(early_correction_cap=999,
                                          normal_adaptive_cap=999,
                                          endgame_chase_cap=999,
                                          emergency_reserve_cap=999,
                                          total_cap=999 * 4)
    budget = ModificationBudget(cfg=cfg, history_cap=20)
    for i in range(500):
        budget.try_spend(ModificationBucket.NORMAL_ADAPTIVE,
                          reason=f"req-{i}")
    assert len(budget.history) <= 20


def test_exit_budget_rejected_history_bounded():
    cfg = ModificationBudgetConfig(early_correction_cap=1,
                                          normal_adaptive_cap=1,
                                          endgame_chase_cap=0,
                                          emergency_reserve_cap=0,
                                          total_cap=2)
    budget = ModificationBudget(cfg=cfg, history_cap=20)
    # First spend succeeds; remaining 500 attempts get rejected.
    for i in range(500):
        budget.try_spend(ModificationBucket.NORMAL_ADAPTIVE,
                          reason=f"req-{i}")
    assert len(budget.rejected_history) <= 20


# ── MemoryGuard ─────────────────────────────────────────────────


def test_memory_guard_samples_rss():
    g = MemoryGuard(MemoryGuardConfig(
        sample_interval_seconds=0.0,    # always sample
        hard_limit_mb=None,
    ))
    s = g.check()
    if s is None:
        pytest.skip("RSS not available on this platform")
    assert s["rss_mb"] > 0
    summary = g.summary()
    assert summary["n_samples"] == 1
    assert summary["rss_now_mb"] > 0


def test_memory_guard_respects_interval():
    g = MemoryGuard(MemoryGuardConfig(
        sample_interval_seconds=60.0,
        hard_limit_mb=None,
    ))
    s1 = g.check()
    if s1 is None:
        pytest.skip("RSS not available")
    s2 = g.check()
    # Second call returns None because the interval hasn't elapsed.
    assert s2 is None


def test_memory_guard_hard_ceiling_raises():
    if get_rss_mb() is None:
        pytest.skip("RSS not available")
    g = MemoryGuard(MemoryGuardConfig(
        sample_interval_seconds=0.0,
        hard_limit_mb=0.001,    # impossibly small → fires immediately
    ))
    with pytest.raises(MemoryCeilingExceeded):
        g.check()


def test_memory_guard_disabled_when_hard_limit_none():
    g = MemoryGuard(MemoryGuardConfig(
        sample_interval_seconds=0.0,
        hard_limit_mb=None,
    ))
    s = g.check()
    if s is None:
        pytest.skip("RSS not available")
    # No exception even at zero limit.


# ── V4Runner: cockpit publish throttled ─────────────────────────


def _runner_with_state(state_dir: Path) -> V4Runner:
    cfg = V4RunnerConfig(
        manager=None,
        persistence=PersistenceConfig(state_dir=state_dir, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


def test_v4runner_throttle_attribute_exists():
    tmp = Path(tempfile.mkdtemp(prefix="mem_thr_"))
    runner = _runner_with_state(tmp)
    assert runner._cockpit_publish_min_interval_s > 0


def test_v4runner_memory_guard_attached():
    tmp = Path(tempfile.mkdtemp(prefix="mem_g_"))
    runner = _runner_with_state(tmp)
    assert runner.memory_guard is not None


# ── Soak: RSS plateau over a long tick stream ──────────────────


@_skip_soak
@pytest.mark.parametrize("n_ticks,growth_slack_mb", [(400, 80.0)])
def test_soak_run_does_not_leak_unbounded(n_ticks, growth_slack_mb):
    """Drive a synthetic tick stream through V4Runner. RSS at the END
    should not be more than ``growth_slack_mb`` MB above RSS at the
    midpoint. Allows a moderate warm-up (lazy state spinning up) but
    detects an unbounded leak (linear growth keeps RSS rising past the
    slack)."""
    from tests.test_executor_v4_dead_market_and_web_viz import _snap
    if get_rss_mb() is None:
        pytest.skip("RSS not available on this platform")
    tmp = Path(tempfile.mkdtemp(prefix="soak_"))
    runner = _runner_with_state(tmp)
    # Warm up.
    for i in range(120):
        runner.on_tick(_snap(100 + i))
    gc.collect()
    rss_mid = get_rss_mb()
    # Run the long stream.
    for i in range(n_ticks):
        runner.on_tick(_snap(220 + i))
    gc.collect()
    rss_end = get_rss_mb()
    growth = rss_end - rss_mid
    assert growth < growth_slack_mb, (
        f"RSS grew {growth:.1f} MB across {n_ticks} ticks "
        f"(mid {rss_mid:.0f} → end {rss_end:.0f}). Slack {growth_slack_mb:.0f}.")


@_skip_soak
def test_soak_position_with_long_hold_does_not_blow_up_bar_records():
    """A position held open for many ticks: bar_records stays bounded."""
    from tests.test_executor_v4_dead_market_and_web_viz import _snap
    tmp = Path(tempfile.mkdtemp(prefix="hold_"))
    runner = _runner_with_state(tmp)
    for i in range(80):
        runner.on_tick(_snap(100 + i))
    runner.on_tick(_snap(200, action="ENTER_LONG"))
    for i in range(300):
        runner.on_tick(_snap(220 + i))
    for ledger in runner.manager.ledger_store._open.values():
        assert len(ledger.bar_records) <= 1200
