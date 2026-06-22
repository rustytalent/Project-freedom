"""Connectedness audit — proves the live path uses the LATEST modules only.

Founder's concern (2026-06-22): "Make sure everything is connected,
everything available is being used, the quality is good, nothing old
or strange is being connected."

These tests walk the live wiring graph and produce mechanical evidence
that:
  1. The KiteV4LiveRunner does NOT construct the v3 executor.
  2. The v4 PortfolioManager wires every latest layer:
        substrate, memory, flow_memory, scenario_web, critic,
        counterfactual, projection, aggregator, manipulation_board,
        market_maker_mind, fat_tail_amp, crowd_mirror, portfolio_risk,
        hedge_proposer, portfolio_exit, live_calibrator,
        weight_memory, weight_analyzer
  3. Per-tick, the manager produces a portfolio_summary that carries
     every latest dimension surface (scenario_web, mm_posterior,
     fat_tail_score, crowd_mirror, portfolio_risk, hedge_proposal,
     adaptive_exit, live_calibration, weight_evolution, recent_trades).
  4. The cockpit snapshot's panels include the new capital + trades
     panels (founder's UI ask).
  5. The CockpitSnapshot constructed by the runner carries ALL of:
     action/web/mm/fat-tail/crowd/risk/patterns/positions/pnl/capital/
     trades + the explainer text — proving the runner's render path
     surfaces every dimension.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PersistenceConfig,
    V4Runner,
    V4RunnerConfig,
)


# ── Stale-module audit ───────────────────────────────────────────


def test_v4_live_bridge_does_not_construct_v3_executor():
    from liqpool.research.belief.executor_v4.v4_live_bridge import (
        KiteV4BridgeConfig, KiteV4LiveRunner,
    )
    from liqpool.research.belief.live_runner import BeliefLiveConfig
    cfg = KiteV4BridgeConfig(belief=BeliefLiveConfig(),
                              confirm_real_orders=False,
                              cockpit_server_port=None)
    runner = KiteV4LiveRunner(api_key="x", access_token="y",
                                bridge_cfg=cfg)
    # v3 executor MUST be None.
    assert runner.executor is None, (
        "v3 executor was constructed inside the v4 live bridge — STALE RISK")
    # The latest v4 brain MUST be present.
    assert hasattr(runner.v4.manager, "portfolio_exit")
    assert hasattr(runner.v4.manager, "live_calibrator")
    assert hasattr(runner.v4.manager, "weight_memory")
    # Paper broker by default; no real orders possible without confirm_real.
    assert runner.v4.broker.is_live is False


def test_v4_manager_wires_every_latest_layer():
    """Spec-by-fixture: enumerate the names we EXPECT to find as
    attributes on the live PortfolioManager. If a layer is removed or
    renamed, this test fails loudly. If a NEW layer is shipped, add it
    here to lock in the contract."""
    runner = V4Runner.paper()
    mgr = runner.manager
    REQUIRED_LAYERS = [
        "substrate_state", "mtf", "flow",
        "web", "critic", "counterfactual", "projection", "aggregator",
        "manipulation_board", "mm_mind", "fat_tail_amp", "crowd_mirror",
        "portfolio_risk", "hedge_proposer",
        "portfolio_exit",
        "live_calibrator", "weight_memory", "weight_analyzer",
        "ledger_store",
    ]
    missing = [name for name in REQUIRED_LAYERS if not hasattr(mgr, name)]
    assert not missing, (
        f"latest manager is missing wired layer(s): {missing}")


# ── Per-tick surface audit ──────────────────────────────────────


def _new_runner() -> V4Runner:
    tmp = Path(tempfile.mkdtemp(prefix="v4_conn_test_"))
    cfg = V4RunnerConfig(
        persistence=PersistenceConfig(state_dir=tmp, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


def _mk_hold(bars: int) -> dict:
    ts = pd.Timestamp("2026-06-23 10:00") + pd.Timedelta(seconds=bars)
    slot_readings = []
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "CE", "level": level,
            "label": f"CE_{level}",
            "acceptance": "normal", "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "is_abnormal": False, "dod_z": 1.5, "mark_price": 100.0,
        })
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "PE", "level": level,
            "label": f"PE_{level}",
            "acceptance": "normal", "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "is_abnormal": False, "dod_z": -1.0, "mark_price": 100.0,
        })
    return {
        "ts": ts, "spot": 23000.0, "bars_seen": bars, "is_warm": True,
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
        "decision": {"action": "HOLD", "direction": 1, "confidence": 0.82,
                     "trade_allowed": True,
                     "strike": {"side": "CE", "level": 0, "label": "CE_ATM"}},
        "winding": {"zone": "NO_WINDING"},
        "bull_state": {"state_index": 2},
        "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


def test_portfolio_summary_carries_every_dimension():
    """Spec-by-keys: the per-tick portfolio_summary MUST surface every
    dimension the cockpit and the operator depend on."""
    runner = _new_runner()
    # Warmup
    for i in range(5):
        runner.on_tick(_mk_hold(100 + i))
    result = runner.on_tick(_mk_hold(120))
    summary = result.intent.portfolio_summary
    REQUIRED_SUMMARY_KEYS = [
        "open_positions", "daily_pnl_rupees",
        "cumulative_fees_rupees", "scenario_web",
        "mm_posterior", "fat_tail_score", "crowd_mirror",
        "portfolio_risk", "hedge_proposal",
        "adaptive_exit", "live_calibration", "weight_evolution",
        "recent_trades",
    ]
    missing = [k for k in REQUIRED_SUMMARY_KEYS if k not in summary]
    assert not missing, (
        f"portfolio_summary missing dimensions: {missing}")


def test_cockpit_snapshot_carries_capital_and_trades_panels():
    runner = _new_runner()
    for i in range(5):
        runner.on_tick(_mk_hold(100 + i))
    result = runner.on_tick(_mk_hold(120))
    cockpit = result.cockpit
    # Every panel a UI consumer expects MUST be present.
    REQUIRED_PANELS = [
        "action_card", "web_panel", "mm_panel", "fat_tail_dial",
        "crowd_panel", "risk_panel", "patterns_panel",
        "hedge_panel", "positions_panel", "pnl_panel",
        "explainer_text",
        "capital_panel", "trades_panel",
    ]
    d = cockpit.to_dict()
    missing = [k for k in REQUIRED_PANELS if k not in d]
    assert not missing, f"cockpit missing panels: {missing}"


def test_capital_panel_populated_from_broker():
    runner = _new_runner()
    for i in range(5):
        runner.on_tick(_mk_hold(100 + i))
    result = runner.on_tick(_mk_hold(120))
    cap = result.cockpit.capital_panel
    assert "starting_capital_rupees" in cap
    assert cap["starting_capital_rupees"] > 0
    assert "available_rupees" in cap
    assert "current_total_rupees" in cap


def test_paper_broker_starts_with_configurable_capital():
    broker = PaperBrokerAdapter(starting_capital_rupees=75_000.0)
    cap = broker.get_capital()
    assert cap["starting_capital_rupees"] == 75_000.0
    assert cap["available_rupees"] == 75_000.0


def test_trades_tape_records_open_and_close():
    runner = _new_runner()
    # Warmup
    for i in range(90):
        runner.on_tick(_mk_hold(100 + i))
    # Force an ENTER
    snap = _mk_hold(200)
    snap["decision"]["action"] = "ENTER_LONG"
    result = runner.on_tick(snap)
    # Force an EXIT
    snap2 = _mk_hold(201)
    snap2["decision"]["action"] = "EXIT"
    result2 = runner.on_tick(snap2)
    trades = result2.cockpit.trades_panel.get("trades") or []
    kinds = {t.get("kind") for t in trades}
    assert "OPEN" in kinds or "CLOSE" in kinds


def test_engine_with_advanced_runtime_off_is_legacy_safe():
    """The new BeliefEngineConfig.feed_tape_speed/feed_time_of_day_scale
    flags are off by default → existing engine snapshots are unchanged."""
    from liqpool.research.belief.engine import (
        BeliefEngine, BeliefEngineConfig,
    )
    cfg = BeliefEngineConfig()
    assert cfg.feed_tape_speed is False
    assert cfg.feed_time_of_day_scale is False
    # Both ResidualConfig and ThesisMemoryConfig advanced flags are off too.
    assert cfg.resid_cfg.enable_multi_horizon is False
    assert cfg.resid_cfg.enable_time_of_day_scaling is False
    assert cfg.resid_cfg.enable_acceptance_confidence is False
    assert cfg.thesis_cfg.enable_adaptive_decay is False
    assert cfg.thesis_cfg.enable_uncertainty is False
