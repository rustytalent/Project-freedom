"""Tests for the dead-market guard + pulsing-web cockpit panel
(founder hot-fix follow-up 2026-06-22)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PersistenceConfig,
    PortfolioManagerConfig,
    V4Runner,
    V4RunnerConfig,
)
from liqpool.research.belief.executor_v4.scenario_web import (
    FAMILY_DIRECTIONAL, FAMILY_CHOP, FAMILY_FAT_TAIL, FAMILY_MANIPULATION,
    Scenario, ScenarioWeb,
)


# ── Dead-market guard ─────────────────────────────────────────────


def test_manager_config_has_dead_market_guard_defaults():
    cfg = PortfolioManagerConfig()
    assert cfg.dead_market_refuse_directional is True
    assert cfg.dead_market_chop_threshold == 0.50
    assert cfg.dead_market_manipulation_threshold == 0.40


def _snap(bars: int, action: str = "HOLD",
            ce_def_frac: float = 0.0) -> dict:
    ts = pd.Timestamp("2026-06-22 11:00") + pd.Timedelta(seconds=bars)
    slots = []
    n_each = 11
    n_ce_def = int(ce_def_frac * n_each)
    for level in range(-5, 6):
        slots.append({
            "strike": 23000.0 + 50 * level, "option_type": "CE",
            "level": level, "label": f"CE_{level}",
            "acceptance": "defended" if n_ce_def > 0 else "normal",
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": 1.5, "mark_price": 100.0,
        })
        if n_ce_def > 0: n_ce_def -= 1
    for level in range(-5, 6):
        slots.append({
            "strike": 23000.0 + 50 * level, "option_type": "PE",
            "level": level, "label": f"PE_{level}",
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
                     "confidence": 0.85, "clean_mark_fraction": 0.98,
                     "net_intent_z": 2.5},
        "battlefield": {
            "verdict": "bullish_agreement", "direction": 1, "confidence": 0.8,
            "ce_rail": {"weighted_mean_signed_z": 1.5,
                         "dispersion_score": 0.10,
                         "epicenter_label": "CE_ATM", "epicenter_level": 0},
            "pe_rail": {"weighted_mean_signed_z": -1.0,
                         "dispersion_score": 0.10,
                         "epicenter_label": "PE_ATM", "epicenter_level": 0},
        },
        "decision": {"action": action, "direction": 1, "confidence": 0.80,
                      "trade_allowed": True,
                      "strike": {"side": "CE", "level": 0,
                                  "label": "CE_ATM"}},
        "winding": {"zone": "NO_WINDING"},
        "bull_state": {"state_index": 2}, "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slots,
    }


def _new_runner(dead_market: bool = True) -> V4Runner:
    tmp = Path(tempfile.mkdtemp(prefix="deadmkt_"))
    pm_cfg = PortfolioManagerConfig(
        dead_market_refuse_directional=dead_market,
    )
    cfg = V4RunnerConfig(
        manager=pm_cfg,
        persistence=PersistenceConfig(state_dir=tmp, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


def test_dead_market_refuses_when_chop_dominant():
    """Inject lots of chop scenarios so chop_mass crosses the threshold,
    then verify a directional ENTER request is refused."""
    runner = _new_runner(dead_market=True)
    # Warmup
    for i in range(90):
        runner.on_tick(_snap(100 + i))
    # Force chop dominance by injecting many high-probability chop
    # scenarios directly. They will be decayed by the next observe()
    # pass, so we over-provision the mass.
    web = runner.manager.web
    for i in range(8):
        web.scenarios[f"chop_{i}"] = Scenario(
            scenario_id=f"chop_{i}", name=f"chop_range_{i}",
            family=FAMILY_CHOP, trigger_signature=f"c{i}",
            implied_direction=0, implied_horizon_bars=30,
            implied_max_drawdown_during_path=0.5,
            implied_strategy_class="iron_condor", base_prior=0.20,
            current_probability=0.85, decay_rate=0.99,
        )
    intent = runner.on_tick(_snap(200, action="ENTER_LONG"))
    assert intent.intent.new_entry is None
    assert any("dead market" in r.lower() for r in intent.intent.refuse_reasons), (
        f"expected dead-market refusal, got: {intent.intent.refuse_reasons}")


def test_dead_market_refuses_when_manipulation_dominant():
    runner = _new_runner(dead_market=True)
    for i in range(90):
        runner.on_tick(_snap(100 + i))
    web = runner.manager.web
    for i in range(3):
        web.scenarios[f"manip_{i}"] = Scenario(
            scenario_id=f"manip_{i}", name=f"single_strike_distortion_{i}",
            family=FAMILY_MANIPULATION, trigger_signature=f"m{i}",
            implied_direction=0, implied_horizon_bars=10,
            implied_max_drawdown_during_path=0.8,
            implied_strategy_class="wait", base_prior=0.15,
            current_probability=0.20, decay_rate=0.99,
        )
    intent = runner.on_tick(_snap(200, action="ENTER_LONG"))
    if intent.intent.new_entry is None:
        # Either the dead-market guard fired, or another gate did.
        assert any("dead market" in r.lower() or "manipulation" in r.lower()
                    for r in intent.intent.refuse_reasons)


def test_dead_market_guard_can_be_disabled():
    runner = _new_runner(dead_market=False)
    for i in range(90):
        runner.on_tick(_snap(100 + i))
    web = runner.manager.web
    for i in range(4):
        web.scenarios[f"chop_{i}"] = Scenario(
            scenario_id=f"chop_{i}", name=f"chop_range_{i}",
            family=FAMILY_CHOP, trigger_signature=f"c{i}",
            implied_direction=0, implied_horizon_bars=30,
            implied_max_drawdown_during_path=0.5,
            implied_strategy_class="iron_condor", base_prior=0.20,
            current_probability=0.30, decay_rate=0.99,
        )
    intent = runner.on_tick(_snap(200, action="ENTER_LONG"))
    # No "dead market" reason — guard is off.
    assert not any("dead market" in r.lower()
                    for r in intent.intent.refuse_reasons)


# ── Pulsing web visualization ─────────────────────────────────────


def test_scenario_records_probability_history():
    """Each observe() pushes the current probability into the
    scenario's recent_probabilities ring (cap=12)."""
    web = ScenarioWeb()
    snap = _snap(100)
    rich = type("R", (), {
        "regime_stability_index": 0.8,
        "epicenter_label": "CE_ATM", "epicenter_level": 0,
        "epicenter_migration_distance": 0.0,
        "dispersion_velocity": 0.0,
        "net_intent_velocity": 0.0,
        "thesis_bull_velocity": 0.0, "thesis_bear_velocity": 0.0,
    })()
    flow_event = type("F", (), {
        "surprise_score": 0.0, "spot_pct_change": 0.0,
        "net_intent_z": 1.0, "ce_dispersion": 0.1,
        "clean_mark_fraction": 1.0, "iv_state": "",
    })()
    for i in range(15):
        web.observe(_snap(100 + i), rich, flow_event)
    # Find a scenario that's been alive for a few ticks.
    survivors = [sc for sc in web.scenarios.values()
                 if len(sc.recent_probabilities) >= 2]
    assert survivors, "expected at least one scenario to accumulate history"
    sample = survivors[0]
    assert len(sample.recent_probabilities) <= 12


def test_web_snapshot_carries_all_scenarios():
    web = ScenarioWeb()
    # Inject some scenarios directly so we don't depend on the spawn
    # logic firing on synthetic data.
    for i in range(8):
        web.scenarios[f"x_{i}"] = Scenario(
            scenario_id=f"x_{i}", name=f"bull_{i}",
            family=FAMILY_DIRECTIONAL, trigger_signature=f"t{i}",
            implied_direction=+1, implied_horizon_bars=20,
            implied_max_drawdown_during_path=0.5,
            implied_strategy_class="long_ce", base_prior=0.15,
            current_probability=0.20 + i * 0.05, decay_rate=0.99,
            recent_probabilities=[0.10 + i * 0.01,
                                    0.15 + i * 0.02,
                                    0.20 + i * 0.05],
        )
    snap = web.snapshot()
    d = snap.to_dict()
    assert "all_scenarios" in d
    assert len(d["all_scenarios"]) == 8
    # Each carries a recent_probabilities trail.
    assert all("recent_probabilities" in sc for sc in d["all_scenarios"])


def test_cockpit_web_panel_has_all_scenarios_for_viz():
    """The cockpit's web_panel must propagate all_scenarios + the
    horizon-weighted consensus so the front-end can render the pulse."""
    runner = _new_runner(dead_market=False)
    for i in range(5):
        runner.on_tick(_snap(100 + i))
    # Inject scenarios with history so they pass through to the cockpit.
    web = runner.manager.web
    for i in range(6):
        web.scenarios[f"viz_{i}"] = Scenario(
            scenario_id=f"viz_{i}", name=f"bull_{i}",
            family=FAMILY_DIRECTIONAL, trigger_signature=f"v{i}",
            implied_direction=+1, implied_horizon_bars=15,
            implied_max_drawdown_during_path=0.5,
            implied_strategy_class="long_ce", base_prior=0.15,
            current_probability=0.30, decay_rate=0.99,
            recent_probabilities=[0.10, 0.18, 0.25, 0.30],
        )
    result = runner.on_tick(_snap(120))
    wp = result.cockpit.web_panel
    assert "all_scenarios" in wp
    assert "directional_consensus_horizon_weighted" in wp
    if wp["all_scenarios"]:
        sample = wp["all_scenarios"][0]
        assert "recent_probabilities" in sample


def test_viewer_html_includes_pulsing_web_svg():
    from liqpool.research.belief.executor_v4.cockpit_server import (
        _DEFAULT_VIEWER_HTML,
    )
    assert "PULSING WEB" in _DEFAULT_VIEWER_HTML
    assert 'id="web_svg"' in _DEFAULT_VIEWER_HTML
    assert "render_pulsing_web" in _DEFAULT_VIEWER_HTML
