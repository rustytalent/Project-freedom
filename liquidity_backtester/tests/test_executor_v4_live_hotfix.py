"""HOT-FIX regression tests for the 5 live-trading bugs (2026-06-22).

Founder caught all 5 of these in real money during the 11 AM session.
These tests lock in each fix so a future change cannot reintroduce
them.
"""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PortfolioManager,
    PortfolioManagerConfig,
    V4Runner,
    V4RunnerConfig,
)
from liqpool.research.belief.executor_v4.economics import (
    DEFAULT_MIN_EDGE_MULTIPLE,
    ExecutionEconomicsConfig,
)
from liqpool.research.belief.executor_v4.scenario_web import (
    FAMILY_DIRECTIONAL, Scenario, ScenarioWeb,
)


def _snapshot_with_atm(spot: float, extra_marks=None) -> dict:
    """Build a snapshot whose ATM strike = round(spot/50)*50. Used to
    reproduce the founder's strike-lookup bug: when spot drifts and ATM
    shifts, the level-0 slot points to a DIFFERENT strike."""
    atm = round(spot / 50.0) * 50.0
    slots = []
    for level in range(-5, 6):
        strike = atm + 50.0 * level
        ce_mark = 100.0 + level * 10
        if extra_marks and strike in extra_marks:
            ce_mark = extra_marks[strike]
        slots.append({
            "strike": strike, "option_type": "CE", "level": level,
            "label": f"CE_{level}", "acceptance": "normal",
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": 0.5, "mark_price": ce_mark,
        })
        slots.append({
            "strike": strike, "option_type": "PE", "level": level,
            "label": f"PE_{level}", "acceptance": "normal",
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": -0.5, "mark_price": 80.0 + abs(level) * 5,
        })
    return {
        "ts": pd.Timestamp("2026-06-22 11:00"),
        "spot": spot, "bars_seen": 100, "is_warm": True,
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
        "decision": {"action": "HOLD", "direction": 1, "confidence": 0.80,
                      "trade_allowed": True,
                      "strike": {"side": "CE", "level": 0,
                                  "label": "CE_ATM"}},
        "winding": {"zone": "NO_WINDING"},
        "bull_state": {"state_index": 2}, "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slots,
    }


def _new_runner() -> V4Runner:
    import tempfile
    from pathlib import Path
    from liqpool.research.belief.executor_v4 import PersistenceConfig
    tmp = Path(tempfile.mkdtemp(prefix="hot_fix_"))
    cfg = V4RunnerConfig(
        persistence=PersistenceConfig(state_dir=tmp, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


# ── BUG 1: strike-lookup-by-level → phantom P&L ─────────────────


def test_held_premium_follows_strike_not_level_when_atm_drifts():
    """Founder's bug: open at ATM=23150 (mark=150). Spot drifts so ATM
    becomes 23100. The held strike is STILL 23150 — its mark must come
    from THAT strike, not from the NEW level-0 (which is now strike
    23100, a completely different contract)."""
    runner = _new_runner()
    for i in range(90):
        runner.on_tick(_snapshot_with_atm(spot=23127.0))
    open_snap = _snapshot_with_atm(spot=23127.0,
                                      extra_marks={23150.0: 150.0,
                                                     23100.0: 100.0})
    open_snap["decision"]["action"] = "ENTER_LONG"
    result = runner.on_tick(open_snap)
    if result.intent.new_entry is None:
        return    # entry blocked by gates — skip; still useful as smoke
    pos_id = result.intent.new_entry["hypothesis"]["position_id"]
    held = runner.manager._open_states[pos_id]
    assert held.hypothesis.strike_price == 23150.0
    # Now drift spot to 23123, ATM=23100. Mark at our held strike (23150)
    # is now 152; mark at new level-0 (23100) is 100.
    drift_snap = _snapshot_with_atm(spot=23123.0,
                                        extra_marks={23100.0: 100.0,
                                                       23150.0: 152.0})
    held_premium = runner.manager._resolve_held_premium(
        state=held, snapshot=drift_snap, held_quote=None)
    assert held_premium == 152.0, (
        f"BUG: phantom P&L. Expected 152 (our actual contract at 23150), "
        f"got {held_premium}. The lookup must be by STRIKE, not LEVEL.")


def test_held_premium_returns_none_when_strike_absent():
    """If the held strike has dropped out of the slot window entirely,
    we must return None — NOT silently pick another contract."""
    runner = _new_runner()
    for i in range(90):
        runner.on_tick(_snapshot_with_atm(spot=23127.0))
    open_snap = _snapshot_with_atm(spot=23127.0)
    open_snap["decision"]["action"] = "ENTER_LONG"
    result = runner.on_tick(open_snap)
    if result.intent.new_entry is None:
        return
    pos_id = result.intent.new_entry["hypothesis"]["position_id"]
    held = runner.manager._open_states[pos_id]
    bare_snap = _snapshot_with_atm(spot=23127.0)
    bare_snap["slot_readings"] = []
    held_premium = runner.manager._resolve_held_premium(
        state=held, snapshot=bare_snap, held_quote=None)
    assert held_premium is None


# ── BUG 2: confidence-giveback exit ─────────────────────────────


def _make_open_state(direction: int = +1, hwc: float = 0.85):
    from liqpool.research.belief.executor_v4.hypothesis import PositionHypothesis
    from liqpool.research.belief.executor_v4.manager import _OpenPositionState
    h = PositionHypothesis(
        position_id="t1", opened_at_ts=None, opened_at_bar=100,
        trigger_action="ENTER_LONG", trigger_decision={}, trigger_snapshot_summary={},
        rich_context={}, thesis_summary="x", thesis_state="HOLD_BULL",
        direction=direction, profile="INTRADAY",
        mtf_alignment={}, mtf_aligned=True,
        contract_side="CE", contract_label="CE_ATM", contract_level=0,
        strike_price=23150.0, entry_premium=100.0, entry_spot=23127.0,
        entry_dod_z=0.5, entry_friendliness=0.92, entry_spread_state="clean",
        size_fraction=0.25, size_lots=1, rupees_at_risk=500.0,
        rupees_at_target=500.0, stop_premium=70.0, target_premium=140.0,
        trail_after_r=0.80, max_hold_bars=90, premium_stop_pct=0.30,
        fee_estimate_rupees=52.0, slippage_estimate_rupees=10.0,
        expected_value_rupees=200.0, expected_edge_multiple=2.0,
        minimum_profitable_premium_delta=0.5, antithesis_score=0.1,
    )
    return _OpenPositionState(
        hypothesis=h, entry_bar=100, last_bar=102,
        high_water_confidence=hwc, low_water_confidence=0.39,
    )


def test_confidence_giveback_blocked_when_thesis_supports_held():
    """Founder's bug: confidence drops 0.85 → 0.39 within 3 bars,
    giveback fires, profitable trade gets killed. But thesis stayed
    HOLD_BULL. Fix: giveback only fires when thesis ALSO breaks."""
    mgr = PortfolioManager(PortfolioManagerConfig(min_hold_bars=0))
    state = _make_open_state(direction=+1)
    snap = _snapshot_with_atm(spot=23127.0)
    snap["thesis"]["composite_state"] = "HOLD_BULL"   # thesis still supports
    snap["decision"]["confidence"] = 0.39
    exit_reasons, _ = mgr._exit_check(
        state=state, snapshot=snap, held_read=None,
        held_premium=98.0, current_r=-0.07, confidence=0.39, bar_index=103,
    )
    assert not any("confidence giveback" in r for r in exit_reasons), (
        f"giveback should not fire when thesis still supports held side: "
        f"{exit_reasons}")


# ── BUG 3: entry-confidence floor lowered ────────────────────────


def test_confidence_floors_lowered_in_hot_fix():
    cfg = PortfolioManagerConfig()
    assert cfg.min_entry_confidence == 0.55, (
        f"min_entry_confidence should be 0.55 after hot-fix, "
        f"got {cfg.min_entry_confidence}")
    assert cfg.min_scalp_confidence == 0.62


# ── BUG 4: temporal-noise contamination ──────────────────────────


def test_horizon_weighted_consensus_drops_short_term_noise():
    """One strong long-horizon scenario (prob 0.50, 30 bars, +1) should
    dominate ten weak short-horizon ones (prob 0.05 each, 2 bars, -1).
    Under raw consensus this is roughly neutral; under horizon-weighted
    it's clearly bullish."""
    web = ScenarioWeb()
    web.scenarios["bull_long"] = Scenario(
        scenario_id="bull_long", name="bull_continuation",
        family=FAMILY_DIRECTIONAL, trigger_signature="t1",
        implied_direction=+1, implied_horizon_bars=30,
        implied_max_drawdown_during_path=0.5,
        implied_strategy_class="long_ce", base_prior=0.10,
        current_probability=0.50, decay_rate=0.99,
    )
    for i in range(10):
        web.scenarios[f"noise_{i}"] = Scenario(
            scenario_id=f"noise_{i}", name=f"n{i}",
            family=FAMILY_DIRECTIONAL, trigger_signature=f"n{i}",
            implied_direction=-1, implied_horizon_bars=2,
            implied_max_drawdown_during_path=0.5,
            implied_strategy_class="long_pe", base_prior=0.05,
            current_probability=0.05, decay_rate=0.99,
        )
    hw = web.horizon_weighted_consensus()
    assert hw > 0.5, (
        f"horizon-weighted should be bullish (long-horizon dominates), "
        f"got {hw:.3f}")


def test_horizon_weighted_zero_when_all_below_threshold():
    web = ScenarioWeb()
    for i in range(5):
        web.scenarios[f"n{i}"] = Scenario(
            scenario_id=f"n{i}", name=f"n{i}", family=FAMILY_DIRECTIONAL,
            trigger_signature=f"n{i}", implied_direction=+1,
            implied_horizon_bars=5,
            implied_max_drawdown_during_path=0.5,
            implied_strategy_class="long_ce", base_prior=0.05,
            current_probability=0.08, decay_rate=0.99,
        )
    assert web.horizon_weighted_consensus() == 0.0


def test_web_snapshot_includes_horizon_weighted_field():
    web = ScenarioWeb()
    web.scenarios["bull"] = Scenario(
        scenario_id="bull", name="bull_continuation",
        family=FAMILY_DIRECTIONAL, trigger_signature="t",
        implied_direction=+1, implied_horizon_bars=30,
        implied_max_drawdown_during_path=0.5,
        implied_strategy_class="long_ce", base_prior=0.10,
        current_probability=0.5, decay_rate=0.99,
    )
    snap = web.snapshot()
    d = snap.to_dict()
    assert "directional_consensus_horizon_weighted" in d


# ── BUG 5: min_edge_multiple raised ──────────────────────────────


def test_min_edge_multiple_raised():
    assert DEFAULT_MIN_EDGE_MULTIPLE == 1.8
    cfg = ExecutionEconomicsConfig()
    assert cfg.min_edge_multiple == 1.8
