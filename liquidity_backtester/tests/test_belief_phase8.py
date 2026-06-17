"""Tests for Belief Engine Phase 8 — decision layer + engine orchestrator.

Decision-layer tests use fabricated upstream readings to drive each
precedence branch. Engine tests drive the live orchestrator end-to-end
on synthetic streams to confirm the wiring + warmup discipline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liqpool.research.belief.battlefield import (
    BattlefieldSnapshot,
    FIELD_BEARISH_AGREEMENT,
    FIELD_BULLISH_AGREEMENT,
    FIELD_QUIET,
    FIELD_SINGLE_DISTORTION,
    FIELD_VOL_EXPANSION,
    RailSummary,
)
from liqpool.research.belief.decision import (
    ACTION_ENTER_LONG,
    ACTION_ENTER_SHORT,
    ACTION_EXIT,
    ACTION_HOLD,
    ACTION_NO_TRADE,
    ACTION_SCALP_CALL,
    ACTION_SCALP_PUT,
    ACTION_WAIT,
    Decision,
    DecisionConfig,
    decide,
)
from liqpool.research.belief.engine import (
    BeliefEngine,
    BeliefEngineConfig,
    BeliefSnapshot,
)
from liqpool.research.belief.iv_state import (
    IVState,
    IV_COMMON_SHOCK,
    IV_DIRECTIONAL_BEAR,
    IV_DIRECTIONAL_BULL,
    IV_DIRTY_DATA,
    IV_LIQUIDITY_DISTORTION,
    IV_NEUTRAL,
)
from liqpool.research.belief.mark_price import Quote
from liqpool.research.belief.state_machines import StateUpdate
from liqpool.research.belief.thesis_memory import (
    BEAR_ENTRY,
    BULL_ENTRY,
    EXIT_BULL,
    HOLD_BULL,
    NEUTRAL_THESIS,
    NO_TRADE_DANGER,
    ThesisSnapshot,
)
from liqpool.research.belief.winding import (
    BEAR_TRAP_WINDING,
    BULL_TRAP_WINDING,
    BULLISH_WINDING_UP,
    NO_WINDING,
    WindingZone,
)


# ─────────────────────────────────────────────────────────────────
# Fabrication helpers
# ─────────────────────────────────────────────────────────────────

def _rail(side: str) -> RailSummary:
    return RailSummary(
        side=side, n_slots=11, weighted_mean_signed_z=0.0,
        weighted_mean_abs_z=0.0, fraction_abnormal=0.0,
        dispersion_score=0.0, n_abnormal_strong=0, n_abnormal_weak=0,
        epicenter_label="", epicenter_level=0, state="strong_bull",
    )


def _bf(verdict: str, *, conf: float = 0.8) -> BattlefieldSnapshot:
    direction = (+1 if verdict == FIELD_BULLISH_AGREEMENT
                 else -1 if verdict == FIELD_BEARISH_AGREEMENT else 0)
    return BattlefieldSnapshot(
        ce_rail=_rail("CE"), pe_rail=_rail("PE"),
        verdict=verdict, direction=direction, confidence=conf, note="",
    )


def _iv(state: str, *, conf: float = 0.9) -> IVState:
    direction = (+1 if state == IV_DIRECTIONAL_BULL
                 else -1 if state == IV_DIRECTIONAL_BEAR else 0)
    return IVState(
        state=state, direction=direction, confidence=conf,
        avg_friendliness=1.0, clean_mark_fraction=1.0,
        ce_defended_fraction=0.0, pe_defended_fraction=0.0,
        ce_rejected_fraction=0.0, pe_rejected_fraction=0.0, note="",
    )


def _thesis(*, composite: str = NEUTRAL_THESIS, bull: float = 0.0,
            bear: float = 0.0, held: int = 0,
            liq: float = 0.0) -> ThesisSnapshot:
    return ThesisSnapshot(
        bull_thesis_score=bull, bear_thesis_score=bear,
        vol_expansion_score=0.0, liquidity_danger_score=liq,
        no_trade_score=max(liq, 0.0),
        held_direction=held, composite_state=composite,
        just_changed=True, note="",
    )


def _state(machine: str, idx: int, *, label: str = "", note: str = "") -> StateUpdate:
    return StateUpdate(machine=machine, state_index=idx,
                        state_label=label or f"state_{idx}",
                        just_entered=True, note=note)


def _winding(zone: str, *, conf: float = 0.8) -> WindingZone:
    return WindingZone(
        zone=zone, scalp_direction="", confidence=conf,
        reference_x=100.0, band_upper=110.0, band_lower=90.0,
        last=105.0, upper_proximity=0.8, lower_proximity=0.0,
        note="",
    )


# ─────────────────────────────────────────────────────────────────
# Decision layer — precedence
# ─────────────────────────────────────────────────────────────────

def test_dirty_data_forces_no_trade():
    d = decide(thesis=_thesis(),
               iv_state=_iv(IV_DIRTY_DATA),
               battlefield=_bf(FIELD_BULLISH_AGREEMENT))
    assert d.action == ACTION_NO_TRADE
    assert not d.trade_allowed
    assert "dirty" in d.no_trade_reason


def test_liquidity_distortion_forces_no_trade():
    d = decide(thesis=_thesis(),
               iv_state=_iv(IV_LIQUIDITY_DISTORTION),
               battlefield=_bf(FIELD_BULLISH_AGREEMENT))
    assert d.action == ACTION_NO_TRADE


def test_no_trade_danger_thesis_blocks_trade():
    d = decide(thesis=_thesis(composite=NO_TRADE_DANGER, liq=80.0),
               iv_state=_iv(IV_DIRECTIONAL_BULL),
               battlefield=_bf(FIELD_BULLISH_AGREEMENT))
    assert d.action == ACTION_NO_TRADE


def test_common_shock_forces_wait():
    d = decide(thesis=_thesis(),
               iv_state=_iv(IV_COMMON_SHOCK),
               battlefield=_bf(FIELD_VOL_EXPANSION))
    assert d.action == ACTION_WAIT
    assert "straddle" in d.no_trade_reason


def test_single_distortion_forces_wait():
    d = decide(thesis=_thesis(),
               iv_state=_iv(IV_NEUTRAL),
               battlefield=_bf(FIELD_SINGLE_DISTORTION))
    assert d.action == ACTION_WAIT


# ─────────────────────────────────────────────────────────────────
# Decision layer — exit
# ─────────────────────────────────────────────────────────────────

def test_bull_invalidation_state_8_forces_exit():
    d = decide(
        thesis=_thesis(composite=HOLD_BULL, bull=60.0, held=+1),
        iv_state=_iv(IV_NEUTRAL),
        battlefield=_bf(FIELD_QUIET),
        bull_state=_state("bull", 8, label="bull_thesis_invalidated"),
    )
    assert d.action == ACTION_EXIT
    assert "invalidated" in d.invalidation_rule


def test_exit_bull_thesis_state_forces_exit():
    d = decide(
        thesis=_thesis(composite=EXIT_BULL, bull=40.0, held=0),
        iv_state=_iv(IV_NEUTRAL),
        battlefield=_bf(FIELD_QUIET),
    )
    assert d.action == ACTION_EXIT


# ─────────────────────────────────────────────────────────────────
# Decision layer — entry
# ─────────────────────────────────────────────────────────────────

def test_bull_entry_picks_atm_ce():
    d = decide(
        thesis=_thesis(composite=BULL_ENTRY, bull=75.0),
        iv_state=_iv(IV_DIRECTIONAL_BULL, conf=0.85),
        battlefield=_bf(FIELD_BULLISH_AGREEMENT, conf=0.85),
    )
    assert d.action == ACTION_ENTER_LONG
    assert d.direction == +1
    assert d.strike.side == "CE"
    assert d.strike.level == 0
    assert d.strike.label == "CE_ATM"
    assert d.confidence >= 0.5
    assert "REGIME_BREAK_DOWN" in d.invalidation_rule


def test_bear_entry_picks_atm_pe():
    d = decide(
        thesis=_thesis(composite=BEAR_ENTRY, bear=75.0),
        iv_state=_iv(IV_DIRECTIONAL_BEAR, conf=0.85),
        battlefield=_bf(FIELD_BEARISH_AGREEMENT, conf=0.85),
    )
    assert d.action == ACTION_ENTER_SHORT
    assert d.direction == -1
    assert d.strike.label == "PE_ATM"


def test_entry_blocked_when_confidence_below_floor():
    """Even with BULL_ENTRY composite, low IV / battlefield confidence
    suppresses the entry."""
    d = decide(
        thesis=_thesis(composite=BULL_ENTRY, bull=72.0),
        iv_state=_iv(IV_DIRECTIONAL_BULL, conf=0.05),
        battlefield=_bf(FIELD_BULLISH_AGREEMENT, conf=0.05),
        cfg=DecisionConfig(min_directional_confidence=0.3),
    )
    assert d.action != ACTION_ENTER_LONG


# ─────────────────────────────────────────────────────────────────
# Decision layer — scalps
# ─────────────────────────────────────────────────────────────────

def test_bull_trap_winding_emits_scalp_put():
    d = decide(
        thesis=_thesis(),
        iv_state=_iv(IV_NEUTRAL),
        battlefield=_bf(FIELD_BEARISH_AGREEMENT),
        winding=_winding(BULL_TRAP_WINDING, conf=0.7),
    )
    assert d.action == ACTION_SCALP_PUT
    assert d.direction == -1
    assert d.strike.side == "PE"


def test_bear_trap_winding_emits_scalp_call():
    d = decide(
        thesis=_thesis(),
        iv_state=_iv(IV_NEUTRAL),
        battlefield=_bf(FIELD_BULLISH_AGREEMENT),
        winding=_winding(BEAR_TRAP_WINDING, conf=0.7),
    )
    assert d.action == ACTION_SCALP_CALL
    assert d.direction == +1


def test_winding_below_confidence_floor_does_not_scalp():
    d = decide(
        thesis=_thesis(),
        iv_state=_iv(IV_NEUTRAL),
        battlefield=_bf(FIELD_QUIET),
        winding=_winding(BULL_TRAP_WINDING, conf=0.1),
    )
    assert d.action != ACTION_SCALP_PUT


def test_sweep_reversal_emits_scalp():
    d = decide(
        thesis=_thesis(),
        iv_state=_iv(IV_DIRECTIONAL_BULL, conf=0.7),
        battlefield=_bf(FIELD_BULLISH_AGREEMENT, conf=0.7),
        sweep_state=_state("sweep", 5, label="reversal_trigger",
                            note="bullish sweep reversal trigger — take CE"),
    )
    # Sweep reversal takes precedence over plain neutral/hold paths and
    # emits a CE scalp.
    assert d.action == ACTION_SCALP_CALL


# ─────────────────────────────────────────────────────────────────
# Decision layer — hold + continuation
# ─────────────────────────────────────────────────────────────────

def test_hold_bull_extends_position():
    d = decide(
        thesis=_thesis(composite=HOLD_BULL, bull=80.0, held=+1),
        iv_state=_iv(IV_NEUTRAL),
        battlefield=_bf(FIELD_QUIET),
    )
    assert d.action == ACTION_HOLD
    assert d.direction == +1


def test_continuation_trigger_extends_held_long():
    d = decide(
        thesis=_thesis(composite=HOLD_BULL, bull=70.0, held=+1),
        iv_state=_iv(IV_NEUTRAL),
        battlefield=_bf(FIELD_QUIET),
        bull_state=_state("bull", 6, label="continuation_trigger"),
    )
    assert d.action == ACTION_HOLD
    assert d.confidence > 0.7
    assert "continuation" in d.notes[0]


def test_default_neutral_when_nothing_actionable():
    d = decide(
        thesis=_thesis(),
        iv_state=_iv(IV_NEUTRAL),
        battlefield=_bf(FIELD_QUIET),
    )
    assert d.action == ACTION_WAIT


def test_decision_to_dict_is_serializable():
    d = decide(
        thesis=_thesis(composite=BULL_ENTRY, bull=75.0),
        iv_state=_iv(IV_DIRECTIONAL_BULL, conf=0.85),
        battlefield=_bf(FIELD_BULLISH_AGREEMENT, conf=0.85),
    )
    import json
    json.dumps(d.to_dict())


# ─────────────────────────────────────────────────────────────────
# Engine orchestrator
# ─────────────────────────────────────────────────────────────────

def _make_quotes(spot: float, strikes, rng) -> dict:
    quotes = {}
    for k in strikes:
        ce_fair = max(2.0, (spot - k + 50.0) * 0.5 + 30.0)
        pe_fair = max(2.0, (k - spot + 50.0) * 0.5 + 30.0)
        ce_fair += float(rng.normal(0.0, 0.3))
        pe_fair += float(rng.normal(0.0, 0.3))
        quotes[(k, "CE")] = Quote(bid=ce_fair - 0.1, ask=ce_fair + 0.1,
                                   bid_qty=300, ask_qty=300,
                                   ltp=ce_fair, ltp_age_s=0.1)
        quotes[(k, "PE")] = Quote(bid=pe_fair - 0.1, ask=pe_fair + 0.1,
                                   bid_qty=300, ask_qty=300,
                                   ltp=pe_fair, ltp_age_s=0.1)
    return quotes


def test_engine_warmup_blocks_decisions():
    """During warmup the engine must refuse to act, regardless of inputs."""
    cfg = BeliefEngineConfig(warmup_bars=30)
    eng = BeliefEngine(cfg=cfg)
    spot = 23000.0
    strikes = [spot + 50.0 * i for i in range(-5, 6)]
    rng = np.random.default_rng(7)
    for bar in range(10):
        spot += 1.0
        snap = eng.observe(
            ts=pd.Timestamp("2026-06-16 09:15") + pd.Timedelta(minutes=bar),
            spot=spot, quotes=_make_quotes(spot, strikes, rng),
        )
        assert snap.decision.action == ACTION_NO_TRADE
        assert "warming up" in snap.decision.no_trade_reason
        assert not snap.is_warm


def test_engine_promotes_to_warm_after_warmup_bars():
    cfg = BeliefEngineConfig(warmup_bars=30)
    eng = BeliefEngine(cfg=cfg)
    spot = 23000.0
    strikes = [spot + 50.0 * i for i in range(-5, 6)]
    rng = np.random.default_rng(11)
    snap = None
    for bar in range(35):
        spot += float(rng.normal(0.0, 2.0))
        snap = eng.observe(
            ts=pd.Timestamp("2026-06-16 09:15") + pd.Timedelta(minutes=bar),
            spot=spot, quotes=_make_quotes(spot, strikes, rng),
        )
    assert snap.is_warm
    assert snap.bars_seen >= cfg.warmup_bars


def test_engine_reset_clears_all_state():
    cfg = BeliefEngineConfig(warmup_bars=10)
    eng = BeliefEngine(cfg=cfg)
    spot = 23000.0
    strikes = [spot + 50.0 * i for i in range(-5, 6)]
    rng = np.random.default_rng(3)
    for bar in range(15):
        spot += float(rng.normal(0.0, 2.0))
        eng.observe(
            ts=pd.Timestamp("2026-06-16 09:15") + pd.Timedelta(minutes=bar),
            spot=spot, quotes=_make_quotes(spot, strikes, rng),
        )
    assert eng.bars_seen >= cfg.warmup_bars
    eng.reset()
    assert eng.bars_seen == 0
    assert not eng.is_warm


def test_engine_snapshot_is_json_serializable():
    cfg = BeliefEngineConfig(warmup_bars=20)
    eng = BeliefEngine(cfg=cfg)
    spot = 23000.0
    strikes = [spot + 50.0 * i for i in range(-5, 6)]
    rng = np.random.default_rng(5)
    snap = None
    for bar in range(25):
        spot += float(rng.normal(0.0, 2.0))
        snap = eng.observe(
            ts=pd.Timestamp("2026-06-16 09:15") + pd.Timedelta(minutes=bar),
            spot=spot, quotes=_make_quotes(spot, strikes, rng),
        )
    import json
    payload = snap.to_dict()
    assert payload["bars_seen"] >= 20
    json.dumps(payload, default=str)


def test_engine_handles_dirty_quotes_gracefully():
    """A crossed quote on one contract → engine continues but reports a
    no-trade reason once the clean-mark fraction drops below threshold."""
    cfg = BeliefEngineConfig(warmup_bars=15)
    eng = BeliefEngine(cfg=cfg)
    spot = 23000.0
    strikes = [spot + 50.0 * i for i in range(-5, 6)]
    rng = np.random.default_rng(9)
    # Build up warmup with clean quotes.
    snap = None
    for bar in range(20):
        spot += float(rng.normal(0.0, 1.5))
        snap = eng.observe(
            ts=pd.Timestamp("2026-06-16 09:15") + pd.Timedelta(minutes=bar),
            spot=spot, quotes=_make_quotes(spot, strikes, rng),
        )
    # Now flip MOST contracts to crossed quotes (no LTP fallback freshness).
    dirty = {}
    for k in strikes:
        for side in ("CE", "PE"):
            dirty[(k, side)] = Quote(bid=10.0, ask=5.0, bid_qty=0, ask_qty=0,
                                      ltp=float("nan"), ltp_age_s=99.0)
    snap = eng.observe(
        ts=pd.Timestamp("2026-06-16 09:50"),
        spot=spot, quotes=dirty,
    )
    # The IV state should detect dirty data and the decision should refuse.
    assert snap.decision.action in (ACTION_NO_TRADE, ACTION_WAIT)


def test_engine_observe_runs_without_quotes_changes_keys():
    """Adding new contracts mid-stream (eg. chain widening) must NOT crash —
    the engine must rebuild moneyness slots on demand."""
    cfg = BeliefEngineConfig(warmup_bars=5)
    eng = BeliefEngine(cfg=cfg)
    spot = 23000.0
    strikes_narrow = [spot + 50.0 * i for i in range(-3, 4)]
    strikes_wide = [spot + 50.0 * i for i in range(-5, 6)]
    rng = np.random.default_rng(13)
    for bar in range(6):
        spot += float(rng.normal(0.0, 1.0))
        eng.observe(
            ts=pd.Timestamp("2026-06-16 09:15") + pd.Timedelta(minutes=bar),
            spot=spot, quotes=_make_quotes(spot, strikes_narrow, rng),
        )
    # Now widen the chain. Engine must accept the new contracts.
    snap = eng.observe(
        ts=pd.Timestamp("2026-06-16 09:25"),
        spot=spot, quotes=_make_quotes(spot, strikes_wide, rng),
    )
    assert snap.bars_seen == 7
