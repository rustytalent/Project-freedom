"""End-to-end tests for the v4 PortfolioManager — Sprint 1 integration."""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4 import (
    PortfolioManager,
    PortfolioManagerConfig,
)
from liqpool.research.belief.executor_v4.manager import (
    INTENT_EXIT,
    INTENT_HOLD,
    INTENT_OPEN_LONG,
)


def _mk_snap(*, bars: int = 100, spot: float = 23000.0,
             action: str = "HOLD", direction: int = 0, confidence: float = 0.80,
             iv: str = "directional_bull", bf: str = "bullish_agreement",
             thesis: str = "HOLD_BULL", bull_score: float = 75.0,
             bear_score: float = 5.0,
             ce_signed: float = 1.4, pe_signed: float = -1.2,
             ce_atm_mark: float = 100.0, pe_atm_mark: float = 80.0,
             clean_mark_frac: float = 0.98, no_trade: float = 5.0,
             is_warm: bool = True,
             slot_acceptance: dict | None = None,
             ts: pd.Timestamp | None = None) -> dict:
    if ts is None:
        ts = pd.Timestamp("2026-06-19 10:00") + pd.Timedelta(seconds=bars)
    slot_acceptance = slot_acceptance or {}
    slots = []
    for level in range(-5, 6):
        label_ce = (f"CE_ATM" if level == 0
                     else f"CE_OTM{level}" if level > 0
                     else f"CE_ITM{abs(level)}")
        label_pe = (f"PE_ATM" if level == 0
                     else f"PE_OTM{abs(level)}" if level < 0
                     else f"PE_ITM{level}")
        slots.append({
            "strike": 23000.0 + 50 * level, "option_type": "CE",
            "level": level, "label": label_ce,
            "moneyness_label": label_ce,
            "behavior": "gamma_atm",
            "acceptance": slot_acceptance.get(label_ce, "normal"),
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": 0.7, "mark_price": ce_atm_mark if level == 0 else max(2.0, 100 - level * 15),
        })
        slots.append({
            "strike": 23000.0 + 50 * level, "option_type": "PE",
            "level": level, "label": label_pe,
            "moneyness_label": label_pe,
            "behavior": "gamma_atm",
            "acceptance": slot_acceptance.get(label_pe, "normal"),
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": -0.7, "mark_price": pe_atm_mark if level == 0 else max(2.0, 80 + level * 15),
        })
    return {
        "ts": ts, "spot": spot, "bars_seen": bars, "is_warm": is_warm,
        "thesis": {"composite_state": thesis,
                   "bull_thesis_score": bull_score,
                   "bear_thesis_score": bear_score,
                   "no_trade_score": no_trade},
        "iv_state": {"state": iv, "direction": direction, "confidence": confidence,
                     "clean_mark_fraction": clean_mark_frac,
                     "net_intent_z": ce_signed - pe_signed},
        "battlefield": {"verdict": bf, "direction": direction, "confidence": confidence,
                        "ce_rail": {"weighted_mean_signed_z": ce_signed,
                                    "dispersion_score": 0.05,
                                    "epicenter_label": "CE_ATM",
                                    "epicenter_level": 0},
                        "pe_rail": {"weighted_mean_signed_z": pe_signed,
                                    "dispersion_score": 0.05,
                                    "epicenter_label": "PE_ATM",
                                    "epicenter_level": 0}},
        "decision": {"action": action, "direction": direction,
                     "confidence": confidence, "trade_allowed": action in ("HOLD", "WAIT") or direction != 0,
                     "spread_friendliness": 0.91,
                     "strike": {"side": "CE" if direction > 0 else "PE",
                                "label": "CE_ATM" if direction > 0 else "PE_ATM",
                                "level": 0},
                     "invalidation_rule": "exit when thesis flips or rail breaks"},
        "winding": {"zone": "NO_WINDING"},
        "bull_state": {"state_index": 2 if direction > 0 else 0,
                       "state_label": "bull_impulse_confirmed"},
        "bear_state": {"state_index": 0, "state_label": "neutral"},
        "sweep_state": {"state_index": 0, "state_label": "neutral"},
        "slot_readings": slots,
    }


def _warm_with_bull_history(mgr: PortfolioManager, n: int = 100) -> None:
    """Build a bull-aligned history across all timeframes."""
    for i in range(n):
        mgr.evaluate(_mk_snap(bars=i, action="HOLD", direction=1,
                                iv="directional_bull", bf="bullish_agreement",
                                thesis="HOLD_BULL"))


def test_manager_warmup_refuses_entries():
    mgr = PortfolioManager(PortfolioManagerConfig(min_warm_bars=80))
    intent = mgr.evaluate(_mk_snap(bars=10, action="ENTER_LONG", direction=1,
                                     is_warm=False))
    assert intent.new_entry is None
    assert any("warmup" in r.lower() for r in intent.refuse_reasons)


def test_manager_opens_long_when_all_gates_pass():
    cfg = PortfolioManagerConfig(min_warm_bars=80)
    mgr = PortfolioManager(cfg)
    _warm_with_bull_history(mgr, n=100)
    # Now fire an ENTER_LONG.
    intent = mgr.evaluate(_mk_snap(bars=101, action="ENTER_LONG",
                                     direction=1, confidence=0.85,
                                     thesis="BULL_ENTRY",
                                     bull_score=80.0, ce_signed=1.6, pe_signed=-1.4))
    assert intent.new_entry is not None, f"refused: {intent.refuse_reasons}"
    assert intent.new_entry["intent"] == INTENT_OPEN_LONG
    h = intent.new_entry["hypothesis"]
    assert h["direction"] == 1
    assert h["contract_label"] == "CE_ATM"
    assert h["strike_price"] == 23000.0
    assert h["entry_premium"] > 0
    assert h["size_lots"] >= 1
    # Hypothesis carries validation/invalidation criteria
    assert h["validation_criteria"]
    assert h["hard_invalidation"]
    assert h["soft_invalidation"]
    # Explanation should be substantial
    assert len(h["explanation"]) > 300


def test_manager_refuses_when_mtf_not_aligned():
    """If L1 is bullish but the longer timeframes are bearish, refuse."""
    mgr = PortfolioManager(PortfolioManagerConfig(min_warm_bars=80))
    # Build a BEARISH history.
    for i in range(100):
        mgr.evaluate(_mk_snap(bars=i, action="HOLD", direction=-1,
                                iv="directional_bear", bf="bearish_agreement",
                                thesis="HOLD_BEAR", bull_score=5.0,
                                bear_score=75.0,
                                ce_signed=-1.4, pe_signed=1.6))
    # Now try to enter LONG (against the higher-tf bias).
    intent = mgr.evaluate(_mk_snap(bars=101, action="ENTER_LONG",
                                     direction=1, confidence=0.85,
                                     thesis="BULL_ENTRY", bull_score=72.0,
                                     iv="directional_bull",
                                     bf="bullish_agreement"))
    # Should refuse — L5/L15/L60 are all bear.
    assert intent.new_entry is None
    assert any("alignment" in r.lower() for r in intent.refuse_reasons)


def test_manager_refuses_when_clean_mark_fraction_low():
    mgr = PortfolioManager(PortfolioManagerConfig(min_warm_bars=80))
    _warm_with_bull_history(mgr, n=100)
    intent = mgr.evaluate(_mk_snap(bars=101, action="ENTER_LONG", direction=1,
                                     confidence=0.85, thesis="BULL_ENTRY",
                                     clean_mark_frac=0.50))
    assert intent.new_entry is None
    assert any("clean marks" in r.lower() for r in intent.refuse_reasons)


def test_manager_hold_then_exit_on_thesis_flip():
    cfg = PortfolioManagerConfig(min_warm_bars=80, min_hold_bars=1,
                                   cooldown_bars_after_entry=0)
    mgr = PortfolioManager(cfg)
    _warm_with_bull_history(mgr, n=100)
    open_intent = mgr.evaluate(_mk_snap(bars=101, action="ENTER_LONG",
                                          direction=1, confidence=0.85,
                                          thesis="BULL_ENTRY", bull_score=80.0))
    assert open_intent.new_entry is not None
    # Bar+1: thesis flips to bear → soft exit fires.
    held = mgr.evaluate(_mk_snap(bars=102, action="HOLD", direction=-1,
                                   confidence=0.7, thesis="BEAR_ENTRY",
                                   iv="directional_bear",
                                   bf="bearish_agreement",
                                   bull_score=20.0, bear_score=70.0,
                                   ce_signed=-1.3, pe_signed=1.4))
    assert held.closed_this_tick, f"updates: {held.position_updates}"
    closed = held.closed_this_tick[0]["outcome"]
    assert closed["exit_severity"] in ("soft", "hard")
    # Realized rupees should be a number (positive or negative)
    assert isinstance(closed["realized_rupees"], (int, float))


def test_manager_hard_exit_on_engine_emit_exit():
    cfg = PortfolioManagerConfig(min_warm_bars=80, min_hold_bars=0,
                                   cooldown_bars_after_entry=0)
    mgr = PortfolioManager(cfg)
    _warm_with_bull_history(mgr, n=100)
    mgr.evaluate(_mk_snap(bars=101, action="ENTER_LONG", direction=1,
                            confidence=0.85, thesis="BULL_ENTRY", bull_score=80.0))
    # Bar+1: engine says EXIT explicitly.
    intent = mgr.evaluate(_mk_snap(bars=102, action="EXIT", direction=0,
                                     confidence=0.6))
    assert intent.closed_this_tick
    assert intent.closed_this_tick[0]["outcome"]["exit_severity"] == "hard"
    assert any("EXIT" in r for r in intent.closed_this_tick[0]["outcome"]["exit_reason"].split(";"))


def test_manager_caps_at_max_open_positions():
    cfg = PortfolioManagerConfig(
        min_warm_bars=80, min_hold_bars=0,
        cooldown_bars_after_entry=0,
        economics=__import__(
            "liqpool.research.belief.executor_v4.economics", fromlist=["x"]
        ).ExecutionEconomicsConfig(max_open_positions=2),
    )
    mgr = PortfolioManager(cfg)
    _warm_with_bull_history(mgr, n=100)
    mgr.evaluate(_mk_snap(bars=101, action="ENTER_LONG", direction=1,
                            confidence=0.85, thesis="BULL_ENTRY", bull_score=80.0))
    mgr.evaluate(_mk_snap(bars=102, action="ENTER_LONG", direction=1,
                            confidence=0.85, thesis="BULL_ENTRY", bull_score=80.0))
    # Third entry — must be refused.
    intent = mgr.evaluate(_mk_snap(bars=103, action="ENTER_LONG", direction=1,
                                     confidence=0.85, thesis="BULL_ENTRY",
                                     bull_score=80.0))
    assert intent.new_entry is None
    assert any("max open positions" in r.lower() for r in intent.refuse_reasons)


def test_manager_daily_bleed_blocks_new_entries():
    cfg = PortfolioManagerConfig(min_warm_bars=80)
    mgr = PortfolioManager(cfg)
    _warm_with_bull_history(mgr, n=100)
    # Force daily PnL below floor manually
    mgr.daily_pnl_rupees = -6000.0
    intent = mgr.evaluate(_mk_snap(bars=101, action="ENTER_LONG", direction=1,
                                     confidence=0.85, thesis="BULL_ENTRY",
                                     bull_score=80.0))
    assert intent.new_entry is None
    assert any("daily bleed" in r.lower() for r in intent.refuse_reasons)


def test_manager_portfolio_summary_serializable():
    mgr = PortfolioManager(PortfolioManagerConfig(min_warm_bars=80))
    _warm_with_bull_history(mgr, n=85)
    intent = mgr.evaluate(_mk_snap(bars=86, action="ENTER_LONG", direction=1,
                                     confidence=0.85, thesis="BULL_ENTRY",
                                     bull_score=80.0))
    import json
    json.dumps(intent.to_dict(), default=str)
    assert intent.portfolio_summary["n_open"] >= 0


def test_manager_reset_daily_clears_counters():
    mgr = PortfolioManager()
    mgr.daily_pnl_rupees = -2000.0
    mgr.cumulative_fees_rupees = 500.0
    mgr.reset_daily()
    assert mgr.daily_pnl_rupees == 0.0
    assert mgr.cumulative_fees_rupees == 0.0
