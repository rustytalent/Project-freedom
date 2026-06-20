"""Integration tests: PortfolioManager + Sprint 2 layers wired together."""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4.manager import (
    PortfolioManager,
    PortfolioManagerConfig,
)


def _mk(*, bars: int = 100, action: str = "HOLD", direction: int = 1,
        thesis: str = "HOLD_BULL", iv: str = "directional_bull",
        bf: str = "bullish_agreement", confidence: float = 0.82,
        bull_score: float = 70.0, bear_score: float = 10.0,
        ce_signed: float = 1.5, pe_signed: float = -1.0,
        ce_def_frac: float = 0.0, pe_def_frac: float = 0.0,
        winding: str = "NO_WINDING") -> dict:
    ts = pd.Timestamp("2026-06-19 10:00") + pd.Timedelta(seconds=bars)
    n_each = 11
    n_ce_def = int(ce_def_frac * n_each); n_pe_def = int(pe_def_frac * n_each)
    slot_readings = []
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "CE", "level": level,
            "label": f"CE_{'ATM' if level == 0 else (f'OTM{level}' if level > 0 else f'ITM{abs(level)}')}",
            "acceptance": "defended" if n_ce_def > 0 else "normal",
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": ce_signed, "mark_price": 100.0,
        })
        if n_ce_def > 0:
            n_ce_def -= 1
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "PE", "level": level,
            "label": f"PE_{'ATM' if level == 0 else (f'OTM{level}' if level > 0 else f'ITM{abs(level)}')}",
            "acceptance": "defended" if n_pe_def > 0 else "normal",
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": pe_signed, "mark_price": 100.0,
        })
        if n_pe_def > 0:
            n_pe_def -= 1
    return {
        "ts": ts, "spot": 23000.0, "bars_seen": bars, "is_warm": True,
        "thesis": {"composite_state": thesis,
                   "bull_thesis_score": bull_score,
                   "bear_thesis_score": bear_score,
                   "no_trade_score": 5.0},
        "iv_state": {"state": iv, "direction": direction, "confidence": 0.85,
                     "clean_mark_fraction": 0.98,
                     "net_intent_z": ce_signed - pe_signed},
        "battlefield": {
            "verdict": bf, "direction": direction, "confidence": 0.8,
            "ce_rail": {"weighted_mean_signed_z": ce_signed,
                        "dispersion_score": 0.10,
                        "epicenter_label": "CE_ATM",
                        "epicenter_level": 0},
            "pe_rail": {"weighted_mean_signed_z": pe_signed,
                        "dispersion_score": 0.10,
                        "epicenter_label": "PE_ATM",
                        "epicenter_level": 0},
        },
        "decision": {"action": action, "direction": direction,
                     "confidence": confidence, "trade_allowed": True,
                     "strike": {"side": "CE" if direction > 0 else "PE",
                                 "level": 0,
                                 "label": f"{'CE' if direction > 0 else 'PE'}_ATM"}},
        "winding": {"zone": winding},
        "bull_state": {"state_index": 2},
        "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


def _warmup(mgr: PortfolioManager, bars: int = 90) -> None:
    """Push enough HOLDs so MTF / substrate are warm."""
    for i in range(bars):
        mgr.evaluate(_mk(bars=100 + i, action="HOLD"))


def test_manager_entry_emits_critic_and_aggregator_in_payload():
    """When the manager opens a position the new_entry dict should
    carry critique, aggregator_decision, projection and counterfactual."""
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    assert intent.new_entry is not None
    assert "critique" in intent.new_entry
    assert "aggregator_decision" in intent.new_entry
    assert "projection" in intent.new_entry
    assert "counterfactual_plan" in intent.new_entry
    assert "web_snapshot" in intent.new_entry


def test_manager_aggregator_refuses_when_critic_finds_strong_opposite_acceptance():
    """A long entry where PE rail is heavily defended → critic should refuse."""
    mgr = PortfolioManager()
    # Warm with PE-defended environment.
    for i in range(90):
        mgr.evaluate(_mk(bars=100 + i, action="HOLD", pe_def_frac=0.70))
    intent = mgr.evaluate(_mk(bars=200, action="ENTER_LONG", pe_def_frac=0.70))
    # Either entry is refused at the critic gate or aggregator refuses.
    refused_or_haircut = (
        intent.new_entry is None
        or intent.new_entry["aggregator_decision"]["recommended_size_multiplier"] < 1.0
    )
    assert refused_or_haircut


def test_manager_counterfactual_kills_long_on_thesis_flip():
    """Open a long, then flip thesis to BEAR within the kill window;
    the counterfactual's thesis_flips_bear hard kill must fire."""
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    assert intent.new_entry is not None
    pos_id = list(mgr._open_states.keys())[0]
    # Now feed a bar where thesis turns BEAR within counterfactual window.
    flip_intent = mgr.evaluate(_mk(
        bars=201, action="HOLD",
        thesis="HOLD_BEAR", bull_score=8, bear_score=72,
        ce_signed=-1.0, pe_signed=1.0,
        iv="directional_bear", bf="bearish_agreement", direction=-1,
    ))
    closed = [c for c in flip_intent.closed_this_tick
              if c["position_id"] == pos_id]
    assert closed, "expected the position to be closed by counterfactual"
    reason = closed[0]["outcome"]["exit_reason"]
    assert ("thesis_flips_bear" in reason
            or "thesis flipped" in reason
            or "EXIT" in reason)


def test_manager_projection_tape_populates_after_closures():
    """After at least one closed position, projection summary's tape grows."""
    mgr = PortfolioManager()
    _warmup(mgr)
    mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    # Force close via belief-engine EXIT.
    mgr.evaluate(_mk(bars=201, action="EXIT"))
    summary = mgr.projection.summary()
    assert summary["tape_size"] >= 1


def test_manager_portfolio_summary_includes_scenario_web():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="HOLD"))
    summary = intent.portfolio_summary
    assert "scenario_web" in summary
    assert "projection_summary" in summary
    # Serialization sanity.
    import json
    json.dumps(intent.to_dict(), default=str)
