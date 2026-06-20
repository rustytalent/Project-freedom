"""Integration tests: PortfolioManager + Sprint 3 layers wired together."""
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
        friendliness: float = 0.92,
        mark_source: str = "microprice",
        spot: float = 23000.0,
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
            "friendliness": friendliness, "spread_state": "clean",
            "mark_source": mark_source, "is_abnormal": False,
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
            "friendliness": friendliness, "spread_state": "clean",
            "mark_source": mark_source, "is_abnormal": False,
            "dod_z": pe_signed, "mark_price": 100.0,
        })
        if n_pe_def > 0:
            n_pe_def -= 1
    return {
        "ts": ts, "spot": spot, "bars_seen": bars, "is_warm": True,
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


def _warmup(mgr: PortfolioManager, bars: int = 90,
             pe_def_frac: float = 0.0) -> None:
    for i in range(bars):
        mgr.evaluate(_mk(bars=100 + i, action="HOLD",
                          pe_def_frac=pe_def_frac))


def test_portfolio_summary_carries_sprint3_outputs():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="HOLD"))
    s = intent.portfolio_summary
    assert "active_patterns" in s
    assert "mm_posterior" in s
    assert "fat_tail_score" in s
    assert "crowd_mirror" in s


def test_entry_payload_carries_sprint3_outputs():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    assert intent.new_entry is not None
    assert "active_patterns" in intent.new_entry
    assert "mm_posterior" in intent.new_entry
    assert "fat_tail_score" in intent.new_entry
    assert "crowd_mirror" in intent.new_entry


def test_manager_refuses_under_fat_tail_amp_refuse():
    """If iv state is common_shock + dirty marks, the fat-tail amp should
    fire REFUSE and block any new entry."""
    mgr = PortfolioManager()
    # Warm with dirty environment.
    for i in range(90):
        mgr.evaluate(_mk(bars=100 + i, action="HOLD",
                          iv="common_shock", friendliness=0.40,
                          mark_source="ltp"))
    intent = mgr.evaluate(_mk(bars=200, action="ENTER_LONG",
                                iv="common_shock", friendliness=0.40,
                                mark_source="ltp"))
    assert intent.new_entry is None
    # One of the refuse reasons should mention either fat-tail or IV state
    # (sprint 1 IV-dirty gate catches it first if Sprint 3 doesn't).
    assert any(("fat-tail" in r.lower() or "iv state" in r.lower()
                 or "common_shock" in r)
               for r in intent.refuse_reasons), intent.refuse_reasons


def test_manager_full_intent_serializable_with_sprint3():
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    import json
    json.dumps(intent.to_dict(), default=str)


def test_crowd_mirror_kicks_in_after_position_open():
    """After opening a long ATM CE the crowd_mirror should record the position."""
    mgr = PortfolioManager()
    _warmup(mgr)
    intent = mgr.evaluate(_mk(bars=200, action="ENTER_LONG"))
    assert intent.new_entry is not None
    # Now another tick — the crowd mirror should see 1 position.
    next_intent = mgr.evaluate(_mk(bars=201, action="HOLD"))
    crowd = next_intent.portfolio_summary["crowd_mirror"]
    assert crowd["crowd_density_long_atm_ce"] == 1.0
