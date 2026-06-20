"""Tests for executor_v4.memory + flow_memory."""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4.flow_memory import (
    FlowMemory,
    FlowMemoryConfig,
)
from liqpool.research.belief.executor_v4.memory import (
    MultiTimeframeMemory,
    TimeframeLevel,
)


def _mk(*, bars: int = 100, spot: float = 23000.0,
        thesis: str = "HOLD_BULL", iv: str = "directional_bull",
        bf: str = "bullish_agreement", direction: int = 1,
        ce_signed: float = 1.0, pe_signed: float = -1.0,
        ce_def_frac: float = 0.0, pe_def_frac: float = 0.0,
        ce_disp: float = 0.05, pe_disp: float = 0.05,
        winding: str = "NO_WINDING",
        ts: pd.Timestamp | None = None) -> dict:
    if ts is None:
        ts = pd.Timestamp("2026-06-19 10:00") + pd.Timedelta(seconds=bars)
    slot_readings = []
    n_ce = 11; n_pe = 11
    n_ce_defended = int(ce_def_frac * n_ce)
    n_pe_defended = int(pe_def_frac * n_pe)
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "CE", "level": level,
            "label": f"CE_{level}",
            "acceptance": ("defended" if n_ce_defended > 0 else "normal"),
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": ce_signed, "mark_price": 100.0,
        })
        if n_ce_defended > 0:
            n_ce_defended -= 1
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "PE", "level": level,
            "label": f"PE_{level}",
            "acceptance": ("defended" if n_pe_defended > 0 else "normal"),
            "friendliness": 0.92, "spread_state": "clean",
            "mark_source": "microprice", "is_abnormal": False,
            "dod_z": pe_signed, "mark_price": 100.0,
        })
        if n_pe_defended > 0:
            n_pe_defended -= 1
    return {
        "ts": ts, "spot": spot, "bars_seen": bars, "is_warm": True,
        "thesis": {"composite_state": thesis,
                   "bull_thesis_score": 70.0, "bear_thesis_score": 10.0,
                   "no_trade_score": 5.0},
        "iv_state": {"state": iv, "direction": direction, "confidence": 0.8,
                     "clean_mark_fraction": 0.98,
                     "net_intent_z": ce_signed - pe_signed},
        "battlefield": {"verdict": bf, "direction": direction, "confidence": 0.8,
                        "ce_rail": {"weighted_mean_signed_z": ce_signed,
                                    "dispersion_score": ce_disp,
                                    "epicenter_label": "CE_ATM",
                                    "epicenter_level": 0},
                        "pe_rail": {"weighted_mean_signed_z": pe_signed,
                                    "dispersion_score": pe_disp,
                                    "epicenter_label": "PE_ATM",
                                    "epicenter_level": 0}},
        "decision": {"action": "HOLD", "direction": direction,
                     "confidence": 0.8, "trade_allowed": True},
        "winding": {"zone": winding},
        "bull_state": {"state_index": 2}, "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


# ── Multi-timeframe memory ────────────────────────────────────────

def test_mtf_records_at_each_level():
    mtf = MultiTimeframeMemory(l5_subsample=5, l15_subsample=15, l60_subsample=60)
    for i in range(70):
        mtf.observe(_mk(bars=100 + i))
    v1 = mtf.view(TimeframeLevel.L1)
    v5 = mtf.view(TimeframeLevel.L5)
    v15 = mtf.view(TimeframeLevel.L15)
    v60 = mtf.view(TimeframeLevel.L60)
    assert v1.n_samples == 60   # cap
    assert v5.n_samples == 14   # 70/5
    assert v15.n_samples == 4   # 70/15
    assert v60.n_samples == 1   # 70/60


def test_mtf_alignment_requires_long_tf_confirmation():
    """L1 alone is retail. Need at least one of L5/L15/L60 to match."""
    mtf = MultiTimeframeMemory()
    # Build all-bull history.
    for i in range(70):
        mtf.observe(_mk(bars=100 + i))
    alignment = mtf.alignment(proposed_direction=1)
    assert alignment["alignment_ok"]
    assert alignment["confirmation_count"] >= 1

    # Propose SHORT against bull history → alignment fails.
    alignment_bear = mtf.alignment(proposed_direction=-1)
    assert not alignment_bear["alignment_ok"]


def test_mtf_view_to_dict_serializable():
    mtf = MultiTimeframeMemory()
    for i in range(10):
        mtf.observe(_mk(bars=100 + i))
    import json
    json.dumps(mtf.view(TimeframeLevel.L1).to_dict())


def test_mtf_view_dominant_thesis_majority():
    mtf = MultiTimeframeMemory()
    # 8 bars of BULL_ENTRY, 2 bars of NEUTRAL → bull dominates
    for i in range(8):
        mtf.observe(_mk(bars=100 + i, thesis="BULL_ENTRY"))
    for i in range(2):
        mtf.observe(_mk(bars=108 + i, thesis="NEUTRAL"))
    v = mtf.view(TimeframeLevel.L1)
    assert v.dominant_thesis == "BULL_ENTRY"
    assert v.direction_bias == 1


# ── Flow memory ───────────────────────────────────────────────────

def test_flow_memory_captures_events():
    fm = FlowMemory()
    for i in range(20):
        fm.observe(_mk(bars=100 + i))
    assert len(fm.events) == 20
    assert fm.events[-1].iv_state == "directional_bull"


def test_flow_memory_surprise_score_jumps_on_intent_swing():
    fm = FlowMemory()
    for i in range(5):
        fm.observe(_mk(bars=100 + i, ce_signed=0.5, pe_signed=-0.5))
    # Big swing
    ev = fm.observe(_mk(bars=105, ce_signed=3.0, pe_signed=-2.0))
    assert ev.surprise_score > 0.4


def test_flow_memory_finds_sustained_ce_defended_streak():
    fm = FlowMemory()
    # 8 bars of high CE defended fraction (>40%)
    for i in range(8):
        fm.observe(_mk(bars=100 + i, ce_def_frac=0.5))
    matches = fm.find_pattern("sustained_ce_defended")
    assert matches
    assert (matches[0][1] - matches[0][0]) >= 4


def test_flow_memory_thesis_oscillation_detected():
    fm = FlowMemory()
    # Many bull/bear flips in a 25-bar window
    states = (["BULL_ENTRY", "BEAR_ENTRY", "BULL_ENTRY", "BEAR_ENTRY",
                "BULL_ENTRY", "BEAR_ENTRY"] * 5)[:25]
    for i, st in enumerate(states):
        fm.observe(_mk(bars=100 + i, thesis=st))
    matches = fm.find_pattern("thesis_oscillation")
    assert matches


def test_flow_memory_within_seconds_filters_by_time():
    fm = FlowMemory()
    base = pd.Timestamp("2026-06-19 10:00")
    for i in range(60):
        fm.observe(_mk(bars=100 + i, ts=base + pd.Timedelta(seconds=i)))
    recent_30s = fm.within_seconds(30)
    assert len(recent_30s) <= 31
    assert len(recent_30s) >= 25  # allow some slop


def test_flow_memory_summary_dict_serializable():
    fm = FlowMemory()
    for i in range(20):
        fm.observe(_mk(bars=100 + i))
    s = fm.summary()
    import json
    json.dumps(s, default=str)
    assert s["event_count"] == 20
