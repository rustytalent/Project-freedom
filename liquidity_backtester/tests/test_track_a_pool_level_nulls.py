from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.run_phase4_track_a_pool_level_nulls import (
    make_atr_offset_candidates,
    make_random_pool_candidates,
    pocket_candidates,
    summarize_trades,
)


def _candidate(
    *,
    sector: str = "AUTO",
    side: str = "above",
    morning: float = 1.0,
    midday: float = 0.0,
    entry: float = 100.0,
    atr: float = 2.0,
    distance_atr: float = 5.0,
) -> dict:
    return {
        "symbol": "MARUTI",
        "sector": sector,
        "pool_side": side,
        "st_session_morning": morning,
        "st_session_midday": midday,
        "entry_ref_for_null": entry,
        "atr_at_decision_for_null": atr,
        "pool_width_atr_for_null": 0.25,
        "distance_atr": distance_atr,
        "price_low": entry + 10.0,
        "price_high": entry + 10.5,
    }


def test_pocket_candidates_direction_hard_filters_time_sector_and_side():
    df = pd.DataFrame([
        _candidate(sector="AUTO", side="above", morning=1.0),
        _candidate(sector="PHARMA", side="below", morning=1.0),
        _candidate(sector="IT", side="above", morning=1.0),
        _candidate(sector="AUTO", side="above", morning=0.0, midday=0.0),
    ])
    out = pocket_candidates(df, "direction_hard")
    assert len(out) == 1
    assert out.iloc[0]["sector"] == "AUTO"
    assert out.iloc[0]["pool_side"] == "above"


def test_pocket_candidates_direction_soft_keeps_both_sides():
    df = pd.DataFrame([
        _candidate(sector="AUTO", side="above", morning=1.0),
        _candidate(sector="PHARMA", side="below", midday=1.0, morning=0.0),
        _candidate(sector="BANKING", side="above", morning=1.0),
    ])
    out = pocket_candidates(df, "direction_soft")
    assert len(out) == 2
    assert set(out["pool_side"]) == {"above", "below"}


def test_atr_offset_candidates_place_levels_on_correct_side():
    df = pd.DataFrame([
        _candidate(side="above", entry=100.0, atr=2.0),
        _candidate(side="below", entry=100.0, atr=2.0),
    ])
    out = make_atr_offset_candidates(df, 3.0)
    above = out[out["pool_side"] == "above"].iloc[0]
    below = out[out["pool_side"] == "below"].iloc[0]
    assert above["price_low"] > 100.0
    assert below["price_high"] < 100.0
    assert np.isclose(above["distance_atr"], 3.0)
    assert np.isclose(below["distance_atr"], 3.0)


def test_random_pool_candidates_preserve_side_and_valid_width():
    df = pd.DataFrame([
        _candidate(side="above", distance_atr=5.0),
        _candidate(side="below", distance_atr=7.0),
    ])
    out = make_random_pool_candidates(df, np.random.default_rng(1))
    assert list(out["pool_side"]) == ["above", "below"]
    assert (out["price_high"] > out["price_low"]).all()


def test_summarize_trades_reports_cost_stress():
    trades = pd.DataFrame([
        {"net_r": 0.5, "risk_inr": 10.0, "total_cost": 1.0},
        {"net_r": -0.1, "risk_inr": 10.0, "total_cost": 1.0},
        {"net_r": 0.7, "risk_inr": 10.0, "total_cost": 1.0},
    ])
    row = summarize_trades(trades)
    assert row["n_trades"] == 3
    assert row["mean_r"] > 0
    assert row["mean_r_cost_1_50x"] < row["mean_r"]
