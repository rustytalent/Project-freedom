from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.run_phase4_track_a_synthetic_nulls import (
    POCKETS,
    filter_trades,
    run_null_suite,
    summarize_frame,
)


def _row(
    *,
    net_r: float = 0.5,
    time_bucket: str = "morning",
    sector: str = "AUTO",
    direction: str = "UP",
    distance_atr: float = 6.0,
    target_fraction: float = 1.0,
    stop_atr_mult: float = 2.0,
    requested_max_hold_bars: int = 60,
) -> dict:
    return {
        "net_r": net_r,
        "risk_inr": 10.0,
        "total_cost": 1.0,
        "time_bucket": time_bucket,
        "sector": sector,
        "direction": direction,
        "distance_atr": distance_atr,
        "distance_bucket": "5-8",
        "target_fraction": target_fraction,
        "stop_atr_mult": stop_atr_mult,
        "requested_max_hold_bars": requested_max_hold_bars,
    }


def test_filter_trades_ands_winning_combo_criteria():
    trades = pd.DataFrame([
        _row(time_bucket="morning", sector="AUTO", direction="UP"),
        _row(time_bucket="midday", sector="PHARMA", direction="UP"),
        _row(time_bucket="afternoon", sector="AUTO", direction="UP"),
        _row(time_bucket="morning", sector="IT", direction="UP"),
        _row(time_bucket="morning", sector="AUTO", direction="DOWN"),
    ])
    out = filter_trades(trades, POCKETS["winning_combo"])
    assert len(out) == 2
    assert set(out["sector"]) == {"AUTO", "PHARMA"}


def test_summarize_frame_computes_ci_and_cost_stress():
    trades = pd.DataFrame([
        _row(net_r=0.5),
        _row(net_r=0.7),
        _row(net_r=-0.1),
    ])
    summary = summarize_frame(trades)
    assert summary["n_trades"] == 3
    assert abs(summary["mean_r"] - np.mean([0.5, 0.7, -0.1])) < 1e-9
    assert summary["ci95_lo"] < summary["mean_r"] < summary["ci95_hi"]
    assert summary["mean_r_cost_1_50x"] < summary["mean_r"]


def test_run_null_suite_returns_all_pocket_null_rows():
    rows = []
    for i in range(30):
        rows.append(_row(
            net_r=0.5 if i % 2 == 0 else -0.2,
            time_bucket="morning" if i % 3 else "midday",
            sector="AUTO" if i % 4 else "IT",
            direction="UP" if i % 5 else "DOWN",
            target_fraction=1.0 if i % 2 else 0.8,
            stop_atr_mult=2.0 if i % 2 else 1.5,
            requested_max_hold_bars=60 if i % 2 else 36,
        ))
    trades = pd.DataFrame(rows)
    results, trials = run_null_suite(
        trades,
        pockets={"winning_combo": POCKETS["winning_combo"]},
        null_tests=("matched_random_rows", "time_bucket_shuffle"),
        trials=12,
        seed=123,
        workers=1,
    )
    assert set(results["null_test"]) == {"matched_random_rows", "time_bucket_shuffle"}
    assert len(results) == 2
    assert len(trials) == 24
    assert results["p_value"].between(0.0, 1.0).all()
