import pandas as pd

from analysis.run_stream_a_rupee_target import _candidate_rows, _summarise_groups


def test_candidate_rows_require_positive_mean_p_value_and_sample_floor():
    rows = [
        {"name": "pass", "n": 200, "mean_R": 0.01, "p_value": 0.049},
        {"name": "thin", "n": 199, "mean_R": 1.00, "p_value": 0.001},
        {"name": "negative", "n": 500, "mean_R": -0.01, "p_value": 0.001},
        {"name": "not_sig", "n": 500, "mean_R": 0.10, "p_value": 0.050},
    ]

    out = _candidate_rows(rows, min_n=200, p_threshold=0.05)

    assert [row["name"] for row in out] == ["pass"]


def test_summarise_groups_adds_mode_label():
    trades = pd.DataFrame({
        "mode": ["touch_confirmed", "touch_confirmed"],
        "net_r": [1.0, -0.5],
        "net_pnl": [100.0, -50.0],
        "gross_pnl": [120.0, -30.0],
        "total_cost": [20.0, 20.0],
        "risk_inr": [100.0, 100.0],
    })

    rows = _summarise_groups(trades, ("mode",))

    assert rows[0]["mode_label"] == "respect_mode"
    assert rows[0]["n"] == 2
