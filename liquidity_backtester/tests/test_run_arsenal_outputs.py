import tempfile
from pathlib import Path

import pandas as pd

from analysis.run_arsenal import _null_results_frame, _write_null_results
from liqpool.arsenal import NullResult


def test_null_results_frame_has_stable_schema_for_empty_results():
    frame = _null_results_frame([])
    assert list(frame.columns) == [
        "alpha_name",
        "test_name",
        "actual_mean_R",
        "null_mean_R_mean",
        "null_mean_R_std",
        "null_p_value",
        "n_trials",
        "actual_n_trades",
    ]
    assert frame.empty


def test_write_null_results_writes_partial_csv():
    result = NullResult(
        alpha_name="alpha",
        test_name="time_shuffle",
        actual_mean_R=0.1,
        null_mean_R_mean=-0.2,
        null_mean_R_std=0.3,
        null_p_value=0.04,
        n_trials=10,
        actual_n_trades=25,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "null_tests.partial.csv"
        _write_null_results([result], path)
        frame = pd.read_csv(path)
    assert frame.loc[0, "alpha_name"] == "alpha"
    assert frame.loc[0, "test_name"] == "time_shuffle"
    assert frame.loc[0, "n_trials"] == 10
