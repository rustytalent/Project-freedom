import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from analysis.run_geometry_mode_sweep import (
    _load_resume_geometry_rows,
    _progress_path,
    _write_json,
)


def test_progress_path_defaults_to_out_plus_partial_suffix():
    args = SimpleNamespace(out="reports/geometry_mode_sweep.json", progress_out="")
    assert _progress_path(args) == Path("reports/geometry_mode_sweep.json.partial")


def test_write_json_is_readable_and_load_resume_filters_geometry_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "geometry.partial.json"
        _write_json(path, {
            "geometry_sweep_results": [
                {"geometry": "stop_1_target_2", "mean_R": -0.1},
                {"not_geometry": True},
            ],
        })
        payload = json.loads(path.read_text())
        rows = _load_resume_geometry_rows(path)
    assert payload["geometry_sweep_results"][0]["geometry"] == "stop_1_target_2"
    assert rows == [{"geometry": "stop_1_target_2", "mean_R": -0.1}]


def test_load_resume_geometry_rows_tolerates_missing_or_bad_file():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "missing.json"
        bad = Path(tmp) / "bad.json"
        bad.write_text("{bad json")
        assert _load_resume_geometry_rows(missing) == []
        assert _load_resume_geometry_rows(bad) == []
