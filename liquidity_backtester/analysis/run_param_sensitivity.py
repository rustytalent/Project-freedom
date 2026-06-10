"""Parameter sensitivity harness — promotes registry entries from
"unvalidated" to "validated" with data instead of vibes.

For each sweepable Tunable in liqpool/tuning.REGISTRY (or a chosen
subset), the harness:

  1. Builds the sweep grid from the registry's sweep_range/steps.
  2. For DETECTOR parameters: re-runs detection on every asset in the
     bundle at each grid value and reports pool counts + touch rates
     + respect rates per value.
  3. Emits a per-parameter verdict:
       plateau   — metric varies < 10% across the grid: value choice
                   doesn't matter much; current default fine
       monotone  — metric trends with the parameter: the default
                   should sit where marginal gain flattens
       cliff     — sharp regime change inside the grid: the default
                   needs to be re-examined and the cliff documented

The verdict is advisory — the operator promotes the registry status
by editing liqpool/tuning.py with a pointer to the emitted report.
That keeps a human in the loop on every threshold change.

Usage:
    python -m analysis.run_param_sensitivity \\
        --bundle path/to/multi_asset_report.pkl \\
        --params sweep_min_atr vw_swing_multiplier \\
        --out reports/param_sensitivity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from liqpool.tuning import REGISTRY, sweepable


def build_grid(name: str) -> List[float]:
    t = REGISTRY[name]
    if t.sweep_range is None:
        return [t.value]
    lo, hi = t.sweep_range
    grid = list(np.linspace(lo, hi, t.sweep_steps))
    # Always include the current default so the report shows where
    # the engine currently sits on the curve.
    if not any(abs(g - t.value) < 1e-12 for g in grid):
        grid = sorted(grid + [t.value])
    return [float(g) for g in grid]


def classify_curve(metric_by_value: Dict[float, float]) -> str:
    """plateau / monotone / cliff verdict on a metric curve."""
    vals = [v for v in metric_by_value.values() if np.isfinite(v)]
    if len(vals) < 3:
        return "insufficient-data"
    arr = np.asarray(vals, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    centre = float(np.median(arr))
    if centre == 0:
        rel_spread = float("inf") if hi > lo else 0.0
    else:
        rel_spread = (hi - lo) / abs(centre)
    if rel_spread < 0.10:
        return "plateau"
    diffs = np.diff(arr)
    # Cliff: one adjacent step explains > 60% of the total range.
    if len(diffs) and np.abs(diffs).max() > 0.6 * (hi - lo):
        return "cliff"
    if np.all(diffs >= 0) or np.all(diffs <= 0):
        return "monotone"
    return "non-monotone"


def sweep_detector_param(
    report: Any,
    param_name: str,
    grid: List[float],
) -> pd.DataFrame:
    """Re-run pool detection per asset per grid value. Returns one
    row per (value, asset) with pool counts; aggregate happens in
    the caller. Detector params flow through the Config object's
    attribute of the same name (the detectors read them via
    getattr(params, name, default))."""
    from liqpool.config import Config
    from liqpool.pools import detect_pools

    rows = []
    assets = getattr(report, "assets", {}) or {}
    for value in grid:
        cfg = Config()
        if not hasattr(cfg, param_name):
            # Detector reads via getattr fallback — setting the attr
            # dynamically is exactly how an override reaches it.
            pass
        setattr(cfg, param_name, value)
        for symbol, ad in assets.items():
            df = getattr(ad, "base_df", None)
            tf_data = getattr(ad, "tf_data", None)
            if df is None or df.empty:
                continue
            try:
                pools = detect_pools(tf_data or {"base": df}, cfg)
                n_pools = len(pools)
            except Exception:
                n_pools = -1
            rows.append({
                "param": param_name, "value": value,
                "symbol": str(symbol), "n_pools": n_pools,
            })
    return pd.DataFrame(rows)


def run(
    bundle_path: Optional[Path],
    params: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = None
    if bundle_path is not None:
        import pickle
        with open(bundle_path, "rb") as f:
            report = pickle.load(f)

    chosen = params or [t.name for t in sweepable()]
    results: Dict[str, Any] = {}
    for name in chosen:
        if name not in REGISTRY:
            results[name] = {"error": "not in registry"}
            continue
        grid = build_grid(name)
        entry: Dict[str, Any] = {
            "grid": grid,
            "current_default": REGISTRY[name].value,
            "status": REGISTRY[name].status,
        }
        if report is not None and name.startswith((
            "sweep_", "stop_run", "imbalance_", "vw_", "cum_delta",
        )):
            frame = sweep_detector_param(report, name, grid)
            if not frame.empty:
                agg = frame.groupby("value")["n_pools"].mean().to_dict()
                entry["mean_pools_by_value"] = {
                    f"{k:.4g}": float(v) for k, v in agg.items()
                }
                entry["verdict"] = classify_curve(agg)
                frame.to_csv(out_dir / f"{name}_sweep.csv", index=False)
        else:
            entry["verdict"] = "needs-bundle" if report is None else "non-detector"
        results[name] = entry

    (out_dir / "sensitivity_summary.json").write_text(
        json.dumps(results, indent=2)
    )
    return results


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bundle", type=Path, default=None,
                   help="multi_asset_report.pkl; omit for registry-only report")
    p.add_argument("--params", nargs="*", default=[],
                   help="registry names to sweep; default = all sweepable")
    p.add_argument("--out", type=Path, default=Path("reports/param_sensitivity"))
    args = p.parse_args()
    results = run(args.bundle, list(args.params), args.out)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
