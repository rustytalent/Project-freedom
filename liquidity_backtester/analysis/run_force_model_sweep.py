"""Compatibility wrapper for :mod:`analysis.force_model_sweep`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis import force_model_sweep


def _model_dir(bundle: str, model_dir: str) -> str:
    if bundle:
        p = Path(bundle).expanduser()
        if (p / "multi_asset_report.pkl").exists():
            return str(p)
        if p.name == str(p):
            return str(Path("output_models") / str(p))
        return str(p)
    return model_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="")
    ap.add_argument("--model-dir", default="output_models/core25_latest")
    ap.add_argument("--use-extended-features", default="false")
    ap.add_argument("--out", default="output_audit/force_model")
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--top-decile-pct", type=float, default=0.10)
    args = ap.parse_args()
    sys.argv = [
        "force_model_sweep.py",
        "--model-dir", _model_dir(args.bundle, args.model_dir),
        "--use-extended-features", str(args.use_extended_features),
        "--out", args.out,
        "--atr-period", str(args.atr_period),
        "--top-decile-pct", str(args.top_decile_pct),
    ]
    return force_model_sweep.main()


if __name__ == "__main__":
    raise SystemExit(main())
