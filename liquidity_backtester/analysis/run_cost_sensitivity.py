"""Compatibility wrapper for :mod:`analysis.cost_sensitivity`.

The canonical implementation is ``analysis/cost_sensitivity.py``. This wrapper
accepts the newer runbook style ``--bundle core25_head_alpha`` and translates it
to ``--model-dir output_models/core25_head_alpha``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis import cost_sensitivity


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
    ap.add_argument("--split", default="oos", choices=("oos", "train"))
    ap.add_argument("--notionals", default="50000,100000,200000")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="output_audit")
    ap.add_argument("--cost-multipliers", default="",
                    help="accepted for runbook compatibility; cost multipliers are reported by run_arsenal.py")
    args = ap.parse_args()
    if args.cost_multipliers:
        print("[run-cost-sens] note: --cost-multipliers is ignored here; "
              "run_arsenal.py emits multiplier-stressed alpha costs.")
    sys.argv = [
        "cost_sensitivity.py",
        "--model-dir", _model_dir(args.bundle, args.model_dir),
        "--split", args.split,
        "--notionals", args.notionals,
        "--workers", str(args.workers),
        "--out", args.out,
    ]
    return cost_sensitivity.main()


if __name__ == "__main__":
    raise SystemExit(main())
