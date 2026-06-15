"""Test bootstrap — make both the package root AND the project root importable.

The package root (parent of ``liqpool/``) is needed so tests can ``import
liqpool.learned_gate``. The project root (parent of ``liquidity_backtester/``)
is needed so cross-codebase tests can pull in the sibling ``sentinel`` package
(live inference -> Sentinel bridge, ledger -> flywheel adapter)."""
from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]    # /home/user/Project-freedom/liquidity_backtester
_PROJECT_ROOT = Path(__file__).resolve().parents[2]    # /home/user/Project-freedom

for _path in (_PKG_ROOT, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
