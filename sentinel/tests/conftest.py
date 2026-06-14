"""Test bootstrap — put the sibling ``liquidity_backtester`` package on
the path so cross-codebase integration tests (the launch-day soul test,
and any future cross-half check) can actually import both halves."""
from __future__ import annotations

import sys
from pathlib import Path

# /home/user/Project-freedom/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Also make the sibling repo importable directly (liqpool lives under
# liquidity_backtester/liqpool, not at the project root).
_LIQ_REPO = _PROJECT_ROOT / "liquidity_backtester"
if _LIQ_REPO.exists() and str(_LIQ_REPO) not in sys.path:
    sys.path.insert(0, str(_LIQ_REPO))
