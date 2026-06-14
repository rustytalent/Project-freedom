"""Test bootstrap — put the sibling ``sentinel`` package on the path so
cross-codebase integration tests (live inference → Sentinel bridge,
sentinel ledger → flywheel adapter) can actually import both halves.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]    # /home/user/Project-freedom
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
