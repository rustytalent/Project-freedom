"""Compatibility wrapper for the safe daily Kite token helper.

Do not hardcode Kite API secrets in this file. Put them in .env or pass them as
CLI arguments to scripts/kite_daily_token.py.
"""
from __future__ import annotations

from scripts.kite_daily_token import main


if __name__ == "__main__":
    raise SystemExit(main())
