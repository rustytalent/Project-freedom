"""Sentinel configuration — env-driven, dry-run by default.

Multi-account ready: ``accounts`` is a list; v1 uses the first entry.
Adding a second account later = adding a second env-var pair (or a
JSON file), not a refactor.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AccountConfig:
    label: str
    api_key: str
    access_token: str


@dataclass
class SentinelConfig:
    accounts: List[AccountConfig] = field(default_factory=list)
    # SAFETY: dry-run unless explicitly disabled. Even with dry_run
    # False, each order call still requires the engine's confirm flag.
    dry_run: bool = True
    demo_mode: bool = False           # synthetic account; no broker at all
    dash_token: Optional[str] = None  # X-Sentinel-Token header, if set
    journal_dir: Path = Path.home() / ".sentinel"
    underlying: str = "NIFTY"         # recommender's default chain
    poll_quote_seconds: float = 1.0   # Kite quote budget: 1 req/s
    poll_portfolio_seconds: float = 15.0
    ledger_resolve_minutes: float = 10.0
    max_orders_per_day: int = 50      # hard self-cap, far below Kite's 3000
    # Optional bridge to the research engine's trained organs.
    flywheel_dir: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "SentinelConfig":
        accounts: List[AccountConfig] = []
        # Primary account from the same env vars the research repo uses.
        api_key = os.environ.get("KITE_API_KEY", "")
        token = os.environ.get("KITE_ACCESS_TOKEN", "")
        if api_key and token:
            accounts.append(AccountConfig("primary", api_key, token))
        # Additional accounts: SENTINEL_ACCOUNTS_JSON = [{label, api_key, access_token}]
        extra = os.environ.get("SENTINEL_ACCOUNTS_JSON")
        if extra:
            try:
                for row in json.loads(extra):
                    accounts.append(AccountConfig(
                        row["label"], row["api_key"], row["access_token"]))
            except Exception:
                pass
        fly = os.environ.get("SENTINEL_FLYWHEEL_DIR")
        return cls(
            accounts=accounts,
            dry_run=os.environ.get("SENTINEL_CONFIRM_REAL", "") != "1",
            demo_mode=os.environ.get("SENTINEL_DEMO", "") == "1",
            dash_token=os.environ.get("SENTINEL_TOKEN") or None,
            journal_dir=Path(os.environ.get(
                "SENTINEL_JOURNAL_DIR", str(Path.home() / ".sentinel"))),
            underlying=os.environ.get("SENTINEL_UNDERLYING", "NIFTY"),
            flywheel_dir=Path(fly) if fly else None,
        )
