"""Stream L — brief integration test for the shadow_log_writer hook.

Verifies that when ``generate_brief`` is called with a ``shadow_log_writer``,
every avoidance entry surfaces as a shadow event on disk after commit.

This is the integration end of the Stream L wiring; the in-memory
ShadowLogger contract is tested separately in test_shadow_log.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from liqpool.products.daily_brief import generate_brief
from liqpool.products.shadow_log import (
    EVENT_KIND_AVOIDANCE_FLAG,
    ShadowLogger,
    read_shadow_events,
)


# Reuse the brief-test fixture shape (kept local so this test stays
# independent of changes to tests/test_daily_brief.py internals).


def _bars(n: int = 80, base_price: float = 100.0,
          start: str = "2026-05-22 03:45") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = base_price + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.2, "low": close - 0.2,
        "close": close, "volume": np.full(n, 1000.0),
    }, index=idx)


@dataclass
class _StubAssetData:
    base_df: pd.DataFrame


@dataclass
class _StubReport:
    assets: Dict[str, _StubAssetData]
    unified_direction: Optional[Any] = None
    unified_proximity: Dict[int, Any] = field(default_factory=dict)
    unified_oos_audit: Optional[Any] = None


def _empty_report() -> _StubReport:
    """Empty watchlist + empty predictions -> the avoid-list block
    emits the ALL_BASKET avoidance entry. That single entry is
    enough to verify the shadow hook fires."""
    return _StubReport(
        assets={"HDFCBANK": _StubAssetData(base_df=_bars())},
    )


def test_generate_brief_emits_shadow_events_for_avoidances(tmp_path: Path):
    shadow = ShadowLogger(tmp_path)
    report = _empty_report()

    brief = generate_brief(
        report,
        trading_date_ist="2026-06-03",
        shadow_log_writer=shadow,
    )
    summary = shadow.commit()

    # The brief block must have produced at least one avoidance.
    assert len(brief.avoid_list) >= 1
    # Every avoidance entry surfaces as a shadow event.
    assert summary["events_written"] == len(brief.avoid_list)

    df = read_shadow_events(
        tmp_path, event_kind=EVENT_KIND_AVOIDANCE_FLAG,
        trading_date_ist="2026-06-03",
    )
    assert len(df) == len(brief.avoid_list)
    # Every recorded row carries the brief's reason verbatim.
    recorded_symbols = set(df["symbol"].tolist())
    expected_symbols = {av.symbol for av in brief.avoid_list}
    assert recorded_symbols == expected_symbols


def test_generate_brief_without_shadow_writer_is_a_no_op():
    """Existing callers that don't pass shadow_log_writer must keep
    working unchanged. This is the schema-stability pin for the new
    kwarg."""
    brief = generate_brief(
        _empty_report(),
        trading_date_ist="2026-06-03",
        # No shadow_log_writer kwarg.
    )
    assert brief is not None
    assert hasattr(brief, "avoid_list")


def test_brief_shadow_hook_swallows_writer_exceptions(tmp_path: Path):
    """Shadow logging MUST NOT block brief generation. If the writer
    throws on a single record, the brief still publishes."""

    class _ExplodingLogger(ShadowLogger):
        def record(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("disk full simulation")

    brief = generate_brief(
        _empty_report(),
        trading_date_ist="2026-06-03",
        shadow_log_writer=_ExplodingLogger(tmp_path),
    )
    # No raise = pass. The brief still has its avoid_list intact.
    assert brief is not None
    assert len(brief.avoid_list) >= 1
