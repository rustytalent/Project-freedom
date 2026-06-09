"""Stream L — counterfactual replay resolver tests.

Pins:
  * avoidance resolver computes ret_close_over_open / MFE / MAE correctly
  * verdict labeling (avoidance_vindicated vs avoidance_regret) matches
    the sign of the next-session return
  * missing next-session data leaves the event unresolved (no crash)
  * unknown event_kind is gracefully unresolvable
  * batch replay is idempotent (re-running collapses to same outputs)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.replay_shadow_counterfactuals import (
    _resolve_avoidance,
    replay_one_date,
)
from liqpool.products.shadow_log import (
    EVENT_KIND_AVOIDANCE_FLAG,
    ShadowLogger,
    join_events_to_resolutions,
    read_shadow_events,
)


def _write_warehouse(
    bundle_root: Path, symbol: str, day: str, open_px: float, close_px: float,
    high_px: float = None, low_px: float = None,
) -> None:
    """Build a 5-minute-bar parquet for one IST trading day."""
    high_px = high_px if high_px is not None else max(open_px, close_px) + 1.0
    low_px = low_px if low_px is not None else min(open_px, close_px) - 1.0
    bars = pd.date_range(f"{day} 03:45", periods=75, freq="5min")  # ~09:15-15:15 IST
    closes = np.linspace(open_px, close_px, len(bars))
    df = pd.DataFrame({
        "open": closes, "high": high_px, "low": low_px,
        "close": closes, "volume": 1000.0,
    }, index=bars)
    bundle_root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(bundle_root / f"{symbol}.parquet")


def _seed_event(shadow_root: Path, symbol: str, event_date: str) -> str:
    logger = ShadowLogger(shadow_root)
    sid = logger.record(
        event_kind=EVENT_KIND_AVOIDANCE_FLAG,
        trading_date_ist=event_date,
        symbol=symbol,
        detail_token="sector_trending_down",
        decision_context={"reason": "sector_trending_down"},
    )
    logger.commit()
    return sid


def test_avoidance_resolver_marks_vindicated_when_next_day_drops(tmp_path):
    bundle = tmp_path / "warehouse"
    _write_warehouse(bundle, "HDFCBANK", "2026-06-10",
                     open_px=100.0, close_px=98.0,
                     high_px=100.5, low_px=97.0)
    sid = _seed_event(tmp_path, "HDFCBANK", "2026-06-09")

    summary = replay_one_date(tmp_path, bundle, "2026-06-09")
    assert summary["scanned"] == 1
    assert summary["resolved"] == 1

    joined = join_events_to_resolutions(tmp_path,
                                        event_kind=EVENT_KIND_AVOIDANCE_FLAG)
    row = joined[joined["shadow_id"] == sid].iloc[0]
    import json as _json
    outcome = _json.loads(row["counterfactual_outcome"])
    assert outcome["verdict"] == "avoidance_vindicated"
    assert outcome["ret_close_over_open"] < 0


def test_avoidance_resolver_marks_regret_when_next_day_rallies(tmp_path):
    bundle = tmp_path / "warehouse"
    _write_warehouse(bundle, "TCS", "2026-06-10",
                     open_px=100.0, close_px=104.0,
                     high_px=105.0, low_px=99.5)
    sid = _seed_event(tmp_path, "TCS", "2026-06-09")

    replay_one_date(tmp_path, bundle, "2026-06-09")

    joined = join_events_to_resolutions(tmp_path,
                                        event_kind=EVENT_KIND_AVOIDANCE_FLAG)
    row = joined[joined["shadow_id"] == sid].iloc[0]
    import json as _json
    outcome = _json.loads(row["counterfactual_outcome"])
    assert outcome["verdict"] == "avoidance_regret"
    assert outcome["ret_close_over_open"] > 0
    # MFE >= close-over-open >= 0 when next day rallies.
    assert outcome["mfe_close_over_open"] >= outcome["ret_close_over_open"]


def test_missing_warehouse_leaves_event_unresolvable(tmp_path):
    """No <symbol>.parquet -> resolver returns None, the event stays
    unresolved, the batch driver doesn't crash."""
    bundle = tmp_path / "warehouse"
    bundle.mkdir()  # exists but empty
    _seed_event(tmp_path, "MISSING_SYMBOL", "2026-06-09")

    summary = replay_one_date(tmp_path, bundle, "2026-06-09")
    assert summary["scanned"] == 1
    assert summary["resolved"] == 0
    assert summary["unresolvable"] == 1


def test_replay_is_idempotent(tmp_path):
    """Running the replay twice on the same date must collapse to one
    resolution per event_id via the shadow_id dedup."""
    bundle = tmp_path / "warehouse"
    _write_warehouse(bundle, "HDFCBANK", "2026-06-10",
                     open_px=100.0, close_px=98.0)
    _seed_event(tmp_path, "HDFCBANK", "2026-06-09")

    replay_one_date(tmp_path, bundle, "2026-06-09")
    replay_one_date(tmp_path, bundle, "2026-06-09")

    joined = join_events_to_resolutions(tmp_path)
    assert len(joined) == 1  # not duplicated
    assert pd.notna(joined.iloc[0]["trading_date_ist_resolved"])


def test_unknown_event_kind_counted_as_unresolvable(tmp_path):
    """A shadow event with an event_kind that has no RESOLVERS entry
    counts as unresolvable rather than blowing up the batch."""
    bundle = tmp_path / "warehouse"
    bundle.mkdir()
    logger = ShadowLogger(tmp_path)
    logger.record(
        event_kind="some_future_kind_not_in_resolvers",
        trading_date_ist="2026-06-09",
        symbol="X", detail_token="d",
        decision_context={},
    )
    logger.commit()

    summary = replay_one_date(tmp_path, bundle, "2026-06-09")
    assert summary["scanned"] == 1
    assert summary["resolved"] == 0
    assert summary["unresolvable"] == 1


def test_resolve_avoidance_returns_none_for_missing_symbol(tmp_path):
    """Direct unit test of the resolver: a NaN/empty symbol yields None
    rather than crashing on the .parquet lookup."""
    ev = pd.Series({
        "symbol": None,
        "trading_date_ist": "2026-06-09",
    })
    assert _resolve_avoidance(tmp_path / "warehouse", ev) is None
