"""Stream L — ShadowLogger tests.

Pins:
  * Append-only semantics (existing partitions are merged, not blown away).
  * Idempotent re-runs (same shadow_id collapses to one row, keep="last").
  * Atomic commit (no half-written partitions visible to readers).
  * Read API filters by date and kind correctly.
  * Resolution join is a left-join (unresolved events surface NaN).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from liqpool.products.shadow_log import (
    EVENT_KIND_AVOIDANCE_FLAG,
    EVENT_KIND_MACRO_GATE_CLOSED,
    EVENT_KIND_SKIP_OPTIONS,
    ShadowLogger,
    join_events_to_resolutions,
    make_shadow_id,
    read_shadow_events,
    read_shadow_resolutions,
)


# ---------------------------------------------------------------------------
# ID determinism
# ---------------------------------------------------------------------------

def test_make_shadow_id_is_deterministic():
    a = make_shadow_id("2026-06-09", EVENT_KIND_SKIP_OPTIONS, "HDFCBANK", "lvl_1800")
    b = make_shadow_id("2026-06-09", EVENT_KIND_SKIP_OPTIONS, "HDFCBANK", "lvl_1800")
    assert a == b


def test_make_shadow_id_distinguishes_kinds():
    a = make_shadow_id("2026-06-09", EVENT_KIND_SKIP_OPTIONS, "HDFCBANK", "x")
    b = make_shadow_id("2026-06-09", EVENT_KIND_MACRO_GATE_CLOSED, "HDFCBANK", "x")
    assert a != b


# ---------------------------------------------------------------------------
# Write + commit
# ---------------------------------------------------------------------------

def test_commit_writes_partitioned_parquet(tmp_path: Path):
    logger = ShadowLogger(tmp_path)
    logger.record(
        event_kind=EVENT_KIND_SKIP_OPTIONS,
        trading_date_ist="2026-06-09",
        symbol="NIFTY",
        detail_token="strike_24500_PE",
        decision_context={"lcs": -0.05, "macro_gate": "closed"},
        regime_tags=["high_vol"],
    )
    logger.record(
        event_kind=EVENT_KIND_AVOIDANCE_FLAG,
        trading_date_ist="2026-06-09",
        symbol="HDFCBANK",
        detail_token="basket_BANKING",
        decision_context={"reason": "sector_trending_down"},
    )
    summary = logger.commit()
    assert summary["events_written"] == 2
    assert summary["resolutions_written"] == 0
    # Partition layout
    assert (tmp_path / "shadow_events" / "trading_date_ist=2026-06-09"
            / "event_kind=skip_options_executor" / "events.parquet").exists()
    assert (tmp_path / "shadow_events" / "trading_date_ist=2026-06-09"
            / "event_kind=avoidance_flag" / "events.parquet").exists()


def test_round_trip_via_read_api(tmp_path: Path):
    logger = ShadowLogger(tmp_path)
    logger.record(
        event_kind=EVENT_KIND_SKIP_OPTIONS,
        trading_date_ist="2026-06-09",
        symbol="NIFTY",
        detail_token="x",
        decision_context={"foo": 1, "bar": "baz"},
        regime_tags=["a", "b"],
    )
    logger.commit()
    df = read_shadow_events(tmp_path, event_kind=EVENT_KIND_SKIP_OPTIONS)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["symbol"] == "NIFTY"
    assert row["trading_date_ist"] == "2026-06-09"
    # decision_context is stored as a JSON string for column stability
    ctx = json.loads(row["decision_context"])
    assert ctx == {"foo": 1, "bar": "baz"}
    tags = json.loads(row["regime_tags"])
    assert tags == ["a", "b"]


# ---------------------------------------------------------------------------
# Idempotence + append
# ---------------------------------------------------------------------------

def test_repeated_record_collapses_via_dedupe(tmp_path: Path):
    """Recording the same shadow row twice across two ShadowLogger
    sessions must not duplicate. The dedupe key is shadow_id; the
    second write replaces the first via keep='last'."""
    # Session 1
    l1 = ShadowLogger(tmp_path)
    l1.record(
        event_kind=EVENT_KIND_SKIP_OPTIONS,
        trading_date_ist="2026-06-09",
        symbol="NIFTY",
        detail_token="t",
        decision_context={"v": 1},
    )
    l1.commit()
    # Session 2 — same logical event, but with updated context
    l2 = ShadowLogger(tmp_path)
    l2.record(
        event_kind=EVENT_KIND_SKIP_OPTIONS,
        trading_date_ist="2026-06-09",
        symbol="NIFTY",
        detail_token="t",
        decision_context={"v": 2},  # corrected
    )
    l2.commit()
    df = read_shadow_events(tmp_path, event_kind=EVENT_KIND_SKIP_OPTIONS)
    assert len(df) == 1, "duplicate shadow_id not deduplicated"
    assert json.loads(df.iloc[0]["decision_context"])["v"] == 2


def test_append_preserves_prior_partitions(tmp_path: Path):
    """A second session adding a NEW event to the same partition must
    preserve the existing row, not overwrite it."""
    l1 = ShadowLogger(tmp_path)
    l1.record(
        event_kind=EVENT_KIND_SKIP_OPTIONS,
        trading_date_ist="2026-06-09",
        symbol="NIFTY",
        detail_token="first",
        decision_context={"k": 1},
    )
    l1.commit()
    l2 = ShadowLogger(tmp_path)
    l2.record(
        event_kind=EVENT_KIND_SKIP_OPTIONS,
        trading_date_ist="2026-06-09",
        symbol="NIFTY",
        detail_token="second",
        decision_context={"k": 2},
    )
    l2.commit()
    df = read_shadow_events(tmp_path)
    assert len(df) == 2
    contexts = sorted(json.loads(c)["k"] for c in df["decision_context"])
    assert contexts == [1, 2]


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def test_no_tmp_files_after_commit(tmp_path: Path):
    """The atomic write goes through a .tmp sibling. After commit() no
    .tmp file may remain — that would mean a half-written partition is
    visible to readers."""
    logger = ShadowLogger(tmp_path)
    logger.record(
        event_kind=EVENT_KIND_AVOIDANCE_FLAG,
        trading_date_ist="2026-06-09",
        symbol="HDFCBANK",
        detail_token="d",
        decision_context={},
    )
    logger.commit()
    leftovers = list((tmp_path / "shadow_events").rglob("*.tmp"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# Resolution join
# ---------------------------------------------------------------------------

def test_resolution_join_left_joins_with_unresolved_nans(tmp_path: Path):
    logger = ShadowLogger(tmp_path)
    sid_resolved = logger.record(
        event_kind=EVENT_KIND_SKIP_OPTIONS,
        trading_date_ist="2026-06-09",
        symbol="NIFTY",
        detail_token="r",
        decision_context={"side": "long"},
    )
    sid_unresolved = logger.record(
        event_kind=EVENT_KIND_SKIP_OPTIONS,
        trading_date_ist="2026-06-09",
        symbol="NIFTY",
        detail_token="u",
        decision_context={"side": "long"},
    )
    logger.record_resolution(
        shadow_id=sid_resolved,
        trading_date_ist_resolved="2026-06-10",
        counterfactual_outcome={"net_r": 0.42},
    )
    logger.commit()

    joined = join_events_to_resolutions(
        tmp_path, event_kind=EVENT_KIND_SKIP_OPTIONS
    )
    assert len(joined) == 2
    by_id = {row["shadow_id"]: row for _, row in joined.iterrows()}
    # Resolved row carries the resolution date
    assert (
        by_id[sid_resolved]["trading_date_ist_resolved"] == "2026-06-10"
    )
    # Unresolved row's resolution columns are NaN
    assert pd.isna(by_id[sid_unresolved]["trading_date_ist_resolved"])


# ---------------------------------------------------------------------------
# Read API filters
# ---------------------------------------------------------------------------

def test_read_filters_by_event_kind(tmp_path: Path):
    logger = ShadowLogger(tmp_path)
    logger.record(
        event_kind=EVENT_KIND_SKIP_OPTIONS,
        trading_date_ist="2026-06-09",
        symbol="NIFTY", detail_token="a",
        decision_context={},
    )
    logger.record(
        event_kind=EVENT_KIND_AVOIDANCE_FLAG,
        trading_date_ist="2026-06-09",
        symbol="HDFCBANK", detail_token="b",
        decision_context={},
    )
    logger.commit()
    skips = read_shadow_events(tmp_path, event_kind=EVENT_KIND_SKIP_OPTIONS)
    assert len(skips) == 1
    assert skips.iloc[0]["event_kind"] == EVENT_KIND_SKIP_OPTIONS


def test_read_filters_by_date(tmp_path: Path):
    logger = ShadowLogger(tmp_path)
    logger.record(
        event_kind=EVENT_KIND_SKIP_OPTIONS,
        trading_date_ist="2026-06-09",
        symbol="NIFTY", detail_token="a",
        decision_context={},
    )
    logger.record(
        event_kind=EVENT_KIND_SKIP_OPTIONS,
        trading_date_ist="2026-06-10",
        symbol="NIFTY", detail_token="b",
        decision_context={},
    )
    logger.commit()
    d1 = read_shadow_events(tmp_path, trading_date_ist="2026-06-09")
    assert len(d1) == 1
    assert d1.iloc[0]["trading_date_ist"] == "2026-06-09"


def test_empty_root_returns_empty_frame(tmp_path: Path):
    df = read_shadow_events(tmp_path)
    assert df.empty
    resos = read_shadow_resolutions(tmp_path)
    assert resos.empty


# ---------------------------------------------------------------------------
# Type safety
# ---------------------------------------------------------------------------

def test_decision_context_must_be_dict(tmp_path: Path):
    logger = ShadowLogger(tmp_path)
    with pytest.raises(TypeError):
        logger.record(
            event_kind=EVENT_KIND_SKIP_OPTIONS,
            trading_date_ist="2026-06-09",
            symbol="NIFTY", detail_token="x",
            decision_context="not_a_dict",  # type: ignore
        )


def test_resolution_outcome_must_be_dict(tmp_path: Path):
    logger = ShadowLogger(tmp_path)
    with pytest.raises(TypeError):
        logger.record_resolution(
            shadow_id="any",
            trading_date_ist_resolved="2026-06-10",
            counterfactual_outcome="not_a_dict",  # type: ignore
        )
