"""Sentinel adapter tests — verify the night trainer eats Sentinel's
shadow ledger via the same shape the flywheel expects.

Pins:
  * read_sentinel_ledger_dir picks up ledger_<date>.jsonl files
  * date-range filtering works
  * exported decision_events directory is read too
  * events → shadow-frame projection has the column shape the flywheel
    needs (shadow_id / event_kind / trading_date_ist / decision_context /
    counterfactual_outcome)
  * load_sentinel_as_shadow_frame returns None on empty input
  * train_flywheel.py's _load_shadow_joined concats liqpool + sentinel
    rows in one pile
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from liqpool.sentinel_adapter import (
    events_to_shadow_frame, load_sentinel_as_shadow_frame,
    read_decision_events_dir, read_sentinel_ledger_dir,
)


def _write_ledger(dirpath: Path, session: str, rows: list) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / f"ledger_{session}.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _sentinel_row(event_id, kind="virtual", session="2026-06-13",
                   scientist="sci", trust="TRUSTED",
                   instrument="NIFTY25000CE", complete=True):
    return {
        "event_id": event_id, "kind": kind,
        "ts_utc": f"{session}T05:00:00Z", "session": session,
        "scientist": scientist, "trust_tier": trust,
        "identity": {"instrument": instrument, "premium": 180,
                      "option_type": "CE", "strike": 25000,
                      "moneyness_key": "NIFTY:CE:+0"},
        "hypothesis": {"suggested_entry": 180, "suggested_stop": 160,
                        "suggested_target": 220, "confidence": 0.6,
                        "reason_codes": ["gamma"]},
        "journey": {"complete": complete, "mfe": 210, "mae": 175,
                     "time_to_target_min": 8,
                     "outcomes": {"t+60": 200}},
        "judgment": {"was_entry_good": True},
    }


def test_read_ledger_dir_collapses_per_event_id(tmp_path):
    """Two writes for the same event_id (a journey + a judgment) collapse
    to one DecisionEvent — last-write-wins per id, like the real ledger."""
    rows = [
        _sentinel_row("EV1", complete=False),
        _sentinel_row("EV1", complete=True),                  # newer
        _sentinel_row("EV2", complete=True),
    ]
    _write_ledger(tmp_path, "2026-06-13", rows)
    events = read_sentinel_ledger_dir(tmp_path)
    assert {ev.event_id for ev in events} == {"EV1", "EV2"}
    e1 = next(ev for ev in events if ev.event_id == "EV1")
    assert e1.journey_complete is True


def test_read_ledger_dir_date_range_pruning(tmp_path):
    _write_ledger(tmp_path, "2026-06-12",
                  [_sentinel_row("A", session="2026-06-12")])
    _write_ledger(tmp_path, "2026-06-13",
                  [_sentinel_row("B", session="2026-06-13")])
    _write_ledger(tmp_path, "2026-06-14",
                  [_sentinel_row("C", session="2026-06-14")])
    only_mid = read_sentinel_ledger_dir(
        tmp_path, since="2026-06-13", until="2026-06-13")
    assert {ev.event_id for ev in only_mid} == {"B"}


def test_read_ledger_dir_skips_malformed_lines(tmp_path):
    """One bad JSON line in the middle of the file doesn't kill the file."""
    rows = [_sentinel_row("EV1"), _sentinel_row("EV2")]
    p = tmp_path / "ledger_2026-06-13.jsonl"
    with p.open("w") as f:
        f.write(json.dumps(rows[0]) + "\n")
        f.write("{not valid json\n")
        f.write(json.dumps(rows[1]) + "\n")
    events = read_sentinel_ledger_dir(tmp_path)
    assert {ev.event_id for ev in events} == {"EV1", "EV2"}


def test_read_decision_events_dir_reads_exported_layout(tmp_path):
    """Exports land at <root>/sentinel_live/session=<date>/decision_events.jsonl."""
    layer = tmp_path / "sentinel_live" / "session=2026-06-13"
    layer.mkdir(parents=True)
    rows = [
        {"event_id": "EV1", "event_type": "VIRTUAL_DECISION",
          "session_date": "2026-06-13", "instrument": "X", "trust_tier": "TRUSTED"},
        {"event_id": "EV2", "event_type": "EXECUTED",
          "session_date": "2026-06-13", "instrument": "Y"},
    ]
    with (layer / "decision_events.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    events = read_decision_events_dir(tmp_path)
    assert {ev.event_id for ev in events} == {"EV1", "EV2"}


def test_events_to_shadow_frame_has_flywheel_columns(tmp_path):
    """Projection must have the same columns the flywheel's
    join_events_to_resolutions already returns, so concat works."""
    rows = [_sentinel_row("EV1", session="2026-06-13", complete=True)]
    _write_ledger(tmp_path, "2026-06-13", rows)
    events = read_sentinel_ledger_dir(tmp_path)
    frame = events_to_shadow_frame(events)
    for col in ("shadow_id", "event_kind", "trading_date_ist",
                "written_at_utc", "symbol", "decision_context",
                "regime_tags", "resolved_at_utc",
                "trading_date_ist_resolved", "counterfactual_outcome"):
        assert col in frame.columns
    assert frame.iloc[0]["symbol"] == "NIFTY25000CE"
    assert frame.iloc[0]["event_kind"].startswith("sentinel_")
    assert "TRUSTED" in frame.iloc[0]["regime_tags"]
    # journey outcome lands in counterfactual_outcome
    co = frame.iloc[0]["counterfactual_outcome"]
    assert co and co["realized_outcome_t60"] == 200


def test_events_to_shadow_frame_marks_incomplete_no_outcome(tmp_path):
    rows = [_sentinel_row("EV1", complete=False)]
    _write_ledger(tmp_path, "2026-06-13", rows)
    events = read_sentinel_ledger_dir(tmp_path)
    frame = events_to_shadow_frame(events)
    assert frame.iloc[0]["counterfactual_outcome"] is None
    assert frame.iloc[0]["resolved_at_utc"] is None


def test_load_sentinel_as_shadow_frame_returns_none_on_empty(tmp_path):
    assert load_sentinel_as_shadow_frame(journal_dir=tmp_path) is None


def test_load_sentinel_as_shadow_frame_combines_journal_and_export(tmp_path):
    journal = tmp_path / "journal"
    export = tmp_path / "export"
    _write_ledger(journal, "2026-06-13",
                  [_sentinel_row("EV_J", session="2026-06-13")])
    layer = export / "sentinel_live" / "session=2026-06-13"
    layer.mkdir(parents=True)
    with (layer / "decision_events.jsonl").open("w") as f:
        f.write(json.dumps({"event_id": "EV_E", "event_type": "EXECUTED",
                             "session_date": "2026-06-13",
                             "instrument": "Y"}) + "\n")
    frame = load_sentinel_as_shadow_frame(
        journal_dir=journal, export_dir=export)
    assert frame is not None
    assert set(frame["shadow_id"]) == {"EV_J", "EV_E"}


def test_train_flywheel_helper_concats_both_sides(tmp_path):
    """The script's _load_shadow_joined must merge liqpool's own shadow
    frame with the Sentinel-converted frame in one pile."""
    # No liqpool shadow dir, but Sentinel input present.
    _write_ledger(tmp_path / "journal", "2026-06-13",
                  [_sentinel_row("EV_J", session="2026-06-13")])
    from scripts.train_flywheel import _load_shadow_joined
    out = _load_shadow_joined(
        root=None,
        sentinel_journal=tmp_path / "journal",
        sentinel_export=None)
    assert out is not None
    assert "EV_J" in set(out["shadow_id"])
