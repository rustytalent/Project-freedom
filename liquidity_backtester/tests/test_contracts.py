"""Tests for the shared contracts package — the cross-codebase spine
Codex called out as missing.

What's pinned:
  * DecisionEvent + ModelSignal + TrustPromotionRecord +
    ResearchContextPack all round-trip through dict cleanly
  * Sentinel-shape rows convert to canonical DecisionEvent
  * ShadowEvent rows convert to canonical DecisionEvent
  * Unknown fields land in `extras` rather than throwing
"""
from __future__ import annotations

import pytest

from liqpool.contracts import (
    CONTRACT_VERSION, DecisionEvent, EVENT_TYPES, ModelSignal,
    ResearchContextPack, RunManifest, TrustPromotionRecord, ZoneLevel,
)
from liqpool.contracts.events import (
    SENTINEL_KIND_TO_EVENT_TYPE, from_sentinel_ledger_row,
    from_shadow_event_row,
)
from liqpool.contracts.manifest import StageStatus, fresh_manifest


# ---------------------------------------------------------------------------
# DecisionEvent
# ---------------------------------------------------------------------------

def test_decision_event_round_trips_minimal():
    ev = DecisionEvent(event_id="X", event_type="VIRTUAL_DECISION",
                        source="sentinel", session_date="2026-06-14",
                        instrument="NIFTY25000CE", confidence=0.7,
                        reason_codes=["a", "b"])
    row = ev.to_row()
    back = DecisionEvent.from_row(row)
    assert back.event_id == "X" and back.confidence == 0.7
    assert back.reason_codes == ["a", "b"]
    assert back.contract_version == CONTRACT_VERSION


def test_decision_event_drops_empty_keys_in_export():
    ev = DecisionEvent(event_id="X", event_type="EXECUTED")
    row = ev.to_row(drop_empty=True)
    assert "confidence" not in row     # None default dropped
    assert "extras" not in row          # empty dict dropped
    # but core identity survives
    assert row["event_id"] == "X"
    assert row["event_type"] == "EXECUTED"
    assert row["contract_version"] == CONTRACT_VERSION


def test_decision_event_unknown_keys_land_in_extras():
    row = {"event_id": "X", "event_type": "EXECUTED",
            "made_up_field": "yes", "another": 42}
    ev = DecisionEvent.from_row(row)
    assert ev.extras["made_up_field"] == "yes"
    assert ev.extras["another"] == 42


def test_sentinel_kind_to_event_type_complete():
    """Every Sentinel ledger kind has a mapping."""
    expected = {"actual", "virtual", "rejected", "alternative",
                "model_suggestion", "counterfactual"}
    assert expected <= set(SENTINEL_KIND_TO_EVENT_TYPE)


def test_from_sentinel_ledger_row_extracts_blocks():
    row = {
        "event_id": "EV_1", "kind": "virtual", "session": "2026-06-14",
        "ts_utc": "2026-06-14T05:00:00Z", "scientist": "second_pullback",
        "trust_tier": "TRUSTED",
        "identity": {"instrument": "NIFTY25000CE", "option_type": "CE",
                      "strike": 25000, "premium": 180, "moneyness_key": "NIFTY:CE:+0"},
        "hypothesis": {"suggested_entry": 180, "suggested_stop": 160,
                        "suggested_target": 230, "confidence": 0.66,
                        "reason_codes": ["gamma", "vwap"]},
        "journey": {"complete": True, "mfe": 220, "mae": 175,
                     "time_to_target_min": 8, "outcomes": {"t+60": 215}},
        "judgment": {"was_entry_good": True},
    }
    ev = from_sentinel_ledger_row(row)
    assert ev.event_type == "VIRTUAL_DECISION"
    assert ev.source == "sentinel" and ev.trust_tier == "TRUSTED"
    assert ev.instrument == "NIFTY25000CE" and ev.strike == 25000
    assert ev.confidence == 0.66
    assert ev.reason_codes == ["gamma", "vwap"]
    assert ev.mfe == 220 and ev.realized_outcome_t60 == 215
    assert ev.journey_complete is True
    assert ev.was_entry_good is True


def test_from_shadow_event_row_handles_string_decision_context():
    """liqpool's ShadowEvent often stores decision_context as a JSON
    string in Parquet — adapter must parse it."""
    import json
    row = {
        "shadow_id": "S1", "event_kind": "skip_options_executor",
        "trading_date_ist": "2026-06-14",
        "written_at_utc": "2026-06-14T05:00:00Z",
        "symbol": "HDFCBANK",
        "decision_context": json.dumps({"confidence": 0.3, "reason": "below_ratio"}),
        "regime_tags": ["choppy"],
    }
    ev = from_shadow_event_row(row)
    assert ev.event_type == "SKIP"
    assert ev.source == "liqpool"
    assert ev.confidence == 0.3
    assert ev.extras["decision_context"]["reason"] == "below_ratio"
    assert ev.reason_codes == ["choppy"]


# ---------------------------------------------------------------------------
# ModelSignal
# ---------------------------------------------------------------------------

def test_model_signal_round_trip_with_zone():
    sig = ModelSignal(ts_ist="10:00:00", asset="NIFTY",
                       model="direction_model", signal="upside 70%",
                       confidence=0.7, zone=(25000.0, 25100.0),
                       reason_codes=["a"], source="liqpool")
    row = sig.to_row()
    assert row["zone"] == [25000.0, 25100.0]
    back = ModelSignal.from_row(row)
    assert back.zone == (25000.0, 25100.0)
    assert back.source == "liqpool"


def test_model_signal_tolerates_unknown_fields():
    row = {"ts_ist": "t", "asset": "X", "model": "m", "signal": "s",
            "confidence": 0.5, "made_up_field": "ok"}
    sig = ModelSignal.from_row(row)
    assert sig.extras["made_up_field"] == "ok"


# ---------------------------------------------------------------------------
# TrustPromotionRecord
# ---------------------------------------------------------------------------

def test_promotion_record_round_trip():
    rec = TrustPromotionRecord(source="sci", from_tier="SHADOW",
                                to_tier="TRUSTED", decision="promoted",
                                ts_utc="t", n=42, hit_rate=0.6)
    back = TrustPromotionRecord.from_row(rec.to_row())
    assert back.decision == "promoted" and back.n == 42


# ---------------------------------------------------------------------------
# ResearchContextPack
# ---------------------------------------------------------------------------

def test_research_context_pack_round_trip_with_zones():
    pack = ResearchContextPack(
        session_date="2026-06-14", direction_probability=0.55,
        sector_regime="BANK_HOT",
        key_zones=[ZoneLevel(label="PDH", price=25180,
                              side="resistance", horizon_minutes=60,
                              reach_probability=0.55)])
    row = pack.to_row()
    back = ResearchContextPack.from_row(row)
    assert back.session_date == "2026-06-14"
    assert back.sector_regime == "BANK_HOT"
    assert back.key_zones[0].label == "PDH"


def test_research_context_pack_is_empty_when_blank():
    assert ResearchContextPack().is_empty


# ---------------------------------------------------------------------------
# RunManifest
# ---------------------------------------------------------------------------

def test_fresh_manifest_carries_mode_and_command():
    m = fresh_manifest(run_tag="nightly_2026_06_14", mode="train",
                       command="python -m scripts.train_flywheel")
    assert m.mode == "train"
    assert m.started_at_utc
    assert m.command


def test_run_manifest_round_trip_saves_and_loads(tmp_path):
    m = RunManifest(run_tag="t1", mode="train")
    m.stages.append(StageStatus(name="load_bundle", status="ok"))
    p = m.save_json(tmp_path / "manifest.json")
    assert p.exists()
    import json
    blob = json.loads(p.read_text())
    back = RunManifest.from_row(blob)
    assert back.run_tag == "t1" and back.stages[0].status == "ok"
