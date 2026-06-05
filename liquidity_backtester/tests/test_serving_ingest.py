"""Tests for the request handler and the raw-output ingestion engine."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from liqpool.scoring import PUBLIC_KEYS
from liqpool.serving import StubScorer, handle_levels_request
from liqpool.ingest import (
    RawFeedScorer,
    read_raw_levels,
    row_to_internal_level,
    levels_from_rows,
)


# ---- request handler ------------------------------------------------------
def test_handle_request_envelope_and_records():
    out = handle_levels_request(StubScorer(), symbol="HDFCBANK.NS",
                                date="2026-05-29", customer_id="cust-a")
    assert out["instrument"] == "HDFCBANK.NS"
    assert out["observation_count"] == len(out["observations"])
    assert "not investment advice" in out["compliance_notice"]["notice"].lower()
    assert out["compliance_notice"]["classification"] == "non_recommendatory_market_analytics"
    assert out["observations"], "expected some observations"
    for rec in out["observations"]:
        assert set(rec.keys()) == set(PUBLIC_KEYS)


def test_demo_tier_fuzzes_served_level_zones():
    licensed = handle_levels_request(StubScorer(n_above=1, n_below=0),
                                     symbol="HDFCBANK.NS",
                                     date="2026-05-29",
                                     customer_id="cust-a",
                                     customer_tier="licensed")
    demo = handle_levels_request(StubScorer(n_above=1, n_below=0),
                                 symbol="HDFCBANK.NS",
                                 date="2026-05-29",
                                 customer_id="cust-a",
                                 customer_tier="demo")
    assert licensed["observations"][0]["level_zone"] != demo["observations"][0]["level_zone"]


def test_handle_request_requires_symbol_and_customer():
    with pytest.raises(ValueError):
        handle_levels_request(StubScorer(), symbol="", date="d", customer_id="c")
    with pytest.raises(ValueError):
        handle_levels_request(StubScorer(), symbol="X", date="d", customer_id="")


# ---- ingestion mapping ----------------------------------------------------
def test_row_mapping_prefers_real_probability():
    row = {"symbol": "X.NS", "side": "above", "pool_low": 99, "pool_high": 101,
           "p_touch": 0.8, "p_up": 0.7, "q": 0.6}
    lvl = row_to_internal_level(row, "2026-05-29")
    assert lvl.symbol == "X.NS"
    assert lvl.p_touch == 0.8 and lvl.p_up == 0.7 and lvl.q == 0.6
    assert lvl.level_mid == 100.0


def test_row_mapping_derives_p_up_from_direction_to_pool():
    row = {"symbol": "X.NS", "side": "below", "pool_low": 99, "pool_high": 101,
           "p_touch": 0.5, "p_direction_to_pool": 0.8, "p_respect": 0.5}
    lvl = row_to_internal_level(row, "2026-05-29")
    # below + toward-pool 0.8 => up-prob 0.2
    assert abs(lvl.p_up - 0.2) < 1e-9


def test_row_mapping_derives_p_up_from_dir_tag():
    row = {"symbol": "X.NS", "side": "above", "pool_mid": 100,
           "t_today": 0.4, "dir_tag": "DIR_ALIGN", "q": 0.5}
    lvl = row_to_internal_level(row, "2026-05-29")
    assert lvl.p_up == 0.65
    assert lvl.p_touch == 0.4  # mapped from t_today alias


def test_row_without_geometry_is_dropped():
    assert row_to_internal_level({"symbol": "X.NS", "side": "above"}, "d") is None


def test_read_raw_levels_from_gate_csv(tmp_path):
    df = pd.DataFrame([
        {"symbol": "A.NS", "side": "above", "pool_low": 99, "pool_high": 101,
         "p_touch": 0.7, "q": 0.6, "dir_tag": "DIR_ALIGN", "direction_sign": 1},
        {"symbol": "A.NS", "side": "below", "pool_low": 90, "pool_high": 92,
         "p_touch": 0.5, "q": 0.4, "dir_tag": "DIR_FIGHT", "direction_sign": -1},
    ])
    df.to_csv(tmp_path / "live_gate_decisions.csv", index=False)
    rows = read_raw_levels(tmp_path)
    assert len(rows) == 2
    levels = levels_from_rows(rows, "2026-05-29")
    assert {l.symbol for l in levels} == {"A.NS"}


def test_raw_feed_scorer_end_to_end(tmp_path):
    df = pd.DataFrame([
        {"symbol": "A.NS", "side": "above", "pool_low": 99, "pool_high": 101,
         "p_touch": 0.7, "q": 0.6, "p_up": 0.7},
        {"symbol": "A.NS", "side": "above", "pool_low": 102, "pool_high": 104,
         "p_touch": 0.4, "q": 0.3, "p_up": 0.55},
        {"symbol": "B.NS", "side": "below", "pool_low": 90, "pool_high": 92,
         "p_touch": 0.6, "q": 0.5, "p_up": 0.3},
    ])
    df.to_csv(tmp_path / "live_gate_decisions.csv", index=False)
    scorer = RawFeedScorer(tmp_path)
    out = handle_levels_request(scorer, symbol="A.NS", date="2026-05-29",
                                customer_id="cust-a")
    assert out["observation_count"] == 2
    for rec in out["observations"]:
        assert rec["instrument"] == "A.NS"
        assert set(rec.keys()) == set(PUBLIC_KEYS)


def test_read_raw_levels_from_live_plan_json(tmp_path):
    plan = {
        "tradeable_setups": [
            {"symbol": "A.NS", "side": "below", "pool_low": 90, "pool_high": 92,
             "p_touch": 0.7, "p_respect": 0.6, "direction_sign": 1},
        ],
        "track_a_pretouch": {"setups": [
            {"symbol": "A.NS", "side": "above", "pool_low": 99, "pool_high": 101,
             "pool_mid": 100, "p_touch": 0.8, "p_up": 0.7, "q": 0.6},
        ]},
    }
    (tmp_path / "live_plan.json").write_text(json.dumps(plan))
    rows = read_raw_levels(tmp_path / "live_plan.json")
    assert len(rows) == 2
