"""Tests for the live Kite WebSocket producer.

Closes Codex's #1 launch blocker. Pins:
  * Bar aggregator boundary math + force-close
  * Feature builder is deterministic + handles cold start
  * LiveFeed start/stop lifecycle + heartbeat
  * Reconnect counter + state transitions
  * End-to-end: KiteTicker-style ticks → bars → features → signals
    published onto JSONL → Sentinel's LiveSignalsTail can read them
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from liqpool.contracts.signals import ModelSignal
from liqpool.live_feed import (
    Bar, BarStateFeatureBuilder, FeedHeartbeat, LiveFeed,
    MinuteBarAggregator, _coerce_ts,
)
from liqpool.live_inference import (
    InMemoryPublisher, JsonlPublisher, LiveInferenceBundle,
)

IST = timezone(timedelta(hours=5, minutes=30))


# ─────────────────────────────────────────────────────────────────
# MinuteBarAggregator
# ─────────────────────────────────────────────────────────────────

def test_aggregator_does_not_emit_within_same_bar():
    agg = MinuteBarAggregator(timeframe_minutes=5)
    ts0 = int(time.time() // 300) * 300            # bar boundary
    for i in range(3):
        closed = agg.feed("X", ts0 + i * 10, 100 + i)
        assert closed == []


def test_aggregator_emits_on_boundary():
    agg = MinuteBarAggregator(timeframe_minutes=5)
    ts0 = int(time.time() // 300) * 300
    agg.feed("X", ts0 + 10, 100.0)
    agg.feed("X", ts0 + 60, 102.0)
    agg.feed("X", ts0 + 200, 101.5)
    # next tick crosses the 300s boundary
    closed = agg.feed("X", ts0 + 305, 103.0)
    assert len(closed) == 1
    b = closed[0]
    assert b.open == 100.0
    assert b.high == 102.0
    assert b.low == 100.0
    assert b.close == 101.5            # last tick before boundary
    assert b.n_ticks == 3


def test_aggregator_per_symbol_isolation():
    agg = MinuteBarAggregator(timeframe_minutes=5)
    ts0 = int(time.time() // 300) * 300
    agg.feed("A", ts0 + 10, 100); agg.feed("A", ts0 + 50, 101)
    agg.feed("B", ts0 + 10, 200); agg.feed("B", ts0 + 50, 202)
    closed = agg.feed("A", ts0 + 305, 103)         # only A crosses
    assert len(closed) == 1
    assert closed[0].symbol == "A"


def test_aggregator_force_close_emits_all_active():
    agg = MinuteBarAggregator(timeframe_minutes=5)
    ts0 = int(time.time() // 300) * 300
    agg.feed("A", ts0 + 10, 100)
    agg.feed("B", ts0 + 10, 200)
    closed = agg.force_close_all()
    assert {b.symbol for b in closed} == {"A", "B"}
    # after a force-close, the aggregator's active map is empty
    closed2 = agg.force_close_all()
    assert closed2 == []


def test_aggregator_rejects_bad_inputs():
    agg = MinuteBarAggregator(timeframe_minutes=5)
    assert agg.feed("X", 0, 100) == []
    assert agg.feed("X", time.time(), 0) == []
    assert agg.feed("X", time.time(), -5) == []


def test_aggregator_on_close_callback_fires():
    fired = []
    agg = MinuteBarAggregator(timeframe_minutes=5,
                                on_bar_close=lambda b: fired.append(b))
    ts0 = int(time.time() // 300) * 300
    agg.feed("X", ts0 + 10, 100)
    agg.feed("X", ts0 + 305, 101)
    assert len(fired) == 1


# ─────────────────────────────────────────────────────────────────
# BarStateFeatureBuilder
# ─────────────────────────────────────────────────────────────────

def _make_bar(start: float, end: float, o: float, h: float, lo: float, c: float):
    return Bar(symbol="X", bar_start_ts=start, bar_end_ts=end,
               open=o, high=h, low=lo, close=c, n_ticks=1)


def test_feature_builder_cold_start():
    fb = BarStateFeatureBuilder()
    features = fb.features()
    assert features.get("insufficient_history") == 1.0


def test_feature_builder_emits_returns_and_range():
    fb = BarStateFeatureBuilder()
    ts0 = int(time.time() // 300) * 300
    for i in range(10):
        fb.push(_make_bar(ts0 + i * 300, ts0 + (i + 1) * 300,
                            100 + i, 102 + i, 99 + i, 101 + i))
    f = fb.features()
    assert "ret_1" in f
    assert "ret_6" in f
    assert "ret_24" in f
    assert "ret_78" in f
    assert f["range_atr_proxy"] > 0
    assert "zscore_close_50" in f
    assert "session_minutes_in" in f


def test_feature_builder_deterministic_for_same_series():
    fb1 = BarStateFeatureBuilder()
    fb2 = BarStateFeatureBuilder()
    ts0 = int(time.time() // 300) * 300
    bars = [_make_bar(ts0 + i * 300, ts0 + (i + 1) * 300,
                       100 + i * 0.5, 101 + i * 0.5,
                       99 + i * 0.5, 100.5 + i * 0.5)
            for i in range(20)]
    for b in bars: fb1.push(b); fb2.push(b)
    assert fb1.features() == fb2.features()


def test_feature_builder_streak_signed():
    fb = BarStateFeatureBuilder()
    ts0 = int(time.time() // 300) * 300
    # build a rising 9-bar series
    for i in range(9):
        fb.push(_make_bar(ts0 + i * 300, ts0 + (i + 1) * 300,
                            100 + i, 101 + i, 99 + i, 100.5 + i))
    f = fb.features()
    assert f["streak"] > 0


# ─────────────────────────────────────────────────────────────────
# LiveFeed lifecycle + reconnect counter
# ─────────────────────────────────────────────────────────────────

class _DuckModel:
    """A 'sklearn-like' model that returns whatever's passed in features['p']."""
    def predict_one(self, features):
        # Use the absolute zscore so we get a real number > 0.30 to clear
        # the publish_floor. Cap at 0.92 to look like a probability.
        z = abs(features.get("zscore_close_50", 0.0))
        return min(0.92, 0.5 + z * 0.1)


class _FakeReport:
    """Mimics MultiAssetReport so LiveInferenceBundle.from_report works."""
    def __init__(self):
        self.unified_direction = _DuckModel()
        self.unified_ml = _DuckModel()
        self.unified_proximity = {12: _DuckModel(), 25: _DuckModel(),
                                   78: _DuckModel()}
        self.model_bundle_version = "test_v1"
        self.feature_version = "fv_1"


def test_live_feed_lifecycle(tmp_path):
    bundle = LiveInferenceBundle.from_report(_FakeReport())
    sink = JsonlPublisher(tmp_path / "live.jsonl")
    audit = []
    feed = LiveFeed(
        bundle=bundle, publisher_path=tmp_path / "live.jsonl",
        asset="NIFTY", timeframe_minutes=5, publish_floor=0.30,
        enforce_mis=False,
        audit_sink=lambda evt, payload: audit.append((evt, payload)))
    feed.start()
    assert feed.heartbeat.state == "running"
    assert any(evt == "live_feed_started" for evt, _ in audit)
    feed.stop()
    assert feed.heartbeat.state == "stopped"
    assert any(evt == "live_feed_stopped" for evt, _ in audit)


def test_live_feed_reconnect_counter(tmp_path):
    bundle = LiveInferenceBundle.from_report(_FakeReport())
    feed = LiveFeed(bundle=bundle, publisher_path=tmp_path / "live.jsonl",
                     enforce_mis=False)
    feed.start()
    feed.mark_reconnect("network blip")
    assert feed.heartbeat.n_reconnects == 1
    assert feed.heartbeat.state == "reconnecting"
    assert feed.heartbeat.last_error == "network blip"
    feed.mark_back_running()
    assert feed.heartbeat.state == "running"


# ─────────────────────────────────────────────────────────────────
# End-to-end: ticks → bars → features → published signals → JSONL
# ─────────────────────────────────────────────────────────────────

def test_live_feed_end_to_end_publishes_to_jsonl(tmp_path):
    """The whole pipeline. We feed 25 minutes of synthetic ticks
    (5 bars worth, enough to warm up the feature builder), then
    verify the JSONL has model signals."""
    bundle = LiveInferenceBundle.from_report(_FakeReport())
    feed = LiveFeed(
        bundle=bundle, publisher_path=tmp_path / "live.jsonl",
        asset="NIFTY", timeframe_minutes=5, publish_floor=0.0,
        enforce_mis=False)
    feed.start()
    # 25 minutes of ticks aligned to 5-min boundaries
    base = int(time.time() // 300) * 300
    for bar_idx in range(7):
        for tick_idx in range(3):
            ts = base + bar_idx * 300 + tick_idx * 90
            price = 25000.0 + bar_idx * 10 + tick_idx * 2
            feed.on_ticks([{"tradingsymbol": "NIFTY",
                             "last_price": price,
                             "timestamp": ts}])
    # force-close everything so the last bar lands as a feature input
    feed.stop()

    assert feed.heartbeat.n_ticks >= 21
    assert feed.heartbeat.n_bars >= 5
    # JSONL has at least one signal row
    lines = (tmp_path / "live.jsonl").read_text().splitlines()
    assert len(lines) >= 1
    row = json.loads(lines[0])
    assert row["source"] == "liqpool"
    assert row["asset"] == "NIFTY"


def test_live_feed_to_sentinel_bridge_roundtrip(tmp_path):
    """End-to-end including Sentinel's LiveSignalsTail reading the
    JSONL the producer wrote. Closes the 'two halves talk' loop."""
    sentinel = pytest.importorskip("sentinel.live_publisher")
    from sentinel.liqpool_bridge import LiveSignalsTail

    # Producer side
    bundle = LiveInferenceBundle.from_report(_FakeReport())
    feed = LiveFeed(bundle=bundle, publisher_path=tmp_path / "live.jsonl",
                     asset="NIFTY", publish_floor=0.0,
                     enforce_mis=False)
    feed.start()
    base = int(time.time() // 300) * 300
    for bar_idx in range(6):
        for tick_idx in range(3):
            ts = base + bar_idx * 300 + tick_idx * 90
            price = 25000.0 + bar_idx * 10 + tick_idx
            feed.on_ticks([{"tradingsymbol": "NIFTY",
                             "last_price": price, "timestamp": ts}])
    feed.stop()

    # Consumer side
    pub = sentinel.LivePublisher()
    tail = LiveSignalsTail(tmp_path / "live.jsonl", publisher=pub)
    n = tail.poll()
    assert n >= 1
    # bus carries at least one NIFTY model
    cur = pub.current()
    assert any(k.startswith("NIFTY|") for k in cur.keys())


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def test_coerce_ts_handles_datetime():
    dt = datetime(2026, 6, 15, 9, 16, tzinfo=IST)
    ts = _coerce_ts(dt)
    assert isinstance(ts, float)
    assert ts > 0


def test_coerce_ts_handles_epoch_int():
    assert _coerce_ts(1718353800) == 1718353800.0


def test_coerce_ts_returns_now_on_none():
    out = _coerce_ts(None)
    assert abs(out - time.time()) < 2


def test_heartbeat_round_trips_through_dict():
    hb = FeedHeartbeat(state="running", n_ticks=42, n_bars=8)
    row = hb.to_row()
    assert row["state"] == "running"
    assert row["n_ticks"] == 42
    assert row["n_bars"] == 8
