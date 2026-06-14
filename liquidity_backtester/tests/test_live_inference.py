"""Live inference tests — the missing 'research engine during market
hours' path.

Pins:
  * LiveInferenceBundle.from_report wraps duck-typed models from a
    MultiAssetReport-shaped object
  * Each head's predict callable is tolerant of failing models (returns
    None rather than crashing)
  * LiveInferenceServer maps (head, prob) -> ModelSignal in the
    canonical shared-contracts shape with the right model name,
    confidence floor, and trust_tier
  * JsonlPublisher appends each signal as a JSONL row
  * Round trip works: liqpool publishes -> Sentinel-side LiveSignalsTail
    reads -> Sentinel's bus carries the signal
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from liqpool.contracts.signals import ModelSignal
from liqpool.live_inference import (
    InMemoryPublisher, JsonlPublisher, LiveInferenceBundle,
    LiveInferenceServer,
)


class _DuckDirectionModel:
    """Fake model that returns whatever was passed in via the first feature."""

    def predict_one(self, features):
        return features["p"]


class _BrokenModel:
    def predict_one(self, features):
        raise RuntimeError("network down")


class _FakeReport:
    """Mimic the shape of MultiAssetReport that from_report unpacks."""
    def __init__(self):
        self.unified_direction = _DuckDirectionModel()
        self.unified_ml = _DuckDirectionModel()         # quality
        self.unified_proximity = {78: _DuckDirectionModel(),
                                   156: _DuckDirectionModel()}
        self.model_bundle_version = "test_v1"
        self.feature_version = "fv_42"


def test_bundle_from_report_discovers_three_heads():
    b = LiveInferenceBundle.from_report(_FakeReport())
    names = set(b.heads)
    assert "direction" in names
    assert "quality" in names
    assert "proximity_h78" in names and "proximity_h156" in names
    assert b.bundle_version == "test_v1"
    assert b.feature_version == "fv_42"


def test_bundle_predict_one_uses_head_callable():
    b = LiveInferenceBundle.from_report(_FakeReport())
    assert b.predict_one("direction", {"p": 0.72}) == pytest.approx(0.72)
    assert b.predict_one("proximity_h78", {"p": 0.41}) == pytest.approx(0.41)


def test_bundle_predict_one_returns_none_on_broken_model():
    b = LiveInferenceBundle(heads={
        "direction": LiveInferenceBundle.from_report(_FakeReport()).heads["direction"]
    })
    # rewrite the head to point at a broken model via wrapping
    from liqpool.live_inference import HeadSlot
    def _bad(features):
        raise RuntimeError("nope")
    b.heads["broken"] = HeadSlot(name="broken", predict=_bad, label="broken")
    assert b.predict_one("broken", {}) is None


def test_server_publishes_direction_signal_above_floor():
    b = LiveInferenceBundle.from_report(_FakeReport())
    sink = InMemoryPublisher()
    s = LiveInferenceServer(bundle=b, publisher=sink, publish_floor=0.20,
                              enforce_mis=False)
    fired = s.on_tick({"direction": {"p": 0.80}})
    assert len(fired) == 1
    sig = fired[0]
    assert sig.model == "direction_model"
    assert sig.source == "liqpool"
    assert sig.confidence == pytest.approx(0.6, abs=1e-6)   # |0.8-0.5|*2
    assert "upside" in sig.signal


def test_server_suppresses_low_confidence_direction():
    b = LiveInferenceBundle.from_report(_FakeReport())
    sink = InMemoryPublisher()
    s = LiveInferenceServer(bundle=b, publisher=sink, publish_floor=0.30,
                              enforce_mis=False)
    # p=0.55 -> conf 0.10 -> below floor -> no publish
    fired = s.on_tick({"direction": {"p": 0.55}})
    assert fired == []


def test_server_publishes_quality_signal_with_grade():
    b = LiveInferenceBundle.from_report(_FakeReport())
    sink = InMemoryPublisher()
    s = LiveInferenceServer(bundle=b, publisher=sink, publish_floor=0.30,
                              enforce_mis=False)
    fired = s.on_tick({"quality": {"p": 0.71}})
    assert len(fired) == 1
    assert fired[0].model == "quality_model"
    assert "HIGH" in fired[0].signal


def test_server_publishes_proximity_signal_with_horizon():
    b = LiveInferenceBundle.from_report(_FakeReport())
    sink = InMemoryPublisher()
    s = LiveInferenceServer(bundle=b, publisher=sink, publish_floor=0.20,
                              enforce_mis=False)
    fired = s.on_tick({"proximity_h78": {"p": 0.62}})
    assert len(fired) == 1
    assert fired[0].model == "proximity_model"
    assert "78" in fired[0].signal


def test_server_assigns_trust_tier_per_head():
    b = LiveInferenceBundle.from_report(_FakeReport())
    sink = InMemoryPublisher()
    s = LiveInferenceServer(
        bundle=b, publisher=sink,
        trust_tier_per_head={"direction": "LOGGED"},
        enforce_mis=False)
    fired = s.on_tick({"direction": {"p": 0.85}})
    assert fired[0].trust_tier == "LOGGED"


def test_jsonl_publisher_appends_one_row_per_signal(tmp_path):
    sink = JsonlPublisher(tmp_path / "live.jsonl")
    sig = ModelSignal(ts_ist="t", asset="NIFTY", model="m",
                       signal="s", confidence=0.6, source="liqpool")
    sink.publish(sig)
    sink.publish(sig)
    lines = (tmp_path / "live.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["asset"] == "NIFTY"


def test_end_to_end_liqpool_to_sentinel_via_jsonl(tmp_path):
    """Producer side: liqpool writes JSONL. Consumer side: Sentinel's
    LiveSignalsTail reads it and publishes on the sentinel bus."""
    sink = JsonlPublisher(tmp_path / "live.jsonl")
    b = LiveInferenceBundle.from_report(_FakeReport())
    server = LiveInferenceServer(bundle=b, publisher=sink,
                                   publish_floor=0.20, enforce_mis=False)
    fired = server.on_tick({
        "direction": {"p": 0.80},
        "proximity_h78": {"p": 0.62},
    })
    assert len(fired) == 2

    # consumer side
    sentinel = pytest.importorskip("sentinel.live_publisher")
    from sentinel.liqpool_bridge import LiveSignalsTail
    pub = sentinel.LivePublisher()
    tail = LiveSignalsTail(tmp_path / "live.jsonl", publisher=pub)
    n = tail.poll()
    assert n == 2
    current = pub.current()
    assert "NIFTY|direction_model" in current
    assert "NIFTY|proximity_model" in current
    # second poll picks up zero new rows (offset memory)
    assert tail.poll() == 0


def test_tail_handles_missing_file_gracefully(tmp_path):
    from sentinel.liqpool_bridge import LiveSignalsTail
    sentinel = pytest.importorskip("sentinel.live_publisher")
    pub = sentinel.LivePublisher()
    tail = LiveSignalsTail(tmp_path / "does_not_exist.jsonl", publisher=pub)
    assert tail.poll() == 0


# ─────────────────────────────────────────────────────────────────
# MIS session-window enforcement (Mother's Audit §3.1 fix)
# ─────────────────────────────────────────────────────────────────

def test_mis_session_window_suppresses_outside_session(monkeypatch):
    """At 23:00 IST nothing should publish — the F&O session is closed."""
    from datetime import datetime, timedelta, timezone
    IST = timezone(timedelta(hours=5, minutes=30))

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 6, 14, 23, 0, 0, tzinfo=tz or IST)

    monkeypatch.setattr("liqpool.live_inference.datetime", _FakeDatetime)
    b = LiveInferenceBundle.from_report(_FakeReport())
    sink = InMemoryPublisher()
    server = LiveInferenceServer(bundle=b, publisher=sink,
                                  enforce_mis=True, publish_floor=0.20)
    fired = server.on_tick({"direction": {"p": 0.85}})
    assert fired == []                      # rejected outside session
    assert sink.all() == []


def test_mis_no_new_entry_tag_after_1430(monkeypatch):
    """At 14:35 IST signals still publish but carry
    mis_no_new_entry=True so Crux can refuse new entries."""
    from datetime import datetime, timedelta, timezone
    IST = timezone(timedelta(hours=5, minutes=30))

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 6, 14, 14, 35, 0, tzinfo=tz or IST)

    monkeypatch.setattr("liqpool.live_inference.datetime", _FakeDatetime)
    b = LiveInferenceBundle.from_report(_FakeReport())
    sink = InMemoryPublisher()
    server = LiveInferenceServer(bundle=b, publisher=sink,
                                  enforce_mis=True, publish_floor=0.20)
    fired = server.on_tick({"direction": {"p": 0.85}})
    assert len(fired) == 1
    assert fired[0].extras.get("mis_no_new_entry") is True
    assert fired[0].extras.get("mis_squareoff") is False


def test_mis_squareoff_tag_after_1515(monkeypatch):
    """At 15:20 IST every signal carries mis_squareoff=True so the
    cockpit can show 'MIS exit window' and refuse new entries
    aggressively."""
    from datetime import datetime, timedelta, timezone
    IST = timezone(timedelta(hours=5, minutes=30))

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 6, 14, 15, 20, 0, tzinfo=tz or IST)

    monkeypatch.setattr("liqpool.live_inference.datetime", _FakeDatetime)
    b = LiveInferenceBundle.from_report(_FakeReport())
    sink = InMemoryPublisher()
    server = LiveInferenceServer(bundle=b, publisher=sink,
                                  enforce_mis=True, publish_floor=0.20)
    fired = server.on_tick({"direction": {"p": 0.85}})
    assert len(fired) == 1
    assert fired[0].extras.get("mis_squareoff") is True
    assert fired[0].extras.get("mis_no_new_entry") is True


def test_mis_enforcement_can_be_disabled_for_backtesting(monkeypatch):
    """enforce_mis=False lets replay scripts publish at any wallclock
    time (e.g. midnight backtest)."""
    from datetime import datetime, timedelta, timezone
    IST = timezone(timedelta(hours=5, minutes=30))

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 6, 14, 23, 0, 0, tzinfo=tz or IST)

    monkeypatch.setattr("liqpool.live_inference.datetime", _FakeDatetime)
    b = LiveInferenceBundle.from_report(_FakeReport())
    sink = InMemoryPublisher()
    server = LiveInferenceServer(bundle=b, publisher=sink,
                                  enforce_mis=False, publish_floor=0.20)
    fired = server.on_tick({"direction": {"p": 0.85}})
    assert len(fired) == 1                  # back-test mode = no window check
