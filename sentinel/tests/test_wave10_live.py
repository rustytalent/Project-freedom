"""Wave 10 tests — Live Model Publisher spine + live cockpit panels.

The founder's diagnosis: until Sentinel shows what the research brain is
thinking live, it's just another panel. The fix is a shared contract:
every research model publishes a structured ModelSignal to one bus; the
dashboard reads from the bus; TRUSTED+ mirrors to the canonical ledger.

These tests pin the contract end-to-end:
  * ModelSignal shape (matches Codex's DecisionEvent fields)
  * LivePublisher dedupe + history ring + ledger mirror
  * Each of the six model stubs returns a sane ModelSignal on the right
    input and stays silent on the wrong one
  * /api/state surfaces live_signals + live_feed + spot_history
  * /api/live/signals exposes the bus + filters by asset
  * The cockpit ships the three new panels (spot chart, research signal
    panel, brief feed)
"""
from __future__ import annotations

import importlib
import json
import time

import pytest
from fastapi.testclient import TestClient

from sentinel.live_models import (
    DEFAULT_MODELS, LiquidityModel, LiveModelPool, ManipulationModel,
    PostReactionModel, ProximityModel, QualityModel, ReactionModel, Tick,
)
from sentinel.live_publisher import LivePublisher, ModelSignal, now_ist_hms
from sentinel.shadow_ledger import ShadowLedger, read_session


# ---------------------------------------------------------------------------
# ModelSignal contract
# ---------------------------------------------------------------------------

def test_model_signal_to_row_round_trips():
    sig = ModelSignal(
        ts_ist="10:23:15", asset="NIFTY", model="reaction_model",
        signal="upside reaction in progress", confidence=0.68,
        trust_tier="LOGGED", zone=(23120.0, 23155.0),
        risk="fails below 23095",
        reason_codes=["liquidity_sweep", "fast_reclaim"])
    row = sig.to_row()
    assert row["asset"] == "NIFTY"
    assert row["confidence"] == 0.68
    assert row["zone"] == [23120.0, 23155.0]      # tuple → list for JSON
    assert "fast_reclaim" in row["reason_codes"]
    assert row["trust_tier"] == "LOGGED"


def test_model_signal_key_collates_per_asset_model():
    a = ModelSignal(ts_ist="t", asset="NIFTY", model="reaction_model",
                    signal="x", confidence=0.5)
    b = ModelSignal(ts_ist="t", asset="HDFCBANK", model="reaction_model",
                    signal="x", confidence=0.5)
    assert a.key != b.key
    assert a.key == "NIFTY|reaction_model"


# ---------------------------------------------------------------------------
# LivePublisher — dedupe + history + canonical mirror
# ---------------------------------------------------------------------------

def _sig(model="reaction_model", signal="x", conf=0.7, asset="NIFTY",
         tier="LOGGED"):
    return ModelSignal(ts_ist=now_ist_hms(), asset=asset, model=model,
                       signal=signal, confidence=conf, trust_tier=tier)


def test_publisher_dedupe_suppresses_identical_signal():
    pub = LivePublisher()
    assert pub.publish(_sig()) is True
    assert pub.publish(_sig()) is False        # same signal+conf -> dedupe
    assert pub.publish(_sig(conf=0.85)) is True  # confidence step crossed
    assert pub.publish(_sig(signal="y")) is True  # signal text changed
    assert len(pub.history()) == 3
    assert "NIFTY|reaction_model" in pub.current()


def test_publisher_history_is_bounded():
    pub = LivePublisher(history=4)
    for i in range(10):
        pub.publish(_sig(signal=f"variant_{i}"))
    assert len(pub.history()) == 4
    # newest-first ordering
    assert pub.history()[0].signal == "variant_9"


def test_publisher_latest_per_asset():
    pub = LivePublisher()
    pub.publish(_sig(asset="NIFTY"))
    pub.publish(_sig(asset="HDFCBANK"))
    pub.publish(_sig(asset="NIFTY", model="proximity_model"))
    nifty = pub.latest_per_asset("NIFTY")
    assert {s.model for s in nifty} == {"reaction_model", "proximity_model"}


def test_publisher_mirrors_trusted_signals_to_shadow_ledger(tmp_path):
    """TRUSTED+ signals are canonicalised as model_suggestion events; SHADOW
    research noise is left off the substrate (one ledger of truths only)."""
    sl = ShadowLedger(tmp_path)
    pub = LivePublisher(shadow_ledger=sl, session="2026-06-13")
    pub.publish(_sig(tier="SHADOW"))         # off-substrate
    pub.publish(_sig(tier="TRUSTED", signal="trusted call"))
    rows = read_session(tmp_path, "2026-06-13")
    assert len(rows) == 1
    assert rows[0]["kind"] == "model_suggestion"
    assert rows[0]["scientist"] == "reaction_model"
    assert rows[0]["trust_tier"] == "TRUSTED"
    assert rows[0]["judgment"]["notes"] == "trusted call"


def test_publisher_does_not_mirror_low_confidence_trusted_signals(tmp_path):
    """Even on TRUSTED, a low-confidence signal stays off the substrate —
    we don't want noise downstream of the calibration loop."""
    sl = ShadowLedger(tmp_path)
    pub = LivePublisher(shadow_ledger=sl, session="2026-06-13")
    pub.publish(_sig(tier="TRUSTED", conf=0.3))   # below 0.5 mirror gate
    assert read_session(tmp_path, "2026-06-13") == []


# ---------------------------------------------------------------------------
# Six model stubs — fire only when they should
# ---------------------------------------------------------------------------

def _ticks(values, base_ts=0.0):
    return [Tick(ts=base_ts + i, spot=v) for i, v in enumerate(values)]


def test_reaction_model_fires_on_sharp_move():
    """Quiet then a sharp up move → upside reaction signal."""
    quiet = [25000 + (i % 3 - 1) for i in range(60)]
    move = [quiet[-1] + 4 * (i + 1) for i in range(5)]
    sig = ReactionModel().tick(_ticks(quiet + move))
    assert sig is not None
    assert "upside" in sig.signal
    assert sig.confidence > 0.4
    assert sig.zone is not None
    assert "z=" in " ".join(sig.reason_codes) or "move=" in " ".join(sig.reason_codes)


def test_reaction_model_silent_in_chop():
    chop = [25000 + (i % 5 - 2) for i in range(80)]
    assert ReactionModel().tick(_ticks(chop)) is None


def test_proximity_model_fires_near_session_high():
    rising = [25000 + i for i in range(120)]
    sig = ProximityModel().tick(_ticks(rising))
    assert sig is not None
    assert "high" in sig.signal


def test_proximity_model_fires_near_session_low():
    falling = [25000 - i for i in range(120)]
    sig = ProximityModel().tick(_ticks(falling))
    assert sig is not None
    assert "low" in sig.signal


def test_quality_model_grades_trend_vs_chop():
    trend = [25000 + i * 0.8 for i in range(40)]
    chop = [25000 + (i % 7 - 3) * 4 for i in range(40)]
    t_sig = QualityModel().tick(_ticks(trend))
    c_sig = QualityModel().tick(_ticks(chop))
    assert "HIGH" in t_sig.signal
    assert "LOW" in c_sig.signal or "MEDIUM" in c_sig.signal


def test_liquidity_model_fires_on_sweep_and_reclaim():
    """Quiet ticks (prior high ~ 25001), spike to 25055, then reclaim
    BACK INSIDE the prior range — classic sweep."""
    quiet = [25000 + (i % 3 - 1) for i in range(40)]
    spike = [25030, 25055, 25060]                      # past prior high
    reclaim = [25025, 25010, 24998]                    # back inside (< 25001)
    sig = LiquidityModel().tick(_ticks(quiet + spike + reclaim))
    assert sig is not None
    assert "sweep" in sig.signal
    assert "fast_reclaim" in sig.reason_codes


def test_post_reaction_model_detects_failed_reaction():
    """A move up that retraces over half → reaction failing."""
    base = [25000] * 20
    up = [25005, 25010, 25015, 25020, 25025, 25030, 25035, 25040, 25045, 25050]
    fail = [25045, 25040, 25030, 25025, 25020]         # gives back > half
    sig = PostReactionModel().tick(_ticks(base + up + fail))
    assert sig is not None
    assert "failing" in sig.signal or "retraced" in sig.signal


def test_manipulation_model_fires_on_fast_spike_and_reclaim():
    quiet = [25000 + (i % 3 - 1) for i in range(25)]
    spike_back = [25000, 25080, 25082, 25005]          # 80-pt spike, reclaim
    sig = ManipulationModel().tick(_ticks(quiet + spike_back))
    assert sig is not None
    assert "manipulation" in sig.signal


def test_pool_runs_all_models_and_publishes_via_bus():
    pool = LiveModelPool()
    pub = LivePublisher()
    # series long enough to satisfy every model's lookback
    series = [25000 + i * 0.5 for i in range(120)]
    sigs = pool.run(_ticks(series))
    assert len(sigs) >= 2                              # several will fire
    pubs = sum(1 for s in sigs if pub.publish(s))
    assert pubs >= 2
    assert all(0.0 <= s.confidence <= 1.0 for s in sigs)


# ---------------------------------------------------------------------------
# Server integration — endpoint shape + cockpit markup
# ---------------------------------------------------------------------------

@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    import sentinel.server as server
    importlib.reload(server)
    return server


def test_state_includes_live_signals_feed_and_spot_history(srv):
    """The /api/state contract gains live_signals, live_feed, spot_history."""
    with TestClient(srv.app) as c:
        srv.CORE._tick_portfolio()
        srv.CORE._tick_models()
        s = c.get("/api/state").json()
        assert "live_signals" in s
        assert "live_feed" in s
        assert "spot_history" in s
        # spot_history rows carry t + spot
        if s["spot_history"]:
            assert "t" in s["spot_history"][0] and "spot" in s["spot_history"][0]


def test_live_signals_endpoint_filters_by_asset(srv):
    """/api/live/signals returns the bus; ?asset=NIFTY filters."""
    srv.CORE.publisher.publish(_sig(asset="NIFTY"))
    srv.CORE.publisher.publish(_sig(asset="HDFCBANK", signal="x"))
    with TestClient(srv.app) as c:
        all_ = c.get("/api/live/signals").json()
        assert "NIFTY|reaction_model" in all_["current"]
        only = c.get("/api/live/signals?asset=NIFTY").json()
        assert all(k.startswith("NIFTY") for k in only["current"])


def test_premium_belief_endpoint_tails_jsonl_on_request(srv):
    """The Premium Belief cockpit must not wait for the background loop.

    The live runner writes JSONL faster than the generic Sentinel loop. The
    endpoint force-polls the tail on each request so the trader sees the newest
    row on the next browser refresh.
    """
    path = srv.CORE.cfg.journal_dir / "liqpool_live_signals.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ts_ist": now_ist_hms(),
        "asset": "NIFTY_DEMO",
        "model": "premium_belief_engine",
        "signal": "WAIT",
        "confidence": 0.42,
        "trust_tier": "SHADOW",
        "reason_codes": ["test_row"],
        "extras": {
            "spot": 25000.0,
            "belief_snapshot": {
                "spot": 25000.0,
                "bars_seen": 101,
                "is_warm": True,
                "decision": {
                    "action": "WAIT",
                    "confidence": 0.42,
                    "direction": 0,
                    "trade_allowed": False,
                },
                "slot_readings": [],
            },
        },
    }) + "\n")

    with TestClient(srv.app) as c:
        payload = c.get("/api/premium_belief?limit=1").json()

    assert payload["status"] == "ok"
    assert payload["tail_rows_polled"] == 1
    assert payload["latest"]["asset"] == "NIFTY_DEMO"
    assert payload["latest"]["extras"]["spot"] == 25000.0
    assert payload["latest_age_seconds"] is not None


def test_cockpit_renders_live_panels(srv):
    """The cockpit ships the three new panels the founder asked for."""
    with TestClient(srv.app) as c:
        html = c.get("/").text
    for marker in (
        "NIFTY · Live spot", "spot_chart", "Live research signals",
        "live_signals", "Live brief feed", "live_feed",
        "drawSpotChart(", "renderSignal(", "renderFeedRow(",
    ):
        assert marker in html


def test_model_tick_pump_publishes_after_history_warms_up(srv):
    """The Sentinel loop's _tick_models pumps the publisher once spot
    history is long enough."""
    core = srv.CORE
    # seed a rising spot series so reaction + proximity + quality fire
    for i in range(120):
        core._tick_hist.append(Tick(ts=time.monotonic() + i, spot=25000 + i))
    # one more tick triggers the model pool
    core._tick_models()
    assert core.publisher.current()              # at least one model fired
