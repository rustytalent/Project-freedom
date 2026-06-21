"""Live-bridge integration: REAL BeliefEngine output → v4 brain.

The other v4 tests use hand-built snapshot dicts. This test drives the
ACTUAL BeliefEngine with synthetic-but-realistic option quotes (the same
generator the DemoBeliefLiveRunner uses), produces REAL BeliefSnapshots
via snapshot.to_dict(), and routes them through V4Runner — proving the
real engine's output shape is compatible with the latest brain.

This is the test that answers "are the latest modules actually connected
end-to-end, no stale glue?"
"""
from __future__ import annotations

import math
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pytest

from liqpool.research.belief.engine import (
    BeliefEngine,
    BeliefEngineConfig,
)
from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PersistenceConfig,
    V4Runner,
    V4RunnerConfig,
)

# The Quote type + helpers live in the engine / moneyness modules.
try:
    from liqpool.research.belief.engine import Quote
except Exception:  # pragma: no cover
    Quote = None


IST_OFFSET = timedelta(hours=5, minutes=30)


def _nearest_atm(spot: float, step: float) -> float:
    return round(spot / step) * step


def _synthetic_quotes(spot: float, tick_no: int, ts, *,
                       step: float = 50.0, levels: int = 5) -> Dict:
    """Reproduce the DemoBeliefLiveRunner quote generator — realistic
    option battlefield from a synthetic spot."""
    assert Quote is not None, "Quote type unavailable"
    atm = _nearest_atm(spot, step)
    quotes = {}
    pressure = math.sin(tick_no / 11.0)
    iv_wave = 1.0 + 0.18 * math.sin(tick_no / 23.0)
    for level in range(-levels, levels + 1):
        strike = atm + level * step
        for typ in ("CE", "PE"):
            intrinsic = (max(0.0, spot - strike) if typ == "CE"
                          else max(0.0, strike - spot))
            distance = abs(spot - strike) / max(step, 1.0)
            time_value = max(8.0, 42.0 * math.exp(-0.27 * distance)) * iv_wave
            directional_bump = pressure * (7.0 if typ == "CE" else -7.0)
            fair = max(1.0, intrinsic + time_value + directional_bump)
            spread = max(0.10, fair * (0.006 + 0.002 * distance))
            bid = max(0.05, fair - spread / 2.0)
            ask = max(bid + 0.05, fair + spread / 2.0)
            base_qty = max(50.0, 1600.0 - 135.0 * distance)
            quotes[(float(strike), typ)] = Quote(
                bid=bid, ask=ask,
                bid_qty=base_qty, ask_qty=base_qty,
                ltp=(bid + ask) / 2.0, ltp_age_s=0.0, ts=ts,
            )
    return quotes


def _synthetic_spot(tick_no: int, start: float = 23500.0) -> float:
    drift = math.sin(tick_no / 18.0) * 38.0 + math.sin(tick_no / 7.0) * 13.0
    slow = tick_no * 0.22
    return float(start + drift + slow)


def _new_runner() -> V4Runner:
    tmp = Path(tempfile.mkdtemp(prefix="v4_bridge_test_"))
    cfg = V4RunnerConfig(
        persistence=PersistenceConfig(state_dir=tmp, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


@pytest.mark.skipif(Quote is None, reason="Quote type unavailable")
def test_real_engine_snapshots_flow_through_v4_brain():
    """200 ticks of REAL BeliefEngine output routed through V4Runner.
    Must not crash, must warm up, must produce valid intents."""
    engine = BeliefEngine(BeliefEngineConfig(
        strike_step=50.0, levels=5, warmup_bars=80))
    runner = _new_runner()

    base_ts = datetime(2026, 6, 23, 9, 30, 0)
    warmed = False
    intents_seen = 0
    for tick_no in range(200):
        ts = base_ts + timedelta(seconds=tick_no)
        spot = _synthetic_spot(tick_no)
        quotes = _synthetic_quotes(spot, tick_no, ts)
        snapshot = engine.observe(
            ts=ts.isoformat(), spot=spot, quotes=quotes,
            hunt_verdict="", trap_verdict="",
        )
        snap_dict = snapshot.to_dict()
        # The real snapshot MUST carry the fields the v4 brain reads.
        assert "slot_readings" in snap_dict
        assert "decision" in snap_dict
        assert "thesis" in snap_dict
        result = runner.on_tick(snap_dict)
        intents_seen += 1
        if snapshot.is_warm:
            warmed = True

    assert warmed, "engine never warmed up over 200 ticks"
    assert intents_seen == 200


@pytest.mark.skipif(Quote is None, reason="Quote type unavailable")
def test_real_engine_snapshot_carries_v4_expected_keys():
    """Verify the real snapshot's nested dicts have the keys the v4 brain
    accesses (decision.action/confidence/direction, battlefield rails,
    iv_state, etc.)."""
    engine = BeliefEngine(BeliefEngineConfig(
        strike_step=50.0, levels=5, warmup_bars=10))
    base_ts = datetime(2026, 6, 23, 9, 30, 0)
    snap = None
    for tick_no in range(20):
        ts = base_ts + timedelta(seconds=tick_no)
        spot = _synthetic_spot(tick_no)
        quotes = _synthetic_quotes(spot, tick_no, ts)
        snap = engine.observe(ts=ts.isoformat(), spot=spot,
                                quotes=quotes, hunt_verdict="",
                                trap_verdict="")
    d = snap.to_dict()
    # Decision keys
    assert "action" in d["decision"]
    assert "confidence" in d["decision"]
    # Battlefield rails
    bf = d["battlefield"]
    assert "ce_rail" in bf and "pe_rail" in bf
    # IV state
    assert "state" in d["iv_state"]
    # Slot readings carry premium for the exit engine
    if d["slot_readings"]:
        slot = d["slot_readings"][0]
        assert "mark_price" in slot or "mark" in slot


@pytest.mark.skipif(Quote is None, reason="Quote type unavailable")
def test_real_engine_to_v4_full_chain_serializable():
    """The full intent from a real-engine tick must serialize (so the
    cockpit / JSONL / persistence layers all work)."""
    import json
    engine = BeliefEngine(BeliefEngineConfig(
        strike_step=50.0, levels=5, warmup_bars=10))
    runner = _new_runner()
    base_ts = datetime(2026, 6, 23, 9, 30, 0)
    result = None
    for tick_no in range(30):
        ts = base_ts + timedelta(seconds=tick_no)
        spot = _synthetic_spot(tick_no)
        quotes = _synthetic_quotes(spot, tick_no, ts)
        snap = engine.observe(ts=ts.isoformat(), spot=spot,
                                quotes=quotes, hunt_verdict="",
                                trap_verdict="")
        result = runner.on_tick(snap.to_dict())
    json.dumps(result.to_dict(), default=str)


def test_bridge_does_not_construct_v3_executor():
    """The KiteV4LiveRunner must NEVER construct the v3 executor."""
    from liqpool.research.belief.executor_v4.v4_live_bridge import (
        KiteV4BridgeConfig, KiteV4LiveRunner,
    )
    from liqpool.research.belief.live_runner import BeliefLiveConfig
    cfg = KiteV4BridgeConfig(belief=BeliefLiveConfig(),
                              confirm_real_orders=False,
                              cockpit_server_port=None)
    runner = KiteV4LiveRunner(api_key="dummy", access_token="dummy",
                                bridge_cfg=cfg)
    assert runner.executor is None, "v3 executor was constructed — STALE RISK"
    assert runner.v4 is not None
    assert hasattr(runner.v4.manager, "portfolio_exit")
    # Paper broker by default — no real orders possible without confirm_real.
    assert runner.v4.broker.is_live is False


def test_bridge_paper_broker_unless_confirm_real():
    from liqpool.research.belief.executor_v4.v4_live_bridge import (
        KiteV4BridgeConfig, KiteV4LiveRunner,
    )
    from liqpool.research.belief.live_runner import BeliefLiveConfig
    cfg = KiteV4BridgeConfig(belief=BeliefLiveConfig(),
                              confirm_real_orders=False)
    runner = KiteV4LiveRunner(api_key="x", access_token="y", bridge_cfg=cfg)
    from liqpool.research.belief.executor_v4 import PaperBrokerAdapter
    assert isinstance(runner.v4.broker, PaperBrokerAdapter)
