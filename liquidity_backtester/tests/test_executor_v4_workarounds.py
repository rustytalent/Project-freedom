"""Tests for the workarounds + latency budget + limit-only order policy."""
from __future__ import annotations

import socket
import tempfile
import time
import urllib.request
import json
from pathlib import Path

import pandas as pd
import pytest

from liqpool.research.belief.executor_v4 import (
    KiteBrokerAdapter,
    KiteBrokerConfig,
    PaperBrokerAdapter,
    PersistenceConfig,
    V4Runner,
    V4RunnerConfig,
)
from liqpool.research.belief.executor_v4.broker import BrokerOrder
from liqpool.research.belief.executor_v4.broker.base import (
    STATE_REJECTED, STATE_FILLED,
)
from liqpool.research.belief.executor_v4.economics import estimate_slippage


def _isolated_paper_runner() -> V4Runner:
    """Build a paper runner with an isolated tmp state dir so previous
    test runs can't leak into this one."""
    tmp = Path(tempfile.mkdtemp(prefix="v4_wa_test_"))
    cfg = V4RunnerConfig(
        persistence=PersistenceConfig(state_dir=tmp, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


# ── Limit-only policy ────────────────────────────────────────────


def _mk_order(otype: str, side: str = "BUY",
                price: float = 100.0) -> BrokerOrder:
    return BrokerOrder(
        client_order_id="t", tradingsymbol="X", exchange="NFO",
        side=side, quantity=65, order_type=otype, product="MIS",
        limit_price=price, position_id="p1",
    )


class _MockKite:
    """Mock account that only exposes place_market_exit (the legacy
    helper) — used to prove the policy guard refuses to silently fall
    through to MARKET for a LIMIT request."""
    def __init__(self):
        self.placed = []
    def place_market_exit(self, *, tradingsymbol, side, quantity, exchange):
        self.placed.append({"tradingsymbol": tradingsymbol, "side": side,
                              "quantity": quantity, "exchange": exchange,
                              "order_id": "K-1", "status": "submitted"})
        return self.placed[-1]
    def positions(self):
        return {"net": []}


class _MockKiteWithLimit(_MockKite):
    """Mock account that DOES expose place_limit_order — proves the
    adapter prefers the limit path when available."""
    def place_limit_order(self, *, tradingsymbol, side, quantity,
                            limit_price, exchange, product):
        self.placed.append({"tradingsymbol": tradingsymbol, "side": side,
                              "quantity": quantity, "exchange": exchange,
                              "limit_price": limit_price,
                              "order_id": "L-1", "status": "submitted"})
        return self.placed[-1]


def test_market_orders_refused_by_default():
    """Without explicit allow_market_orders, MARKET requests are refused."""
    mk = _MockKiteWithLimit()
    a = KiteBrokerAdapter(kite_account=mk,
                            cfg=KiteBrokerConfig(confirm_real=True))
    a.begin_tick()
    r = a.place_order(_mk_order("MARKET"))
    assert r.state == STATE_REJECTED
    assert "MARKET orders disallowed" in (r.rejection_reason or "")


def test_kite_limit_path_uses_place_limit_order():
    mk = _MockKiteWithLimit()
    a = KiteBrokerAdapter(kite_account=mk,
                            cfg=KiteBrokerConfig(confirm_real=True))
    a.begin_tick()
    r = a.place_order(_mk_order("LIMIT", price=98.0))
    assert r.state == STATE_FILLED or r.state == "PENDING"
    # Confirm the limit-order helper was actually called.
    assert mk.placed
    assert mk.placed[0].get("limit_price") == 98.0


def test_kite_refuses_silent_market_fallthrough_for_limit_request():
    """When the account does NOT expose place_limit_order, the adapter
    must REFUSE to fall through to place_market_exit. Otherwise a LIMIT
    request would silently become MARKET in production."""
    mk = _MockKite()   # no place_limit_order
    a = KiteBrokerAdapter(kite_account=mk,
                            cfg=KiteBrokerConfig(confirm_real=True))
    a.begin_tick()
    r = a.place_order(_mk_order("LIMIT"))
    assert r.state == STATE_REJECTED
    assert "place_limit_order" in (r.rejection_reason or "")
    # And place_market_exit must NOT have been called.
    assert mk.placed == []


def test_v4_runner_only_sends_limit_orders():
    """End-to-end: regardless of strategy, every order the runner places
    is a LIMIT. This catches accidental MARKET introductions in the
    routing layer."""
    runner = _isolated_paper_runner()
    seen_types = []

    original_place = runner.broker.place_order

    def _spy(order):
        seen_types.append(order.order_type)
        return original_place(order)
    runner.broker.place_order = _spy

    # Force an entry through.
    snap = _hold_snap(bars=200, action="ENTER_LONG")
    for i in range(95):
        runner.on_tick(_hold_snap(bars=100 + i))
    runner.on_tick(snap)
    if seen_types:
        assert set(seen_types) == {"LIMIT"}, (
            f"non-LIMIT order types observed: {set(seen_types)}")


# ── Slippage feedback into EV gate ──────────────────────────────


def test_estimate_slippage_blends_realized_with_more_samples():
    # No samples → pure theoretical.
    theo = estimate_slippage(lots=2, friendliness=0.9, spread_state="clean")
    # Many samples of high realized bps → blended estimate higher than theo.
    blended = estimate_slippage(
        lots=2, friendliness=0.9, spread_state="clean",
        realized_mean_bps=80.0, realized_n_samples=40,
        reference_premium=100.0,
    )
    assert blended > theo


def test_estimate_slippage_falls_back_when_no_reference_premium():
    # If reference_premium <= 0, the override is ignored.
    theo = estimate_slippage(lots=2, friendliness=0.9)
    blend = estimate_slippage(
        lots=2, friendliness=0.9,
        realized_mean_bps=50.0, realized_n_samples=10,
        reference_premium=0.0,
    )
    assert theo == blend


# ── Per-strategy attribution ────────────────────────────────────


def _hold_snap(bars: int, action: str = "HOLD") -> dict:
    ts = pd.Timestamp("2026-06-23 10:00") + pd.Timedelta(seconds=bars)
    slot_readings = []
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level, "option_type": "CE",
            "level": level, "label": f"CE_{level}",
            "acceptance": "normal", "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "is_abnormal": False, "dod_z": 1.5, "mark_price": 100.0,
        })
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level, "option_type": "PE",
            "level": level, "label": f"PE_{level}",
            "acceptance": "normal", "friendliness": 0.92,
            "spread_state": "clean", "mark_source": "microprice",
            "is_abnormal": False, "dod_z": -1.0, "mark_price": 100.0,
        })
    return {
        "ts": ts, "spot": 23000.0, "bars_seen": bars, "is_warm": True,
        "thesis": {"composite_state": "HOLD_BULL",
                   "bull_thesis_score": 70.0, "bear_thesis_score": 10.0,
                   "no_trade_score": 5.0},
        "iv_state": {"state": "directional_bull", "direction": 1,
                     "confidence": 0.85,
                     "clean_mark_fraction": 0.98, "net_intent_z": 2.5},
        "battlefield": {
            "verdict": "bullish_agreement", "direction": 1, "confidence": 0.8,
            "ce_rail": {"weighted_mean_signed_z": 1.5,
                        "dispersion_score": 0.10,
                        "epicenter_label": "CE_ATM", "epicenter_level": 0},
            "pe_rail": {"weighted_mean_signed_z": -1.0,
                        "dispersion_score": 0.10,
                        "epicenter_label": "PE_ATM", "epicenter_level": 0},
        },
        "decision": {"action": action, "direction": 1, "confidence": 0.82,
                     "trade_allowed": True,
                     "strike": {"side": "CE", "level": 0, "label": "CE_ATM"}},
        "winding": {"zone": "NO_WINDING"},
        "bull_state": {"state_index": 2},
        "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


def test_strategy_attribution_records_close():
    runner = _isolated_paper_runner()
    for i in range(90):
        runner.on_tick(_hold_snap(100 + i))
    runner.on_tick(_hold_snap(200, action="ENTER_LONG"))
    runner.on_tick(_hold_snap(201, action="EXIT"))
    result = runner.on_tick(_hold_snap(202))
    attribution = result.cockpit.attribution_panel
    assert attribution["n_strategies_seen"] >= 1
    assert attribution["rows"], "expected at least one strategy row"
    row = attribution["rows"][0]
    assert "win_rate" in row
    assert "avg_realized_r" in row
    assert "size_multiplier_now" in row


def test_confidence_gate_throttles_losing_strategy():
    """Force the same strategy to close 5 losing trades; the
    size_multiplier_now should drop to 0.40 (throttle)."""
    runner = _isolated_paper_runner()
    mgr = runner.manager
    # Inject a synthetic losing track record directly into stats.
    mgr._strategy_stats["ENTER_LONG"]["n_trades"] = 6
    mgr._strategy_stats["ENTER_LONG"]["wins"] = 1
    mgr._strategy_stats["ENTER_LONG"]["losses"] = 5
    mgr._strategy_stats["ENTER_LONG"]["avg_realized_r"] = -0.8
    mult = mgr._strategy_confidence_multiplier("ENTER_LONG")
    assert mult == 0.40


def test_confidence_gate_boosts_winning_strategy():
    runner = _isolated_paper_runner()
    mgr = runner.manager
    mgr._strategy_stats["ENTER_LONG"]["n_trades"] = 6
    mgr._strategy_stats["ENTER_LONG"]["wins"] = 4
    mgr._strategy_stats["ENTER_LONG"]["losses"] = 2
    mgr._strategy_stats["ENTER_LONG"]["avg_realized_r"] = 0.35
    mult = mgr._strategy_confidence_multiplier("ENTER_LONG")
    assert mult > 1.0
    assert mult <= 1.25


def test_confidence_gate_neutral_with_few_trades():
    runner = _isolated_paper_runner()
    mgr = runner.manager
    # < min_trades → 1.0 (no haircut, no bonus).
    mgr._strategy_stats["NEW_STRAT"]["n_trades"] = 2
    mult = mgr._strategy_confidence_multiplier("NEW_STRAT")
    assert mult == 1.0


# ── Latency budget ──────────────────────────────────────────────


def test_latency_panel_populated_after_ticks():
    runner = _isolated_paper_runner()
    for i in range(10):
        runner.on_tick(_hold_snap(100 + i))
    result = runner.on_tick(_hold_snap(110))
    lat = result.cockpit.latency_panel
    assert lat["n_samples"] >= 1
    assert lat["total_ms_p50"] is not None


def test_per_tick_total_within_one_second_budget():
    """The brain MUST run end-to-end in well under 1 second so the
    1-Hz Kite poll cycle is keepable. This test gives a generous 750ms
    p95 budget on synthetic data; it's a regression fence, not a
    benchmark."""
    runner = _isolated_paper_runner()
    for i in range(50):
        runner.on_tick(_hold_snap(100 + i))
    summary = runner.latency_summary()
    total = summary.get("total_ms") or {}
    p95 = total.get("p95_ms") or 0.0
    assert p95 < 750.0, (
        f"per-tick p95 latency {p95}ms exceeds 750ms budget — risk of "
        "lagging the 1-Hz Kite poll cycle")


# ── HTTP control: limit-only state surfaced ─────────────────────


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]; s.close()
    return p


def test_viewer_html_carries_new_panels():
    """The embedded HTML viewer must include the new panel divs so a
    browser can render them."""
    from liqpool.research.belief.executor_v4.cockpit_server import (
        _DEFAULT_VIEWER_HTML,
    )
    assert "LATENCY (rolling 60 ticks)" in _DEFAULT_VIEWER_HTML
    assert "STRATEGY ATTRIBUTION" in _DEFAULT_VIEWER_HTML
    assert "SLIPPAGE TRACKER" in _DEFAULT_VIEWER_HTML
