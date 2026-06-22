"""Tests for executor_v4.broker — paper + Kite adapters."""
from __future__ import annotations

import pytest

from liqpool.research.belief.executor_v4.broker import (
    BrokerOrder,
    KiteBrokerAdapter,
    KiteBrokerConfig,
    PaperBrokerAdapter,
)
from liqpool.research.belief.executor_v4.broker.base import (
    STATE_FILLED,
    STATE_REJECTED,
)


def _mk_order(*, side: str = "BUY", qty: int = 65,
                price: float = 100.0,
                tradingsymbol: str = "NIFTY24JUN23000CE",
                otype: str = "LIMIT") -> BrokerOrder:
    return BrokerOrder(
        client_order_id=f"test-{side}-{qty}",
        tradingsymbol=tradingsymbol,
        exchange="NFO", side=side, quantity=qty,
        order_type=otype, product="MIS",
        limit_price=price, position_id="p1",
    )


# ── Paper adapter ──────────────────────────────────────────────────


def test_paper_fills_limit_order_at_price():
    p = PaperBrokerAdapter()
    o = _mk_order(side="BUY", qty=65, price=100.0)
    r = p.place_order(o)
    assert r.state == STATE_FILLED
    assert r.filled_quantity == 65
    assert r.avg_fill_price == 100.0


def test_paper_tracks_position_after_fill():
    p = PaperBrokerAdapter()
    p.place_order(_mk_order(side="BUY", qty=65, price=100.0))
    positions = p.get_positions()
    assert len(positions) == 1
    assert positions[0].quantity == 65
    assert positions[0].avg_price == 100.0


def test_paper_position_closes_on_offsetting_sell():
    p = PaperBrokerAdapter()
    p.place_order(_mk_order(side="BUY", qty=65, price=100.0))
    p.place_order(_mk_order(side="SELL", qty=65, price=110.0))
    assert p.get_positions() == []


def test_paper_avg_price_averages_on_pyramiding():
    p = PaperBrokerAdapter()
    p.place_order(_mk_order(side="BUY", qty=65, price=100.0))
    p.place_order(_mk_order(side="BUY", qty=65, price=120.0))
    positions = p.get_positions()
    assert positions[0].quantity == 130
    # Average of 100 and 120 (equal qty) = 110.
    assert abs(positions[0].avg_price - 110.0) < 0.01


def test_paper_market_order_requires_mark_lookup():
    p = PaperBrokerAdapter()
    o = _mk_order(otype="MARKET", price=0.0)
    r = p.place_order(o)
    assert r.state == STATE_REJECTED
    assert "no mark" in (r.rejection_reason or "")


def test_paper_market_order_uses_mark_lookup_with_slippage():
    p = PaperBrokerAdapter(mark_lookup=lambda s: 100.0,
                            default_slippage_bps=10.0)
    o = _mk_order(otype="MARKET", side="BUY")
    r = p.place_order(o)
    assert r.state == STATE_FILLED
    # 10bps slippage on buy → +0.1%
    assert abs(r.avg_fill_price - 100.10) < 0.01


def test_paper_kill_switch_rejects_orders():
    p = PaperBrokerAdapter()
    p.kill_switch("test")
    r = p.place_order(_mk_order())
    assert r.state == STATE_REJECTED
    assert "kill" in (r.rejection_reason or "")


def test_paper_order_result_serializable():
    p = PaperBrokerAdapter()
    r = p.place_order(_mk_order())
    import json
    json.dumps(r.to_dict())


# ── Kite adapter ───────────────────────────────────────────────────


class _MockKite:
    def __init__(self):
        self.placed = []
        self._positions = []

    def place_market_exit(self, *, tradingsymbol, side, quantity, exchange):
        order_id = f"KITE-{len(self.placed)}"
        self.placed.append({
            "tradingsymbol": tradingsymbol, "side": side,
            "quantity": quantity, "order_id": order_id,
            "status": "submitted",
        })
        return self.placed[-1]

    def place_limit_order(self, *, tradingsymbol, side, quantity,
                            limit_price, exchange, product):
        # Founder policy (2026-06-22): the executor uses LIMIT orders;
        # any account adapter must expose this. Test mock now satisfies
        # the contract so the live-policy guard doesn't reject it.
        order_id = f"KITE-LMT-{len(self.placed)}"
        self.placed.append({
            "tradingsymbol": tradingsymbol, "side": side,
            "quantity": quantity, "limit_price": limit_price,
            "order_id": order_id, "status": "submitted",
        })
        return self.placed[-1]

    def positions(self):
        return {"net": self._positions}


def test_kite_dry_run_default_does_not_call_broker():
    mk = _MockKite()
    a = KiteBrokerAdapter(kite_account=mk,
                            cfg=KiteBrokerConfig(confirm_real=False))
    r = a.place_order(_mk_order())
    assert r.state == STATE_FILLED
    assert r.raw.get("mode") == "dry_run"
    assert len(mk.placed) == 0


def test_kite_live_mode_calls_broker():
    mk = _MockKite()
    a = KiteBrokerAdapter(kite_account=mk,
                            cfg=KiteBrokerConfig(confirm_real=True))
    a.begin_tick()
    r = a.place_order(_mk_order())
    assert r.state == STATE_FILLED
    assert len(mk.placed) == 1
    assert mk.placed[0]["tradingsymbol"] == "NIFTY24JUN23000CE"


def test_kite_respects_per_tick_cap():
    mk = _MockKite()
    a = KiteBrokerAdapter(kite_account=mk,
                            cfg=KiteBrokerConfig(confirm_real=False,
                                                    max_orders_per_tick=2))
    a.begin_tick()
    a.place_order(_mk_order())
    a.place_order(_mk_order())
    r = a.place_order(_mk_order())
    assert r.state == STATE_REJECTED
    assert "per-tick" in r.rejection_reason


def test_kite_kill_switch_blocks():
    mk = _MockKite()
    a = KiteBrokerAdapter(kite_account=mk,
                            cfg=KiteBrokerConfig(confirm_real=False))
    a.kill_switch("emergency")
    r = a.place_order(_mk_order())
    assert r.state == STATE_REJECTED


def test_kite_refuses_opposing_position():
    mk = _MockKite()
    mk._positions = [{
        "tradingsymbol": "NIFTY24JUN23000CE",
        "quantity": 65, "average_price": 100.0, "last_price": 110.0,
        "exchange": "NFO",
    }]
    a = KiteBrokerAdapter(kite_account=mk,
                            cfg=KiteBrokerConfig(confirm_real=True))
    a.begin_tick()
    # Try to SELL twice the existing long — would over-position to net short.
    sell = _mk_order(side="SELL", qty=130)
    r = a.place_order(sell)
    assert r.state == STATE_REJECTED
    assert "opposing" in r.rejection_reason


def test_kite_healthcheck_serializable():
    mk = _MockKite()
    a = KiteBrokerAdapter(kite_account=mk,
                            cfg=KiteBrokerConfig(confirm_real=False))
    import json
    json.dumps(a.healthcheck())
