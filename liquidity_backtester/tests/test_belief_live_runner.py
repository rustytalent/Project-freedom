from __future__ import annotations

from datetime import date, datetime

from liqpool.research.belief.live_runner import (
    BeliefLiveConfig,
    DemoBeliefLiveRunner,
    kite_quote_to_belief_quote,
    make_live_row,
    select_belief_contracts,
)
from liqpool.research.belief.engine import BeliefEngine, BeliefEngineConfig
from liqpool.research.belief.mark_price import Quote


def _instrument(strike: float, typ: str, expiry: date = date(2026, 6, 18)):
    return {
        "tradingsymbol": f"NIFTY{expiry:%y%m%d}{int(strike)}{typ}",
        "name": "NIFTY",
        "strike": strike,
        "instrument_type": typ,
        "expiry": expiry,
        "instrument_token": int(strike) * (1 if typ == "CE" else 2),
        "exchange": "NFO",
    }


def test_select_belief_contracts_picks_nearest_expiry_atm_band():
    instruments = []
    for exp in [date(2026, 6, 18), date(2026, 6, 25)]:
        for strike in [24900.0, 24950.0, 25000.0, 25050.0, 25100.0]:
            instruments.append(_instrument(strike, "CE", exp))
            instruments.append(_instrument(strike, "PE", exp))
    out = select_belief_contracts(
        instruments,
        spot=25012.0,
        underlying="NIFTY",
        strike_step=50.0,
        levels=2,
        now=date(2026, 6, 17),
    )
    assert len(out) == 10
    assert (25000.0, "CE") in out
    assert (25100.0, "PE") in out
    assert {c.expiry for c in out.values()} == {date(2026, 6, 18)}


def test_kite_quote_to_belief_quote_uses_top_of_book_quantities():
    raw = {
        "last_price": 101.25,
        "depth": {
            "buy": [{"price": 101.0, "quantity": 1800}],
            "sell": [{"price": 101.5, "quantity": 900}],
        },
    }
    q = kite_quote_to_belief_quote(raw, ts=datetime(2026, 6, 17, 9, 20))
    assert q.bid == 101.0
    assert q.ask == 101.5
    assert q.bid_qty == 1800
    assert q.ask_qty == 900
    assert q.ltp == 101.25


def test_make_live_row_is_sentinel_model_signal_compatible():
    engine = BeliefEngine(BeliefEngineConfig(warmup_bars=1))
    quotes = {
        (25000.0, "CE"): Quote(bid=100, ask=101, bid_qty=1000, ask_qty=1000, ltp=100.5),
        (25000.0, "PE"): Quote(bid=95, ask=96, bid_qty=1000, ask_qty=1000, ltp=95.5),
    }
    snap = engine.observe(ts="2026-06-17T09:15:00+05:30", spot=25010.0, quotes=quotes)
    contracts = select_belief_contracts(
        [_instrument(25000.0, "CE"), _instrument(25000.0, "PE")],
        spot=25010.0,
        levels=0,
        now=date(2026, 6, 17),
    )
    row = make_live_row(snap, asset="NIFTY", contracts=contracts).to_json_row()
    assert row["model"] == "premium_belief_engine"
    assert row["asset"] == "NIFTY"
    assert row["trust_tier"] == "SHADOW"
    assert row["extras"]["belief_snapshot"]["decision"]["action"]


def test_demo_runner_writes_sentinel_compatible_jsonl(tmp_path):
    out = tmp_path / "liqpool_live_signals.jsonl"
    cfg = BeliefLiveConfig(
        output_jsonl=out,
        max_ticks=3,
        poll_seconds=0.0,
        warmup_bars=1,
        levels=1,
    )
    DemoBeliefLiveRunner(cfg=cfg).run_forever()
    rows = out.read_text().strip().splitlines()
    assert len(rows) == 3
    assert "premium_belief_engine" in rows[-1]
    assert "belief_snapshot" in rows[-1]
