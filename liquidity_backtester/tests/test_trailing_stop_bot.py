"""Tests for the trailing-stop bot.

The bot's job is small and well-defined; the tests pin every state
transition exhaustively. All run against a fake broker + scripted
quote function so there are zero real network calls.
"""
from __future__ import annotations

import json
import time
from datetime import time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from liqpool.broker_zerodha import OrderResult, ZerodhaBroker, ZerodhaConfig
from liqpool.trading.trailing_stop_bot import (
    MIN_CUSHION_RUPEES,
    TrailingStopBot,
    TrailJournal,
    TrailState,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeBroker:
    """Records every order placement; never connects to anything."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def place_market_order(self, *, tradingsymbol: str, side: str,
                            quantity: int, exchange: str = "NFO",
                            confirm: bool = False) -> OrderResult:
        call = {
            "tradingsymbol": tradingsymbol, "side": side,
            "quantity": quantity, "exchange": exchange,
            "confirm": confirm,
        }
        self.calls.append(call)
        return OrderResult(
            order_id=f"FAKE-{len(self.calls):04d}",
            status="simulated",
            dry_run=True,
            raw_response=call,
        )


class FailingBroker(FakeBroker):
    def place_market_order(self, **kw) -> OrderResult:
        self.calls.append(kw)
        return OrderResult(order_id="", status="rejected", dry_run=False,
                            error="market closed simulation")


def _bot(broker, journal_path: Path, quotes: List[Optional[float]],
         # Default to 23:59 IST so unit tests aren't poisoned by the
         # real wall-clock time (running this suite at 16:00 IST would
         # otherwise force-flatten every trail on the first tick).
         # The explicit auto-flatten test overrides this.
         auto_flatten_at_ist: dtime = dtime(23, 59)) -> TrailingStopBot:
    """Build a bot whose quote function returns ``quotes`` in order,
    repeating the last value once the script is exhausted. Lets the
    test drive premium-path scenarios deterministically."""
    iterator = {"i": 0}

    def _q(_exch, _sym):
        i = iterator["i"]
        iterator["i"] += 1
        if i < len(quotes):
            return quotes[i]
        return quotes[-1] if quotes else None

    bot = TrailingStopBot(
        broker=broker, journal_path=journal_path,
        quote_fn=_q,
        poll_seconds=0.001,
        confirm_real_orders=False,
        auto_flatten_at_ist=auto_flatten_at_ist,
    )
    return bot


# ---------------------------------------------------------------------------
# arm() validation
# ---------------------------------------------------------------------------

def test_arm_rejects_invalid_side(tmp_path):
    bot = _bot(FakeBroker(), tmp_path / "j.jsonl", quotes=[100.0])
    with pytest.raises(ValueError, match="side"):
        bot.arm("X", "weird", 10, 30.0, 100.0)


def test_arm_rejects_non_positive_quantity(tmp_path):
    bot = _bot(FakeBroker(), tmp_path / "j.jsonl", quotes=[100.0])
    with pytest.raises(ValueError, match="quantity"):
        bot.arm("X", "long", 0, 30.0, 100.0)


def test_arm_rejects_too_small_cushion(tmp_path):
    bot = _bot(FakeBroker(), tmp_path / "j.jsonl", quotes=[100.0])
    with pytest.raises(ValueError, match="cushion"):
        bot.arm("X", "long", 10, MIN_CUSHION_RUPEES / 2, 100.0)


def test_arm_rejects_non_positive_premium(tmp_path):
    bot = _bot(FakeBroker(), tmp_path / "j.jsonl", quotes=[100.0])
    with pytest.raises(ValueError, match="premium"):
        bot.arm("X", "long", 10, 30.0, 0.0)


def test_arm_writes_to_journal(tmp_path):
    journal = tmp_path / "j.jsonl"
    bot = _bot(FakeBroker(), journal, quotes=[100.0])
    state = bot.arm("NIFTY26JUN24500CE", "long", 75, 30.0, 200.0,
                    note="post-Fed gap fade")
    assert state.state == "ARMED"
    assert state.peak_premium == 200.0
    assert state.cushion_rupees == 30.0
    # Journal has one line with our trail.
    lines = journal.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["trail_id"] == state.trail_id
    assert row["note"] == "post-Fed gap fade"


# ---------------------------------------------------------------------------
# Long position — peak tracking + trigger
# ---------------------------------------------------------------------------

def test_long_position_tracks_peak_up_and_triggers_on_giveback(tmp_path):
    broker = FakeBroker()
    # Premium path: arm at 200; runs up to 250; pulls back to 219 (gave back 31).
    quotes = [220.0, 230.0, 250.0, 240.0, 219.0]
    bot = _bot(broker, tmp_path / "j.jsonl", quotes=quotes)
    bot.arm("OPTX", "long", 75, cushion_rupees=30.0, current_premium=200.0)
    # Run enough ticks to exhaust the script.
    for _ in range(len(quotes)):
        bot._tick()
    # Exactly one exit fired, on the SELL side (exit a long = sell).
    assert len(broker.calls) == 1
    assert broker.calls[0]["side"] == "SELL"
    assert broker.calls[0]["quantity"] == 75
    # And peak should have been the highest premium seen.
    journal_rows = [json.loads(l) for l in
                    (tmp_path / "j.jsonl").read_text().strip().splitlines()]
    peaks = [r["peak_premium"] for r in journal_rows]
    assert max(peaks) == 250.0


def test_long_position_does_not_fire_inside_cushion(tmp_path):
    broker = FakeBroker()
    # Peak hits 230; pull back to 205 = give back 25 < 30 cushion.
    quotes = [210.0, 230.0, 220.0, 205.0]
    bot = _bot(broker, tmp_path / "j.jsonl", quotes=quotes)
    bot.arm("OPTX", "long", 50, cushion_rupees=30.0, current_premium=200.0)
    for _ in range(len(quotes)):
        bot._tick()
    assert broker.calls == []


# ---------------------------------------------------------------------------
# Short position — trough tracking + trigger
# ---------------------------------------------------------------------------

def test_short_position_tracks_trough_down_and_triggers_on_rise(tmp_path):
    """For a short (option writer), profit grows as premium falls.
    Cushion fires when premium RISES from the trough by cushion."""
    broker = FakeBroker()
    # Arm at 100; trough goes to 60; bounces to 91 (rose 31 from trough).
    quotes = [90.0, 70.0, 60.0, 80.0, 91.0]
    bot = _bot(broker, tmp_path / "j.jsonl", quotes=quotes)
    bot.arm("OPTX", "short", 75, cushion_rupees=30.0, current_premium=100.0)
    for _ in range(len(quotes)):
        bot._tick()
    assert len(broker.calls) == 1
    # Exit a short = BUY.
    assert broker.calls[0]["side"] == "BUY"
    journal_rows = [json.loads(l) for l in
                    (tmp_path / "j.jsonl").read_text().strip().splitlines()]
    # "peak_premium" stores the TROUGH for shorts.
    troughs = [r["peak_premium"] for r in journal_rows]
    assert min(troughs) == 60.0


# ---------------------------------------------------------------------------
# Bad quotes / robustness
# ---------------------------------------------------------------------------

def test_bad_quotes_do_not_fire_or_corrupt_peak(tmp_path):
    """If the broker quote returns None / 0, the tick is a no-op —
    must NOT fire an exit, must NOT update peak."""
    broker = FakeBroker()
    # Run good price up to 250 then mix in two bad ticks.
    quotes = [250.0, None, 0.0, 245.0]
    bot = _bot(broker, tmp_path / "j.jsonl", quotes=quotes)
    bot.arm("X", "long", 50, cushion_rupees=30.0, current_premium=200.0)
    for _ in range(len(quotes)):
        bot._tick()
    # 245 from peak 250 = give back 5 < 30 cushion, so no fire.
    assert broker.calls == []
    # The recorded peak is the legitimate 250, not None or 0.
    rows = [json.loads(l) for l in
            (tmp_path / "j.jsonl").read_text().strip().splitlines()]
    assert max(r["peak_premium"] for r in rows) == 250.0


# ---------------------------------------------------------------------------
# Cancel / update
# ---------------------------------------------------------------------------

def test_cancel_stops_a_trail_before_fire(tmp_path):
    broker = FakeBroker()
    quotes = [210.0, 230.0, 150.0]   # would fire on the 150
    bot = _bot(broker, tmp_path / "j.jsonl", quotes=quotes)
    s = bot.arm("X", "long", 10, cushion_rupees=30.0, current_premium=200.0)
    bot._tick()
    assert bot.cancel(s.trail_id) is True
    bot._tick()      # would have fired, but trail is gone
    bot._tick()
    assert broker.calls == []


def test_cancel_unknown_trail_returns_false(tmp_path):
    bot = _bot(FakeBroker(), tmp_path / "j.jsonl", quotes=[100.0])
    assert bot.cancel("not-a-real-id") is False


def test_update_cushion_changes_trigger_distance(tmp_path):
    broker = FakeBroker()
    # Arm with cushion 50; peak hits 240; pull back to 195 = give back 45
    # — would NOT fire at cushion 50. Tighten to 30 mid-flight → fires.
    bot = _bot(broker, tmp_path / "j.jsonl",
               quotes=[210.0, 240.0, 195.0, 195.0])
    s = bot.arm("X", "long", 10, cushion_rupees=50.0, current_premium=200.0)
    bot._tick()
    bot._tick()      # peak now 240
    assert bot.update_cushion(s.trail_id, 30.0) is True
    bot._tick()      # 195 from peak 240 = 45 give-back >= 30 NEW cushion
    assert len(broker.calls) == 1


def test_update_cushion_rejects_too_small(tmp_path):
    bot = _bot(FakeBroker(), tmp_path / "j.jsonl", quotes=[100.0])
    s = bot.arm("X", "long", 10, 30.0, 200.0)
    with pytest.raises(ValueError):
        bot.update_cushion(s.trail_id, 0.0)


# ---------------------------------------------------------------------------
# Auto-flatten before MIS square-off
# ---------------------------------------------------------------------------

def test_auto_flatten_fires_after_squareoff_time(tmp_path):
    """If the clock has already passed 15:14 IST, every tick force-
    flattens any ARMED trail at market."""
    broker = FakeBroker()
    # Use a flatten time of 00:00 IST so 'now' is always past it.
    bot = _bot(broker, tmp_path / "j.jsonl",
               quotes=[195.0],
               auto_flatten_at_ist=dtime(0, 0))
    bot.arm("X", "long", 10, cushion_rupees=999.0, current_premium=200.0)
    bot._tick()
    # Even though cushion 999 wouldn't fire on price 195, square-off does.
    assert len(broker.calls) == 1
    rows = [json.loads(l) for l in
            (tmp_path / "j.jsonl").read_text().strip().splitlines()]
    # State machine: final state is EXITED (the broker accepted),
    # but the REASON is SQUAREOFF (not a peak-pullback TRIGGER).
    final = rows[-1]
    assert final["state"] == "EXITED"
    assert final["exit_reason"] == "SQUAREOFF"


# ---------------------------------------------------------------------------
# Crash recovery via journal replay
# ---------------------------------------------------------------------------

def test_active_trails_recovered_from_journal_on_restart(tmp_path):
    """Arm a trail, "crash" (drop the bot), build a new one — the
    fresh bot must pick the trail back up from disk."""
    journal_path = tmp_path / "j.jsonl"
    b1 = _bot(FakeBroker(), journal_path, quotes=[210.0])
    s = b1.arm("X", "long", 10, 30.0, 200.0)
    # Simulate crash: drop b1.
    del b1
    # New process starts from scratch.
    b2 = _bot(FakeBroker(), journal_path, quotes=[210.0])
    b2.start()
    try:
        active = b2.list_active()
        assert any(a.trail_id == s.trail_id for a in active)
    finally:
        b2.stop()


def test_terminal_states_not_recovered(tmp_path):
    journal_path = tmp_path / "j.jsonl"
    broker = FakeBroker()
    bot = _bot(broker, journal_path, quotes=[210.0, 250.0, 200.0])
    bot.arm("X", "long", 10, 30.0, 200.0)
    bot._tick(); bot._tick(); bot._tick()    # fires
    # Recovery on a fresh bot must NOT bring it back as active.
    fresh = TrailJournal(journal_path).replay_active()
    assert fresh == {}


# ---------------------------------------------------------------------------
# Failure modes — broker rejection
# ---------------------------------------------------------------------------

def test_broker_rejection_lands_trail_in_error_state(tmp_path):
    broker = FailingBroker()
    bot = _bot(broker, tmp_path / "j.jsonl",
               quotes=[210.0, 250.0, 200.0])
    bot.arm("X", "long", 10, 30.0, 200.0)
    bot._tick(); bot._tick(); bot._tick()
    rows = [json.loads(l) for l in
            (tmp_path / "j.jsonl").read_text().strip().splitlines()]
    final = rows[-1]
    assert final["state"] == "ERROR"
    assert "market closed" in (final["exit_error"] or "")


# ---------------------------------------------------------------------------
# Idempotence — no double fire
# ---------------------------------------------------------------------------

def test_post_fire_ticks_do_nothing(tmp_path):
    broker = FakeBroker()
    bot = _bot(broker, tmp_path / "j.jsonl",
               quotes=[210.0, 250.0, 200.0, 180.0, 100.0])
    bot.arm("X", "long", 10, 30.0, 200.0)
    for _ in range(5):
        bot._tick()
    # Only one exit ever fires, even though every subsequent tick
    # also looks "below trigger".
    assert len(broker.calls) == 1
