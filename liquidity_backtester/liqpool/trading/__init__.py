"""Live trading utilities — what you USE while trading manually.

Distinct from liqpool/execution/* (which is the BACKTEST simulator
suite) and liqpool/options/* (which is the research-side options
methodology). This package is the small set of tools that talk to
the live broker on behalf of a human who's trading by hand.

v1 contents:
  trailing_stop_bot  — the "lock in profit when I say so" bot
                       described in docs/trailing_stop_bot_spec.md
"""
from .trailing_stop_bot import (
    TrailJournal,
    TrailState,
    TrailingStopBot,
    default_kite_quote,
)

__all__ = [
    "TrailJournal",
    "TrailState",
    "TrailingStopBot",
    "default_kite_quote",
]
