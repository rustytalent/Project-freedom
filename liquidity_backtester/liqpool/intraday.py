"""Intraday MIS (margin-intraday-square-off) primitives.

Single source of truth for the rules every MIS-aware piece of code obeys:
session boundaries, EOD square-off cap, no-new-entry-after cap, and the
"same IST trading day" helpers.

Extracted to its own module so it can be imported by both the
SIMULATOR (``liqpool.execution_backtest``) and the LABELLING pipeline
(``liqpool.tester`` and ``liqpool.timing``) without a circular dependency.

Convention: the codebase normalizes parquet timestamps to tz-naive UTC
(see ``liqpool.data._normalise_parquet_ohlcv``). IST is derived by adding
5h30m. ``nse_session`` in ``liqpool.regime`` follows the same convention.
"""
from __future__ import annotations

import pandas as pd


# NSE session boundaries in IST minutes-of-day.
SESSION_OPEN_IST_MIN: int = 9 * 60 + 15           # 09:15 IST
SESSION_CLOSE_IST_MIN: int = 15 * 60 + 30         # 15:30 IST (NSE close)

# MIS-trader operational caps:
EOD_SQUAREOFF_IST_MIN: int = 15 * 60 + 15         # 15:15 IST — last bar a label
                                                  # is allowed to read from, and the
                                                  # last bar the simulator allows an
                                                  # exit on.
NO_NEW_ENTRY_AFTER_IST_MIN: int = 14 * 60 + 30    # 14:30 IST — last allowed entry
                                                  # bar (a stop+target trade needs
                                                  # ~60 min of runway).


def ist_minute_of_day(ts: pd.Timestamp) -> int:
    """IST minute-of-day. Treats tz-naive as UTC (production data convention)."""
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    ist = ts + pd.Timedelta(hours=5, minutes=30)
    return int(ist.hour * 60 + ist.minute)


def ist_date(ts: pd.Timestamp) -> pd.Timestamp:
    """IST calendar date as a midnight Timestamp. Same TZ convention."""
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return (ts + pd.Timedelta(hours=5, minutes=30)).normalize()


def last_intraday_bar_idx(idx: pd.DatetimeIndex, anchor_idx: int,
                          hard_max_idx: int) -> int:
    """Largest bar index j in [anchor_idx, hard_max_idx] such that bar j's
    timestamp is on the SAME IST trading day as anchor_idx AND its IST
    minute-of-day is <= ``EOD_SQUAREOFF_IST_MIN``.

    Returns ``anchor_idx`` if none of the following bars qualify (caller treats
    that as "no same-session bars left").

    This is the function that converts "wall-clock 200-bar window" into
    "same-session intraday-MIS window."
    """
    if anchor_idx < 0 or anchor_idx >= len(idx):
        return anchor_idx
    anchor_date = ist_date(idx[anchor_idx])
    last = anchor_idx
    for j in range(anchor_idx + 1, min(hard_max_idx, len(idx) - 1) + 1):
        ts = idx[j]
        if ist_date(ts) != anchor_date:
            break                                  # crossed to next session
        if ist_minute_of_day(ts) > EOD_SQUAREOFF_IST_MIN:
            break                                  # past hard square-off
        last = j
    return last


def is_session_minute(minute_of_day: int) -> bool:
    """True if `minute_of_day` is inside the NSE intraday session window."""
    return SESSION_OPEN_IST_MIN <= int(minute_of_day) <= SESSION_CLOSE_IST_MIN
