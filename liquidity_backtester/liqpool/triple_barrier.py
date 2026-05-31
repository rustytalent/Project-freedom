"""Triple-barrier labels for path-conditional supervised learning.

Standard Lopez de Prado / Bailey & Lopez de Prado triple-barrier method:
from an entry timestamp, simulate forward bar-by-bar until one of three
barriers is reached -- profit barrier, adverse barrier, or time horizon.
Returns the realized R outcome.

This is the LABEL the user's "force model" needs: a number that says "if you
had entered with these specific barriers, this is the actual R you would
have realized," rather than the binary endpoint labels the direction /
proximity / reaction models train on.

Implementation notes:
  * Intraday MIS is enforced via the same helpers as
    :func:`liqpool.execution_backtest.simulate_pool_trade`: time horizon is
    capped at the last bar of the entry's IST trading day with minute
    <= 15:15.
  * R is normalized to the stop distance, so a trade that hits target at
    target_atr=2.0 with stop_atr=0.5 returns R = +4.0. A stop-out returns
    -1.0 (or worse if the bar gaps through). A time/EOD exit returns the
    realized close-to-entry distance in stop units (continuous).
  * Conservative tie-breaking: when a single bar's range includes both stop
    AND target, the stop is assumed hit first (mirrors V1 simulator).
  * Returns (realized_r, exit_reason, exit_bar_idx). exit_reason is one of
    ``"target"``, ``"stop"``, ``"time_exit"``, ``"eod_squareoff"``,
    ``"no_room"`` (no same-day bars after entry).
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from liqpool.execution_backtest import _last_intraday_bar_idx


def triple_barrier_label(
    df_base: pd.DataFrame,
    entry_idx: int,
    side: str,
    entry_price: float,
    stop_atr: float,
    target_atr: float,
    horizon_bars: int,
    atr_value: float,
) -> Tuple[float, str, int]:
    """Triple-barrier realized R.

    Parameters
    ----------
    df_base : OHLCV frame indexed by timestamp (tz-naive UTC per the codebase
        convention; IST is derived by +5:30).
    entry_idx : bar index of the entry.
    side : ``"long"`` or ``"short"``.
    entry_price : actual fill price at entry_idx (may differ from idx[entry_idx]
        close if there is slippage; the function does NOT model slippage).
    stop_atr : stop distance in ATR units (positive number).
    target_atr : target distance in ATR units (positive number).
    horizon_bars : maximum bars before time-barrier exits.
    atr_value : ATR in price units at entry_idx.

    Returns
    -------
    (realized_r, exit_reason, exit_bar_idx)
        realized_r is signed: positive for profit, negative for loss, in
        units of the stop distance. exit_reason is one of "target", "stop",
        "time_exit", "eod_squareoff", "no_room".
    """
    if df_base.empty or entry_idx < 0 or entry_idx >= len(df_base):
        return 0.0, "no_room", max(0, entry_idx)
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")

    atr = max(float(atr_value), 1e-9)
    stop_dist = float(stop_atr) * atr
    target_dist = float(target_atr) * atr
    if stop_dist <= 0:
        return 0.0, "no_room", entry_idx

    if side == "long":
        stop_price = entry_price - stop_dist
        target_price = entry_price + target_dist
    else:
        stop_price = entry_price + stop_dist
        target_price = entry_price - target_dist

    # MIS: cap the horizon at the same-day EOD bar.
    idx = df_base.index
    natural_end = min(entry_idx + int(horizon_bars), len(df_base) - 1)
    eod_end = _last_intraday_bar_idx(idx, entry_idx, natural_end)
    effective_end = min(natural_end, eod_end)
    if effective_end <= entry_idx:
        return 0.0, "no_room", entry_idx

    # Bar-by-bar scan. Stop checked BEFORE target on the same bar — conservative
    # (mirrors the V1 simulator's _exit_trade) and the tighter end of the
    # ambiguity range when a bar straddles both.
    for j in range(entry_idx + 1, effective_end + 1):
        row = df_base.iloc[j]
        low = float(row["low"])
        high = float(row["high"])
        open_ = float(row["open"])

        if side == "long":
            stop_hit = low <= stop_price
            target_hit = high >= target_price
            if stop_hit:
                # Gap-down past stop fills at the (worse) open.
                fill = min(open_, stop_price)
                return float((fill - entry_price) / stop_dist), "stop", j
            if target_hit:
                fill = target_price                            # limit capped
                return float((fill - entry_price) / stop_dist), "target", j
        else:
            stop_hit = high >= stop_price
            target_hit = low <= target_price
            if stop_hit:
                fill = max(open_, stop_price)
                return float((entry_price - fill) / stop_dist), "stop", j
            if target_hit:
                fill = target_price
                return float((entry_price - fill) / stop_dist), "target", j

    # Time / EOD exit at the effective_end close.
    final_close = float(df_base["close"].iloc[effective_end])
    if side == "long":
        r = (final_close - entry_price) / stop_dist
    else:
        r = (entry_price - final_close) / stop_dist
    reason = "eod_squareoff" if effective_end < natural_end else "time_exit"
    return float(r), reason, int(effective_end)
