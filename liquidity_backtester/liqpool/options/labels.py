"""Options label generator (Stream D.2).

Produces the per-trade target the OptionsExpectedReturnModel (D.3) is
trained against:

  realized_premium_pct_60min   — premium percent change over the
                                  next 12 bars (60 minutes at 5-min
                                  cadence), AFTER slippage and STT.
  realized_premium_atr_units_60min
                                — the above, normalised by the
                                  per-strike rolling 60-day ATR of
                                  daily premium percent moves.

This is the *label* layer. Winsorization happens at FIT time (D.3),
not here — the label generator records the truth as observed; the
model decides how to handle the tails.

Pinned contracts (six tests in `test_options_labels.py`):

  1. Round-trip math: entry 100 → exit 120 (buy side, zero slippage)
     gives realized_pct = +0.20.
  2. Side sign: same forward move yields +R for buy, −R for sell
     (modulo the side-asymmetric STT cost on the sell leg).
  3. Slippage subtracted at the correct magnitude (`4 ticks × tick_inr
     / entry_premium` round-trip default).
  4. Zero-volume entry OR exit bar → label = NaN, `label_valid = False`.
  5. Causality: the per-strike ATR denominator uses ONLY data from
     trading days strictly before the label window opens (1-day lag
     enforced via `align_daily_to_intraday`).
  6. No-lookahead under truncation: same label at a given bar whether
     the option_bars frame is truncated to that bar + horizon or run
     in full.

Design notes:

  * Slippage and STT are baked into the LABEL, not into the model.
    This keeps the model's predicted_R directly interpretable as
    "net realized return per trade" without a post-fit adjustment.
  * STT applies on the SELL side only (Indian options selling pays
    STT on premium received; buying does not). Parameterised
    (defaults to 0.0625% per the current SEBI grid; trivially
    updated when the grid changes).
  * No brokerage at v1: Indian retail brokers charge ₹0 on options
    BUY trades; SELL is typically ₹20 flat which we treat as
    negligible relative to typical contract values. Parameter
    `brokerage_pct_per_side` allows callers to override.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..warehouse import align_daily_to_intraday


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class OptionsLabelParams:
    """Configuration for the label generator.

    All defaults are v1 production assumptions per
    ``docs/options_strategy_methodology.md``. Override per-environment
    as the SEBI grid moves.
    """
    horizon_bars: int = 12              # 60 minutes at 5-min cadence
    tick_inr: float = 0.05              # NIFTY/BANKNIFTY weekly option tick
    slippage_ticks_per_leg: int = 2     # 2 ticks half-spread + 0 ticks
                                        # extra → 2 ticks per fill. Round
                                        # trip = 2 * 2 = 4 ticks.
    stt_sell_pct: float = 0.000625      # 0.0625% on premium for SELL leg
    brokerage_pct_per_side: float = 0.0 # retail buying is brokerage-free
    atr_window_days: int = 60           # per-strike rolling daily ATR window
    min_atr_days_warmup: int = 30       # minimum days before ATR is usable


# Output column names — appended to whatever the input frame already has.
LABEL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "premium_entry", "premium_exit",
    "slippage_pct", "stt_pct",
    "realized_premium_pct_60min",
    "atr_premium_pct_60d",
    "realized_premium_atr_units_60min",
    "label_valid",
)


# ---------------------------------------------------------------------------
# Per-strike daily ATR of premium percent moves
# ---------------------------------------------------------------------------

def _daily_premium_returns(option_bars: pd.DataFrame) -> pd.DataFrame:
    """Compute daily premium percent returns per (underlying, strike).

    The premium series is intraday; we resample to daily by taking
    the last close of each trading date. The day-over-day return is
    used as the input to the per-strike ATR computation.
    """
    df = option_bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["trading_date"] = df["timestamp"].dt.normalize()
    daily = (df.sort_values("timestamp")
               .groupby(["underlying", "strike", "trading_date"],
                        as_index=False)
               .agg(premium_eod=("premium_close", "last")))
    daily = daily.sort_values(
        ["underlying", "strike", "trading_date"]).reset_index(drop=True)
    daily["premium_return_pct"] = (
        daily.groupby(["underlying", "strike"])["premium_eod"]
             .pct_change(fill_method=None)
    )
    return daily


def _rolling_premium_atr(daily: pd.DataFrame,
                          window: int, warmup: int) -> pd.DataFrame:
    """Per-strike rolling 60-day mean of |daily return| — the
    'ATR-of-percent' denominator used to normalise labels.

    Causal: the window ends at the row of the daily frame; no row is
    a function of future rows. The intraday-side bar gets the
    previous-day value via ``align_daily_to_intraday``, so the label
    on an intraday bar at date D never sees the daily ATR computed
    on a window that ends on date D — only on D-1.
    """
    daily = daily.copy()
    daily["atr_premium_pct_60d"] = (
        daily.groupby(["underlying", "strike"])["premium_return_pct"]
             .transform(
                 lambda s: s.abs().rolling(window, min_periods=warmup).mean()
             )
    )
    return daily[["trading_date", "underlying", "strike",
                  "atr_premium_pct_60d"]]


# ---------------------------------------------------------------------------
# Slippage + STT formulas
# ---------------------------------------------------------------------------

def _slippage_pct(entry_premium: pd.Series,
                  params: OptionsLabelParams) -> pd.Series:
    """Round-trip slippage as a fraction of entry premium.

    Computed as ``(2 * slippage_ticks_per_leg * tick_inr) /
    entry_premium``. Two legs = entry + exit.
    """
    cost_inr = 2 * params.slippage_ticks_per_leg * params.tick_inr
    return cost_inr / entry_premium.replace(0, np.nan)


def _stt_pct(side: pd.Series,
             params: OptionsLabelParams) -> pd.Series:
    """STT cost as a fraction of entry premium. Only on SELL leg.

    Returns 0 for buy rows, ``stt_sell_pct`` for sell rows. NaN
    propagates through if `side` is NaN somehow.
    """
    return np.where(side == "sell", params.stt_sell_pct, 0.0)


# ---------------------------------------------------------------------------
# Top-level: append labels to a feature frame
# ---------------------------------------------------------------------------

def add_options_labels(features: pd.DataFrame,
                       option_bars: pd.DataFrame,
                       params: Optional[OptionsLabelParams] = None,
                       ) -> pd.DataFrame:
    """Append label columns to a feature frame.

    Parameters
    ----------
    features
        Output of :func:`liqpool.options.featurizer.build_options_feature_frame`.
        Must carry at least ``bar_ts, underlying, strike, side``.
    option_bars
        Intraday 5-min option bars per (underlying × strike × bar) with
        columns ``timestamp, underlying, strike, premium_close,
        premium_volume``. The premium_volume column is the gate for
        the ``label_valid`` flag.
    params
        Label configuration; defaults to the v1 production values.

    Returns
    -------
    pandas.DataFrame
        ``features`` with the columns in :data:`LABEL_OUTPUT_COLUMNS`
        appended. Rows where the label cannot be computed (missing
        bars, zero volume, insufficient warmup) carry ``label_valid =
        False`` and ``NaN`` for the numeric label columns.
    """
    p = params or OptionsLabelParams()
    out = features.copy()
    if "bar_ts" not in out.columns:
        raise ValueError("features frame must carry a 'bar_ts' column")
    out["bar_ts"] = pd.to_datetime(out["bar_ts"])

    # ---- Build (entry_ts, strike, side) → premium lookup ------------------
    bars = option_bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.sort_values(
        ["underlying", "strike", "timestamp"]).reset_index(drop=True)

    # The exit bar is `horizon_bars` 5-min steps after entry. We compute
    # the exit ts per-row, then point-merge to the option_bars frame.
    horizon = pd.Timedelta(minutes=5 * p.horizon_bars)
    out["exit_ts"] = out["bar_ts"] + horizon

    # Entry-bar premium + volume per (underlying, strike, bar_ts).
    entry_lookup = bars[[
        "timestamp", "underlying", "strike",
        "premium_close", "premium_volume",
    ]].rename(columns={
        "timestamp": "bar_ts",
        "premium_close": "premium_entry",
        "premium_volume": "volume_entry",
    })
    out = out.merge(entry_lookup,
                     on=["bar_ts", "underlying", "strike"],
                     how="left")

    # Exit-bar premium + volume per (underlying, strike, exit_ts).
    exit_lookup = bars[[
        "timestamp", "underlying", "strike",
        "premium_close", "premium_volume",
    ]].rename(columns={
        "timestamp": "exit_ts",
        "premium_close": "premium_exit",
        "premium_volume": "volume_exit",
    })
    out = out.merge(exit_lookup,
                     on=["exit_ts", "underlying", "strike"],
                     how="left")

    # ---- Slippage + STT ---------------------------------------------------
    out["slippage_pct"] = _slippage_pct(out["premium_entry"], p)
    out["stt_pct"] = _stt_pct(out["side"], p)

    # ---- Gross then net realised return -----------------------------------
    raw_pct = (out["premium_exit"] - out["premium_entry"]) / out[
        "premium_entry"].replace(0, np.nan)
    # Side flip for sell: profit is negative-of-price-change.
    raw_pct = np.where(out["side"] == "sell", -raw_pct, raw_pct)
    realised_net = raw_pct - out["slippage_pct"] - out["stt_pct"] \
        - 2 * p.brokerage_pct_per_side
    out["realized_premium_pct_60min"] = realised_net

    # ---- ATR-of-premium-% denominator (causal, 1-day lagged) --------------
    daily = _daily_premium_returns(option_bars)
    atr = _rolling_premium_atr(
        daily, p.atr_window_days, p.min_atr_days_warmup)

    # Join the daily ATR onto the intraday rows with the standard 1-day
    # lag. We per-strike split because align_daily_to_intraday expects
    # a single feature frame keyed by date.
    pieces: list[pd.DataFrame] = []
    for (underlying, strike), grp in out.groupby(["underlying", "strike"],
                                                  sort=False):
        a = atr[(atr["underlying"] == underlying)
                & (np.isclose(atr["strike"], strike))][
                    ["trading_date", "atr_premium_pct_60d"]].copy()
        if a.empty:
            grp = grp.copy()
            grp["atr_premium_pct_60d"] = np.nan
        else:
            grp = grp.copy()
            grp = align_daily_to_intraday(
                grp, a, intraday_ts="bar_ts", daily_date="trading_date",
            )
        pieces.append(grp)
    out = pd.concat(pieces, ignore_index=True)

    # ---- Normalised label ------------------------------------------------
    out["realized_premium_atr_units_60min"] = (
        out["realized_premium_pct_60min"]
        / out["atr_premium_pct_60d"].replace(0, np.nan)
    )

    # ---- Validity flag ---------------------------------------------------
    valid = (
        out["premium_entry"].notna()
        & out["premium_exit"].notna()
        & (out["premium_entry"] > 0)
        & out.get("volume_entry", pd.Series(True, index=out.index)).fillna(0).gt(0)
        & out.get("volume_exit", pd.Series(True, index=out.index)).fillna(0).gt(0)
        & out["atr_premium_pct_60d"].notna()
        & (out["atr_premium_pct_60d"] > 0)
    )
    out["label_valid"] = valid.astype(bool)
    # Mask the numeric labels where invalid so downstream consumers
    # don't accidentally train on zeros.
    out.loc[~out["label_valid"], "realized_premium_pct_60min"] = np.nan
    out.loc[~out["label_valid"], "realized_premium_atr_units_60min"] = np.nan

    # Drop the helper columns we used for merging — they were never
    # part of the public output schema.
    drop_cols = [c for c in ("exit_ts", "volume_entry", "volume_exit")
                 if c in out.columns]
    out = out.drop(columns=drop_cols)

    # Deterministic row order: the per-strike concat above + the
    # align_daily_to_intraday internal sort can scramble bars on
    # the same day. Lock the order so iloc/index-based downstream
    # consumers see truncation as truncation (not as reordering).
    return out.sort_values(
        ["underlying", "strike", "side", "bar_ts"]
    ).reset_index(drop=True)
