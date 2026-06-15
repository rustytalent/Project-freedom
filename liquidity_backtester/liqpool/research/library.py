"""Hypothesis library — the recipes we are systematically testing.

The Profitability Doctrine ranks 10 candidates. This module ships the
first few coded against ``HypothesisHarness``. Each lives as a tiny
pure function so it can be replaced, parametrised, or A/B'd without
touching the harness or the rest of the codebase.

Discipline:
  * Every hypothesis carries a docstring with the THESIS (why we
    think it might work), the KILL CONDITION (what falsifies it),
    and the REGIME WHERE IT FAILS (the operator's eyes during live).
  * No hypothesis is "promoted to live" without a HypothesisReport
    showing OOS Sharpe ≥ 1.0 on 18+ months of warehouse data AND
    surviving 4 weeks of paper-mode validation.
  * Hypotheses are pure functions of the bars they're given. No
    network calls. No file reads. Deterministic for the same input.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from .harness import HypothesisFn, HypothesisSpec

HYPOTHESIS_LIBRARY: Dict[str, HypothesisSpec] = {}


def register_hypothesis(spec: HypothesisSpec) -> HypothesisSpec:
    """Decorator-friendly registrar so a library author can write
    ``HYPOTHESIS_LIBRARY['name'] = register_hypothesis(spec)`` or
    just call this with a populated spec."""
    HYPOTHESIS_LIBRARY[spec.name] = spec
    return spec


# ─────────────────────────────────────────────────────────────────
# H1 — Thursday theta scalp
# ─────────────────────────────────────────────────────────────────
def h1_thursday_theta_scalp(bars, qty: int = 75,
                              enter_minute_of_day: int = 30 * 60 + 0,
                              exit_minute_of_day: int = 14 * 60 + 30 + 0,
                              ) -> List[Tuple[int, int, int, int]]:
    """Thursday theta scalp: SHORT the ATM straddle at 09:30 IST, CLOSE
    by 14:30 IST. Holds through the bulk of theta-decay.

    THESIS: Theta is non-linear in last day. Intraday realised vol
    on expiry day historically underprices the implied. Selling
    premium captures the decay.

    KILL CONDITION: Sharpe < 1.0 on OOS expiry-day data after costs,
    OR max drawdown > 30% of peak equity.

    REGIME WHERE IT FAILS: Gap moves on expiry mornings (FOMC, RBI,
    bank result days), BankNIFTY-shock days that pull NIFTY 1%+
    intraday.

    For backtest purposes here, the hypothesis operates on a single
    bar series (the underlying NIFTY spot, or an ATM-straddle premium
    series the caller pre-computed). The harness treats the result
    as a single short position on that series. In production, you
    open both legs separately — that wiring lives in the runner.

    Params:
        bars: DataFrame with columns ts, open, high, low, close
        qty: position quantity (1 lot = 75 for NIFTY)
        enter_minute_of_day: IST minute-of-day for entry (default 09:30)
        exit_minute_of_day: IST minute-of-day for exit (default 14:30)

    Returns:
        list of (entry_idx, exit_idx, side, qty) tuples — one per
        Thursday in the bar series.
    """
    if "ts" not in bars.columns or len(bars) < 2:
        return []
    out: List[Tuple[int, int, int, int]] = []
    try:
        import pandas as pd
        ts = pd.to_datetime(bars["ts"])
    except Exception:
        return []
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    day_of_week = ts.dt.dayofweek
    date_key = ts.dt.strftime("%Y-%m-%d")

    # For each Thursday (dow=3) in the series, find the first bar at
    # or after enter_minute_of_day and the first bar at or after
    # exit_minute_of_day, on the same date.
    df = bars.copy()
    df["mod"] = minute_of_day.values
    df["dow"] = day_of_week.values
    df["date"] = date_key.values
    by_day = df[df["dow"] == 3].groupby("date")
    for date_str, group in by_day:
        idxs = group.index.values
        entries = group[group["mod"] >= enter_minute_of_day]
        exits = group[group["mod"] >= exit_minute_of_day]
        if len(entries) == 0 or len(exits) == 0:
            continue
        entry_idx = int(entries.index[0])
        exit_idx = int(exits.index[0])
        if exit_idx <= entry_idx:
            continue
        # SHORT side = -1; harness's P&L direction handles it
        out.append((entry_idx, exit_idx, -1, qty))
    return out


HYPOTHESIS_LIBRARY["h1_thursday_theta_scalp"] = register_hypothesis(
    HypothesisSpec(
        name="h1_thursday_theta_scalp",
        fn=h1_thursday_theta_scalp,
        params={"qty": 75},
        instrument_kind="options",
        lot_size=75,
        note="Sell ATM straddle 09:30 → 14:30 every Thursday. Captures "
             "non-linear theta decay on expiry day.",
    ))


# ─────────────────────────────────────────────────────────────────
# H2 — Opening-range failure fade
# ─────────────────────────────────────────────────────────────────
def h2_opening_range_failure_fade(
    bars, qty: int = 75,
    or_minutes: int = 30,
    fail_threshold_pct: float = 0.5,
    hold_minutes: int = 60,
) -> List[Tuple[int, int, int, int]]:
    """Fade the opening range when it fails to extend.

    THESIS: Indian retail flow chases the open. When the first ``or_minutes``
    creates a range that does NOT extend by ``fail_threshold_pct`` of
    its size in the next ``or_minutes``, the range typically mean-reverts.
    Fading at the wrong end has positive expectancy historically.

    KILL CONDITION: Sharpe < 1.0 OR DD > 30%. Specifically dies on
    trending news-driven days (RBI policy, US CPI Wednesday).

    REGIME WHERE IT FAILS: One-way trending days. Use the regime
    classifier in `liqpool.regime` to filter when going live.

    Logic per day:
        1. After the first or_minutes, define OR = [low, high].
        2. After ``or_minutes`` MORE bars, if neither side has extended
           by ``fail_threshold_pct * (high - low)``, the range "failed."
        3. Fade in the direction of mean-reversion: if last close > OR
           midpoint → SHORT (qty units); if below → LONG.
        4. Exit ``hold_minutes`` later, OR at session close.
    """
    if "ts" not in bars.columns or len(bars) < 2:
        return []
    out: List[Tuple[int, int, int, int]] = []
    try:
        import pandas as pd
    except Exception:
        return []
    ts = pd.to_datetime(bars["ts"])
    df = bars.copy()
    df["ts"] = ts
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mod"] = ts.dt.hour * 60 + ts.dt.minute

    bar_minutes = bars.attrs.get("bar_minutes", 5)
    or_bars = max(1, int(or_minutes / bar_minutes))
    hold_bars = max(1, int(hold_minutes / bar_minutes))

    for date_str, group in df.groupby("date"):
        idxs = group.index.values.tolist()
        if len(idxs) < or_bars * 2 + 1:
            continue
        or_window = group.iloc[:or_bars]
        or_high = float(or_window["high"].max())
        or_low = float(or_window["low"].min())
        or_size = or_high - or_low
        if or_size <= 0:
            continue
        threshold = or_size * fail_threshold_pct
        extend_window = group.iloc[or_bars: or_bars * 2]
        extended_up = extend_window["high"].max() > or_high + threshold
        extended_dn = extend_window["low"].min() < or_low - threshold
        if extended_up or extended_dn:
            continue                                # OR did extend; skip
        # Fade decision
        decision_bar = group.iloc[or_bars * 2 - 1]
        decision_idx = int(group.index[or_bars * 2 - 1])
        last_close = float(decision_bar["close"])
        mid = (or_high + or_low) / 2.0
        side = -1 if last_close > mid else 1
        exit_idx = min(decision_idx + hold_bars, int(group.index[-1]))
        if exit_idx <= decision_idx:
            continue
        out.append((decision_idx, exit_idx, side, qty))
    return out


HYPOTHESIS_LIBRARY["h2_opening_range_failure_fade"] = register_hypothesis(
    HypothesisSpec(
        name="h2_opening_range_failure_fade",
        fn=h2_opening_range_failure_fade,
        params={"qty": 75, "or_minutes": 30, "fail_threshold_pct": 0.5,
                "hold_minutes": 60},
        instrument_kind="equity",       # for the spot series; option mapping at runtime
        note="When the first 30-min OR fails to extend by 50% in the "
             "next 30 min, fade toward the mid for 60 min.",
    ))


# ─────────────────────────────────────────────────────────────────
# H5 — Time-of-day expectancy filter (baseline / sanity check)
# ─────────────────────────────────────────────────────────────────
def h5_time_of_day_buy_and_hold(
    bars, qty: int = 75,
    enter_minute_of_day: int = 9 * 60 + 45,
    exit_minute_of_day: int = 14 * 60 + 45,
) -> List[Tuple[int, int, int, int]]:
    """Simplest possible baseline: BUY the underlying at 09:45 IST,
    SELL at 14:45 IST, every trading day.

    THESIS: A 30-minute warmup after open avoids opening noise; a
    45-minute pre-close exit avoids closing auction noise. If THIS
    has no edge then we're not getting hurt by noise in the harness.

    KILL CONDITION: Almost certain. Sharpe near zero; this is a noise
    benchmark. If the harness shows Sharpe > 0.5 on a buy-and-hold
    spot baseline, our cost model is too lax.

    REGIME WHERE IT FAILS: Choppy days. Wins on trending days.
    Average: near zero. Useful as a calibration check.
    """
    if "ts" not in bars.columns or len(bars) < 2:
        return []
    out: List[Tuple[int, int, int, int]] = []
    try:
        import pandas as pd
        ts = pd.to_datetime(bars["ts"])
    except Exception:
        return []
    df = bars.copy()
    df["mod"] = ts.dt.hour * 60 + ts.dt.minute
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["dow"] = ts.dt.dayofweek

    for date_str, group in df.groupby("date"):
        if group["dow"].iloc[0] >= 5:
            continue                            # weekend (shouldn't happen, defensive)
        entries = group[group["mod"] >= enter_minute_of_day]
        exits = group[group["mod"] >= exit_minute_of_day]
        if len(entries) == 0 or len(exits) == 0:
            continue
        entry_idx = int(entries.index[0])
        exit_idx = int(exits.index[0])
        if exit_idx <= entry_idx:
            continue
        out.append((entry_idx, exit_idx, +1, qty))
    return out


HYPOTHESIS_LIBRARY["h5_time_of_day_buy_and_hold"] = register_hypothesis(
    HypothesisSpec(
        name="h5_time_of_day_buy_and_hold",
        fn=h5_time_of_day_buy_and_hold,
        params={"qty": 75},
        instrument_kind="equity",
        note="Noise baseline: long the spot from 09:45 to 14:45 every "
             "trading day. If THIS earns Sharpe > 0.5 our cost model "
             "is broken.",
    ))


def hypothesis_names() -> List[str]:
    """Return the registered hypothesis names in stable order."""
    return sorted(HYPOTHESIS_LIBRARY)


def get_hypothesis(name: str) -> HypothesisSpec:
    if name not in HYPOTHESIS_LIBRARY:
        raise KeyError(f"unknown hypothesis: {name}; "
                       f"available: {hypothesis_names()}")
    return HYPOTHESIS_LIBRARY[name]
