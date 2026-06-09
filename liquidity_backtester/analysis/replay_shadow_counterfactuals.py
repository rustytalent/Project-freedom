"""Replay shadow-log events against the next trading session and
write counterfactual outcomes.

Stream L delivery — the part of the flywheel that closes the loop on
shadow events emitted by the brief, options executor, drift monitor,
etc. (See ``docs/CONSOLIDATED_ENHANCEMENT_PLAN.md`` §2 Stream L.)

For each shadow event recorded on trading_date_ist=D, this script:
  1. Loads the next trading session's data (D+1 NSE day).
  2. Computes the counterfactual outcome that maps to the event_kind.
     - avoidance_flag: did the avoided basket actually rally?
     - skip_options_executor: what would the SKIPped trade have
       returned? (placeholder until Stream D wires the SKIP recorder)
     - below_target_to_cost_ratio: same — what would the would-be
       trade have netted?
  3. Records the resolution via ShadowLogger.record_resolution.

Defensive: missing data, schema mismatches, and exceptions on a
single event MUST NOT block the whole batch. The job is idempotent
via shadow_id dedup, so partial runs can be re-run safely.

Usage:
    python -m analysis.replay_shadow_counterfactuals \\
        --shadow-root data/shadow_log \\
        --bundle-root data/warehouse \\
        --event-date 2026-06-09
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from liqpool.products.shadow_log import (
    EVENT_KIND_AVOIDANCE_FLAG,
    EVENT_KIND_BELOW_TARGET_TO_COST,
    EVENT_KIND_DRIFT_FLAG_FIRED,
    EVENT_KIND_MACRO_GATE_CLOSED,
    EVENT_KIND_Q_BELOW_THRESHOLD,
    EVENT_KIND_SKIP_OPTIONS,
    ShadowLogger,
    read_shadow_events,
)


LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-event resolvers
# ---------------------------------------------------------------------------

def _next_session_bars(
    bundle_root: Path, symbol: str, after_date: str,
) -> Optional[pd.DataFrame]:
    """Best-effort next-session OHLCV lookup. Reads
    ``<bundle_root>/<symbol>.parquet`` if present and returns the
    subset of bars on the next IST trading day after ``after_date``.

    Returns None when no data is available so the caller can fall
    back to "unresolvable today" rather than crashing.
    """
    candidate = bundle_root / f"{symbol}.parquet"
    if not candidate.exists():
        return None
    try:
        df = pd.read_parquet(candidate)
    except Exception as exc:
        LOG.warning("failed to read %s: %s", candidate, exc)
        return None
    if df.empty:
        return None
    after = pd.Timestamp(after_date) + pd.Timedelta(days=1)
    # Strip any tz to the warehouse convention (tz-naive UTC).
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
        df.index = idx
    next_day = df[(df.index >= after) & (df.index < after + pd.Timedelta(days=1))]
    if next_day.empty:
        return None
    return next_day


def _resolve_avoidance(
    bundle_root: Path,
    event: pd.Series,
) -> Optional[Dict[str, Any]]:
    """For an avoidance_flag: did the avoided basket actually rally?
    Returns a dict with the next-session close-over-open return and
    a verdict bucket; None if data is unavailable.

    The interpretation: a NEGATIVE return tomorrow vindicates the
    avoidance; a POSITIVE return is the regret signal."""
    symbol = event["symbol"]
    if not symbol or pd.isna(symbol):
        return None
    after_date = event["trading_date_ist"]
    bars = _next_session_bars(bundle_root, symbol, after_date)
    if bars is None or bars.empty:
        return None
    open_px = float(bars["open"].iloc[0])
    close_px = float(bars["close"].iloc[-1])
    high_px = float(bars["high"].max())
    low_px = float(bars["low"].min())
    if open_px <= 0:
        return None
    ret_close_over_open = (close_px - open_px) / open_px
    mfe = (high_px - open_px) / open_px   # max favourable excursion for a long
    mae = (low_px - open_px) / open_px    # max adverse excursion for a long
    verdict = (
        "avoidance_vindicated" if ret_close_over_open <= 0.0
        else "avoidance_regret"
    )
    return {
        "next_session_date_ist": str(bars.index[0].date()),
        "ret_close_over_open": ret_close_over_open,
        "mfe_close_over_open": mfe,
        "mae_close_over_open": mae,
        "verdict": verdict,
    }


def _resolve_below_threshold_long_signal(
    bundle_root: Path,
    event: pd.Series,
) -> Optional[Dict[str, Any]]:
    """For Q-below-threshold / below-target-to-cost rejections that
    look like a "would have been a long if not for the gate" — did
    the next session actually reach the predicted level?

    Uses the decision_context.key_level (for Q-below-threshold) or
    derived target from entry+target_atr (for below-target-to-cost).
    """
    symbol = event.get("symbol")
    if symbol is None or pd.isna(symbol):
        return None
    after_date = event["trading_date_ist"]
    try:
        ctx = json.loads(event["decision_context"])
    except Exception:
        return None
    bars = _next_session_bars(bundle_root, str(symbol), after_date)
    if bars is None or bars.empty:
        return None
    # Find the implied target. Two payload shapes both supported:
    target = ctx.get("key_level")
    side = ctx.get("side_trade") or ctx.get("side")
    if target is None and "entry_price" in ctx and "target_atr" in ctx:
        entry = float(ctx["entry_price"])
        target_atr = float(ctx["target_atr"])
        atr_at_entry = float(ctx.get("atr_at_entry", 0.0))
        if atr_at_entry > 0:
            if side == "long":
                target = entry + target_atr * atr_at_entry
            else:
                target = entry - target_atr * atr_at_entry
    if target is None or side is None:
        return None
    try:
        target = float(target)
    except (TypeError, ValueError):
        return None
    high_px = float(bars["high"].max())
    low_px = float(bars["low"].min())
    if side == "long":
        touched = high_px >= target
    else:
        touched = low_px <= target
    verdict = (
        "rejection_regret" if touched else "rejection_vindicated"
    )
    return {
        "next_session_date_ist": str(bars.index[0].date()),
        "target_level": target,
        "next_session_high": high_px,
        "next_session_low": low_px,
        "side": side,
        "target_touched": bool(touched),
        "verdict": verdict,
    }


def _resolve_macro_gate_closed(
    bundle_root: Path,
    event: pd.Series,
) -> Optional[Dict[str, Any]]:
    """For macro-gate-closed days: did the broader market actually
    move next session? Without an explicit underlying we can't grade
    the SKIP precisely, but we can record next-session range info to
    let the future drift-imminent / macro-gate-cost model see whether
    the gate's caution was justified."""
    symbol = event.get("symbol")
    if symbol is None or pd.isna(symbol):
        return None
    after_date = event["trading_date_ist"]
    bars = _next_session_bars(bundle_root, str(symbol), after_date)
    if bars is None or bars.empty:
        return None
    open_px = float(bars["open"].iloc[0])
    if open_px <= 0:
        return None
    high_px = float(bars["high"].max())
    low_px = float(bars["low"].min())
    close_px = float(bars["close"].iloc[-1])
    realised_range_atr = (high_px - low_px) / open_px
    return {
        "next_session_date_ist": str(bars.index[0].date()),
        "realised_range_open_norm": realised_range_atr,
        "realised_close_over_open": (close_px - open_px) / open_px,
    }


def _resolve_drift_alert(
    bundle_root: Path,
    event: pd.Series,
) -> Optional[Dict[str, Any]]:
    """Drift events resolve over a longer window than next-session;
    the recovery / non-recovery pattern is what the M.6 drift-imminent
    model needs. For now we just record that the event happened on
    `trading_date_ist`; a downstream batch can pair it with later
    drift checks. Returns a stub outcome marking it as "deferred"
    so it's countable but not yet truly resolved."""
    return {
        "resolution_kind": "deferred",
        "note": "drift events resolve over multi-session recovery windows",
    }


# Map event_kind to its resolver. Adding a new kind = adding a new entry.
RESOLVERS = {
    EVENT_KIND_AVOIDANCE_FLAG: _resolve_avoidance,
    EVENT_KIND_Q_BELOW_THRESHOLD: _resolve_below_threshold_long_signal,
    EVENT_KIND_BELOW_TARGET_TO_COST: _resolve_below_threshold_long_signal,
    EVENT_KIND_MACRO_GATE_CLOSED: _resolve_macro_gate_closed,
    EVENT_KIND_SKIP_OPTIONS: _resolve_macro_gate_closed,  # same shape: range-norm read
    EVENT_KIND_DRIFT_FLAG_FIRED: _resolve_drift_alert,
}


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def replay_one_date(
    shadow_root: Path,
    bundle_root: Path,
    event_date: str,
) -> Dict[str, int]:
    """Replay every shadow event written on event_date, attempting to
    resolve each against the next session's data. Returns a summary
    of {scanned, resolved, unresolvable}.

    Idempotent: re-running on the same date re-resolves the same
    shadow_ids; the ShadowLogger's dedupe keeps the latest version.
    """
    events = read_shadow_events(shadow_root, trading_date_ist=event_date)
    if events.empty:
        LOG.info("no events for %s", event_date)
        return {"scanned": 0, "resolved": 0, "unresolvable": 0}

    logger = ShadowLogger(shadow_root)
    resolved = 0
    unresolvable = 0
    for _, ev in events.iterrows():
        kind = ev["event_kind"]
        resolver = RESOLVERS.get(kind)
        if resolver is None:
            unresolvable += 1
            continue
        try:
            outcome = resolver(bundle_root, ev)
        except Exception as exc:
            LOG.warning("resolver failed for %s: %s", ev["shadow_id"], exc)
            outcome = None
        if outcome is None:
            unresolvable += 1
            continue
        # The resolved-on date is the next session's date the resolver
        # found. Fall back to event_date + 1 if absent.
        resolved_on = outcome.get("next_session_date_ist") or str(
            (pd.Timestamp(event_date) + pd.Timedelta(days=1)).date()
        )
        logger.record_resolution(
            shadow_id=ev["shadow_id"],
            trading_date_ist_resolved=resolved_on,
            counterfactual_outcome=outcome,
        )
        resolved += 1
    summary = logger.commit()
    return {
        "scanned": len(events),
        "resolved": resolved,
        "unresolvable": unresolvable,
        **summary,
    }


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shadow-root", required=True, type=Path)
    p.add_argument("--bundle-root", required=True, type=Path,
                   help="root containing per-symbol parquet OHLCV warehouses")
    p.add_argument("--event-date", required=True,
                   help="IST trading date the shadow events were written on, YYYY-MM-DD")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = replay_one_date(args.shadow_root, args.bundle_root, args.event_date)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
