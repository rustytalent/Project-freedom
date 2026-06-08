#!/usr/bin/env python
"""End-to-end Options OOS training run (Stream D Gate-1).

Loads data from the Kite warehouse, builds the D.1 feature frame +
D.2 labels + D.5a layer scores per row, fits the
OptionsExpectedReturnModelSuite (D.3), and persists the per-head
metrics + Gate-1 verdict.

Yahoo 429 / firewall on the VPS is normal — the macro snapshot
will be empty and the macro_score column will be 0 everywhere.
LightGBM handles that natively (the model just doesn't see global-
index features). Training proceeds.

Usage:

  PYTHONPATH=. .venv/bin/python scripts/train_options_model.py \\
    --underlying NIFTY50 \\
    --option-index NIFTY \\
    --start 2024-06-01 \\
    --end 2026-05-31 \\
    --output-dir output_models/options_v1_nifty \\
    --max-expiries 0          # 0 = all; use a small number for smoke tests

  PYTHONPATH=. .venv/bin/python scripts/train_options_model.py \\
    --underlying BANKNIFTY --option-index BANKNIFTY \\
    --start 2024-06-01 --end 2026-05-31 \\
    --output-dir output_models/options_v1_banknifty

Outputs (in --output-dir):
  summary.pkl              — full SuiteSummary pickle for inspection
  per_head_metrics.json    — human-readable per-head dump
  gate1_report.md          — decision summary
  features_sample.parquet  — first 10k rows of the labelled frame
                             (small audit fixture)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Engine modules
from liqpool.warehouse import WarehouseReader, OPTION_INDEX_NAMES
from liqpool.options import (
    OptionsExpectedReturnModelSuite,
    OptionsFeaturizerParams,
    OptionsLabelParams,
    YahooMacroAdapter,
    add_options_labels,
    build_options_feature_frame,
    compute_layer_scores_from_inputs,
    empty_snapshot,
)


LOGGER = logging.getLogger("train_options_model")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--underlying", required=True,
                   help="Index symbol for the SPOT layer (e.g. NIFTY50)")
    p.add_argument("--option-index", required=True,
                   choices=list(OPTION_INDEX_NAMES) + ["NIFTY50"],
                   help="Index symbol for the OPTIONS layer (NIFTY/BANKNIFTY)")
    p.add_argument("--start", required=True,
                   help="ISO date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True,
                   help="ISO date YYYY-MM-DD (inclusive)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to persist summary.pkl + reports")
    p.add_argument("--warehouse-root",
                   default=os.environ.get("GFEED_WAREHOUSE_ROOT", None),
                   help="Override warehouse root. Defaults to env "
                        "GFEED_WAREHOUSE_ROOT or the WarehouseReader default.")
    p.add_argument("--max-expiries", type=int, default=0,
                   help="If > 0, cap the number of expiries loaded for a "
                        "smoke test. 0 = all expiries in the window.")
    p.add_argument("--strikes-around-atm", type=int, default=3,
                   help="Strikes per side per expiry (ATM±N).")
    p.add_argument("--sides", default="buy",
                   help="Comma-separated sides to train: buy / sell / both. "
                        "Both fits 2x heads; buy alone for v1 quick path.")
    p.add_argument("--min-trades", type=int, default=200)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--embargo-bars", type=int, default=13,
                   help="Bars to embargo between train and val.")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--workers", type=int, default=0,
                   help="Threads for the per-expiry data-prep loop. "
                        "0 = auto (min(8, cpu_count)). Pandas releases the "
                        "GIL on the heavy numpy paths so threads scale well "
                        "for the I/O + featurizer work. LightGBM itself "
                        "uses all cores per head fit independently.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_spot_bars(reader: WarehouseReader, underlying: str,
                    start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load 5-min spot bars in the window."""
    df, rep = reader.load_spot(underlying, tf="5min", normalize_utc=False)
    if df is None or df.empty:
        raise RuntimeError(
            f"warehouse returned empty spot frame for {underlying}; "
            f"report={rep}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    mask = (df["timestamp"] >= start) & (
        df["timestamp"] <= end + pd.Timedelta(days=1))
    df = df[mask].reset_index(drop=True)
    LOGGER.info("loaded spot bars: %s rows in window %s → %s",
                f"{len(df):,}", start.date(), end.date())
    return df


def load_greeks(reader: WarehouseReader, option_index: str,
                start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load EOD Greeks for the window. Returns columns the featurizer
    expects (renaming implied_volatility → iv, etc.)."""
    df, rep = reader.load_model_options_with_greeks(option_index,
                                                      drop_missing_iv=True)
    if df is None or df.empty:
        raise RuntimeError(
            f"warehouse returned empty Greeks frame for {option_index}; "
            f"report={rep}")
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.normalize()
    mask = (df["trading_date"] >= start.normalize()) & (
        df["trading_date"] <= end.normalize())
    df = df[mask].copy()
    # Rename to featurizer schema.
    df = df.rename(columns={
        "implied_volatility": "iv",
        "option_price_for_iv": "premium_close",
        "symbol": "underlying",
    })
    # Some Greeks frames carry both CALL and PUT rows per (date, strike).
    # The featurizer asks for one Greeks row per (underlying × strike ×
    # date). We aggregate by averaging the call/put Greeks (delta sign
    # cancels for puts so we use the absolute value to keep magnitude
    # interpretable for the model; the executor sign-flips per side).
    if "side" in df.columns:
        df["delta"] = df["delta"].abs()
        df = (df.groupby(["underlying", "strike", "trading_date"],
                          as_index=False)
                .agg({
                    "delta": "mean", "gamma": "mean",
                    "theta": "mean", "vega": "mean",
                    "iv": "mean", "premium_close": "mean",
                    "days_to_expiry": "min",
                    "spot_close": "first",
                }))
    # Featurizer also wants pcr_oi optionally; we don't have it daily here.
    df["pcr_oi"] = np.nan
    LOGGER.info("loaded Greeks: %s rows in window", f"{len(df):,}")
    return df


def load_macro(reader: WarehouseReader,
                start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Load daily macro (risk-free + India VIX). Builds the frame the
    featurizer's macro_daily kwarg expects."""
    try:
        df, _ = reader.load_macro()
    except Exception as exc:                                      # noqa: BLE001
        LOGGER.warning("macro layer load failed: %s — proceeding without",
                       exc)
        return None
    if df is None or df.empty:
        return None
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.normalize()
    df = df[(df["trading_date"] >= start.normalize())
            & (df["trading_date"] <= end.normalize())]
    # The featurizer's `macro_daily` wants `india_vix_close` + `usdinr_close`.
    # The warehouse macro frame may not carry both; we synthesize the
    # missing column as NaN and rely on align_daily_to_intraday to lag.
    if "india_vix_close" not in df.columns:
        try:
            vix_df, _ = reader.load_spot("INDIAVIX", tf="day")
            if vix_df is not None and not vix_df.empty:
                vix_df = vix_df.copy()
                vix_df["trading_date"] = pd.to_datetime(
                    vix_df["timestamp"]).dt.normalize()
                df = df.merge(
                    vix_df[["trading_date", "close"]].rename(
                        columns={"close": "india_vix_close"}),
                    on="trading_date", how="left")
        except Exception:
            df["india_vix_close"] = np.nan
    if "usdinr_close" not in df.columns:
        df["usdinr_close"] = np.nan
    LOGGER.info("loaded macro: %s rows", f"{len(df):,}")
    return df[["trading_date", "india_vix_close", "usdinr_close"]]


def discover_expiries(greeks: pd.DataFrame, max_expiries: int) -> List[str]:
    """Return distinct expiry dates from the Greeks frame, ascending.

    The warehouse Greeks frame doesn't carry an explicit expiry column,
    only days_to_expiry. We approximate: each (trading_date,
    days_to_expiry) → expiry_date = trading_date + days_to_expiry. The
    distinct set of expiry_dates lands here.
    """
    if "days_to_expiry" not in greeks.columns:
        raise RuntimeError("Greeks frame lacks days_to_expiry column")
    expiry_dates = (
        pd.to_datetime(greeks["trading_date"])
        + pd.to_timedelta(greeks["days_to_expiry"], unit="D")
    ).dt.normalize().unique()
    expiry_strs = sorted(
        pd.to_datetime(expiry_dates).strftime("%Y-%m-%d").tolist())
    if max_expiries and len(expiry_strs) > max_expiries:
        LOGGER.info("capping to %d of %d expiries (smoke test mode)",
                    max_expiries, len(expiry_strs))
        expiry_strs = expiry_strs[:max_expiries]
    return expiry_strs


def select_strikes_per_expiry(greeks: pd.DataFrame, expiry: str,
                                strikes_around_atm: int) -> List[float]:
    """For an expiry, pick ATM ± N strikes from the earliest day
    available in the Greeks frame for that expiry."""
    e_norm = pd.to_datetime(expiry).normalize()
    sub = greeks[
        (pd.to_datetime(greeks["trading_date"])
         + pd.to_timedelta(greeks["days_to_expiry"], unit="D")
        ).dt.normalize() == e_norm
    ].copy()
    if sub.empty:
        return []
    earliest_day = sub["trading_date"].min()
    that_day = sub[sub["trading_date"] == earliest_day]
    if "spot_close" not in that_day.columns or that_day["spot_close"].isna().all():
        # Fallback: use the median of strikes
        return sorted(that_day["strike"].unique().tolist())[
            : 2 * strikes_around_atm + 1]
    spot = float(that_day["spot_close"].dropna().iloc[0])
    strikes = sorted(that_day["strike"].unique().tolist())
    if not strikes:
        return []
    # Find nearest strike to spot
    atm_idx = int(np.argmin([abs(s - spot) for s in strikes]))
    lo = max(0, atm_idx - strikes_around_atm)
    hi = min(len(strikes), atm_idx + strikes_around_atm + 1)
    return strikes[lo:hi]


def load_option_intraday_bars(reader: WarehouseReader, option_index: str,
                                expiry: str, strikes: List[float],
                                sides: List[str]) -> pd.DataFrame:
    """Load intraday option bars for the given (expiry × strike × side)
    grid. Returns a frame with columns expected by add_options_labels:
    timestamp, underlying, strike, premium_close, premium_volume."""
    pieces: List[pd.DataFrame] = []
    for strike in strikes:
        for side in sides:
            try:
                df, _ = reader.load_option_active(
                    option_index, expiry, int(strike), side,
                    normalize_utc=False,
                )
            except Exception as exc:                              # noqa: BLE001
                LOGGER.debug("skipping %s %s %s %s (%s)",
                             option_index, expiry, strike, side, exc)
                continue
            if df is None or df.empty:
                continue
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["underlying"] = option_index
            df["strike"] = float(strike)
            df["premium_close"] = df["close"]
            df["premium_volume"] = df["volume"]
            pieces.append(df[[
                "timestamp", "underlying", "strike",
                "premium_close", "premium_volume",
            ]])
    if not pieces:
        return pd.DataFrame(columns=[
            "timestamp", "underlying", "strike",
            "premium_close", "premium_volume"])
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-expiry worker (thread pool)
# ---------------------------------------------------------------------------

def _process_one_expiry(
    *,
    expiry: str,
    reader: WarehouseReader,
    spot: pd.DataFrame,
    greeks: pd.DataFrame,
    macro_daily: Optional[pd.DataFrame],
    option_index: str,
    sides: List[str],
    strikes_around_atm: int,
    feat_params: OptionsFeaturizerParams,
    label_params: OptionsLabelParams,
) -> Tuple[Optional[pd.DataFrame], str, float]:
    """Worker: build features + load bars + label for one expiry.

    Returns ``(labelled_or_None, status_message, seconds_elapsed)``.
    Safe to call from multiple threads concurrently — only reads from
    ``spot`` / ``greeks`` / ``macro_daily``, and the WarehouseReader
    parquet reads are independent per file.
    """
    t0 = time.perf_counter()
    strikes = select_strikes_per_expiry(greeks, expiry, strikes_around_atm)
    if not strikes:
        return None, "no strikes", time.perf_counter() - t0
    strike_dicts = [
        {"strike": float(k), "side": s, "expiry_date": expiry}
        for k in strikes for s in sides
    ]
    features = build_options_feature_frame(
        underlying_bars=spot, greeks_daily=greeks,
        macro_daily=macro_daily, strikes=strike_dicts,
        underlying=option_index, params=feat_params,
    )
    if features.empty:
        return None, "empty features", time.perf_counter() - t0
    option_bars = load_option_intraday_bars(
        reader, option_index, expiry, strikes, sides)
    if option_bars.empty:
        return None, "no option bars", time.perf_counter() - t0
    labelled = add_options_labels(
        features=features, option_bars=option_bars, params=label_params)
    n_valid = int(labelled["label_valid"].sum())
    elapsed = time.perf_counter() - t0
    status = f"strikes={len(strikes)} valid={n_valid}/{len(labelled)}"
    if n_valid == 0:
        return None, status + " (no valid rows)", elapsed
    return labelled, status, elapsed


# ---------------------------------------------------------------------------
# Layer-score join
# ---------------------------------------------------------------------------

def join_layer_scores(labelled: pd.DataFrame,
                       macro_snapshot) -> pd.DataFrame:
    """Compute the six cascade layer scores per row and append them
    as new columns. The macro snapshot is whatever the YahooMacroAdapter
    produced (empty when the VPS is rate-limited or firewalled).
    """
    LOGGER.info("computing layer scores for %s rows", f"{len(labelled):,}")
    t0 = time.perf_counter()
    rows = []
    # Vectorise the inputs that don't depend on the row.
    # The score functions accept None gracefully.
    cols = labelled.columns
    for r in labelled.itertuples(index=False):
        scores = compute_layer_scores_from_inputs(
            macro_snapshot=macro_snapshot,
            underlying_30m_return=getattr(r, "underlying_30m_return", None),
            underlying_5m_return=getattr(r, "underlying_5m_return", None),
            iv_percentile_60d=getattr(r, "iv_percentile_60d", None),
            theta_per_day_pct=getattr(r, "theta_per_day_pct_yday", None),
            dte_trading_days=getattr(r, "dte_trading_days", None),
            vega_per_volpoint_pct=getattr(r, "vega_per_volpoint_pct_yday",
                                           None),
            # The warehouse path doesn't yet surface p_up / path_efficiency
            # at this layer — pass None; the regime score falls back to the
            # underlying_30m_return proxy.
            p_up=None,
            path_efficiency_30=None,
            direction_changes_30=None,
            vol_regime_zscore_20d=None,
            # Pool: no proximity-model output here yet → score 0.
            proximity_p_60min=None,
            pool_side_from_spot=None,
        )
        rows.append((
            scores["macro"], scores["regime"], scores["pool"],
            scores["options"], scores["micro"], scores["manipulation"],
        ))
    out = labelled.copy()
    arr = np.array(rows, dtype=float)
    out["macro_score"] = arr[:, 0]
    out["regime_score"] = arr[:, 1]
    out["pool_score"] = arr[:, 2]
    out["options_score"] = arr[:, 3]
    out["micro_score"] = arr[:, 4]
    out["manipulation_score"] = arr[:, 5]
    LOGGER.info("layer scores computed in %.1fs",
                time.perf_counter() - t0)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_gate1_markdown(summary, args) -> str:
    """Render the Gate-1 verdict as a Markdown report."""
    lines = []
    lines.append(f"# Gate-1 report — {args.underlying}\n")
    lines.append(f"- Window: {args.start} → {args.end}")
    lines.append(f"- Heads attempted: {summary.head_count}")
    lines.append(f"- Heads fit: {summary.fit_count}")
    lines.append(f"- Heads skipped: {summary.skipped_count}")
    lines.append(f"- **Gate 1 pass: {summary.gate1_pass}**")
    if summary.gate1_winning_buckets:
        lines.append(f"- Winning buckets: {', '.join(summary.gate1_winning_buckets)}")
    lines.append("")
    lines.append("## Per-head metrics")
    for label, m in sorted(summary.per_head.items()):
        lines.append(f"\n### `{label}`")
        lines.append(f"- status: **{m.status}**")
        if m.reason:
            lines.append(f"- reason: `{m.reason}`")
        if m.status != "fit":
            continue
        lines.append(f"- n_train={m.n_train}, n_oos={m.n_oos}")
        lines.append(f"- spearman = **{m.spearman_pred_vs_real:+.3f}**")
        lines.append(f"- top_decile_realized_R = **{m.top_decile_mean_realized:+.3f}**")
        lines.append(f"- top_decile_hit_rate = {m.top_decile_hit_rate:.1%}")
        lines.append(f"- mean_predicted_oos = {m.mean_predicted_oos:+.3f}")
        lines.append(f"- mean_realized_oos = {m.mean_realized_oos:+.3f}")
        if m.calibration_table:
            lines.append("- decile reliability:")
            for r in m.calibration_table:
                lines.append(
                    f"  - d{r.decile:>2}: n={r.n:>4} "
                    f"pred={r.mean_predicted:+.2f} real={r.mean_realized:+.2f} "
                    f"err={r.calibration_error:+.2f}")
        if m.sub_slice_calibration_by_iv_decile:
            max_err = max(abs(r.calibration_error)
                           for r in m.sub_slice_calibration_by_iv_decile)
            lines.append(f"- max |IV-decile sub-slice err| = {max_err:.2f}")
        if m.feature_importance_top_20:
            lines.append("- top features by gain:")
            for name, gain in m.feature_importance_top_20[:10]:
                lines.append(f"  - `{name}` gain={gain:.1f}")
    return "\n".join(lines) + "\n"


def metrics_to_json(summary) -> str:
    """Flat JSON dump of the per-head metrics (CalibrationRow is
    nested; we asdict it carefully)."""
    out = {
        "head_count": summary.head_count,
        "fit_count": summary.fit_count,
        "skipped_count": summary.skipped_count,
        "gate1_pass": summary.gate1_pass,
        "gate1_winning_buckets": list(summary.gate1_winning_buckets),
        "per_head": {},
    }
    for label, m in summary.per_head.items():
        out["per_head"][label] = {
            "status": m.status,
            "reason": m.reason,
            "n_train": m.n_train,
            "n_oos": m.n_oos,
            "mean_predicted_oos": m.mean_predicted_oos,
            "mean_realized_oos": m.mean_realized_oos,
            "spearman_pred_vs_real": m.spearman_pred_vs_real,
            "top_decile_mean_realized": m.top_decile_mean_realized,
            "top_decile_hit_rate": m.top_decile_hit_rate,
            "calibration_table": [asdict(r) for r in m.calibration_table],
            "sub_slice_calibration_by_iv_decile": [
                asdict(r) for r in m.sub_slice_calibration_by_iv_decile
            ],
            "sub_slice_calibration_by_dte_bucket":
                dict(m.sub_slice_calibration_by_dte_bucket),
            "feature_importance_top_20": [
                [n, float(g)] for n, g in m.feature_importance_top_20
            ],
        }
    return json.dumps(out, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    if "both" in sides:
        sides = ["buy", "sell"]
    if not sides:
        sides = ["buy"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    # Warehouse
    reader = WarehouseReader(root=args.warehouse_root) if args.warehouse_root \
        else WarehouseReader()
    LOGGER.info("warehouse root: %s", reader.root)

    # 1. Spot bars
    spot = load_spot_bars(reader, args.underlying, start, end)

    # 2. Greeks
    greeks = load_greeks(reader, args.option_index, start, end)

    # 3. Macro (daily, lagged)
    macro_daily = load_macro(reader, start, end)

    # 4. Overnight macro snapshot (Yahoo) — empty on 429
    try:
        snap = YahooMacroAdapter().fetch()
        if not snap.has_any_data:
            LOGGER.warning("Yahoo snapshot empty (firewalled/rate-limited); "
                           "macro layer will be 0 — training proceeds")
            snap = empty_snapshot()
    except Exception as exc:                                      # noqa: BLE001
        LOGGER.warning("Yahoo fetch raised: %s — using empty snapshot", exc)
        snap = empty_snapshot()

    # 5. Discover expiries + per-expiry strikes
    expiries = discover_expiries(greeks, args.max_expiries)
    LOGGER.info("training on %d expiries", len(expiries))

    # 6. Per-expiry: load option bars, build features, label rows.
    # Runs in a thread pool — pandas releases the GIL on the heavy paths
    # (parquet I/O, groupby, numpy ops), so threads give a real speedup
    # without the pickle overhead of multiprocessing.
    feat_params = OptionsFeaturizerParams()
    label_params = OptionsLabelParams()
    n_workers = (args.workers or min(8, os.cpu_count() or 4))
    LOGGER.info("processing %d expiries with %d worker thread(s)",
                len(expiries), n_workers)
    all_labelled: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_to_expiry = {
            pool.submit(
                _process_one_expiry,
                expiry=e, reader=reader, spot=spot, greeks=greeks,
                macro_daily=macro_daily,
                option_index=args.option_index, sides=sides,
                strikes_around_atm=args.strikes_around_atm,
                feat_params=feat_params, label_params=label_params,
            ): e for e in expiries
        }
        for i, fut in enumerate(as_completed(future_to_expiry), start=1):
            expiry = future_to_expiry[fut]
            try:
                labelled, status, elapsed = fut.result()
            except Exception as exc:                                  # noqa: BLE001
                LOGGER.warning("[%d/%d] %s: FAILED — %s",
                                i, len(expiries), expiry, exc)
                continue
            LOGGER.info("[%d/%d] %s: %s (%.1fs)",
                        i, len(expiries), expiry, status, elapsed)
            if labelled is not None:
                all_labelled.append(labelled)

    if not all_labelled:
        LOGGER.error("no labelled data — aborting before fit")
        return 2

    big = pd.concat(all_labelled, ignore_index=True)
    LOGGER.info("total labelled rows: %s  valid: %s",
                f"{len(big):,}", f"{int(big['label_valid'].sum()):,}")

    # 7. Join layer scores
    big = join_layer_scores(big, snap)

    # Persist a sample of the full frame for audit
    sample_n = min(10_000, len(big))
    big.head(sample_n).to_parquet(
        args.output_dir / "features_sample.parquet", index=False)

    # 8. Fit the suite
    suite = OptionsExpectedReturnModelSuite(
        min_trades=args.min_trades,
        val_frac=args.val_frac,
        embargo_bars=args.embargo_bars,
        seed=args.seed,
    )
    LOGGER.info("fitting suite (min_trades=%d, val_frac=%.2f, embargo=%d)",
                args.min_trades, args.val_frac, args.embargo_bars)
    t0 = time.perf_counter()
    summary = suite.fit(big)
    LOGGER.info("fit complete in %.1fs — heads=%d fit=%d skipped=%d",
                time.perf_counter() - t0,
                summary.head_count, summary.fit_count,
                summary.skipped_count)

    # 9. Persist results
    summary_path = args.output_dir / "summary.pkl"
    with summary_path.open("wb") as f:
        pickle.dump(summary, f)
    LOGGER.info("wrote %s", summary_path)

    metrics_json = args.output_dir / "per_head_metrics.json"
    metrics_json.write_text(metrics_to_json(summary))
    LOGGER.info("wrote %s", metrics_json)

    report_md = args.output_dir / "gate1_report.md"
    report_md.write_text(render_gate1_markdown(summary, args))
    LOGGER.info("wrote %s", report_md)

    # 10. Final Gate-1 verdict to stdout
    print()
    print("=" * 60)
    print(f"Gate 1 verdict: {'PASS' if summary.gate1_pass else 'FAIL'}")
    print(f"Winning buckets: "
          f"{summary.gate1_winning_buckets or '(none)'}")
    print(f"Heads fit: {summary.fit_count} / attempted {summary.head_count}")
    print("=" * 60)
    return 0 if summary.gate1_pass else 1


if __name__ == "__main__":
    sys.exit(main())
