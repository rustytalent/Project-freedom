"""Phase 4 Task 2: Track A pre-touch directional sweep.

This is a research runner for the pre-touch thesis:

* proximity says a pool is likely to be reached,
* direction agrees with the pool side,
* enter on the next bar and target the approach move before the touch.

The script deliberately keeps Q as a post-hoc cohort, enforces intraday
session-end exits, and reports direction/sector/time stability so the sweep
does not create pretty but untradeable false positives.
"""

from __future__ import annotations

import argparse
import itertools
import math
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqpool.execution_simulator_v2 import (  # noqa: E402
    _buy_sell_prices,
    _load_symbol_1m,
    _slipped_price,
    compute_slippage_bps,
    compute_zerodha_intraday_costs,
    resolve_intrabar_path,
)
from liqpool.indicators import atr  # noqa: E402
try:  # noqa: E402
    from q_audit_common import fnum, markdown_table, pct
except ModuleNotFoundError:  # pragma: no cover - package import path
    from analysis.q_audit_common import fnum, markdown_table, pct
try:  # noqa: E402
    from q_rescale import exact_top_mask
except ModuleNotFoundError:  # pragma: no cover - package import path
    from analysis.q_rescale import exact_top_mask


DEFAULT_MODEL_REPORT = Path("output_models/core25_phase4_v2_neutral/multi_asset_report.pkl")
DEFAULT_FEATURE_STORE = Path("output_feature_store/core25_phase4_v2_neutral")
DEFAULT_OUT_DIR = Path("output_phase4_track_a_pretouch_sweep")
DEFAULT_REPORT = Path("reports/phase4_track_a_sweep.md")
DEFAULT_CSV = Path("reports/phase4_track_a_sweep_cells.csv")
DEFAULT_PREFLIGHT = Path("reports/phase4_track_a_sweep_preflight.csv")

Q_COHORTS = [
    ("all", 1.00),
    ("top_50pct", 0.50),
    ("top_25pct", 0.25),
    ("top_10pct", 0.10),
    ("top_5pct", 0.05),
]


@dataclass(frozen=True)
class Geometry:
    target_fraction: float
    stop_atr_mult: float
    max_hold_bars: int


@dataclass(frozen=True)
class Gate:
    min_p_touch: float
    min_p_direction: float
    distance_low: float
    distance_high: float

    @property
    def label(self) -> str:
        return (
            f"T>={self.min_p_touch:.2f}|D>={self.min_p_direction:.2f}|"
            f"{self.distance_low:g}-{self.distance_high:g}ATR"
        )


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ranges(raw: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split("-", 1)
        out.append((float(left), float(right)))
    return out


def _parse_symbols(raw: str) -> Optional[set[str]]:
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _target_notional(raw: Optional[float]) -> Optional[float]:
    if raw is None:
        return None
    value = float(raw)
    return value if np.isfinite(value) and value > 0 else None


def _quantity_for_entry(entry_price: float, *, quantity: int, notional_inr: Optional[float]) -> int:
    if notional_inr is not None:
        price = abs(float(entry_price))
        if not np.isfinite(price) or price <= 0:
            return max(1, int(quantity))
        return max(1, int(float(notional_inr) // price))
    return max(1, int(quantity))


def _load_report(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"model report not found: {path}")
    with path.open("rb") as fh:
        return pickle.load(fh)


def _proximity_files(root: Path, horizon: int, split: str) -> List[Path]:
    return sorted((root / "proximity").glob(f"horizon={int(horizon)}/symbol=*/split={split}/fold=*.parquet"))


def _parse_symbol_fold(path: Path) -> Tuple[str, int]:
    text = str(path)
    sym_match = re.search(r"symbol=([^/]+)", text)
    fold_match = re.search(r"fold=(\d+)\.parquet$", text)
    if not sym_match or not fold_match:
        raise ValueError(f"could not parse symbol/fold from path: {path}")
    return sym_match.group(1), int(fold_match.group(1))


def _load_direction_frame(root: Path, symbol: str, split: str, fold: int) -> pd.DataFrame:
    path = root / f"direction/symbol={symbol}/split={split}/fold={fold}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing direction shard: {path}")
    df = pd.read_parquet(path)
    keep = ["ts", "bar_idx", "direction_label"]
    missing = set(keep).difference(df.columns)
    if missing:
        raise ValueError(f"direction shard missing {sorted(missing)}: {path}")
    out = df[keep].copy()
    out["ts"] = out["ts"].astype(str)
    out["bar_idx"] = pd.to_numeric(out["bar_idx"], errors="coerce").astype("Int64")
    return out.dropna(subset=["bar_idx"]).drop_duplicates(["ts", "bar_idx"])


def _load_pool_lookup(root: Path, symbol: str) -> pd.DataFrame:
    frames = []
    for split in ("train", "oos"):
        path = root / f"pools/symbol={symbol}/{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path).copy()
            df["_source_split"] = split
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no pool metadata parquet for {symbol} under {root}")
    pools = pd.concat(frames, ignore_index=True)
    pools = pools.reset_index(drop=True)
    pools["proximity_pool_idx"] = np.arange(len(pools), dtype=int)
    keep = [
        "proximity_pool_idx",
        "price_low",
        "price_high",
        "mid",
        "side",
        "score",
        "tf_bucket",
        "headline_factor",
        "n_contributors",
        "outcome",
        "mfe_after_touch_atr",
        "mae_after_touch_atr",
    ]
    return pools[keep]


def _to_ist(ts: pd.Timestamp, data_timestamps_utc: bool) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts + pd.Timedelta(minutes=330) if data_timestamps_utc else ts


def _from_ist(ts: pd.Timestamp, data_timestamps_utc: bool) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts - pd.Timedelta(minutes=330) if data_timestamps_utc else ts


def _session_phase_ist(ts: pd.Timestamp, data_timestamps_utc: bool) -> str:
    t = _to_ist(ts, data_timestamps_utc).time()
    if pd.Timestamp("09:15").time() <= t < pd.Timestamp("10:00").time():
        return "open"
    if pd.Timestamp("14:30").time() <= t <= pd.Timestamp("15:30").time():
        return "close"
    return "mid"


def _time_bucket_ist(ts: pd.Timestamp, data_timestamps_utc: bool) -> str:
    t = _to_ist(ts, data_timestamps_utc).time()
    if pd.Timestamp("09:15").time() <= t < pd.Timestamp("11:30").time():
        return "morning"
    if pd.Timestamp("11:30").time() <= t < pd.Timestamp("13:30").time():
        return "midday"
    if pd.Timestamp("13:30").time() <= t < pd.Timestamp("14:45").time():
        return "afternoon"
    if pd.Timestamp("14:45").time() <= t <= pd.Timestamp("15:10").time():
        return "closing"
    return "off_session"


def _effective_hold_bars(
    entry_ts: pd.Timestamp,
    requested_max_hold_bars: int,
    *,
    data_timestamps_utc: bool,
    session_exit_time: str,
) -> int:
    entry_ts = pd.Timestamp(entry_ts)
    entry_ist = _to_ist(entry_ts, data_timestamps_utc)
    hh, mm = [int(x) for x in session_exit_time.split(":", 1)]
    cutoff_ist = entry_ist.normalize() + pd.Timedelta(hours=hh, minutes=mm)
    cutoff_data = _from_ist(cutoff_ist, data_timestamps_utc)
    bars_to_cutoff = int(math.floor((cutoff_data - entry_ts).total_seconds() / 300.0))
    return max(0, min(int(requested_max_hold_bars), bars_to_cutoff))


def _safe_auc(y: Sequence[int], p: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=int)
    pp = np.asarray(p, dtype=float)
    if len(yy) == 0 or len(np.unique(yy)) < 2:
        return float("nan")
    return float(roc_auc_score(yy, pp))


def _profit_factor(r: pd.Series) -> float:
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    if neg == 0.0:
        return math.inf if pos > 0 else 0.0
    return pos / neg


def _max_drawdown(r: pd.Series) -> float:
    if r.empty:
        return float("nan")
    equity = r.cumsum()
    dd = equity - equity.cummax()
    return float(dd.min())


def _trade_sharpe(r: pd.Series) -> float:
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float(r.mean() / sd)


def _deflated_sharpe(r: pd.Series, n_trials: int) -> float:
    if len(r) < 3:
        return float("nan")
    sr = _trade_sharpe(r)
    if not np.isfinite(sr):
        return float("nan")
    n_obs = len(r)
    sk = float(skew(r, bias=False)) if n_obs >= 3 else 0.0
    ku = float(kurtosis(r, fisher=False, bias=False)) if n_obs >= 4 else 3.0
    trials = max(int(n_trials), 2)
    emc = 0.5772156649
    expected_max_sr = (
        (1.0 - emc) * norm.ppf(1.0 - 1.0 / trials)
        + emc * norm.ppf(1.0 - 1.0 / (trials * math.e))
    )
    variance = (1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr * sr) / max(n_obs - 1, 1)
    if variance <= 0 or not np.isfinite(variance):
        return float("nan")
    return float(norm.cdf((sr - expected_max_sr) / math.sqrt(variance)))


def collect_candidates(
    *,
    report,
    feature_store: Path,
    split: str,
    horizon: int,
    symbols: Optional[set[str]],
    min_p_touch_floor: float,
    min_p_direction_floor: float,
    distance_union: Tuple[float, float],
    max_files: Optional[int],
    max_candidates: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    direction_model = getattr(report, "unified_direction", None)
    proximity_model = (getattr(report, "unified_proximity", {}) or {}).get(int(horizon))
    if direction_model is None:
        raise ValueError("saved report has no unified_direction model")
    if proximity_model is None:
        raise ValueError(f"saved report has no unified_proximity model for h={horizon}")

    model_cols = sorted(set(
        list(getattr(direction_model, "feature_names", []))
        + list(getattr(proximity_model, "feature_names", []))
        + [
            "symbol", "sector", "split", "fold", "ts", "bar_idx", "pool_idx",
            "touch_label", "distance_atr", "side_above", "pool_quality",
        ]
    ))
    paths = _proximity_files(feature_store, horizon, split)
    if symbols is not None:
        paths = [p for p in paths if _parse_symbol_fold(p)[0] in symbols]
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"no proximity shards for h={horizon} split={split}")

    direction_cache: Dict[Tuple[str, int], pd.DataFrame] = {}
    pool_cache: Dict[str, pd.DataFrame] = {}
    candidate_frames: List[pd.DataFrame] = []
    preflight_parts: Dict[str, Dict[str, List[np.ndarray]]] = {}
    joined_rows = 0
    dropped_no_direction = 0

    def append_preflight(label: str, y: np.ndarray, p: np.ndarray) -> None:
        if label not in preflight_parts:
            preflight_parts[label] = {"y": [], "p": []}
        preflight_parts[label]["y"].append(np.asarray(y, dtype=int))
        preflight_parts[label]["p"].append(np.asarray(p, dtype=float))

    for path in paths:
        symbol, fold = _parse_symbol_fold(path)
        frame = pd.read_parquet(path, columns=model_cols)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["ts"] = frame["ts"].astype(str)
        frame["bar_idx"] = pd.to_numeric(frame["bar_idx"], errors="coerce").astype("Int64")

        dkey = (symbol, fold)
        if dkey not in direction_cache:
            direction_cache[dkey] = _load_direction_frame(feature_store, symbol, split, fold)
        merged = frame.merge(direction_cache[dkey], on=["ts", "bar_idx"], how="left")
        missing_dir = merged["direction_label"].isna()
        dropped_no_direction += int(missing_dir.sum())
        merged = merged[~missing_dir].copy()
        if merged.empty:
            continue
        joined_rows += int(len(merged))

        p_touch = proximity_model.predict_frame(merged)
        p_up = direction_model.predict_batch(merged)
        side_above = merged["side_above"].astype(float).to_numpy() >= 0.5
        y_up = merged["direction_label"].astype(int).to_numpy()
        y_to_pool = np.where(side_above, y_up, 1 - y_up)
        p_to_pool = np.where(side_above, p_up, 1.0 - p_up)
        distance = merged["distance_atr"].astype(float).to_numpy()

        for lo, hi in [(2.0, 5.0), (3.0, 8.0), (5.0, 10.0)]:
            mask = (distance >= lo) & (distance < hi)
            if mask.any():
                append_preflight(f"{lo:g}-{hi:g} ATR", y_to_pool[mask], p_to_pool[mask])

        keep = (
            (distance >= distance_union[0])
            & (distance < distance_union[1])
            & (p_touch >= min_p_touch_floor)
            & (p_to_pool >= min_p_direction_floor)
        )
        if not keep.any():
            continue
        kept = merged.loc[keep].copy()
        kept["p_touch"] = p_touch[keep]
        kept["p_up"] = p_up[keep]
        kept["p_direction_to_pool"] = p_to_pool[keep]
        kept["direction_target_to_pool"] = y_to_pool[keep]
        kept["pool_side"] = np.where(side_above[keep], "above", "below")

        if symbol not in pool_cache:
            pool_cache[symbol] = _load_pool_lookup(feature_store, symbol)
        kept = kept.merge(
            pool_cache[symbol],
            left_on="pool_idx",
            right_on="proximity_pool_idx",
            how="left",
            validate="many_to_one",
        )
        missing_pools = int(kept["price_low"].isna().sum())
        if missing_pools:
            raise ValueError(f"{symbol} fold={fold}: missing pool metadata for {missing_pools} rows")

        candidate_frames.append(kept)
        if max_candidates is not None:
            current = sum(len(x) for x in candidate_frames)
            if current >= max_candidates:
                break

    if candidate_frames:
        candidates = pd.concat(candidate_frames, ignore_index=True)
        if max_candidates is not None and len(candidates) > max_candidates:
            candidates = candidates.head(max_candidates).copy()
    else:
        candidates = pd.DataFrame()

    preflight_rows = []
    for label, parts in preflight_parts.items():
        y = np.concatenate(parts["y"]) if parts["y"] else np.array([], dtype=int)
        p = np.concatenate(parts["p"]) if parts["p"] else np.array([], dtype=float)
        preflight_rows.append({
            "distance_bucket": label,
            "n": int(len(y)),
            "base_rate_to_pool": float(y.mean()) if len(y) else np.nan,
            "mean_p_to_pool": float(p.mean()) if len(p) else np.nan,
            "auc": _safe_auc(y, p),
        })
    preflight = pd.DataFrame(preflight_rows).sort_values("distance_bucket").reset_index(drop=True)
    meta = {
        "proximity_files": len(paths),
        "joined_rows": int(joined_rows),
        "dropped_no_direction": int(dropped_no_direction),
        "candidate_rows": int(len(candidates)),
    }
    return candidates, preflight, meta


def _simulate_one(
    row: pd.Series,
    *,
    base_df: pd.DataFrame,
    atr_values: pd.Series,
    intrabar_1m: Optional[pd.DataFrame],
    geometry: Geometry,
    data_timestamps_utc: bool,
    session_exit_time: str,
    use_1m_resolution: bool,
    slippage_model: str,
    base_slippage_bps: float,
    exchange: str,
    quantity: int,
    notional_inr: Optional[float],
) -> Optional[Dict]:
    idx = pd.DatetimeIndex(base_df.index)
    decision_idx = int(row["bar_idx"])
    entry_idx = decision_idx + 1
    if decision_idx < 0 or entry_idx >= len(base_df):
        return None
    entry_ts = pd.Timestamp(idx[entry_idx])
    effective_hold = _effective_hold_bars(
        entry_ts,
        geometry.max_hold_bars,
        data_timestamps_utc=data_timestamps_utc,
        session_exit_time=session_exit_time,
    )
    if effective_hold <= 0:
        return None
    deadline_idx = min(entry_idx + effective_hold, len(base_df) - 1)
    if deadline_idx <= entry_idx:
        return None

    entry_ref = float(base_df["open"].iloc[entry_idx])
    atr_at_decision = float(atr_values.iloc[decision_idx])
    if not np.isfinite(atr_at_decision) or atr_at_decision <= 0:
        return None

    price_low = float(row["price_low"])
    price_high = float(row["price_high"])
    pool_side = str(row["pool_side"])
    if pool_side == "above":
        direction = "UP"
        direction_sign = 1
        target = entry_ref + geometry.target_fraction * (price_low - entry_ref)
        stop = entry_ref - geometry.stop_atr_mult * atr_at_decision
        if not (stop < entry_ref < target):
            return None
    else:
        direction = "DOWN"
        direction_sign = -1
        target = entry_ref - geometry.target_fraction * (entry_ref - price_high)
        stop = entry_ref + geometry.stop_atr_mult * atr_at_decision
        if not (target < entry_ref < stop):
            return None

    deadline_ts = pd.Timestamp(idx[deadline_idx])
    resolution_source = "5m"
    exit_ref = float(base_df["close"].iloc[deadline_idx])
    exit_reason = "time_exit"
    exit_ts = deadline_ts
    if use_1m_resolution and intrabar_1m is not None and not intrabar_1m.empty:
        resolved = resolve_intrabar_path(
            entry_ts,
            entry_ref,
            stop,
            target,
            str(row["symbol"]),
            direction,
            intrabar_1m,
            exit_deadline=deadline_ts,
        )
        resolution_source = "1m_no_hit"
        if resolved["resolution"] in ("stop_hit", "target_hit"):
            exit_ref = float(resolved["price"])
            exit_reason = "stop" if resolved["resolution"] == "stop_hit" else "target"
            exit_ts = pd.Timestamp(resolved["timestamp"])
            resolution_source = "1m"
    else:
        for j in range(entry_idx, deadline_idx + 1):
            low = float(base_df["low"].iloc[j])
            high = float(base_df["high"].iloc[j])
            if direction == "UP":
                stop_hit = low <= stop
                target_hit = high >= target
            else:
                stop_hit = high >= stop
                target_hit = low <= target
            if stop_hit:
                exit_ref = float(stop)
                exit_reason = "stop"
                exit_ts = pd.Timestamp(idx[j])
                break
            if target_hit:
                exit_ref = float(target)
                exit_reason = "target"
                exit_ts = pd.Timestamp(idx[j])
                break

    distance_atr = float(row["distance_atr"])
    vol_frac = atr_at_decision / max(abs(entry_ref), 1e-9)
    if slippage_model == "flat":
        entry_slip = exit_slip = max(float(base_slippage_bps), 0.0)
    else:
        entry_slip = compute_slippage_bps(
            vol_frac,
            distance_atr,
            _session_phase_ist(entry_ts, data_timestamps_utc),
            base_bps=base_slippage_bps,
        )
        exit_slip = compute_slippage_bps(
            vol_frac,
            distance_atr,
            _session_phase_ist(exit_ts, data_timestamps_utc),
            base_bps=base_slippage_bps,
        )

    entry_exec = _slipped_price(entry_ref, direction, "entry", entry_slip)
    exit_exec = _slipped_price(exit_ref, direction, "exit", exit_slip)
    trade_quantity = _quantity_for_entry(
        entry_exec,
        quantity=quantity,
        notional_inr=notional_inr,
    )
    buy_price, sell_price = _buy_sell_prices(direction, entry_exec, exit_exec)
    costs = compute_zerodha_intraday_costs(buy_price, sell_price, trade_quantity, exchange=exchange)
    if direction == "UP":
        gross_ref = (exit_ref - entry_ref) * trade_quantity
        slip_cost = max(0.0, (entry_exec - entry_ref) + (exit_ref - exit_exec)) * trade_quantity
        risk = (entry_exec - stop) * trade_quantity
    else:
        gross_ref = (entry_ref - exit_ref) * trade_quantity
        slip_cost = max(0.0, (entry_ref - entry_exec) + (exit_exec - exit_ref)) * trade_quantity
        risk = (stop - entry_exec) * trade_quantity
    total_cost = float(costs["total_cost_inr"] + slip_cost)
    net = float(gross_ref - total_cost)
    risk = float(max(risk, 1e-9))
    bars_held = max(0, int(np.searchsorted(idx.values, np.datetime64(exit_ts), side="right") - 1 - entry_idx))

    return {
        "symbol": row["symbol"],
        "sector": row["sector"],
        "fold": int(row["fold"]),
        "pool_idx": int(row["pool_idx"]),
        "bar_idx": int(row["bar_idx"]),
        "decision_at": str(pd.Timestamp(row["ts"])),
        "entry_at": str(entry_ts),
        "exit_at": str(exit_ts),
        "time_bucket": _time_bucket_ist(entry_ts, data_timestamps_utc),
        "direction": direction,
        "direction_sign": direction_sign,
        "pool_side": pool_side,
        "factor": row.get("headline_factor", ""),
        "tf_bucket": row.get("tf_bucket", ""),
        "target_fraction": float(geometry.target_fraction),
        "stop_atr_mult": float(geometry.stop_atr_mult),
        "requested_max_hold_bars": int(geometry.max_hold_bars),
        "effective_max_hold_bars": int(effective_hold),
        "bars_held": int(bars_held),
        "entry_reference": float(entry_ref),
        "exit_reference": float(exit_ref),
        "entry": float(entry_exec),
        "exit": float(exit_exec),
        "quantity": int(trade_quantity),
        "entry_notional_inr": float(abs(entry_exec) * trade_quantity),
        "target_notional_inr": float(notional_inr) if notional_inr is not None else np.nan,
        "stop": float(stop),
        "target": float(target),
        "exit_reason": exit_reason,
        "resolution_source": resolution_source,
        "distance_atr": distance_atr,
        "p_touch": float(row["p_touch"]),
        "p_up": float(row["p_up"]),
        "p_direction_to_pool": float(row["p_direction_to_pool"]),
        "pool_quality": float(row["pool_quality"]),
        "touch_label": int(row["touch_label"]) if not pd.isna(row.get("touch_label")) else np.nan,
        "direction_target_to_pool": int(row["direction_target_to_pool"]),
        "gross_pnl": float(gross_ref),
        "statutory_cost": float(costs["total_cost_inr"]),
        "slippage_cost": float(slip_cost),
        "total_cost": total_cost,
        "risk_inr": risk,
        "net_pnl": net,
        "net_r": float(net / risk),
        "directional_net_r": float(abs(net / risk) * direction_sign),
        **costs,
    }


def simulate_trades(
    candidates: pd.DataFrame,
    *,
    report,
    raw_1m_dir: Optional[Path],
    geometries: Sequence[Geometry],
    out_dir: Path,
    data_timestamps_utc: bool,
    session_exit_time: str,
    use_1m_resolution: bool,
    slippage_model: str,
    base_slippage_bps: float,
    exchange: str,
    quantity: int,
    notional_inr: Optional[float],
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = out_dir / "pretouch_trade_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    base_cache: Dict[str, pd.DataFrame] = {}
    atr_cache: Dict[str, pd.Series] = {}
    intrabar_cache: Dict[str, Optional[pd.DataFrame]] = {}
    frames: List[pd.DataFrame] = []

    for symbol, sdf in candidates.groupby("symbol", sort=True):
        asset = getattr(report, "assets", {}).get(symbol)
        if asset is None:
            raise ValueError(f"model report has no asset payload for {symbol}")
        base = getattr(asset, "base_df", None)
        if base is None or base.empty:
            raise ValueError(f"empty base_df for {symbol}")
        base_cache[symbol] = base
        cfg = getattr(asset, "final_cfg", None)
        atr_period = getattr(getattr(cfg, "detect", None), "atr_period", 14)
        atr_cache[symbol] = atr(base, atr_period).bfill()
        intrabar_cache[symbol] = _load_symbol_1m(symbol, raw_1m_dir) if use_1m_resolution else None

        for geometry in geometries:
            rows = []
            for _, row in sdf.iterrows():
                trade = _simulate_one(
                    row,
                    base_df=base_cache[symbol],
                    atr_values=atr_cache[symbol],
                    intrabar_1m=intrabar_cache[symbol],
                    geometry=geometry,
                    data_timestamps_utc=data_timestamps_utc,
                    session_exit_time=session_exit_time,
                    use_1m_resolution=use_1m_resolution,
                    slippage_model=slippage_model,
                    base_slippage_bps=base_slippage_bps,
                    exchange=exchange,
                    quantity=quantity,
                    notional_inr=notional_inr,
                )
                if trade is not None:
                    rows.append(trade)
            if rows:
                chunk = pd.DataFrame(rows)
                chunk_path = (
                    chunk_dir
                    / f"symbol={symbol}_tf={geometry.target_fraction:g}_sl={geometry.stop_atr_mult:g}_hold={geometry.max_hold_bars}.parquet"
                )
                chunk.to_parquet(chunk_path, index=False)
                frames.append(chunk)

    if not frames:
        return pd.DataFrame()
    trades = pd.concat(frames, ignore_index=True)
    trades.to_parquet(out_dir / "pretouch_sweep_trades.parquet", index=False)
    return trades


def _sizing_matches(trades: pd.DataFrame, *, quantity: int, notional_inr: Optional[float]) -> bool:
    if trades.empty:
        return True
    if notional_inr is not None:
        if "target_notional_inr" not in trades.columns:
            return False
        observed = pd.to_numeric(trades["target_notional_inr"], errors="coerce").dropna()
        if observed.empty:
            return False
        return bool(np.allclose(observed.to_numpy(dtype=float), float(notional_inr), rtol=0.0, atol=0.01))
    if "target_notional_inr" in trades.columns:
        observed = pd.to_numeric(trades["target_notional_inr"], errors="coerce").dropna()
        if not observed.empty:
            return False
    if "quantity" not in trades.columns:
        return False
    observed_qty = pd.to_numeric(trades["quantity"], errors="coerce").dropna()
    return bool(not observed_qty.empty and (observed_qty.astype(int) == int(quantity)).all())


def load_existing_trades(out_dir: Path, *, quantity: int, notional_inr: Optional[float]) -> Optional[pd.DataFrame]:
    trade_path = out_dir / "pretouch_sweep_trades.parquet"
    if trade_path.exists():
        trades = pd.read_parquet(trade_path)
        return trades if _sizing_matches(trades, quantity=quantity, notional_inr=notional_inr) else None
    chunk_dir = out_dir / "pretouch_trade_chunks"
    if not chunk_dir.exists():
        return None
    paths = sorted(chunk_dir.glob("*.parquet"))
    if not paths:
        return None
    trades = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if not _sizing_matches(trades, quantity=quantity, notional_inr=notional_inr):
        return None
    trades.to_parquet(trade_path, index=False)
    return trades


def _cohort_mask(df: pd.DataFrame, q_col: str, fraction: float) -> pd.Series:
    if fraction >= 1.0:
        return pd.Series(True, index=df.index)
    return pd.Series(exact_top_mask(df.reset_index(drop=True), q_col, fraction), index=df.index)


def _summarize_subset(
    df: pd.DataFrame,
    *,
    q_col: str,
    q_cohort: str,
    q_fraction: float,
    gate: Gate,
    geometry: Geometry,
    n_trials: int,
    min_trades: int,
    min_direction_trades: int,
    min_sector_trades: int,
    min_sectors_with_trades: int,
) -> Dict:
    cdf = df.sort_values("entry_at").copy()
    r = pd.to_numeric(cdf["net_r"], errors="coerce").dropna()
    sectors = cdf["sector"].value_counts()
    directions = cdf["direction"].value_counts()
    long_r = pd.to_numeric(cdf.loc[cdf["direction"] == "UP", "net_r"], errors="coerce").dropna()
    short_r = pd.to_numeric(cdf.loc[cdf["direction"] == "DOWN", "net_r"], errors="coerce").dropna()
    by_time = cdf.groupby("time_bucket")["net_r"].mean().to_dict()
    if len(cdf):
        ts = pd.to_datetime(cdf["entry_at"])
        midpoint = ts.min() + (ts.max() - ts.min()) / 2
        first = cdf.loc[ts < midpoint, "net_r"]
        second = cdf.loc[ts >= midpoint, "net_r"]
    else:
        first = second = pd.Series(dtype=float)

    sectors_with_min = int((sectors >= min_sector_trades).sum())
    long_n = int(len(long_r))
    short_n = int(len(short_r))
    passes = (
        len(r) >= min_trades
        and sectors_with_min >= min_sectors_with_trades
        and long_n >= min_direction_trades
        and short_n >= min_direction_trades
    )
    temporal_gap = float(abs(first.mean() - second.mean())) if len(first) and len(second) else np.nan
    cost_125 = r - ((cdf["total_cost"] * 0.25) / cdf["risk_inr"])
    cost_150 = r - ((cdf["total_cost"] * 0.50) / cdf["risk_inr"])

    return {
        "gate": gate.label,
        "min_p_touch": gate.min_p_touch,
        "min_p_direction": gate.min_p_direction,
        "distance_low": gate.distance_low,
        "distance_high": gate.distance_high,
        "target_fraction": geometry.target_fraction,
        "stop_atr_mult": geometry.stop_atr_mult,
        "max_hold_bars": geometry.max_hold_bars,
        "q_cohort": q_cohort,
        "q_fraction": q_fraction,
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "mean_r": float(r.mean()) if len(r) else np.nan,
        "median_r": float(r.median()) if len(r) else np.nan,
        "total_r": float(r.sum()) if len(r) else 0.0,
        "profit_factor": float(_profit_factor(r)) if len(r) else np.nan,
        "max_drawdown_r": float(_max_drawdown(r)) if len(r) else np.nan,
        "trade_sharpe": float(_trade_sharpe(r)) if len(r) else np.nan,
        "dsr": float(_deflated_sharpe(r, n_trials=n_trials)) if len(r) else np.nan,
        "mean_r_cost_1_25x": float(cost_125.mean()) if len(cost_125) else np.nan,
        "mean_r_cost_1_50x": float(cost_150.mean()) if len(cost_150) else np.nan,
        "long_n": long_n,
        "short_n": short_n,
        "long_mean_r": float(long_r.mean()) if len(long_r) else np.nan,
        "short_mean_r": float(short_r.mean()) if len(short_r) else np.nan,
        "long_pf": float(_profit_factor(long_r)) if len(long_r) else np.nan,
        "short_pf": float(_profit_factor(short_r)) if len(short_r) else np.nan,
        "sectors_with_min_trades": sectors_with_min,
        "top_sector": str(sectors.index[0]) if len(sectors) else "",
        "top_sector_trades": int(sectors.iloc[0]) if len(sectors) else 0,
        "morning_mean_r": float(by_time.get("morning", np.nan)),
        "midday_mean_r": float(by_time.get("midday", np.nan)),
        "afternoon_mean_r": float(by_time.get("afternoon", np.nan)),
        "closing_mean_r": float(by_time.get("closing", np.nan)),
        "first_half_mean_r": float(first.mean()) if len(first) else np.nan,
        "second_half_mean_r": float(second.mean()) if len(second) else np.nan,
        "temporal_gap_r": temporal_gap,
        "temporal_stable": bool(np.isfinite(temporal_gap) and temporal_gap < 0.15),
        "avg_bars_held": float(cdf["bars_held"].mean()) if len(cdf) else np.nan,
        "target_exit_rate": float((cdf["exit_reason"] == "target").mean()) if len(cdf) else np.nan,
        "stop_exit_rate": float((cdf["exit_reason"] == "stop").mean()) if len(cdf) else np.nan,
        "time_exit_rate": float((cdf["exit_reason"] == "time_exit").mean()) if len(cdf) else np.nan,
        "mean_p_touch": float(cdf["p_touch"].mean()) if len(cdf) else np.nan,
        "mean_p_direction": float(cdf["p_direction_to_pool"].mean()) if len(cdf) else np.nan,
        "mean_q": float(cdf[q_col].mean()) if len(cdf) else np.nan,
        "passes_min_filters": bool(passes),
    }


def summarize_cells(
    trades: pd.DataFrame,
    *,
    gates: Sequence[Gate],
    geometries: Sequence[Geometry],
    q_col: str,
    n_trials: int,
    min_trades: int,
    min_direction_trades: int,
    min_sector_trades: int,
    min_sectors_with_trades: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    if trades.empty:
        return pd.DataFrame()
    for geometry in geometries:
        gdf = trades[
            (trades["target_fraction"] == geometry.target_fraction)
            & (trades["stop_atr_mult"] == geometry.stop_atr_mult)
            & (trades["requested_max_hold_bars"] == geometry.max_hold_bars)
        ]
        if gdf.empty:
            continue
        for gate in gates:
            cdf = gdf[
                (gdf["p_touch"] >= gate.min_p_touch)
                & (gdf["p_direction_to_pool"] >= gate.min_p_direction)
                & (gdf["distance_atr"] >= gate.distance_low)
                & (gdf["distance_atr"] < gate.distance_high)
            ]
            if cdf.empty:
                continue
            for q_label, q_fraction in Q_COHORTS:
                mask = _cohort_mask(cdf, q_col, q_fraction)
                qdf = cdf.loc[mask]
                rows.append(_summarize_subset(
                    qdf,
                    q_col=q_col,
                    q_cohort=q_label,
                    q_fraction=q_fraction,
                    gate=gate,
                    geometry=geometry,
                    n_trials=n_trials,
                    min_trades=min_trades,
                    min_direction_trades=min_direction_trades,
                    min_sector_trades=min_sector_trades,
                    min_sectors_with_trades=min_sectors_with_trades,
                ))
    return pd.DataFrame(rows)


def _fmt_pf(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    if math.isinf(float(x)):
        return "inf"
    return fnum(x, 3)


def _rows_for_report(df: pd.DataFrame, limit: int) -> List[Dict]:
    rows = []
    for row in df.head(limit).to_dict(orient="records"):
        rows.append({
            "gate": row["gate"],
            "geom": f"tf={row['target_fraction']:g}, sl={row['stop_atr_mult']:g}, h={int(row['max_hold_bars'])}",
            "Q": row["q_cohort"],
            "n": int(row["trades"]),
            "win": pct(row["win_rate"], 1),
            "mean_R": fnum(row["mean_r"], 3),
            "PF": _fmt_pf(row["profit_factor"]),
            "DSR": pct(row["dsr"], 1),
            "long/short": f"{int(row['long_n'])}/{int(row['short_n'])}",
            "sectors": int(row["sectors_with_min_trades"]),
            "stable": str(bool(row["temporal_stable"])),
        })
    return rows


def _preflight_rows(df: pd.DataFrame) -> List[Dict]:
    rows = []
    for row in df.to_dict(orient="records"):
        rows.append({
            "distance": row["distance_bucket"],
            "n": int(row["n"]),
            "base": pct(row["base_rate_to_pool"], 1),
            "mean_p": pct(row["mean_p_to_pool"], 1),
            "AUC": fnum(row["auc"], 3),
        })
    return rows


def decide(summary: pd.DataFrame, limited: bool) -> Tuple[str, str]:
    if limited:
        return "SMOKE_ONLY", "Run was limited by --max-files/--max-candidates/--max-cells; no viability verdict."
    if summary.empty:
        return "NOT_VIABLE", "No cells produced executable trades."
    passing = summary[summary["passes_min_filters"]].copy()
    positive = passing[passing["mean_r"] > 0.0]
    viable = positive[
        (positive["mean_r"] > 0.05)
        & (positive["dsr"] > 0.95)
        & (positive["trades"] > 500)
        & (positive["sectors_with_min_trades"] >= 3)
        & (positive["temporal_stable"])
    ]
    if not viable.empty:
        best = viable.sort_values(["mean_r", "dsr"], ascending=False).iloc[0]
        return "VIABLE", f"Best robust cell has mean_R={best['mean_r']:+.3f}, DSR={best['dsr']:.1%}."
    if not positive.empty:
        best = positive.sort_values("mean_r", ascending=False).iloc[0]
        return "MARGINAL", f"Positive cells exist, but best deployable evidence fails robustness filters: mean_R={best['mean_r']:+.3f}."
    best = summary.sort_values("mean_r", ascending=False).iloc[0]
    return "NOT_VIABLE", f"No passing cell is positive. Best observed mean_R={best['mean_r']:+.3f}."


def build_report(
    *,
    summary: pd.DataFrame,
    preflight: pd.DataFrame,
    meta: Dict,
    decision: str,
    reason: str,
    args: argparse.Namespace,
) -> str:
    passing = summary[summary["passes_min_filters"]].copy() if not summary.empty else pd.DataFrame()
    by_mean = passing.sort_values("mean_r", ascending=False) if not passing.empty else summary.sort_values("mean_r", ascending=False)
    by_dsr = passing.sort_values("dsr", ascending=False) if not passing.empty else summary.sort_values("dsr", ascending=False)
    sizing = (
        f"target_notional_inr={float(args.notional_inr):g}"
        if _target_notional(args.notional_inr) is not None else
        f"fixed quantity={int(args.quantity)}"
    )
    lines = [
        "# Phase 4 Track A Pre-Touch Directional Sweep",
        "",
        "## Verdict",
        "",
        f"- Decision: `{decision}`",
        f"- {reason}",
        f"- Cumulative DSR trials used: `{args.dsr_trials}`",
        f"- Intraday session exit enforced at `{args.session_exit_time}` IST.",
        "",
        "## Pre-Flight Direction Stability",
        "",
        markdown_table(_preflight_rows(preflight), ["distance", "n", "base", "mean_p", "AUC"]),
        "",
        "## Inputs",
        "",
        f"- Model report: `{args.model_report}`",
        f"- Feature store: `{args.feature_store}`",
        f"- Raw 1m dir: `{args.raw_1m_dir or ''}`",
        f"- Sizing: `{sizing}`",
        f"- Candidate rows after floor gates: `{meta.get('candidate_rows', 0):,}`",
        f"- Joined proximity rows: `{meta.get('joined_rows', 0):,}`",
        f"- Dropped rows without direction labels: `{meta.get('dropped_no_direction', 0):,}`",
        "",
        "## Top Cells By Mean R",
        "",
        markdown_table(
            _rows_for_report(by_mean, 20),
            ["gate", "geom", "Q", "n", "win", "mean_R", "PF", "DSR", "long/short", "sectors", "stable"],
        ),
        "",
        "## Top Cells By DSR",
        "",
        markdown_table(
            _rows_for_report(by_dsr, 10),
            ["gate", "geom", "Q", "n", "win", "mean_R", "PF", "DSR", "long/short", "sectors", "stable"],
        ),
        "",
        "## Interpretation",
        "",
        "- Q is applied post-hoc within each cell, not as a separate sweep dimension.",
        "- Cells must clear minimum total trades, long/short trades, and sector breadth before they count as passing.",
        "- `mean_R_cost_1_25x` and `mean_R_cost_1_50x` are included in the CSV for slippage/cost stress checks.",
        "- This report is research-only and does not change live/predict gates.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-report", default=str(DEFAULT_MODEL_REPORT))
    parser.add_argument("--feature-store", default=str(DEFAULT_FEATURE_STORE))
    parser.add_argument("--raw-1m-dir", default="")
    parser.add_argument("--split", default="oos")
    parser.add_argument("--horizon", type=int, default=78)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--min-p-touch", default="0.75,0.80,0.85")
    parser.add_argument("--min-p-direction", default="0.55,0.60,0.65")
    parser.add_argument("--distance-ranges", default="3-8")
    parser.add_argument("--target-fractions", default="0.6,0.8,1.0")
    parser.add_argument("--stop-atr-mults", default="1.0,1.5,2.0")
    parser.add_argument("--max-hold-bars", default="12,24,36,60")
    parser.add_argument("--session-exit-time", default="15:10")
    parser.add_argument("--data-timestamps-local-ist", action="store_true")
    parser.add_argument("--use-1m-resolution", action="store_true")
    parser.add_argument("--slippage-model", choices=["flat", "state_dependent"], default="state_dependent")
    parser.add_argument("--base-slippage-bps", type=float, default=2.0)
    parser.add_argument("--exchange", default="NSE")
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument(
        "--notional-inr",
        type=float,
        default=100_000.0,
        help="Target per-trade notional. Set 0 to use fixed --quantity instead.",
    )
    parser.add_argument("--dsr-trials", type=int, default=550)
    parser.add_argument("--min-trades", type=int, default=200)
    parser.add_argument("--min-direction-trades", type=int, default=50)
    parser.add_argument("--min-sector-trades", type=int, default=30)
    parser.add_argument("--min-sectors-with-trades", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=None, help="Smoke helper: only keep this many gate/geometry pairs before Q cohorts.")
    parser.add_argument("--reuse-trades", action="store_true", help="Reuse existing trade parquet/chunks in --out-dir and only rebuild summaries.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--summary-out", default=str(DEFAULT_CSV))
    parser.add_argument("--preflight-out", default=str(DEFAULT_PREFLIGHT))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    model_report = Path(args.model_report)
    feature_store = Path(args.feature_store)
    raw_1m_dir = Path(args.raw_1m_dir).expanduser() if args.raw_1m_dir else None
    out_dir = Path(args.out_dir)
    data_timestamps_utc = not bool(args.data_timestamps_local_ist)
    notional_inr = _target_notional(args.notional_inr)

    report = _load_report(model_report)
    min_touch = _parse_csv_floats(args.min_p_touch)
    min_dir = _parse_csv_floats(args.min_p_direction)
    distance_ranges = _parse_ranges(args.distance_ranges)
    target_fractions = _parse_csv_floats(args.target_fractions)
    stop_mults = _parse_csv_floats(args.stop_atr_mults)
    holds = _parse_csv_ints(args.max_hold_bars)
    symbols = _parse_symbols(args.symbols)
    distance_union = (min(lo for lo, _ in distance_ranges), max(hi for _, hi in distance_ranges))

    candidates, preflight, meta = collect_candidates(
        report=report,
        feature_store=feature_store,
        split=args.split,
        horizon=args.horizon,
        symbols=symbols,
        min_p_touch_floor=min(min_touch),
        min_p_direction_floor=min(min_dir),
        distance_union=distance_union,
        max_files=args.max_files,
        max_candidates=args.max_candidates,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(out_dir / "pretouch_candidates.parquet", index=False)

    geometries = [
        Geometry(target_fraction=tf, stop_atr_mult=sl, max_hold_bars=hold)
        for tf, sl, hold in itertools.product(target_fractions, stop_mults, holds)
    ]
    gates = [
        Gate(min_p_touch=t, min_p_direction=d, distance_low=lo, distance_high=hi)
        for t, d, (lo, hi) in itertools.product(min_touch, min_dir, distance_ranges)
    ]
    if args.max_cells is not None:
        pairs = list(itertools.product(gates, geometries))[: int(args.max_cells)]
        gates = sorted({pair[0] for pair in pairs}, key=lambda g: g.label)
        geometries = sorted(
            {pair[1] for pair in pairs},
            key=lambda g: (g.target_fraction, g.stop_atr_mult, g.max_hold_bars),
        )

    trades = (
        load_existing_trades(out_dir, quantity=args.quantity, notional_inr=notional_inr)
        if args.reuse_trades else None
    )
    if trades is None:
        trades = simulate_trades(
            candidates,
            report=report,
            raw_1m_dir=raw_1m_dir,
            geometries=geometries,
            out_dir=out_dir,
            data_timestamps_utc=data_timestamps_utc,
            session_exit_time=args.session_exit_time,
            use_1m_resolution=bool(args.use_1m_resolution),
            slippage_model=args.slippage_model,
            base_slippage_bps=args.base_slippage_bps,
            exchange=args.exchange,
            quantity=args.quantity,
            notional_inr=notional_inr,
        )

    summary = summarize_cells(
        trades,
        gates=gates,
        geometries=geometries,
        q_col="pool_quality",
        n_trials=args.dsr_trials,
        min_trades=args.min_trades,
        min_direction_trades=args.min_direction_trades,
        min_sector_trades=args.min_sector_trades,
        min_sectors_with_trades=args.min_sectors_with_trades,
    )
    limited = any(x is not None for x in (args.max_files, args.max_candidates, args.max_cells)) or symbols is not None
    decision, reason = decide(summary, limited=limited)

    summary_out = Path(args.summary_out)
    preflight_out = Path(args.preflight_out)
    report_out = Path(args.report_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    preflight_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    preflight.to_csv(preflight_out, index=False)
    report_out.write_text(
        build_report(summary=summary, preflight=preflight, meta=meta, decision=decision, reason=reason, args=args),
        encoding="utf-8",
    )
    print(f"wrote {summary_out}")
    print(f"wrote {preflight_out}")
    print(f"wrote {report_out}")
    print(f"decision: {decision} - {reason}")


if __name__ == "__main__":
    main()
