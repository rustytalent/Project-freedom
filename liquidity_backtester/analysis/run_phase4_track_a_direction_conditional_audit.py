"""Track A direction-conditional viability audit.

The proximity gate passed: P_touch is predictive at tradeable distances. This
script checks the next question: does the direction model still help inside
high-P_touch, tradeable-distance setup rows?

It loads saved OOS direction/proximity feature-store shards and saved models.
For each proximity setup row, direction is interpreted relative to the pool:

* pool above current price -> desired direction is UP/long
* pool below current price -> desired direction is DOWN/short

The audit does not retrain models.
"""

from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


DEFAULT_MODEL_REPORT = Path("output_models/core25_latest/multi_asset_report.pkl")
DEFAULT_FEATURE_STORE = Path("output_feature_store/core25_fresh_may25")
DEFAULT_REPORT = Path("reports/phase4_track_a_direction_conditional_audit.md")
DEFAULT_CSV = Path("reports/phase4_track_a_direction_conditional_audit.csv")
DEFAULT_SPLIT = "oos"
DEFAULT_PROXIMITY_HORIZON = 78
PRIMARY_SEGMENT = "p_touch>=0.85 & distance 2-8 ATR"


@dataclass
class SegmentMetrics:
    segment: str
    n: int
    positives: int
    base_rate: float
    mean_p_dir: float
    auc: float
    brier: float
    logloss: float
    accuracy_50: float
    top_quartile_accuracy: float
    long_rows: int
    short_rows: int
    long_auc: float
    short_auc: float


def _load_report(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"model report not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _proximity_files(root: Path, horizon: int, split: str) -> List[Path]:
    return sorted((root / "proximity").glob(f"horizon={int(horizon)}/symbol=*/split={split}/fold=*.parquet"))


def _parse_symbol_fold(path: Path) -> tuple[str, int]:
    text = str(path)
    sym_match = re.search(r"symbol=([^/]+)", text)
    fold_match = re.search(r"fold=(\d+)\.parquet$", text)
    if not sym_match or not fold_match:
        raise ValueError(f"could not parse symbol/fold from proximity path: {path}")
    return sym_match.group(1), int(fold_match.group(1))


def _load_direction_frame(root: Path, symbol: str, split: str, fold: int) -> pd.DataFrame:
    path = root / f"direction/symbol={symbol}/split={split}/fold={fold}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing direction shard: {path}")
    df = pd.read_parquet(path)
    keep = ["ts", "bar_idx", "direction_label"]
    missing = set(keep).difference(df.columns)
    if missing:
        raise ValueError(f"direction shard missing columns {sorted(missing)}: {path}")
    out = df[keep].copy()
    out["ts"] = out["ts"].astype(str)
    out["bar_idx"] = pd.to_numeric(out["bar_idx"], errors="coerce").astype("Int64")
    return out.dropna(subset=["bar_idx"]).drop_duplicates(["ts", "bar_idx"])


def _append(acc: Dict[str, Dict[str, List[np.ndarray]]], segment: str,
            y: np.ndarray, p: np.ndarray, side_above: np.ndarray) -> None:
    if segment not in acc:
        acc[segment] = {"y": [], "p": [], "side": []}
    acc[segment]["y"].append(np.asarray(y, dtype=int))
    acc[segment]["p"].append(np.asarray(p, dtype=float))
    acc[segment]["side"].append(np.asarray(side_above, dtype=bool))


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _metrics(segment: str, y: np.ndarray, p: np.ndarray, side_above: np.ndarray) -> SegmentMetrics:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    side_above = np.asarray(side_above, dtype=bool)
    n = int(len(y))
    positives = int(y.sum())
    base = float(y.mean()) if n else 0.0
    mean_p = float(p.mean()) if n else 0.0
    auc = _safe_auc(y, p)
    brier = float(brier_score_loss(y, p)) if n else float("nan")
    ll = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])) if n else float("nan")
    acc = float(((p >= 0.5).astype(int) == y).mean()) if n else float("nan")
    top_acc = float("nan")
    if n:
        conf = np.abs(p - 0.5)
        threshold = float(np.quantile(conf, 0.75))
        mask = conf >= threshold
        if mask.any():
            top_acc = float(((p[mask] >= 0.5).astype(int) == y[mask]).mean())
    long_mask = side_above
    short_mask = ~side_above
    return SegmentMetrics(
        segment=segment,
        n=n,
        positives=positives,
        base_rate=base,
        mean_p_dir=mean_p,
        auc=auc,
        brier=brier,
        logloss=ll,
        accuracy_50=acc,
        top_quartile_accuracy=top_acc,
        long_rows=int(long_mask.sum()),
        short_rows=int(short_mask.sum()),
        long_auc=_safe_auc(y[long_mask], p[long_mask]) if long_mask.any() else float("nan"),
        short_auc=_safe_auc(y[short_mask], p[short_mask]) if short_mask.any() else float("nan"),
    )


def collect_metrics(
    *,
    report,
    feature_store: Path,
    split: str,
    proximity_horizon: int,
    max_files: int | None = None,
) -> tuple[pd.DataFrame, Dict]:
    direction_model = getattr(report, "unified_direction", None)
    proximity_model = (getattr(report, "unified_proximity", {}) or {}).get(int(proximity_horizon))
    if direction_model is None:
        raise ValueError("saved report has no unified_direction model")
    if proximity_model is None:
        raise ValueError(f"saved report has no unified_proximity model for h={proximity_horizon}")

    paths = _proximity_files(feature_store, int(proximity_horizon), split)
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"no proximity shards found for h={proximity_horizon} split={split}")

    direction_cache: Dict[Tuple[str, int], pd.DataFrame] = {}
    acc: Dict[str, Dict[str, List[np.ndarray]]] = {}
    joined_rows = 0
    dropped_no_direction = 0
    columns = sorted(set(
        list(getattr(direction_model, "feature_names", []))
        + list(getattr(proximity_model, "feature_names", []))
        + ["ts", "bar_idx", "touch_label", "distance_atr", "side_above"]
    ))

    for path in paths:
        symbol, fold = _parse_symbol_fold(path)
        frame = pd.read_parquet(path, columns=columns)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["ts"] = frame["ts"].astype(str)
        frame["bar_idx"] = pd.to_numeric(frame["bar_idx"], errors="coerce").astype("Int64")
        key = (symbol, fold)
        if key not in direction_cache:
            direction_cache[key] = _load_direction_frame(feature_store, symbol, split, fold)
        merged = frame.merge(direction_cache[key], on=["ts", "bar_idx"], how="left")
        missing_dir = merged["direction_label"].isna()
        dropped_no_direction += int(missing_dir.sum())
        merged = merged[~missing_dir].copy()
        if merged.empty:
            continue
        joined_rows += int(len(merged))

        p_touch = proximity_model.predict_frame(merged)
        p_up = direction_model.predict_batch(merged)
        y_up = merged["direction_label"].astype(int).to_numpy()
        side_above = merged["side_above"].astype(float).to_numpy() >= 0.5
        distance = merged["distance_atr"].astype(float).to_numpy()

        # Re-express the direction target around the pool side:
        # above-pool setups want UP; below-pool setups want DOWN.
        y_to_pool = np.where(side_above, y_up, 1 - y_up)
        p_to_pool = np.where(side_above, p_up, 1.0 - p_up)

        _append(acc, "all proximity setup rows", y_to_pool, p_to_pool, side_above)

        masks = {
            "distance 2-8 ATR": (distance >= 2.0) & (distance < 8.0),
            "distance 3-8 ATR": (distance >= 3.0) & (distance < 8.0),
            "p_touch>=0.80 & distance 2-8 ATR": (p_touch >= 0.80) & (distance >= 2.0) & (distance < 8.0),
            PRIMARY_SEGMENT: (p_touch >= 0.85) & (distance >= 2.0) & (distance < 8.0),
            "p_touch>=0.90 & distance 2-8 ATR": (p_touch >= 0.90) & (distance >= 2.0) & (distance < 8.0),
            "p_touch>=0.85 & distance 3-8 ATR": (p_touch >= 0.85) & (distance >= 3.0) & (distance < 8.0),
            "p_touch>=0.85 & distance 5-8 ATR": (p_touch >= 0.85) & (distance >= 5.0) & (distance < 8.0),
            "p_touch<0.50 all distances": p_touch < 0.50,
        }
        for segment, mask in masks.items():
            if np.any(mask):
                _append(acc, segment, y_to_pool[mask], p_to_pool[mask], side_above[mask])

    rows: List[SegmentMetrics] = []
    for segment, parts in acc.items():
        y = np.concatenate(parts["y"]) if parts["y"] else np.array([], dtype=int)
        p = np.concatenate(parts["p"]) if parts["p"] else np.array([], dtype=float)
        side = np.concatenate(parts["side"]) if parts["side"] else np.array([], dtype=bool)
        rows.append(_metrics(segment, y, p, side))

    order = [
        "all proximity setup rows",
        "distance 2-8 ATR",
        "distance 3-8 ATR",
        "p_touch>=0.80 & distance 2-8 ATR",
        PRIMARY_SEGMENT,
        "p_touch>=0.90 & distance 2-8 ATR",
        "p_touch>=0.85 & distance 3-8 ATR",
        "p_touch>=0.85 & distance 5-8 ATR",
        "p_touch<0.50 all distances",
    ]
    order_map = {name: i for i, name in enumerate(order)}
    out = pd.DataFrame([asdict(row) for row in rows])
    if not out.empty:
        out["_order"] = out["segment"].map(order_map).fillna(999)
        out = out.sort_values(["_order", "segment"]).drop(columns=["_order"]).reset_index(drop=True)
    meta = {
        "proximity_files": len(paths),
        "joined_rows": joined_rows,
        "dropped_no_direction": dropped_no_direction,
        "proximity_horizon": int(proximity_horizon),
    }
    return out, meta


def _full_direction_metrics(report, feature_store: Path, split: str) -> SegmentMetrics:
    direction_model = getattr(report, "unified_direction", None)
    if direction_model is None:
        raise ValueError("saved report has no unified_direction model")
    paths = sorted((feature_store / "direction").glob(f"symbol=*/split={split}/fold=*.parquet"))
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    y = frame["direction_label"].astype(int).to_numpy()
    p = direction_model.predict_batch(frame)
    side = np.ones(len(y), dtype=bool)
    return _metrics("full direction rows: p_up vs y_up", y, p, side)


def _decision(primary: pd.Series | None) -> tuple[str, str]:
    if primary is None or primary.empty or pd.isna(primary.get("auc", np.nan)):
        return "BLOCKED", "Primary high-P_touch tradeable-distance segment was not available."
    auc = float(primary["auc"])
    if auc >= 0.60:
        return "VIABLE", "Direction signal is preserved inside high-P_touch, 2-8 ATR setup rows."
    if auc >= 0.55:
        return "MARGINAL", "Direction signal is weak but not dead inside high-P_touch, 2-8 ATR setup rows."
    return "NOT_VIABLE", "Direction signal collapses inside high-P_touch, 2-8 ATR setup rows."


def _fmt_pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{100.0 * float(x):.{digits}f}%"


def _fmt_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{float(x):.{digits}f}"


def _write_report(
    *,
    report_path: Path,
    csv_path: Path,
    model_report: Path,
    feature_store: Path,
    split: str,
    metrics: pd.DataFrame,
    full_direction: SegmentMetrics,
    meta: Dict,
) -> None:
    primary_df = metrics[metrics["segment"] == PRIMARY_SEGMENT]
    primary = primary_df.iloc[0] if not primary_df.empty else None
    decision, reason = _decision(primary)

    rows = []
    for row in metrics.to_dict("records"):
        rows.append(
            "| {segment} | {n} | {base} | {mean_p} | {auc} | {acc} | {top_acc} | {long_rows} | {short_rows} | {long_auc} | {short_auc} |".format(
                segment=row["segment"],
                n=f"{int(row['n']):,}",
                base=_fmt_pct(row["base_rate"]),
                mean_p=_fmt_pct(row["mean_p_dir"]),
                auc=_fmt_num(row["auc"]),
                acc=_fmt_pct(row["accuracy_50"]),
                top_acc=_fmt_pct(row["top_quartile_accuracy"]),
                long_rows=f"{int(row['long_rows']):,}",
                short_rows=f"{int(row['short_rows']):,}",
                long_auc=_fmt_num(row["long_auc"]),
                short_auc=_fmt_num(row["short_auc"]),
            )
        )

    lines = [
        "# Phase 4 Track A Direction-Conditional Audit",
        "",
        f"**Decision:** {decision}",
        "",
        reason,
        "",
        "This audit tests whether direction still helps after conditioning on the setup rows",
        "that Track A would care about: high P_touch and tradeable distance. Metrics are",
        "pool-side relative: above-pool rows want UP, below-pool rows want DOWN.",
        "",
        "## Scope",
        "",
        f"- Model report: `{model_report}`",
        f"- Feature store: `{feature_store}`",
        f"- Split: `{split}`",
        f"- Proximity horizon: `{meta.get('proximity_horizon')}`",
        f"- Proximity shards read: `{meta.get('proximity_files')}`",
        f"- Joined setup rows: `{int(meta.get('joined_rows', 0)):,}`",
        f"- Rows dropped with no direction label: `{int(meta.get('dropped_no_direction', 0)):,}`",
        f"- Metrics CSV: `{csv_path}`",
        "",
        "## Baseline Direction Model",
        "",
        f"- Full OOS direction rows: `{full_direction.n:,}`",
        f"- Full OOS p_up AUC: `{_fmt_num(full_direction.auc)}`",
        f"- Full OOS accuracy at 0.50: `{_fmt_pct(full_direction.accuracy_50)}`",
        "",
        "## Primary Gate",
        "",
    ]
    if primary is not None:
        lines.extend([
            f"- Segment: `{PRIMARY_SEGMENT}`",
            f"- Rows: `{int(primary['n']):,}`",
            f"- AUC: `{_fmt_num(primary['auc'])}`",
            f"- Accuracy at 0.50: `{_fmt_pct(primary['accuracy_50'])}`",
            f"- Top-quartile confidence accuracy: `{_fmt_pct(primary['top_quartile_accuracy'])}`",
            f"- Long/short rows: `{int(primary['long_rows']):,}` / `{int(primary['short_rows']):,}`",
            f"- Long/short AUC: `{_fmt_num(primary['long_auc'])}` / `{_fmt_num(primary['short_auc'])}`",
        ])
        if int(primary["n"]) < 500:
            lines.append("- Sample caution: this strict high-P_touch segment is small; use broader sensitivity bands in the sweep.")
    else:
        lines.append("- Primary segment was not available.")

    lines.extend([
        "",
        "Decision rules:",
        "",
        "- `AUC >= 0.60`: Direction conditionality passes.",
        "- `0.55 <= AUC < 0.60`: Marginal; continue cautiously.",
        "- `AUC < 0.55`: Track A direction edge is not preserved.",
        "",
        "## Conditional Metrics",
        "",
        "| Segment | N | Base Correct Dir | Mean P_dir | AUC | Acc@0.50 | Top-Q Acc | Long Rows | Short Rows | Long AUC | Short AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## Interpretation",
        "",
    ])
    if decision == "VIABLE":
        lines.extend([
            "Direction remains useful inside high-P_touch tradeable-distance setup rows.",
            "Track A can proceed, but the next implementation should be Execution Simulator",
            "v2 before any large pre-touch sweep, because profitability still depends on fills,",
            "slippage, stops, targets, and costs.",
        ])
    elif decision == "MARGINAL":
        lines.extend([
            "Direction has some conditional signal but not enough to trust blindly.",
            "A pre-touch sweep should use strict execution assumptions and DSR correction.",
        ])
    else:
        lines.extend([
            "The direction edge does not survive the Track A conditioning step.",
            "Do not run a large pre-touch parameter sweep without redefining the direction layer.",
        ])
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", type=Path, default=DEFAULT_MODEL_REPORT)
    parser.add_argument("--feature-store-dir", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--proximity-horizon", type=int, default=DEFAULT_PROXIMITY_HORIZON)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all proximity shards.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = _load_report(args.model_report)
    max_files = args.max_files if args.max_files > 0 else None
    metrics, meta = collect_metrics(
        report=report,
        feature_store=args.feature_store_dir,
        split=args.split,
        proximity_horizon=args.proximity_horizon,
        max_files=max_files,
    )
    full_direction = _full_direction_metrics(report, args.feature_store_dir, args.split)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.csv, index=False)
    _write_report(
        report_path=args.report,
        csv_path=args.csv,
        model_report=args.model_report,
        feature_store=args.feature_store_dir,
        split=args.split,
        metrics=metrics,
        full_direction=full_direction,
        meta=meta,
    )
    primary = metrics[metrics["segment"] == PRIMARY_SEGMENT]
    decision, reason = _decision(primary.iloc[0] if not primary.empty else None)
    print(f"Track A direction-conditional audit decision: {decision}")
    print(reason)
    print(f"report: {args.report}")
    print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
