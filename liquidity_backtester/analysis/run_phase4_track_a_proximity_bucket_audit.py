"""Track A proximity distance-bucket viability audit.

This Phase 4 gate checks whether the strong proximity model is still useful at
tradeable distances, or whether its headline AUC is mostly a trivial "pool is
already close" effect.

The script loads saved proximity models and OOS proximity feature-store shards,
predicts P_touch per row, and reports AUC/lift by distance bucket. It does not
retrain models.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


DEFAULT_MODEL_REPORT = Path("output_models/core25_latest/multi_asset_report.pkl")
DEFAULT_FEATURE_STORE = Path("output_feature_store/core25_fresh_may25")
DEFAULT_REPORT = Path("reports/phase4_track_a_proximity_bucket_audit.md")
DEFAULT_CSV = Path("reports/phase4_track_a_proximity_bucket_audit.csv")
DEFAULT_SPLIT = "oos"
DEFAULT_HORIZONS = (78, 156, 312)

BUCKETS: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 1.0, "0-1 ATR"),
    (1.0, 3.0, "1-3 ATR"),
    (3.0, 5.0, "3-5 ATR"),
    (5.0, 8.0, "5-8 ATR"),
    (8.0, 12.0, "8-12 ATR"),
    (12.0, float("inf"), "12+ ATR"),
)
TRADEABLE_BUCKET = "3-8 ATR"


@dataclass
class BucketMetrics:
    horizon: int
    bucket: str
    n: int
    positives: int
    base_rate: float
    mean_p_touch: float
    auc: float
    brier: float
    logloss: float
    top_decile_actual_touched: float
    top_decile_mean_p_touch: float
    lift_over_base_pp: float


def _parse_csv_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_report(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"model report not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _proximity_files(root: Path, horizon: int, split: str) -> List[Path]:
    return sorted((root / "proximity").glob(f"horizon={int(horizon)}/symbol=*/split={split}/fold=*.parquet"))


def _bucket_label(distance: np.ndarray) -> np.ndarray:
    labels = np.full(len(distance), "12+ ATR", dtype=object)
    for low, high, label in BUCKETS:
        if np.isinf(high):
            mask = distance >= low
        else:
            mask = (distance >= low) & (distance < high)
        labels[mask] = label
    return labels


def _metrics_for_arrays(horizon: int, bucket: str, y: np.ndarray, p: np.ndarray) -> BucketMetrics:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = int(len(y))
    positives = int(y.sum())
    base = float(y.mean()) if n else 0.0
    mean_p = float(p.mean()) if n else 0.0
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    brier = float(brier_score_loss(y, p)) if n else float("nan")
    ll = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])) if n else float("nan")
    top_actual = float("nan")
    top_mean = float("nan")
    lift_pp = float("nan")
    if n:
        k = max(1, int(np.ceil(n * 0.10)))
        top_idx = np.argsort(p)[-k:]
        top_actual = float(y[top_idx].mean())
        top_mean = float(p[top_idx].mean())
        lift_pp = float((top_actual - base) * 100.0)
    return BucketMetrics(
        horizon=int(horizon),
        bucket=bucket,
        n=n,
        positives=positives,
        base_rate=base,
        mean_p_touch=mean_p,
        auc=auc,
        brier=brier,
        logloss=ll,
        top_decile_actual_touched=top_actual,
        top_decile_mean_p_touch=top_mean,
        lift_over_base_pp=lift_pp,
    )


def _append_group(acc: Dict[Tuple[int, str], Dict[str, List[np.ndarray]]],
                  horizon: int, bucket: str, y: np.ndarray, p: np.ndarray) -> None:
    key = (int(horizon), bucket)
    if key not in acc:
        acc[key] = {"y": [], "p": []}
    acc[key]["y"].append(np.asarray(y, dtype=int))
    acc[key]["p"].append(np.asarray(p, dtype=float))


def collect_predictions(
    *,
    report,
    feature_store: Path,
    split: str,
    horizons: Sequence[int],
    max_files_per_horizon: int | None = None,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    models = getattr(report, "unified_proximity", {}) or {}
    rows: List[BucketMetrics] = []
    accum: Dict[Tuple[int, str], Dict[str, List[np.ndarray]]] = {}
    file_counts: Dict[int, int] = {}

    for horizon in horizons:
        model = models.get(int(horizon))
        if model is None:
            continue
        paths = _proximity_files(feature_store, int(horizon), split)
        if max_files_per_horizon is not None:
            paths = paths[:max_files_per_horizon]
        file_counts[int(horizon)] = len(paths)
        columns = sorted(set(list(getattr(model, "feature_names", [])) + ["touch_label", "distance_atr"]))
        for path in paths:
            frame = pd.read_parquet(path, columns=columns)
            if frame.empty:
                continue
            y = frame["touch_label"].astype(int).to_numpy()
            p = model.predict_frame(frame)
            d = frame["distance_atr"].astype(float).to_numpy()
            labels = _bucket_label(d)
            for _low, _high, label in BUCKETS:
                mask = labels == label
                if mask.any():
                    _append_group(accum, int(horizon), label, y[mask], p[mask])
            trade_mask = (d >= 3.0) & (d < 8.0)
            if trade_mask.any():
                _append_group(accum, int(horizon), TRADEABLE_BUCKET, y[trade_mask], p[trade_mask])

    for (horizon, bucket), parts in sorted(accum.items(), key=lambda x: (x[0][0], x[0][1])):
        y = np.concatenate(parts["y"]) if parts["y"] else np.array([], dtype=int)
        p = np.concatenate(parts["p"]) if parts["p"] else np.array([], dtype=float)
        rows.append(_metrics_for_arrays(horizon, bucket, y, p))
    return pd.DataFrame([asdict(row) for row in rows]), file_counts


def _fmt_pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{100.0 * float(x):.{digits}f}%"


def _fmt_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{float(x):.{digits}f}"


def _decision(primary: pd.Series | None) -> Tuple[str, str]:
    if primary is None or primary.empty or pd.isna(primary.get("auc", np.nan)):
        return "BLOCKED", "No primary h=78 3-8 ATR bucket metric was available."
    auc = float(primary["auc"])
    if auc > 0.70:
        return "VIABLE", "Primary 3-8 ATR proximity AUC is above 0.70."
    if auc < 0.55:
        return "NOT_VIABLE", "Primary 3-8 ATR proximity AUC is below 0.55."
    return "MARGINAL", "Primary 3-8 ATR proximity AUC is between 0.55 and 0.70."


def _write_report(
    *,
    report_path: Path,
    csv_path: Path,
    model_report: Path,
    feature_store: Path,
    split: str,
    metrics: pd.DataFrame,
    file_counts: Dict[int, int],
) -> None:
    primary_df = metrics[(metrics["horizon"] == 78) & (metrics["bucket"] == TRADEABLE_BUCKET)]
    primary = primary_df.iloc[0] if not primary_df.empty else None
    decision, reason = _decision(primary)

    rows = []
    if not metrics.empty:
        order = {label: i for i, (_lo, _hi, label) in enumerate(BUCKETS)}
        order[TRADEABLE_BUCKET] = 99
        table = metrics.copy()
        table["_order"] = table["bucket"].map(order).fillna(50)
        table = table.sort_values(["horizon", "_order"])
        for row in table.to_dict("records"):
            rows.append(
                "| {horizon} | {bucket} | {n} | {base} | {mean_p} | {auc} | {top_actual} | {lift} |".format(
                    horizon=int(row["horizon"]),
                    bucket=row["bucket"],
                    n=f"{int(row['n']):,}",
                    base=_fmt_pct(row["base_rate"]),
                    mean_p=_fmt_pct(row["mean_p_touch"]),
                    auc=_fmt_num(row["auc"]),
                    top_actual=_fmt_pct(row["top_decile_actual_touched"]),
                    lift=f"{float(row['lift_over_base_pp']):+.2f}pp" if not pd.isna(row["lift_over_base_pp"]) else "n/a",
                )
            )

    lines = [
        "# Phase 4 Track A Proximity Distance-Bucket Audit",
        "",
        f"**Decision:** {decision}",
        "",
        reason,
        "",
        "This is the Track A viability gate. It tests whether P_touch remains predictive",
        "at tradeable pre-touch distances, rather than only looking strong because very",
        "nearby pools are easy to classify.",
        "",
        "## Scope",
        "",
        f"- Model report: `{model_report}`",
        f"- Feature store: `{feature_store}`",
        f"- Split: `{split}`",
        f"- Shards read by horizon: `{file_counts}`",
        f"- Metrics CSV: `{csv_path}`",
        "",
        "## Primary Gate",
        "",
    ]
    if primary is not None:
        lines.extend([
            f"- Primary horizon: `78` bars",
            f"- Tradeable bucket: `3-8 ATR`",
            f"- Rows: `{int(primary['n']):,}`",
            f"- Base touch rate: `{_fmt_pct(primary['base_rate'])}`",
            f"- AUC: `{_fmt_num(primary['auc'])}`",
            f"- Top-decile actual touched: `{_fmt_pct(primary['top_decile_actual_touched'])}`",
            f"- Lift over bucket base: `{float(primary['lift_over_base_pp']):+.2f}pp`",
        ])
    else:
        lines.append("- Primary h=78 3-8 ATR bucket was not available.")

    lines.extend([
        "",
        "Decision rules from the Phase 4 brief:",
        "",
        "- `AUC > 0.70`: Track A is viable, run the full pre-touch sweep.",
        "- `AUC < 0.55`: Track A is not viable; proximity is mostly trivial/too weak here.",
        "- `0.55 <= AUC <= 0.70`: Track A is marginal; continue cautiously.",
        "",
        "## Bucket Metrics",
        "",
        "| Horizon | Bucket | N | Base Touch | Mean P_touch | AUC | Top-Decile Actual | Lift |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## Interpretation",
        "",
    ])
    if decision == "VIABLE":
        lines.extend([
            "The 3-8 ATR bucket clears the strict viability threshold on the primary horizon.",
            "Track A can proceed to the direction-conditional audit and then the reduced",
            "pre-touch parameter sweep under the v2 execution simulator.",
        ])
    elif decision == "MARGINAL":
        lines.extend([
            "The 3-8 ATR bucket has some signal, but not enough to call the pre-touch thesis",
            "proven. The next step should be the direction-conditional audit before any",
            "large parameter sweep.",
        ])
    else:
        lines.extend([
            "The primary tradeable-distance proximity signal does not clear the minimum",
            "threshold. Track A should not move to a large sweep unless the user explicitly",
            "chooses to investigate another horizon or distance definition.",
        ])
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", type=Path, default=DEFAULT_MODEL_REPORT)
    parser.add_argument("--feature-store-dir", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    parser.add_argument("--max-files-per-horizon", type=int, default=0, help="0 means all shards.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = _load_report(args.model_report)
    horizons = _parse_csv_ints(args.horizons)
    max_files = args.max_files_per_horizon if args.max_files_per_horizon > 0 else None
    metrics, file_counts = collect_predictions(
        report=report,
        feature_store=args.feature_store_dir,
        split=args.split,
        horizons=horizons,
        max_files_per_horizon=max_files,
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.csv, index=False)
    _write_report(
        report_path=args.report,
        csv_path=args.csv,
        model_report=args.model_report,
        feature_store=args.feature_store_dir,
        split=args.split,
        metrics=metrics,
        file_counts=file_counts,
    )
    primary = metrics[(metrics["horizon"] == 78) & (metrics["bucket"] == TRADEABLE_BUCKET)]
    decision, reason = _decision(primary.iloc[0] if not primary.empty else None)
    print(f"Track A proximity bucket audit decision: {decision}")
    print(reason)
    print(f"report: {args.report}")
    print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
