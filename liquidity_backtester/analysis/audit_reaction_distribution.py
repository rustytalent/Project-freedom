"""Stage A — Reaction prediction distribution audit.

Loads the saved model bundle and the OOS post-touch event shards from the
feature store, runs each reaction sub-model (strict_reaction, reclaim_success,
break_continuation) over the FULL OOS population, and emits a distribution +
reliability report.

This is a one-off diagnostic. It tests whether the "every confirmation = 98%"
pattern visible in the predict terminal is genuine model saturation, or a
display artifact of `reaction_alerts[:10]` being sorted by descending
probability and truncated.

Usage (typically run on the Mac where the bundle + feature store live):

    PYTHONPATH=. .venv/bin/python analysis/audit_reaction_distribution.py \
        --model-dir output_models/core25_latest \
        --feature-store output_feature_store/core25 \
        --out output_audit

Outputs to ``output_audit/``:
  - reaction_distribution.csv    one row per (target, bin) histogram
  - reaction_reliability.csv     one row per (target, decile) reliability bin
  - reaction_summary.csv         one row per target with summary stats
  - reaction_summary.txt         human-readable summary

Decision gate the script makes explicit at the end:
  - If <15% of predictions are >=0.85 and the distribution spans 0.2-0.85 with
    reasonable density throughout, the "all 98%" finding is a display artifact;
    proceed to Stage B (display only).
  - Otherwise the distribution really is saturated and a scoped recalibration
    is warranted.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from liqpool.reaction_model import (
    TARGET_COLUMNS,
    ReactionBinaryModel,
    ReactionModelSuite,
)


HISTOGRAM_EDGES = np.linspace(0.0, 1.0, 11)   # 10 equal-width bins
RELIABILITY_BINS = 10                          # deciles


def _load_bundle(model_dir: Path):
    bundle = model_dir / "multi_asset_report.pkl"
    if not bundle.exists():
        raise SystemExit(f"missing model bundle: {bundle}. Run --mode train first.")
    with bundle.open("rb") as f:
        return pickle.load(f)


def _load_oos_events(feature_store: Path) -> pd.DataFrame:
    """Read every OOS reaction-event parquet under the feature store root."""
    pattern = feature_store / "reaction_events" / "symbol=*" / "oos.parquet"
    files = sorted(feature_store.glob("reaction_events/symbol=*/oos.parquet"))
    if not files:
        raise SystemExit(
            f"no OOS reaction-event parquet under {pattern}. "
            "Was --mode train run with feature-store writes enabled?"
        )
    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def _suite_from_bundle(report) -> ReactionModelSuite:
    suite = getattr(report, "reaction_model", None)
    if suite is None or not getattr(suite, "models", None):
        raise SystemExit(
            "model bundle has no reaction_model suite — was Phase 3 reaction "
            "training enabled when the bundle was produced?"
        )
    return suite


def _histogram_rows(target: str, p: np.ndarray) -> List[Dict]:
    counts, _ = np.histogram(p, bins=HISTOGRAM_EDGES)
    total = max(1, len(p))
    rows = []
    for i, c in enumerate(counts):
        rows.append({
            "target": target,
            "bin_lo": float(HISTOGRAM_EDGES[i]),
            "bin_hi": float(HISTOGRAM_EDGES[i + 1]),
            "n": int(c),
            "share": float(c) / float(total),
        })
    return rows


def _reliability_rows(target: str, p: np.ndarray, y: np.ndarray
                      ) -> List[Dict]:
    if len(p) == 0 or len(np.unique(p)) < 2:
        return []
    try:
        bucket = pd.qcut(p, q=min(RELIABILITY_BINS, len(np.unique(p))),
                         duplicates="drop")
    except ValueError:
        return []
    rows = []
    df = pd.DataFrame({"p": p, "y": y, "bucket": bucket})
    for b, sub in df.groupby("bucket", observed=False):
        if sub.empty:
            continue
        rows.append({
            "target": target,
            "bucket": str(b),
            "n": int(len(sub)),
            "mean_prediction": float(sub["p"].mean()),
            "actual_rate": float(sub["y"].mean()),
            "calibration_error": float(sub["p"].mean() - sub["y"].mean()),
        })
    return rows


def _summary_row(target: str, model: ReactionBinaryModel,
                 p: np.ndarray, y: Optional[np.ndarray]) -> Dict:
    saturated_85 = float((p >= 0.85).mean()) if len(p) else 0.0
    saturated_95 = float((p >= 0.95).mean()) if len(p) else 0.0
    decisive_50 = float((p >= 0.50).mean()) if len(p) else 0.0
    auc = None
    if y is not None and len(np.unique(y)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p))
    return {
        "target": target,
        "n_predictions": int(len(p)),
        "n_labelled": int(len(y)) if y is not None else 0,
        "mean": float(p.mean()) if len(p) else 0.0,
        "median": float(np.median(p)) if len(p) else 0.0,
        "p10": float(np.quantile(p, 0.10)) if len(p) else 0.0,
        "p25": float(np.quantile(p, 0.25)) if len(p) else 0.0,
        "p75": float(np.quantile(p, 0.75)) if len(p) else 0.0,
        "p90": float(np.quantile(p, 0.90)) if len(p) else 0.0,
        "share_ge_0.50": decisive_50,
        "share_ge_0.85": saturated_85,
        "share_ge_0.95": saturated_95,
        "oos_auc": auc,
        "calibration_clip_low": 0.02,
        "calibration_clip_high": 0.98,
    }


def _predict_target(model: ReactionBinaryModel, events: pd.DataFrame
                    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run the calibrated predict over EVERY OOS event (not just labelled)."""
    p = model.predict_frame(events)
    if model.label_col in events.columns:
        y_raw = events[model.label_col]
        mask = y_raw.notna()
        y = y_raw[mask].astype(int).to_numpy() if mask.any() else None
    else:
        mask = pd.Series(False, index=events.index)
        y = None
    p_lab = p[mask.to_numpy()] if y is not None else None
    return p, (p_lab, y) if y is not None else (None, None)


def _verdict(summary_rows: List[Dict]) -> str:
    """Plain-language decision the script renders for the operator."""
    over85 = max((r["share_ge_0.85"] for r in summary_rows), default=0.0)
    medians = [r["median"] for r in summary_rows]
    median = max(medians) if medians else 0.0
    if over85 < 0.15 and 0.20 <= median <= 0.85:
        return (
            "DISPLAY ARTIFACT: distribution is healthy "
            f"(max share>=0.85 is {over85:.1%}, median is "
            f"{median:.2f}). The 'every confirmation = 98%' pattern is the "
            "predict terminal's top-10 sort, not model saturation. "
            "Proceed to Stage B (display fix only)."
        )
    return (
        "POSSIBLE SATURATION: max share>=0.85 is "
        f"{over85:.1%}, median is {median:.2f}. The full OOS distribution "
        "is heavier near the ceiling than expected. Re-plan a scoped "
        "recalibration before Stage B."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="output_models/core25_latest",
                    help="directory holding multi_asset_report.pkl")
    ap.add_argument("--feature-store", default="output_feature_store/core25",
                    help="feature-store root (parent of reaction_events/)")
    ap.add_argument("--out", default="output_audit",
                    help="output directory for the audit CSVs + summary")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    feature_store = Path(args.feature_store).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[audit] loading bundle: {model_dir / 'multi_asset_report.pkl'}")
    report = _load_bundle(model_dir)
    suite = _suite_from_bundle(report)
    print(f"[audit] reaction targets in bundle: {sorted(suite.models)}")

    print(f"[audit] reading OOS reaction events from: {feature_store}")
    events = _load_oos_events(feature_store)
    print(f"[audit] {len(events):,} OOS event rows loaded")

    hist_rows: List[Dict] = []
    rel_rows: List[Dict] = []
    sum_rows: List[Dict] = []
    for target in TARGET_COLUMNS:
        model = suite.models.get(target)
        if model is None:
            print(f"[audit] skip {target}: not present in bundle")
            continue
        p_all, labelled = _predict_target(model, events)
        sum_rows.append(_summary_row(target, model, p_all,
                                     labelled[1] if labelled[1] is not None else None))
        hist_rows.extend(_histogram_rows(target, p_all))
        if labelled[0] is not None and labelled[1] is not None:
            rel_rows.extend(_reliability_rows(target, labelled[0], labelled[1]))

    hist_df = pd.DataFrame(hist_rows)
    rel_df = pd.DataFrame(rel_rows)
    sum_df = pd.DataFrame(sum_rows)

    hist_path = out_dir / "reaction_distribution.csv"
    rel_path = out_dir / "reaction_reliability.csv"
    sum_path = out_dir / "reaction_summary.csv"
    txt_path = out_dir / "reaction_summary.txt"
    hist_df.to_csv(hist_path, index=False)
    rel_df.to_csv(rel_path, index=False)
    sum_df.to_csv(sum_path, index=False)

    verdict = _verdict(sum_rows)
    lines: List[str] = []
    lines.append("=== Reaction Prediction Distribution Audit ===")
    lines.append(f"OOS events: {len(events):,}")
    lines.append("")
    for r in sum_rows:
        lines.append(f"[{r['target']}]")
        lines.append(f"  n_predictions:    {r['n_predictions']:,}")
        lines.append(f"  n_labelled:       {r['n_labelled']:,}")
        lines.append(f"  mean / median:    {r['mean']:.3f} / {r['median']:.3f}")
        lines.append(f"  p10/p25/p75/p90:  {r['p10']:.3f} / {r['p25']:.3f} / "
                     f"{r['p75']:.3f} / {r['p90']:.3f}")
        lines.append(f"  share >= 0.50:    {r['share_ge_0.50']:.1%}")
        lines.append(f"  share >= 0.85:    {r['share_ge_0.85']:.1%}")
        lines.append(f"  share >= 0.95:    {r['share_ge_0.95']:.1%}")
        if r["oos_auc"] is not None:
            lines.append(f"  OOS AUC:          {r['oos_auc']:.3f}")
        lines.append("")
    lines.append(f"VERDICT: {verdict}")
    summary_text = "\n".join(lines)
    txt_path.write_text(summary_text + "\n", encoding="utf-8")

    print("\n" + summary_text)
    print()
    print(f"[audit] wrote {hist_path}")
    print(f"[audit] wrote {rel_path}")
    print(f"[audit] wrote {sum_path}")
    print(f"[audit] wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
