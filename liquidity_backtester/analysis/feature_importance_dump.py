"""Dump feature importance from a saved multi-asset bundle.

This is intentionally read-only: it does not retrain, re-score, or mutate the
bundle. It answers the Phase 4/Arsenal question: which of the newly added
context features actually land near the top of the trained models?

Example:
    PYTHONPATH=. .venv/bin/python -u analysis/feature_importance_dump.py \
        --bundle core25_head_alpha \
        --models direction,proximity_12,proximity_36,proximity_60,reaction \
        --top 20 \
        --out reports/feature_importance_head.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


NEW_FEATURE_GROUPS: Dict[str, List[str]] = {
    "MTF-CTX": [
        "htf_today_range_atr",
        "htf_today_pos_in_range",
        "htf_today_open_to_now_atr",
        "htf_session_volume_ratio",
        "htf_overnight_gap_atr",
    ],
    "AVWAP-FRVP": [
        "avwap_today_dist_atr",
        "avwap_today_slope_5_atr",
        "avwap_today_dev_sigmas",
        "avwap_week_dist_atr",
        "avwap_week_slope_5_atr",
        "poc_today_dist_atr",
        "vah_today_dist_atr",
        "val_today_dist_atr",
        "in_value_area_today",
    ],
    "EXPIRY-CTX": [
        "days_to_monthly_expiry",
        "is_weekly_expiry_day",
        "is_morning_after_expiry",
        "is_monthly_expiry_week",
    ],
}
NEW_FEATURE_TO_GROUP = {
    feature: group
    for group, features in NEW_FEATURE_GROUPS.items()
    for feature in features
}


def _resolve_bundle(bundle: str) -> Path:
    p = Path(bundle).expanduser()
    if p.is_file():
        return p
    if (p / "multi_asset_report.pkl").exists():
        return p / "multi_asset_report.pkl"
    if p.name == str(p):
        return Path("output_models") / str(p) / "multi_asset_report.pkl"
    return p / "multi_asset_report.pkl"


def _load_report(bundle: str) -> Any:
    p = _resolve_bundle(bundle)
    if not p.exists():
        raise SystemExit(f"missing bundle: {p}")
    with p.open("rb") as f:
        return pickle.load(f)


def _importance_pairs(model: Any) -> List[Tuple[str, float]]:
    if model is None:
        return []
    if hasattr(model, "_gbm") and hasattr(model, "feature_names"):
        gains = model._gbm.feature_importance(importance_type="gain")
        pairs = list(zip(list(model.feature_names), [float(x) for x in gains]))
        return sorted(pairs, key=lambda kv: -kv[1])
    if hasattr(model, "feature_importance"):
        raw = model.feature_importance(top_k=100000)
        pairs: List[Tuple[str, float]] = []
        for item in raw:
            if isinstance(item, dict):
                name = item.get("feature") or item.get("name")
                gain = item.get("importance_gain", item.get("gain", item.get("importance", 0.0)))
                if name is not None:
                    pairs.append((str(name), float(gain)))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                pairs.append((str(item[0]), float(item[1])))
        return sorted(pairs, key=lambda kv: -kv[1])
    return []


def _row(model_name: str, rank: int, feature: str, gain: float,
         horizon: Optional[int] = None,
         target: Optional[str] = None) -> Dict[str, Any]:
    return {
        "model": model_name,
        "horizon": horizon,
        "target": target,
        "rank": int(rank),
        "feature": feature,
        "gain": float(gain),
        "is_new_feature": feature in NEW_FEATURE_TO_GROUP,
        "new_feature_group": NEW_FEATURE_TO_GROUP.get(feature, ""),
        "top10": int(rank) <= 10,
    }


def _add_model_rows(rows: List[Dict[str, Any]], model_name: str, model: Any,
                    top: int, horizon: Optional[int] = None,
                    target: Optional[str] = None) -> None:
    for rank, (feature, gain) in enumerate(_importance_pairs(model)[:top], start=1):
        rows.append(_row(model_name, rank, feature, gain, horizon=horizon, target=target))


def _reaction_rows(report: Any, top: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    suite = getattr(report, "reaction_model", None)
    if suite is not None and hasattr(suite, "models"):
        for target, model in sorted(suite.models.items()):
            _add_model_rows(rows, f"reaction_{target}", model, top, target=str(target))
        return rows

    stored = getattr(report, "reaction_feature_importance", []) or []
    if stored:
        frame = pd.DataFrame(stored)
        if {"target", "feature", "importance_gain"}.issubset(frame.columns):
            for target, g in frame.groupby("target"):
                g = g.sort_values("importance_gain", ascending=False).head(top)
                for rank, r in enumerate(g.itertuples(index=False), start=1):
                    rows.append(_row(
                        f"reaction_{target}",
                        rank,
                        str(getattr(r, "feature")),
                        float(getattr(r, "importance_gain")),
                        target=str(target),
                    ))
    return rows


def _validation_metrics(report: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    direction = getattr(report, "unified_direction", None)
    if direction is not None:
        out.append({
            "model": "direction",
            "horizon": getattr(direction, "horizon", None),
            "val_auc": getattr(direction, "val_auc", None),
            "val_brier": getattr(direction, "val_brier", None),
            "base_rate": getattr(direction, "base_rate", None),
        })
    for h, model in sorted((getattr(report, "unified_proximity", {}) or {}).items()):
        out.append({
            "model": f"proximity_{h}",
            "horizon": int(h),
            "val_auc": getattr(model, "val_auc", None),
            "val_brier": getattr(model, "val_brier", None),
            "base_rate": getattr(model, "base_rate", None),
            "val_decile_lift": getattr(model, "val_decile_lift", None),
        })
    for row in getattr(report, "reaction_model_report", []) or []:
        row = dict(row)
        target = row.get("target", "")
        out.append({
            "model": f"reaction_{target}",
            "target": target,
            "val_auc": row.get("val_auc"),
            "oos_auc": row.get("oos_auc"),
            "oos_brier": row.get("oos_brier"),
            "base_rate": row.get("base_rate"),
        })
    return out


def _requested_model_rows(report: Any, model_specs: Iterable[str], top: int
                          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    prox = getattr(report, "unified_proximity", {}) or {}
    for spec in [s.strip() for s in model_specs if s.strip()]:
        low = spec.lower()
        if low == "direction":
            model = getattr(report, "unified_direction", None)
            if model is None:
                unavailable.append({"model": spec, "reason": "missing report.unified_direction"})
            else:
                _add_model_rows(rows, "direction", model, top,
                                horizon=getattr(model, "horizon", None))
        elif low.startswith("proximity_"):
            try:
                h = int(low.split("_", 1)[1])
            except ValueError:
                unavailable.append({"model": spec, "reason": "invalid proximity horizon"})
                continue
            model = prox.get(h)
            if model is None:
                unavailable.append({
                    "model": spec,
                    "reason": f"missing report.unified_proximity[{h}]",
                    "available_horizons": sorted(int(x) for x in prox.keys()),
                })
            else:
                _add_model_rows(rows, f"proximity_{h}", model, top, horizon=h)
        elif low == "reaction":
            reaction = _reaction_rows(report, top)
            if not reaction:
                unavailable.append({"model": spec, "reason": "missing reaction importance"})
            rows.extend(reaction)
        elif low == "quality":
            model = getattr(report, "unified_ml", None)
            if model is None:
                unavailable.append({"model": spec, "reason": "missing report.unified_ml"})
            else:
                _add_model_rows(rows, "quality", model, top)
        else:
            unavailable.append({"model": spec, "reason": "unknown model spec"})
    return rows, unavailable


def _proximity_gain_split(report: Any) -> Dict[str, Any]:
    prox = getattr(report, "unified_proximity", {}) or {}
    rows = []
    total = 0.0
    for h in (12, 36, 60):
        pairs = _importance_pairs(prox.get(h))
        gain = float(sum(max(0.0, float(g)) for _, g in pairs))
        rows.append({"horizon": h, "gain": gain})
        total += gain
    for row in rows:
        row["share"] = float(row["gain"] / total) if total > 0 else 0.0
    h12_share = next((row["share"] for row in rows if row["horizon"] == 12), 0.0)
    return {
        "rows": rows,
        "h12_share": float(h12_share),
        "h12_dominates_gt_70pct": bool(h12_share > 0.70),
        "note": (
            "Shares use full LightGBM gain when available, not only the printed top-k rows."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True,
                    help="bundle name under output_models/, model directory, or pickle path")
    ap.add_argument("--models", default="direction,proximity_12,proximity_36,proximity_60,reaction")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="reports/feature_importance_head.json")
    args = ap.parse_args()

    report = _load_report(args.bundle)
    rows, unavailable = _requested_model_rows(
        report, args.models.split(","), max(1, int(args.top))
    )
    hits_top10 = [r for r in rows if r["is_new_feature"] and r["top10"]]
    hits_top20 = [r for r in rows if r["is_new_feature"] and r["rank"] <= 20]
    output = {
        "bundle": str(_resolve_bundle(args.bundle)),
        "top": int(args.top),
        "new_feature_groups": NEW_FEATURE_GROUPS,
        "rows": rows,
        "new_feature_hits_top10": hits_top10,
        "new_feature_hits_top20": hits_top20,
        "proximity_gain_by_horizon": _proximity_gain_split(report),
        "validation_metrics": _validation_metrics(report),
        "unavailable": unavailable,
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"wrote {out_path}")
    print(f"wrote {csv_path}")
    print("\n=== New Features In Top 10 ===")
    if not hits_top10:
        print("(none)")
    else:
        for r in hits_top10:
            print(f"{r['model']:>24} rank={r['rank']:>2} "
                  f"{r['feature']} ({r['new_feature_group']}) gain={r['gain']:.1f}")
    print("\n=== Proximity Gain Split ===")
    for row in output["proximity_gain_by_horizon"]["rows"]:
        print(f"h={row['horizon']:<3} gain={row['gain']:.1f} share={row['share']:.1%}")
    if output["proximity_gain_by_horizon"]["h12_dominates_gt_70pct"]:
        print("h=12 dominates >70%; multi-horizon shape may be redundant.")
    if unavailable:
        print("\n=== Unavailable ===")
        for item in unavailable:
            print(f"{item['model']}: {item['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
