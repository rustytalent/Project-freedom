"""Track 5: model-drift detection.

Markets change. The models trained on Feb-May 2026 data won't generalise forever. This module
tracks key metrics across weekly walk-forward runs and alerts when any of them degrade beyond
a threshold — that's the signal to retrain (or pause trading until you understand why).

Metrics watched:
  - OOS broad respect          (pooled across assets / folds)
  - Mean per-asset overfit gap
  - Direction model OOS AUC
  - Direction top-quartile-confidence accuracy
  - Proximity model OOS AUC by active horizon
  - ML feature importance stability (Spearman rank correlation vs baseline)

State is persisted to `drift_state.json` so we can compare current run to history.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pandas as pd


@dataclass
class DriftThresholds:
    min_oos_broad_respect: float = 0.40
    """Pooled OOS broad respect below this = serious degradation. Currently 45% baseline."""

    max_overfit_gap: float = 0.18
    """Above 18% mean gap = optimizer overfitting hard."""

    min_direction_auc: float = 0.60
    """Below 0.60 = direction model losing signal."""

    min_top_quartile_confidence: float = 0.65
    """Below 65% = high-confidence calls aren't holding up."""

    min_proximity_auc: float = 0.85
    """Proximity is normally 0.95+. Below 0.85 = something's broken with state features."""

    min_feature_importance_corr: float = 0.50
    """Spearman corr of feature importances vs baseline. Below 0.5 = model has fundamentally
    shifted what it relies on — regime change signal."""


@dataclass
class DriftMetrics:
    """Snapshot of a single run's headline metrics."""
    run_timestamp: str
    oos_broad_respect: float
    oos_strict_respect: float
    mean_overfit_gap: float
    direction_auc: float
    direction_top_quartile_confidence: float
    n_oos_pools: int
    proximity_auc_h78: float = 0.0
    """Legacy compatibility field for older h=78 bundles/baselines."""
    proximity_auc_by_horizon: Dict[str, float] = field(default_factory=dict)
    """Active proximity horizons, e.g. {"12": 0.78, "36": 0.82, "60": 0.85}."""
    feature_importance_top10: Dict[str, float] = field(default_factory=dict)
    """Top-10 features by gain, name → gain. Used for rank-correlation stability check."""


@dataclass
class DriftReport:
    current: DriftMetrics
    baseline: Optional[DriftMetrics] = None
    alerts: List[Dict] = field(default_factory=list)
    feature_importance_corr: Optional[float] = None
    overall_status: str = "OK"     # "OK" | "WARNING" | "DRIFT_DETECTED"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman_dict_corr(a: Dict[str, float], b: Dict[str, float]) -> Optional[float]:
    """Spearman correlation of two name→gain dicts, restricted to keys present in both."""
    common = sorted(set(a.keys()) & set(b.keys()))
    if len(common) < 3:
        return None
    sa = pd.Series([a[k] for k in common])
    sb = pd.Series([b[k] for k in common])
    if sa.std() == 0 or sb.std() == 0:
        return 0.0
    return float(sa.rank().corr(sb.rank()))


# ---------------------------------------------------------------------------
# Build a DriftMetrics from a MultiAssetReport
# ---------------------------------------------------------------------------

def extract_drift_metrics(report) -> DriftMetrics:
    """Pull headline metrics out of a MultiAssetReport for drift comparison."""
    if report is None:
        raise ValueError("report is None")

    ml = report.unified_ml
    tr = report.unified_timing_report

    fi_top10 = {}
    if ml is not None:
        for name, gain in ml.feature_importance(10):
            fi_top10[name] = float(gain)

    proximity_auc_h78 = 0.0
    proximity_auc_by_horizon: Dict[str, float] = {}
    if tr is not None and tr.proximity_per_horizon:
        for s in tr.proximity_per_horizon:
            horizon = int(getattr(s, "horizon", 0) or 0)
            auc = float(getattr(s, "auc", 0.0) or 0.0)
            if horizon > 0:
                proximity_auc_by_horizon[str(horizon)] = auc
            if s.horizon == 78:
                proximity_auc_h78 = auc

    return DriftMetrics(
        run_timestamp=pd.Timestamp.utcnow().isoformat(),
        oos_broad_respect=float(report.pooled_oos_respect),
        oos_strict_respect=float(report.pooled_oos_strict),
        mean_overfit_gap=float(report.mean_overfit_gap),
        direction_auc=float(tr.direction_auc) if tr else 0.0,
        direction_top_quartile_confidence=float(tr.direction_top_quartile_acc) if tr else 0.0,
        n_oos_pools=int(report.total_oos_pools),
        proximity_auc_h78=proximity_auc_h78,
        proximity_auc_by_horizon=proximity_auc_by_horizon,
        feature_importance_top10=fi_top10,
    )


def check_drift(current: DriftMetrics, baseline: Optional[DriftMetrics],
                 thresholds: DriftThresholds = DriftThresholds()) -> DriftReport:
    """Compare current vs baseline; emit alerts for each metric that breaches its threshold."""
    rpt = DriftReport(current=current, baseline=baseline)

    def alert(metric: str, value: float, threshold: float, msg: str, severity: str = "warning"):
        rpt.alerts.append({
            "metric": metric, "value": float(value), "threshold": float(threshold),
            "severity": severity, "msg": msg,
        })

    # Absolute-threshold checks (don't require a baseline)
    if current.oos_broad_respect < thresholds.min_oos_broad_respect:
        alert("oos_broad_respect", current.oos_broad_respect,
              thresholds.min_oos_broad_respect,
              f"OOS broad respect {current.oos_broad_respect:.1%} below "
              f"{thresholds.min_oos_broad_respect:.0%} threshold", "critical")
    if current.mean_overfit_gap > thresholds.max_overfit_gap:
        alert("overfit_gap", current.mean_overfit_gap, thresholds.max_overfit_gap,
              f"Mean overfit gap {current.mean_overfit_gap:+.1%} above "
              f"{thresholds.max_overfit_gap:.0%} cap", "warning")
    if current.direction_auc < thresholds.min_direction_auc:
        alert("direction_auc", current.direction_auc, thresholds.min_direction_auc,
              f"Direction AUC {current.direction_auc:.3f} below "
              f"{thresholds.min_direction_auc:.2f} threshold", "warning")
    if current.direction_top_quartile_confidence < thresholds.min_top_quartile_confidence:
        alert("direction_top_quartile_confidence", current.direction_top_quartile_confidence,
              thresholds.min_top_quartile_confidence,
              f"Top-quartile-confidence accuracy {current.direction_top_quartile_confidence:.1%}"
              f" below {thresholds.min_top_quartile_confidence:.0%} threshold", "warning")
    proximity_by_h = getattr(current, "proximity_auc_by_horizon", {}) or {}
    if proximity_by_h:
        for horizon, auc in sorted(proximity_by_h.items(), key=lambda kv: int(kv[0])):
            if float(auc) < thresholds.min_proximity_auc:
                alert(f"proximity_auc_h{horizon}", float(auc),
                      thresholds.min_proximity_auc,
                      f"Proximity h={horizon} AUC {float(auc):.3f} below "
                      f"{thresholds.min_proximity_auc:.2f} threshold", "critical")
    elif current.proximity_auc_h78 < thresholds.min_proximity_auc:
        alert("proximity_auc_h78", current.proximity_auc_h78,
              thresholds.min_proximity_auc,
              f"Proximity h=78 AUC {current.proximity_auc_h78:.3f} below "
              f"{thresholds.min_proximity_auc:.2f} threshold", "critical")

    # Comparison vs baseline (feature-importance stability, regime shift detector)
    if baseline is not None:
        fi_corr = _spearman_dict_corr(current.feature_importance_top10,
                                       baseline.feature_importance_top10)
        rpt.feature_importance_corr = fi_corr
        if fi_corr is not None and fi_corr < thresholds.min_feature_importance_corr:
            alert("feature_importance_corr", fi_corr,
                  thresholds.min_feature_importance_corr,
                  f"Feature importance ranking correlation {fi_corr:.2f} below "
                  f"{thresholds.min_feature_importance_corr:.2f} — "
                  f"model relying on different features than baseline (possible regime shift)",
                  "warning")

    # Determine overall status
    has_critical = any(a["severity"] == "critical" for a in rpt.alerts)
    has_warning = any(a["severity"] == "warning" for a in rpt.alerts)
    if has_critical:
        rpt.overall_status = "DRIFT_DETECTED"
    elif has_warning:
        rpt.overall_status = "WARNING"
    else:
        rpt.overall_status = "OK"
    return rpt


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_baseline(path: Path | str) -> Optional[DriftMetrics]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return DriftMetrics(**data)
    except Exception as e:
        print(f"[drift] failed to load baseline {p}: {e}")
        return None


def save_baseline(metrics: DriftMetrics, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(metrics), indent=2, default=str))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def record_drift_alerts_to_shadow(
    writer: Any,
    report: DriftReport,
    trading_date_ist: str,
    model_bundle_version: str = "unknown_bundle",
) -> int:
    """Stream L hook — record one shadow event per drift alert.

    Adding this to the drift-check cron lets the Stream M.6 drift-
    imminent model train on the precursor trajectory of every alert
    that has fired: which metric, what value, what threshold, how it
    cleared (or didn't). Defensive: if anything goes wrong with the
    writer, we swallow and return 0 — drift monitoring MUST NOT crash
    because of shadow logging.

    Caller does:
        rpt = check_drift(current, baseline)
        record_drift_alerts_to_shadow(writer, rpt, today_ist)

    Returns the number of events successfully buffered. The caller
    still needs to call ``writer.commit()`` to flush them.
    """
    if writer is None or not report.alerts:
        return 0
    from .products.shadow_log import EVENT_KIND_DRIFT_FLAG_FIRED
    written = 0
    for alert_payload in report.alerts:
        try:
            metric = str(alert_payload.get("metric", "unknown"))
            writer.record(
                event_kind=EVENT_KIND_DRIFT_FLAG_FIRED,
                trading_date_ist=trading_date_ist,
                symbol=None,  # drift is on the unified-model layer
                detail_token=f"{metric}_{model_bundle_version}",
                decision_context={
                    "metric": metric,
                    "value": alert_payload.get("value"),
                    "threshold": alert_payload.get("threshold"),
                    "severity": alert_payload.get("severity"),
                    "msg": alert_payload.get("msg"),
                    "model_bundle_version": model_bundle_version,
                    "overall_status": report.overall_status,
                },
            )
            written += 1
        except Exception:
            continue
    return written


def print_drift_report(report: DriftReport, file=None) -> None:
    print("\n================ MODEL DRIFT CHECK ================", file=file)
    print(f"  Overall status:           {report.overall_status}", file=file)
    print(f"  Run timestamp:            {report.current.run_timestamp}", file=file)
    print(f"  OOS broad respect:        {report.current.oos_broad_respect:.1%}", file=file)
    print(f"  OOS strict respect:       {report.current.oos_strict_respect:.1%}", file=file)
    print(f"  Mean overfit gap:         {report.current.mean_overfit_gap:+.1%}", file=file)
    print(f"  Direction AUC:            {report.current.direction_auc:.3f}", file=file)
    print(f"  Direction top-qrt-conf:   {report.current.direction_top_quartile_confidence:.1%}",
          file=file)
    proximity_by_h = getattr(report.current, "proximity_auc_by_horizon", {}) or {}
    if proximity_by_h:
        horizon_bits = ", ".join(
            f"h={h}: {float(v):.3f}"
            for h, v in sorted(proximity_by_h.items(), key=lambda kv: int(kv[0]))
        )
        print(f"  Proximity AUCs:           {horizon_bits}", file=file)
    else:
        print(f"  Proximity AUC h=78:       {report.current.proximity_auc_h78:.3f}", file=file)
    print(f"  OOS pools tested:         {report.current.n_oos_pools}", file=file)

    if report.baseline is not None:
        print(f"\n  Baseline (last good run): {report.baseline.run_timestamp}", file=file)
        if report.feature_importance_corr is not None:
            print(f"  Feature importance corr:  {report.feature_importance_corr:.2f}  "
                  f"(>= 0.50 ideal)", file=file)
    else:
        print(f"\n  Baseline:                 NONE  (saving current as baseline)", file=file)

    if report.alerts:
        print(f"\n  ⚠  Alerts:", file=file)
        for a in report.alerts:
            sev = a["severity"].upper()
            print(f"    [{sev}] {a['metric']}: {a['msg']}", file=file)
    else:
        print(f"\n  No alerts — all metrics within thresholds.", file=file)
