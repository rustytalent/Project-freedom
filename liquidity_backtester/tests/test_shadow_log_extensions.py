"""Stream L extensions — tests for the additional integration sites
and resolvers added on top of the avoidance baseline.

Sites covered here:
  * brief Q-below-threshold rejection -> shadow event
  * drift.record_drift_alerts_to_shadow helper
  * options.executor.record_pre_trade_skip_shadow helper
  * arsenal evaluator below-target-to-cost rejection -> shadow event
  * the corresponding next-session resolvers in the replay script
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from analysis.replay_shadow_counterfactuals import (
    _resolve_below_threshold_long_signal,
    _resolve_drift_alert,
    _resolve_macro_gate_closed,
    replay_one_date,
)
from liqpool.drift import (
    DriftMetrics,
    DriftReport,
    record_drift_alerts_to_shadow,
)
from liqpool.products.shadow_log import (
    EVENT_KIND_BELOW_TARGET_TO_COST,
    EVENT_KIND_DRIFT_FLAG_FIRED,
    EVENT_KIND_MACRO_GATE_CLOSED,
    EVENT_KIND_Q_BELOW_THRESHOLD,
    EVENT_KIND_SKIP_OPTIONS,
    ShadowLogger,
    read_shadow_events,
)


# ---------------------------------------------------------------------------
# Brief Q-below-threshold integration
# ---------------------------------------------------------------------------


def _bars_for_brief(n: int = 80, start: str = "2026-05-22 03:45") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.2, "low": close - 0.2,
        "close": close, "volume": np.full(n, 1000.0),
    }, index=idx)


class _AssetData:
    def __init__(self, df):
        self.base_df = df


class _Report:
    def __init__(self, assets, **kwargs):
        self.assets = assets
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.unified_direction = None
        self.unified_proximity = {}


def test_brief_emits_below_threshold_event_for_dropped_predictions(tmp_path: Path):
    """When the brief's watchlist filter drops predictions below
    p_long < min_p_threshold, each dropped prediction is a paired
    training row. Verified by injecting a mock _gather_active_pool_predictions
    that returns a mix of above- and below-threshold predictions."""
    from liqpool.products import daily_brief as db
    from liqpool.products.daily_brief import generate_brief

    def _pred(symbol: str, sector: str, side: str, key_level: float,
              p_long: float, p_short: float = 0.10) -> Dict[str, Any]:
        return {
            "symbol": symbol, "sector": sector,
            "side_trade": side,
            "side_from_close": "below" if side == "long" else "above",
            "key_level": key_level,
            "key_level_type": (
                "demand_pool" if side == "long" else "supply_pool"
            ),
            "dist_atr_at_last_bar": 1.0,
            "q_pred": 0.5,
            "p_short": p_short,
            "p_long": p_long,
            "p_per_horizon": {"12": p_short, "60": p_long},
            "horizons_evaluated": [12, 60],
        }

    fake_predictions = [
        _pred("HDFCBANK", "BANKING", "long", 95.0, p_long=0.45),  # above
        _pred("TCS", "IT", "short", 502.0, p_long=0.12),  # below
        _pred("ICICI", "BANKING", "long", 850.0, p_long=0.08),  # below
    ]

    def _patched(report, max_entries: int = 200):
        return fake_predictions

    real = db._gather_active_pool_predictions
    db._gather_active_pool_predictions = _patched  # type: ignore[assignment]
    try:
        shadow = ShadowLogger(tmp_path)
        report = _Report({"HDFCBANK": _AssetData(_bars_for_brief())})
        generate_brief(
            report,
            trading_date_ist="2026-06-03",
            shadow_log_writer=shadow,
        )
        shadow.commit()
    finally:
        db._gather_active_pool_predictions = real  # type: ignore[assignment]

    dropped = read_shadow_events(
        tmp_path, event_kind=EVENT_KIND_Q_BELOW_THRESHOLD,
    )
    # Two predictions dropped: TCS and ICICI.
    assert len(dropped) == 2
    syms = set(dropped["symbol"].tolist())
    assert syms == {"TCS", "ICICI"}
    # The one above threshold (HDFCBANK) must NOT appear as Q-below.
    assert "HDFCBANK" not in syms


# ---------------------------------------------------------------------------
# Drift helper
# ---------------------------------------------------------------------------


def _drift_report_with_alert() -> DriftReport:
    metrics = DriftMetrics(
        run_timestamp="2026-06-09T03:00:00Z",
        oos_broad_respect=0.35,
        oos_strict_respect=0.30,
        mean_overfit_gap=0.02,
        direction_auc=0.55,
        direction_top_quartile_confidence=0.55,
        n_oos_pools=1000,
        proximity_auc_h78=0.92,
    )
    rpt = DriftReport(current=metrics, baseline=None)
    rpt.alerts.append({
        "metric": "oos_broad_respect", "value": 0.35,
        "threshold": 0.40, "severity": "critical",
        "msg": "below threshold",
    })
    rpt.alerts.append({
        "metric": "direction_auc", "value": 0.55,
        "threshold": 0.56, "severity": "warning",
        "msg": "borderline",
    })
    rpt.overall_status = "DRIFT_DETECTED"
    return rpt


def test_record_drift_alerts_to_shadow_emits_one_per_alert(tmp_path: Path):
    logger = ShadowLogger(tmp_path)
    rpt = _drift_report_with_alert()
    n = record_drift_alerts_to_shadow(
        logger, rpt, trading_date_ist="2026-06-09",
        model_bundle_version="bundle_v9_xyz",
    )
    assert n == 2
    logger.commit()
    df = read_shadow_events(tmp_path, event_kind=EVENT_KIND_DRIFT_FLAG_FIRED)
    assert len(df) == 2
    metrics_logged = set()
    for _, row in df.iterrows():
        ctx = json.loads(row["decision_context"])
        metrics_logged.add(ctx["metric"])
        assert ctx["model_bundle_version"] == "bundle_v9_xyz"
        assert ctx["overall_status"] == "DRIFT_DETECTED"
    assert metrics_logged == {"oos_broad_respect", "direction_auc"}


def test_record_drift_alerts_to_shadow_is_no_op_when_no_alerts(tmp_path: Path):
    logger = ShadowLogger(tmp_path)
    rpt = DriftReport(current=DriftMetrics(
        run_timestamp="2026-06-09T03:00:00Z",
        oos_broad_respect=0.5, oos_strict_respect=0.45,
        mean_overfit_gap=0.0,
        direction_auc=0.60, direction_top_quartile_confidence=0.60,
        n_oos_pools=1000,
        proximity_auc_h78=0.95,
    ), baseline=None)
    rpt.overall_status = "OK"
    assert record_drift_alerts_to_shadow(logger, rpt, "2026-06-09") == 0


def test_record_drift_alerts_swallows_writer_exceptions(tmp_path: Path):
    class _ExplodingLogger(ShadowLogger):
        def record(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("disk full")

    rpt = _drift_report_with_alert()
    # Must return 0 (no events written) and NOT raise.
    n = record_drift_alerts_to_shadow(
        _ExplodingLogger(tmp_path), rpt, "2026-06-09",
    )
    assert n == 0


# ---------------------------------------------------------------------------
# Options pre-trade SKIP helper
# ---------------------------------------------------------------------------


def _make_ccv(macro: float, lcs_target: float) -> Any:
    """Construct a minimal CCV with the requested macro score and LCS."""
    from liqpool.options.ccv import CCV
    return CCV(
        macro_score=macro,
        regime_score=0.0,
        pool_score=0.0,
        options_score=0.0,
        micro_score=0.0,
        manipulation_score=0.0,
        micro_forced_flag=False,
        pool_distortion_flag=False,
        micro_organic_score=0.0,
        pool_holding_strength=0.0,
        manip_micro_alignment=0.0,
        manip_pool_alignment=0.0,
        regime_pool_alignment=0.0,
        macro_regime_alignment=0.0,
        lcs=lcs_target,
    )


def test_record_pre_trade_skip_for_macro_gate_closed(tmp_path: Path):
    from liqpool.options.executor import (
        pre_trade_decision,
        record_pre_trade_skip_shadow,
    )
    logger = ShadowLogger(tmp_path)
    # macro_score 0.05 < MACRO_GATE_THRESHOLD (0.10) -> SKIP, gate closed
    ccv = _make_ccv(macro=0.05, lcs_target=0.40)
    d = pre_trade_decision(ccv)
    assert d.action == "SKIP"
    assert d.reason.startswith("macro_gate_closed")
    sid = record_pre_trade_skip_shadow(
        logger, d, ccv, trading_date_ist="2026-06-09",
        underlying="NIFTY", strike_label="24500_PE",
    )
    assert sid is not None
    logger.commit()
    df = read_shadow_events(
        tmp_path, event_kind=EVENT_KIND_MACRO_GATE_CLOSED,
    )
    assert len(df) == 1


def test_record_pre_trade_skip_for_signal_too_weak(tmp_path: Path):
    from liqpool.options.executor import (
        pre_trade_decision,
        record_pre_trade_skip_shadow,
    )
    logger = ShadowLogger(tmp_path)
    # macro OK; |lcs| 0.05 < WAIT_THRESHOLD -> SKIP, lcs below floor.
    ccv = _make_ccv(macro=0.50, lcs_target=0.05)
    d = pre_trade_decision(ccv)
    assert d.action == "SKIP"
    record_pre_trade_skip_shadow(
        logger, d, ccv, trading_date_ist="2026-06-09",
        underlying="NIFTY", strike_label="24500_CE",
    )
    logger.commit()
    df = read_shadow_events(tmp_path, event_kind=EVENT_KIND_SKIP_OPTIONS)
    assert len(df) == 1


def test_record_pre_trade_skip_skips_enter_decisions(tmp_path: Path):
    """ENTER actions are NOT shadow-logged (they take the trade and
    get tracked in the outcome log via the normal pipeline)."""
    from liqpool.options.executor import (
        pre_trade_decision,
        record_pre_trade_skip_shadow,
    )
    logger = ShadowLogger(tmp_path)
    ccv = _make_ccv(macro=0.50, lcs_target=0.40)  # |lcs| >= 0.30
    d = pre_trade_decision(ccv)
    assert d.action == "ENTER"
    sid = record_pre_trade_skip_shadow(
        logger, d, ccv, trading_date_ist="2026-06-09",
        underlying="NIFTY", strike_label="x",
    )
    assert sid is None  # nothing logged
    summary = logger.commit()
    assert summary["events_written"] == 0


# ---------------------------------------------------------------------------
# Replay resolvers for new kinds
# ---------------------------------------------------------------------------


def _write_warehouse(bundle: Path, sym: str, day: str,
                     open_px: float, close_px: float,
                     high_px: float, low_px: float) -> None:
    idx = pd.date_range(f"{day} 03:45", periods=75, freq="5min")
    closes = np.linspace(open_px, close_px, len(idx))
    df = pd.DataFrame({
        "open": closes, "high": high_px, "low": low_px,
        "close": closes, "volume": 1000.0,
    }, index=idx)
    bundle.mkdir(parents=True, exist_ok=True)
    df.to_parquet(bundle / f"{sym}.parquet")


def test_resolve_below_threshold_long_signal_uses_key_level(tmp_path: Path):
    """Q-below-threshold payload: side='long', key_level=105.
    Next session high reaches 106 -> rejection_regret."""
    bundle = tmp_path / "warehouse"
    _write_warehouse(bundle, "HDFCBANK", "2026-06-10",
                     open_px=100, close_px=104,
                     high_px=106, low_px=99)
    ev = pd.Series({
        "symbol": "HDFCBANK",
        "trading_date_ist": "2026-06-09",
        "decision_context": json.dumps({
            "side_trade": "long",
            "key_level": 105.0,
        }),
    })
    out = _resolve_below_threshold_long_signal(bundle, ev)
    assert out is not None
    assert out["target_touched"] is True
    assert out["verdict"] == "rejection_regret"


def test_resolve_below_threshold_uses_entry_plus_target_atr(tmp_path: Path):
    """below-target-to-cost payload shape: entry_price + target_atr +
    atr_at_entry give the target; verdict is computed against next-
    session high/low."""
    bundle = tmp_path / "warehouse"
    _write_warehouse(bundle, "TCS", "2026-06-10",
                     open_px=500, close_px=503,
                     high_px=504, low_px=499)  # high < 510 -> vindicated
    ev = pd.Series({
        "symbol": "TCS",
        "trading_date_ist": "2026-06-09",
        "decision_context": json.dumps({
            "side": "long",
            "entry_price": 500.0,
            "target_atr": 2.0,
            "atr_at_entry": 5.0,  # target = 500 + 2 * 5 = 510
        }),
    })
    out = _resolve_below_threshold_long_signal(bundle, ev)
    assert out is not None
    assert out["target_touched"] is False
    assert out["verdict"] == "rejection_vindicated"


def test_resolve_macro_gate_closed_records_realised_range(tmp_path: Path):
    bundle = tmp_path / "warehouse"
    _write_warehouse(bundle, "NIFTY", "2026-06-10",
                     open_px=24000, close_px=24200,
                     high_px=24300, low_px=23950)
    ev = pd.Series({
        "symbol": "NIFTY",
        "trading_date_ist": "2026-06-09",
        "decision_context": json.dumps({"reason": "macro_gate_closed_x"}),
    })
    out = _resolve_macro_gate_closed(bundle, ev)
    assert out is not None
    assert out["realised_close_over_open"] > 0
    assert out["realised_range_open_norm"] > 0


def test_resolve_drift_alert_marks_deferred():
    ev = pd.Series({
        "symbol": None,
        "trading_date_ist": "2026-06-09",
        "decision_context": json.dumps({}),
    })
    out = _resolve_drift_alert(Path("/nonexistent"), ev)
    assert out is not None
    assert out["resolution_kind"] == "deferred"


def test_replay_batch_handles_mixed_event_kinds(tmp_path: Path):
    """End-to-end: a single date with three different event kinds
    runs through one replay call without crashing, and each
    resolver path attempts its own resolution."""
    bundle = tmp_path / "warehouse"
    _write_warehouse(bundle, "HDFCBANK", "2026-06-10",
                     open_px=100, close_px=102,
                     high_px=106, low_px=99)
    logger = ShadowLogger(tmp_path)
    logger.record(
        event_kind=EVENT_KIND_Q_BELOW_THRESHOLD,
        trading_date_ist="2026-06-09",
        symbol="HDFCBANK", detail_token="level_105",
        decision_context={"side_trade": "long", "key_level": 105.0},
    )
    logger.record(
        event_kind=EVENT_KIND_DRIFT_FLAG_FIRED,
        trading_date_ist="2026-06-09",
        symbol=None, detail_token="oos_broad_respect_bundle_x",
        decision_context={"metric": "oos_broad_respect"},
    )
    logger.commit()

    summary = replay_one_date(tmp_path, bundle, "2026-06-09")
    assert summary["scanned"] == 2
    # Drift event resolves to a deferred outcome (counted as resolved).
    # Q-below resolves to a touched/not-touched outcome (also resolved).
    assert summary["resolved"] == 2
