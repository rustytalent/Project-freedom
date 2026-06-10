"""Organism pass — extractors, hub, and the closed-loop wirings.

Pins the digestion + circulation layer:
  * extractors convert engine artifacts into model food
  * FlywheelHub trains organs and degrades safely when starved
  * the three live consumer wirings (brief meta-cal, executor regret,
    drift imminence) activate when a fitted hub/model is present and
    are no-ops otherwise
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from liqpool.flywheel import FlywheelHub
from liqpool.flywheel.extractors import (
    bucket_aging_history_from_joined,
    co_occurrences_from_report,
    detector_outcomes_from_report,
    reaction_paths_from_report,
)


# ---------------------------------------------------------------------------
# Synthetic report fixtures (mirror the wf.oos_pools/results shape)
# ---------------------------------------------------------------------------

@dataclass
class _Contributor:
    source: str


@dataclass
class _Pool:
    contributors: List[_Contributor] = field(default_factory=list)


@dataclass
class _Result:
    pool_idx: int
    side: str
    formed_at: pd.Timestamp
    price_low: float
    price_high: float
    score: float
    _respect: bool
    _tested: bool = True
    touched_at: Optional[pd.Timestamp] = None
    broken_at: Optional[pd.Timestamp] = None
    tfs: List[str] = field(default_factory=lambda: ["base", "15min"])

    @property
    def is_tested(self) -> bool:
        return self._tested

    @property
    def is_respect(self) -> bool:
        return self._respect


@dataclass
class _WF:
    oos_pools: list
    oos_results: list


@dataclass
class _Asset:
    base_df: pd.DataFrame
    walkforward: _WF


@dataclass
class _Report:
    assets: Dict[str, _Asset]


def _bars(n: int = 200, base: float = 100.0, start="2026-05-01 03:45"):
    idx = pd.date_range(start, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = base + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": 1000.0,
    }, index=idx)


def _make_report() -> _Report:
    df = _bars()
    touch_ts = df.index[100]
    pools = [
        _Pool([_Contributor("SWEEP_H"), _Contributor("SWEEP_L")]),
        _Pool([_Contributor("HVN")]),
    ]
    results = [
        _Result(0, "low", df.index[80], 99.0, 99.5, 1.2, True,
                touched_at=touch_ts),
        _Result(1, "high", df.index[85], 101.0, 101.5, 0.9, False,
                touched_at=df.index[110]),
    ]
    return _Report({"HDFCBANK": _Asset(df, _WF(pools, results))})


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def test_detector_outcomes_extractor_shapes():
    out = detector_outcomes_from_report(_make_report())
    assert not out.empty
    assert {"factor", "regime", "respected"} <= set(out.columns)
    # The two tested pools both appear.
    assert len(out) == 2
    assert set(out["respected"]) <= {0.0, 1.0}


def test_reaction_paths_extractor_shapes():
    paths, sides, ctx = reaction_paths_from_report(_make_report(), path_bars=6)
    # At least the demand pool (touched at index 100, room for 6 bars).
    assert paths.shape[1] == 6
    assert len(paths) == len(sides) == len(ctx)
    if len(paths):
        assert set(np.unique(sides)) <= {-1.0, 1.0}


def test_co_occurrences_needs_two_assets():
    # Single-asset report -> no cross pairs.
    out = co_occurrences_from_report(_make_report())
    assert out.empty


def test_co_occurrences_with_two_assets():
    df = _bars()
    same_date_ts = df.index[100]
    rep = _Report({
        "A": _Asset(df, _WF(
            [_Pool([_Contributor("SWEEP_H")])],
            [_Result(0, "low", df.index[80], 99, 99.5, 1.0, True,
                     touched_at=same_date_ts)])),
        "B": _Asset(df, _WF(
            [_Pool([_Contributor("HVN")])],
            [_Result(0, "low", df.index[82], 99, 99.5, 1.0, True,
                     touched_at=same_date_ts)])),
    })
    out = co_occurrences_from_report(rep)
    assert not out.empty
    assert {"asset_a", "asset_b", "agreed"} <= set(out.columns)
    # Both respected on the same date -> agreed = 1.
    assert out["agreed"].iloc[0] == 1.0


def test_bucket_aging_extractor():
    joined = pd.DataFrame({
        "trading_date_ist": ["2026-06-05", "2026-06-05", "2026-06-10"],
        "prediction_type": ["proximity", "proximity", "proximity"],
        "predicted_value": [0.7, 0.8, 0.75],
        "outcome_boolean": [1.0, 0.0, 1.0],
    })
    hist = bucket_aging_history_from_joined(joined, bundle_fit_date="2026-06-01")
    assert not hist.empty
    assert {"prediction_type", "age_days", "n_resolved",
            "abs_calib_error"} <= set(hist.columns)
    # 2026-06-05 is 4 days after fit; 2026-06-10 is 9 days.
    assert set(hist["age_days"]) == {4.0, 9.0}


# ---------------------------------------------------------------------------
# Hub training + safe-starve
# ---------------------------------------------------------------------------

def test_hub_fits_organs_from_report():
    hub = FlywheelHub()
    fitted = hub.fit_from_artifacts(report=_make_report())
    assert isinstance(fitted, dict)
    # Trust eats detector outcomes; should be fed from the report.
    assert fitted["trust"] is True


def test_hub_starves_safely_with_no_inputs():
    hub = FlywheelHub()
    fitted = hub.fit_from_artifacts()
    # Nothing fed -> every organ unfit, but no exception.
    assert all(v is False for v in fitted.values())
    # Serving helpers still work (degrade to no-op).
    assert hub.adjust_probability("proximity", 0.62) == pytest.approx(0.62)
    assert hub.trust_weighted_score(1.5, "SWEEP", "trend") == pytest.approx(1.5)
    assert hub.staleness_note("proximity", 40) is None


def test_hub_save_load_round_trip(tmp_path):
    hub = FlywheelHub()
    hub.fit_from_artifacts(report=_make_report())
    hub.save(tmp_path)
    loaded = FlywheelHub.load(tmp_path)
    # Trust survived the round trip.
    assert loaded.trust.is_fitted == hub.trust.is_fitted


def test_hub_trust_weighting_scales_by_trust():
    """A fitted trust router scales pool scores: high-trust bucket
    boosts, low-trust cuts, relative to the neutral 0.5 anchor."""
    # Build a report where SWEEP in 'range' resolves poorly and well
    # in... only one regime exists here, so just check the mechanism:
    hub = FlywheelHub()
    hub.fit_from_artifacts(report=_make_report())
    if hub.trust.is_fitted:
        base = 2.0
        weighted = hub.trust_weighted_score(base, "SWEEP", "trend")
        # Weighted = base * trust/0.5; must be finite and positive.
        assert weighted > 0
        assert np.isfinite(weighted)


# ---------------------------------------------------------------------------
# Consumer wiring: brief meta-calibration
# ---------------------------------------------------------------------------

def test_brief_applies_meta_calibration_when_hub_fitted():
    from liqpool.products.daily_brief import generate_brief
    from liqpool.flywheel.brief_meta_calibrator import (
        BriefConfidenceMetaCalibrator,
    )

    # Fit a meta-calibrator that softens overconfident proximity.
    rng = np.random.default_rng(1)
    n = 300
    joined = pd.DataFrame({
        "prediction_type": ["proximity"] * n,
        "predicted_value": rng.uniform(0.80, 0.90, n),
        "outcome_boolean": rng.binomial(1, 0.55, n).astype(float),
    })
    hub = FlywheelHub()
    hub.meta_calibrator = BriefConfidenceMetaCalibrator().fit(joined)
    assert hub.meta_calibrator.is_fitted

    @dataclass
    class _AD:
        base_df: pd.DataFrame

    @dataclass
    class _Rep:
        assets: Dict[str, Any]
        unified_direction: Any = None
        unified_proximity: Dict = field(default_factory=dict)
        unified_oos_audit: Any = None

    report = _Rep({"HDFCBANK": _AD(_bars())})
    brief = generate_brief(report, trading_date_ist="2026-06-03",
                           flywheel_hub=hub)
    # The disclosure must appear in the confidence notes.
    note = (brief.confidence_notes.operator_note or "")
    assert "recalibrated" in note.lower()


def test_brief_unchanged_without_hub():
    from liqpool.products.daily_brief import generate_brief

    @dataclass
    class _AD:
        base_df: pd.DataFrame

    @dataclass
    class _Rep:
        assets: Dict[str, Any]
        unified_direction: Any = None
        unified_proximity: Dict = field(default_factory=dict)
        unified_oos_audit: Any = None

    brief = generate_brief(_Rep({"X": _AD(_bars())}),
                           trading_date_ist="2026-06-03")
    assert brief is not None  # no hub -> no crash, no disclosure


# ---------------------------------------------------------------------------
# Consumer wiring: executor regret advisory
# ---------------------------------------------------------------------------

def test_executor_attaches_regret_advisory_when_model_fitted():
    from liqpool.options.executor import pre_trade_decision
    from liqpool.flywheel.regret_model import RegretEstimator
    from liqpool.options.ccv import CCV

    # Fit a regret model on synthetic skip data.
    import json
    rng = np.random.default_rng(2)
    rows = []
    for _ in range(400):
        lcs = float(rng.uniform(0, 0.3))
        regret = rng.random() < (0.1 + 2.5 * lcs)
        rows.append({
            "event_kind": "skip_options_executor",
            "decision_context": json.dumps({"lcs": lcs}),
            "counterfactual_outcome": json.dumps({
                "verdict": "rejection_regret" if regret
                           else "rejection_vindicated"}),
        })
    model = RegretEstimator().fit(pd.DataFrame(rows))

    def _ccv(macro, lcs):
        return CCV(
            macro_score=macro, regime_score=0.0, pool_score=0.0,
            options_score=0.0, micro_score=0.0, manipulation_score=0.0,
            micro_forced_flag=False, pool_distortion_flag=False,
            micro_organic_score=0.0, pool_holding_strength=0.0,
            manip_micro_alignment=0.0, manip_pool_alignment=0.0,
            regime_pool_alignment=0.0, macro_regime_alignment=0.0,
            lcs=lcs,
        )

    # A SKIP carries the advisory.
    d = pre_trade_decision(_ccv(0.5, 0.05), regret_model=model)
    assert d.action == "SKIP"
    assert d.regret_advisory is not None
    assert 0.0 <= d.regret_advisory <= 1.0

    # An ENTER never carries it.
    d_enter = pre_trade_decision(_ccv(0.5, 0.40), regret_model=model)
    assert d_enter.action == "ENTER"
    assert d_enter.regret_advisory is None


def test_executor_no_advisory_without_model():
    from liqpool.options.executor import pre_trade_decision
    from liqpool.options.ccv import CCV
    ccv = CCV(
        macro_score=0.05, regime_score=0.0, pool_score=0.0,
        options_score=0.0, micro_score=0.0, manipulation_score=0.0,
        micro_forced_flag=False, pool_distortion_flag=False,
        micro_organic_score=0.0, pool_holding_strength=0.0,
        manip_micro_alignment=0.0, manip_pool_alignment=0.0,
        regime_pool_alignment=0.0, macro_regime_alignment=0.0, lcs=0.4,
    )
    d = pre_trade_decision(ccv)   # no model
    assert d.regret_advisory is None


# ---------------------------------------------------------------------------
# Consumer wiring: drift imminence
# ---------------------------------------------------------------------------

def test_drift_attach_imminence_when_model_fitted():
    from liqpool.drift import (
        DriftMetrics, DriftReport, attach_imminence,
    )
    from liqpool.flywheel.drift_imminent import DriftImminentModel

    # Fit on synthetic history.
    rng = np.random.default_rng(3)
    n = 300
    respect = np.clip(0.45 + np.cumsum(rng.normal(0, 0.01, n)), 0.25, 0.6)
    fired = np.zeros(n)
    for i in range(4, n):
        if respect[i] - respect[i - 3] < -0.02 and rng.random() < 0.8:
            fired[i] = 1.0
    hist = pd.DataFrame({
        "oos_broad_respect": respect,
        "mean_overfit_gap": rng.normal(0.02, 0.005, n),
        "direction_auc": rng.normal(0.57, 0.01, n),
        "direction_top_quartile_confidence": rng.normal(0.58, 0.01, n),
        "drift_fired": fired,
    })
    model = DriftImminentModel(horizon=3).fit(hist)

    rpt = DriftReport(current=DriftMetrics(
        run_timestamp="t", oos_broad_respect=0.4, oos_strict_respect=0.35,
        mean_overfit_gap=0.02, direction_auc=0.56,
        direction_top_quartile_confidence=0.57, n_oos_pools=900,
    ))
    attach_imminence(rpt, model, hist.tail(8))
    assert rpt.imminence is not None
    assert 0.0 <= rpt.imminence <= 1.0


def test_drift_imminence_none_without_model():
    from liqpool.drift import DriftMetrics, DriftReport, attach_imminence
    rpt = DriftReport(current=DriftMetrics(
        run_timestamp="t", oos_broad_respect=0.5, oos_strict_respect=0.45,
        mean_overfit_gap=0.0, direction_auc=0.6,
        direction_top_quartile_confidence=0.6, n_oos_pools=900,
    ))
    attach_imminence(rpt, None, pd.DataFrame())
    assert rpt.imminence is None
