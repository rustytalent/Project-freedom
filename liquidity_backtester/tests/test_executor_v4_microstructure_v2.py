"""Microstructure pipeline + 5 new detectors + ShadowLedger + rehearsal pre-calibration.

Founder 2026-06-22 follow-up: "we have L2 + OI from Kite, use it.
Build shadow ledger. Pre-calibrate from the rehearsal layer." This is
that commit.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter, PersistenceConfig, V4Runner, V4RunnerConfig,
)
from liqpool.research.belief.executor_v4.live_data_v2 import (
    CancelRateTracker, DepthLevel, EnrichedQuote,
    OIHistoryTracker, build_microstructure_features,
    enrich_kite_quote, extract_pressure_ratio,
    slot_reading_from_enriched,
)
from liqpool.research.belief.executor_v4.manipulation_v2 import (
    CancelRateDetector, DepthPressureDetector, IcebergDetector,
    LayeringDetector, ManipulationEngineV2, OIVelocityDetector,
)
from liqpool.research.belief.executor_v4.manipulation_v2.detectors.base import (
    DetectorPosterior,
)
from liqpool.research.belief.executor_v4.manipulation_v2.rehearsal_calibration import (
    RehearsalCalibrationConfig, RehearsalCalibrator,
)
from liqpool.research.belief.executor_v4.shadow_ledger import (
    ShadowLedger, ShadowLedgerConfig,
)


# ── Enriched quote / pipeline ─────────────────────────────────


def test_enrich_kite_quote_captures_depth_and_oi():
    raw = {
        "last_price": 100.5, "last_quantity": 50, "volume": 1500000,
        "oi": 2_500_000, "oi_day_high": 2_600_000,
        "buy_quantity": 12000, "sell_quantity": 8000,
        "depth": {
            "buy":  [{"price": 100.0, "quantity": 500, "orders": 5},
                       {"price": 99.5,  "quantity": 700, "orders": 7}],
            "sell": [{"price": 101.0, "quantity": 300, "orders": 3}],
        },
    }
    q = enrich_kite_quote(raw, tradingsymbol="NIFTY23000CE")
    assert q.bid == 100.0
    assert q.ask == 101.0
    assert q.spread == 1.0
    assert q.oi == 2_500_000
    assert q.buy_quantity_total == 12000
    assert len(q.depth_buy) == 2


def test_pressure_ratio_signed():
    q = EnrichedQuote(
        tradingsymbol="x", ltp=100.0, last_quantity=0,
        buy_quantity_total=3000, sell_quantity_total=1000)
    assert extract_pressure_ratio(q) == pytest.approx(0.5)


def test_oi_history_velocity_after_two_samples():
    tr = OIHistoryTracker(history_bars=10)
    r1 = tr.observe(strike=23000.0, option_type="CE", oi=1_000_000.0,
                       ts=0.0)
    assert r1.oi_velocity_per_min == 0.0
    r2 = tr.observe(strike=23000.0, option_type="CE", oi=1_120_000.0,
                       ts=60.0)
    assert r2.oi_velocity_per_min == pytest.approx(120_000.0)


def test_cancel_rate_detects_persistent_book():
    tr = CancelRateTracker(persistence_threshold_ticks=2)
    db = [{"price": 100.0, "quantity": 500},
            {"price": 99.5, "quantity": 700}]
    ds = [{"price": 101.0, "quantity": 300}]
    for _ in range(5):
        r = tr.observe(tradingsymbol="X", depth_buy=db, depth_sell=ds)
    assert r.persistence_score > 0.5


def test_cancel_rate_detects_spoofy_book():
    tr = CancelRateTracker(persistence_threshold_ticks=3)
    # Every tick the buy/sell prices change → high cancel rate.
    for i in range(10):
        tr.observe(
            tradingsymbol="X",
            depth_buy=[{"price": 100.0 + 0.1 * i, "quantity": 50}],
            depth_sell=[{"price": 101.0 + 0.1 * i, "quantity": 50}],
        )
    r = tr.observe(
        tradingsymbol="X",
        depth_buy=[{"price": 200.0, "quantity": 50}],
        depth_sell=[{"price": 201.0, "quantity": 50}],
    )
    assert r.persistence_score <= 0.5
    assert r.cancel_rate_per_tick > 0.0


def test_slot_reading_from_enriched_carries_v2_fields():
    q = enrich_kite_quote({
        "last_price": 80.0, "last_quantity": 100,
        "oi": 1_000_000, "buy_quantity": 5000, "sell_quantity": 4000,
        "depth": {"buy": [{"price": 80.0, "quantity": 200}],
                    "sell": [{"price": 80.5, "quantity": 150}]},
    }, tradingsymbol="NIFTY23000CE")
    s = slot_reading_from_enriched(
        q, strike=23000.0, option_type="CE", level=0,
        label="CE_ATM", dod_z=0.5)
    for key in ("depth_buy_levels", "depth_sell_levels", "oi",
                  "buy_quantity_pending", "sell_quantity_pending",
                  "last_trade_size"):
        assert key in s


# ── New detectors ────────────────────────────────────────────


def test_oi_velocity_fires_on_defended_ceiling():
    det = OIVelocityDetector(velocity_threshold_per_min=100.0)
    # Seed spot history rising.
    for i in range(6):
        det.observe(snapshot={"spot": 23000.0 + 10 * i,
                                 "slot_readings": []},
                       bar_index=i)
    out = det.observe(
        snapshot={"spot": 23080.0,
                   "slot_readings": [
                       {"option_type": "CE",
                         "strike": 23100.0,
                         "oi_velocity_per_min": 500.0},
                   ]},
        bar_index=6)
    assert out.fire
    assert out.direction == -1
    assert "ceiling" in out.classification


def test_depth_pressure_fires_on_persistent_imbalance():
    det = DepthPressureDetector(
        persistence_bars=3, pressure_threshold=0.20,
        min_persistence_score=0.20)
    base_slots = [
        {"option_type": "CE", "strike": 23000.0,
          "buy_quantity_pending": 4000,
          "sell_quantity_pending": 1000,
          "persistence_score": 0.80},
    ]
    out = None
    for i in range(6):
        out = det.observe(snapshot={"slot_readings": base_slots},
                            bar_index=i)
    assert out is not None
    assert out.fire
    assert out.direction == 1


def test_layering_fires_only_when_persistent():
    det = LayeringDetector(min_slots_layered=1,
                              min_persistence_score=0.50,
                              meaningful_level_floor=50)
    slots = [{
        "strike": 23000.0, "option_type": "CE",
        "persistence_score": 0.90,
        "depth_buy_levels": [
            {"price": 100.0, "quantity": 200},
            {"price": 99.5, "quantity": 300},
            {"price": 99.0, "quantity": 200}],
        "depth_sell_levels": [{"price": 101.0, "quantity": 10}],
    }]
    out = det.observe(snapshot={"slot_readings": slots}, bar_index=0)
    assert out.fire
    assert out.direction == -1     # buy-side layering → push DOWN
    assert "buy_side_layering" in out.classification

    # Same scenario but low persistence → ignored.
    slots_spoof = [{
        **slots[0], "persistence_score": 0.10,
    }]
    out2 = det.observe(snapshot={"slot_readings": slots_spoof},
                          bar_index=1)
    assert out2.fire is False


def test_iceberg_fires_on_recurrent_fills():
    det = IcebergDetector(history_ticks=8, min_recurrences=3,
                              size_tolerance_pct=0.10,
                              min_persistence_score=0.30)
    slots = [{
        "strike": 23000.0, "option_type": "CE", "label": "NIFTY23000CE",
        "persistence_score": 0.60,
        "ltp": 100.0, "last_trade_size": 100,
    }]
    out = None
    for i in range(5):
        out = det.observe(snapshot={"slot_readings": slots},
                            bar_index=i)
    assert out is not None
    assert out.fire
    assert out.direction == 1     # iceberg on CE → upside bid


def test_cancel_rate_classifies_spoof_vs_clean():
    det = CancelRateDetector(spoof_persistence_ceiling=0.30,
                                  clean_persistence_floor=0.70)
    # Spoof-heavy book.
    out = det.observe(snapshot={"slot_readings": [
        {"persistence_score": 0.10, "cancel_rate_per_tick": 2.5},
        {"persistence_score": 0.12, "cancel_rate_per_tick": 2.0},
    ]}, bar_index=0)
    assert out.fire
    assert out.classification == "spoofing_dominant"
    # Clean book.
    out2 = det.observe(snapshot={"slot_readings": [
        {"persistence_score": 0.85, "cancel_rate_per_tick": 0.10},
        {"persistence_score": 0.90, "cancel_rate_per_tick": 0.05},
    ]}, bar_index=1)
    assert out2.fire
    assert out2.classification == "clean_book"


def test_engine_has_12_detectors():
    e = ManipulationEngineV2()
    assert len(e.detectors) == 12
    for name in ("oi_velocity", "depth_pressure", "layering",
                  "iceberg", "cancel_rate"):
        assert name in e.detectors


# ── ShadowLedger ─────────────────────────────────────────────


def test_shadow_ledger_writes_jsonl_with_flush():
    tmp = Path(tempfile.mkdtemp(prefix="sl_"))
    sl = ShadowLedger(state_dir=tmp,
                          cfg=ShadowLedgerConfig(flush_every_n_events=2))
    sl.record_tick(bar_index=1, snapshot={"ts": "x", "spot": 100.0,
                                                "slot_readings": []})
    sl.record_mm_intent(bar_index=1,
                              mm_intent_dict={"direction": 1})
    # Should auto-flush.
    path = sl.path_for_today()
    assert path.exists()
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 2
    assert {l["kind"] for l in lines} == {"tick", "mm_intent"}


def test_shadow_ledger_disabled_when_cfg_off():
    tmp = Path(tempfile.mkdtemp(prefix="sl_off_"))
    sl = ShadowLedger(state_dir=tmp,
                          cfg=ShadowLedgerConfig(enabled=False))
    sl.record_tick(bar_index=1, snapshot={})
    sl.flush()
    path = sl.path_for_today()
    assert not path.exists()


def test_shadow_ledger_summary_includes_counters():
    tmp = Path(tempfile.mkdtemp(prefix="sl_sum_"))
    sl = ShadowLedger(state_dir=tmp)
    sl.record_tick(bar_index=1, snapshot={"slot_readings": []})
    sl.record_close(bar_index=2, position_id="p1",
                        outcome_dict={"realized_r": 0.5})
    s = sl.summary()
    assert s["counters"]["ticks"] == 1
    assert s["counters"]["closes"] == 1


# ── RehearsalCalibrator ──────────────────────────────────────


class _FakePerturbation:
    def __init__(self, name, mean_r):
        self.name = name
        self.mean_r = mean_r


class _FakeRehearsalDecision:
    def __init__(self, ran, confidence, n_analogues,
                  current_perturbation_name="as_proposed",
                  per_perturbation=None):
        self.ran = ran
        self.confidence = confidence
        self.n_analogues_total = n_analogues
        self.current_perturbation_name = current_perturbation_name
        self.per_perturbation = per_perturbation or [
            _FakePerturbation("as_proposed", 0.5)]


class _FakeCalibrator:
    def __init__(self):
        self.calls = []

    def record_outcome(self, *, per_detector_posteriors, realised_r):
        self.calls.append((per_detector_posteriors, realised_r))


def _post(direction=1, fire=True, confidence=0.7):
    return DetectorPosterior(
        fire=fire, probability=0.6, direction=direction,
        confidence=confidence)


def test_rehearsal_calibrator_pushes_virtual_when_rehearsal_confident():
    rc = RehearsalCalibrator(cfg=RehearsalCalibrationConfig(
        virtual_weight=0.25, min_rehearsal_confidence=0.30,
        min_analogues=4,
    ))
    fake = _FakeCalibrator()
    dec = _FakeRehearsalDecision(
        ran=True, confidence=0.65, n_analogues=12,
        per_perturbation=[_FakePerturbation("as_proposed", 0.8)],
    )
    fired = rc.maybe_calibrate(
        self_calibrator=fake,
        per_detector_posteriors={"a": _post()},
        mm_intent_direction=1, mm_intent_confidence=0.6,
        rehearsal_decision=dec,
    )
    assert fired is True
    assert len(fake.calls) == 1
    # virtual_r = 0.8 × 0.25 = 0.2
    assert fake.calls[0][1] == pytest.approx(0.2)


def test_rehearsal_calibrator_skips_when_intent_neutral():
    rc = RehearsalCalibrator()
    fake = _FakeCalibrator()
    dec = _FakeRehearsalDecision(ran=True, confidence=0.7, n_analogues=10)
    fired = rc.maybe_calibrate(
        self_calibrator=fake,
        per_detector_posteriors={"a": _post()},
        mm_intent_direction=0, mm_intent_confidence=0.0,
        rehearsal_decision=dec,
    )
    assert fired is False


def test_rehearsal_calibrator_skips_when_rehearsal_thin():
    rc = RehearsalCalibrator(cfg=RehearsalCalibrationConfig(
        min_rehearsal_confidence=0.50, min_analogues=10,
    ))
    fake = _FakeCalibrator()
    dec = _FakeRehearsalDecision(ran=True, confidence=0.20, n_analogues=4)
    fired = rc.maybe_calibrate(
        self_calibrator=fake,
        per_detector_posteriors={"a": _post()},
        mm_intent_direction=1, mm_intent_confidence=0.7,
        rehearsal_decision=dec,
    )
    assert fired is False


# ── Manager wiring ───────────────────────────────────────────


def test_manager_has_shadow_ledger_and_rehearsal_calibrator():
    tmp = Path(tempfile.mkdtemp(prefix="msr_"))
    runner = V4Runner(
        cfg=V4RunnerConfig(
            persistence=PersistenceConfig(state_dir=tmp, enabled=True),
            emit_explainer_to_log=False,
        ),
        broker=PaperBrokerAdapter(),
    )
    assert runner.manager.shadow_ledger is not None
    assert runner.manager.rehearsal_calibrator is not None
    assert len(runner.manager.manipulation_v2.detectors) == 12


def test_manager_writes_shadow_ledger_on_ticks():
    tmp = Path(tempfile.mkdtemp(prefix="msr_tape_"))
    runner = V4Runner(
        cfg=V4RunnerConfig(
            persistence=PersistenceConfig(state_dir=tmp, enabled=True),
            emit_explainer_to_log=False,
        ),
        broker=PaperBrokerAdapter(),
    )
    from tests.test_executor_v4_dead_market_and_web_viz import _snap
    for i in range(60):
        runner.on_tick(_snap(100 + i))
    runner.manager.shadow_ledger.flush()
    path = runner.manager.shadow_ledger.path_for_today()
    assert path.exists()
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) > 30
