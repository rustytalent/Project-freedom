"""Tests for executor_v4.projection — forward Bayesian conditional distributions."""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4.projection import (
    ForwardProjection,
    ProjectionConfig,
    ProjectionRecord,
)


def _mk_rec(*, ts: pd.Timestamp, bar: int,
             thesis: str = "HOLD_BULL",
             iv: str = "directional_bull",
             bf: str = "bullish_agreement",
             winding: str = "NO_WINDING",
             direction: int = 1,
             regime: float = 0.80,
             realized_r: float = 1.0,
             bars_to_res: int = 12,
             via_target: bool = True,
             via_stop: bool = False,
             via_neither: bool = False) -> ProjectionRecord:
    return ProjectionRecord(
        ts=ts, bar_index=bar, thesis_state=thesis,
        iv_state=iv, battlefield_verdict=bf,
        winding_zone=winding, direction=direction,
        regime_stability=regime,
        bull_score=70.0, bear_score=10.0, net_intent_z=1.5,
        realized_premium_change_pct=0.10,
        realized_r=realized_r, bars_to_resolution=bars_to_res,
        closed_via_target=via_target, closed_via_stop=via_stop,
        closed_via_neither=via_neither,
    )


def test_projection_empty_returns_zero_confidence():
    pj = ForwardProjection()
    dist = pj.project(thesis_state="HOLD_BULL", iv_state="directional_bull",
                       battlefield_verdict="bullish_agreement",
                       winding_zone="NO_WINDING", direction=1,
                       regime_stability=0.80)
    assert dist.n_samples == 0
    assert dist.confidence == 0.0


def test_projection_records_and_queries_matching():
    pj = ForwardProjection()
    base_ts = pd.Timestamp("2026-06-15 10:00")
    for i in range(25):
        pj.record(_mk_rec(ts=base_ts + pd.Timedelta(minutes=i),
                            bar=i, realized_r=1.2, via_target=True))
    dist = pj.project(thesis_state="HOLD_BULL", iv_state="directional_bull",
                       battlefield_verdict="bullish_agreement",
                       winding_zone="NO_WINDING", direction=1,
                       regime_stability=0.80)
    assert dist.n_samples == 25
    assert dist.confidence > 0.9
    assert dist.p_target_hit_first == 1.0
    assert dist.p_stop_hit_first == 0.0
    assert dist.mean_r > 1.0


def test_projection_handles_mixed_outcomes():
    pj = ForwardProjection()
    base_ts = pd.Timestamp("2026-06-15 10:00")
    # 14 winners, 6 losers, 5 neither = 25 total
    for i in range(14):
        pj.record(_mk_rec(ts=base_ts + pd.Timedelta(minutes=i),
                            bar=i, realized_r=1.5, via_target=True))
    for i in range(6):
        pj.record(_mk_rec(ts=base_ts + pd.Timedelta(minutes=14 + i),
                            bar=14 + i, realized_r=-1.0,
                            via_target=False, via_stop=True))
    for i in range(5):
        pj.record(_mk_rec(ts=base_ts + pd.Timedelta(minutes=20 + i),
                            bar=20 + i, realized_r=0.2,
                            via_target=False, via_stop=False,
                            via_neither=True))
    dist = pj.project(thesis_state="HOLD_BULL", iv_state="directional_bull",
                       battlefield_verdict="bullish_agreement",
                       winding_zone="NO_WINDING", direction=1,
                       regime_stability=0.80)
    assert dist.n_samples == 25
    assert abs(dist.p_target_hit_first - 0.56) < 0.05
    assert abs(dist.p_stop_hit_first - 0.24) < 0.05
    assert dist.p_continuation > dist.p_reversal


def test_projection_low_samples_neutral_confidence_note():
    pj = ForwardProjection(ProjectionConfig(min_samples_for_confidence=30))
    base_ts = pd.Timestamp("2026-06-15 10:00")
    for i in range(5):
        pj.record(_mk_rec(ts=base_ts + pd.Timedelta(minutes=i),
                            bar=i, realized_r=1.0))
    dist = pj.project(thesis_state="HOLD_BULL", iv_state="directional_bull",
                       battlefield_verdict="bullish_agreement",
                       winding_zone="NO_WINDING", direction=1,
                       regime_stability=0.80)
    assert dist.n_samples == 5
    assert dist.confidence < 1.0
    assert any("similar historical records" in n for n in dist.notes)


def test_projection_dissimilar_context_no_match():
    pj = ForwardProjection()
    base_ts = pd.Timestamp("2026-06-15 10:00")
    for i in range(25):
        pj.record(_mk_rec(ts=base_ts + pd.Timedelta(minutes=i),
                            bar=i, realized_r=1.0))
    # Query a completely different context — no matches.
    dist = pj.project(thesis_state="HOLD_BEAR", iv_state="dirty_data",
                       battlefield_verdict="bearish_agreement",
                       winding_zone="STRONG_BEAR_TRAP_WINDING",
                       direction=-1, regime_stability=0.20)
    # similarity_threshold = 0.55 default; near-zero overlap should miss.
    assert dist.n_samples == 0


def test_projection_thesis_family_fallback():
    pj = ForwardProjection()
    base_ts = pd.Timestamp("2026-06-15 10:00")
    for i in range(25):
        # All records HOLD_BULL.
        pj.record(_mk_rec(ts=base_ts + pd.Timedelta(minutes=i),
                            bar=i, thesis="HOLD_BULL", realized_r=1.0))
    # Query BULL_ENTRY — same family, should still match (partial credit).
    dist = pj.project(thesis_state="BULL_ENTRY", iv_state="directional_bull",
                       battlefield_verdict="bullish_agreement",
                       winding_zone="NO_WINDING", direction=1,
                       regime_stability=0.80)
    assert dist.n_samples >= 1


def test_projection_serializable():
    pj = ForwardProjection()
    base_ts = pd.Timestamp("2026-06-15 10:00")
    for i in range(25):
        pj.record(_mk_rec(ts=base_ts + pd.Timedelta(minutes=i),
                            bar=i, realized_r=1.0))
    dist = pj.project(thesis_state="HOLD_BULL", iv_state="directional_bull",
                       battlefield_verdict="bullish_agreement",
                       winding_zone="NO_WINDING", direction=1,
                       regime_stability=0.80)
    import json
    json.dumps(dist.to_dict())
    rec = pj.tape[0]
    json.dumps(rec.to_dict(), default=str)
    json.dumps(pj.summary())


def test_projection_history_cap_evicts_old():
    pj = ForwardProjection(ProjectionConfig(history_cap=10))
    base_ts = pd.Timestamp("2026-06-15 10:00")
    for i in range(30):
        pj.record(_mk_rec(ts=base_ts + pd.Timedelta(minutes=i),
                            bar=i, realized_r=1.0))
    assert len(pj.tape) == 10
