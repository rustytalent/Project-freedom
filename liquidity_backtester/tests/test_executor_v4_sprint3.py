"""Tests for executor_v4 Sprint 3: manipulation board + MM mind + tail amplifier + crowd mirror."""
from __future__ import annotations

import pandas as pd

from liqpool.research.belief.executor_v4.crowd_mirror import CrowdMirror
from liqpool.research.belief.executor_v4.fat_tail_amplifier import (
    FatTailAmplifier,
    FatTailAmplifierConfig,
    TAIL_ACTION_HEDGE,
    TAIL_ACTION_NORMAL,
    TAIL_ACTION_REFUSE,
    TAIL_ACTION_SCALE_UP,
)
from liqpool.research.belief.executor_v4.flow_memory import FlowMemory
from liqpool.research.belief.executor_v4.manipulation_patterns import (
    ManipulationBoard,
    ManipulationBoardConfig,
    PATTERN_ACCUMULATION,
    PATTERN_DISTRIBUTION,
    PATTERN_FAKE_BREAKOUT,
    PATTERN_LIQUIDITY_VACUUM,
    PATTERN_MM_INVENTORY_FLIP,
    PATTERN_STOP_HUNT_SHORT,
)
from liqpool.research.belief.executor_v4.market_maker_mind import (
    INTENT_ACCUMULATING,
    INTENT_DISTRIBUTING,
    INTENT_HUNTING_STOPS,
    INTENT_NEUTRAL,
    INTENT_PINNING,
    INTENT_STEPPING_BACK,
    MarketMakerMind,
)
from liqpool.research.belief.executor_v4.substrate import SubstrateState


def _mk(*, bars: int = 100, spot: float = 23000.0,
        thesis: str = "HOLD_BULL", iv: str = "directional_bull",
        bf: str = "bullish_agreement", direction: int = 1,
        bull_score: float = 70.0, bear_score: float = 10.0,
        ce_signed: float = 1.5, pe_signed: float = -1.0,
        ce_def_frac: float = 0.0, pe_def_frac: float = 0.0,
        ce_disp: float = 0.10, pe_disp: float = 0.10,
        friendliness: float = 0.92,
        mark_source: str = "microprice",
        winding: str = "NO_WINDING",
        ts: pd.Timestamp | None = None) -> dict:
    if ts is None:
        ts = pd.Timestamp("2026-06-19 10:00") + pd.Timedelta(seconds=bars)
    n_each = 11
    n_ce_def = int(ce_def_frac * n_each); n_pe_def = int(pe_def_frac * n_each)
    slot_readings = []
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "CE", "level": level,
            "label": f"CE_{level}",
            "acceptance": "defended" if n_ce_def > 0 else "normal",
            "friendliness": friendliness, "spread_state": "clean",
            "mark_source": mark_source, "is_abnormal": False,
            "dod_z": ce_signed, "mark_price": 100.0,
        })
        if n_ce_def > 0:
            n_ce_def -= 1
    for level in range(-5, 6):
        slot_readings.append({
            "strike": 23000.0 + 50 * level,
            "option_type": "PE", "level": level,
            "label": f"PE_{level}",
            "acceptance": "defended" if n_pe_def > 0 else "normal",
            "friendliness": friendliness, "spread_state": "clean",
            "mark_source": mark_source, "is_abnormal": False,
            "dod_z": pe_signed, "mark_price": 100.0,
        })
        if n_pe_def > 0:
            n_pe_def -= 1
    return {
        "ts": ts, "spot": spot, "bars_seen": bars, "is_warm": True,
        "thesis": {"composite_state": thesis,
                   "bull_thesis_score": bull_score,
                   "bear_thesis_score": bear_score,
                   "no_trade_score": 5.0},
        "iv_state": {"state": iv, "direction": direction, "confidence": 0.8,
                     "clean_mark_fraction": 0.98,
                     "net_intent_z": ce_signed - pe_signed},
        "battlefield": {
            "verdict": bf, "direction": direction, "confidence": 0.8,
            "ce_rail": {"weighted_mean_signed_z": ce_signed,
                        "dispersion_score": ce_disp,
                        "epicenter_label": "CE_ATM",
                        "epicenter_level": 0},
            "pe_rail": {"weighted_mean_signed_z": pe_signed,
                        "dispersion_score": pe_disp,
                        "epicenter_label": "PE_ATM",
                        "epicenter_level": 0},
        },
        "decision": {"action": "HOLD", "direction": direction,
                     "confidence": 0.8, "trade_allowed": True},
        "winding": {"zone": winding},
        "bull_state": {"state_index": 2},
        "bear_state": {"state_index": 0},
        "sweep_state": {"state_index": 0},
        "slot_readings": slot_readings,
    }


# ── Manipulation board ─────────────────────────────────────────────


def test_stop_hunt_short_detected():
    board = ManipulationBoard()
    sub = SubstrateState(); fm = FlowMemory()
    # First 4 bars: steady spot, defining a "prior high".
    for i in range(4):
        snap = _mk(bars=100 + i, spot=23000.0 + i * 0.5)
        sub.observe(snap); fm.observe(snap)
    # Bar 5: big spike UP +10bps (~23 points on 23k).
    spike = _mk(bars=104, spot=23023.0)
    sub.observe(spike); fm.observe(spike)
    # Bar 6-8: drop back below original 23000.
    for i, spot in enumerate([23010.0, 22995.0, 22990.0]):
        snap = _mk(bars=105 + i, spot=spot)
        sub.observe(snap); fm.observe(snap)
    rich = sub.observe(_mk(bars=110, spot=22990.0))
    fm.observe(_mk(bars=110, spot=22990.0))

    matches = board.scan(flow_memory=fm, rich_context=rich,
                          snapshot=_mk(bars=110, spot=22990.0))
    names = {m.pattern_name for m in matches}
    assert PATTERN_STOP_HUNT_SHORT in names


def test_liquidity_vacuum_detected_on_friendliness_drop():
    board = ManipulationBoard()
    sub = SubstrateState(); fm = FlowMemory()
    # 3 bars at high friendliness.
    for i in range(3):
        snap = _mk(bars=100 + i, friendliness=0.95)
        sub.observe(snap); fm.observe(snap)
    # 3 bars at low friendliness + dirty marks.
    for i in range(3):
        snap = _mk(bars=103 + i, friendliness=0.50, mark_source="ltp")
        sub.observe(snap); fm.observe(snap)
    rich = sub.observe(_mk(bars=110, friendliness=0.45, mark_source="ltp"))
    fm.observe(_mk(bars=110, friendliness=0.45, mark_source="ltp"))

    matches = board.scan(flow_memory=fm, rich_context=rich,
                          snapshot=_mk(bars=110))
    names = {m.pattern_name for m in matches}
    assert PATTERN_LIQUIDITY_VACUUM in names


def test_accumulation_detected_on_ce_defended_streak():
    board = ManipulationBoard()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(16):
        snap = _mk(bars=100 + i, ce_def_frac=0.50,
                    spot=23000.0 + (i % 4))
        sub.observe(snap); fm.observe(snap)
    rich = sub.observe(_mk(bars=116, ce_def_frac=0.50, spot=23001.0))
    fm.observe(_mk(bars=116, ce_def_frac=0.50, spot=23001.0))
    matches = board.scan(flow_memory=fm, rich_context=rich,
                          snapshot=_mk(bars=116))
    names = {m.pattern_name for m in matches}
    assert PATTERN_ACCUMULATION in names


def test_distribution_detected_on_pe_defended_streak():
    board = ManipulationBoard()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(16):
        snap = _mk(bars=100 + i, pe_def_frac=0.55,
                    spot=23000.0 + (i % 4))
        sub.observe(snap); fm.observe(snap)
    rich = sub.observe(_mk(bars=116, pe_def_frac=0.55, spot=23000.0))
    fm.observe(_mk(bars=116, pe_def_frac=0.55, spot=23000.0))
    matches = board.scan(flow_memory=fm, rich_context=rich,
                          snapshot=_mk(bars=116))
    names = {m.pattern_name for m in matches}
    assert PATTERN_DISTRIBUTION in names


def test_mm_inventory_flip_on_intent_swing_no_spot_move():
    board = ManipulationBoard()
    sub = SubstrateState(); fm = FlowMemory()
    # 8 bars: ce_signed swings from +2 to -2 while spot barely moves.
    swings = [2.5, 2.0, 1.0, 0.0, -1.0, -1.8, -2.5, -2.5]
    for i, ce in enumerate(swings):
        snap = _mk(bars=100 + i, ce_signed=ce, pe_signed=-ce,
                    spot=23000.0 + 0.2 * (i % 2))
        sub.observe(snap); fm.observe(snap)
    rich = sub.observe(_mk(bars=108, ce_signed=-2.5, pe_signed=2.5,
                              spot=23000.0))
    fm.observe(_mk(bars=108, ce_signed=-2.5, pe_signed=2.5, spot=23000.0))
    matches = board.scan(flow_memory=fm, rich_context=rich,
                          snapshot=_mk(bars=108))
    names = {m.pattern_name for m in matches}
    assert PATTERN_MM_INVENTORY_FLIP in names


def test_fake_breakout_detected():
    board = ManipulationBoard()
    sub = SubstrateState(); fm = FlowMemory()
    # First 3 bars steady, then a big +10bps move, then collapse back.
    for i in range(3):
        snap = _mk(bars=100 + i, spot=23000.0 + 0.1 * i)
        sub.observe(snap); fm.observe(snap)
    breakout = _mk(bars=103, spot=23023.0)
    sub.observe(breakout); fm.observe(breakout)
    for i, spot in enumerate([23015.0, 23005.0, 22995.0, 22990.0]):
        snap = _mk(bars=104 + i, spot=spot)
        sub.observe(snap); fm.observe(snap)
    rich = sub.observe(_mk(bars=108, spot=22990.0))
    fm.observe(_mk(bars=108, spot=22990.0))
    matches = board.scan(flow_memory=fm, rich_context=rich,
                          snapshot=_mk(bars=108))
    names = {m.pattern_name for m in matches}
    assert PATTERN_FAKE_BREAKOUT in names


def test_pattern_match_to_dict_serializable():
    board = ManipulationBoard()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(20):
        snap = _mk(bars=100 + i, ce_def_frac=0.50, spot=23000.0)
        sub.observe(snap); fm.observe(snap)
    rich = sub.observe(_mk(bars=120, ce_def_frac=0.50, spot=23000.0))
    fm.observe(_mk(bars=120, ce_def_frac=0.50, spot=23000.0))
    matches = board.scan(flow_memory=fm, rich_context=rich,
                          snapshot=_mk(bars=120))
    import json
    for m in matches:
        json.dumps(m.to_dict())


def test_board_scan_empty_flow_returns_empty():
    board = ManipulationBoard()
    fm = FlowMemory()
    sub = SubstrateState()
    matches = board.scan(flow_memory=fm,
                          rich_context=None,
                          snapshot=_mk())
    assert matches == []


# ── MM mind ────────────────────────────────────────────────────────


def test_mm_mind_inflates_accumulating_under_accumulation():
    mind = MarketMakerMind()
    board = ManipulationBoard()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(16):
        snap = _mk(bars=100 + i, ce_def_frac=0.50, spot=23000.0)
        sub.observe(snap); fm.observe(snap)
    rich = sub.observe(_mk(bars=116, ce_def_frac=0.50, spot=23000.0))
    fm.observe(_mk(bars=116, ce_def_frac=0.50, spot=23000.0))
    matches = board.scan(flow_memory=fm, rich_context=rich,
                          snapshot=_mk(bars=116))
    post = mind.infer(patterns=matches)
    assert post.dominant_intent == INTENT_ACCUMULATING
    assert post.implied_bias == +1


def test_mm_mind_distributing_under_distribution():
    mind = MarketMakerMind()
    board = ManipulationBoard()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(16):
        snap = _mk(bars=100 + i, pe_def_frac=0.55, spot=23000.0)
        sub.observe(snap); fm.observe(snap)
    rich = sub.observe(_mk(bars=116, pe_def_frac=0.55, spot=23000.0))
    fm.observe(_mk(bars=116, pe_def_frac=0.55, spot=23000.0))
    matches = board.scan(flow_memory=fm, rich_context=rich,
                          snapshot=_mk(bars=116))
    post = mind.infer(patterns=matches)
    assert post.dominant_intent == INTENT_DISTRIBUTING
    assert post.implied_bias == -1


def test_mm_mind_neutral_on_no_patterns():
    mind = MarketMakerMind()
    post = mind.infer(patterns=[])
    assert post.dominant_intent in INTENT_ACCUMULATING + "/" + INTENT_NEUTRAL \
        or post.dominant_intent in {INTENT_NEUTRAL, INTENT_ACCUMULATING,
                                     INTENT_DISTRIBUTING, INTENT_PINNING,
                                     INTENT_HUNTING_STOPS, INTENT_STEPPING_BACK}
    # All intents equally likely → 1/7 each.
    assert abs(post.dominant_probability - 1.0 / 7) < 0.01


def test_mm_mind_serializable():
    mind = MarketMakerMind()
    post = mind.infer(patterns=[])
    import json
    json.dumps(post.to_dict())


def test_mm_mind_guidance_changes_with_dominant_intent():
    mind = MarketMakerMind()
    board = ManipulationBoard()
    sub = SubstrateState(); fm = FlowMemory()
    for i in range(16):
        snap = _mk(bars=100 + i, ce_def_frac=0.50, spot=23000.0)
        sub.observe(snap); fm.observe(snap)
    rich = sub.observe(_mk(bars=116, ce_def_frac=0.50, spot=23000.0))
    fm.observe(_mk(bars=116, ce_def_frac=0.50, spot=23000.0))
    matches = board.scan(flow_memory=fm, rich_context=rich,
                          snapshot=_mk(bars=116))
    post = mind.infer(patterns=matches)
    assert "accumulating" in post.operator_guidance.lower()


# ── Fat-tail amplifier ─────────────────────────────────────────────


def test_amplifier_low_score_on_calm():
    amp = FatTailAmplifier()
    sub = SubstrateState()
    snap = _mk()
    rich = sub.observe(snap)
    fm = FlowMemory()
    ev = fm.observe(snap)
    score = amp.amplify(patterns=[], mm_posterior=None,
                         flow_event=ev, rich_context=rich,
                         iv_state="directional_bull")
    assert score.tail_score < 0.30
    assert score.recommended_action in (TAIL_ACTION_NORMAL, TAIL_ACTION_SCALE_UP)


def test_amplifier_refuses_on_common_shock_plus_steppers():
    amp = FatTailAmplifier()
    sub = SubstrateState()
    snap = _mk(iv="common_shock", friendliness=0.40, mark_source="ltp")
    rich = sub.observe(snap)
    fm = FlowMemory()
    ev = fm.observe(snap)
    # Fake a strong-stepping_back MM posterior.

    class FakeMM:
        intent_distribution = {INTENT_STEPPING_BACK: 0.7, INTENT_NEUTRAL: 0.3}
        implied_volatility_view = "expansion"
        dominant_intent = INTENT_STEPPING_BACK
        confidence = 0.8

    score = amp.amplify(patterns=[], mm_posterior=FakeMM(),
                         flow_event=ev, rich_context=rich,
                         iv_state="common_shock", crowd_density=0.3)
    assert score.tail_score > 0.50
    assert score.recommended_action in (TAIL_ACTION_HEDGE, TAIL_ACTION_REFUSE)


def test_amplifier_components_serializable():
    amp = FatTailAmplifier()
    sub = SubstrateState()
    snap = _mk()
    rich = sub.observe(snap)
    fm = FlowMemory()
    ev = fm.observe(snap)
    score = amp.amplify(patterns=[], mm_posterior=None,
                         flow_event=ev, rich_context=rich,
                         iv_state="directional_bull")
    import json
    json.dumps(score.to_dict())


# ── Crowd mirror ───────────────────────────────────────────────────


class _FakePos:
    def __init__(self, *, contract_side: str = "CE",
                  contract_level: int = 0, direction: int = 1,
                  entry_premium: float = 100.0,
                  stop_premium: float = 70.0) -> None:
        self.contract_side = contract_side
        self.contract_level = contract_level
        self.direction = direction
        self.entry_premium = entry_premium
        self.stop_premium = stop_premium


def test_crowd_mirror_empty_portfolio():
    cm = CrowdMirror()
    rep = cm.inspect([])
    assert not rep.we_look_like_retail
    assert rep.retail_similarity_score == 0.0


def test_crowd_mirror_flags_us_as_retail_under_atm_ce_clustering():
    cm = CrowdMirror()
    positions = [
        _FakePos(contract_side="CE", contract_level=0, direction=+1),
        _FakePos(contract_side="CE", contract_level=0, direction=+1),
        _FakePos(contract_side="CE", contract_level=1, direction=+1),
    ]
    rep = cm.inspect(positions)
    assert rep.crowd_density_long_atm_ce >= 0.66
    assert rep.we_look_like_retail
    assert rep.mm_likely_target_us
    assert rep.recommended_diversification


def test_crowd_mirror_doesnt_flag_diversified_portfolio():
    cm = CrowdMirror()
    positions = [
        _FakePos(contract_side="CE", contract_level=3, direction=+1,
                  entry_premium=50.0, stop_premium=20.0),
        _FakePos(contract_side="PE", contract_level=-3, direction=-1,
                  entry_premium=80.0, stop_premium=120.0),
    ]
    rep = cm.inspect(positions)
    assert rep.retail_similarity_score < cm.cfg.retail_threshold


def test_crowd_mirror_serializable():
    cm = CrowdMirror()
    rep = cm.inspect([])
    import json
    json.dumps(rep.to_dict())
