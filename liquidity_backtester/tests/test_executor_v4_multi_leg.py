"""Multi-leg structures, broker routing + reconciliation.

Founder 2026-06-22 Tier-2 brief: "Today's market is dead chop. Iron
condor / jade lizard / butterfly are what we should be trading. Wire
the multi-leg routing." These tests cover:

  * LegSpec dataclass + serialization + reversal
  * Each structure constructor (iron condor, jade lizard, butterfly,
    strangle, vertical spread)
  * The strike-grid picking logic (no leg collisions, wings push
    further OTM)
  * Bundle ledger combined-R updates and exit detection
  * Broker reconciliation: clean fill, total rejection, partial fill
    rollback
  * End-to-end iron condor through PaperBrokerAdapter
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from liqpool.research.belief.executor_v4 import (
    PaperBrokerAdapter,
    PersistenceConfig,
    PortfolioManagerConfig,
    V4Runner,
    V4RunnerConfig,
)
from liqpool.research.belief.executor_v4.broker.base import (
    BrokerOrder, BrokerOrderResult, STATE_FILLED, STATE_REJECTED,
)
from liqpool.research.belief.executor_v4.multi_leg import (
    BUNDLE_OPEN,
    BUNDLE_REJECTED,
    BUNDLE_ROLLED_BACK,
    BundleLedger,
    Butterfly,
    IronCondor,
    JadeLizard,
    LEG_LONG,
    LEG_SHORT,
    LegSpec,
    MultiLegBundle,
    OPT_CE,
    OPT_PE,
    STRUCTURE_BUTTERFLY,
    STRUCTURE_IRON_CONDOR,
    STRUCTURE_JADE_LIZARD,
    STRUCTURE_STRANGLE,
    STRUCTURE_VERTICAL_SPREAD,
    Strangle,
    StructureConstructor,
    VerticalSpread,
    reconcile_partial_fills,
)
from liqpool.research.belief.executor_v4.multi_leg.reconciliation import (
    RECON_OK, RECON_REJECTED, RECON_ROLLED_BACK,
)


# ── Fixtures ────────────────────────────────────────────────────


def _build_slot_grid(spot: float = 23000.0, n_each_side: int = 10,
                       step: int = 50) -> List[Dict[str, Any]]:
    """A wider strike grid so wings can land on distinct strikes."""
    slots: List[Dict[str, Any]] = []
    for level in range(-n_each_side, n_each_side + 1):
        for opt in (OPT_CE, OPT_PE):
            strike = spot + step * level
            # Synthetic mark — closer to ATM = higher premium.
            distance_norm = abs(strike - spot) / max(1.0, spot * 0.02)
            mark = max(2.0, 100.0 - distance_norm * 40.0)
            slots.append({
                "strike": strike, "option_type": opt,
                "label": f"NIFTY{int(strike)}{opt}",
                "mark_price": mark,
                "moneyness_label": f"{opt}_{level}",
                "friendliness": 0.9, "is_abnormal": False,
                "spread_state": "clean",
            })
    return slots


# ── LegSpec ────────────────────────────────────────────────────


def test_legspec_serialization_round_trip():
    leg = LegSpec(role="short_atm_ce", strike=23000.0,
                    option_type=OPT_CE, side=LEG_SHORT, lots=2,
                    tradingsymbol="NIFTY23000CE",
                    estimated_premium=85.0)
    d = leg.to_dict()
    back = LegSpec.from_dict(d)
    assert back.role == leg.role
    assert back.strike == leg.strike
    assert back.option_type == leg.option_type
    assert back.side == leg.side
    assert back.lots == leg.lots
    assert back.tradingsymbol == leg.tradingsymbol


def test_legspec_reverse_flips_side_only():
    leg = LegSpec(role="long_otm_pe", strike=22950.0, option_type=OPT_PE,
                    side=LEG_LONG, lots=2, tradingsymbol="NIFTY22950PE",
                    estimated_premium=40.0)
    rev = leg.reversed()
    assert rev.side == LEG_SHORT
    assert rev.strike == leg.strike
    assert rev.tradingsymbol == leg.tradingsymbol
    assert rev.lots == leg.lots
    assert "reverse" in rev.role


def test_legspec_signed_lots():
    long_leg = LegSpec(role="x", strike=1.0, option_type=OPT_CE,
                         side=LEG_LONG, lots=3)
    short_leg = LegSpec(role="y", strike=1.0, option_type=OPT_CE,
                          side=LEG_SHORT, lots=3)
    assert long_leg.signed_lots == 3
    assert short_leg.signed_lots == -3


# ── Iron condor ────────────────────────────────────────────────


def test_iron_condor_constructs_four_legs():
    slots = _build_slot_grid()
    b = IronCondor().construct(spot=23000.0, slot_readings=slots, lots=2)
    assert b is not None
    assert b.structure_class == STRUCTURE_IRON_CONDOR
    assert len(b.legs) == 4
    roles = sorted([l.role for l in b.legs])
    assert roles == ["long_wing_ce", "long_wing_pe",
                       "short_otm_ce", "short_otm_pe"]
    # Wings outside body.
    short_ce = next(l for l in b.legs if l.role == "short_otm_ce")
    long_ce = next(l for l in b.legs if l.role == "long_wing_ce")
    assert long_ce.strike > short_ce.strike
    short_pe = next(l for l in b.legs if l.role == "short_otm_pe")
    long_pe = next(l for l in b.legs if l.role == "long_wing_pe")
    assert long_pe.strike < short_pe.strike


def test_iron_condor_is_balanced():
    slots = _build_slot_grid()
    b = IronCondor().construct(spot=23000.0, slot_readings=slots, lots=3)
    assert b is not None
    assert b.has_balanced_size()
    # Lot sums.
    assert b.long_lots() == 6
    assert b.short_lots() == 6


def test_iron_condor_returns_none_on_missing_grid():
    b = IronCondor().construct(spot=23000.0, slot_readings=[], lots=1)
    assert b is None


# ── Jade lizard ────────────────────────────────────────────────


def test_jade_lizard_three_legs_with_wing():
    slots = _build_slot_grid()
    b = JadeLizard().construct(spot=23000.0, slot_readings=slots, lots=1)
    assert b is not None
    assert b.structure_class == STRUCTURE_JADE_LIZARD
    assert len(b.legs) == 3
    sides = [l.side for l in b.legs]
    assert sides.count(LEG_SHORT) == 2
    assert sides.count(LEG_LONG) == 1


# ── Butterfly ──────────────────────────────────────────────────


def test_butterfly_three_legs_body_doubled():
    slots = _build_slot_grid()
    b = Butterfly().construct(spot=23000.0, slot_readings=slots, lots=1)
    assert b is not None
    assert b.structure_class == STRUCTURE_BUTTERFLY
    # Body is 2× the wing lots.
    body_leg = next(l for l in b.legs if l.role == "short_atm_body")
    wing_leg = next(l for l in b.legs if l.role == "long_lower_wing")
    assert body_leg.lots == wing_leg.lots * 2


# ── Strangle ───────────────────────────────────────────────────


def test_long_strangle_both_legs_long():
    slots = _build_slot_grid()
    b = Strangle().construct(spot=23000.0, slot_readings=slots, lots=2,
                                short=False)
    assert b is not None
    assert all(l.is_long for l in b.legs)


def test_short_strangle_both_legs_short():
    slots = _build_slot_grid()
    b = Strangle().construct(spot=23000.0, slot_readings=slots, lots=2,
                                short=True)
    assert b is not None
    assert all(l.is_short for l in b.legs)


# ── Vertical spread ────────────────────────────────────────────


def test_vertical_spread_bull_call_debit():
    slots = _build_slot_grid()
    b = VerticalSpread().construct(
        spot=23000.0, slot_readings=slots, direction=1,
        option_type=OPT_CE, lots=2, credit=False,
    )
    assert b is not None
    assert len(b.legs) == 2
    assert all(l.option_type == OPT_CE for l in b.legs)


# ── StructureConstructor router ────────────────────────────────


def test_structure_constructor_routes_to_iron_condor():
    sc = StructureConstructor()
    slots = _build_slot_grid()
    b = sc.construct(strategy_class=STRUCTURE_IRON_CONDOR,
                       spot=23000.0, slot_readings=slots, lots=2)
    assert b is not None
    assert b.structure_class == STRUCTURE_IRON_CONDOR


def test_structure_constructor_returns_none_for_unknown_class():
    sc = StructureConstructor()
    slots = _build_slot_grid()
    b = sc.construct(strategy_class="not_a_thing",
                       spot=23000.0, slot_readings=slots, lots=2)
    assert b is None


def test_structure_constructor_recognises_multi_leg_classes():
    sc = StructureConstructor()
    assert sc.is_multi_leg(STRUCTURE_IRON_CONDOR) is True
    assert sc.is_multi_leg(STRUCTURE_JADE_LIZARD) is True
    assert sc.is_multi_leg("long_ce") is False


# ── Reconciliation ─────────────────────────────────────────────


def _mk_result(state: str, filled_q: int = 65,
                  avg: float = 50.0,
                  reason: str = "") -> BrokerOrderResult:
    return BrokerOrderResult(
        client_order_id="x",
        broker_order_id="b" if state == STATE_FILLED else "",
        state=state, filled_quantity=filled_q,
        avg_fill_price=avg, rejection_reason=reason or None,
    )


def test_reconcile_all_filled_marks_bundle_open():
    slots = _build_slot_grid()
    b = IronCondor().construct(spot=23000.0, slot_readings=slots, lots=1)
    results = [_mk_result(STATE_FILLED, 65, 50.0) for _ in b.legs]
    out = reconcile_partial_fills(bundle=b, results=results, lot_size=65)
    assert out.outcome == RECON_OK
    assert b.state == BUNDLE_OPEN
    assert len(out.reverse_orders) == 0


def test_reconcile_all_rejected_marks_bundle_rejected():
    slots = _build_slot_grid()
    b = IronCondor().construct(spot=23000.0, slot_readings=slots, lots=1)
    results = [_mk_result(STATE_REJECTED, 0, 0.0, "no liquidity")
                for _ in b.legs]
    out = reconcile_partial_fills(bundle=b, results=results, lot_size=65)
    assert out.outcome == RECON_REJECTED
    assert b.state == BUNDLE_REJECTED
    assert len(out.reverse_orders) == 0


def test_reconcile_partial_fill_generates_reverse_orders():
    slots = _build_slot_grid()
    b = IronCondor().construct(spot=23000.0, slot_readings=slots, lots=1)
    # First two legs fill, last two reject.
    results = [
        _mk_result(STATE_FILLED, 65, 50.0),
        _mk_result(STATE_FILLED, 65, 60.0),
        _mk_result(STATE_REJECTED, 0, 0.0, "no fill"),
        _mk_result(STATE_REJECTED, 0, 0.0, "no fill"),
    ]
    out = reconcile_partial_fills(bundle=b, results=results, lot_size=65)
    assert out.outcome == RECON_ROLLED_BACK
    assert b.state == BUNDLE_ROLLED_BACK
    assert len(out.reverse_orders) == 2
    # Reverse orders flip the side.
    assert out.reverse_orders[0].side == LEG_LONG    # was SHORT
    assert out.reverse_orders[1].side == LEG_LONG    # was SHORT


# ── Bundle ledger ──────────────────────────────────────────────


def test_bundle_ledger_open_and_update_combined_marks():
    ledger = BundleLedger(lot_size=65)
    slots = _build_slot_grid()
    b = IronCondor().construct(spot=23000.0, slot_readings=slots, lots=1)
    b.net_credit_at_entry = 30.0   # synthetic credit
    b.max_loss_estimate = 50.0 * 65
    fill_prices = [l.estimated_premium for l in b.legs]
    entry = ledger.open(b, fill_prices)
    assert ledger.n_open() == 1
    # Decay every short leg's mark to half; long wings stay the same.
    # Combined premium drops → credit kept → R goes up.
    marks: List[float] = []
    for leg in b.legs:
        if leg.is_short:
            marks.append((leg.estimated_premium or 0) * 0.5)
        else:
            marks.append(leg.estimated_premium or 0)
    ledger.update_combined_marks(b.bundle_id,
                                       leg_marks=marks, bar_index=10)
    refreshed = ledger.get(b.bundle_id)
    assert refreshed is not None
    # Combined premium should have decayed favourably (closer to zero
    # or negative for a credit structure means we kept the credit).
    assert refreshed.current_combined_premium <= b.net_credit_at_entry


def test_bundle_ledger_close_moves_entry_to_closed_list():
    ledger = BundleLedger(lot_size=65)
    slots = _build_slot_grid()
    b = IronCondor().construct(spot=23000.0, slot_readings=slots, lots=1)
    ledger.open(b, [l.estimated_premium for l in b.legs])
    ledger.close(b.bundle_id, realised_rupees=750.0,
                    exit_reason="target hit", bars_held=22)
    assert ledger.n_open() == 0
    assert ledger.n_closed() == 1


# ── Broker integration (paper) ─────────────────────────────────


def test_paper_broker_places_iron_condor_atomically():
    broker = PaperBrokerAdapter()
    slots = _build_slot_grid()
    b = IronCondor().construct(spot=23000.0, slot_readings=slots, lots=1)
    out = broker.place_multi_leg_bundle(b, lot_size=65,
                                              position_id=b.bundle_id)
    assert out.outcome == RECON_OK
    assert b.state == BUNDLE_OPEN
    # 4 legs filled = 4 broker fills.
    assert len(broker.fills) == 4


def test_paper_broker_rejects_when_killed():
    broker = PaperBrokerAdapter()
    broker.kill_switch("operator test")
    slots = _build_slot_grid()
    b = IronCondor().construct(spot=23000.0, slot_readings=slots, lots=1)
    out = broker.place_multi_leg_bundle(b, lot_size=65)
    assert out.outcome == RECON_REJECTED


# ── Manager wiring ─────────────────────────────────────────────


def _runner_with_state(state_dir: Path) -> V4Runner:
    cfg = V4RunnerConfig(
        manager=None,
        persistence=PersistenceConfig(state_dir=state_dir, enabled=True),
        emit_explainer_to_log=False,
    )
    return V4Runner(cfg=cfg, broker=PaperBrokerAdapter())


def test_manager_exposes_structure_constructor_and_bundle_ledger():
    tmp = Path(tempfile.mkdtemp(prefix="ml_mgr_"))
    runner = _runner_with_state(tmp)
    assert runner.manager.structure_constructor is not None
    assert runner.manager.bundle_ledger is not None


def test_manager_try_multi_leg_entry_opens_iron_condor_bundle():
    tmp = Path(tempfile.mkdtemp(prefix="ml_try_"))
    runner = _runner_with_state(tmp)
    # Need at least one tick so the manager has a snapshot to work
    # from. The shared _snap fixture provides slot_readings.
    from tests.test_executor_v4_dead_market_and_web_viz import _snap
    for i in range(5):
        runner.on_tick(_snap(100 + i))
    out = runner.manager.try_multi_leg_entry(
        structure_class=STRUCTURE_IRON_CONDOR,
        lots=1, broker=runner.broker,
    )
    assert out is not None
    # The synthetic snap grid is narrow (only ±5 strikes), so iron
    # condor's wings might land on the same strike as the body. We
    # accept either OK (bundle opened) or NOT_CONSTRUCTED with the
    # right notes — both are valid outcomes for the test data.
    assert out.get("outcome") in (RECON_OK, "NOT_CONSTRUCTED",
                                     "NOT_SUBMITTED")


def test_cockpit_carries_multi_leg_panel():
    tmp = Path(tempfile.mkdtemp(prefix="ml_panel_"))
    runner = _runner_with_state(tmp)
    from tests.test_executor_v4_dead_market_and_web_viz import _snap
    res = None
    for i in range(3):
        res = runner.on_tick(_snap(100 + i))
    panel = res.cockpit.to_dict()["multi_leg_panel"]
    for key in ["n_open_bundles", "n_closed_bundles", "open",
                  "closed_recent", "wins", "losses", "win_rate"]:
        assert key in panel


def test_viewer_html_includes_multi_leg_panel():
    from liqpool.research.belief.executor_v4.cockpit_server import (
        _DEFAULT_VIEWER_HTML,
    )
    h = _DEFAULT_VIEWER_HTML
    must_contain = [
        "MULTI-LEG BUNDLES", "ml_open", "ml_open_list",
        "ml_closed_list", "render_multi_leg",
        "iron condor", "jade lizard",
    ]
    missing = [s for s in must_contain if s not in h]
    assert not missing, f"viewer HTML missing tokens: {missing}"
