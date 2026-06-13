"""Sentinel core tests — greeks, scenario engine, advisor, ledger."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sentinel.advisor import (
    Maximizer, SuggestionLedger, recommend_dips,
)
from sentinel.greeks import bs_price, greeks, implied_vol, reprice
from sentinel.kite_client import InstrumentMeta, Quote
from sentinel.portfolio import PortfolioState

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Greeks / IV
# ---------------------------------------------------------------------------

def test_iv_round_trip():
    """price -> implied_vol -> price closes the loop."""
    p = bs_price(25000, 25000, 7 / 365, 0.14, "CE")
    iv = implied_vol(p, 25000, 25000, 7 / 365, "CE")
    assert iv is not None and abs(iv - 0.14) < 1e-3
    p2 = bs_price(25000, 25000, 7 / 365, iv, "CE")
    assert abs(p2 - p) < 0.05


def test_iv_none_below_intrinsic():
    # Deep ITM CE quoted below intrinsic — arbitrage print -> None.
    assert implied_vol(50.0, 25000, 24000, 7 / 365, "CE") is None


def test_put_delta_negative_call_positive():
    g_ce = greeks(25000, 25000, 7 / 365, 0.14, "CE")
    g_pe = greeks(25000, 25000, 7 / 365, 0.14, "PE")
    assert 0.4 < g_ce.delta < 0.65
    assert -0.65 < g_pe.delta < -0.35
    # ATM long options decay: theta negative both sides.
    assert g_ce.theta_per_day < 0 and g_pe.theta_per_day < 0


def test_reprice_direction():
    """Spot up -> CE worth more, PE worth less. The founder's core
    intuition, verified by the pricing engine."""
    t, iv = 7 / 365, 0.14
    ce0 = reprice(25000, 25000, t, iv, "CE")
    ce1 = reprice(25150, 25000, t, iv, "CE")
    pe0 = reprice(25000, 25000, t, iv, "PE")
    pe1 = reprice(25150, 25000, t, iv, "PE")
    assert ce1 > ce0 and pe1 < pe0


# ---------------------------------------------------------------------------
# Portfolio + scenario engine
# ---------------------------------------------------------------------------

def _metas(expiry_days: float = 7.0):
    exp = datetime.now(IST) + timedelta(days=expiry_days)
    return {
        "NIFTY25000CE": InstrumentMeta("NIFTY25000CE", "NIFTY", 25000, "CE", exp, 75),
        "NIFTY24800PE": InstrumentMeta("NIFTY24800PE", "NIFTY", 24800, "PE", exp, 75),
    }


def _pf(spot=25000.0, ce_avg_mult=0.8, pe_avg_mult=1.2):
    pf = PortfolioState()
    metas = _metas()
    ce_p = bs_price(spot, 25000, 7 / 365, 0.14, "CE")
    pe_p = bs_price(spot, 24800, 7 / 365, 0.15, "PE")
    raw = [
        {"tradingsymbol": "NIFTY25000CE", "quantity": 75,
         "average_price": ce_p * ce_avg_mult, "last_price": ce_p},
        {"tradingsymbol": "NIFTY24800PE", "quantity": 75,
         "average_price": pe_p * pe_avg_mult, "last_price": pe_p},
    ]
    quotes = {
        "NIFTY25000CE": Quote("NIFTY25000CE", ltp=ce_p, day_high=ce_p * 1.3),
        "NIFTY24800PE": Quote("NIFTY24800PE", ltp=pe_p, day_high=pe_p * 1.4),
    }
    pf.refresh(raw, quotes, metas, spot, {"cash": 100000, "utilised": 0, "net": 100000})
    return pf


def test_portfolio_enrichment_computes_greeks():
    pf = _pf()
    ce = pf.positions["NIFTY25000CE"]
    assert ce.iv is not None and 0.10 < ce.iv < 0.20
    assert ce.delta is not None and ce.delta > 0.4
    pe = pf.positions["NIFTY24800PE"]
    assert pe.delta is not None and pe.delta < 0


def test_pair_detection_finds_ce_pe_pair():
    pf = _pf()
    assert len(pf.pairs) == 1
    pair = pf.pairs[0]
    assert pair.underlying == "NIFTY"
    assert set(pair.legs) == {"NIFTY25000CE", "NIFTY24800PE"}
    # CE winning (avg below ltp), PE losing -> the funding note exists.
    assert "funding" in pair.note


def test_scenario_curve_strangle_shape_and_breakevens():
    """A long strangle bought ABOVE fair (both legs at 1.15x) loses in
    the middle and wins on big moves: the curve must be V-ish (ends
    above the centre) and cross zero at least once — the breakeven
    diamonds the dashboard draws."""
    pf = _pf(ce_avg_mult=1.15, pe_avg_mult=1.15)
    sc = pf.scenario_curve(move_pct_range=2.0, steps=41)
    pnls = [p["pnl"] for p in sc["points"]]
    centre = pnls[len(pnls) // 2]
    assert pnls[0] > centre and pnls[-1] > centre     # V shape
    assert centre < 0                                  # paid-up book bleeds at rest
    assert len(sc["breakevens"]) >= 1


def test_scenario_curve_no_breakeven_when_always_profitable():
    """A book deep in profit everywhere in the window has NO breakeven
    — the engine must report an empty list, not invent one."""
    pf = _pf(ce_avg_mult=0.5, pe_avg_mult=0.5)        # both legs bought cheap
    sc = pf.scenario_curve(move_pct_range=1.0, steps=21)
    assert all(p["pnl"] > 0 for p in sc["points"])
    assert sc["breakevens"] == []


def test_what_if_direction_matches_founder_intuition():
    """Spot +150: the CE gains premium, the PE loses premium."""
    pf = _pf()
    legs = {l["tradingsymbol"]: l for l in pf.what_if(150.0)}
    assert legs["NIFTY25000CE"]["premium_change"] > 0
    assert legs["NIFTY24800PE"]["premium_change"] < 0


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

def _chain(spot=25000.0):
    exp = datetime.now(IST) + timedelta(days=7)
    metas, quotes = [], {}
    for k in range(-3, 4):
        strike = spot + k * 100
        for t in ("CE", "PE"):
            sym = f"N{int(strike)}{t}"
            metas.append(InstrumentMeta(sym, "NIFTY", strike, t, exp, 75))
            fair = bs_price(spot, strike, 7 / 365, 0.14, t)
            # Make one specific put deeply marked down from its high.
            day_high = fair * (2.0 if sym == f"N{int(spot - 100)}PE" else 1.15)
            quotes[sym] = Quote(sym, ltp=max(fair, 0.5),
                                day_high=day_high, day_low=fair * 0.7,
                                volume=1_000_000, oi=1_000_000,
                                bid=fair - 0.5, ask=fair + 0.5)
    return metas, quotes


def test_recommender_up_returns_puts_ranked_by_fall():
    spot = 25000.0
    metas, quotes = _chain(spot)
    recs = recommend_dips("UP", spot, metas, quotes, top_n=3)
    assert recs and all(r.option_type == "PE" for r in recs)
    # The deliberately-marked-down put ranks first.
    assert recs[0].tradingsymbol == f"N{int(spot-100)}PE"
    assert recs[0].fall_from_high_pct > 40


def test_recommender_down_returns_calls():
    spot = 25000.0
    metas, quotes = _chain(spot)
    recs = recommend_dips("DOWN", spot, metas, quotes, top_n=3)
    assert recs and all(r.option_type == "CE" for r in recs)


def test_recommender_wide_spread_penalised():
    """Blowing out the spread on the top candidate must cut its score
    roughly in half versus the tight-spread baseline — the haircut is
    the liquidity-reality check on 'it fell a lot, buy it'."""
    spot = 25000.0
    sym = f"N{int(spot-100)}PE"
    metas, quotes = _chain(spot)
    baseline = {r.tradingsymbol: r.score
                for r in recommend_dips("UP", spot, metas, quotes, top_n=20)}
    q = quotes[sym]
    quotes[sym] = Quote(sym, q.ltp, q.day_high, q.day_low, q.volume, q.oi,
                        bid=q.ltp * 0.8, ask=q.ltp * 1.2)
    wide = {r.tradingsymbol: r.score
            for r in recommend_dips("UP", spot, metas, quotes, top_n=20)}
    assert sym in baseline and sym in wide
    assert wide[sym] < baseline[sym] * 0.6


# ---------------------------------------------------------------------------
# Maximizer + ledger
# ---------------------------------------------------------------------------

def test_maximizer_suggests_trail_for_unprotected_winner(tmp_path):
    pf = _pf()
    ledger = SuggestionLedger(tmp_path / "s.jsonl")
    mx = Maximizer(ledger)
    sugg = mx.run(pf, armed_symbols=[])
    rules = {s.rule_id for s in sugg}
    assert "R1_unprotected_winner" in rules        # CE is up with no trail
    assert "R2_paired_leg_bleed" in rules          # PE bleeding vs CE winning


def test_maximizer_respects_armed_symbols(tmp_path):
    pf = _pf()
    ledger = SuggestionLedger(tmp_path / "s.jsonl")
    mx = Maximizer(ledger)
    sugg = mx.run(pf, armed_symbols=["NIFTY25000CE"])
    r1 = [s for s in sugg if s.rule_id == "R1_unprotected_winner"]
    assert all(s.tradingsymbol != "NIFTY25000CE" for s in r1)


def test_ledger_scores_close_suggestion(tmp_path):
    ledger = SuggestionLedger(tmp_path / "s.jsonl", resolve_minutes=0.0)
    pf = _pf()
    mx = Maximizer(ledger)
    sugg = mx.run(pf, armed_symbols=[])
    close = next(s for s in sugg if s.action == "CLOSE")
    # Premium FELL after the CLOSE call -> the call was right.
    n = ledger.resolve_due({close.tradingsymbol: close.premium_at_suggestion * 0.8})
    assert n >= 1
    stats = ledger.stats()
    assert stats["rules"]["R2_paired_leg_bleed"]["n_resolved"] >= 1
    assert stats["rules"]["R2_paired_leg_bleed"]["hit_rate"] > 0.5


def test_ledger_hit_rates_persist_across_restart(tmp_path):
    path = tmp_path / "s.jsonl"
    l1 = SuggestionLedger(path, resolve_minutes=0.0)
    pf = _pf()
    Maximizer(l1).run(pf, armed_symbols=[])
    l1.resolve_due({s: 1.0 for s in pf.positions})
    rates_before = l1.stats()["rules"]
    l2 = SuggestionLedger(path)
    assert l2.stats()["rules"].keys() == rates_before.keys()


def test_ledger_does_not_restack_same_live_suggestion(tmp_path):
    ledger = SuggestionLedger(tmp_path / "s.jsonl", resolve_minutes=999.0)
    pf = _pf()
    mx = Maximizer(ledger)
    mx.run(pf, armed_symbols=[])
    n1 = ledger.stats()["pending"]
    mx.run(pf, armed_symbols=[])   # same conditions again
    assert ledger.stats()["pending"] == n1
