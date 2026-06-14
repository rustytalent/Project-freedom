"""Wave 14 tests — Monte Carlo Lite + Model overlay switcher + Crux
meta-signal composer.

The 'one verdict' fusion layer + 1000-path scenario probe + the model
overlay layer on the spot chart. Tests pin:
  * realised_sigma_per_day on synthetic series
  * simulate returns deterministic results, sensible P(profit)/P(stop)
  * each Crux verdict triggers on its expected input
  * /api/state carries `crux` + `model_zones`
  * /api/monte_carlo runs end-to-end (PRO-gated)
  * cockpit ships the new panels + overlay toggles
"""
from __future__ import annotations

import importlib
import math

import pytest
from fastapi.testclient import TestClient

from sentinel.crux_signal import (
    CruxVerdict, VERDICTS, _CruxInputs, compose, compose_from_state,
)
from sentinel.live_publisher import LivePublisher, ModelSignal, now_ist_hms
from sentinel.monte_carlo import (
    LegPayoff, MonteCarloResult, realised_sigma_per_day, simulate,
)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def test_realised_sigma_positive_on_volatile_series():
    prices = [25000, 25010, 24995, 25020, 24990, 25030, 25015]
    sigma = realised_sigma_per_day(prices)
    assert sigma > 0 and sigma < 5.0


def test_realised_sigma_falls_back_on_short_series():
    assert realised_sigma_per_day([25000]) == 0.15


def test_simulate_deterministic_with_seed():
    legs = [LegPayoff(symbol="X", qty=75, entry_premium=180,
                       delta=0.5, gamma=0.0008, theta_per_day=-5)]
    a = simulate(legs, spot=25000, sigma_annualised=0.15, n_paths=500)
    b = simulate(legs, spot=25000, sigma_annualised=0.15, n_paths=500)
    assert a.to_row() == b.to_row()


def test_simulate_shape_and_horizons():
    legs = [LegPayoff(symbol="X", qty=75, entry_premium=180,
                       delta=0.5, gamma=0.0008, theta_per_day=-5)]
    r = simulate(legs, spot=25000, sigma_annualised=0.15,
                  horizons_min=(5, 15, 30), n_paths=400)
    assert len(r.horizons) == 3
    assert {h.horizon_min for h in r.horizons} == {5, 15, 30}
    for h in r.horizons:
        assert 0.0 <= h.p_profit <= 1.0
        assert 0.0 <= h.p_target <= 1.0
        assert 0.0 <= h.p_stop <= 1.0
        assert h.p10_r <= h.median_r <= h.p90_r


def test_simulate_long_call_more_likely_profit_at_higher_sigma():
    legs = [LegPayoff(symbol="X", qty=75, entry_premium=180,
                       delta=0.5, gamma=0.001, theta_per_day=-5)]
    low = simulate(legs, spot=25000, sigma_annualised=0.10,
                    horizons_min=(15,), n_paths=600, seed=99)
    high = simulate(legs, spot=25000, sigma_annualised=0.30,
                     horizons_min=(15,), n_paths=600, seed=99)
    # With positive Δ + Γ, higher vol -> higher tail outcomes -> higher
    # P(target) (cushioned by the asymmetric payoff)
    assert high.horizons[0].p_target > low.horizons[0].p_target


def test_simulate_empty_legs_short_circuits():
    r = simulate([], spot=25000, sigma_annualised=0.15)
    assert r.horizons == []


# ---------------------------------------------------------------------------
# Crux meta-signal composer
# ---------------------------------------------------------------------------

def _sig(model, signal, conf=0.7, reason_codes=None, zone=None):
    return ModelSignal(
        ts_ist=now_ist_hms(), asset="NIFTY", model=model,
        signal=signal, confidence=conf, trust_tier="LOGGED",
        reason_codes=reason_codes or [], zone=zone)


def test_crux_returns_exit_now_when_intention_violated():
    v = compose(_CruxInputs(intention_violated=True))
    assert v.verdict == "EXIT_NOW"
    assert "intention" in v.rationale.lower()


def test_crux_returns_exit_now_when_killed():
    v = compose(_CruxInputs(killed=True))
    assert v.verdict == "EXIT_NOW"
    assert "kill" in v.rationale.lower()


def test_crux_exit_now_at_red_tilt_with_losing_book():
    v = compose(_CruxInputs(tilt_band="RED", open_pnl=-5000))
    assert v.verdict == "EXIT_NOW"
    assert "tilt" in v.rationale.lower()


def test_crux_trail_up_when_winner_open_and_direction_firing():
    sigs = [_sig("reaction_model", "upside reaction in progress", conf=0.7,
                  reason_codes=["upside_reaction"])]
    v = compose(_CruxInputs(signals=sigs, has_winner=True, open_pnl=1500))
    assert v.verdict == "TRAIL_UP"


def test_crux_wait_when_avoid_signals_dominate():
    sigs = [
        _sig("manipulation_model", "fast upside spike + reclaim — manipulation",
             conf=0.85, reason_codes=["fast_spike", "stop_hunt_risk"]),
        _sig("quality_model", "trend quality LOW (chop)", conf=0.7,
             reason_codes=["LOW (chop)"]),
    ]
    v = compose(_CruxInputs(signals=sigs))
    assert v.verdict == "WAIT"


def test_crux_trade_on_strong_bull_consensus():
    sigs = [
        _sig("reaction_model", "upside reaction in progress", conf=0.75,
             reason_codes=["upside_reaction"]),
        _sig("proximity_model", "approaching session low — demand_zone",
             conf=0.7, reason_codes=["near_low", "demand_zone"]),
    ]
    v = compose(_CruxInputs(signals=sigs))
    assert v.verdict == "TRADE"
    assert v.supporting


def test_crux_watch_on_lone_signal():
    sigs = [_sig("reaction_model", "upside reaction in progress", conf=0.6,
                  reason_codes=["upside_reaction"])]
    v = compose(_CruxInputs(signals=sigs))
    assert v.verdict == "WATCH"


def test_crux_wait_on_silent_bus():
    v = compose(_CruxInputs(signals=[]))
    assert v.verdict == "WAIT"


def test_crux_published_to_bus_via_compose_from_state():
    pub = LivePublisher()
    sigs = {"NIFTY|reaction_model": _sig(
        "reaction_model", "upside reaction in progress", conf=0.8,
        reason_codes=["upside_reaction"])}
    compose_from_state(sigs, tilt_band="GREEN", publisher=pub)
    assert "NIFTY|crux" in pub.current()


# ---------------------------------------------------------------------------
# Server integration
# ---------------------------------------------------------------------------

@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    import sentinel.server as server
    importlib.reload(server)
    return server


def test_state_carries_crux_and_zones(srv):
    core = srv.CORE
    # warm up so models fire and publish zones
    for i in range(120):
        core._tick_hist.append(__import__("sentinel.live_models", fromlist=["Tick"]).Tick(
            ts=i, spot=25000 + i * 0.5))
    core._tick_models()
    core._tick_psychology()
    core._tick_crux()
    with TestClient(srv.app) as c:
        s = c.get("/api/state").json()
        assert "crux" in s
        assert "model_zones" in s
        if s["crux"]:
            assert s["crux"]["verdict"] in VERDICTS


def test_monte_carlo_endpoint_runs_for_pro(srv):
    body = {
        "legs": [{"symbol": "X", "qty": 75, "entry_premium": 180,
                   "delta": 0.5, "gamma": 0.001, "theta_per_day": -5}],
        "spot": 25000, "horizons_min": [5, 15],
        "n_paths": 200, "target_pnl_rupees": 1500, "stop_pnl_rupees": 1500,
    }
    with TestClient(srv.app) as c:
        r = c.post("/api/monte_carlo",
                   headers={"X-Sentinel-Plan": "PRO"}, json=body)
        assert r.status_code == 200
        payload = r.json()
        assert "horizons" in payload and len(payload["horizons"]) == 2


def test_monte_carlo_endpoint_gates_retail(srv):
    body = {"legs": [], "spot": 25000, "horizons_min": [5]}
    with TestClient(srv.app) as c:
        r = c.post("/api/monte_carlo",
                   headers={"X-Sentinel-Plan": "RETAIL"}, json=body)
        assert r.status_code == 402


def test_crux_endpoint_returns_verdict_after_cycle(srv):
    core = srv.CORE
    core._tick_crux()
    with TestClient(srv.app) as c:
        r = c.get("/api/crux")
        assert r.status_code == 200
        # may be empty if no signals yet but key must be present
        body = r.json()
        if body:
            assert "verdict" in body


def test_cockpit_ships_crux_and_mc_and_overlays(srv):
    with TestClient(srv.app) as c:
        html = c.get("/").text
    for marker in (
        "Crux · live verdict", "crux_body", "renderCrux(",
        "EXIT_NOW", "TRAIL_UP", "TRADE", "DO_NOT_CHASE", "WATCH", "WAIT", "HOLD",
        "Monte Carlo", "runMonteCarlo(", "mc_target", "mc_stop",
        "overlay_toggles", "OVERLAY_ON", "toggleOverlay(",
    ):
        assert marker in html
