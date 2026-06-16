"""Tests for Belief Engine Phase 4 — fair response, deviation-of-deviation,
and spread friendliness.

Pins:
  * Effective delta is estimated live and converges toward the true delta,
    shrinking toward the moneyness prior when the sample is thin.
  * The residual is actual − fair change; deviation-of-deviation is the
    robust z of the residual against its own slot-specific band.
  * "Defended" acceptance fires when the residual is persistently and
    abnormally positive (premium held above fair).
  * Spread friendliness drops and the state turns widening/dangerous when
    the spread blows out, recovering to clean when it tightens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liqpool.research.belief.fair_response import (
    FairResponseConfig,
    estimate_effective_delta,
    fair_response_frame,
)
from liqpool.research.belief.moneyness import classify_moneyness
from liqpool.research.belief.residual import (
    ResidualConfig,
    deviation_of_deviation,
    slot_residual_frame,
)
from liqpool.research.belief.spread import (
    CLEAN,
    DANGEROUS,
    WIDENING,
    SpreadConfig,
    is_execution_friendly,
    spread_friendliness_frame,
)


SPOT0 = 23000.0


def _ce_slot(level_strike=23000):
    return classify_moneyness(SPOT0, level_strike, "CE", 50)


def _pe_slot(level_strike=23000):
    return classify_moneyness(SPOT0, level_strike, "PE", 50)


# ─────────────────────────────────────────────────────────────────
# Fair response / effective delta
# ─────────────────────────────────────────────────────────────────

def test_effective_delta_converges_to_true_delta():
    rng = np.random.default_rng(1)
    n = 300
    spot = SPOT0 + np.cumsum(rng.normal(0, 5, n))
    d = np.diff(spot, prepend=spot[0])
    true_delta = 0.5
    mark = 100 + np.cumsum(true_delta * d + rng.normal(0, 0.3, n))
    df = pd.DataFrame({"spot": spot, "mark": np.clip(mark, 1, None)})
    eff = estimate_effective_delta(df["mark"], df["spot"], _ce_slot())
    # In the body of the series the estimate should sit near the truth.
    assert abs(eff.iloc[60:].mean() - true_delta) < 0.12


def test_effective_delta_respects_option_sign():
    rng = np.random.default_rng(2)
    n = 200
    spot = SPOT0 + np.cumsum(rng.normal(0, 5, n))
    d = np.diff(spot, prepend=spot[0])
    # A put: premium falls when spot rises → negative delta.
    mark = 100 + np.cumsum(-0.5 * d + rng.normal(0, 0.3, n))
    eff = estimate_effective_delta(pd.Series(np.clip(mark, 1, None)),
                                   pd.Series(spot), _pe_slot())
    assert (eff <= 0).all(), "put effective delta must be non-positive"


def test_effective_delta_uses_prior_when_thin():
    # Only a couple of bars → estimate should be close to the moneyness prior.
    df = pd.DataFrame({"spot": [23000.0, 23005.0, 23010.0],
                       "mark": [100.0, 102.0, 104.0]})
    slot = _ce_slot()
    eff = estimate_effective_delta(df["mark"], df["spot"], slot)
    # Early values dominated by prior (0.50 for ATM CE).
    assert abs(eff.iloc[1] - slot.expected_signed_delta) < 0.3


def test_fair_response_frame_residual_identity():
    rng = np.random.default_rng(3)
    n = 120
    spot = SPOT0 + np.cumsum(rng.normal(0, 4, n))
    mark = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"spot": spot, "mark": np.clip(mark, 1, None)})
    fr = fair_response_frame(df, _ce_slot())
    # residual ≡ actual_change − fair_change.
    np.testing.assert_allclose(
        fr["residual"].iloc[1:],
        (fr["actual_change"] - fr["fair_change"]).iloc[1:],
        rtol=1e-9, atol=1e-9,
    )


def test_fair_response_requires_columns():
    with pytest.raises(ValueError, match="mark"):
        fair_response_frame(pd.DataFrame({"spot": [1, 2, 3]}), _ce_slot())


# ─────────────────────────────────────────────────────────────────
# Deviation of deviation
# ─────────────────────────────────────────────────────────────────

def test_dod_z_is_bounded():
    rng = np.random.default_rng(4)
    resid = pd.Series(rng.normal(0, 1, 200))
    resid.iloc[100] = 1e6     # extreme outlier
    dod = deviation_of_deviation(resid)
    assert dod["dod_z"].abs().max() <= 8.0 + 1e-9


def test_dod_z_zero_centered_on_stationary_residual():
    rng = np.random.default_rng(5)
    resid = pd.Series(rng.normal(0, 1, 300))
    dod = deviation_of_deviation(resid)
    assert abs(dod["dod_z"].iloc[60:].median()) < 0.5


def test_defended_acceptance_on_sustained_positive_residual():
    """Flat spot + steadily rising premium = pure sustained positive
    residual → acceptance must reach 'defended'."""
    rng = np.random.default_rng(6)
    n = 200
    spot = SPOT0 + np.cumsum(rng.normal(0, 4, n))
    d = np.diff(spot, prepend=spot[0])
    mark = 100 + np.cumsum(0.5 * d + rng.normal(0, 0.4, n))
    spot[120:133] = spot[119]                      # flat spot
    mark[120:133] = mark[119] + np.cumsum(np.full(13, 1.2))  # premium pushed up
    df = pd.DataFrame({"spot": spot, "mark": np.clip(mark, 1, None)})
    fr = slot_residual_frame(df, _ce_slot())
    assert (fr.loc[120:135, "acceptance"] == "defended").any()
    assert (fr.loc[120:132, "is_abnormal"]).any()


def test_rejected_acceptance_on_sustained_negative_residual():
    rng = np.random.default_rng(7)
    n = 200
    spot = SPOT0 + np.cumsum(rng.normal(0, 4, n))
    d = np.diff(spot, prepend=spot[0])
    mark = 100 + np.cumsum(0.5 * d + rng.normal(0, 0.4, n))
    spot[120:133] = spot[119]
    mark[120:133] = mark[119] - np.cumsum(np.full(13, 1.2))  # premium pushed down
    df = pd.DataFrame({"spot": spot, "mark": np.clip(mark, 1, None)})
    fr = slot_residual_frame(df, _ce_slot())
    assert (fr.loc[120:135, "acceptance"] == "rejected").any()


def test_slot_residual_frame_is_causal():
    rng = np.random.default_rng(8)
    n = 200
    spot = SPOT0 + np.cumsum(rng.normal(0, 4, n))
    d = np.diff(spot, prepend=spot[0])
    mark = 100 + np.cumsum(0.5 * d + rng.normal(0, 0.4, n))
    df = pd.DataFrame({"spot": spot, "mark": np.clip(mark, 1, None)})
    full = slot_residual_frame(df, _ce_slot())
    k = 150
    trunc = slot_residual_frame(df.iloc[:k + 1], _ce_slot())
    assert trunc["dod_z"].iloc[k] == pytest.approx(full["dod_z"].iloc[k], abs=1e-9)


# ─────────────────────────────────────────────────────────────────
# Spread friendliness
# ─────────────────────────────────────────────────────────────────

def _spread_df(n=200, seed=9):
    rng = np.random.default_rng(seed)
    mark = 100 + np.cumsum(rng.normal(0, 0.5, n))
    mark = np.clip(mark, 5, None)
    bid = mark - 0.15
    ask = mark + 0.15
    return pd.DataFrame({"mark": mark, "bid": bid, "ask": ask,
                         "bid_qty": 500.0, "ask_qty": 500.0})


def test_spread_clean_when_tight():
    df = _spread_df()
    sp = spread_friendliness_frame(df)
    body = sp.iloc[60:]
    assert (body["spread_state"] == CLEAN).mean() > 0.8
    assert body["friendliness"].mean() > 0.7


def test_spread_widening_and_dangerous_on_blowout():
    df = _spread_df()
    # Blow the spread out wide for bars 150..165.
    df.loc[150:165, "bid"] = df.loc[150:165, "mark"] - 5.0
    df.loc[150:165, "ask"] = df.loc[150:165, "mark"] + 5.0
    df.loc[150:165, "bid_qty"] = 5.0
    df.loc[150:165, "ask_qty"] = 5.0
    sp = spread_friendliness_frame(df)
    blown = sp.loc[150:165]
    assert (blown["spread_state"] != CLEAN).any()
    assert (blown["spread_state"] == DANGEROUS).any()
    # Friendliness collapses in the blowout.
    assert blown["friendliness"].min() < 0.3


def test_spread_recovers_to_clean_after_blowout():
    df = _spread_df()
    df.loc[150:160, "bid"] = df.loc[150:160, "mark"] - 5.0
    df.loc[150:160, "ask"] = df.loc[150:160, "mark"] + 5.0
    sp = spread_friendliness_frame(df)
    # Well after the blowout, state returns to clean.
    assert (sp.loc[180:199, "spread_state"] == CLEAN).mean() > 0.7


def test_is_execution_friendly_blocks_dangerous():
    assert not is_execution_friendly(0.9, DANGEROUS)
    assert is_execution_friendly(0.6, CLEAN)
    assert not is_execution_friendly(0.2, CLEAN)


def test_spread_requires_columns():
    with pytest.raises(ValueError, match="mark"):
        spread_friendliness_frame(pd.DataFrame({"bid": [1.0], "ask": [1.1]}))


# ─────────────────────────────────────────────────────────────────
# Config validation
# ─────────────────────────────────────────────────────────────────

def test_config_validation():
    with pytest.raises(ValueError):
        FairResponseConfig(delta_window=2)
    with pytest.raises(ValueError):
        ResidualConfig(band_window=4)
    with pytest.raises(ValueError):
        SpreadConfig(tight_pct=0.1, wide_pct=0.05)
