"""Tests for the cost-sensitivity analysis aggregation.

The verdict the script renders hinges on _summarise correctly computing, per
mode, gross_R (cost-free ceiling), net_R (after costs), and cost_R (the drag).
These tests pin that arithmetic with a hand-built trade frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.cost_sensitivity import _summarise


def _trade_row(mode, gross_pnl, total_cost, risk_inr, entry=100.0, qty=1):
    net_pnl = gross_pnl - total_cost
    return {
        "mode": mode,
        "gross_pnl": gross_pnl,
        "total_cost": total_cost,
        "net_pnl": net_pnl,
        "risk_inr": risk_inr,
        "net_r": net_pnl / risk_inr,
        "entry": entry,
        "quantity": qty,
    }


def test_summarise_empty_frame():
    assert _summarise(pd.DataFrame()).empty


def test_gross_net_cost_r_arithmetic():
    # Two trades in one mode. risk=10 each.
    #  trade A: gross +20 (gross_R +2.0), cost 2 -> net +18 (net_R +1.8)
    #  trade B: gross -10 (gross_R -1.0), cost 2 -> net -12 (net_R -1.2)
    # mean gross_R = +0.5, mean net_R = +0.3, cost_R = +0.2
    df = pd.DataFrame([
        _trade_row("touch_confirmed", 20.0, 2.0, 10.0),
        _trade_row("touch_confirmed", -10.0, 2.0, 10.0),
    ])
    out = _summarise(df).set_index("mode")
    row = out.loc["touch_confirmed"]
    assert row["trades"] == 2
    assert abs(row["gross_R"] - 0.5) < 1e-9
    assert abs(row["net_R"] - 0.3) < 1e-9
    assert abs(row["cost_R"] - 0.2) < 1e-9
    assert row["win"] == 0.5            # one of two net_pnl > 0


def test_profit_factor_and_all_losses_finite():
    # All losing trades -> PF is 0 (wins sum 0 / losses > 0), finite.
    df = pd.DataFrame([
        _trade_row("blind_limit", -5.0, 1.0, 10.0),
        _trade_row("blind_limit", -8.0, 1.0, 10.0),
    ])
    out = _summarise(df).set_index("mode")
    assert out.loc["blind_limit", "PF"] == 0.0
    assert out.loc["blind_limit", "win"] == 0.0


def test_avg_notional_uses_entry_times_quantity():
    df = pd.DataFrame([
        _trade_row("touch_confirmed", 1.0, 0.1, 5.0, entry=200.0, qty=50),
        _trade_row("touch_confirmed", 1.0, 0.1, 5.0, entry=100.0, qty=50),
    ])
    out = _summarise(df).set_index("mode")
    # (200*50 + 100*50)/2 = (10000 + 5000)/2 = 7500
    assert out.loc["touch_confirmed", "avg_notional_inr"] == 7500.0


def test_multiple_modes_separated():
    df = pd.DataFrame([
        _trade_row("touch_confirmed", 20.0, 2.0, 10.0),
        _trade_row("blind_limit", -10.0, 2.0, 10.0),
    ])
    out = _summarise(df)
    assert set(out["mode"]) == {"touch_confirmed", "blind_limit"}
