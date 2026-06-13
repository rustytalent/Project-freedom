"""Top-10 NIFTY equity contextual layer.

The founder's insight: NIFTY moves because its top constituents move.
RELIANCE + HDFCBANK + ICICIBANK + INFY + TCS + ITC + LT + AXIS + KOTAK
+ SBIN carry roughly half the index weight. When the index is FLAT but
RELIANCE is +2%, the index is masking a bullish dispersion — the kind
of regime where a directional CE works even though the screen says
"chop". And the opposite: index +0.5% on broad weakness with two
heavyweights covering for everyone is a fragile rally.

This module computes:

  * contribution to NIFTY per stock (weight * return)
  * residual = stock_return - weight_adjusted_index_return
  * hidden-bull / hidden-bear / true-flat verdict
  * breadth (count up / down within the top 10)

Methodology: index-decomposition / dispersion analytics (standard at
prop desks — published as 'CBOE Implied Correlation' for SPX, no
equivalent listed for NIFTY which is why we compute it directly).
Sentinel tier: PRO. Feeds the `MarketSnapshot.context` so scientists can
read the regime.

The weights are approximate (updated quarterly via NSE indices). When
NSE publishes new weights, edit ``NIFTY_TOP10_WEIGHTS`` — the
``effective_date`` field tracks when this snapshot was last refreshed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from .io_decl import IOSpec, declare


# NIFTY 50 top-10 weights as of Q2-2026 (approx; refresh quarterly via NSE).
NIFTY_TOP10_WEIGHTS: Dict[str, float] = {
    "RELIANCE":  0.0998,
    "HDFCBANK":  0.0892,
    "ICICIBANK": 0.0772,
    "INFY":      0.0612,
    "TCS":       0.0432,
    "ITC":       0.0408,
    "LT":        0.0387,
    "AXISBANK":  0.0322,
    "KOTAKBANK": 0.0298,
    "SBIN":      0.0276,
}
NIFTY_TOP10_EFFECTIVE_DATE = "2026-04-01"


@dataclass
class Contribution:
    symbol: str
    weight: float
    return_pct: float          # stock return today, %
    contribution_pct: float    # weight * return_pct (in % of index)
    residual_pct: float        # return_pct - index_return (idiosyncratic)


@dataclass
class EquityContext:
    index_return_pct: float
    top10_summed_contribution_pct: float
    breadth_up: int
    breadth_down: int
    regime: str                # "HIDDEN_BULL" | "HIDDEN_BEAR" | "BROAD_BULL" | "BROAD_BEAR" | "TRUE_FLAT"
    leaders: List[Contribution] = field(default_factory=list)
    laggards: List[Contribution] = field(default_factory=list)
    contributions: List[Contribution] = field(default_factory=list)
    effective_date: str = NIFTY_TOP10_EFFECTIVE_DATE


def _classify(index_return: float,
              top10_contrib: float,
              breadth_up: int) -> str:
    """The regime classifier.

    HIDDEN_BULL: index < +0.2% BUT top-10 contributed > +0.4% — heavyweights
                 are pulling but broad market is dragging. A directional CE
                 trade still has wind.
    HIDDEN_BEAR: index > -0.2% BUT top-10 contributed < -0.4% — heavyweights
                 are dragging despite a flat index. Fragile rally.
    BROAD_BULL : index > +0.2% AND breadth_up >= 7 — real strength.
    BROAD_BEAR : index < -0.2% AND breadth_up <= 3 — real weakness.
    TRUE_FLAT  : index and top-10 contribution both within +/- 0.2%.
    """
    if index_return < 0.2 and top10_contrib > 0.4:
        return "HIDDEN_BULL"
    if index_return > -0.2 and top10_contrib < -0.4:
        return "HIDDEN_BEAR"
    if index_return > 0.2 and breadth_up >= 7:
        return "BROAD_BULL"
    if index_return < -0.2 and breadth_up <= 3:
        return "BROAD_BEAR"
    return "TRUE_FLAT"


def top10_contextual_layer(stock_returns_pct: Mapping[str, float],
                            index_return_pct: float,
                            weights: Mapping[str, float] | None = None
                            ) -> EquityContext:
    """Compute the equity contextual layer for a NIFTY session.

    Parameters
    ----------
    stock_returns_pct
        ``{symbol: today's return in %}``. Missing symbols default to 0.
    index_return_pct
        NIFTY 50 return in % for the same period.
    weights
        Override the canned weights (useful when NSE publishes a refresh
        between releases). Defaults to ``NIFTY_TOP10_WEIGHTS``.

    Returns
    -------
    ``EquityContext`` with full per-stock decomposition + regime label.
    """
    w = dict(weights or NIFTY_TOP10_WEIGHTS)
    contribs: List[Contribution] = []
    breadth_up = breadth_down = 0
    summed = 0.0
    for sym, weight in w.items():
        r = float(stock_returns_pct.get(sym, 0.0))
        c = weight * r
        residual = r - index_return_pct
        contribs.append(Contribution(
            symbol=sym, weight=round(weight, 4), return_pct=round(r, 2),
            contribution_pct=round(c, 3), residual_pct=round(residual, 2),
        ))
        summed += c
        if r > 0:
            breadth_up += 1
        elif r < 0:
            breadth_down += 1
    regime = _classify(index_return_pct, summed, breadth_up)
    contribs.sort(key=lambda x: x.contribution_pct, reverse=True)
    return EquityContext(
        index_return_pct=round(index_return_pct, 2),
        top10_summed_contribution_pct=round(summed, 3),
        breadth_up=breadth_up,
        breadth_down=breadth_down,
        regime=regime,
        leaders=contribs[:3],
        laggards=contribs[-3:],
        contributions=contribs,
    )


def weightage_divergence(stock_returns_pct: Mapping[str, float],
                          index_return_pct: float) -> float:
    """The single scalar: top-10 weighted contribution minus index return.
    Positive = heavyweights leading the index; negative = index leading
    despite heavyweight weakness. Tier: RETAIL."""
    ctx = top10_contextual_layer(stock_returns_pct, index_return_pct)
    return round(ctx.top10_summed_contribution_pct - index_return_pct, 3)


declare(IOSpec(
    module="sentinel.equity_layer",
    purpose="NIFTY top-10 contextual layer — heavyweight contribution + "
            "breadth + regime classification (HIDDEN_BULL / HIDDEN_BEAR / "
            "BROAD_BULL / BROAD_BEAR / TRUE_FLAT) so scientists can read the "
            "real regime under a flat-looking index",
    inputs=["stock returns map", "index return", "optional weight override"],
    outputs=["EquityContext (regime, leaders, laggards, per-stock contribs)",
             "weightage_divergence scalar"],
    consumes_from=["sentinel.kite_client (top-10 quotes)"],
    produces_for=["sentinel.scientists (MarketSnapshot.context)",
                  "sentinel.reports"],
    tier="TRUSTED",
))
