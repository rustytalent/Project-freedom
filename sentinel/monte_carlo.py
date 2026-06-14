"""Monte Carlo Lite — 1k GBM paths over the next 5/15/30 min for the
current book.

The founder asked for "not heavy ones, we can just give it once a month
according to the plans" — so this module is intentionally LIGHT:

  * Single-asset geometric Brownian motion with drift = 0 and σ drawn
    from realised vol over the latest spot history.
  * 1000 paths, three horizons, deterministic seed → reproducible.
  * Output is intentionally narrow: P(profit), P(stop), P(target),
    expected R-multiple distribution, worst-10% and best-10% bands.

This is NOT a derivative-pricing engine. It's a fast scenario probe
the operator runs in 50 ms. The institutional layer (SVI, Cornish-
Fisher VaR, etc.) lives in ``sentinel.institutional``.

Methodology citation: Boyle (1977) "Options: A Monte Carlo Approach."
Wikipedia: Geometric Brownian Motion. Tier: PRO.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .io_decl import IOSpec, declare


@dataclass(frozen=True)
class LegPayoff:
    """One leg of the book in the Monte Carlo. Premium response is a
    linear delta + 0.5*gamma*move² approximation; good enough for 5-30
    min forecasts on near-ATM contracts (the institutional layer can
    do the SVI re-price for longer horizons)."""
    symbol: str
    qty: int                            # signed
    entry_premium: float
    delta: float = 0.0
    gamma: float = 0.0
    theta_per_day: float = 0.0


@dataclass
class HorizonResult:
    horizon_min: int
    n_paths: int
    p_profit: float
    p_target: float
    p_stop: float
    mean_r: float
    median_r: float
    p10_r: float
    p90_r: float
    expected_pnl_rupees: float

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MonteCarloResult:
    asset: str
    spot: float
    sigma_annualised: float
    horizons: List[HorizonResult] = field(default_factory=list)

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["horizons"] = [h.to_row() for h in self.horizons]
        return d


def realised_sigma_per_day(prices: Sequence[float], annualise: bool = True
                            ) -> float:
    """Daily log-return σ from a short price series (e.g. spot ticks
    in a session). Returns annualised σ if ``annualise``."""
    if len(prices) < 3:
        return 0.15
    rets = []
    for a, b in zip(prices, prices[1:]):
        if a > 0 and b > 0:
            rets.append(math.log(b / a))
    if not rets or statistics.pstdev(rets) == 0:
        return 0.15
    sd = statistics.pstdev(rets)
    # roughly 75 minutes between consecutive 1-s ticks for sessions of
    # interest; scale to daily using session/total
    minutes_per_step = 1.0
    steps_per_day = 6.25 * 60 / minutes_per_step
    daily_sd = sd * math.sqrt(steps_per_day)
    if annualise:
        return daily_sd * math.sqrt(252)
    return daily_sd


def _leg_pnl_at_spot(legs: Sequence[LegPayoff], spot: float,
                     base_spot: float, time_passed_min: float) -> float:
    """Sum of legs' P&L when spot moves base_spot -> spot over
    ``time_passed_min`` minutes."""
    move = spot - base_spot
    days = time_passed_min / (6.25 * 60.0)
    pnl = 0.0
    for l in legs:
        d_premium = (l.delta * move
                     + 0.5 * l.gamma * move * move
                     + l.theta_per_day * days)
        pnl += l.qty * d_premium
    return pnl


def simulate(legs: Sequence[LegPayoff],
             spot: float,
             sigma_annualised: float = 0.15,
             horizons_min: Sequence[int] = (5, 15, 30),
             n_paths: int = 1000,
             target_pnl_rupees: float = 1500.0,
             stop_pnl_rupees: float = 1500.0,
             seed: int = 7) -> MonteCarloResult:
    """Run ``n_paths`` GBM paths from ``spot`` with the given σ, evaluate
    each leg's payoff at every horizon, return the standard
    institutional Monte Carlo readouts."""
    rng = random.Random(seed)
    out = MonteCarloResult(asset="NIFTY", spot=float(spot),
                           sigma_annualised=round(sigma_annualised, 4))
    if not legs or spot <= 0 or sigma_annualised <= 0:
        return out
    minutes_per_year = 252 * 6.25 * 60
    for horizon in horizons_min:
        dt = horizon / minutes_per_year     # in trading years
        sd = sigma_annualised * math.sqrt(dt)
        pnls: List[float] = []
        for _ in range(n_paths):
            z = rng.gauss(0.0, 1.0)
            terminal = spot * math.exp(-0.5 * sd * sd + sd * z)
            pnls.append(_leg_pnl_at_spot(legs, terminal, spot, horizon))
        pnls.sort()
        mean = sum(pnls) / len(pnls)
        median = pnls[len(pnls) // 2]
        p10 = pnls[int(len(pnls) * 0.1)]
        p90 = pnls[int(len(pnls) * 0.9)]
        p_profit = sum(1 for p in pnls if p > 0) / len(pnls)
        p_target = sum(1 for p in pnls if p >= target_pnl_rupees) / len(pnls)
        p_stop = sum(1 for p in pnls if p <= -abs(stop_pnl_rupees)) / len(pnls)
        # R = P&L / stop magnitude (the 'risk dollar')
        risk = max(abs(stop_pnl_rupees), 1.0)
        out.horizons.append(HorizonResult(
            horizon_min=horizon, n_paths=n_paths,
            p_profit=round(p_profit, 3),
            p_target=round(p_target, 3),
            p_stop=round(p_stop, 3),
            mean_r=round(mean / risk, 3),
            median_r=round(median / risk, 3),
            p10_r=round(p10 / risk, 3),
            p90_r=round(p90 / risk, 3),
            expected_pnl_rupees=round(mean, 0),
        ))
    return out


declare(IOSpec(
    module="sentinel.monte_carlo",
    purpose="light scenario probe — 1000 GBM paths over next 5/15/30 min "
            "on the held book, returns P(profit/target/stop) + R-multiple "
            "distribution. Boyle 1977 methodology, intentionally tiny so "
            "it runs in 50 ms every cycle a customer requests",
    inputs=["LegPayoff list (signed qty + delta/gamma/theta)",
            "current spot + realised σ + horizon list"],
    outputs=["MonteCarloResult with per-horizon p_profit / p_target / "
             "p_stop / mean R / quantile R"],
    consumes_from=["sentinel.portfolio (legs)", "sentinel.live_publisher (spot σ)"],
    produces_for=["sentinel.server", "sentinel.live_publisher (monte_carlo signal)"],
    tier="TRUSTED",
))
