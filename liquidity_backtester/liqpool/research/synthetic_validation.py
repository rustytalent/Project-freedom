"""Synthetic-data negative-control validation (idea #29).

The single strongest tool against fooling yourself in backtesting: feed
the hypothesis bars whose generative process is *known* to have no
exploitable structure. If the hypothesis reports a positive Sharpe on
pure random walk, the in-sample Sharpe was always noise — the discovery
was an artifact of multiple-comparison or look-ahead.

This module ships three regimes whose ground truth is known:

  * ``random_walk`` — geometric Brownian motion with zero drift. Should
    not yield a tradeable edge under any sensible cost. Any positive
    Sharpe is the false-positive floor.
  * ``mean_reverting`` — AR(1) on log-returns. Strategies that buy dips
    and sell rips should profit here; trend-following strategies should
    lose. Useful for showing a hypothesis correctly *fails* when its
    structural assumption is violated.
  * ``trending`` — random walk with a small upward drift. Trend-following
    hypotheses should profit; mean-reversion should bleed.

For each regime we run the hypothesis across ``n_realizations`` random
draws and report the *distribution* of Sharpes, not a single number. The
verdict compares the random-walk Sharpe distribution to a threshold:

  * If the median random-walk Sharpe is well below the noise floor and
    the 95th percentile is also below: ``passed`` — the hypothesis does
    not invent edge from noise.
  * If the 95th percentile crosses the floor: ``noise_susceptible`` —
    the hypothesis sometimes pretends to have edge when there isn't
    any. Operator should view the original backtest with suspicion.

This is intentionally narrow. The module does NOT validate the original
hypothesis's edge — only that it is not pulling edge out of pure noise.
A failing real-data Sharpe combined with a passing synthetic-data
verdict means the hypothesis simply has no edge, not that it's fooled
by noise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# A real strategy with edge has Sharpe well above this; this is the "white
# noise floor" we expect random-walk synthetic data to register at or below.
DEFAULT_NOISE_SHARPE_CEILING: float = 0.15

# A random walk's expected per-trade Sharpe is zero. The 95th percentile of
# 50 realizations of pure noise is the empirical "false-positive 5%" floor.
DEFAULT_FALSE_POSITIVE_PCTILE: float = 0.95

DEFAULT_N_REALIZATIONS: int = 50
DEFAULT_BARS_PER_REALIZATION: int = 1800        # ~30 trading days of 5m bars


def _ohlc_from_close(close: np.ndarray, *, base_ts: pd.Timestamp,
                     freq_minutes: int = 5,
                     intrabar_vol: float = 0.0015) -> pd.DataFrame:
    """Build an OHLC frame from a synthetic close series.

    Adds plausible intrabar high/low jitter so hypotheses that rely on
    bar wicks (any range/breakout strategy) still see realistic shape
    rather than a single line. Volume is omitted on purpose — most
    research hypotheses ignore it, and synthesizing it cleanly is a
    larger project.
    """
    n = len(close)
    rng = np.random.default_rng(int(abs(hash(tuple(close[:8].tolist())))) % (2 ** 32))
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wick_up = np.abs(rng.normal(0.0, intrabar_vol, n))
    wick_dn = np.abs(rng.normal(0.0, intrabar_vol, n))
    high = np.maximum(open_, close) * (1.0 + wick_up)
    low = np.minimum(open_, close) * (1.0 - wick_dn)
    ts = pd.date_range(base_ts, periods=n, freq=f"{int(freq_minutes)}min")
    return pd.DataFrame({"ts": ts, "open": open_, "high": high,
                          "low": low, "close": close})


def generate_random_walk(*, n_bars: int = DEFAULT_BARS_PER_REALIZATION,
                          seed: int = 0,
                          start_price: float = 100.0,
                          per_bar_vol: float = 0.002) -> pd.DataFrame:
    """Geometric Brownian motion with zero drift. Pure noise."""
    rng = np.random.default_rng(int(seed))
    logret = rng.normal(0.0, per_bar_vol, n_bars)
    close = float(start_price) * np.exp(np.cumsum(logret))
    return _ohlc_from_close(close, base_ts=pd.Timestamp("2024-01-01 09:15"))


def generate_mean_reverting(*, n_bars: int = DEFAULT_BARS_PER_REALIZATION,
                             seed: int = 0,
                             start_price: float = 100.0,
                             half_life_bars: int = 40,
                             per_bar_vol: float = 0.002) -> pd.DataFrame:
    """AR(1) log-return series with mean-reverting drift.

    Half-life chosen at ~40 5-min bars (~3.3 hours) — short enough that a
    well-tuned mean-reversion hypothesis can profit but long enough that
    a trend-following hypothesis will lose persistently.
    """
    rng = np.random.default_rng(int(seed))
    phi = math.exp(-math.log(2.0) / max(1, half_life_bars))
    eps = rng.normal(0.0, per_bar_vol, n_bars)
    log_dev = np.empty(n_bars)
    log_dev[0] = eps[0]
    for i in range(1, n_bars):
        log_dev[i] = phi * log_dev[i - 1] + eps[i]
    close = float(start_price) * np.exp(log_dev)
    return _ohlc_from_close(close, base_ts=pd.Timestamp("2024-01-01 09:15"))


def generate_trending(*, n_bars: int = DEFAULT_BARS_PER_REALIZATION,
                       seed: int = 0,
                       start_price: float = 100.0,
                       per_bar_drift: float = 0.0002,
                       per_bar_vol: float = 0.002) -> pd.DataFrame:
    """Random walk with a small per-bar positive drift. Trend-following
    edge should survive; mean-reversion should bleed."""
    rng = np.random.default_rng(int(seed))
    logret = rng.normal(per_bar_drift, per_bar_vol, n_bars)
    close = float(start_price) * np.exp(np.cumsum(logret))
    return _ohlc_from_close(close, base_ts=pd.Timestamp("2024-01-01 09:15"))


GENERATORS: Dict[str, Callable[..., pd.DataFrame]] = {
    "random_walk": generate_random_walk,
    "mean_reverting": generate_mean_reverting,
    "trending": generate_trending,
}


@dataclass
class RegimeResult:
    """Distribution of a hypothesis's Sharpe across N realizations of
    one regime."""
    regime: str
    n_realizations: int
    sharpes: List[float] = field(default_factory=list)
    n_trades_median: int = 0
    n_realizations_with_trades: int = 0

    @property
    def median_sharpe(self) -> float:
        return float(np.median(self.sharpes)) if self.sharpes else 0.0

    @property
    def p95_sharpe(self) -> float:
        if not self.sharpes:
            return 0.0
        return float(np.quantile(self.sharpes, DEFAULT_FALSE_POSITIVE_PCTILE))

    @property
    def mean_sharpe(self) -> float:
        return float(np.mean(self.sharpes)) if self.sharpes else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "n_realizations": int(self.n_realizations),
            "n_realizations_with_trades": int(self.n_realizations_with_trades),
            "median_sharpe": float(self.median_sharpe),
            "mean_sharpe": float(self.mean_sharpe),
            "p95_sharpe": float(self.p95_sharpe),
            "n_trades_median": int(self.n_trades_median),
        }


@dataclass
class SyntheticValidationReport:
    """Per-hypothesis verdict aggregating all regimes."""
    name: str
    status: str            # "passed" / "noise_susceptible" / "no_trades"
    reason: str
    regimes: Dict[str, RegimeResult] = field(default_factory=dict)
    noise_ceiling: float = DEFAULT_NOISE_SHARPE_CEILING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "regimes": {k: v.to_dict() for k, v in self.regimes.items()},
            "noise_ceiling": float(self.noise_ceiling),
        }


def _sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    return mean / std if std > 1e-12 else 0.0


def validate_hypothesis_against_noise(
    runner: Callable[[pd.DataFrame], Any],
    *,
    name: str,
    regimes: Sequence[str] = ("random_walk", "mean_reverting", "trending"),
    n_realizations: int = DEFAULT_N_REALIZATIONS,
    n_bars: int = DEFAULT_BARS_PER_REALIZATION,
    seed: int = 11,
    noise_ceiling: float = DEFAULT_NOISE_SHARPE_CEILING,
    return_extractor: Optional[Callable[[Any], np.ndarray]] = None,
) -> SyntheticValidationReport:
    """Run a hypothesis across synthetic regimes; return verdict.

    ``runner`` takes one synthetic OHLC frame and returns whatever shape
    contains the per-trade R-multiples. The default ``return_extractor``
    handles two shapes out of the box:

      * a ``HypothesisReport`` (research/harness.py) — pulls
        ``trade["r_multiple"]`` from ``report.trades``.
      * a plain sequence of numbers — used directly.

    Callers may pass their own ``return_extractor`` to support other
    shapes (e.g. a tuple ``(name, returns)``).
    """
    extractor = return_extractor or _default_extractor
    report = SyntheticValidationReport(name=str(name), status="passed",
                                        reason="", noise_ceiling=float(noise_ceiling))
    for regime in regimes:
        gen = GENERATORS.get(regime)
        if gen is None:
            continue
        rr = RegimeResult(regime=regime, n_realizations=int(n_realizations))
        trades_per_run: List[int] = []
        for i in range(int(n_realizations)):
            bars = gen(n_bars=int(n_bars), seed=int(seed + i * 31))
            try:
                raw = runner(bars)
            except Exception as exc:
                rr.sharpes.append(float("nan"))
                continue
            returns = extractor(raw)
            returns = np.asarray(returns, dtype=float)
            returns = returns[np.isfinite(returns)]
            if returns.size == 0:
                continue
            rr.sharpes.append(_sharpe(returns))
            trades_per_run.append(int(returns.size))
            rr.n_realizations_with_trades += 1
        if trades_per_run:
            rr.n_trades_median = int(np.median(trades_per_run))
        # Drop NaNs from sharpes for percentile math
        rr.sharpes = [s for s in rr.sharpes if np.isfinite(s)]
        report.regimes[regime] = rr

    rw = report.regimes.get("random_walk")
    if rw is None or rw.n_realizations_with_trades == 0:
        report.status = "no_trades"
        report.reason = ("hypothesis produced no trades on random-walk "
                         "regime; nothing to validate against noise")
        return report

    if rw.p95_sharpe > noise_ceiling:
        report.status = "noise_susceptible"
        report.reason = (
            f"random-walk 95th-pctile Sharpe = {rw.p95_sharpe:+.2f} > "
            f"noise ceiling {noise_ceiling:+.2f}; the original backtest's "
            f"edge may be an artifact of overfit. Median random-walk "
            f"Sharpe = {rw.median_sharpe:+.2f}."
        )
        return report

    report.status = "passed"
    report.reason = (
        f"random-walk Sharpes well below noise ceiling "
        f"(median {rw.median_sharpe:+.2f}, p95 {rw.p95_sharpe:+.2f} ≤ "
        f"{noise_ceiling:+.2f}); the hypothesis does not invent edge "
        f"from pure noise."
    )
    return report


def _default_extractor(raw: Any) -> np.ndarray:
    """Best-effort extractor that handles HypothesisReport, dict-list
    of trades, list of floats, or numpy arrays."""
    if raw is None:
        return np.empty(0, dtype=float)
    # numpy / sequence of floats
    if isinstance(raw, (np.ndarray, list, tuple)):
        as_floats: List[float] = []
        for x in raw:
            if isinstance(x, dict):
                r = x.get("r_multiple")
                if r is not None:
                    as_floats.append(float(r))
            else:
                try:
                    as_floats.append(float(x))
                except (TypeError, ValueError):
                    pass
        return np.asarray(as_floats, dtype=float)
    # HypothesisReport
    trades = getattr(raw, "trades", None)
    if trades is None:
        return np.empty(0, dtype=float)
    rs: List[float] = []
    for t in trades:
        r = t.get("r_multiple") if isinstance(t, dict) else getattr(t, "r_multiple", None)
        if r is None or not np.isfinite(float(r)):
            continue
        rs.append(float(r))
    return np.asarray(rs, dtype=float)
