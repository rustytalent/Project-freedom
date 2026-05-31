"""Alpha arsenal foundation: AlphaSignal + Alpha abstract base class.

An ALPHA is an independent signal generator. The contract:

  alpha.name        — unique string identifier.
  alpha.candidates() — given a per-asset df_base + atr_series + symbol/sector
                       metadata, returns a list of AlphaSignal objects.
  alpha.regime_tags() — given a signal, returns a dict of regime dimensions
                       (session, vol_regime, day_of_week, ...) used by the
                       evaluator to stratify performance.

An AlphaSignal is a SINGLE candidate trade. It does NOT specify cost,
fill price, or realized P&L — those are computed downstream by the
evaluator under the SHARED rules every alpha is subject to (MIS exits,
realistic Zerodha costs, identical slippage model). This separation
ensures every alpha is judged on the same playing field: no alpha can
hide behind generous assumptions.

The signal carries the alpha's PREFERRED barriers (stop_atr, target_atr,
horizon_bars). Different alphas can prefer different geometries — a
mean-reversion alpha might want stop_atr=0.5 / target_atr=1.5 / horizon=12,
while a momentum-continuation alpha might want 0.75 / 2.0 / 24. The
evaluator respects each alpha's preferences but applies the same MIS cap.

Why an abstract base class instead of duck-typed callables: the explicit
contract lets the registry/evaluator validate before running, surfaces
introspection (regime_tags) as a first-class concern, and forces every
new alpha author to make their design choices visible in code rather than
buried in defaults.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class AlphaSignal:
    """A single trade candidate emitted by an alpha at a specific bar.

    ``decision_at`` is the bar timestamp at which the signal *would have
    been actionable* — i.e. the entry decision is made on this bar's
    close. The evaluator enters the trade at the NEXT bar's open to avoid
    look-ahead bias (same convention as the V1 execution simulator for
    confirmation modes).

    ``entry_reference`` is the alpha's notional entry price (typically the
    current close, or pool boundary, or whatever the alpha thinks is the
    fair entry). The evaluator may adjust this for realistic fill.
    """
    alpha_name: str
    symbol: str
    decision_at: pd.Timestamp
    decision_idx: int                    # bar index in df_base for fast lookup
    side: str                            # "long" or "short"
    entry_reference: float
    stop_atr: float                      # alpha-preferred stop in ATR units
    target_atr: float                    # alpha-preferred target in ATR units
    horizon_bars: int                    # alpha-preferred max-hold
    confidence: float = 0.5              # 0-1, alpha's internal confidence
    state: Dict[str, Any] = field(default_factory=dict)   # introspection only

    def __post_init__(self) -> None:
        if self.side not in ("long", "short"):
            raise ValueError(f"side must be 'long' or 'short', got {self.side!r}")
        if self.stop_atr <= 0:
            raise ValueError(f"stop_atr must be > 0, got {self.stop_atr}")
        if self.target_atr <= 0:
            raise ValueError(f"target_atr must be > 0, got {self.target_atr}")
        if self.horizon_bars <= 0:
            raise ValueError(f"horizon_bars must be > 0, got {self.horizon_bars}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


class Alpha(ABC):
    """Abstract base class for an independent signal generator.

    Subclasses MUST implement :meth:`name`, :meth:`description`, and
    :meth:`candidates`. They MAY override :meth:`regime_tags`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used in the registry, reports, and CSV outputs."""

    @property
    def description(self) -> str:
        """One-paragraph description of the alpha's thesis. Default empty."""
        return ""

    @abstractmethod
    def candidates(self,
                   *,
                   symbol: str,
                   df_base: pd.DataFrame,
                   atr_series: pd.Series,
                   extra: Optional[Dict[str, Any]] = None,
                   ) -> List[AlphaSignal]:
        """Walk through df_base bars and emit AlphaSignal candidates.

        ``df_base`` is the per-asset OHLCV (5-min). ``atr_series`` is the
        precomputed ATR aligned to df_base's index. ``extra`` is an
        alpha-specific kwargs bag — e.g. for the pool-reach alpha it
        contains the asset's pool list; for the force-model alpha it
        contains the saved force model and feature inputs.

        Implementations should be DETERMINISTIC and ITERATE FORWARD in
        time. They should NOT look at bars at index > the current
        decision_idx when deciding to emit a signal at that bar.
        """

    def regime_tags(self, signal: AlphaSignal,
                    df_base: pd.DataFrame, atr_series: pd.Series,
                    ) -> Dict[str, str]:
        """Return regime stratification tags for the signal.

        Default implementation tags by IST session and trade side. Override
        in subclasses to add alpha-specific dimensions (e.g.
        ``factor_type`` for pool-reach, ``zscore_bucket`` for mean
        reversion).
        """
        from liqpool.execution_backtest import _ist_minute_of_day
        from liqpool.regime import nse_session
        ts = signal.decision_at
        return {
            "session": nse_session(ts),
            "side": signal.side,
            "ist_minute_of_day_bucket": _bucket_minute(
                _ist_minute_of_day(ts)
            ),
        }


def _bucket_minute(minute_of_day: int) -> str:
    """Coarse IST minute-of-day bucket: pre / morning / midday / afternoon / post."""
    if minute_of_day < 9 * 60 + 15:
        return "pre_open"
    if minute_of_day < 10 * 60 + 15:
        return "open_hour"
    if minute_of_day < 12 * 60:
        return "morning"
    if minute_of_day < 13 * 60 + 30:
        return "midday"
    if minute_of_day <= 15 * 60 + 30:
        return "afternoon"
    return "post_close"


def signals_to_frame(signals: Iterable[AlphaSignal]) -> pd.DataFrame:
    """Convenience: convert a list of AlphaSignal into a flat DataFrame.

    Used by the evaluator and reports. Drops the opaque ``state`` field
    since it can contain arbitrary objects; if a caller needs it they can
    inspect the AlphaSignal objects directly.
    """
    rows = []
    for s in signals:
        rows.append({
            "alpha_name": s.alpha_name,
            "symbol": s.symbol,
            "decision_at": s.decision_at,
            "decision_idx": s.decision_idx,
            "side": s.side,
            "entry_reference": s.entry_reference,
            "stop_atr": s.stop_atr,
            "target_atr": s.target_atr,
            "horizon_bars": s.horizon_bars,
            "confidence": s.confidence,
        })
    return pd.DataFrame(rows)
