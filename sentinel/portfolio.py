"""Portfolio intelligence — positions enriched with Greeks, pairs,
peaks, and the what-if scenario engine.

This is the layer that understands "I hold a call AND a put on the
same underlying": pair detection groups CE/PE legs per underlying ×
expiry, the scenario engine reprices every leg at user-chosen spot
moves (constant-IV Black-Scholes), and the breakeven curve answers
"above what level am I in profit, below what level in loss" — the
chart the founder asked for, computed honestly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .greeks import GreeksView, greeks, implied_vol, reprice
from .kite_client import InstrumentMeta, Quote

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class PositionView:
    tradingsymbol: str
    quantity: int                 # signed: + long, - short
    average_price: float
    ltp: float
    pnl: float
    underlying: str = ""
    strike: float = 0.0
    option_type: str = ""         # CE | PE | "" for non-options
    expiry: Optional[str] = None
    t_years: float = 0.0
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta_per_day: Optional[float] = None
    # session-tracking
    peak_pnl: float = 0.0
    drawdown_from_peak: float = 0.0


@dataclass
class PairView:
    underlying: str
    expiry: Optional[str]
    legs: List[str]                       # tradingsymbols
    combined_pnl: float
    combined_delta: Optional[float]
    note: str = ""


class PortfolioState:
    """Holds the enriched live picture; refreshed by the server loops."""

    def __init__(self) -> None:
        self.positions: Dict[str, PositionView] = {}
        self.pairs: List[PairView] = []
        self.funds: Dict[str, float] = {}
        self.spot: Optional[float] = None
        self.total_pnl: float = 0.0
        self.total_pnl_peak: float = float("-inf")
        self.updated_at_utc: Optional[str] = None
        self._peaks: Dict[str, float] = {}

    # -- enrichment ------------------------------------------------------

    def refresh(self, raw_positions: List[Dict[str, Any]],
                quotes: Dict[str, Quote],
                metas: Dict[str, InstrumentMeta],
                spot: Optional[float],
                funds: Dict[str, float]) -> None:
        now = datetime.now(IST)
        views: Dict[str, PositionView] = {}
        total = 0.0
        for p in raw_positions:
            sym = str(p.get("tradingsymbol") or "")
            qty = int(p.get("quantity") or 0)
            if not sym or qty == 0:
                continue
            q = quotes.get(sym)
            ltp = float(q.ltp) if q else float(p.get("last_price") or 0.0)
            avg = float(p.get("average_price") or 0.0)
            pnl = (ltp - avg) * qty
            meta = metas.get(sym)
            v = PositionView(
                tradingsymbol=sym, quantity=qty, average_price=avg,
                ltp=ltp, pnl=round(pnl, 2),
            )
            if meta and meta.instrument_type in ("CE", "PE") and spot:
                v.underlying = meta.name
                v.strike = meta.strike
                v.option_type = meta.instrument_type
                if meta.expiry:
                    v.expiry = meta.expiry.strftime("%Y-%m-%d")
                    v.t_years = max(
                        (meta.expiry - now).total_seconds(), 3600,
                    ) / (365.0 * 86400.0)
                    iv = implied_vol(ltp, spot, meta.strike, v.t_years,
                                     meta.instrument_type)
                    if iv is not None:
                        g = greeks(spot, meta.strike, v.t_years, iv,
                                   meta.instrument_type)
                        v.iv = round(iv, 4)
                        v.delta = round(g.delta, 4)
                        v.gamma = round(g.gamma, 6)
                        v.theta_per_day = round(g.theta_per_day, 2)
            # session peak tracking
            peak = max(self._peaks.get(sym, float("-inf")), pnl)
            self._peaks[sym] = peak
            v.peak_pnl = round(peak, 2)
            v.drawdown_from_peak = round(peak - pnl, 2)
            views[sym] = v
            total += pnl
        self.positions = views
        self.total_pnl = round(total, 2)
        if total > self.total_pnl_peak:
            self.total_pnl_peak = total
        self.funds = dict(funds)
        self.spot = spot
        self.updated_at_utc = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        self.pairs = self._detect_pairs()

    def _detect_pairs(self) -> List[PairView]:
        groups: Dict[tuple, List[PositionView]] = {}
        for v in self.positions.values():
            if v.option_type in ("CE", "PE"):
                groups.setdefault((v.underlying, v.expiry), []).append(v)
        pairs = []
        for (und, exp), legs in groups.items():
            types = {l.option_type for l in legs}
            if len(legs) < 2 or types != {"CE", "PE"}:
                continue
            combined = sum(l.pnl for l in legs)
            deltas = [l.delta * l.quantity for l in legs if l.delta is not None]
            combined_delta = round(sum(deltas), 1) if deltas else None
            winners = [l for l in legs if l.pnl > 0]
            losers = [l for l in legs if l.pnl < 0]
            note = ""
            if winners and losers:
                w, lo = winners[0], losers[0]
                note = (f"{w.option_type} leg funding the {lo.option_type} "
                        f"leg ({w.pnl:+.0f} vs {lo.pnl:+.0f})")
            pairs.append(PairView(
                underlying=und, expiry=exp,
                legs=[l.tradingsymbol for l in legs],
                combined_pnl=round(combined, 2),
                combined_delta=combined_delta, note=note,
            ))
        return pairs

    # -- scenario engine -----------------------------------------------------

    def scenario_curve(self, move_pct_range: float = 2.0,
                       steps: int = 41) -> Dict[str, Any]:
        """Portfolio P&L vs spot move. For each candidate spot in
        ±move_pct_range%, reprice every option leg at constant IV and
        sum. Returns the curve + breakeven spots (where total P&L
        crosses zero) — the founder's 'above this amount profit,
        below this loss' chart, computed not drawn by hand."""
        if not self.spot or not self.positions:
            return {"points": [], "breakevens": []}
        moves = [(-move_pct_range + i * (2 * move_pct_range) / (steps - 1))
                 for i in range(steps)]
        points = []
        for mv in moves:
            s_new = self.spot * (1.0 + mv / 100.0)
            total = 0.0
            for v in self.positions.values():
                if v.option_type in ("CE", "PE") and v.iv is not None:
                    new_premium = reprice(s_new, v.strike, v.t_years,
                                          v.iv, v.option_type)
                    total += (new_premium - v.average_price) * v.quantity
                else:
                    total += v.pnl     # non-option legs held flat
            points.append({"move_pct": round(mv, 3),
                           "spot": round(s_new, 1),
                           "pnl": round(total, 0)})
        breakevens = []
        for a, b in zip(points, points[1:]):
            if a["pnl"] == 0 or (a["pnl"] < 0) != (b["pnl"] < 0):
                # linear interpolation of the crossing spot
                pa, pb = a["pnl"], b["pnl"]
                if pb != pa:
                    frac = abs(pa) / abs(pb - pa)
                    breakevens.append(round(
                        a["spot"] + frac * (b["spot"] - a["spot"]), 1))
        return {"points": points, "breakevens": breakevens}

    def what_if(self, move_points: float) -> List[Dict[str, Any]]:
        """Per-leg answer to the founder's slider: 'if the index moves
        by X points, how much cheaper/dearer does each option get?'"""
        out = []
        if not self.spot:
            return out
        s_new = self.spot + move_points
        for v in self.positions.values():
            if v.option_type not in ("CE", "PE") or v.iv is None:
                continue
            new_premium = reprice(s_new, v.strike, v.t_years, v.iv,
                                  v.option_type)
            out.append({
                "tradingsymbol": v.tradingsymbol,
                "current_premium": v.ltp,
                "premium_at_move": round(new_premium, 2),
                "premium_change": round(new_premium - v.ltp, 2),
                "pnl_change": round((new_premium - v.ltp) * v.quantity, 0),
            })
        return out

    # -- serialisation for the dashboard ------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "updated_at_utc": self.updated_at_utc,
            "spot": self.spot,
            "funds": self.funds,
            "total_pnl": self.total_pnl,
            "total_pnl_peak": (None if self.total_pnl_peak == float("-inf")
                               else round(self.total_pnl_peak, 2)),
            "positions": [vars(v) for v in self.positions.values()],
            "pairs": [vars(p) for p in self.pairs],
        }
