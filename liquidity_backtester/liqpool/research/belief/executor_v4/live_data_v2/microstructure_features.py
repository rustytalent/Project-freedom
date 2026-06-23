"""Per-slot microstructure feature extraction.

Combines a single EnrichedQuote + OI history + cancel-rate history
into one numerical feature vector that the new V2 detectors consume.

Output is a small dataclass (not a dict) so detectors get attribute
access with autocompletion and zero string lookups in hot paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .cancel_rate_tracker import CancelRateReading, CancelRateTracker
from .enriched_quote import EnrichedQuote, extract_pressure_ratio
from .oi_history import OIHistoryTracker, OIVelocityReading


@dataclass(frozen=True)
class SlotMicrostructureFeatures:
    """All microstructure-derived features for one slot."""
    tradingsymbol: str
    strike: float
    option_type: str
    # OI dynamics.
    oi: float
    oi_velocity_per_min: float
    oi_acceleration_per_min2: float
    # Pending pressure (chain-wide totals + L5-summed depth).
    pressure_ratio: float            # in [-1, +1]
    total_depth_buy_qty: int
    total_depth_sell_qty: int
    buy_quantity_pending: int
    sell_quantity_pending: int
    # Spread + microprice.
    spread: float
    mid: float
    # Persistence / cancel rate.
    persistence_score: float
    cancel_rate_per_tick: float
    # Last trade size — recent trades give us tape pace.
    last_trade_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tradingsymbol": self.tradingsymbol,
            "strike": float(self.strike),
            "option_type": self.option_type,
            "oi": float(self.oi),
            "oi_velocity_per_min": round(self.oi_velocity_per_min, 2),
            "oi_acceleration_per_min2": round(
                self.oi_acceleration_per_min2, 4),
            "pressure_ratio": round(self.pressure_ratio, 4),
            "total_depth_buy_qty": int(self.total_depth_buy_qty),
            "total_depth_sell_qty": int(self.total_depth_sell_qty),
            "buy_quantity_pending": int(self.buy_quantity_pending),
            "sell_quantity_pending": int(self.sell_quantity_pending),
            "spread": round(self.spread, 4),
            "mid": round(self.mid, 4),
            "persistence_score": round(self.persistence_score, 4),
            "cancel_rate_per_tick": round(self.cancel_rate_per_tick, 4),
            "last_trade_size": int(self.last_trade_size),
        }


def build_microstructure_features(*,
                                       quote: EnrichedQuote,
                                       strike: float,
                                       option_type: str,
                                       oi_history: Optional[OIHistoryTracker] = None,
                                       cancel_tracker: Optional[CancelRateTracker] = None,
                                       ) -> SlotMicrostructureFeatures:
    """Build features for one slot. The trackers are observed-in-place
    so the caller gets the latest readings for free."""
    # OI dynamics.
    oi_reading: OIVelocityReading
    if oi_history is not None:
        oi_reading = oi_history.observe(
            strike=strike, option_type=option_type, oi=quote.oi,
        )
    else:
        oi_reading = OIVelocityReading(
            strike=float(strike), option_type=str(option_type),
            oi=quote.oi, oi_velocity_per_min=0.0,
            oi_acceleration_per_min2=0.0, n_samples=0,
        )
    # Cancel rate.
    cancel_reading: Optional[CancelRateReading] = None
    if cancel_tracker is not None:
        cancel_reading = cancel_tracker.observe(
            tradingsymbol=quote.tradingsymbol,
            depth_buy=[d.to_dict() for d in quote.depth_buy],
            depth_sell=[d.to_dict() for d in quote.depth_sell],
        )
    persistence = (cancel_reading.persistence_score
                    if cancel_reading is not None else 1.0)
    cancel_rate = (cancel_reading.cancel_rate_per_tick
                    if cancel_reading is not None else 0.0)

    return SlotMicrostructureFeatures(
        tradingsymbol=quote.tradingsymbol,
        strike=float(strike),
        option_type=str(option_type),
        oi=float(quote.oi),
        oi_velocity_per_min=float(oi_reading.oi_velocity_per_min),
        oi_acceleration_per_min2=float(
            oi_reading.oi_acceleration_per_min2),
        pressure_ratio=float(extract_pressure_ratio(quote)),
        total_depth_buy_qty=int(quote.total_depth_buy_qty),
        total_depth_sell_qty=int(quote.total_depth_sell_qty),
        buy_quantity_pending=int(quote.buy_quantity_total),
        sell_quantity_pending=int(quote.sell_quantity_total),
        spread=float(quote.spread),
        mid=float(quote.mid),
        persistence_score=float(persistence),
        cancel_rate_per_tick=float(cancel_rate),
        last_trade_size=int(quote.last_quantity),
    )
