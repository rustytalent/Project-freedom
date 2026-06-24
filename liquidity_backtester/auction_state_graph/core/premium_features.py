"""Optional CE/PE premium response features."""

from __future__ import annotations

from .models import Tick


def premium_response(ticks: list[Tick]) -> tuple[float | None, float | None, float | None, float | None]:
    """Return CE efficiency, PE efficiency, premium bias, compression.

    Efficiency is a normalized premium response to spot movement. If replay data
    has no premium columns, all outputs are None so the layer remains usable.
    """
    ce_ticks = [t for t in ticks if t.ce_price is not None]
    pe_ticks = [t for t in ticks if t.pe_price is not None]
    if len(ce_ticks) < 2 or len(pe_ticks) < 2:
        return None, None, None, None

    spot_move = ticks[-1].price - ticks[0].price
    ce_move = float(ce_ticks[-1].ce_price or 0.0) - float(ce_ticks[0].ce_price or 0.0)
    pe_move = float(pe_ticks[-1].pe_price or 0.0) - float(pe_ticks[0].pe_price or 0.0)
    denom = max(abs(spot_move), 1e-6)

    ce_eff = ce_move / denom
    pe_eff = -pe_move / denom
    premium_bias = ce_eff - pe_eff

    total_premium_0 = float(ce_ticks[0].ce_price or 0.0) + float(pe_ticks[0].pe_price or 0.0)
    total_premium_1 = float(ce_ticks[-1].ce_price or 0.0) + float(pe_ticks[-1].pe_price or 0.0)
    premium_compression = (total_premium_1 - total_premium_0) / max(abs(total_premium_0), 1e-6)
    return ce_eff, pe_eff, premium_bias, premium_compression
