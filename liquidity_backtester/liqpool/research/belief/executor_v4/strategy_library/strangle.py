"""Strangle — cheaper than straddle, wider breakeven, fatter tails.

Long strangle: long OTM CE + long OTM PE (default OTM2 on each side).
Used when vol expansion is expected but the founder doesn't want to
pay ATM premium. Risk: theta bleed if range stays narrow.

When preferred over straddle:
  * Crowd-mirror says we look retail (cheaper = less obvious)
  * Tail-mass is moderate but not extreme
  * MM mind is volatile-but-undirectional
"""
from __future__ import annotations

import math
from typing import List

from .base import (
    BaseStrategy,
    FAMILY_VOL_LONG,
    StrategyContext,
    StrategyEntryDecision,
    StrategyLeg,
    StrategyLegs,
)


_WING_LEVEL = 2


class LongStrangleStrategy(BaseStrategy):
    name = "long_strangle"
    family = FAMILY_VOL_LONG

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        ce_info = self._lookup(ctx, "CE", _WING_LEVEL)
        pe_info = self._lookup(ctx, "PE", -_WING_LEVEL)
        if ce_info is None or pe_info is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["missing OTM wings"],
                fitness_score=0.0,
            )
        reasons: List[str] = []
        fitness = 0.35

        consensus = float(getattr(ctx.web_snapshot, "directional_consensus",
                                    0.0)) if ctx.web_snapshot else 0.0
        if abs(consensus) > 0.40:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"web consensus {consensus:+.2f} too directional"],
                fitness_score=0.0,
            )
        regime_stab = float(getattr(ctx.rich_context,
                                       "regime_stability_index", 1.0)
                            if ctx.rich_context else 1.0)
        if regime_stab < 0.55:
            fitness += 0.20
            reasons.append(f"regime stability {regime_stab:.2f} — vol expansion")

        if ctx.mm_posterior is not None:
            vol_view = getattr(ctx.mm_posterior,
                                 "implied_volatility_view", "")
            if vol_view == "expansion":
                fitness += 0.20
                reasons.append("MM-mind vol expansion view")

        # Crowd-mirror prefers strangle over straddle (cheaper, less obvious).
        if ctx.crowd_report is not None and getattr(
                ctx.crowd_report, "we_look_like_retail", False):
            fitness += 0.10
            reasons.append("crowd-mirror retail flag — strangle is less retail")

        fitness = max(0.0, min(1.0, fitness))
        can_enter = fitness >= 0.40
        if not can_enter:
            reasons.append(f"strangle fitness {fitness:.2f} below 0.40")
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=can_enter, reasons=reasons,
            fitness_score=fitness,
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        ce_info = self._lookup(ctx, "CE", _WING_LEVEL)
        pe_info = self._lookup(ctx, "PE", -_WING_LEVEL)
        if ce_info is None or pe_info is None:
            raise ValueError("missing strangle leg quotes")
        ce_strike, ce_prem, _, _ = ce_info
        pe_strike, pe_prem, _, _ = pe_info
        legs = [
            StrategyLeg(contract_side="CE", contract_level=_WING_LEVEL,
                        strike_price=ce_strike, direction=+1,
                        lots=lots, entry_premium=ce_prem),
            StrategyLeg(contract_side="PE", contract_level=-_WING_LEVEL,
                        strike_price=pe_strike, direction=+1,
                        lots=lots, entry_premium=pe_prem),
        ]
        debit = (ce_prem + pe_prem) * lots * ctx.lot_size
        be_high = ce_strike + (ce_prem + pe_prem)
        be_low = pe_strike - (ce_prem + pe_prem)
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=legs, primary_direction=0,
            max_loss_rupees=debit,
            max_gain_rupees=math.inf,
            breakeven_low=be_low, breakeven_high=be_high,
            net_premium_paid=debit,
            notes=[
                f"long CE_OTM{_WING_LEVEL} ₹{ce_prem:.2f} + "
                f"long PE_OTM{_WING_LEVEL} ₹{pe_prem:.2f}",
                f"breakeven ₹{be_low:.0f} / ₹{be_high:.0f}",
            ],
        )

    def early_warning(self, legs, ctx):
        return [
            "spot stays inside breakeven band — theta bleeds both legs",
            "MM mind flips to crush",
            "regime stabilizes — strangle loses optionality value",
        ]
