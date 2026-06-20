"""1x2 ratio spread — directional with built-in hedge.

Bull ratio: long 1 ATM CE + short 2 OTM CE. Pays a small net credit;
profits if spot drifts up to the short strike. Loses if spot rips
through the short strike.

When preferred:
  * Mild bullish thesis but high tail-mass for explosive move
  * MM mind hints at pinning OR moderate accumulation
  * Want premium received, not paid
"""
from __future__ import annotations

import math
from typing import List

from .base import (
    BaseStrategy,
    FAMILY_DIRECTIONAL,
    StrategyContext,
    StrategyEntryDecision,
    StrategyLeg,
    StrategyLegs,
)


_LONG_LEVEL = 0
_SHORT_LEVEL = 2


class BullRatioSpreadStrategy(BaseStrategy):
    name = "bull_ratio_spread"
    family = FAMILY_DIRECTIONAL

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        if ctx.thesis_direction <= 0:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["needs bullish bias"],
                fitness_score=0.0,
            )
        long_info = self._lookup(ctx, "CE", _LONG_LEVEL)
        short_info = self._lookup(ctx, "CE", _SHORT_LEVEL)
        if long_info is None or short_info is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["missing CE legs"],
                fitness_score=0.0,
            )
        reasons: List[str] = []
        fitness = 0.35

        consensus = float(getattr(ctx.web_snapshot, "directional_consensus",
                                    0.0)) if ctx.web_snapshot else 0.0
        # Strong consensus → prefer single leg; mild consensus is ideal for ratio.
        if 0.10 < consensus < 0.40:
            fitness += 0.15
            reasons.append(f"mild bullish consensus {consensus:+.2f} fits ratio")
        elif consensus > 0.50:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"strong consensus {consensus:+.2f} — use single leg"],
                fitness_score=0.0,
            )

        if ctx.mm_posterior is not None:
            intent = getattr(ctx.mm_posterior, "dominant_intent", "")
            if intent in ("pinning_near_expiry", "accumulating"):
                fitness += 0.15
                reasons.append(f"MM intent {intent} supports ratio bull")

        # Tail risk too high — ratio is unbounded on the short side.
        if ctx.fat_tail_score is not None:
            action = getattr(ctx.fat_tail_score, "recommended_action", "NORMAL")
            if action in ("HEDGE", "REFUSE"):
                return StrategyEntryDecision(
                    strategy_name=self.name, family=self.family,
                    can_enter=False,
                    reasons=[f"fat-tail {action} — naked short leg unsafe"],
                    fitness_score=0.0,
                )

        fitness = max(0.0, min(1.0, fitness))
        can_enter = fitness >= 0.40
        if not can_enter:
            reasons.append(f"ratio fitness {fitness:.2f} below 0.40")
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=can_enter, reasons=reasons,
            fitness_score=fitness,
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        long_info = self._lookup(ctx, "CE", _LONG_LEVEL)
        short_info = self._lookup(ctx, "CE", _SHORT_LEVEL)
        if long_info is None or short_info is None:
            raise ValueError("missing ratio legs")
        long_strike, long_prem, _, _ = long_info
        short_strike, short_prem, _, _ = short_info
        legs = [
            StrategyLeg(contract_side="CE", contract_level=_LONG_LEVEL,
                        strike_price=long_strike, direction=+1,
                        lots=lots, entry_premium=long_prem),
            StrategyLeg(contract_side="CE", contract_level=_SHORT_LEVEL,
                        strike_price=short_strike, direction=-1,
                        lots=lots * 2, entry_premium=short_prem),
        ]
        net = (long_prem - 2 * short_prem) * lots * ctx.lot_size
        # Net negative = credit.
        max_gain = ((short_strike - long_strike) * lots * ctx.lot_size) - net
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=legs, primary_direction=+1,
            max_loss_rupees=math.inf,    # naked short = unbounded
            max_gain_rupees=max(0.0, max_gain),
            breakeven_low=long_strike + max(0, net / max(1, lots * ctx.lot_size)),
            breakeven_high=short_strike + (short_strike - long_strike),
            net_premium_paid=net,
            notes=[
                f"1× long CE_ATM ₹{long_prem:.2f}, 2× short CE_OTM{_SHORT_LEVEL} "
                f"₹{short_prem:.2f}",
                "WARNING: naked short upper leg — needs strict spot monitoring",
            ],
        )

    def early_warning(self, legs, ctx):
        return [
            f"spot rips above short strike — naked leg uncapped loss",
            "fat-tail score crosses HEDGE band → close immediately",
            "thesis flips bearish → close before gap-down",
        ]
