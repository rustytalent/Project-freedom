"""Vertical spreads — bull call / bear put with capped risk and reward.

Bull call: long ATM CE + short OTM CE → bullish, capped both sides.
Bear put:  long ATM PE + short OTM PE → bearish, capped both sides.

When to prefer over single_leg:
  * Crowd mirror says we look like retail
  * Fat-tail in HEDGE band
  * MM mind in stepping_back or hunting_stops (we'd otherwise get
    shaken out of a naked long)
  * We want defined risk for sizing precision
"""
from __future__ import annotations

from typing import List

from .base import (
    BaseStrategy,
    FAMILY_DIRECTIONAL,
    StrategyContext,
    StrategyEntryDecision,
    StrategyLeg,
    StrategyLegs,
)


_WING_LEVEL = 2   # short the OTM2 wing


class BullVerticalStrategy(BaseStrategy):
    name = "bull_vertical"
    family = FAMILY_DIRECTIONAL

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        if ctx.thesis_direction <= 0:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["bull_vertical requires bullish thesis"],
                fitness_score=0.0,
            )
        # Require both legs exist.
        long_info = self._lookup(ctx, "CE", 0)
        short_info = self._lookup(ctx, "CE", _WING_LEVEL)
        if long_info is None or short_info is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["missing leg quotes"],
                fitness_score=0.0,
            )
        reasons: List[str] = []
        fitness = 0.55
        if ctx.crowd_report is not None and getattr(
                ctx.crowd_report, "we_look_like_retail", False):
            fitness += 0.15
            reasons.append("crowd-mirror retail flag — defined-risk preferred")
        tail_action = (getattr(ctx.fat_tail_score, "recommended_action", "")
                       if ctx.fat_tail_score else "NORMAL")
        if tail_action == "HEDGE":
            fitness += 0.15
            reasons.append("fat-tail HEDGE band — defined-risk preferred")
        if ctx.mm_posterior is not None and getattr(
                ctx.mm_posterior, "dominant_intent", "") in {
                "stepping_back", "hunting_stops"}:
            fitness += 0.10
            reasons.append("MM stepping back / hunting — cap downside")
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=True, reasons=reasons,
            fitness_score=max(0.0, min(1.0, fitness)),
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        long_info = self._lookup(ctx, "CE", 0)
        short_info = self._lookup(ctx, "CE", _WING_LEVEL)
        if long_info is None or short_info is None:
            raise ValueError("missing leg quotes for bull vertical")
        long_strike, long_premium, _, _ = long_info
        short_strike, short_premium, _, _ = short_info
        legs = [
            StrategyLeg(contract_side="CE", contract_level=0,
                        strike_price=long_strike, direction=+1,
                        lots=lots, entry_premium=long_premium),
            StrategyLeg(contract_side="CE", contract_level=_WING_LEVEL,
                        strike_price=short_strike, direction=-1,
                        lots=lots, entry_premium=short_premium),
        ]
        net_debit = (long_premium - short_premium) * lots * ctx.lot_size
        max_loss = net_debit
        max_gain = ((short_strike - long_strike) * lots * ctx.lot_size) - net_debit
        be_low = long_strike + (long_premium - short_premium)
        be_high = short_strike
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=legs, primary_direction=+1,
            max_loss_rupees=max(0.0, max_loss),
            max_gain_rupees=max(0.0, max_gain),
            breakeven_low=be_low, breakeven_high=be_high,
            net_premium_paid=net_debit,
            notes=[
                f"long CE_ATM @ ₹{long_premium:.2f}, "
                f"short CE_OTM{_WING_LEVEL} @ ₹{short_premium:.2f}",
                f"net debit ₹{net_debit:.0f}, max gain ₹{max_gain:.0f}",
            ],
        )

    def early_warning(self, legs: StrategyLegs,
                       ctx: StrategyContext) -> List[str]:
        return [
            "spot moves above short wing strike — gain capped early",
            "thesis turns BEAR — both legs lose synchronously",
            "regime stability drops — vertical can stall in chop",
        ]


class BearVerticalStrategy(BaseStrategy):
    name = "bear_vertical"
    family = FAMILY_DIRECTIONAL

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        if ctx.thesis_direction >= 0:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["bear_vertical requires bearish thesis"],
                fitness_score=0.0,
            )
        long_info = self._lookup(ctx, "PE", 0)
        short_info = self._lookup(ctx, "PE", _WING_LEVEL)
        if long_info is None or short_info is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["missing leg quotes"],
                fitness_score=0.0,
            )
        reasons: List[str] = []
        fitness = 0.55
        if ctx.crowd_report is not None and getattr(
                ctx.crowd_report, "we_look_like_retail", False):
            fitness += 0.15
            reasons.append("crowd-mirror retail flag — defined-risk preferred")
        tail_action = (getattr(ctx.fat_tail_score, "recommended_action", "")
                       if ctx.fat_tail_score else "NORMAL")
        if tail_action == "HEDGE":
            fitness += 0.15
            reasons.append("fat-tail HEDGE band — defined-risk preferred")
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=True, reasons=reasons,
            fitness_score=max(0.0, min(1.0, fitness)),
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        long_info = self._lookup(ctx, "PE", 0)
        short_info = self._lookup(ctx, "PE", _WING_LEVEL)
        if long_info is None or short_info is None:
            raise ValueError("missing leg quotes for bear vertical")
        long_strike, long_premium, _, _ = long_info
        short_strike, short_premium, _, _ = short_info
        legs = [
            StrategyLeg(contract_side="PE", contract_level=0,
                        strike_price=long_strike, direction=+1,
                        lots=lots, entry_premium=long_premium),
            StrategyLeg(contract_side="PE", contract_level=_WING_LEVEL,
                        strike_price=short_strike, direction=-1,
                        lots=lots, entry_premium=short_premium),
        ]
        net_debit = (long_premium - short_premium) * lots * ctx.lot_size
        max_loss = net_debit
        max_gain = ((long_strike - short_strike) * lots * ctx.lot_size) - net_debit
        be_high = long_strike - (long_premium - short_premium)
        be_low = short_strike
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=legs, primary_direction=-1,
            max_loss_rupees=max(0.0, max_loss),
            max_gain_rupees=max(0.0, max_gain),
            breakeven_low=be_low, breakeven_high=be_high,
            net_premium_paid=net_debit,
            notes=[
                f"long PE_ATM @ ₹{long_premium:.2f}, "
                f"short PE_OTM{_WING_LEVEL} @ ₹{short_premium:.2f}",
                f"net debit ₹{net_debit:.0f}, max gain ₹{max_gain:.0f}",
            ],
        )

    def early_warning(self, legs: StrategyLegs,
                       ctx: StrategyContext) -> List[str]:
        return [
            "spot drops below short wing strike — gain capped early",
            "thesis turns BULL — both legs lose synchronously",
        ]
