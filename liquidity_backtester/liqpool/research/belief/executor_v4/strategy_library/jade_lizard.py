"""Jade lizard — short PE + short call spread, neutral-to-bullish.

Structure:
  * short PE_OTM2
  * short CE_OTM2
  * long  CE_OTM4
Net effect: collect premium, profit if spot stays above PE strike,
capped upside risk via the call spread.

When preferred:
  * Mild bullish bias + chop dominance
  * MM accumulating
  * Crowd-mirror NOT retail-flagged (jade lizard is institutional-feel)
"""
from __future__ import annotations

from typing import List

from .base import (
    BaseStrategy,
    FAMILY_NEUTRAL,
    StrategyContext,
    StrategyEntryDecision,
    StrategyLeg,
    StrategyLegs,
)


class JadeLizardStrategy(BaseStrategy):
    name = "jade_lizard"
    family = FAMILY_NEUTRAL

    SHORT_PE = -2
    SHORT_CE = 2
    LONG_CE = 4

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        reasons: List[str] = []
        fitness = 0.30
        needed = [("PE", self.SHORT_PE), ("CE", self.SHORT_CE),
                   ("CE", self.LONG_CE)]
        for side, lvl in needed:
            if self._lookup(ctx, side, lvl) is None:
                return StrategyEntryDecision(
                    strategy_name=self.name, family=self.family,
                    can_enter=False,
                    reasons=[f"missing {side} level {lvl}"],
                    fitness_score=0.0,
                )

        consensus = float(getattr(ctx.web_snapshot, "directional_consensus",
                                    0.0)) if ctx.web_snapshot else 0.0
        chop = float(getattr(ctx.web_snapshot, "chop_mass", 0.0)
                      if ctx.web_snapshot else 0.0)
        if consensus < -0.15:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"web consensus {consensus:+.2f} bearish"],
                fitness_score=0.0,
            )
        if consensus > 0.10 and chop > 0.30:
            fitness += 0.20
            reasons.append(f"mild bull + chop {chop:.2f} ideal for jade lizard")

        if ctx.fat_tail_score is not None:
            action = getattr(ctx.fat_tail_score, "recommended_action", "NORMAL")
            if action in ("HEDGE", "REFUSE"):
                return StrategyEntryDecision(
                    strategy_name=self.name, family=self.family,
                    can_enter=False,
                    reasons=[f"fat-tail {action} — short PE unsafe"],
                    fitness_score=0.0,
                )

        if ctx.mm_posterior is not None:
            intent = getattr(ctx.mm_posterior, "dominant_intent", "")
            if intent in ("accumulating", "pinning_near_expiry"):
                fitness += 0.20
                reasons.append(f"MM {intent} supports jade lizard")

        if ctx.crowd_report is not None and getattr(
                ctx.crowd_report, "we_look_like_retail", False):
            fitness += 0.10
            reasons.append("crowd-retail flag — institutional structure preferred")

        can_enter = fitness >= 0.40
        if not can_enter:
            reasons.append(f"jade lizard fitness {fitness:.2f} below 0.40")
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=can_enter, reasons=reasons,
            fitness_score=max(0.0, min(1.0, fitness)),
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        short_pe = self._lookup(ctx, "PE", self.SHORT_PE)
        short_ce = self._lookup(ctx, "CE", self.SHORT_CE)
        long_ce = self._lookup(ctx, "CE", self.LONG_CE)
        if not all([short_pe, short_ce, long_ce]):
            raise ValueError("missing jade lizard legs")
        spe_strike, spe_prem, _, _ = short_pe
        sce_strike, sce_prem, _, _ = short_ce
        lce_strike, lce_prem, _, _ = long_ce

        legs = [
            StrategyLeg(contract_side="PE", contract_level=self.SHORT_PE,
                        strike_price=spe_strike, direction=-1,
                        lots=lots, entry_premium=spe_prem),
            StrategyLeg(contract_side="CE", contract_level=self.SHORT_CE,
                        strike_price=sce_strike, direction=-1,
                        lots=lots, entry_premium=sce_prem),
            StrategyLeg(contract_side="CE", contract_level=self.LONG_CE,
                        strike_price=lce_strike, direction=+1,
                        lots=lots, entry_premium=lce_prem),
        ]
        net_credit = (spe_prem + sce_prem - lce_prem) * lots * ctx.lot_size
        call_spread_width = (lce_strike - sce_strike) * lots * ctx.lot_size
        # Classic jade lizard: net_credit > call_spread_width → no upside risk.
        upside_risk = max(0.0, call_spread_width - net_credit)
        max_gain = net_credit
        # Downside loss bounded only by short PE: significant.
        max_loss = spe_strike * lots * ctx.lot_size - net_credit
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=legs, primary_direction=0,
            max_loss_rupees=max(0.0, max_loss),
            max_gain_rupees=max_gain,
            breakeven_low=spe_strike - (net_credit / max(1, lots * ctx.lot_size)),
            breakeven_high=sce_strike + (net_credit / max(1, lots * ctx.lot_size)),
            net_premium_paid=-net_credit,
            notes=[
                f"jade lizard: -PE_{self.SHORT_PE}, -CE_{self.SHORT_CE}, "
                f"+CE_{self.LONG_CE}",
                f"net credit ₹{net_credit:.0f}, "
                f"upside_risk ₹{upside_risk:.0f}",
            ],
        )

    def early_warning(self, legs, ctx):
        return [
            "spot drops below short PE strike — large unrealized loss",
            "fat-tail crosses HEDGE band — close",
            "MM mind shifts to distributing — close",
        ]
