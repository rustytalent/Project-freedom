"""Butterfly — the max-pin trade near expiry.

Structure: long 1 ATM CE + short 2 OTM1 CE + long 1 OTM2 CE
(symmetric body around ATM). Cheap, capped both ways.

When to prefer:
  * MM mind dominant intent = pinning_near_expiry
  * scenario_web's dominant strategy class is butterfly or iron_condor
  * fat-tail in NORMAL band
"""
from __future__ import annotations

from typing import List

from .base import (
    BaseStrategy,
    FAMILY_PIN,
    StrategyContext,
    StrategyEntryDecision,
    StrategyLeg,
    StrategyLegs,
)


class ButterflyStrategy(BaseStrategy):
    name = "butterfly"
    family = FAMILY_PIN

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        reasons: List[str] = []
        fitness = 0.30

        atm_info = self._lookup(ctx, "CE", 0)
        wing1_info = self._lookup(ctx, "CE", 1)
        wing2_info = self._lookup(ctx, "CE", 2)
        if not (atm_info and wing1_info and wing2_info):
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["missing CE leg quotes"],
                fitness_score=0.0,
            )

        # Strong indicator: MM in pinning mode.
        if ctx.mm_posterior is not None:
            intent = getattr(ctx.mm_posterior, "dominant_intent", "")
            if intent == "pinning_near_expiry":
                fitness += 0.30
                reasons.append("MM-mind: pinning_near_expiry — butterfly ideal")

        # Web dominant strategy
        web_strat = (getattr(ctx.web_snapshot, "dominant_strategy_class", "")
                      if ctx.web_snapshot else "")
        if web_strat in ("butterfly", "iron_condor"):
            fitness += 0.20
            reasons.append(f"web dominant_strategy={web_strat}")

        # Tail risk
        tail_action = (getattr(ctx.fat_tail_score, "recommended_action", "")
                       if ctx.fat_tail_score else "NORMAL")
        if tail_action == "REFUSE":
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["fat-tail REFUSE"],
                fitness_score=0.0,
            )

        fitness = max(0.0, min(1.0, fitness))
        can_enter = fitness >= 0.40
        if not can_enter:
            reasons.append(f"butterfly fitness {fitness:.2f} below 0.40")
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=can_enter, reasons=reasons,
            fitness_score=fitness,
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        atm_info = self._lookup(ctx, "CE", 0)
        wing1_info = self._lookup(ctx, "CE", 1)
        wing2_info = self._lookup(ctx, "CE", 2)
        if not (atm_info and wing1_info and wing2_info):
            raise ValueError("missing butterfly legs")
        atm_strike, atm_prem, _, _ = atm_info
        w1_strike, w1_prem, _, _ = wing1_info
        w2_strike, w2_prem, _, _ = wing2_info

        legs = [
            StrategyLeg(contract_side="CE", contract_level=0,
                        strike_price=atm_strike, direction=+1,
                        lots=lots, entry_premium=atm_prem),
            StrategyLeg(contract_side="CE", contract_level=1,
                        strike_price=w1_strike, direction=-1,
                        lots=lots * 2, entry_premium=w1_prem),
            StrategyLeg(contract_side="CE", contract_level=2,
                        strike_price=w2_strike, direction=+1,
                        lots=lots, entry_premium=w2_prem),
        ]
        net_debit = (atm_prem - 2 * w1_prem + w2_prem) * lots * ctx.lot_size
        max_loss = max(0.0, net_debit)
        # Max gain at the middle (W1) strike at expiry:
        body_width = w1_strike - atm_strike
        max_gain = body_width * lots * ctx.lot_size - net_debit
        be_low = atm_strike + (net_debit / max(1, lots * ctx.lot_size))
        be_high = w2_strike - (net_debit / max(1, lots * ctx.lot_size))
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=legs, primary_direction=0,
            max_loss_rupees=max_loss,
            max_gain_rupees=max(0.0, max_gain),
            breakeven_low=be_low, breakeven_high=be_high,
            net_premium_paid=net_debit,
            notes=[
                f"butterfly @ ₹{w1_strike:.0f}, net debit ₹{net_debit:.0f}",
                f"max gain ₹{max_gain:.0f} at the middle strike",
            ],
        )

    def early_warning(self, legs: StrategyLegs,
                       ctx: StrategyContext) -> List[str]:
        return [
            "spot drifts away from the body strike — gain decays",
            "MM intent flips away from pinning",
            "fat-tail rises — short wing exposure",
        ]
