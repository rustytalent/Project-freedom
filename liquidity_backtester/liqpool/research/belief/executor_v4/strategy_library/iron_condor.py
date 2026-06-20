"""Iron condor — short ATM±k strangle hedged by further OTM strangle.

The chop / pin / range-bound trade. Capped loss on either side.

When to prefer:
  * scenario_web chop_mass dominant
  * fat-tail in NORMAL band
  * MM mind in pinning or neutral_inventory
  * scenario_web directional consensus near zero

Sprint-4 structure: short OTM2 wings, long OTM4 wings.
Net credit (premium received).
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


class IronCondorStrategy(BaseStrategy):
    name = "iron_condor"
    family = FAMILY_NEUTRAL

    SHORT_WING = 2     # short the OTM2 contracts
    LONG_WING = 4      # long the OTM4 contracts (protective)

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        reasons: List[str] = []
        fitness = 0.30

        # Need all four legs.
        needed = [("CE", self.SHORT_WING), ("CE", self.LONG_WING),
                   ("PE", -self.SHORT_WING), ("PE", -self.LONG_WING)]
        for side, lvl in needed:
            if self._lookup(ctx, side, lvl) is None:
                return StrategyEntryDecision(
                    strategy_name=self.name, family=self.family,
                    can_enter=False,
                    reasons=[f"missing {side} level {lvl} quote"],
                    fitness_score=0.0,
                )

        # Need chop / no strong direction.
        chop_mass = float(getattr(ctx.web_snapshot, "chop_mass", 0.0)
                          if ctx.web_snapshot else 0.0)
        consensus = float(getattr(ctx.web_snapshot, "directional_consensus",
                                    0.0) if ctx.web_snapshot else 0.0)
        if chop_mass >= 0.35:
            fitness += 0.25
            reasons.append(f"web chop_mass {chop_mass:.2f} supports condor")
        if abs(consensus) > 0.30:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"web consensus {consensus:+.2f} too directional"],
                fitness_score=0.0,
            )

        # MM mind
        if ctx.mm_posterior is not None:
            intent = getattr(ctx.mm_posterior, "dominant_intent", "")
            vol_view = getattr(ctx.mm_posterior,
                                 "implied_volatility_view", "")
            if intent in ("pinning_near_expiry", "neutral_inventory"):
                fitness += 0.20
                reasons.append(f"MM intent {intent} supports range-bound")
            if vol_view == "expansion":
                return StrategyEntryDecision(
                    strategy_name=self.name, family=self.family,
                    can_enter=False,
                    reasons=["MM vol view expansion — condor unsafe"],
                    fitness_score=0.0,
                )

        # Fat-tail
        tail_action = (getattr(ctx.fat_tail_score, "recommended_action", "")
                       if ctx.fat_tail_score else "NORMAL")
        if tail_action == "REFUSE":
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=["fat-tail REFUSE — condor short legs unsafe"],
                fitness_score=0.0,
            )
        if tail_action in ("NORMAL", "SCALE_UP"):
            fitness += 0.15

        fitness = max(0.0, min(1.0, fitness))
        can_enter = fitness >= 0.40
        if not can_enter:
            reasons.append(f"condor fitness {fitness:.2f} below 0.40")
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=can_enter, reasons=reasons,
            fitness_score=fitness,
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        ce_short = self._lookup(ctx, "CE", self.SHORT_WING)
        ce_long = self._lookup(ctx, "CE", self.LONG_WING)
        pe_short = self._lookup(ctx, "PE", -self.SHORT_WING)
        pe_long = self._lookup(ctx, "PE", -self.LONG_WING)
        if not all([ce_short, ce_long, pe_short, pe_long]):
            raise ValueError("missing iron condor legs")
        ce_short_strike, ce_short_prem, _, _ = ce_short
        ce_long_strike, ce_long_prem, _, _ = ce_long
        pe_short_strike, pe_short_prem, _, _ = pe_short
        pe_long_strike, pe_long_prem, _, _ = pe_long

        legs = [
            StrategyLeg(contract_side="CE", contract_level=self.SHORT_WING,
                        strike_price=ce_short_strike, direction=-1,
                        lots=lots, entry_premium=ce_short_prem),
            StrategyLeg(contract_side="CE", contract_level=self.LONG_WING,
                        strike_price=ce_long_strike, direction=+1,
                        lots=lots, entry_premium=ce_long_prem),
            StrategyLeg(contract_side="PE", contract_level=-self.SHORT_WING,
                        strike_price=pe_short_strike, direction=-1,
                        lots=lots, entry_premium=pe_short_prem),
            StrategyLeg(contract_side="PE", contract_level=-self.LONG_WING,
                        strike_price=pe_long_strike, direction=+1,
                        lots=lots, entry_premium=pe_long_prem),
        ]
        net_credit = (ce_short_prem + pe_short_prem
                       - ce_long_prem - pe_long_prem) * lots * ctx.lot_size
        wing_width = ce_long_strike - ce_short_strike
        max_loss = wing_width * lots * ctx.lot_size - net_credit
        max_gain = net_credit
        be_high = ce_short_strike + (net_credit / max(1, lots * ctx.lot_size))
        be_low = pe_short_strike - (net_credit / max(1, lots * ctx.lot_size))
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=legs, primary_direction=0,
            max_loss_rupees=max(0.0, max_loss),
            max_gain_rupees=max(0.0, max_gain),
            breakeven_low=be_low, breakeven_high=be_high,
            net_premium_paid=-net_credit,    # credit received
            notes=[
                f"net credit ₹{net_credit:.0f}, max loss ₹{max_loss:.0f}",
                f"wings: CE_OTM{self.SHORT_WING}/{self.LONG_WING}, "
                f"PE_ITM{self.SHORT_WING}/{self.LONG_WING}",
            ],
        )

    def early_warning(self, legs: StrategyLegs,
                       ctx: StrategyContext) -> List[str]:
        return [
            "scenario web tail_mass crosses 0.30 — condor short legs at risk",
            "spot breaks above CE short wing — left short call goes ITM",
            "spot breaks below PE short wing — right short put goes ITM",
            "MM mind flips to faking_direction or hunting_stops",
        ]
