"""Single-leg long CE or PE — the baseline directional bet.

The retail-default. We use it ONLY when:
  * Strong directional consensus from the scenario web
  * MM mind not in pinning / hunting / faking_direction
  * Fat-tail score below the HEDGE threshold
  * Crowd mirror does NOT flag us as retail-like already (or this is
    the first position in the book)

Max loss: bounded at the premium paid × lots × lot_size.
Max gain: unbounded (long calls / puts).
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


class SingleLegStrategy(BaseStrategy):
    name = "single_leg"
    family = FAMILY_DIRECTIONAL

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        reasons: List[str] = []
        fitness = 0.50

        if ctx.thesis_direction == 0:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=["no directional thesis — single_leg unsuitable"],
                fitness_score=0.0,
            )

        # Require a quote for the proposed contract.
        side = ctx.preferred_side or ("CE" if ctx.thesis_direction > 0 else "PE")
        if self._lookup(ctx, side, ctx.preferred_level) is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"no quote for {side} level {ctx.preferred_level}"],
                fitness_score=0.0,
            )

        consensus = float(getattr(ctx.web_snapshot, "directional_consensus",
                                    0.0)) if ctx.web_snapshot else 0.0
        if ctx.thesis_direction > 0 and consensus < -0.10:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"web consensus {consensus:+.2f} contradicts long"],
                fitness_score=0.0,
            )
        if ctx.thesis_direction < 0 and consensus > 0.10:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"web consensus {consensus:+.2f} contradicts short"],
                fitness_score=0.0,
            )
        # MM intent override
        if ctx.mm_posterior is not None:
            intent = getattr(ctx.mm_posterior, "dominant_intent", "")
            if intent in ("pinning_near_expiry", "faking_direction",
                          "hunting_stops"):
                return StrategyEntryDecision(
                    strategy_name=self.name, family=self.family,
                    can_enter=False,
                    reasons=[f"MM intent {intent} unsafe for single leg"],
                    fitness_score=0.0,
                )

        # Fat-tail
        tail_action = (getattr(ctx.fat_tail_score, "recommended_action", "")
                       if ctx.fat_tail_score else "NORMAL")
        if tail_action == "REFUSE":
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=["fat-tail amp REFUSE"],
                fitness_score=0.0,
            )
        if tail_action == "HEDGE":
            reasons.append("fat-tail recommends a hedged structure instead")
            fitness -= 0.20

        # Crowd mirror
        if ctx.crowd_report is not None and getattr(
                ctx.crowd_report, "we_look_like_retail", False):
            reasons.append("crowd mirror: portfolio already retail-shaped — penalty")
            fitness -= 0.15

        # Fitness boosts when consensus aligns strongly with thesis.
        if ctx.thesis_direction > 0 and consensus > 0.30:
            fitness += 0.20
            reasons.append(f"web consensus {consensus:+.2f} supports long")
        elif ctx.thesis_direction < 0 and consensus < -0.30:
            fitness += 0.20
            reasons.append(f"web consensus {consensus:+.2f} supports short")

        fitness = max(0.0, min(1.0, fitness))
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=True, reasons=reasons, fitness_score=fitness,
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        side = ctx.preferred_side or ("CE" if ctx.thesis_direction > 0 else "PE")
        level = ctx.preferred_level
        info = self._lookup(ctx, side, level)
        if info is None:
            raise ValueError(f"no quote for {side} level {level}")
        strike, premium, _, _ = info
        leg = StrategyLeg(
            contract_side=side, contract_level=level,
            strike_price=strike, direction=+1,
            lots=lots, entry_premium=premium,
        )
        max_loss = premium * lots * ctx.lot_size
        # Unbounded gain — represent as inf, manager caps with R-multiple.
        be_low = strike - premium if side == "PE" else strike + premium
        be_high = strike + premium if side == "CE" else strike - premium
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=[leg],
            primary_direction=ctx.thesis_direction,
            max_loss_rupees=max_loss,
            max_gain_rupees=math.inf,
            breakeven_low=be_low,
            breakeven_high=be_high,
            net_premium_paid=premium * lots * ctx.lot_size,
            notes=[f"single long {side} at level {level}"],
        )

    def early_warning(self, legs: StrategyLegs,
                       ctx: StrategyContext) -> List[str]:
        leg = legs.legs[0]
        return [
            f"{leg.contract_side} rail signed-z reverses by >1σ within 6 bars",
            "engine thesis flips against the position",
            "held leg acceptance turns 'rejected'",
            "fat-tail score climbs above HEDGE threshold mid-position",
        ]
