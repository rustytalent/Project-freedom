"""Long straddle — long ATM CE + long ATM PE.

Plays vol expansion. The trade is right when:
  * fat-tail score is moderate-to-high (HEDGE band)
  * MM mind suggests stepping back / vol expansion
  * regime stability is low
  * scenario web's directional consensus is near zero (no committed
    direction)
  * IV state hasn't pre-expanded already (or we'd be paying the high)

Max loss: total premium paid (both legs).
Max gain: theoretically unbounded if spot moves far enough either way.
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


class LongStraddleStrategy(BaseStrategy):
    name = "long_straddle"
    family = FAMILY_VOL_LONG

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        reasons: List[str] = []
        fitness = 0.40

        ce_info = self._lookup(ctx, "CE", 0)
        pe_info = self._lookup(ctx, "PE", 0)
        if ce_info is None or pe_info is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["missing ATM leg quotes"],
                fitness_score=0.0,
            )

        consensus = float(getattr(ctx.web_snapshot, "directional_consensus",
                                    0.0)) if ctx.web_snapshot else 0.0
        if abs(consensus) > 0.40:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"web consensus {consensus:+.2f} too directional "
                          "for straddle"],
                fitness_score=0.0,
            )
        if abs(consensus) < 0.15:
            fitness += 0.15
            reasons.append(
                f"web consensus {consensus:+.2f} near zero — non-directional"
            )

        regime_stab = float(getattr(ctx.rich_context,
                                     "regime_stability_index", 1.0)
                            if ctx.rich_context else 1.0)
        if regime_stab < 0.50:
            fitness += 0.15
            reasons.append(
                f"regime stability {regime_stab:.2f} low — vol expansion likely"
            )

        # IV state — avoid straddle if vol already crushed (premium high).
        if ctx.iv_state in ("dirty_data", "common_shock", "vol_expansion"):
            fitness -= 0.10
            reasons.append(
                f"iv_state {ctx.iv_state} — vol may already be expanded"
            )

        if ctx.mm_posterior is not None:
            vol_view = getattr(ctx.mm_posterior,
                                 "implied_volatility_view", "")
            if vol_view == "expansion":
                fitness += 0.20
                reasons.append("MM-mind: expansion view supports straddle")
            elif vol_view == "crush":
                fitness -= 0.30
                reasons.append(
                    "MM-mind: crush view — straddle bleeds via theta"
                )

        # Fat-tail
        tail_action = (getattr(ctx.fat_tail_score, "recommended_action", "")
                       if ctx.fat_tail_score else "NORMAL")
        if tail_action == "HEDGE":
            fitness += 0.15
            reasons.append("fat-tail HEDGE band — straddle defends")

        fitness = max(0.0, min(1.0, fitness))
        can_enter = fitness >= 0.45
        if not can_enter:
            reasons.append(f"straddle fitness {fitness:.2f} below 0.45")
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=can_enter, reasons=reasons,
            fitness_score=fitness,
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        ce_info = self._lookup(ctx, "CE", 0)
        pe_info = self._lookup(ctx, "PE", 0)
        if ce_info is None or pe_info is None:
            raise ValueError("missing ATM leg quotes")
        ce_strike, ce_premium, _, _ = ce_info
        pe_strike, pe_premium, _, _ = pe_info
        legs = [
            StrategyLeg(contract_side="CE", contract_level=0,
                        strike_price=ce_strike, direction=+1,
                        lots=lots, entry_premium=ce_premium),
            StrategyLeg(contract_side="PE", contract_level=0,
                        strike_price=pe_strike, direction=+1,
                        lots=lots, entry_premium=pe_premium),
        ]
        debit = (ce_premium + pe_premium) * lots * ctx.lot_size
        be_high = ce_strike + (ce_premium + pe_premium)
        be_low = pe_strike - (ce_premium + pe_premium)
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=legs, primary_direction=0,
            max_loss_rupees=debit,
            max_gain_rupees=math.inf,
            breakeven_low=be_low, breakeven_high=be_high,
            net_premium_paid=debit,
            notes=[
                f"long CE_ATM @ ₹{ce_premium:.2f} + long PE_ATM @ ₹{pe_premium:.2f}",
                f"breakeven ₹{be_low:.0f} / ₹{be_high:.0f}",
            ],
        )

    def early_warning(self, legs: StrategyLegs,
                       ctx: StrategyContext) -> List[str]:
        return [
            "spot remains in a tight band — theta bleeds both legs",
            "MM mind flips to crush — IV bleed accelerates",
            "regime stabilizes — straddle loses optionality value",
        ]
