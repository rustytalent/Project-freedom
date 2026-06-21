"""Ecosystem-specific strategies — built for the Premium Belief Engine.

Standard strategies (single_leg, vertical, straddle, condor, butterfly,
strangle, ratio, jade_lizard) consume signals that exist in any options
context. The strategies in this module ONLY make sense because of the
unique signals OUR engine produces:

  * MM-mind Bayesian posterior over intent
  * Scenario web with explicit probabilities per pathway
  * Manipulation pattern catalogue (stop hunts, traps, accumulation,
    fake breakouts)
  * Crowd-mirror self-awareness ("we look like retail")
  * Epicenter migration tracking
  * Regime stability index from substrate
  * MTF alignment with per-level direction match flags

These are the strategies that take money FROM the retail crowd and the
MMs who are positioned against them — not the textbook plays.

Catalogue (all ship in this module):

  1. **MMIntentMimicryStrategy**       — direct copy of dominant MM intent
                                          when MM-mind confidence is high
  2. **AntiCrowdContrarian**            — when crowd_mirror flags retail-like
                                          portfolio, FADE the retail trade
  3. **EpicenterMigrationStrategy**    — ride MM rebalancing as epicenter
                                          migrates from one strike to another
  4. **StopHuntFadeStrategy**          — long-fade after detected stop hunt
  5. **AccumulationBreakoutStrategy** — long-aggressive when CE acceptance
                                          has been sustained-defended >5 bars
                                          AND spot stayed in tight range
  6. **MTFDivergenceStrategy**         — fade L1 when L15/L60 disagree
  7. **WebDominantPathwayStrategy**    — take a position aligned with the
                                          web's dominant scenario when its
                                          probability > 0.40
  8. **RegimeTransitionStrategy**     — take a directional position at the
                                          edge of a regime stability flip
  9. **CounterfactualInversionStrategy** — when the counterfactual kill
                                              criteria for one position are
                                              firing on the others, FLIP to
                                              the inverse side

These were designed specifically for the founder's "we are the smart
money — take money from the dumb money and the MMs who target the
dumb money" thesis.
"""
from __future__ import annotations

import math
from typing import List

from .base import (
    BaseStrategy,
    FAMILY_DIRECTIONAL,
    FAMILY_NEUTRAL,
    StrategyContext,
    StrategyEntryDecision,
    StrategyLeg,
    StrategyLegs,
)


# ── 1. MM Intent Mimicry ────────────────────────────────────────────


class MMIntentMimicryStrategy(BaseStrategy):
    """Copy the MM. When MM-mind is confidently accumulating →
    long ATM CE. When distributing → long ATM PE.

    Ecosystem signals consumed:
      * ctx.mm_posterior.dominant_intent
      * ctx.mm_posterior.dominant_probability
      * ctx.mm_posterior.confidence
    """
    name = "mm_intent_mimicry"
    family = FAMILY_DIRECTIONAL

    MIN_DOMINANT_PROB = 0.50
    MIN_MM_CONFIDENCE = 0.45

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        if ctx.mm_posterior is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["no MM posterior available"],
                fitness_score=0.0,
            )
        mm = ctx.mm_posterior
        intent = getattr(mm, "dominant_intent", "")
        prob = float(getattr(mm, "dominant_probability", 0.0))
        confidence = float(getattr(mm, "confidence", 0.0))
        if intent not in ("accumulating", "distributing"):
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"MM intent {intent} not actionable for mimicry"],
                fitness_score=0.0,
            )
        if prob < self.MIN_DOMINANT_PROB:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"MM dominant prob {prob:.2f} below floor "
                          f"{self.MIN_DOMINANT_PROB:.2f}"],
                fitness_score=0.0,
            )
        if confidence < self.MIN_MM_CONFIDENCE:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"MM read confidence {confidence:.2f} too low"],
                fitness_score=0.0,
            )
        side = "CE" if intent == "accumulating" else "PE"
        if self._lookup(ctx, side, 0) is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=[f"no ATM {side} quote"],
                fitness_score=0.0,
            )
        fitness = 0.50 + 0.40 * prob + 0.10 * confidence
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=True,
            reasons=[f"MM {intent} @ {prob:.0%} (conf {confidence:.2f}) — "
                      f"mimic with long {side}"],
            fitness_score=min(1.0, fitness),
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        intent = getattr(ctx.mm_posterior, "dominant_intent", "")
        side = "CE" if intent == "accumulating" else "PE"
        info = self._lookup(ctx, side, 0)
        if info is None:
            raise ValueError(f"no ATM {side} quote")
        strike, premium, _, _ = info
        direction = +1
        leg = StrategyLeg(
            contract_side=side, contract_level=0,
            strike_price=strike, direction=direction,
            lots=lots, entry_premium=premium,
        )
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=[leg],
            primary_direction=(+1 if side == "CE" else -1),
            max_loss_rupees=premium * lots * ctx.lot_size,
            max_gain_rupees=math.inf,
            breakeven_low=strike - premium if side == "PE"
                         else strike + premium,
            breakeven_high=strike + premium if side == "CE"
                         else strike - premium,
            net_premium_paid=premium * lots * ctx.lot_size,
            notes=[f"MM-mimicry: long {side}_ATM (intent={intent})"],
        )

    def early_warning(self, legs, ctx):
        return [
            "MM dominant intent shifts AWAY from current side",
            "MM dominant_probability drops below 0.40",
            "Engine thesis flips against the MM-inferred direction",
        ]


# ── 2. Anti-Crowd Contrarian ────────────────────────────────────────


class AntiCrowdContrarianStrategy(BaseStrategy):
    """When the crowd-mirror says we look like retail, take the OPPOSITE
    side of the textbook retail trade.

    Founder's thesis: "we are the smart money, taking money from the
    dumb money." When crowd_mirror flags us as retail-shaped, that
    means our directional thesis is the retail consensus → fade it.

    Ecosystem signals consumed:
      * ctx.crowd_report.we_look_like_retail
      * ctx.crowd_report.crowd_density_long_atm_ce/pe
      * ctx.thesis_direction
    """
    name = "anti_crowd_contrarian"
    family = FAMILY_DIRECTIONAL

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        if ctx.crowd_report is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["no crowd_report available"],
                fitness_score=0.0,
            )
        retail = bool(getattr(ctx.crowd_report, "we_look_like_retail", False))
        score = float(getattr(ctx.crowd_report, "retail_similarity_score", 0.0))
        if not retail or score < 0.65:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"crowd_mirror retail_score {score:.2f} not high enough"],
                fitness_score=0.0,
            )
        # The contrarian trade: fade the dominant retail position.
        long_ce_density = float(getattr(ctx.crowd_report,
                                          "crowd_density_long_atm_ce", 0.0))
        long_pe_density = float(getattr(ctx.crowd_report,
                                          "crowd_density_long_atm_pe", 0.0))
        if long_ce_density > 0.50:
            # Crowd is long ATM CE → contrarian = long PE
            side = "PE"
        elif long_pe_density > 0.50:
            side = "CE"
        else:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=["no clear retail density to fade"],
                fitness_score=0.0,
            )
        # Use OTM2 to keep cost low.
        level = 2 if side == "CE" else -2
        if self._lookup(ctx, side, level) is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=[f"no {side} OTM{abs(level)} quote"],
                fitness_score=0.0,
            )
        fitness = 0.55 + 0.30 * (score - 0.65)
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=True,
            reasons=[f"retail similarity {score:.2f} — fade crowd with long {side}"],
            fitness_score=min(1.0, fitness),
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        long_ce_density = float(getattr(ctx.crowd_report,
                                          "crowd_density_long_atm_ce", 0.0))
        side = "PE" if long_ce_density > 0.50 else "CE"
        level = -2 if side == "PE" else 2
        info = self._lookup(ctx, side, level)
        if info is None:
            raise ValueError(f"no quote for {side} OTM{abs(level)}")
        strike, premium, _, _ = info
        leg = StrategyLeg(
            contract_side=side, contract_level=level,
            strike_price=strike, direction=+1,
            lots=lots, entry_premium=premium,
        )
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=[leg],
            primary_direction=(+1 if side == "CE" else -1),
            max_loss_rupees=premium * lots * ctx.lot_size,
            max_gain_rupees=math.inf,
            breakeven_low=strike - premium,
            breakeven_high=strike + premium,
            net_premium_paid=premium * lots * ctx.lot_size,
            notes=[f"anti-crowd: long {side}_OTM{abs(level)} fading retail crowd"],
        )

    def early_warning(self, legs, ctx):
        return [
            "crowd_density flips — retail rotated",
            "we no longer look like retail (mirror flag clears)",
            "spot accelerates IN the retail direction (we may be early)",
        ]


# ── 3. Epicenter Migration ──────────────────────────────────────────


class EpicenterMigrationStrategy(BaseStrategy):
    """Ride the MM as the epicenter migrates from one strike to another.

    When rich_context.epicenter_migration_distance is >=3 strikes and
    rich_context.epicenter_level has moved positively (toward CE OTM),
    the MM is rebalancing toward bullish; piggyback by going long CE
    at the NEW epicenter level. Conversely for negative migration.

    Ecosystem signals consumed:
      * ctx.rich_context.epicenter_migration_distance
      * ctx.rich_context.epicenter_label / epicenter_level
    """
    name = "epicenter_migration"
    family = FAMILY_DIRECTIONAL

    MIN_MIGRATION_DISTANCE = 3.0

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        if ctx.rich_context is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["no rich_context"],
                fitness_score=0.0,
            )
        mig = float(getattr(ctx.rich_context,
                              "epicenter_migration_distance", 0.0))
        epi_level = int(getattr(ctx.rich_context, "epicenter_level", 0))
        if mig < self.MIN_MIGRATION_DISTANCE:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"epicenter migration {mig:.1f} below "
                          f"{self.MIN_MIGRATION_DISTANCE:.1f}"],
                fitness_score=0.0,
            )
        side = "CE"
        level = max(0, min(3, epi_level))
        if self._lookup(ctx, side, level) is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=[f"no quote for {side}_OTM{level}"],
                fitness_score=0.0,
            )
        # Fat-tail HEDGE → refuse (migration storms are tail-risky).
        if (ctx.fat_tail_score is not None
                and getattr(ctx.fat_tail_score, "recommended_action", "")
                == "REFUSE"):
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["fat-tail REFUSE"],
                fitness_score=0.0,
            )
        fitness = 0.55 + 0.05 * mig
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=True,
            reasons=[f"epicenter migrated {mig:.0f} strikes to level "
                      f"{epi_level} — ride to {side}_OTM{level}"],
            fitness_score=min(1.0, fitness),
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        epi_level = int(getattr(ctx.rich_context, "epicenter_level", 0))
        side = "CE"
        level = max(0, min(3, epi_level))
        info = self._lookup(ctx, side, level)
        if info is None:
            raise ValueError("no leg quote")
        strike, premium, _, _ = info
        leg = StrategyLeg(
            contract_side=side, contract_level=level,
            strike_price=strike, direction=+1,
            lots=lots, entry_premium=premium,
        )
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=[leg], primary_direction=+1,
            max_loss_rupees=premium * lots * ctx.lot_size,
            max_gain_rupees=math.inf,
            breakeven_low=strike + premium,
            breakeven_high=strike + premium,
            net_premium_paid=premium * lots * ctx.lot_size,
            notes=[f"epicenter at level {epi_level}; "
                    f"long {side}_OTM{level} riding migration"],
        )

    def early_warning(self, legs, ctx):
        return [
            "epicenter migration reverses direction",
            "ce_dispersion drops sharply (MM done rebalancing)",
            "regime_stability_index rises >0.70 (move complete)",
        ]


# ── 4. Stop-Hunt Fade ───────────────────────────────────────────────


class StopHuntFadeStrategy(BaseStrategy):
    """When the manipulation board detects a stop hunt, take the OPPOSITE
    side of the hunt — the MM has cleared liquidity and the move usually
    reverses.

    Ecosystem signals consumed:
      * web_snapshot's top scenarios for stop_hunt_up/down_reverse
      * (and the underlying manipulation_patterns board)
    """
    name = "stop_hunt_fade"
    family = FAMILY_DIRECTIONAL

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        if ctx.web_snapshot is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["no web snapshot"],
                fitness_score=0.0,
            )
        # Find stop hunt scenarios in the top 5.
        top = getattr(ctx.web_snapshot, "top_scenarios", None) or []
        for sc in top:
            name = sc.get("name") if isinstance(sc, dict) else getattr(sc, "name", "")
            prob = (sc.get("current_probability") if isinstance(sc, dict)
                     else getattr(sc, "current_probability", 0.0))
            if name in ("stop_hunt_up_reverse", "stop_hunt_down_reverse") \
                    and float(prob) > 0.10:
                # Direction: if "stop_hunt_up" → hunt pushed spot UP, so we fade DOWN.
                if name == "stop_hunt_up_reverse":
                    side = "PE"; level = -1
                else:
                    side = "CE"; level = 1
                if self._lookup(ctx, side, level) is None:
                    continue
                fitness = 0.55 + 1.20 * float(prob)
                return StrategyEntryDecision(
                    strategy_name=self.name, family=self.family,
                    can_enter=True,
                    reasons=[f"stop hunt detected (prob {float(prob):.2f}) "
                              f"— fade with long {side}_{level}"],
                    fitness_score=min(1.0, fitness),
                )
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=False, reasons=["no stop-hunt scenario active"],
            fitness_score=0.0,
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        top = getattr(ctx.web_snapshot, "top_scenarios", None) or []
        target_name = None
        for sc in top:
            name = sc.get("name") if isinstance(sc, dict) else getattr(sc, "name", "")
            if name in ("stop_hunt_up_reverse", "stop_hunt_down_reverse"):
                target_name = name
                break
        if target_name == "stop_hunt_up_reverse":
            side = "PE"; level = -1
        else:
            side = "CE"; level = 1
        info = self._lookup(ctx, side, level)
        if info is None:
            raise ValueError("no leg quote")
        strike, premium, _, _ = info
        leg = StrategyLeg(
            contract_side=side, contract_level=level,
            strike_price=strike, direction=+1,
            lots=lots, entry_premium=premium,
        )
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=[leg],
            primary_direction=(+1 if side == "CE" else -1),
            max_loss_rupees=premium * lots * ctx.lot_size,
            max_gain_rupees=math.inf,
            breakeven_low=strike - premium,
            breakeven_high=strike + premium,
            net_premium_paid=premium * lots * ctx.lot_size,
            notes=[f"stop_hunt_fade: long {side}_{level} after hunt detected"],
        )

    def early_warning(self, legs, ctx):
        return [
            "spot moves further in the hunt direction (we read it wrong)",
            "scenario web's stop-hunt scenario probability decays below 0.05",
        ]


# ── 5. Accumulation Breakout ───────────────────────────────────────


class AccumulationBreakoutStrategy(BaseStrategy):
    """When the manipulation board detects sustained CE accumulation
    (>5 bars of high CE-defended fraction in a tight range), the MM
    has accumulated their inventory. Bias long aggressively because
    the breakout is imminent.

    Ecosystem signals consumed:
      * web's accumulation scenario probability OR
      * recent flow patterns of sustained_ce_defended
    """
    name = "accumulation_breakout"
    family = FAMILY_DIRECTIONAL

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        # Two paths: scenario web OR MM mind.
        accumulating_signal = False
        confidence = 0.0
        if ctx.mm_posterior is not None:
            intent = getattr(ctx.mm_posterior, "dominant_intent", "")
            prob = float(getattr(ctx.mm_posterior, "dominant_probability", 0.0))
            if intent == "accumulating" and prob >= 0.50:
                accumulating_signal = True
                confidence = prob
        if not accumulating_signal and ctx.web_snapshot is not None:
            top = getattr(ctx.web_snapshot, "top_scenarios", None) or []
            for sc in top:
                name = (sc.get("name") if isinstance(sc, dict)
                         else getattr(sc, "name", ""))
                prob = (sc.get("current_probability") if isinstance(sc, dict)
                         else getattr(sc, "current_probability", 0.0))
                if "shakeout_then_bull" in name or "bull_continuation" in name:
                    if float(prob) >= 0.20:
                        accumulating_signal = True
                        confidence = float(prob)
        if not accumulating_signal:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=["no accumulation signal in web or MM-mind"],
                fitness_score=0.0,
            )
        if self._lookup(ctx, "CE", 0) is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["no ATM CE quote"],
                fitness_score=0.0,
            )
        fitness = 0.55 + 0.40 * confidence
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=True,
            reasons=[f"accumulation detected (confidence {confidence:.2f}) "
                      "— breakout long"],
            fitness_score=min(1.0, fitness),
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        info = self._lookup(ctx, "CE", 0)
        if info is None:
            raise ValueError("no ATM CE quote")
        strike, premium, _, _ = info
        leg = StrategyLeg(
            contract_side="CE", contract_level=0,
            strike_price=strike, direction=+1,
            lots=lots, entry_premium=premium,
        )
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=[leg], primary_direction=+1,
            max_loss_rupees=premium * lots * ctx.lot_size,
            max_gain_rupees=math.inf,
            breakeven_low=strike + premium,
            breakeven_high=strike + premium,
            net_premium_paid=premium * lots * ctx.lot_size,
            notes=[f"accumulation_breakout: long CE_ATM at ₹{premium:.2f}"],
        )

    def early_warning(self, legs, ctx):
        return [
            "MM intent flips to distributing (we were wrong about the side)",
            "CE acceptance frame drops below 30%",
            "spot drifts down — breakout never came",
        ]


# ── 6. MTF Divergence ──────────────────────────────────────────────


class MTFDivergenceStrategy(BaseStrategy):
    """When L1 is bullish but L5/L15/L60 disagree, the L1 signal is
    retail noise. Fade L1.

    Ecosystem signals consumed:
      * ctx.web_snapshot or mtf_alignment is exposed via the manager;
        here we re-use thesis_direction and infer divergence from
        the rich_context's velocity sides.
    """
    name = "mtf_divergence"
    family = FAMILY_DIRECTIONAL

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        # The engine's MTF alignment is consulted via the rich_context's
        # 'thesis_velocity_dominant_side' for now (proxy for short-term
        # direction). We compare to the thesis_direction (which is the
        # longer-term composite).
        if ctx.rich_context is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["no rich_context"],
                fitness_score=0.0,
            )
        dom_side = str(getattr(ctx.rich_context,
                                  "thesis_velocity_dominant_side", "flat"))
        if dom_side == "flat":
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=["thesis velocity dominant side is flat"],
                fitness_score=0.0,
            )
        if dom_side == "bull" and ctx.thesis_direction >= 0:
            # No divergence
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["bull velocity agrees with thesis"],
                fitness_score=0.0,
            )
        if dom_side == "bear" and ctx.thesis_direction <= 0:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["bear velocity agrees with thesis"],
                fitness_score=0.0,
            )
        # Divergence: short-term velocity vs longer-term thesis disagree.
        # Take the LONGER-TERM side.
        if ctx.thesis_direction > 0:
            side = "CE"
        else:
            side = "PE"
        if self._lookup(ctx, side, 0) is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=[f"no ATM {side}"],
                fitness_score=0.0,
            )
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=True,
            reasons=[f"MTF divergence: velocity {dom_side} vs thesis "
                      f"{ctx.thesis_direction:+d} — take longer-term side"],
            fitness_score=0.60,
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        side = "CE" if ctx.thesis_direction > 0 else "PE"
        info = self._lookup(ctx, side, 0)
        if info is None:
            raise ValueError("no quote")
        strike, premium, _, _ = info
        leg = StrategyLeg(
            contract_side=side, contract_level=0,
            strike_price=strike, direction=+1,
            lots=lots, entry_premium=premium,
        )
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=[leg],
            primary_direction=(+1 if side == "CE" else -1),
            max_loss_rupees=premium * lots * ctx.lot_size,
            max_gain_rupees=math.inf,
            breakeven_low=strike - premium,
            breakeven_high=strike + premium,
            net_premium_paid=premium * lots * ctx.lot_size,
            notes=[f"mtf_divergence: long {side}_ATM (longer-term side)"],
        )

    def early_warning(self, legs, ctx):
        return [
            "thesis composite flips to agree with short-term velocity",
            "regime stability climbs above 0.70 (divergence resolves)",
        ]


# ── 7. Web Dominant Pathway ─────────────────────────────────────────


class WebDominantPathwayStrategy(BaseStrategy):
    """When the scenario web has a single dominant pathway with prob > 0.40
    and clear implied direction, take that side at the implied strategy
    class's typical strike.

    Ecosystem signal: scenario_web.currently_dominant_pathway with its
    implied_direction and implied_strategy_class hint.
    """
    name = "web_dominant_pathway"
    family = FAMILY_DIRECTIONAL

    MIN_DOMINANT_PROB = 0.40

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        if ctx.web_snapshot is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["no web snapshot"],
                fitness_score=0.0,
            )
        pathway = getattr(ctx.web_snapshot, "currently_dominant_pathway", None)
        if pathway is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["no dominant pathway"],
                fitness_score=0.0,
            )
        prob = (pathway.get("current_probability") if isinstance(pathway, dict)
                 else getattr(pathway, "current_probability", 0.0))
        prob = float(prob)
        direction = (pathway.get("implied_direction")
                       if isinstance(pathway, dict)
                       else getattr(pathway, "implied_direction", 0))
        direction = int(direction)
        if prob < self.MIN_DOMINANT_PROB:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"dominant prob {prob:.2f} below "
                          f"{self.MIN_DOMINANT_PROB:.2f}"],
                fitness_score=0.0,
            )
        if direction == 0:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["dominant direction is neutral"],
                fitness_score=0.0,
            )
        side = "CE" if direction > 0 else "PE"
        if self._lookup(ctx, side, 0) is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=[f"no ATM {side}"],
                fitness_score=0.0,
            )
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=True,
            reasons=[f"dominant pathway prob {prob:.2f} → long {side}"],
            fitness_score=min(1.0, 0.55 + prob * 0.5),
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        pathway = getattr(ctx.web_snapshot, "currently_dominant_pathway", None)
        direction = int((pathway.get("implied_direction")
                          if isinstance(pathway, dict)
                          else getattr(pathway, "implied_direction", 0)))
        side = "CE" if direction > 0 else "PE"
        info = self._lookup(ctx, side, 0)
        if info is None:
            raise ValueError("no quote")
        strike, premium, _, _ = info
        leg = StrategyLeg(
            contract_side=side, contract_level=0,
            strike_price=strike, direction=+1,
            lots=lots, entry_premium=premium,
        )
        pathway_name = (pathway.get("name") if isinstance(pathway, dict)
                          else getattr(pathway, "name", ""))
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=[leg],
            primary_direction=direction,
            max_loss_rupees=premium * lots * ctx.lot_size,
            max_gain_rupees=math.inf,
            breakeven_low=strike - premium,
            breakeven_high=strike + premium,
            net_premium_paid=premium * lots * ctx.lot_size,
            notes=[f"web dominant pathway '{pathway_name}' — long {side}_ATM"],
        )

    def early_warning(self, legs, ctx):
        return [
            "dominant pathway probability decays below 0.20",
            "new pathway with opposite direction takes dominance",
        ]


# ── 8. Regime Transition ─────────────────────────────────────────────


class RegimeTransitionStrategy(BaseStrategy):
    """When regime_stability_index is at a transition (mid-range with
    rising or falling velocity in surrounding signals), the market is
    at a structural change point. Take the longer-term-aligned position.

    Ecosystem signal: rich_context.regime_stability_index + dispersion_velocity
    """
    name = "regime_transition"
    family = FAMILY_DIRECTIONAL

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        if ctx.rich_context is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["no rich_context"],
                fitness_score=0.0,
            )
        stab = float(getattr(ctx.rich_context,
                                "regime_stability_index", 1.0))
        disp_v = float(getattr(ctx.rich_context,
                                  "dispersion_velocity", 0.0))
        # Transition zone: stab in [0.35, 0.60] AND dispersion_velocity > 0.05
        if not (0.35 <= stab <= 0.60 and abs(disp_v) > 0.05):
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"not in transition: stab {stab:.2f}, "
                          f"disp_v {disp_v:+.2f}"],
                fitness_score=0.0,
            )
        if ctx.thesis_direction == 0:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["no directional thesis"],
                fitness_score=0.0,
            )
        side = "CE" if ctx.thesis_direction > 0 else "PE"
        if self._lookup(ctx, side, 0) is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=[f"no ATM {side}"],
                fitness_score=0.0,
            )
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=True,
            reasons=[f"regime transition: stab {stab:.2f}, "
                      f"disp_v {disp_v:+.2f}"],
            fitness_score=0.60,
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        side = "CE" if ctx.thesis_direction > 0 else "PE"
        info = self._lookup(ctx, side, 0)
        if info is None:
            raise ValueError("no quote")
        strike, premium, _, _ = info
        leg = StrategyLeg(
            contract_side=side, contract_level=0,
            strike_price=strike, direction=+1,
            lots=lots, entry_premium=premium,
        )
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=[leg],
            primary_direction=ctx.thesis_direction,
            max_loss_rupees=premium * lots * ctx.lot_size,
            max_gain_rupees=math.inf,
            breakeven_low=strike - premium,
            breakeven_high=strike + premium,
            net_premium_paid=premium * lots * ctx.lot_size,
            notes=[f"regime_transition: long {side}_ATM at structural break"],
        )

    def early_warning(self, legs, ctx):
        return [
            "regime stabilizes back above 0.65 (transition failed)",
            "dispersion velocity reverses sign — was a head-fake",
        ]


# ── 9. Counterfactual Inversion ─────────────────────────────────────


class CounterfactualInversionStrategy(BaseStrategy):
    """When ALL other open positions in the portfolio are seeing their
    counterfactual kill criteria fire, the engine's directional read
    is systematically wrong RIGHT NOW. Flip to the inverse side.

    This is a META-strategy — it consumes the manager's own state.

    Implementation note: the StrategyContext doesn't carry the per-
    position counterfactual results directly, so this strategy uses
    a heuristic: if rich_context's regime is unstable AND thesis
    velocity dominant side conflicts with thesis_direction, that's
    the inversion signal.

    Ecosystem signals consumed:
      * rich_context.regime_stability_index
      * rich_context.thesis_velocity_dominant_side
      * thesis_direction
    """
    name = "counterfactual_inversion"
    family = FAMILY_DIRECTIONAL

    def can_enter(self, ctx: StrategyContext) -> StrategyEntryDecision:
        if ctx.rich_context is None or ctx.thesis_direction == 0:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["missing context"],
                fitness_score=0.0,
            )
        stab = float(getattr(ctx.rich_context,
                                "regime_stability_index", 1.0))
        dom = str(getattr(ctx.rich_context,
                            "thesis_velocity_dominant_side", "flat"))
        # Need unstable regime + conflicting velocity.
        if stab > 0.50:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=[f"regime stable ({stab:.2f}) — no inversion needed"],
                fitness_score=0.0,
            )
        if dom == "flat":
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=["no velocity signal"],
                fitness_score=0.0,
            )
        conflict = ((dom == "bull" and ctx.thesis_direction < 0)
                     or (dom == "bear" and ctx.thesis_direction > 0))
        if not conflict:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False,
                reasons=["no velocity/thesis conflict"],
                fitness_score=0.0,
            )
        # Inversion: take the velocity-aligned side (not the thesis side).
        side = "CE" if dom == "bull" else "PE"
        if self._lookup(ctx, side, 0) is None:
            return StrategyEntryDecision(
                strategy_name=self.name, family=self.family,
                can_enter=False, reasons=[f"no ATM {side}"],
                fitness_score=0.0,
            )
        return StrategyEntryDecision(
            strategy_name=self.name, family=self.family,
            can_enter=True,
            reasons=[f"counterfactual inversion: velocity={dom}, "
                      f"thesis direction={ctx.thesis_direction:+d}, "
                      f"stab={stab:.2f}"],
            fitness_score=0.55,
        )

    def build(self, ctx: StrategyContext, lots: int) -> StrategyLegs:
        dom = str(getattr(ctx.rich_context,
                            "thesis_velocity_dominant_side", "flat"))
        side = "CE" if dom == "bull" else "PE"
        info = self._lookup(ctx, side, 0)
        if info is None:
            raise ValueError("no quote")
        strike, premium, _, _ = info
        leg = StrategyLeg(
            contract_side=side, contract_level=0,
            strike_price=strike, direction=+1,
            lots=lots, entry_premium=premium,
        )
        return StrategyLegs(
            strategy_name=self.name, family=self.family,
            legs=[leg],
            primary_direction=(+1 if side == "CE" else -1),
            max_loss_rupees=premium * lots * ctx.lot_size,
            max_gain_rupees=math.inf,
            breakeven_low=strike - premium,
            breakeven_high=strike + premium,
            net_premium_paid=premium * lots * ctx.lot_size,
            notes=[f"counterfactual_inversion: long {side}_ATM "
                    f"(velocity-aligned, NOT thesis-aligned)"],
        )

    def early_warning(self, legs, ctx):
        return [
            "regime stability climbs above 0.55 — inversion read invalidated",
            "thesis composite resyncs with velocity",
        ]
