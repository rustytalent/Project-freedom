"""AdaptiveExitEngine — per-position adaptive exit decision loop.

For each open position, this engine:

  1. Observes the latest premium tick via ``observe(market_state)``
  2. Updates the LocalHighTracker
  3. Forecasts revisit probability for the current ghost target
  4. Decides the exit mode (HARVEST / SHADE / CHASE_FILL / DE_RISK / KILL)
  5. Re-costs the ghost exit price for that mode
  6. Asks the ModificationGate: should we modify the broker order?
  7. If yes, charges the ModificationBudget and emits an ExitDecision
     with the new broker_price

The engine is STATEFUL per position. The PortfolioExitManager owns one
engine per open position.

Founder's golden rule:
  "The model may think every tick, but the broker order should move
   only when the expected value justifies spending one of the 25
   modifications."
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .budget import ModificationBudget, ModificationBudgetConfig
from .gate import ModificationGate, ModificationGateConfig
from .local_highs import LocalHighTracker, LocalHighTrackerConfig, RevisitForecast
from .modes import (
    EXIT_MODE_CHASE_FILL,
    EXIT_MODE_DE_RISK,
    EXIT_MODE_HARVEST,
    EXIT_MODE_KILL,
    EXIT_MODE_SHADE,
    ExitMode,
    ExitModeDecision,
    _ModeHysteresis,
    decide_exit_mode,
)
from .thesis import ExitThesis, ExitThesisConfig


# ── Configuration ─────────────────────────────────────────────────


@dataclass
class AdaptiveExitEngineConfig:
    """Top-level knobs."""
    thesis: ExitThesisConfig = field(default_factory=ExitThesisConfig)
    local_highs: LocalHighTrackerConfig = field(
        default_factory=LocalHighTrackerConfig)
    budget: ModificationBudgetConfig = field(
        default_factory=ModificationBudgetConfig)
    gate: ModificationGateConfig = field(default_factory=ModificationGateConfig)
    # Engine internals
    max_hold_bars_default: int = 90
    near_exit_band_pct: float = 0.005      # within 0.5% of ghost = "near exit"


# ── Market state passed in by the manager ────────────────────────


@dataclass
class MarketState:
    """Per-tick market state for one position."""
    bar_index: int
    ts: Any
    held_premium: float                # current mid / mark
    held_bid: float = 0.0
    held_ask: float = 0.0
    spread: float = 0.0
    friendliness: float = 1.0
    thesis_intact: bool = True
    thesis_confidence_decay: float = 0.0
    current_r: float = 0.0
    bars_held: int = 0
    fat_tail_action: str = "NORMAL"
    portfolio_under_pressure: bool = False
    micro_volatility: float = 0.0     # rolling stdev of recent premium

    def derived_spread(self) -> float:
        """If spread not provided, derive from bid/ask."""
        if self.spread > 0:
            return self.spread
        if self.held_bid > 0 and self.held_ask > 0:
            return max(0.0, self.held_ask - self.held_bid)
        # Fallback: use held_premium * tiny pct
        return max(0.05, self.held_premium * 0.002)


# ── Decision output ───────────────────────────────────────────────


@dataclass
class ExitDecision:
    """Per-tick output from the engine for one position."""
    position_id: str
    exit_mode: ExitMode
    mode_transitioned: bool
    ghost_exit_price: float
    broker_exit_price: float
    broker_modification_proposed: bool
    revisit_probability: float
    high_trajectory: str
    bars_held: int
    # Diagnostics
    mode_decision: Dict[str, Any]
    gate_decision: Optional[Dict[str, Any]]
    budget_summary: Dict[str, Any]
    thesis_snapshot: Dict[str, Any]
    revisit_forecast: Dict[str, Any]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "exit_mode": self.exit_mode,
            "mode_transitioned": self.mode_transitioned,
            "ghost_exit_price": round(self.ghost_exit_price, 4),
            "broker_exit_price": round(self.broker_exit_price, 4),
            "broker_modification_proposed": self.broker_modification_proposed,
            "revisit_probability": round(self.revisit_probability, 3),
            "high_trajectory": self.high_trajectory,
            "bars_held": self.bars_held,
            "mode_decision": dict(self.mode_decision),
            "gate_decision": (dict(self.gate_decision)
                                if self.gate_decision else None),
            "budget_summary": dict(self.budget_summary),
            "thesis_snapshot": dict(self.thesis_snapshot),
            "revisit_forecast": dict(self.revisit_forecast),
            "notes": list(self.notes),
        }


# ── The engine ────────────────────────────────────────────────────


class AdaptiveExitEngine:
    """Per-position adaptive exit decision loop."""

    def __init__(self, *,
                  exit_thesis: ExitThesis,
                  cfg: Optional[AdaptiveExitEngineConfig] = None) -> None:
        self.cfg = cfg or AdaptiveExitEngineConfig()
        self.thesis = exit_thesis
        self.local_highs = LocalHighTracker(self.cfg.local_highs)
        self.budget = ModificationBudget(cfg=self.cfg.budget)
        self.gate = ModificationGate(self.cfg.gate)
        self._hysteresis = _ModeHysteresis()
        self._last_decision: Optional[ExitDecision] = None
        # Track the previous broker price after the engine confirms a
        # modification has been sent (so the gate compares against the
        # CURRENT live broker price, not the ghost).
        self._broker_price_after_last_mod = exit_thesis.current_broker_exit_price

    # ── Per-tick driver ────────────────────────────────────────────

    def observe(self, market: MarketState) -> ExitDecision:
        """Update internal state for one bar and decide what (if anything)
        to do with the broker order."""
        cfg = self.cfg
        thesis = self.thesis

        # 1. Update local highs.
        self.local_highs.observe(market.bar_index, market.held_premium)

        # 2. Revisit forecast against current ghost.
        forecast = self.local_highs.forecast_revisit(
            thesis.current_ghost_exit_price)

        # 3. Decide exit mode.
        near_exit = self._is_near_exit(market.held_premium,
                                          thesis.current_ghost_exit_price)
        mode_decision = decide_exit_mode(
            hysteresis=self._hysteresis,
            thesis_intact=market.thesis_intact,
            thesis_confidence_decay=market.thesis_confidence_decay,
            current_r=market.current_r,
            revisit_probability=forecast.revisit_probability,
            high_trajectory=forecast.high_trajectory,
            bars_held=market.bars_held,
            max_hold_bars=cfg.max_hold_bars_default,
            fat_tail_action=market.fat_tail_action,
            portfolio_under_pressure=market.portfolio_under_pressure,
            price_near_exit=near_exit,
        )

        # 4. Re-cost ghost price based on mode.
        new_ghost = self._compute_ghost_price(
            market=market, mode=mode_decision.mode,
            forecast=forecast,
        )
        if thesis.direction > 0:
            if new_ghost < thesis.current_ghost_exit_price:
                thesis.reduce_ghost_toward(new_ghost, cfg=cfg.thesis)
            elif new_ghost > thesis.current_ghost_exit_price:
                thesis.raise_ghost_toward(new_ghost, cfg=cfg.thesis)
        else:
            if new_ghost > thesis.current_ghost_exit_price:
                thesis.reduce_ghost_toward(new_ghost, cfg=cfg.thesis)
            elif new_ghost < thesis.current_ghost_exit_price:
                thesis.raise_ghost_toward(new_ghost, cfg=cfg.thesis)

        # 5. Modification gate.
        gate_decision = None
        modification_proposed = False
        if mode_decision.mode == EXIT_MODE_KILL:
            # KILL: always propose a modification (will be charged to
            # emergency reserve via the gate's bypass).
            gate_decision = self.gate.evaluate(
                old_broker_price=self._broker_price_after_last_mod,
                new_ghost_price=thesis.current_ghost_exit_price,
                bars_held=market.bars_held,
                total_mods_used=self.budget.total_used,
                budget_pressure=self.budget.budget_pressure(),
                exit_mode=mode_decision.mode,
                spread=market.derived_spread(),
                micro_volatility=market.micro_volatility,
                fill_probability_gain=1.0,    # always max for KILL
                thesis_confidence_decay=market.thesis_confidence_decay,
                portfolio_risk_relief=1.0,
            )
        else:
            # Only call the gate when ghost has moved enough from broker.
            ghost_vs_broker_change = abs(
                thesis.current_ghost_exit_price
                - self._broker_price_after_last_mod)
            if ghost_vs_broker_change > 0:
                fill_gain = self._estimate_fill_probability_gain(
                    market=market, forecast=forecast,
                    new_price=thesis.current_ghost_exit_price)
                gate_decision = self.gate.evaluate(
                    old_broker_price=self._broker_price_after_last_mod,
                    new_ghost_price=thesis.current_ghost_exit_price,
                    bars_held=market.bars_held,
                    total_mods_used=self.budget.total_used,
                    budget_pressure=self.budget.budget_pressure(),
                    exit_mode=mode_decision.mode,
                    spread=market.derived_spread(),
                    micro_volatility=market.micro_volatility,
                    fill_probability_gain=fill_gain,
                    thesis_confidence_decay=market.thesis_confidence_decay,
                )

        # 6. Spend budget if approved.
        if gate_decision is not None and gate_decision.approve:
            granted = self.budget.try_spend(
                gate_decision.suggested_bucket,
                reason=f"{mode_decision.mode}: "
                        f"ghost {self._broker_price_after_last_mod:.2f}"
                        f"→{thesis.current_ghost_exit_price:.2f}",
                fallback_to_emergency_on_kill=(
                    mode_decision.mode == EXIT_MODE_KILL),
            )
            if granted is not None:
                modification_proposed = True
                # Update broker price tracker.
                thesis.current_broker_exit_price = thesis.current_ghost_exit_price
                self._broker_price_after_last_mod = thesis.current_ghost_exit_price

        decision = ExitDecision(
            position_id=thesis.position_id,
            exit_mode=mode_decision.mode,
            mode_transitioned=mode_decision.transitioned,
            ghost_exit_price=thesis.current_ghost_exit_price,
            broker_exit_price=thesis.current_broker_exit_price,
            broker_modification_proposed=modification_proposed,
            revisit_probability=forecast.revisit_probability,
            high_trajectory=forecast.high_trajectory,
            bars_held=market.bars_held,
            mode_decision=mode_decision.to_dict(),
            gate_decision=(gate_decision.to_dict()
                             if gate_decision is not None else None),
            budget_summary=self.budget.to_dict(),
            thesis_snapshot=thesis.to_dict(),
            revisit_forecast=forecast.to_dict(),
        )
        self._last_decision = decision
        return decision

    # ── Pure helpers ───────────────────────────────────────────────

    def _is_near_exit(self, held_premium: float,
                        ghost_price: float) -> bool:
        if ghost_price <= 0:
            return False
        return (abs(held_premium - ghost_price) / ghost_price
                 <= self.cfg.near_exit_band_pct)

    def _estimate_fill_probability_gain(self, *, market: MarketState,
                                            forecast: RevisitForecast,
                                            new_price: float) -> float:
        """How much does moving the broker order to ``new_price`` improve
        fill probability vs the old price?

        Heuristic: if new_price is closer to the held_premium (more
        attainable), gain ≈ revisit_probability * proximity_factor.
        """
        old = self._broker_price_after_last_mod
        # For long: lower price = easier to fill.
        if self.thesis.direction > 0:
            improvement = max(0.0, old - new_price) / max(1e-6, old)
        else:
            improvement = max(0.0, new_price - old) / max(1e-6, old)
        gain = improvement * forecast.revisit_probability
        # Squash to [0, 1] band.
        return max(0.0, min(1.0, gain * 3.0))

    def _compute_ghost_price(self, *, market: MarketState,
                                mode: ExitMode,
                                forecast: RevisitForecast) -> float:
        """Re-cost the ghost exit price based on the current mode + signals."""
        thesis = self.thesis

        if mode == EXIT_MODE_KILL:
            # Exit at-market: ghost = current premium (or slightly below
            # for long longs so the limit fills).
            if thesis.direction > 0:
                return max(0.01, market.held_premium * 0.98)
            return market.held_premium * 1.02

        if mode == EXIT_MODE_DE_RISK:
            # Aggressive: ghost → minimum_profit_exit (or slightly better)
            if thesis.direction > 0:
                return max(thesis.minimum_profit_exit * 0.98,
                            market.held_premium * 1.005)
            return min(thesis.minimum_profit_exit * 1.02,
                         market.held_premium * 0.995)

        if mode == EXIT_MODE_CHASE_FILL:
            # Move limit slightly toward market to grab the fill.
            if thesis.direction > 0:
                return max(thesis.acceptable_exit_price * 0.95,
                            market.held_premium + market.derived_spread() * 0.5)
            return min(thesis.acceptable_exit_price * 1.05,
                         market.held_premium - market.derived_spread() * 0.5)

        if mode == EXIT_MODE_SHADE:
            # Lower target toward acceptable (long) / raise (short).
            if thesis.direction > 0:
                # Shade by 30% of distance to current price.
                gap = thesis.ideal_exit_price - market.held_premium
                shaded = thesis.ideal_exit_price - 0.30 * gap
                return max(thesis.acceptable_exit_price, shaded)
            gap = market.held_premium - thesis.ideal_exit_price
            shaded = thesis.ideal_exit_price + 0.30 * gap
            return min(thesis.acceptable_exit_price, shaded)

        # HARVEST: hold ideal target. If the last peak exceeded ideal,
        # consider raising slightly.
        if forecast.recent_peak_price is not None and forecast.high_trajectory == "rising":
            if thesis.direction > 0:
                return max(thesis.ideal_exit_price,
                            forecast.recent_peak_price * 0.99)
            return min(thesis.ideal_exit_price,
                         forecast.recent_peak_price * 1.01)
        return thesis.ideal_exit_price
