"""PortfolioExitManager — coordinates exits across all open positions.

Five positions in the book ≠ five independent exit engines firing
modifications without coordination. The founder's exact concern:

  "For five positions, never manage exits independently like five dumb
   orders. The portfolio manager tracks net bullish exposure, net
   bearish exposure, which side is currently being proven right, which
   position has weakest thesis, which order deserves modification
   budget first."

This module sits ABOVE the per-position AdaptiveExitEngines and:

  1. Tags each position into a directional cluster (bullish / bearish /
     neutral)
  2. Computes the regime confidence — which side is being proven right
  3. Identifies the weakest-thesis position (likeliest to need rescue)
  4. PRIORITIZES modification approvals when multiple positions want
     to modify in the same tick — only the top-priority N get through,
     subject to a per-tick cap (so we don't burn 5 mods on one bad
     market lurch)
  5. Surfaces a portfolio-level decision report for the cockpit

It does NOT override the per-position engines — they still compute
their own ExitDecision. The manager simply gates which modifications
actually get sent to the broker each tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .engine import AdaptiveExitEngine, ExitDecision, MarketState


# ── Config + decision shape ───────────────────────────────────────


@dataclass
class PortfolioExitManagerConfig:
    """Knobs."""
    max_modifications_per_tick: int = 2   # don't burn 5 mods on one tick
    weak_thesis_decay_threshold: float = 0.20
    high_priority_modes: tuple = (
        "KILL", "DE_RISK", "CHASE_FILL",
    )


@dataclass
class PortfolioExitDecision:
    """Per-tick portfolio-level exit decision."""
    per_position: List[Dict[str, Any]]   # per-position ExitDecision dicts
    cluster_bullish_count: int
    cluster_bearish_count: int
    weakest_thesis_position_id: Optional[str]
    modifications_approved_this_tick: List[str]   # position_ids
    modifications_deferred_this_tick: List[str]
    regime_winning_side: int             # +1 / -1 / 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_position": list(self.per_position),
            "cluster_bullish_count": self.cluster_bullish_count,
            "cluster_bearish_count": self.cluster_bearish_count,
            "weakest_thesis_position_id": self.weakest_thesis_position_id,
            "modifications_approved_this_tick": list(
                self.modifications_approved_this_tick),
            "modifications_deferred_this_tick": list(
                self.modifications_deferred_this_tick),
            "regime_winning_side": self.regime_winning_side,
            "notes": list(self.notes),
        }


# ── Manager ───────────────────────────────────────────────────────


class PortfolioExitManager:
    """Coordinates per-position AdaptiveExitEngines."""

    def __init__(self,
                  cfg: Optional[PortfolioExitManagerConfig] = None) -> None:
        self.cfg = cfg or PortfolioExitManagerConfig()
        self.engines: Dict[str, AdaptiveExitEngine] = {}
        # Position metadata for clustering: position_id → direction
        self._position_directions: Dict[str, int] = {}
        self._last_decision: Optional[PortfolioExitDecision] = None

    # ── Lifecycle ──────────────────────────────────────────────────

    def register(self, position_id: str,
                   engine: AdaptiveExitEngine) -> None:
        self.engines[position_id] = engine
        self._position_directions[position_id] = engine.thesis.direction

    def unregister(self, position_id: str) -> None:
        self.engines.pop(position_id, None)
        self._position_directions.pop(position_id, None)

    # ── Per-tick driver ────────────────────────────────────────────

    def evaluate(self,
                   market_by_position: Dict[str, MarketState],
                   *,
                   regime_winning_side: int = 0,
                   ) -> PortfolioExitDecision:
        """Drive one tick across all registered engines.

        ``market_by_position`` maps position_id → MarketState (the
        manager builds these from snapshot + held_quotes).

        ``regime_winning_side`` is the engine's current best read on
        which side the regime is favoring (+1 bull / -1 bear / 0).
        """
        cfg = self.cfg
        notes: List[str] = []

        # 1. Per-position observation (always run — even if we defer
        #    the actual modification, the ghost prices update for free).
        raw_decisions: List[ExitDecision] = []
        for pid, market in market_by_position.items():
            if pid not in self.engines:
                continue
            decision = self.engines[pid].observe(market)
            raw_decisions.append(decision)

        # 2. Cluster counts.
        bullish_count = sum(1 for d in raw_decisions
                              if self._position_directions.get(d.position_id, 0) > 0)
        bearish_count = sum(1 for d in raw_decisions
                              if self._position_directions.get(d.position_id, 0) < 0)

        # 3. Find weakest-thesis position (highest decay this tick).
        weakest_id: Optional[str] = None
        weakest_decay = 0.0
        for d, ms in zip(raw_decisions, market_by_position.values()):
            if ms.thesis_confidence_decay > weakest_decay:
                weakest_decay = ms.thesis_confidence_decay
                weakest_id = d.position_id

        # 4. Prioritize modifications.
        wanted_mods: List[ExitDecision] = [d for d in raw_decisions
                                              if d.broker_modification_proposed]
        approved_ids: List[str] = []
        deferred_ids: List[str] = []

        if not wanted_mods:
            notes.append("no positions wanted a modification this tick")
        else:
            # Priority sort: KILL first, then DE_RISK, then CHASE_FILL,
            # then others. Among same priority: weakest thesis first.
            def _priority_key(d: ExitDecision) -> tuple:
                mode_rank = {
                    "KILL": 0, "DE_RISK": 1, "CHASE_FILL": 2,
                    "SHADE": 3, "HARVEST": 4,
                }.get(d.exit_mode, 5)
                # Higher decay = should be addressed first (negate for asc sort)
                decay = market_by_position[d.position_id].thesis_confidence_decay
                return (mode_rank, -decay)

            wanted_mods.sort(key=_priority_key)

            for d in wanted_mods:
                if (len(approved_ids) >= cfg.max_modifications_per_tick
                        and d.exit_mode not in cfg.high_priority_modes):
                    deferred_ids.append(d.position_id)
                    continue
                approved_ids.append(d.position_id)

            if deferred_ids:
                notes.append(
                    f"deferred {len(deferred_ids)} mod(s) — per-tick cap "
                    f"{cfg.max_modifications_per_tick} reached"
                )

        # 5. For any DEFERRED position, roll back the gate-approved spend
        #    that the per-position engine already charged.
        for pid in deferred_ids:
            engine = self.engines.get(pid)
            if engine is None:
                continue
            # The engine's budget already incremented; refund the most
            # recent history entry by undoing the bucket count.
            if engine.budget.history:
                last = engine.budget.history.pop()
                if last.bucket in engine.budget.used:
                    engine.budget.used[last.bucket] = max(
                        0, engine.budget.used[last.bucket] - 1)
                # Revert the engine's tracked broker price (the engine
                # bumped it after gate approval).
                # We re-read from the prior thesis broker price snapshot.
                # Simpler: reset to the ghost minus the approved jump.
                engine._broker_price_after_last_mod = (
                    engine.thesis.current_broker_exit_price
                    if engine.thesis.current_broker_exit_price > 0 else 0.0)
            # Mark on the decision (mutate the per-position dict).
            for d in raw_decisions:
                if d.position_id == pid:
                    d.broker_modification_proposed = False
                    d.notes.append("portfolio-deferred: per-tick cap")
                    break

        decision = PortfolioExitDecision(
            per_position=[d.to_dict() for d in raw_decisions],
            cluster_bullish_count=bullish_count,
            cluster_bearish_count=bearish_count,
            weakest_thesis_position_id=weakest_id,
            modifications_approved_this_tick=approved_ids,
            modifications_deferred_this_tick=deferred_ids,
            regime_winning_side=regime_winning_side,
            notes=notes,
        )
        self._last_decision = decision
        return decision

    def summary(self) -> Dict[str, Any]:
        return {
            "n_engines": len(self.engines),
            "directions": dict(self._position_directions),
            "last_decision": (self._last_decision.to_dict()
                                if self._last_decision is not None else None),
        }
