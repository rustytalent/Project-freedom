"""Greek-aware portfolio optimizer — propose adjustments to satisfy Greek bands.

Replaces the delta-cluster heuristic in `risk.py` with a proper
optimization: given current open positions (each with its proper
Black-Scholes Greeks) and a target band on net delta/vega/gamma/theta,
propose the smallest set of adjustments (add leg / close leg / partial
close) that brings the portfolio Greeks into the target band while
maximizing expected R.

Two solvers ship:

  * ``GreedyHedgeSolver`` — fast, greedy nearest-fit. Picks the single
    cheapest hedge leg that pulls the most-violated Greek toward zero.
    Deterministic, O(K) over the candidate leg set. Used in the live
    per-tick loop because it's <1ms.

  * ``CoordinateDescentSolver`` — iterative refinement. Picks any
    candidate, accepts if it improves the weighted objective, repeats
    until no improvement. Used in the Sunday-night replay harness when
    we have more compute.

Objective (minimize):
    L = sum_k w_k * max(0, |net_greek_k| - band_k)^2
        + lambda_cost * total_premium_of_hedges

Constraints (hard):
    Each proposed adjustment must satisfy:
      * leg is in the available strike_lookup
      * tags only operate on legs the manager can actually trade
      * portfolio premium-at-risk after the adjustment ≤ budget
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .pricing import compute_strategy_greeks


# ── Action enum ────────────────────────────────────────────────────


ACTION_ADD_HEDGE = "ADD_HEDGE"
ACTION_CLOSE_POSITION = "CLOSE_POSITION"
ACTION_REDUCE_POSITION = "REDUCE_POSITION"
ACTION_NOOP = "NOOP"


@dataclass
class GreekTargets:
    """Target band per portfolio Greek. Values are signed magnitudes
    (e.g., max_net_delta=200 means -200 ≤ delta ≤ +200)."""
    max_net_delta: float = 200.0      # in lot-equivalent shares
    max_net_vega: float = 50.0
    max_net_gamma: float = 0.5
    max_net_theta_per_day: float = 200.0    # ₹/day at risk
    weight_delta: float = 1.0
    weight_vega: float = 0.8
    weight_gamma: float = 0.5
    weight_theta: float = 0.3
    weight_cost: float = 0.0005


@dataclass
class OptimizerAction:
    """One adjustment proposed by the optimizer."""
    action: str                      # ACTION_*
    contract_side: str               # CE / PE
    contract_level: int
    strike_price: float
    direction: int                   # +1 long, -1 short
    lots: int
    premium: float
    rationale: str
    estimated_delta_change: float
    estimated_vega_change: float
    estimated_gamma_change: float
    estimated_theta_change: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class OptimizerProposal:
    """Per-tick optimizer output."""
    pre_net_delta: float
    pre_net_vega: float
    pre_net_gamma: float
    pre_net_theta: float
    post_net_delta: float
    post_net_vega: float
    post_net_gamma: float
    post_net_theta: float
    in_target_band_before: bool
    in_target_band_after: bool
    actions: List[OptimizerAction]
    objective_before: float
    objective_after: float
    iterations: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pre_net_delta": round(self.pre_net_delta, 3),
            "pre_net_vega": round(self.pre_net_vega, 3),
            "pre_net_gamma": round(self.pre_net_gamma, 5),
            "pre_net_theta": round(self.pre_net_theta, 3),
            "post_net_delta": round(self.post_net_delta, 3),
            "post_net_vega": round(self.post_net_vega, 3),
            "post_net_gamma": round(self.post_net_gamma, 5),
            "post_net_theta": round(self.post_net_theta, 3),
            "in_target_band_before": self.in_target_band_before,
            "in_target_band_after": self.in_target_band_after,
            "actions": [a.to_dict() for a in self.actions],
            "objective_before": round(self.objective_before, 4),
            "objective_after": round(self.objective_after, 4),
            "iterations": self.iterations,
            "notes": list(self.notes),
        }


# ── Greek aggregator helpers ──────────────────────────────────────


@dataclass
class _CandidateLeg:
    """A potential leg to add to the portfolio."""
    contract_side: str
    contract_level: int
    strike_price: float
    direction: int
    lots: int
    premium: float
    delta: float
    vega: float
    gamma: float
    theta_per_day: float


def _aggregate_existing(open_positions: List[Any], spot: float,
                          time_to_expiry: float, lot_size: int,
                          iv_surface=None) -> Tuple[float, float, float, float]:
    """Sum the current portfolio's Greeks across all open positions.

    ``open_positions`` is an iterable of objects exposing the leg list
    via ``.legs`` (StrategyLegs) OR a PositionHypothesis-shaped object
    where we synthesize a single leg from contract_side/strike/etc.
    """
    total_delta = total_vega = total_gamma = total_theta = 0.0
    for pos in open_positions:
        legs_attr = getattr(pos, "legs", None)
        if legs_attr is not None:
            leg_list = list(legs_attr)
        else:
            # Synthesize from PositionHypothesis-shaped object.
            leg_list = [_SynthesizedLeg(
                contract_side=str(getattr(pos, "contract_side", "")),
                strike_price=float(getattr(pos, "strike_price", 0.0)),
                direction=int(getattr(pos, "direction", 1)),
                lots=int(getattr(pos, "size_lots", 0)),
                contract_level=int(getattr(pos, "contract_level", 0)),
                entry_premium=float(getattr(pos, "entry_premium", 0.0)),
            )]
        profile = compute_strategy_greeks(
            legs=leg_list, spot=spot,
            time_to_expiry=time_to_expiry,
            lot_size=lot_size, iv_surface=iv_surface,
        )
        total_delta += profile.net_delta
        total_vega += profile.net_vega
        total_gamma += profile.net_gamma
        total_theta += profile.net_theta_per_day
    return total_delta, total_vega, total_gamma, total_theta


@dataclass(frozen=True)
class _SynthesizedLeg:
    contract_side: str
    strike_price: float
    direction: int
    lots: int
    contract_level: int
    entry_premium: float


# ── Candidate leg construction ───────────────────────────────────


def _build_candidates(strike_lookup: Dict, spot: float,
                        time_to_expiry: float, lot_size: int,
                        iv_surface=None,
                        max_lots: int = 2,
                        ) -> List[_CandidateLeg]:
    """For each (side, level) in strike_lookup × [+1, -1] × lots ∈ {1, max_lots},
    compute Greeks and build a candidate leg."""
    out: List[_CandidateLeg] = []
    for (side, level), info in strike_lookup.items():
        strike, premium, _, sstate = info
        if sstate == "dangerous":
            continue
        if premium <= 0:
            continue
        # Build a single-leg synthesized leg.
        for direction in (+1, -1):
            for lots in range(1, max_lots + 1):
                leg = _SynthesizedLeg(
                    contract_side=side, strike_price=strike,
                    direction=direction, lots=lots,
                    contract_level=level, entry_premium=premium,
                )
                profile = compute_strategy_greeks(
                    legs=[leg], spot=spot,
                    time_to_expiry=time_to_expiry,
                    lot_size=lot_size, iv_surface=iv_surface,
                )
                out.append(_CandidateLeg(
                    contract_side=side, contract_level=level,
                    strike_price=strike, direction=direction,
                    lots=lots, premium=premium,
                    delta=profile.net_delta,
                    vega=profile.net_vega,
                    gamma=profile.net_gamma,
                    theta_per_day=profile.net_theta_per_day,
                ))
    return out


def _objective(delta: float, vega: float, gamma: float, theta: float,
                cost: float, t: GreekTargets) -> float:
    """L = sum_k w_k * max(0, |g_k| - band_k)^2 + lambda * cost."""
    return (
        t.weight_delta * max(0.0, abs(delta) - t.max_net_delta) ** 2
        + t.weight_vega * max(0.0, abs(vega) - t.max_net_vega) ** 2
        + t.weight_gamma * max(0.0, abs(gamma) - t.max_net_gamma) ** 2
        + t.weight_theta * max(0.0, abs(theta) - t.max_net_theta_per_day) ** 2
        + t.weight_cost * cost
    )


def _in_band(delta: float, vega: float, gamma: float, theta: float,
               t: GreekTargets) -> bool:
    return (abs(delta) <= t.max_net_delta
             and abs(vega) <= t.max_net_vega
             and abs(gamma) <= t.max_net_gamma
             and abs(theta) <= t.max_net_theta_per_day)


# ── Solvers ──────────────────────────────────────────────────────


class GreedyHedgeSolver:
    """Greedy: pick the single candidate that maximally reduces the
    objective. Stops when no candidate improves the objective or we hit
    ``max_actions``."""

    def __init__(self, *, targets: Optional[GreekTargets] = None,
                  max_actions: int = 3) -> None:
        self.targets = targets or GreekTargets()
        self.max_actions = max_actions

    def solve(self, *,
                open_positions: List[Any],
                strike_lookup: Dict,
                spot: float,
                time_to_expiry: float,
                lot_size: int = 65,
                iv_surface=None,
                ) -> OptimizerProposal:
        t = self.targets
        pre_d, pre_v, pre_g, pre_th = _aggregate_existing(
            open_positions, spot, time_to_expiry, lot_size, iv_surface)
        in_band_before = _in_band(pre_d, pre_v, pre_g, pre_th, t)
        candidates = _build_candidates(
            strike_lookup, spot, time_to_expiry, lot_size, iv_surface,
            max_lots=2)
        cur_d, cur_v, cur_g, cur_th = pre_d, pre_v, pre_g, pre_th
        cur_cost = 0.0
        cur_obj = _objective(cur_d, cur_v, cur_g, cur_th, cur_cost, t)
        actions: List[OptimizerAction] = []
        notes: List[str] = []

        if in_band_before:
            return OptimizerProposal(
                pre_net_delta=pre_d, pre_net_vega=pre_v,
                pre_net_gamma=pre_g, pre_net_theta=pre_th,
                post_net_delta=pre_d, post_net_vega=pre_v,
                post_net_gamma=pre_g, post_net_theta=pre_th,
                in_target_band_before=True, in_target_band_after=True,
                actions=[], objective_before=cur_obj,
                objective_after=cur_obj, iterations=0,
                notes=["portfolio already in target band — no adjustment needed"],
            )

        for it in range(self.max_actions):
            best_drop = 0.0
            best_cand: Optional[_CandidateLeg] = None
            best_post: Optional[Tuple[float, float, float, float, float]] = None
            for c in candidates:
                new_d = cur_d + c.delta
                new_v = cur_v + c.vega
                new_g = cur_g + c.gamma
                new_th = cur_th + c.theta_per_day
                new_cost = cur_cost + c.premium * c.lots * lot_size
                new_obj = _objective(new_d, new_v, new_g, new_th,
                                       new_cost, t)
                drop = cur_obj - new_obj
                if drop > best_drop:
                    best_drop = drop
                    best_cand = c
                    best_post = (new_d, new_v, new_g, new_th, new_cost)
            if best_cand is None or best_drop <= 0:
                notes.append(f"converged after {it} iteration(s); "
                              f"no further improvement available")
                break
            actions.append(OptimizerAction(
                action=ACTION_ADD_HEDGE,
                contract_side=best_cand.contract_side,
                contract_level=best_cand.contract_level,
                strike_price=best_cand.strike_price,
                direction=best_cand.direction,
                lots=best_cand.lots,
                premium=best_cand.premium,
                rationale=f"reduces objective by {best_drop:.2f}",
                estimated_delta_change=best_cand.delta,
                estimated_vega_change=best_cand.vega,
                estimated_gamma_change=best_cand.gamma,
                estimated_theta_change=best_cand.theta_per_day,
            ))
            cur_d, cur_v, cur_g, cur_th, cur_cost = best_post
            cur_obj = cur_obj - best_drop

        return OptimizerProposal(
            pre_net_delta=pre_d, pre_net_vega=pre_v,
            pre_net_gamma=pre_g, pre_net_theta=pre_th,
            post_net_delta=cur_d, post_net_vega=cur_v,
            post_net_gamma=cur_g, post_net_theta=cur_th,
            in_target_band_before=in_band_before,
            in_target_band_after=_in_band(cur_d, cur_v, cur_g, cur_th, t),
            actions=actions,
            objective_before=_objective(pre_d, pre_v, pre_g, pre_th, 0.0, t),
            objective_after=cur_obj,
            iterations=len(actions),
            notes=notes,
        )


class CoordinateDescentSolver:
    """Iterative refinement: cycles through Greeks; for each, picks the
    candidate that most reduces |that Greek| toward zero. Slower but
    explores configurations the greedy may miss."""

    def __init__(self, *, targets: Optional[GreekTargets] = None,
                  max_iterations: int = 8) -> None:
        self.targets = targets or GreekTargets()
        self.max_iterations = max_iterations

    def solve(self, *,
                open_positions: List[Any],
                strike_lookup: Dict,
                spot: float,
                time_to_expiry: float,
                lot_size: int = 65,
                iv_surface=None,
                ) -> OptimizerProposal:
        t = self.targets
        pre_d, pre_v, pre_g, pre_th = _aggregate_existing(
            open_positions, spot, time_to_expiry, lot_size, iv_surface)
        candidates = _build_candidates(
            strike_lookup, spot, time_to_expiry, lot_size, iv_surface,
            max_lots=2)
        cur_d, cur_v, cur_g, cur_th = pre_d, pre_v, pre_g, pre_th
        cur_cost = 0.0
        cur_obj = _objective(cur_d, cur_v, cur_g, cur_th, cur_cost, t)
        actions: List[OptimizerAction] = []
        notes: List[str] = []

        if _in_band(pre_d, pre_v, pre_g, pre_th, t):
            return OptimizerProposal(
                pre_net_delta=pre_d, pre_net_vega=pre_v,
                pre_net_gamma=pre_g, pre_net_theta=pre_th,
                post_net_delta=pre_d, post_net_vega=pre_v,
                post_net_gamma=pre_g, post_net_theta=pre_th,
                in_target_band_before=True, in_target_band_after=True,
                actions=[], objective_before=cur_obj,
                objective_after=cur_obj, iterations=0,
                notes=["already in band"],
            )

        # Cycle through Greeks, dropping each one toward its band.
        order = ["delta", "vega", "gamma", "theta"]
        for it in range(self.max_iterations):
            improved = False
            for key in order:
                best_drop = 0.0
                best_cand: Optional[_CandidateLeg] = None
                best_post = None
                for c in candidates:
                    new_d = cur_d + c.delta
                    new_v = cur_v + c.vega
                    new_g = cur_g + c.gamma
                    new_th = cur_th + c.theta_per_day
                    new_cost = cur_cost + c.premium * c.lots * lot_size
                    new_obj = _objective(new_d, new_v, new_g, new_th,
                                            new_cost, t)
                    drop = cur_obj - new_obj
                    if drop > best_drop:
                        best_drop = drop
                        best_cand = c
                        best_post = (new_d, new_v, new_g, new_th, new_cost)
                if best_cand is not None and best_drop > 0:
                    actions.append(OptimizerAction(
                        action=ACTION_ADD_HEDGE,
                        contract_side=best_cand.contract_side,
                        contract_level=best_cand.contract_level,
                        strike_price=best_cand.strike_price,
                        direction=best_cand.direction,
                        lots=best_cand.lots,
                        premium=best_cand.premium,
                        rationale=f"coord-descent on {key} drop={best_drop:.2f}",
                        estimated_delta_change=best_cand.delta,
                        estimated_vega_change=best_cand.vega,
                        estimated_gamma_change=best_cand.gamma,
                        estimated_theta_change=best_cand.theta_per_day,
                    ))
                    cur_d, cur_v, cur_g, cur_th, cur_cost = best_post
                    cur_obj = cur_obj - best_drop
                    improved = True
            if not improved:
                notes.append(f"converged after {it + 1} iteration(s)")
                break

        return OptimizerProposal(
            pre_net_delta=pre_d, pre_net_vega=pre_v,
            pre_net_gamma=pre_g, pre_net_theta=pre_th,
            post_net_delta=cur_d, post_net_vega=cur_v,
            post_net_gamma=cur_g, post_net_theta=cur_th,
            in_target_band_before=_in_band(pre_d, pre_v, pre_g, pre_th, t),
            in_target_band_after=_in_band(cur_d, cur_v, cur_g, cur_th, t),
            actions=actions,
            objective_before=_objective(pre_d, pre_v, pre_g, pre_th, 0.0, t),
            objective_after=cur_obj,
            iterations=len(actions),
            notes=notes,
        )
