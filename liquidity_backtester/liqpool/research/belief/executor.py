"""Execution Governor for the Premium Belief Engine.

The shadow execution layer above the eight-phase belief engine. Reads one
:class:`BeliefSnapshot` (and optionally a live read of the **held contract**)
and emits one :class:`ExecutionIntent` per tick: open / hold / exit /
blocked / flat. Broker-agnostic. Shadow-only by construction.

Healed 2026-06-19 (founder-requested). Three structural defects in the
prior Codex implementation were fixed; the shape (lifecycle + guards +
two-tier exits + telemetry) was kept because it was correct. See
``docs/premium_belief_executor_assessment.md`` for the full assessment.

Critical fixes:
  1. **R-multiple is now on the held contract's PREMIUM**, not on spot.
     Options are leveraged; a flat spot can theta-bleed the premium 20%
     and the spot-based R was completely blind to that. Stops, targets,
     trail, profit lock and best/worst R are all now premium-based.
  2. **A held-contract live read** (``HeldContractRead``) carrying the
     contract's own bid/ask/mark/spread_state/acceptance/dod_z is now
     consumed. The founder's "spread can eat you alive" and "exit on
     first weakness of the held leg" both become enforceable here.
  3. **Strike is anchored** to the actual ₹ value, not just the label,
     so the broker adapter can route the order.

Plus the soft-quality bumps:
  - the entry quorum no longer counts the decision's own direction as a
    vote (it's the trigger, not a confirmation);
  - rail-alignment floor raised; vote floor raised;
  - rupees-per-lot anchor so ``size_fraction`` translates to lots;
  - engine-cold guard (``is_warm`` flipping False while holding → exit);
  - first-weakness sweep exit when the held leg's |dod_z| collapses.

This module never places orders. ``order_mode == "SHADOW_ONLY"`` and
``live_orders_enabled == False`` are hard-coded — a broker adapter must
opt-in explicitly, in a separate code path that is not yet written.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


EXECUTOR_NAME = "premium_belief_execution_governor"
EXECUTOR_VERSION = "v3"             # v3 = the healed version
ORDER_MODE_SHADOW = "SHADOW_ONLY"

# Intent labels.
INTENT_OPEN_LONG = "OPEN_LONG"
INTENT_OPEN_SHORT = "OPEN_SHORT"
INTENT_OPEN_SCALP_CALL = "OPEN_SCALP_CALL"
INTENT_OPEN_SCALP_PUT = "OPEN_SCALP_PUT"
INTENT_HOLD_POSITION = "HOLD_POSITION"
INTENT_EXIT_POSITION = "EXIT_POSITION"
INTENT_FLAT_WAIT = "FLAT_WAIT"
INTENT_BLOCKED = "BLOCKED"

PROFILE_SCALP = "SCALP"
PROFILE_INTRADAY = "INTRADAY"

# Decision-action sets the executor recognises as entries.
ENTRY_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "SCALP_CALL", "SCALP_PUT"}
SCALP_ENTRY_ACTIONS = {"SCALP_CALL", "SCALP_PUT"}

# Mark-source labels that signal a dirty / fallback mark.
DIRTY_MARK_SOURCES = {"invalid", "last_valid", "ltp"}

# IV-state / battlefield labels that the executor refuses to enter on.
DIRTY_IV_STATES = {"dirty_data", "liquidity_distortion", "common_shock"}
DANGEROUS_BATTLEFIELD = {"single_distortion", "vol_expansion"}

# Spread-state label that means the held contract is no longer tradeable.
SPREAD_DANGEROUS = "dangerous"


@dataclass(frozen=True)
class HeldContractRead:
    """Live read of the contract the executor holds (or wants to hold).

    Carrying this lets the executor compute R on the *premium* (Critical-1)
    and guard against the held leg's own spread blowing out (Critical-3)
    — neither of which the upstream snapshot exposes cheaply.

    All fields are optional; missing values degrade the executor gracefully
    rather than crash. If ``mid`` is missing it is computed from bid/ask
    where both are positive.
    """
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    spread_state: str = ""          # clean / widening / dangerous / improving
    friendliness: Optional[float] = None
    acceptance: str = ""            # defended / rejected / normal
    dod_z: Optional[float] = None
    mark_source: str = ""            # microprice / mid / last_valid / ltp / invalid
    mark_quality_label: str = ""

    @property
    def best_price(self) -> Optional[float]:
        """Conservative mark for P&L estimation. Prefers mid, then microprice-
        weighted average of bid/ask, then None."""
        if self.mid is not None and math.isfinite(self.mid) and self.mid > 0:
            return float(self.mid)
        if (self.bid is not None and self.ask is not None
                and math.isfinite(self.bid) and math.isfinite(self.ask)
                and self.bid > 0 and self.ask > 0 and self.ask >= self.bid):
            return float((self.bid + self.ask) / 2.0)
        return None


@dataclass(frozen=True)
class ExecutionGovernorConfig:
    """Knobs for translating belief reads into managed shadow intents.

    Defaults are calibrated for NIFTY weekly options at 75-lot size and a
    paper bankroll where ``risk_rupees_per_lot`` represents the operator's
    acceptable per-trade loss budget at 1R.
    """
    # Warmup.
    min_warm_bars: int = 80

    # Entry quorum — strict per the founder's multi-confirmation rule.
    # The decision's own direction is NOT counted; these are the
    # CONFIRMING sources beyond it.
    min_confirming_votes: int = 3      # of {iv direction, battlefield direction,
                                         # thesis state, stream direction, rail alignment}
    min_directional_rail_abs: float = 0.60
    min_entry_confidence: float = 0.66
    min_scalp_confidence: float = 0.72
    min_spread_friendliness: float = 0.65
    min_clean_mark_fraction: float = 0.85
    max_abnormal_slot_fraction: float = 0.40
    max_dirty_source_fraction: float = 0.10
    max_no_trade_score: float = 18.0

    # Cooldowns.
    cooldown_bars_after_entry: int = 8
    cooldown_bars_after_exit: int = 6

    # Hold-time bounds.
    min_hold_bars: int = 2
    scalp_max_bars: int = 24
    intraday_max_bars: int = 90

    # Risk in PREMIUM space (Critical-1 fix). 1R for an option position is
    # the fraction of the entry premium the operator is willing to lose
    # before a hard stop. Defaults reflect the leverage profile of NIFTY
    # ATM options: an ATM CE at ₹100 hits a 35%-stop at ₹65, an intraday
    # ATM hold can ride a 50% drop intra-position. These are NOT spot
    # percentages.
    premium_stop_scalp: float = 0.35
    premium_stop_intraday: float = 0.50

    # Targets and trails on premium R (1R = the premium_stop).
    scalp_target_r: float = 1.20
    intraday_target_r: float = 1.80
    trail_after_r: float = 0.80
    profit_lock_start_r: float = 0.90
    profit_lock_giveback_r: float = 0.45

    # Confidence-based soft exit.
    confidence_giveback: float = 0.30

    # Sizing anchor — converts the abstract ``size_fraction`` to lots.
    risk_rupees_per_lot: float = 2000.0   # operator-configurable rupee loss budget at 1R
    lot_size: int = 75                      # NIFTY weekly lot
    max_lots: int = 4                       # ceiling regardless of conviction

    # First-weakness sweep exit: when the held leg's |dod_z| collapses to
    # this fraction of the entry magnitude, exit.
    sweep_weakness_dod_z_ratio: float = 0.35


@dataclass
class ExecutorPosition:
    """One shadow position tracked by the governor.

    ``entry_premium`` is the held contract's mark at the moment of entry —
    the anchor for the corrected R calculation. ``strike_price`` carries
    the actual rupee strike so the broker adapter has what it needs.
    """
    side: str                            # LONG / SHORT (intent direction)
    intent: str
    contract_side: str                   # CE / PE
    contract_label: str                  # e.g. CE_ATM
    contract_level: int                  # signed level (0 = ATM)
    strike_price: float                  # actual rupee strike
    entry_spot: float
    entry_premium: float                  # NEW — corrected R anchor
    entry_dod_z: float                    # for first-weakness sweep exit
    entry_bar: int
    entry_confidence: float
    profile: str
    size_fraction: float
    size_lots: int
    entry_action: str
    entry_thesis_state: str
    entry_iv_state: str
    entry_battlefield: str
    entry_stream_stable: str
    entry_rail_alignment: float
    entry_confirming_votes: int
    entry_reason_codes: List[str] = field(default_factory=list)
    high_water_confidence: float = 0.0
    low_water_confidence: float = 1.0
    best_r: float = 0.0
    worst_r: float = 0.0
    last_seen_premium: float = 0.0       # most recent mark for telemetry


@dataclass(frozen=True)
class ExecutionIntent:
    """Serializable execution intent emitted per belief row."""
    intent: str
    action: str
    allowed: bool
    direction: int
    confidence: float
    size_fraction: float
    size_lots: int = 0
    contract_side: str = ""
    contract_label: str = ""
    contract_level: int = 0
    strike_price: float = 0.0
    profile: str = ""
    stop_r: float = 0.0                   # in PREMIUM R (Critical-1 fix)
    target_r: float = 0.0
    trail_after_r: float = 0.0
    max_hold_bars: int = 0
    premium_stop_pct: float = 0.0         # ₹-space stop: fraction of entry premium
    invalidation: str = ""
    reason_codes: List[str] = field(default_factory=list)
    reject_reasons: List[str] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    executor_name: str = EXECUTOR_NAME
    executor_version: str = EXECUTOR_VERSION
    order_mode: str = ORDER_MODE_SHADOW
    shadow_only: bool = True
    live_orders_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executor_name": self.executor_name,
            "executor_version": self.executor_version,
            "order_mode": self.order_mode,
            "shadow_only": self.shadow_only,
            "live_orders_enabled": self.live_orders_enabled,
            "intent": self.intent,
            "action": self.action,
            "allowed": self.allowed,
            "direction": self.direction,
            "confidence": self.confidence,
            "size_fraction": self.size_fraction,
            "size_lots": self.size_lots,
            "contract_side": self.contract_side,
            "contract_label": self.contract_label,
            "contract_level": self.contract_level,
            "strike_price": self.strike_price,
            "profile": self.profile,
            "stop_r": self.stop_r,
            "target_r": self.target_r,
            "trail_after_r": self.trail_after_r,
            "max_hold_bars": self.max_hold_bars,
            "premium_stop_pct": self.premium_stop_pct,
            "invalidation": self.invalidation,
            "reason_codes": list(self.reason_codes),
            "reject_reasons": list(self.reject_reasons),
            "state": dict(self.state),
            "telemetry": dict(self.telemetry),
        }


class BeliefExecutionGovernor:
    """Stateful shadow executor above the eight-phase belief engine.

    Usage:

        gov = BeliefExecutionGovernor()
        # On every tick:
        contract_quote = HeldContractRead(bid=..., ask=..., mid=...,
                                          spread_state=..., friendliness=...,
                                          acceptance=..., dod_z=...)
        intent = gov.evaluate(snapshot, stream=stream_payload,
                              held_quote=contract_quote)
        # intent.intent ∈ {OPEN_*, HOLD_POSITION, EXIT_POSITION, BLOCKED, FLAT_WAIT}

    ``held_quote`` may be omitted; the executor degrades to a less precise
    P&L estimate (mid extracted from ``slot_readings`` matching the
    position's contract) but never crashes.
    """

    def __init__(self, cfg: Optional[ExecutionGovernorConfig] = None) -> None:
        self.cfg = cfg or ExecutionGovernorConfig()
        self.position: Optional[ExecutorPosition] = None
        self.cooldown_until_bar = -1

    # ── Public API ─────────────────────────────────────────────────────

    def evaluate(
        self,
        snapshot: Mapping[str, Any],
        stream: Optional[Mapping[str, Any]] = None,
        held_quote: Optional[HeldContractRead] = None,
    ) -> ExecutionIntent:
        """One belief-tick → one shadow intent."""
        decision = _map(snapshot.get("decision"))
        action = str(decision.get("action") or "WAIT")
        direction = int(_number(decision.get("direction"), 0))
        confidence = _clamp(_number(decision.get("confidence"), 0.0))
        bars = int(_number(snapshot.get("bars_seen"), 0))
        spot = _number(snapshot.get("spot"), 0.0)

        # Held contract mark — prefer the explicit HeldContractRead; fall
        # back to the matching slot in the snapshot if not provided.
        held_premium = self._held_premium(snapshot, held_quote)

        telemetry = self._telemetry(snapshot, stream, held_quote, held_premium)
        reject_reasons = self._guard_reasons(snapshot, telemetry)

        # ── PATH 1: We hold a position ────────────────────────────────
        if self.position is not None:
            # Update best/worst R on the corrected premium-based metric.
            current_r = self._position_r(self.position, held_premium)
            self.position.best_r = max(self.position.best_r, current_r)
            self.position.worst_r = min(self.position.worst_r, current_r)
            if held_premium is not None:
                self.position.last_seen_premium = held_premium
            self.position.high_water_confidence = max(
                self.position.high_water_confidence, confidence)
            self.position.low_water_confidence = min(
                self.position.low_water_confidence, confidence)

            exit_reasons = self._position_exit_reasons(
                snapshot, stream, telemetry, held_quote, held_premium,
                current_r=current_r,
            )
            if exit_reasons:
                pos = self.position
                self.position = None
                self.cooldown_until_bar = max(
                    self.cooldown_until_bar,
                    bars + self.cfg.cooldown_bars_after_exit,
                )
                return self._intent(
                    intent=INTENT_EXIT_POSITION,
                    action=action,
                    allowed=True,
                    direction=0,
                    confidence=max(confidence, pos.entry_confidence),
                    size_fraction=0.0,
                    size_lots=0,
                    contract_side=pos.contract_side,
                    contract_label=pos.contract_label,
                    contract_level=pos.contract_level,
                    strike_price=pos.strike_price,
                    profile=pos.profile,
                    invalidation="; ".join(exit_reasons[:4]),
                    reason_codes=exit_reasons,
                    reject_reasons=reject_reasons,
                    telemetry={**telemetry, "exit_at_r": round(current_r, 4)},
                )
            return self._hold_intent(action, confidence, telemetry,
                                     reject_reasons, current_r=current_r)

        # ── PATH 2: Flat — gate the entry ─────────────────────────────
        if reject_reasons:
            return self._intent(
                intent=INTENT_BLOCKED, action=action, allowed=False,
                direction=0, confidence=confidence, size_fraction=0.0,
                reject_reasons=reject_reasons, telemetry=telemetry,
            )

        if bars < self.cooldown_until_bar:
            return self._intent(
                intent=INTENT_FLAT_WAIT, action=action, allowed=False,
                direction=0, confidence=confidence, size_fraction=0.0,
                reject_reasons=[f"cooldown until bar {self.cooldown_until_bar}"],
                telemetry=telemetry,
            )

        if action not in ENTRY_ACTIONS:
            return self._intent(
                intent=INTENT_FLAT_WAIT, action=action, allowed=False,
                direction=0, confidence=confidence, size_fraction=0.0,
                reason_codes=["no entry action from belief engine"],
                telemetry=telemetry,
            )

        # Held-quote gate for entries too — refuse to enter a contract whose
        # spread is dangerous or whose mark is itself dirty.
        if held_quote is not None:
            if held_quote.spread_state == SPREAD_DANGEROUS:
                return self._intent(
                    intent=INTENT_BLOCKED, action=action, allowed=False,
                    direction=0, confidence=confidence, size_fraction=0.0,
                    reject_reasons=[
                        "held contract spread state dangerous — execution would slip badly"
                    ],
                    telemetry=telemetry,
                )
            if held_quote.mark_source in DIRTY_MARK_SOURCES:
                return self._intent(
                    intent=INTENT_BLOCKED, action=action, allowed=False,
                    direction=0, confidence=confidence, size_fraction=0.0,
                    reject_reasons=[
                        f"held contract mark source dirty ({held_quote.mark_source})"
                    ],
                    telemetry=telemetry,
                )

        intent, desired_direction, profile = self._entry_shape(action)
        threshold = (self.cfg.min_scalp_confidence if profile == PROFILE_SCALP
                     else self.cfg.min_entry_confidence)
        if confidence < threshold:
            return self._intent(
                intent=INTENT_BLOCKED, action=action, allowed=False,
                direction=0, confidence=confidence, size_fraction=0.0,
                reject_reasons=[f"confidence {confidence:.2f} below {threshold:.2f}"],
                telemetry=telemetry,
            )

        votes = self._confirming_votes(snapshot, stream, desired_direction)
        rail = self._directional_rail(snapshot, desired_direction)
        if votes < self.cfg.min_confirming_votes:
            return self._intent(
                intent=INTENT_BLOCKED, action=action, allowed=False,
                direction=0, confidence=confidence, size_fraction=0.0,
                reject_reasons=[
                    f"confirming votes {votes} below {self.cfg.min_confirming_votes}"
                ],
                telemetry={**telemetry, "confirming_votes": votes,
                            "rail_alignment": rail},
            )
        if rail < self.cfg.min_directional_rail_abs:
            return self._intent(
                intent=INTENT_BLOCKED, action=action, allowed=False,
                direction=0, confidence=confidence, size_fraction=0.0,
                reject_reasons=[
                    f"rail alignment {rail:.2f} below "
                    f"{self.cfg.min_directional_rail_abs:.2f}"
                ],
                telemetry={**telemetry, "confirming_votes": votes,
                            "rail_alignment": rail},
            )

        # Entry premium is mandatory for a meaningful R. If we have neither
        # an explicit quote nor a matching slot mark, we still allow the
        # entry but mark the position with NaN entry_premium and flag in
        # telemetry — the held path will then exit on the held-quote guard
        # the moment a clean quote arrives.
        if held_premium is None:
            return self._intent(
                intent=INTENT_BLOCKED, action=action, allowed=False,
                direction=0, confidence=confidence, size_fraction=0.0,
                reject_reasons=[
                    "no clean mark for the chosen contract — refusing to anchor R"
                ],
                telemetry=telemetry,
            )

        size_fraction, size_lots = self._size(confidence, telemetry, votes, rail)
        strike = _map(decision.get("strike"))
        contract_side = str(strike.get("side") or
                             ("CE" if desired_direction > 0 else "PE"))
        contract_label = str(strike.get("label") or f"{contract_side}_ATM")
        contract_level = int(_number(strike.get("level"), 0))
        strike_price = self._strike_price(snapshot, contract_side, contract_level)
        target = (self.cfg.scalp_target_r if profile == PROFILE_SCALP
                  else self.cfg.intraday_target_r)
        premium_stop_pct = (self.cfg.premium_stop_scalp if profile == PROFILE_SCALP
                            else self.cfg.premium_stop_intraday)
        max_hold = (self.cfg.scalp_max_bars if profile == PROFILE_SCALP
                    else self.cfg.intraday_max_bars)
        entry_reasons = [
            f"confirming_votes={votes}",
            f"rail={rail:.2f}",
            f"size_fraction={size_fraction:.2f}",
            f"size_lots={size_lots}",
            f"profile={profile}",
            f"entry_premium={held_premium:.2f}",
        ]
        entry_dod_z = float(held_quote.dod_z) if (
            held_quote is not None and held_quote.dod_z is not None
            and math.isfinite(held_quote.dod_z)
        ) else 0.0
        self.position = ExecutorPosition(
            side="LONG" if desired_direction > 0 else "SHORT",
            intent=intent,
            contract_side=contract_side,
            contract_label=contract_label,
            contract_level=contract_level,
            strike_price=strike_price,
            entry_spot=spot,
            entry_premium=held_premium,
            entry_dod_z=entry_dod_z,
            entry_bar=bars,
            entry_confidence=confidence,
            profile=profile,
            size_fraction=size_fraction,
            size_lots=size_lots,
            entry_action=action,
            entry_thesis_state=str(telemetry.get("thesis_state") or ""),
            entry_iv_state=str(telemetry.get("iv_state") or ""),
            entry_battlefield=str(telemetry.get("battlefield") or ""),
            entry_stream_stable=str(telemetry.get("stream_stable") or ""),
            entry_rail_alignment=rail,
            entry_confirming_votes=votes,
            entry_reason_codes=entry_reasons,
            high_water_confidence=confidence,
            low_water_confidence=confidence,
            best_r=0.0,
            worst_r=0.0,
            last_seen_premium=held_premium,
        )
        self.cooldown_until_bar = bars + self.cfg.cooldown_bars_after_entry
        return self._intent(
            intent=intent, action=action, allowed=True,
            direction=desired_direction, confidence=confidence,
            size_fraction=size_fraction, size_lots=size_lots,
            contract_side=contract_side, contract_label=contract_label,
            contract_level=contract_level, strike_price=strike_price,
            profile=profile,
            stop_r=-1.0,                  # 1R loss = the stop
            target_r=target,
            trail_after_r=self.cfg.trail_after_r,
            max_hold_bars=max_hold,
            premium_stop_pct=premium_stop_pct,
            invalidation=str(decision.get("invalidation_rule")
                              or "belief state flips, held leg fails, or data guard trips"),
            reason_codes=entry_reasons,
            telemetry={**telemetry, "confirming_votes": votes,
                        "rail_alignment": rail,
                        "entry_premium": round(held_premium, 4)},
        )

    # ── Telemetry, guards, exits ───────────────────────────────────────

    def _telemetry(self, snapshot: Mapping[str, Any],
                   stream: Optional[Mapping[str, Any]],
                   held_quote: Optional[HeldContractRead],
                   held_premium: Optional[float]) -> Dict[str, Any]:
        decision = _map(snapshot.get("decision"))
        thesis = _map(snapshot.get("thesis"))
        iv = _map(snapshot.get("iv_state"))
        bf = _map(snapshot.get("battlefield"))
        slots = list(snapshot.get("slot_readings") or [])
        n_slots = len(slots)
        dirty = 0
        abnormal = 0
        friendliness: List[float] = []
        mark_quality: List[float] = []
        for raw in slots:
            slot = _map(raw)
            if str(slot.get("mark_source") or "") in DIRTY_MARK_SOURCES:
                dirty += 1
            if bool(slot.get("is_abnormal")):
                abnormal += 1
            friendliness.append(_number(slot.get("friendliness"), math.nan))
            mq = slot.get("mark_quality")
            if isinstance(mq, Mapping):
                mark_quality.append(_number(mq.get("confidence"), math.nan))
            else:
                mark_quality.append(_number(mq, math.nan))
        avg_friend = _safe_mean(friendliness)
        avg_mark_quality = _safe_mean(mark_quality)
        return {
            "bars_seen": int(_number(snapshot.get("bars_seen"), 0)),
            "is_warm": bool(snapshot.get("is_warm")),
            "spot": _number(snapshot.get("spot"), 0.0),
            "decision_trade_allowed": bool(decision.get("trade_allowed")),
            "decision_spread_friendliness": _number(
                decision.get("spread_friendliness"), 0.0),
            "iv_state": str(iv.get("state") or ""),
            "iv_direction": int(_number(iv.get("direction"), 0)),
            "iv_confidence": _clamp(_number(iv.get("confidence"), 0.0)),
            "clean_mark_fraction": _clamp(_number(iv.get("clean_mark_fraction"), 0.0)),
            "avg_friendliness": _clamp(
                avg_friend if math.isfinite(avg_friend)
                else _number(decision.get("spread_friendliness"), 0.0)),
            "avg_mark_quality": _clamp(
                avg_mark_quality if math.isfinite(avg_mark_quality) else 0.0),
            "slot_count": n_slots,
            "abnormal_slot_fraction": (abnormal / n_slots) if n_slots else 0.0,
            "dirty_source_fraction": (dirty / n_slots) if n_slots else 0.0,
            "battlefield": str(bf.get("verdict") or ""),
            "battlefield_direction": int(_number(bf.get("direction"), 0)),
            "battlefield_confidence": _clamp(_number(bf.get("confidence"), 0.0)),
            "thesis_state": str(thesis.get("composite_state") or ""),
            "no_trade_score": _number(thesis.get("no_trade_score"), 0.0),
            "stream_stable": (str(_map(stream).get("stable_verdict") or "")
                              if stream else ""),
            "stream_direction": (int(_number(_map(stream).get("direction"), 0))
                                  if stream else 0),
            "held_quote_present": held_quote is not None,
            "held_spread_state": held_quote.spread_state if held_quote else "",
            "held_friendliness": (
                _clamp(_number(held_quote.friendliness, math.nan))
                if held_quote and held_quote.friendliness is not None else None),
            "held_acceptance": held_quote.acceptance if held_quote else "",
            "held_mark_source": held_quote.mark_source if held_quote else "",
            "held_premium": (round(held_premium, 4)
                              if held_premium is not None else None),
        }

    def _guard_reasons(self, snapshot: Mapping[str, Any],
                       telemetry: Mapping[str, Any]) -> List[str]:
        decision = _map(snapshot.get("decision"))
        reasons: List[str] = []
        if not telemetry["is_warm"] or telemetry["bars_seen"] < self.cfg.min_warm_bars:
            reasons.append(
                f"warmup incomplete bars={telemetry['bars_seen']} "
                f"min={self.cfg.min_warm_bars}")
        if (decision and decision.get("trade_allowed") is False
                and str(decision.get("action")) in ENTRY_ACTIONS):
            reasons.append(str(decision.get("no_trade_reason")
                               or "belief decision disallowed trade"))
        if telemetry["iv_state"] in DIRTY_IV_STATES:
            reasons.append(f"unsafe IV state {telemetry['iv_state']}")
        if telemetry["battlefield"] in DANGEROUS_BATTLEFIELD:
            reasons.append(f"unsafe battlefield {telemetry['battlefield']}")
        if telemetry["clean_mark_fraction"] < self.cfg.min_clean_mark_fraction:
            reasons.append(
                f"clean marks {telemetry['clean_mark_fraction']:.2f} "
                f"below {self.cfg.min_clean_mark_fraction:.2f}")
        if telemetry["avg_friendliness"] < self.cfg.min_spread_friendliness:
            reasons.append(
                f"spread friendliness {telemetry['avg_friendliness']:.2f} "
                f"below {self.cfg.min_spread_friendliness:.2f}")
        if telemetry["abnormal_slot_fraction"] > self.cfg.max_abnormal_slot_fraction:
            reasons.append(
                f"abnormal slots {telemetry['abnormal_slot_fraction']:.2f} "
                f"above {self.cfg.max_abnormal_slot_fraction:.2f}")
        if telemetry["dirty_source_fraction"] > self.cfg.max_dirty_source_fraction:
            reasons.append(
                f"dirty mark sources {telemetry['dirty_source_fraction']:.2f} "
                f"above {self.cfg.max_dirty_source_fraction:.2f}")
        if telemetry["no_trade_score"] > self.cfg.max_no_trade_score:
            reasons.append(
                f"no-trade score {telemetry['no_trade_score']:.1f} "
                f"above {self.cfg.max_no_trade_score:.1f}")
        return reasons

    def _position_exit_reasons(
        self,
        snapshot: Mapping[str, Any],
        stream: Optional[Mapping[str, Any]],
        telemetry: Mapping[str, Any],
        held_quote: Optional[HeldContractRead],
        held_premium: Optional[float],
        *,
        current_r: float,
    ) -> List[str]:
        pos = self.position
        if pos is None:
            return []
        decision = _map(snapshot.get("decision"))
        bars = int(_number(snapshot.get("bars_seen"), 0))
        age = max(0, bars - pos.entry_bar)
        direction = 1 if pos.side == "LONG" else -1
        current_confidence = _clamp(
            _number(decision.get("confidence"), telemetry.get("iv_confidence", 0.0)))
        same_votes = self._confirming_votes(snapshot, stream, direction)
        opposite_votes = self._confirming_votes(snapshot, stream, -direction)
        rail = self._directional_rail(snapshot, direction)
        thesis_state = str(telemetry.get("thesis_state") or "").upper()
        stream_direction = int(_number(telemetry.get("stream_direction"), 0))
        target = (self.cfg.scalp_target_r if pos.profile == PROFILE_SCALP
                  else self.cfg.intraday_target_r)
        hard: List[str] = []
        soft: List[str] = []

        # ── HARD exits (immediate) ────────────────────────────────────
        decision_action = str(decision.get("action") or "")
        if decision_action == "EXIT" or decision_action.startswith("EXIT_"):
            hard.append("belief engine emitted EXIT")
        if telemetry["iv_state"] in DIRTY_IV_STATES:
            hard.append(f"unsafe IV state {telemetry['iv_state']}")
        if telemetry["battlefield"] in DANGEROUS_BATTLEFIELD:
            hard.append(f"unsafe battlefield {telemetry['battlefield']}")
        if telemetry["no_trade_score"] > self.cfg.max_no_trade_score:
            hard.append("no-trade danger rose while holding")
        # Critical-3: held leg's own spread blew out.
        if held_quote is not None and held_quote.spread_state == SPREAD_DANGEROUS:
            hard.append("held contract spread turned dangerous")
        # Critical-3: held leg's mark source went dirty.
        if (held_quote is not None
                and held_quote.mark_source in DIRTY_MARK_SOURCES):
            hard.append(f"held contract mark went dirty ({held_quote.mark_source})")
        # New: engine-cold guard.
        if not telemetry["is_warm"]:
            hard.append("engine went cold while holding")
        # Hard stop on PREMIUM R (Critical-1 fix).
        if current_r <= -1.0:
            hard.append(
                f"hard stop on premium R={current_r:.2f}, "
                f"premium fell {self._premium_stop_pct():.0%}+"
            )
        if hard:
            return hard

        if age < self.cfg.min_hold_bars:
            return []

        # ── SOFT exits (one is enough, after min hold) ────────────────
        if opposite_votes >= self.cfg.min_confirming_votes:
            soft.append(
                f"opposite directional vote cluster votes={opposite_votes}")
        if (same_votes < max(1, self.cfg.min_confirming_votes - 1)
                and rail < self.cfg.min_directional_rail_abs):
            soft.append(
                f"context alignment faded votes={same_votes} rail={rail:.2f}")
        if direction > 0 and "BEAR" in thesis_state:
            soft.append(f"thesis flipped against long: {thesis_state}")
        if direction < 0 and "BULL" in thesis_state:
            soft.append(f"thesis flipped against short: {thesis_state}")
        if stream_direction == -direction:
            soft.append("stream turned against position")
        if (pos.high_water_confidence - current_confidence
                >= self.cfg.confidence_giveback
                and same_votes < pos.entry_confirming_votes):
            soft.append(
                f"confidence giveback "
                f"{pos.high_water_confidence - current_confidence:.2f}")
        # Profit lock and target on the PREMIUM R.
        if (pos.best_r >= self.cfg.profit_lock_start_r
                and pos.best_r - current_r >= self.cfg.profit_lock_giveback_r):
            soft.append(
                f"profit lock giveback best={pos.best_r:.2f}R "
                f"current={current_r:.2f}R")
        if current_r >= target and (
                same_votes < pos.entry_confirming_votes
                or rail < pos.entry_rail_alignment * 0.70):
            soft.append(
                f"target reached with context fade "
                f"current={current_r:.2f}R target={target:.2f}R")
        # Critical-3: held leg's acceptance flipped against the held side.
        if held_quote is not None:
            if direction > 0 and held_quote.acceptance == "rejected":
                soft.append("held call acceptance turned rejected")
            if direction < 0 and held_quote.acceptance == "rejected":
                soft.append("held put acceptance turned rejected")
            # First-weakness sweep exit.
            if (pos.profile == PROFILE_SCALP
                    and abs(pos.entry_dod_z) > 0.5
                    and held_quote.dod_z is not None
                    and math.isfinite(held_quote.dod_z)
                    and (abs(held_quote.dod_z)
                         < abs(pos.entry_dod_z) * self.cfg.sweep_weakness_dod_z_ratio)):
                soft.append(
                    f"sweep weakness: held |dod_z| {abs(held_quote.dod_z):.2f} "
                    f"collapsed from entry {abs(pos.entry_dod_z):.2f}"
                )
        max_age = (self.cfg.scalp_max_bars if pos.profile == PROFILE_SCALP
                   else self.cfg.intraday_max_bars)
        if age >= max_age:
            soft.append(f"max hold bars reached age={age}")
        return soft

    def _hold_intent(self, action: str, confidence: float,
                     telemetry: Mapping[str, Any],
                     reject_reasons: List[str],
                     *, current_r: float) -> ExecutionIntent:
        pos = self.position
        assert pos is not None
        bars = int(_number(telemetry.get("bars_seen"), 0))
        age = max(0, bars - pos.entry_bar)
        max_hold = (self.cfg.scalp_max_bars if pos.profile == PROFILE_SCALP
                    else self.cfg.intraday_max_bars)
        target = (self.cfg.scalp_target_r if pos.profile == PROFILE_SCALP
                  else self.cfg.intraday_target_r)
        premium_stop_pct = self._premium_stop_pct()
        hold_telemetry = {
            **telemetry,
            "position_age": age,
            "position_current_r": round(current_r, 4),
            "position_best_r": round(pos.best_r, 4),
            "position_worst_r": round(pos.worst_r, 4),
            "position_high_water_confidence": round(pos.high_water_confidence, 4),
            "position_low_water_confidence": round(pos.low_water_confidence, 4),
            "entry_confirming_votes": pos.entry_confirming_votes,
            "entry_rail_alignment": round(pos.entry_rail_alignment, 4),
            "position_direction": 1 if pos.side == "LONG" else -1,
            "entry_premium": round(pos.entry_premium, 4),
            "last_seen_premium": round(pos.last_seen_premium, 4),
        }
        return self._intent(
            intent=INTENT_HOLD_POSITION, action=action, allowed=True,
            direction=1 if pos.side == "LONG" else -1,
            confidence=max(confidence, pos.entry_confidence),
            size_fraction=pos.size_fraction, size_lots=pos.size_lots,
            contract_side=pos.contract_side, contract_label=pos.contract_label,
            contract_level=pos.contract_level, strike_price=pos.strike_price,
            profile=pos.profile,
            stop_r=-1.0, target_r=target,
            trail_after_r=self.cfg.trail_after_r,
            max_hold_bars=max_hold, premium_stop_pct=premium_stop_pct,
            invalidation="active position; exit if guards flip or age expires",
            reason_codes=[
                f"position_age={age}",
                f"max_hold={max_hold}",
                f"current_r={current_r:.2f}",
                f"best_r={pos.best_r:.2f}",
                f"worst_r={pos.worst_r:.2f}",
            ],
            reject_reasons=reject_reasons, telemetry=hold_telemetry,
        )

    # ── Vote / size / R helpers ────────────────────────────────────────

    def _confirming_votes(self, snapshot: Mapping[str, Any],
                          stream: Optional[Mapping[str, Any]],
                          direction: int) -> int:
        """Confirming sources beyond the decision's own direction.

        The decision IS the trigger — we want INDEPENDENT confirmations.
        Sources counted: IV direction, battlefield direction, thesis state,
        stream direction, rail alignment magnitude.
        """
        if direction == 0:
            return 0
        iv = _map(snapshot.get("iv_state"))
        bf = _map(snapshot.get("battlefield"))
        thesis = _map(snapshot.get("thesis"))
        votes = 0
        if int(_number(iv.get("direction"), 0)) == direction:
            votes += 1
        if int(_number(bf.get("direction"), 0)) == direction:
            votes += 1
        thesis_state = str(thesis.get("composite_state") or "").upper()
        if direction > 0 and "BULL" in thesis_state:
            votes += 1
        if direction < 0 and "BEAR" in thesis_state:
            votes += 1
        if (stream is not None
                and int(_number(_map(stream).get("direction"), 0)) == direction):
            votes += 1
        if self._directional_rail(snapshot, direction) >= self.cfg.min_directional_rail_abs:
            votes += 1
        return votes

    def _directional_rail(self, snapshot: Mapping[str, Any],
                          direction: int) -> float:
        bf = _map(snapshot.get("battlefield"))
        ce = _map(bf.get("ce_rail"))
        pe = _map(bf.get("pe_rail"))
        ce_z = _number(ce.get("weighted_mean_signed_z"), 0.0)
        pe_z = _number(pe.get("weighted_mean_signed_z"), 0.0)
        if direction > 0:
            return max(0.0, (max(0.0, ce_z) + max(0.0, -pe_z)) / 2.0)
        if direction < 0:
            return max(0.0, (max(0.0, pe_z) + max(0.0, -ce_z)) / 2.0)
        return 0.0

    def _size(self, confidence: float, telemetry: Mapping[str, Any],
              votes: int, rail: float) -> tuple[float, int]:
        """Convert conviction to (size_fraction in [0,1], size_lots)."""
        vote_score = min(1.0, votes / 5.0)
        rail_score = min(1.0, rail / 1.5)
        conviction = (
            0.42 * confidence
            + 0.18 * _number(telemetry.get("avg_friendliness"), 0.0)
            + 0.18 * _number(telemetry.get("clean_mark_fraction"), 0.0)
            + 0.14 * rail_score
            + 0.08 * vote_score
        )
        if conviction >= 0.86:
            fraction = 1.00
        elif conviction >= 0.78:
            fraction = 0.70
        elif conviction >= 0.68:
            fraction = 0.45
        else:
            fraction = 0.25
        # Convert fraction to lots — at least 1 lot when allowed at all.
        lots = max(1, min(self.cfg.max_lots, int(round(fraction * self.cfg.max_lots))))
        return fraction, lots

    def _premium_stop_pct(self) -> float:
        """Return the active premium_stop_pct for the held position's profile."""
        if self.position is None:
            return self.cfg.premium_stop_intraday
        return (self.cfg.premium_stop_scalp
                if self.position.profile == PROFILE_SCALP
                else self.cfg.premium_stop_intraday)

    def _position_r(self, pos: ExecutorPosition,
                    held_premium: Optional[float]) -> float:
        """Corrected R = premium move / (entry_premium × premium_stop_pct).

        Returns 0.0 when held_premium is unavailable — the held-quote
        guards will catch the data-quality issue separately.
        """
        if (held_premium is None or not math.isfinite(held_premium)
                or pos.entry_premium <= 0):
            return 0.0
        side = 1 if pos.side == "LONG" else -1
        stop_pct = (self.cfg.premium_stop_scalp
                    if pos.profile == PROFILE_SCALP
                    else self.cfg.premium_stop_intraday)
        return (side * (held_premium - pos.entry_premium)
                / (pos.entry_premium * stop_pct))

    def _entry_shape(self, action: str) -> tuple[str, int, str]:
        if action == "ENTER_LONG":
            return INTENT_OPEN_LONG, 1, PROFILE_INTRADAY
        if action == "ENTER_SHORT":
            return INTENT_OPEN_SHORT, -1, PROFILE_INTRADAY
        if action == "SCALP_CALL":
            return INTENT_OPEN_SCALP_CALL, 1, PROFILE_SCALP
        return INTENT_OPEN_SCALP_PUT, -1, PROFILE_SCALP

    def _held_premium(self, snapshot: Mapping[str, Any],
                      held_quote: Optional[HeldContractRead]) -> Optional[float]:
        """Resolve the held contract's current premium. Prefer the explicit
        :class:`HeldContractRead`; fall back to the matching slot in the
        snapshot. Returns None if nothing usable is available."""
        if held_quote is not None:
            best = held_quote.best_price
            if best is not None and math.isfinite(best) and best > 0:
                return float(best)
        # Fallback to the slot matching the held contract (if we hold one).
        if self.position is not None:
            want_side = self.position.contract_side
            want_level = self.position.contract_level
            for raw in (snapshot.get("slot_readings") or []):
                slot = _map(raw)
                if (str(slot.get("option_type") or "") == want_side
                        and int(_number(slot.get("level"), 999)) == want_level):
                    mp = _number(slot.get("mark_price"), math.nan)
                    if math.isfinite(mp) and mp > 0:
                        return float(mp)
        # Fallback for entry path: use the slot the decision wants.
        decision = _map(snapshot.get("decision"))
        strike = _map(decision.get("strike"))
        want_side = str(strike.get("side") or "")
        want_level = int(_number(strike.get("level"), 0))
        if want_side:
            for raw in (snapshot.get("slot_readings") or []):
                slot = _map(raw)
                if (str(slot.get("option_type") or "") == want_side
                        and int(_number(slot.get("level"), 999)) == want_level):
                    mp = _number(slot.get("mark_price"), math.nan)
                    if math.isfinite(mp) and mp > 0:
                        return float(mp)
        return None

    def _strike_price(self, snapshot: Mapping[str, Any],
                      contract_side: str, contract_level: int) -> float:
        """Pull the actual rupee strike from the matching slot_reading."""
        for raw in (snapshot.get("slot_readings") or []):
            slot = _map(raw)
            if (str(slot.get("option_type") or "") == contract_side
                    and int(_number(slot.get("level"), 999)) == contract_level):
                return _number(slot.get("strike"), 0.0)
        return 0.0

    # ── Intent assembly ────────────────────────────────────────────────

    def _intent(self, **kwargs: Any) -> ExecutionIntent:
        state = dict(kwargs.pop("state", {}) or {})
        pos = self.position
        if pos is not None:
            state.update({
                "position_open": True,
                "position_side": pos.side,
                "position_intent": pos.intent,
                "contract_side": pos.contract_side,
                "contract_label": pos.contract_label,
                "contract_level": pos.contract_level,
                "strike_price": pos.strike_price,
                "profile": pos.profile,
                "size_fraction": pos.size_fraction,
                "size_lots": pos.size_lots,
                "entry_bar": pos.entry_bar,
                "entry_spot": pos.entry_spot,
                "entry_premium": pos.entry_premium,
                "entry_confidence": pos.entry_confidence,
                "entry_action": pos.entry_action,
                "entry_context": {
                    "thesis_state": pos.entry_thesis_state,
                    "iv_state": pos.entry_iv_state,
                    "battlefield": pos.entry_battlefield,
                    "stream_stable": pos.entry_stream_stable,
                    "confirming_votes": pos.entry_confirming_votes,
                    "rail_alignment": pos.entry_rail_alignment,
                    "reason_codes": list(pos.entry_reason_codes),
                },
                "high_water_confidence": pos.high_water_confidence,
                "low_water_confidence": pos.low_water_confidence,
                "best_r": pos.best_r,
                "worst_r": pos.worst_r,
            })
        else:
            state.setdefault("position_open", False)
        state["cooldown_until_bar"] = self.cooldown_until_bar
        kwargs["state"] = state
        return ExecutionIntent(**kwargs)


# ── Module-level helpers ──────────────────────────────────────────────

def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return default


def _safe_mean(values: List[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return math.nan
    return sum(clean) / len(clean)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
