"""Execution Governor for the Premium Belief Engine.

This module is intentionally broker-agnostic and shadow-only. It turns one
Premium Belief snapshot into an execution intent with position lifecycle,
guardrails, sizing, invalidation, and exit guidance. A later broker adapter can
consume the same intent, but this layer never places orders by itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


EXECUTOR_NAME = "premium_belief_execution_governor"
EXECUTOR_VERSION = "v1"
ORDER_MODE_SHADOW = "SHADOW_ONLY"

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

DIRTY_MARK_SOURCES = {"invalid", "last_valid", "ltp"}
ENTRY_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "SCALP_CALL", "SCALP_PUT"}
DIRTY_IV_STATES = {"dirty_data", "liquidity_distortion", "common_shock"}
DANGEROUS_BATTLEFIELD = {"single_distortion", "vol_expansion"}


@dataclass(frozen=True)
class ExecutionGovernorConfig:
    """Knobs for translating belief reads into managed shadow intents."""

    min_warm_bars: int = 80
    min_entry_confidence: float = 0.66
    min_scalp_confidence: float = 0.72
    min_spread_friendliness: float = 0.65
    min_clean_mark_fraction: float = 0.85
    max_abnormal_slot_fraction: float = 0.40
    max_dirty_source_fraction: float = 0.10
    min_directional_votes: int = 2
    min_directional_rail_abs: float = 0.30
    max_no_trade_score: float = 18.0
    cooldown_bars_after_entry: int = 8
    cooldown_bars_after_exit: int = 6
    scalp_max_bars: int = 24
    intraday_max_bars: int = 90
    stop_r: float = 1.00
    scalp_target_r: float = 0.70
    intraday_target_r: float = 1.25
    trail_after_r: float = 0.55
    max_unit_fraction: float = 1.00
    adverse_spot_stop_pct: float = 0.0025


@dataclass
class ExecutorPosition:
    """One shadow position tracked by the governor."""

    side: str
    intent: str
    contract_side: str
    contract_label: str
    contract_level: int
    entry_spot: float
    entry_bar: int
    entry_confidence: float
    profile: str


@dataclass(frozen=True)
class ExecutionIntent:
    """Serializable execution intent emitted per belief row."""

    intent: str
    action: str
    allowed: bool
    direction: int
    confidence: float
    size_fraction: float
    contract_side: str = ""
    contract_label: str = ""
    contract_level: int = 0
    profile: str = ""
    stop_r: float = 0.0
    target_r: float = 0.0
    trail_after_r: float = 0.0
    max_hold_bars: int = 0
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
            "contract_side": self.contract_side,
            "contract_label": self.contract_label,
            "contract_level": self.contract_level,
            "profile": self.profile,
            "stop_r": self.stop_r,
            "target_r": self.target_r,
            "trail_after_r": self.trail_after_r,
            "max_hold_bars": self.max_hold_bars,
            "invalidation": self.invalidation,
            "reason_codes": list(self.reason_codes),
            "reject_reasons": list(self.reject_reasons),
            "state": dict(self.state),
            "telemetry": dict(self.telemetry),
        }


class BeliefExecutionGovernor:
    """Stateful shadow executor above the eight-phase belief engine."""

    def __init__(self, cfg: Optional[ExecutionGovernorConfig] = None) -> None:
        self.cfg = cfg or ExecutionGovernorConfig()
        self.position: Optional[ExecutorPosition] = None
        self.cooldown_until_bar = -1

    def evaluate(
        self,
        snapshot: Mapping[str, Any],
        stream: Optional[Mapping[str, Any]] = None,
    ) -> ExecutionIntent:
        """Evaluate one belief snapshot and return the next shadow intent."""

        decision = _map(snapshot.get("decision"))
        action = str(decision.get("action") or "WAIT")
        direction = int(_number(decision.get("direction"), 0))
        confidence = _clamp(_number(decision.get("confidence"), 0.0))
        bars = int(_number(snapshot.get("bars_seen"), 0))
        spot = _number(snapshot.get("spot"), 0.0)
        telemetry = self._telemetry(snapshot, stream)
        reject_reasons = self._guard_reasons(snapshot, telemetry)

        if self.position is not None:
            exit_reasons = self._position_exit_reasons(snapshot, stream, telemetry)
            if exit_reasons:
                pos = self.position
                self.position = None
                self.cooldown_until_bar = max(self.cooldown_until_bar, bars + self.cfg.cooldown_bars_after_exit)
                return self._intent(
                    intent=INTENT_EXIT_POSITION,
                    action=action,
                    allowed=True,
                    direction=0,
                    confidence=max(confidence, pos.entry_confidence),
                    size_fraction=0.0,
                    contract_side=pos.contract_side,
                    contract_label=pos.contract_label,
                    contract_level=pos.contract_level,
                    profile=pos.profile,
                    invalidation="; ".join(exit_reasons[:4]),
                    reason_codes=exit_reasons,
                    reject_reasons=reject_reasons,
                    telemetry=telemetry,
                )
            return self._hold_intent(action, confidence, telemetry, reject_reasons)

        if reject_reasons:
            return self._intent(
                intent=INTENT_BLOCKED,
                action=action,
                allowed=False,
                direction=0,
                confidence=confidence,
                size_fraction=0.0,
                reject_reasons=reject_reasons,
                telemetry=telemetry,
            )

        if bars < self.cooldown_until_bar:
            return self._intent(
                intent=INTENT_FLAT_WAIT,
                action=action,
                allowed=False,
                direction=0,
                confidence=confidence,
                size_fraction=0.0,
                reject_reasons=[f"cooldown until bar {self.cooldown_until_bar}"],
                telemetry=telemetry,
            )

        if action not in ENTRY_ACTIONS:
            return self._intent(
                intent=INTENT_FLAT_WAIT,
                action=action,
                allowed=False,
                direction=0,
                confidence=confidence,
                size_fraction=0.0,
                reason_codes=["no entry action from belief engine"],
                telemetry=telemetry,
            )

        intent, desired_direction, profile = self._entry_shape(action)
        threshold = self.cfg.min_scalp_confidence if profile == PROFILE_SCALP else self.cfg.min_entry_confidence
        if confidence < threshold:
            return self._intent(
                intent=INTENT_BLOCKED,
                action=action,
                allowed=False,
                direction=0,
                confidence=confidence,
                size_fraction=0.0,
                reject_reasons=[f"confidence {confidence:.2f} below {threshold:.2f}"],
                telemetry=telemetry,
            )

        votes = self._directional_votes(snapshot, stream, desired_direction)
        rail = self._directional_rail(snapshot, desired_direction)
        if votes < self.cfg.min_directional_votes:
            return self._intent(
                intent=INTENT_BLOCKED,
                action=action,
                allowed=False,
                direction=0,
                confidence=confidence,
                size_fraction=0.0,
                reject_reasons=[f"directional votes {votes} below {self.cfg.min_directional_votes}"],
                telemetry={**telemetry, "directional_votes": votes, "rail_alignment": rail},
            )
        if rail < self.cfg.min_directional_rail_abs:
            return self._intent(
                intent=INTENT_BLOCKED,
                action=action,
                allowed=False,
                direction=0,
                confidence=confidence,
                size_fraction=0.0,
                reject_reasons=[f"rail alignment {rail:.2f} below {self.cfg.min_directional_rail_abs:.2f}"],
                telemetry={**telemetry, "directional_votes": votes, "rail_alignment": rail},
            )

        size_fraction = self._size_fraction(confidence, telemetry, votes, rail)
        strike = _map(decision.get("strike"))
        contract_side = str(strike.get("side") or ("CE" if desired_direction > 0 else "PE"))
        contract_label = str(strike.get("label") or f"{contract_side}_ATM")
        contract_level = int(_number(strike.get("level"), 0))
        target = self.cfg.scalp_target_r if profile == PROFILE_SCALP else self.cfg.intraday_target_r
        max_hold = self.cfg.scalp_max_bars if profile == PROFILE_SCALP else self.cfg.intraday_max_bars
        self.position = ExecutorPosition(
            side="LONG" if desired_direction > 0 else "SHORT",
            intent=intent,
            contract_side=contract_side,
            contract_label=contract_label,
            contract_level=contract_level,
            entry_spot=spot,
            entry_bar=bars,
            entry_confidence=confidence,
            profile=profile,
        )
        self.cooldown_until_bar = bars + self.cfg.cooldown_bars_after_entry
        return self._intent(
            intent=intent,
            action=action,
            allowed=True,
            direction=desired_direction,
            confidence=confidence,
            size_fraction=size_fraction,
            contract_side=contract_side,
            contract_label=contract_label,
            contract_level=contract_level,
            profile=profile,
            stop_r=self.cfg.stop_r,
            target_r=target,
            trail_after_r=self.cfg.trail_after_r,
            max_hold_bars=max_hold,
            invalidation=str(decision.get("invalidation_rule") or "belief state flips or data guard trips"),
            reason_codes=[
                f"votes={votes}",
                f"rail={rail:.2f}",
                f"size={size_fraction:.2f}",
                f"profile={profile}",
            ],
            telemetry={**telemetry, "directional_votes": votes, "rail_alignment": rail},
        )

    def _telemetry(self, snapshot: Mapping[str, Any], stream: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        decision = _map(snapshot.get("decision"))
        thesis = _map(snapshot.get("thesis"))
        iv = _map(snapshot.get("iv_state"))
        field = _map(snapshot.get("battlefield"))
        slots = list(snapshot.get("slot_readings") or [])
        n_slots = len(slots)
        dirty = 0
        abnormal = 0
        friendliness = []
        mark_quality = []
        for raw in slots:
            slot = _map(raw)
            if str(slot.get("mark_source") or "") in DIRTY_MARK_SOURCES:
                dirty += 1
            if bool(slot.get("is_abnormal")):
                abnormal += 1
            friendliness.append(_number(slot.get("friendliness"), math.nan))
            mq = _map(slot.get("mark_quality"))
            mark_quality.append(_number(mq.get("confidence"), math.nan))
        avg_friend = _safe_mean(friendliness)
        avg_mark_quality = _safe_mean(mark_quality)
        return {
            "bars_seen": int(_number(snapshot.get("bars_seen"), 0)),
            "is_warm": bool(snapshot.get("is_warm")),
            "spot": _number(snapshot.get("spot"), 0.0),
            "decision_trade_allowed": bool(decision.get("trade_allowed")),
            "decision_spread_friendliness": _number(decision.get("spread_friendliness"), 0.0),
            "iv_state": str(iv.get("state") or ""),
            "iv_direction": int(_number(iv.get("direction"), 0)),
            "iv_confidence": _clamp(_number(iv.get("confidence"), 0.0)),
            "clean_mark_fraction": _clamp(_number(iv.get("clean_mark_fraction"), 0.0)),
            "avg_friendliness": _clamp(avg_friend if math.isfinite(avg_friend) else _number(decision.get("spread_friendliness"), 0.0)),
            "avg_mark_quality": _clamp(avg_mark_quality if math.isfinite(avg_mark_quality) else 0.0),
            "slot_count": n_slots,
            "abnormal_slot_fraction": (abnormal / n_slots) if n_slots else 0.0,
            "dirty_source_fraction": (dirty / n_slots) if n_slots else 0.0,
            "battlefield": str(field.get("verdict") or ""),
            "battlefield_direction": int(_number(field.get("direction"), 0)),
            "battlefield_confidence": _clamp(_number(field.get("confidence"), 0.0)),
            "thesis_state": str(thesis.get("composite_state") or ""),
            "no_trade_score": _number(thesis.get("no_trade_score"), 0.0),
            "stream_stable": str(_map(stream).get("stable_verdict") or "") if stream else "",
            "stream_direction": int(_number(_map(stream).get("direction"), 0)) if stream else 0,
            "cost_wall_pass": _number(decision.get("spread_friendliness"), 0.0) >= self.cfg.min_spread_friendliness,
            "same_bar_policy": "wait_one_tick_after_open",
        }

    def _guard_reasons(self, snapshot: Mapping[str, Any], telemetry: Mapping[str, Any]) -> List[str]:
        decision = _map(snapshot.get("decision"))
        reasons: List[str] = []
        if not telemetry["is_warm"] or telemetry["bars_seen"] < self.cfg.min_warm_bars:
            reasons.append(f"warmup incomplete bars={telemetry['bars_seen']} min={self.cfg.min_warm_bars}")
        if decision and decision.get("trade_allowed") is False and str(decision.get("action")) in ENTRY_ACTIONS:
            reasons.append(str(decision.get("no_trade_reason") or "belief decision disallowed trade"))
        if telemetry["iv_state"] in DIRTY_IV_STATES:
            reasons.append(f"unsafe IV state {telemetry['iv_state']}")
        if telemetry["battlefield"] in DANGEROUS_BATTLEFIELD:
            reasons.append(f"unsafe battlefield {telemetry['battlefield']}")
        if telemetry["clean_mark_fraction"] < self.cfg.min_clean_mark_fraction:
            reasons.append(f"clean marks {telemetry['clean_mark_fraction']:.2f} below {self.cfg.min_clean_mark_fraction:.2f}")
        if telemetry["avg_friendliness"] < self.cfg.min_spread_friendliness:
            reasons.append(f"spread friendliness {telemetry['avg_friendliness']:.2f} below {self.cfg.min_spread_friendliness:.2f}")
        if telemetry["abnormal_slot_fraction"] > self.cfg.max_abnormal_slot_fraction:
            reasons.append(f"abnormal slots {telemetry['abnormal_slot_fraction']:.2f} above {self.cfg.max_abnormal_slot_fraction:.2f}")
        if telemetry["dirty_source_fraction"] > self.cfg.max_dirty_source_fraction:
            reasons.append(f"dirty mark sources {telemetry['dirty_source_fraction']:.2f} above {self.cfg.max_dirty_source_fraction:.2f}")
        if telemetry["no_trade_score"] > self.cfg.max_no_trade_score:
            reasons.append(f"no-trade score {telemetry['no_trade_score']:.1f} above {self.cfg.max_no_trade_score:.1f}")
        return reasons

    def _position_exit_reasons(
        self,
        snapshot: Mapping[str, Any],
        stream: Optional[Mapping[str, Any]],
        telemetry: Mapping[str, Any],
    ) -> List[str]:
        pos = self.position
        if pos is None:
            return []
        decision = _map(snapshot.get("decision"))
        bars = int(_number(snapshot.get("bars_seen"), 0))
        spot = _number(snapshot.get("spot"), pos.entry_spot)
        age = max(0, bars - pos.entry_bar)
        direction = 1 if pos.side == "LONG" else -1
        spot_move = direction * ((spot - pos.entry_spot) / max(1e-9, pos.entry_spot))
        reasons: List[str] = []
        if str(decision.get("action") or "") == "EXIT":
            reasons.append("belief engine emitted EXIT")
        if telemetry["iv_state"] in DIRTY_IV_STATES:
            reasons.append(f"unsafe IV state {telemetry['iv_state']}")
        if telemetry["battlefield"] in DANGEROUS_BATTLEFIELD:
            reasons.append(f"unsafe battlefield {telemetry['battlefield']}")
        if telemetry["no_trade_score"] > self.cfg.max_no_trade_score:
            reasons.append("no-trade danger rose while holding")
        if self._directional_votes(snapshot, stream, -direction) >= 2:
            reasons.append("opposite directional vote cluster")
        max_age = self.cfg.scalp_max_bars if pos.profile == PROFILE_SCALP else self.cfg.intraday_max_bars
        if age >= max_age:
            reasons.append(f"max hold bars reached age={age}")
        if spot_move <= -self.cfg.adverse_spot_stop_pct:
            reasons.append(f"adverse spot move {spot_move:.4f}")
        return reasons

    def _hold_intent(
        self,
        action: str,
        confidence: float,
        telemetry: Mapping[str, Any],
        reject_reasons: List[str],
    ) -> ExecutionIntent:
        pos = self.position
        assert pos is not None
        bars = int(_number(telemetry.get("bars_seen"), 0))
        age = max(0, bars - pos.entry_bar)
        max_hold = self.cfg.scalp_max_bars if pos.profile == PROFILE_SCALP else self.cfg.intraday_max_bars
        return self._intent(
            intent=INTENT_HOLD_POSITION,
            action=action,
            allowed=True,
            direction=1 if pos.side == "LONG" else -1,
            confidence=max(confidence, pos.entry_confidence),
            size_fraction=0.0,
            contract_side=pos.contract_side,
            contract_label=pos.contract_label,
            contract_level=pos.contract_level,
            profile=pos.profile,
            stop_r=self.cfg.stop_r,
            target_r=self.cfg.scalp_target_r if pos.profile == PROFILE_SCALP else self.cfg.intraday_target_r,
            trail_after_r=self.cfg.trail_after_r,
            max_hold_bars=max_hold,
            invalidation="active position; exit if guards flip or age expires",
            reason_codes=[f"position_age={age}", f"max_hold={max_hold}"],
            reject_reasons=reject_reasons,
            telemetry=telemetry,
        )

    def _intent(self, **kwargs: Any) -> ExecutionIntent:
        state = dict(kwargs.pop("state", {}) or {})
        pos = self.position
        if pos is not None:
            state.update({
                "position_open": True,
                "position_side": pos.side,
                "entry_bar": pos.entry_bar,
                "entry_spot": pos.entry_spot,
                "entry_confidence": pos.entry_confidence,
            })
        else:
            state.setdefault("position_open", False)
        state["cooldown_until_bar"] = self.cooldown_until_bar
        kwargs["state"] = state
        return ExecutionIntent(**kwargs)

    def _entry_shape(self, action: str) -> tuple[str, int, str]:
        if action == "ENTER_LONG":
            return INTENT_OPEN_LONG, 1, PROFILE_INTRADAY
        if action == "ENTER_SHORT":
            return INTENT_OPEN_SHORT, -1, PROFILE_INTRADAY
        if action == "SCALP_CALL":
            return INTENT_OPEN_SCALP_CALL, 1, PROFILE_SCALP
        return INTENT_OPEN_SCALP_PUT, -1, PROFILE_SCALP

    def _directional_votes(
        self,
        snapshot: Mapping[str, Any],
        stream: Optional[Mapping[str, Any]],
        direction: int,
    ) -> int:
        if direction == 0:
            return 0
        decision = _map(snapshot.get("decision"))
        iv = _map(snapshot.get("iv_state"))
        field = _map(snapshot.get("battlefield"))
        thesis = _map(snapshot.get("thesis"))
        votes = 0
        if int(_number(decision.get("direction"), 0)) == direction:
            votes += 1
        if int(_number(iv.get("direction"), 0)) == direction:
            votes += 1
        if int(_number(field.get("direction"), 0)) == direction:
            votes += 1
        thesis_state = str(thesis.get("composite_state") or "").upper()
        if direction > 0 and "BULL" in thesis_state:
            votes += 1
        if direction < 0 and "BEAR" in thesis_state:
            votes += 1
        if stream is not None and int(_number(_map(stream).get("direction"), 0)) == direction:
            votes += 1
        return votes

    def _directional_rail(self, snapshot: Mapping[str, Any], direction: int) -> float:
        field = _map(snapshot.get("battlefield"))
        ce = _map(field.get("ce_rail"))
        pe = _map(field.get("pe_rail"))
        ce_z = _number(ce.get("weighted_mean_signed_z"), 0.0)
        pe_z = _number(pe.get("weighted_mean_signed_z"), 0.0)
        if direction > 0:
            return max(0.0, (max(0.0, ce_z) + max(0.0, -pe_z)) / 2.0)
        if direction < 0:
            return max(0.0, (max(0.0, pe_z) + max(0.0, -ce_z)) / 2.0)
        return 0.0

    def _size_fraction(
        self,
        confidence: float,
        telemetry: Mapping[str, Any],
        votes: int,
        rail: float,
    ) -> float:
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
            size = 1.00
        elif conviction >= 0.78:
            size = 0.70
        elif conviction >= 0.68:
            size = 0.45
        else:
            size = 0.25
        return min(self.cfg.max_unit_fraction, size)


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
