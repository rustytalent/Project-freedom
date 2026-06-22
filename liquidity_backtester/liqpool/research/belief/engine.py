"""Belief Engine orchestrator (Phase 8 — the live wire).

The whole stack, end to end, in one streaming object. Feed it ticks for
the 22 contracts on every quote update and it returns one
:class:`BeliefSnapshot` per tick — the full state of the engine plus a
:class:`Decision` ready for the operator (or, eventually, the executor).

What it owns:

  * the per-contract :class:`MarkPriceTracker` (Phase 2);
  * a rolling per-contract quote/spot history so Phase-4 modules can
    compute rolling residuals + spread-friendliness without re-reading
    a parquet;
  * the 22-slot moneyness identity (Phase 3) — rebuilt when the ATM
    shifts;
  * the per-contract Phase-4 readings (residual + spread) computed on
    each tick;
  * the Phase-5 battlefield + IV state classifier;
  * the Phase-6 :class:`ThesisMemory` (the only persistent score store);
  * the Phase-7 winding detector + 3 state machines;
  * the Phase-8 decision layer.

The design ethic — the founder's "good data, bad interpretation kills":
the engine refuses to act until enough history has accrued for rolling
stats to be stable. ``warmup_bars`` (default 80) sets that floor; while
warming up, decisions are NO_TRADE("warming up — N more bars needed").
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Deque, Dict, List, Optional

import pandas as pd

from .battlefield import (
    BattlefieldSnapshot,
    SlotReading,
    battlefield_snapshot,
)
from .decision import (
    ACTION_NO_TRADE,
    Decision,
    DecisionConfig,
    StrikeRecommendation,
    decide,
)
from .fair_response import FairResponseConfig
from .iv_state import IVState, IVStateConfig, classify_iv_state, IV_DIRTY_DATA
from .mark_price import MarkPrice, MarkPriceConfig, MarkPriceTracker, Quote
from .moneyness import MoneynessSlot, classify_moneyness
from .residual import ResidualConfig, slot_residual_frame
from .spread import SpreadConfig, spread_friendliness_frame, CLEAN as SPREAD_CLEAN
from .state_machines import (
    BearContinuationMachine,
    BullContinuationMachine,
    LiquiditySweepMachine,
    StateUpdate,
)
from .thesis_memory import (
    NEUTRAL_THESIS,
    ThesisMemory,
    ThesisMemoryConfig,
    ThesisSnapshot,
)
from .winding import WindingConfig, WindingDetector, WindingZone


# Per-contract rolling history cap. The Phase-4 residual + spread modules
# need ~60-bar windows; we keep ~3x that so freshly-rolled stats are stable.
_HISTORY_BARS: int = 240


def _finite_float(value: Any) -> Optional[float]:
    """Return a JSON-safe float, preserving missing/non-finite values as None."""
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _slot_reading_payload(
    reading: SlotReading,
    *,
    mark: Optional[MarkPrice],
    spread_state: str,
) -> Dict[str, Any]:
    """Compact Phase-4 slot row for Sentinel's 22-cell CE/PE heatmap.

    The aggregate battlefield tells us what the rails think. This row keeps
    the individual slot evidence that produced that aggregate verdict, including
    the mark source and quality so the cockpit can distinguish real pressure
    from dirty quote artifacts.
    """
    slot = reading.slot
    return {
        "strike": _finite_float(slot.strike),
        "option_type": slot.option_type,
        "moneyness_label": slot.label,
        "label": slot.label,
        "level": int(slot.level),
        "behavior": slot.behavior,
        "expected_abs_delta": _finite_float(slot.expected_abs_delta),
        "expected_signed_delta": _finite_float(slot.expected_signed_delta),
        "mark_source": mark.source if mark is not None else "unknown",
        "mark_quality": _finite_float(mark.quality) if mark is not None else None,
        "mark_quality_label": mark.quality_label if mark is not None else "unknown",
        "mark_price": _finite_float(mark.price) if mark is not None else None,
        "mark_spread": _finite_float(mark.spread) if mark is not None else None,
        "mark_spread_pct": _finite_float(mark.spread_pct) if mark is not None else None,
        "ltp_confirms": bool(mark.ltp_confirms) if mark is not None else False,
        "mark_flags": list(mark.flags) if mark is not None else [],
        "friendliness": _finite_float(reading.friendliness),
        "spread_state": spread_state,
        "acceptance": reading.acceptance,
        "dod_z": _finite_float(reading.dod_z),
        "is_abnormal": bool(reading.is_abnormal),
    }


@dataclass
class BeliefEngineConfig:
    """Top-level knobs for the engine."""
    strike_step: float = 50.0
    levels: int = 5                       # ATM ± N strikes (5 → 22 contracts)
    warmup_bars: int = 80                  # bars of history before decisions act
    rebuild_atm_threshold_strikes: float = 1.0  # rebuild slots when ATM has shifted this far
    mark_cfg: MarkPriceConfig = field(default_factory=MarkPriceConfig)
    fair_cfg: FairResponseConfig = field(default_factory=FairResponseConfig)
    resid_cfg: ResidualConfig = field(default_factory=ResidualConfig)
    spread_cfg: SpreadConfig = field(default_factory=SpreadConfig)
    iv_cfg: IVStateConfig = field(default_factory=IVStateConfig)
    thesis_cfg: ThesisMemoryConfig = field(default_factory=ThesisMemoryConfig)
    winding_cfg: WindingConfig = field(default_factory=WindingConfig)
    decision_cfg: DecisionConfig = field(default_factory=DecisionConfig)
    # Advanced runtime (Sun 2026-06-22 audit fix). When True, the engine
    # feeds tape_speed (realized vol ratio) into ThesisMemory.update and
    # time_of_day_scale (session-phase factor) into slot_residual_frame,
    # so the opt-in upgrades in those modules actually fire at runtime.
    # Off by default = legacy bit-identical behavior.
    feed_tape_speed: bool = False
    feed_time_of_day_scale: bool = False
    realized_vol_window_bars: int = 30
    realized_vol_baseline_window_bars: int = 120


@dataclass(frozen=True)
class BeliefSnapshot:
    """Engine output for one tick."""
    ts: Any
    spot: float
    bars_seen: int
    is_warm: bool
    thesis: ThesisSnapshot
    iv_state: IVState
    battlefield: BattlefieldSnapshot
    winding: WindingZone
    bull_state: StateUpdate
    bear_state: StateUpdate
    sweep_state: StateUpdate
    decision: Decision
    slot_readings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "spot": self.spot,
            "bars_seen": self.bars_seen,
            "is_warm": self.is_warm,
            "thesis": self.thesis.to_dict(),
            "iv_state": self.iv_state.to_dict(),
            "battlefield": self.battlefield.to_dict(),
            "winding": self.winding.to_dict(),
            "bull_state": self.bull_state.to_dict(),
            "bear_state": self.bear_state.to_dict(),
            "sweep_state": self.sweep_state.to_dict(),
            "decision": self.decision.to_dict(),
            "slot_readings": list(self.slot_readings),
        }


@dataclass
class _ContractHistory:
    """Per-contract rolling deque of (ts, spot, mark, bid, ask, bid_qty,
    ask_qty). Phase 4 reads this as a DataFrame on each tick."""
    rows: Deque[Dict[str, Any]] = field(default_factory=deque)

    def push(self, row: Dict[str, Any], cap: int) -> None:
        self.rows.append(row)
        while len(self.rows) > cap:
            self.rows.popleft()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.rows))


@dataclass
class BeliefEngine:
    """The live Belief Engine.

    Usage:

        eng = BeliefEngine()
        for tick in stream:
            quotes_by_strike = {(strike, "CE"): Quote(...), (strike, "PE"): Quote(...), ...}
            snap = eng.observe(ts=tick.ts, spot=tick.spot, quotes=quotes_by_strike,
                                hunt_verdict=hunt, trap_verdict=trap)
            # act on snap.decision

    ``hunt_verdict`` and ``trap_verdict`` are the upstream strings from
    ``liquidity_hunt`` and ``premium_divergence`` for the current bar; the
    engine wires them through the thesis memory + state machines.
    """
    cfg: BeliefEngineConfig = field(default_factory=BeliefEngineConfig)
    _mark_tracker: MarkPriceTracker = field(init=False)
    _hist: Dict[Any, _ContractHistory] = field(default_factory=dict, init=False)
    _spot_hist: Deque[float] = field(default_factory=deque, init=False)
    _atm_strike: Optional[float] = field(default=None, init=False)
    _slots: Dict[Any, MoneynessSlot] = field(default_factory=dict, init=False)
    _thesis: ThesisMemory = field(init=False)
    _winding: WindingDetector = field(init=False)
    _bull: BullContinuationMachine = field(default_factory=BullContinuationMachine, init=False)
    _bear: BearContinuationMachine = field(default_factory=BearContinuationMachine, init=False)
    _sweep: LiquiditySweepMachine = field(default_factory=LiquiditySweepMachine, init=False)
    _bars_seen: int = field(default=0, init=False)
    _last_friendliness: float = field(default=1.0, init=False)
    _last_spread_state: str = field(default=SPREAD_CLEAN, init=False)
    # Bookkeeping for the advanced runtime feeders.
    _spot_returns: Deque[float] = field(default_factory=deque, init=False)
    _last_tape_speed: float = field(default=1.0, init=False)
    _last_tod_scale: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        self._mark_tracker = MarkPriceTracker(cfg=self.cfg.mark_cfg)
        self._thesis = ThesisMemory(cfg=self.cfg.thesis_cfg)
        self._winding = WindingDetector(cfg=self.cfg.winding_cfg)

    def reset(self) -> None:
        self._mark_tracker.reset()
        self._hist.clear()
        self._spot_hist.clear()
        self._atm_strike = None
        self._slots.clear()
        self._thesis.reset()
        self._winding.reset()
        self._bull.reset()
        self._bear.reset()
        self._sweep.reset()
        self._bars_seen = 0

    @property
    def bars_seen(self) -> int:
        return self._bars_seen

    @property
    def is_warm(self) -> bool:
        return self._bars_seen >= self.cfg.warmup_bars

    # ── Runtime feeders for the advanced (opt-in) Phase-4/6 layers ──

    def _compute_tape_speed(self) -> float:
        """Realized-vol ratio (recent / baseline). >1 = fast tape, <1 = slow.

        Falls back to 1.0 until we have enough samples to be meaningful."""
        import statistics
        cfg = self.cfg
        rets = list(self._spot_returns)
        if len(rets) < cfg.realized_vol_window_bars + 4:
            return 1.0
        recent = rets[-cfg.realized_vol_window_bars:]
        baseline_n = min(len(rets), cfg.realized_vol_baseline_window_bars)
        baseline = rets[-baseline_n:]
        try:
            sd_recent = statistics.pstdev(recent)
            sd_base = statistics.pstdev(baseline)
        except Exception:
            return 1.0
        if sd_base <= 1e-9:
            return 1.0
        ratio = sd_recent / sd_base
        # Bound aggressively — the calibrator inside ThesisMemoryConfig
        # already clips the resulting decay, but a tame input is safer.
        return max(0.25, min(4.0, ratio))

    def _compute_time_of_day_scale(self, ts: Any) -> float:
        """Per-bar band scale factor based on IST session phase.

        * 09:15-09:30 morning open: 1.30 (wider)
        * 11:30-13:30 lunch chop:   1.20 (wider)
        * 14:45-15:30 closing run:  1.25 (wider)
        * Other:                    1.00 (normal)
        """
        try:
            from datetime import datetime, timezone, timedelta
            if isinstance(ts, str):
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, datetime):
                dt = ts
            else:
                return 1.0
            ist = timezone(timedelta(hours=5, minutes=30))
            now = dt.astimezone(ist) if dt.tzinfo else dt.replace(tzinfo=ist)
            minute = now.hour * 60 + now.minute
            if minute < 9 * 60 + 15:
                return 1.0
            if minute < 9 * 60 + 30:
                return 1.30
            if 11 * 60 + 30 <= minute < 13 * 60 + 30:
                return 1.20
            if 14 * 60 + 45 <= minute <= 15 * 60 + 30:
                return 1.25
            return 1.0
        except Exception:
            return 1.0

    def _rebuild_slots(self, spot: float, quote_keys: List[Any]) -> None:
        """Build (or rebuild on ATM drift) the moneyness slots for every
        contract key in the current tick."""
        for key in quote_keys:
            try:
                strike, option_type = key
            except Exception:
                continue
            self._slots[key] = classify_moneyness(
                spot, float(strike), str(option_type), self.cfg.strike_step)
        self._atm_strike = round(spot / self.cfg.strike_step) * self.cfg.strike_step

    def observe(self, *,
                ts: Any,
                spot: float,
                quotes: Dict[Any, Quote],
                hunt_verdict: str = "",
                trap_verdict: str = "",
                ) -> BeliefSnapshot:
        """Run one tick through the whole stack and return the snapshot."""
        cfg = self.cfg
        self._bars_seen += 1
        # Track spot log-returns for the realized-vol-based tape-speed signal.
        prev_spot = self._spot_hist[-1] if self._spot_hist else None
        self._spot_hist.append(float(spot))
        while len(self._spot_hist) > _HISTORY_BARS:
            self._spot_hist.popleft()
        if prev_spot is not None and prev_spot > 0:
            import math
            try:
                self._spot_returns.append(math.log(float(spot) / prev_spot))
            except Exception:
                pass
            cap = max(cfg.realized_vol_baseline_window_bars,
                       cfg.realized_vol_window_bars) + 8
            while len(self._spot_returns) > cap:
                self._spot_returns.popleft()

        # 1. Phase 2 — mark price per contract; track clean-mark fraction.
        clean_count = 0
        total = max(1, len(quotes))
        marks: Dict[Any, float] = {}
        mark_details: Dict[Any, MarkPrice] = {}
        for key, q in quotes.items():
            m = self._mark_tracker.update(key, q)
            marks[key] = m.price
            mark_details[key] = m
            if m.source in ("microprice", "mid"):
                clean_count += 1
        clean_fraction = clean_count / total

        # 2. Phase 3 — refresh moneyness slots (rebuild on ATM drift).
        rebuild = (self._atm_strike is None
                   or abs(spot - self._atm_strike) >
                   cfg.rebuild_atm_threshold_strikes * cfg.strike_step
                   or set(quotes.keys()) != set(self._slots.keys()))
        if rebuild:
            self._rebuild_slots(spot, list(quotes.keys()))

        # 3. Keep per-contract rolling history.
        for key, q in quotes.items():
            self._hist.setdefault(key, _ContractHistory())
            self._hist[key].push({
                "ts": ts, "spot": float(spot), "mark": marks[key],
                "bid": float(q.bid), "ask": float(q.ask),
                "bid_qty": float(q.bid_qty), "ask_qty": float(q.ask_qty),
            }, cap=_HISTORY_BARS)

        # 4. Phase 4 — per-slot residual + spread friendliness on the
        # rolling history (only when enough bars exist; otherwise default
        # to dod_z=0, normal acceptance, full friendliness).
        readings: List[SlotReading] = []
        slot_readings: List[Dict[str, Any]] = []
        friendliness_acc = 0.0
        spread_state_counts: Dict[str, int] = {}
        for key, hist in self._hist.items():
            slot = self._slots.get(key)
            if slot is None:
                continue
            df = hist.to_frame()
            if len(df) < max(cfg.fair_cfg.delta_window, cfg.resid_cfg.band_window) + 4:
                reading = SlotReading(
                    slot=slot, dod_z=0.0, is_abnormal=False,
                    friendliness=1.0, acceptance="normal",
                )
                readings.append(reading)
                slot_readings.append(_slot_reading_payload(
                    reading,
                    mark=mark_details.get(key),
                    spread_state=SPREAD_CLEAN,
                ))
                friendliness_acc += 1.0
                spread_state_counts[SPREAD_CLEAN] = spread_state_counts.get(SPREAD_CLEAN, 0) + 1
                continue
            try:
                tod_series = None
                if (cfg.feed_time_of_day_scale
                        and cfg.resid_cfg.enable_time_of_day_scaling):
                    # df.index carries the per-bar ts; compute the scale series.
                    import pandas as _pd
                    scale_vals = [self._compute_time_of_day_scale(ix)
                                  for ix in df.index]
                    tod_series = _pd.Series(scale_vals, index=df.index)
                resid = slot_residual_frame(
                    df, slot, cfg.fair_cfg, cfg.resid_cfg,
                    time_of_day_scale=tod_series,
                )
                last_resid = resid.iloc[-1]
                spread = spread_friendliness_frame(df, cfg.spread_cfg)
                last_sp = spread.iloc[-1]
                friend = float(last_sp["friendliness"])
                sstate = str(last_sp["spread_state"])
            except Exception:
                last_resid = {"dod_z": 0.0, "is_abnormal": False, "acceptance": "normal"}
                friend = 1.0; sstate = SPREAD_CLEAN
            reading = SlotReading(
                slot=slot, dod_z=float(last_resid["dod_z"]),
                is_abnormal=bool(last_resid["is_abnormal"]),
                friendliness=friend,
                acceptance=str(last_resid["acceptance"]),
            )
            readings.append(reading)
            slot_readings.append(_slot_reading_payload(
                reading,
                mark=mark_details.get(key),
                spread_state=sstate,
            ))
            friendliness_acc += friend
            spread_state_counts[sstate] = spread_state_counts.get(sstate, 0) + 1

        avg_friendliness = friendliness_acc / max(1, len(readings))
        self._last_friendliness = avg_friendliness
        # Take the modal spread state (rough — good enough for the
        # liquidity-sweep machine's normalisation check).
        self._last_spread_state = (
            max(spread_state_counts, key=spread_state_counts.get)
            if spread_state_counts else SPREAD_CLEAN
        )

        # 5. Phase 5 — battlefield + IV state.
        bf = battlefield_snapshot(readings)
        iv = classify_iv_state(bf, readings,
                                clean_mark_fraction=clean_fraction,
                                cfg=cfg.iv_cfg)

        # 6. Phase 6 — thesis memory.
        tape_speed_arg = None
        if cfg.feed_tape_speed and cfg.thesis_cfg.enable_adaptive_decay:
            tape_speed_arg = self._compute_tape_speed()
            self._last_tape_speed = tape_speed_arg
        thesis = self._thesis.update(iv_state=iv,
                                      hunt_verdict=hunt_verdict,
                                      trap_verdict=trap_verdict,
                                      tape_speed=tape_speed_arg)

        # 7. Phase 7 — winding + state machines.
        winding = self._winding.observe(
            float(spot), battlefield=bf, iv_state=iv,
            spread_friendly=(self._last_spread_state == SPREAD_CLEAN
                             and avg_friendliness >= 0.45),
        )
        bull_state = self._bull.update(
            spot=float(spot), battlefield=bf, iv_state=iv,
            winding=winding, hunt_verdict=hunt_verdict,
            trap_verdict=trap_verdict)
        bear_state = self._bear.update(
            spot=float(spot), battlefield=bf, iv_state=iv,
            winding=winding, hunt_verdict=hunt_verdict,
            trap_verdict=trap_verdict)
        sweep_state = self._sweep.update(
            spot=float(spot), battlefield=bf, iv_state=iv,
            spread_state=self._last_spread_state, hunt_verdict=hunt_verdict)

        # 8. Phase 8 — decision. During warmup, refuse to act.
        if not self.is_warm:
            need = max(0, cfg.warmup_bars - self._bars_seen)
            decision = Decision(
                action=ACTION_NO_TRADE, trade_allowed=False, direction=0,
                confidence=0.0,
                strike=StrikeRecommendation("", 0, ""),
                thesis_state=thesis.composite_state,
                invalidation_rule="", exit_rule="",
                no_trade_reason=f"warming up — {need} more bars needed",
                spread_friendliness=avg_friendliness,
                notes=[f"warmup: {self._bars_seen}/{cfg.warmup_bars} bars"],
            )
        else:
            decision = decide(
                thesis=thesis, iv_state=iv, battlefield=bf,
                bull_state=bull_state, bear_state=bear_state,
                sweep_state=sweep_state, winding=winding,
                avg_friendliness=avg_friendliness, cfg=cfg.decision_cfg,
            )

        return BeliefSnapshot(
            ts=ts, spot=float(spot),
            bars_seen=self._bars_seen, is_warm=self.is_warm,
            thesis=thesis, iv_state=iv, battlefield=bf,
            winding=winding, bull_state=bull_state, bear_state=bear_state,
            sweep_state=sweep_state, decision=decision,
            slot_readings=sorted(
                slot_readings,
                key=lambda r: (
                    str(r.get("option_type") or ""),
                    int(r.get("level") or 0),
                    float(r.get("strike") or 0.0),
                ),
            ),
        )
