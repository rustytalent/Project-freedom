"""Thesis memory with hysteresis (Belief Engine, Phase 6).

The founder's principle: "Without memory, the engine will flip like a
scared retail trader. The model should not enter and exit like a
mosquito."

This module is the assimilation + emotional-stability layer. It takes
the per-bar reads from Phase 5 (IV state + battlefield) plus the trap /
hunt verdicts already in the codebase (premium_divergence, liquidity_hunt)
and accumulates them into FIVE persistent scores in [0,100]:

  * bull_thesis_score      — confidence that direction is bullish
  * bear_thesis_score      — confidence that direction is bearish
  * vol_expansion_score    — confidence the regime is common-IV-shock
  * liquidity_danger_score — confidence the book is unreliable (no-trade)
  * no_trade_score         — composite "don't act" score

The scores have HYSTERESIS via three thresholds (the founder's numbers):

    entry_confidence   = 70   # must reach this to ENTER long/short
    exit_confidence    = 45   # falls below this to EXIT (gap = 25 → no flip-flop)
    no_trade_danger    = 65   # any score above this on liquidity → block all

Update dynamics — "pullback that survives nicks the score a little;
pullback that breaks structure cuts it hard":

    score_t+1 = decay·score_t + delta_t

  * decay = 0.94 (≈ 16-bar half-life) — without confirmation, conviction
    bleeds gently;
  * a CONFIRMING signal adds a positive delta (bigger when battlefield
    agreement is clean; biggest when an acceptance skew across rails is
    visible);
  * a DAMAGING signal subtracts:
      - a counter-direction battlefield read → small nick
      - a regime_break verdict from liquidity_hunt → large cut
      - a STRONG_BULL_TRAP / STRONG_BEAR_TRAP on the held direction → cut
  * scores are clamped to [0, 100].

State machine output (single label per bar):

    BULL_ENTRY        — bull_thesis ≥ entry AND no_trade < danger → enter long
    BEAR_ENTRY        — bear_thesis ≥ entry AND no_trade < danger → enter short
    HOLD_BULL         — sitting long, bull_thesis ≥ exit
    HOLD_BEAR         — sitting short, bear_thesis ≥ exit
    EXIT_BULL         — was long, bull_thesis fell below exit
    EXIT_BEAR         — was short, bear_thesis fell below exit
    NO_TRADE_DANGER   — liquidity or vol score above danger (regardless)
    NEUTRAL           — nothing to do

The exit-vs-entry GAP (25 points) is the hysteresis: a score that just
touched 71 doesn't immediately exit when noise pushes it to 69. It exits
only when conviction has actually decayed to below 45 — that's what
"don't flip like a mosquito" means.

The module is strictly causal — every score update uses only data from
bar T and the prior score state. Streaming-safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .iv_state import (
    IV_COMMON_SHOCK,
    IV_DIRECTIONAL_BEAR,
    IV_DIRECTIONAL_BULL,
    IV_DIRTY_DATA,
    IV_LIQUIDITY_DISTORTION,
    IV_NEUTRAL,
    IV_VOL_CONTRACTION,
    IVState,
)


# Composite state labels.
BULL_ENTRY = "BULL_ENTRY"
BEAR_ENTRY = "BEAR_ENTRY"
HOLD_BULL = "HOLD_BULL"
HOLD_BEAR = "HOLD_BEAR"
EXIT_BULL = "EXIT_BULL"
EXIT_BEAR = "EXIT_BEAR"
NO_TRADE_DANGER = "NO_TRADE_DANGER"
NEUTRAL_THESIS = "NEUTRAL"


@dataclass
class ThesisMemoryConfig:
    """Knobs for the thesis-memory state. Defaults are the founder's numbers.

    The advanced fields below are ALL opt-in (default = off → identical
    behavior to the original Phase-6 builder). Added on Sun 2026-06-22
    in response to the founder's audit, which flagged two primitives:
      1. Fixed decay rate regardless of tape speed — in fast tape,
         conviction should bleed faster; in slow tape, slower.
      2. No uncertainty propagation — scores are point estimates with
         no confidence carry; downstream can't tell "72 ± 8" from "72 ± 1".
    """
    entry_confidence: float = 70.0
    exit_confidence: float = 45.0
    no_trade_danger: float = 65.0
    decay_per_bar: float = 0.94          # ≈ 16-bar half-life
    # Per-signal score deltas. Positive = adds to the named score.
    delta_battlefield_bull_agreement: float = 18.0
    delta_battlefield_bear_agreement: float = 18.0
    delta_acceptance_skew_bonus: float = 6.0      # extra when IV state was acceptance-skew
    delta_vol_expansion: float = 14.0
    delta_liquidity_distortion: float = 22.0
    delta_dirty_data: float = 28.0
    # Damage costs.
    cost_counter_direction_read: float = 6.0     # battlefield disagrees with held side
    cost_regime_break: float = 24.0              # liquidity_hunt REGIME_BREAK on held side
    cost_strong_trap_on_held: float = 18.0       # STRONG_BULL/BEAR_TRAP against held side
    # Confidence-confirmation amplifier — the high-confidence IV state pushes harder.
    confidence_amp: float = 1.0
    eps: float = 1e-9

    # ── Advanced (opt-in) — tape-speed-aware decay ────────────────
    # When enabled, the effective decay each bar = decay_per_bar adjusted
    # by an observed-tape-speed signal. The signal is provided per-update
    # via `tape_speed` (a ratio: realized / baseline volatility, ≈1.0 in
    # normal tape, >1 in fast tape, <1 in slow tape). Effective decay is
    # bounded to [decay_min, decay_max].
    enable_adaptive_decay: bool = False
    decay_min: float = 0.86             # ≈ 5-bar half-life cap (fast tape)
    decay_max: float = 0.975            # ≈ 27-bar half-life cap (slow tape)
    adaptive_decay_sensitivity: float = 0.06   # per-unit tape-speed pull

    # ── Advanced (opt-in) — score uncertainty ─────────────────────
    # When enabled, every score gets a confidence interval inferred from
    # the recent stability of the inputs (frequent IV-state flips = wide
    # CI; consistent reads = tight CI). Added as fields on ThesisSnapshot;
    # the point estimates are unchanged.
    enable_uncertainty: bool = False
    uncertainty_window: int = 12
    uncertainty_max_band: float = 18.0      # max ± band around any score

    def __post_init__(self) -> None:
        if not (0.0 < self.decay_per_bar < 1.0):
            raise ValueError("decay_per_bar must be in (0,1)")
        if self.exit_confidence >= self.entry_confidence:
            raise ValueError("exit_confidence must be < entry_confidence (hysteresis)")
        if not (0.0 <= self.no_trade_danger <= 100.0):
            raise ValueError("no_trade_danger must be in [0,100]")
        if self.enable_adaptive_decay:
            if not (0 < self.decay_min < self.decay_max < 1.0):
                raise ValueError(
                    "adaptive decay: 0 < decay_min < decay_max < 1 required")
            if not (self.decay_min <= self.decay_per_bar <= self.decay_max):
                raise ValueError(
                    "adaptive decay: decay_per_bar must lie in "
                    "[decay_min, decay_max]")
        if self.enable_uncertainty:
            if self.uncertainty_window < 3:
                raise ValueError("uncertainty_window must be >= 3")
            if self.uncertainty_max_band <= 0:
                raise ValueError("uncertainty_max_band must be > 0")


@dataclass
class ThesisSnapshot:
    """Per-bar output of the memory layer.

    The advanced CI fields default to 0.0 (zero-width band) so that any
    downstream consumer that doesn't care about uncertainty sees the
    same point-estimate-only world it always saw. When the memory has
    ``enable_uncertainty`` on, the half-width bands report how confident
    the score is.
    """
    bull_thesis_score: float
    bear_thesis_score: float
    vol_expansion_score: float
    liquidity_danger_score: float
    no_trade_score: float
    held_direction: int                  # +1 long, -1 short, 0 flat
    composite_state: str                 # one of the BULL_ENTRY / HOLD_BULL / ... labels
    just_changed: bool                   # composite_state different from previous bar
    note: str
    # Advanced (opt-in) — confidence half-widths around each score
    # (i.e. score ± ci). 0.0 = the legacy "no uncertainty surfaced" world.
    bull_thesis_ci: float = 0.0
    bear_thesis_ci: float = 0.0
    vol_expansion_ci: float = 0.0
    liquidity_danger_ci: float = 0.0
    no_trade_ci: float = 0.0
    # Realized decay applied this bar (informational; defaults to cfg's
    # static decay_per_bar when adaptive decay is off).
    effective_decay: float = 0.94

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class _MutableState:
    bull: float = 0.0
    bear: float = 0.0
    vol_exp: float = 0.0
    liq_danger: float = 0.0
    held_direction: int = 0
    last_composite: str = NEUTRAL_THESIS
    # Advanced uncertainty history — rolling window of recent IV-state
    # labels so we can detect flip-flopping (= wide CI) vs consistent
    # reads (= tight CI). Bounded by ThesisMemoryConfig.uncertainty_window.
    iv_state_history: list = field(default_factory=list)
    delta_history: list = field(default_factory=list)
    last_effective_decay: float = 0.94


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class ThesisMemory:
    """Stateful memory: feed one bar at a time, get a :class:`ThesisSnapshot`.

    Usage:

        mem = ThesisMemory()
        for bar in stream:
            snap = mem.update(iv_state=..., hunt_verdict=..., trap_verdict=...)
            # snap.composite_state tells you what to do this bar.

    The memory is reset to a blank slate by ``reset()``.
    """
    cfg: ThesisMemoryConfig = field(default_factory=ThesisMemoryConfig)
    _state: _MutableState = field(default_factory=_MutableState, init=False)

    def reset(self) -> None:
        self._state = _MutableState()

    @property
    def held_direction(self) -> int:
        return self._state.held_direction

    @property
    def last_composite(self) -> str:
        return self._state.last_composite

    def snapshot(self) -> ThesisSnapshot:
        s = self._state
        no_trade = max(s.liq_danger, s.vol_exp * 0.5)
        ci = self._compute_uncertainty()
        return ThesisSnapshot(
            bull_thesis_score=s.bull,
            bear_thesis_score=s.bear,
            vol_expansion_score=s.vol_exp,
            liquidity_danger_score=s.liq_danger,
            no_trade_score=no_trade,
            held_direction=s.held_direction,
            composite_state=s.last_composite,
            just_changed=False,
            note="snapshot (no update)",
            bull_thesis_ci=ci, bear_thesis_ci=ci,
            vol_expansion_ci=ci, liquidity_danger_ci=ci,
            no_trade_ci=ci,
            effective_decay=s.last_effective_decay,
        )

    def _compute_uncertainty(self) -> float:
        """Confidence half-width band around each score.

        When the IV-state history is unstable (flip-flopping) AND/OR
        the recent deltas have high variance, the band widens. When
        everything is consistent, the band collapses toward 0.

        Returns 0.0 when ``enable_uncertainty`` is off (legacy behavior).
        """
        cfg = self.cfg
        if not cfg.enable_uncertainty:
            return 0.0
        s = self._state
        history = s.iv_state_history[-cfg.uncertainty_window:]
        deltas = s.delta_history[-cfg.uncertainty_window:]
        if len(history) < 3:
            # Early in the session — wide band by default.
            return cfg.uncertainty_max_band * 0.6

        # Flip-flop fraction: how often did the IV state CHANGE label?
        flips = sum(
            1 for i in range(1, len(history)) if history[i] != history[i - 1])
        flip_rate = flips / max(1, len(history) - 1)

        # Delta variance: how erratic have the recent updates been?
        if len(deltas) >= 3:
            mean_d = sum(deltas) / len(deltas)
            var_d = sum((d - mean_d) ** 2 for d in deltas) / len(deltas)
            std_d = var_d ** 0.5
            # Normalize against the typical bull/bear delta magnitude.
            std_norm = min(1.0, std_d / max(1.0, cfg.delta_battlefield_bull_agreement))
        else:
            std_norm = 0.5

        # Blend: more flips OR more delta variance = wider band.
        instability = 0.6 * flip_rate + 0.4 * std_norm
        band = cfg.uncertainty_max_band * instability
        return max(0.0, min(cfg.uncertainty_max_band, band))

    def update(self,
               *,
               iv_state: Optional[IVState] = None,
               hunt_verdict: str = "",
               trap_verdict: str = "",
               tape_speed: Optional[float] = None,
               ) -> ThesisSnapshot:
        """Advance the memory by one bar.

        Inputs:
          * ``iv_state``      — the Phase-5 :class:`IVState` for this bar
          * ``hunt_verdict``  — string from ``liquidity_hunt`` (e.g.
                                ``"LIQUIDITY_HUNT_UP"`` / ``"REGIME_BREAK_DOWN"``)
          * ``trap_verdict``  — string from ``premium_divergence`` (e.g.
                                ``"STRONG_BULL_TRAP"`` / ``"BULL_TRAP"``)
          * ``tape_speed``    — OPTIONAL float used only when
                                ``enable_adaptive_decay`` is True. Ratio of
                                realized to baseline volatility: ~1.0 in normal
                                tape, >1.0 in fast tape (decay faster), <1.0 in
                                slow tape (decay slower). When omitted with the
                                flag on, defaults to 1.0 (no adjustment).

        Any input may be omitted / empty — the memory simply decays.
        """
        cfg = self.cfg
        s = self._state

        # 1. Bleed all scores by the decay factor.
        # If adaptive decay is enabled, scale the legacy decay by the
        # observed tape speed (clipped to [decay_min, decay_max]). Default
        # behavior (flag off) is bit-identical to the original.
        if cfg.enable_adaptive_decay:
            ts = float(tape_speed) if tape_speed is not None else 1.0
            # Faster tape → smaller decay coefficient (faster bleed).
            # The sensitivity scales how much the coefficient shifts per
            # unit of tape-speed deviation from 1.0.
            adjust = (ts - 1.0) * cfg.adaptive_decay_sensitivity
            eff_decay = cfg.decay_per_bar - adjust
            eff_decay = max(cfg.decay_min, min(cfg.decay_max, eff_decay))
        else:
            eff_decay = cfg.decay_per_bar
        s.last_effective_decay = eff_decay
        s.bull *= eff_decay
        s.bear *= eff_decay
        s.vol_exp *= eff_decay
        s.liq_danger *= eff_decay

        notes: list[str] = []

        # 2. IV state — the primary belief signal.
        if iv_state is not None:
            iv_conf = max(0.0, min(1.0, float(iv_state.confidence or 0.0)))
            amp = 1.0 + cfg.confidence_amp * iv_conf

            if iv_state.state == IV_DIRECTIONAL_BULL:
                bonus = cfg.delta_acceptance_skew_bonus if "acceptance" in (iv_state.note or "") else 0.0
                s.bull = _clamp(s.bull + amp * (cfg.delta_battlefield_bull_agreement + bonus))
                notes.append(f"IV bull (+{amp * (cfg.delta_battlefield_bull_agreement + bonus):.1f})")
                # Damaging to bear thesis.
                s.bear = _clamp(s.bear - cfg.cost_counter_direction_read)
            elif iv_state.state == IV_DIRECTIONAL_BEAR:
                bonus = cfg.delta_acceptance_skew_bonus if "acceptance" in (iv_state.note or "") else 0.0
                s.bear = _clamp(s.bear + amp * (cfg.delta_battlefield_bear_agreement + bonus))
                notes.append(f"IV bear (+{amp * (cfg.delta_battlefield_bear_agreement + bonus):.1f})")
                s.bull = _clamp(s.bull - cfg.cost_counter_direction_read)
            elif iv_state.state == IV_COMMON_SHOCK:
                s.vol_exp = _clamp(s.vol_exp + cfg.delta_vol_expansion * amp)
                notes.append("IV common shock")
            elif iv_state.state == IV_LIQUIDITY_DISTORTION:
                s.liq_danger = _clamp(s.liq_danger + cfg.delta_liquidity_distortion * amp)
                notes.append("liquidity distortion")
            elif iv_state.state == IV_DIRTY_DATA:
                s.liq_danger = _clamp(s.liq_danger + cfg.delta_dirty_data * amp)
                notes.append("dirty data")
            elif iv_state.state == IV_VOL_CONTRACTION:
                # Calm market — vol_exp bleeds extra.
                s.vol_exp *= 0.85
            # IV_NEUTRAL: pure decay.

        # 3. Liquidity-hunt verdicts — the founder's "during a pullback,
        # positioning beats price." A regime_break against the held side
        # cuts the thesis hard; a liquidity hunt CONFIRMING the held side
        # nicks the held score UP (positioning defended the regime).
        if hunt_verdict == "REGIME_BREAK_DOWN" and s.held_direction == +1:
            s.bull = _clamp(s.bull - cfg.cost_regime_break)
            notes.append("regime_break_down on held long (−)")
        elif hunt_verdict == "REGIME_BREAK_UP" and s.held_direction == -1:
            s.bear = _clamp(s.bear - cfg.cost_regime_break)
            notes.append("regime_break_up on held short (−)")
        elif hunt_verdict == "LIQUIDITY_HUNT_UP" and s.held_direction == +1:
            s.bull = _clamp(s.bull + 0.5 * cfg.delta_battlefield_bull_agreement)
            notes.append("hunt confirms held long (+)")
        elif hunt_verdict == "LIQUIDITY_HUNT_DOWN" and s.held_direction == -1:
            s.bear = _clamp(s.bear + 0.5 * cfg.delta_battlefield_bear_agreement)
            notes.append("hunt confirms held short (+)")

        # 4. Trap verdicts — STRONG traps against the held side cut hard.
        if trap_verdict in ("STRONG_BULL_TRAP", "BULL_TRAP") and s.held_direction == +1:
            cost = (cfg.cost_strong_trap_on_held
                    if trap_verdict == "STRONG_BULL_TRAP"
                    else 0.5 * cfg.cost_strong_trap_on_held)
            s.bull = _clamp(s.bull - cost)
            notes.append(f"{trap_verdict} on held long (−)")
        elif trap_verdict in ("STRONG_BEAR_TRAP", "BEAR_TRAP") and s.held_direction == -1:
            cost = (cfg.cost_strong_trap_on_held
                    if trap_verdict == "STRONG_BEAR_TRAP"
                    else 0.5 * cfg.cost_strong_trap_on_held)
            s.bear = _clamp(s.bear - cost)
            notes.append(f"{trap_verdict} on held short (−)")
        # Trap on opposite side → small positive for the contrarian thesis.
        elif trap_verdict == "STRONG_BULL_TRAP" and s.held_direction != +1:
            s.bear = _clamp(s.bear + 0.5 * cfg.delta_battlefield_bear_agreement)
        elif trap_verdict == "STRONG_BEAR_TRAP" and s.held_direction != -1:
            s.bull = _clamp(s.bull + 0.5 * cfg.delta_battlefield_bull_agreement)

        # 5. Composite state with hysteresis.
        no_trade = max(s.liq_danger, s.vol_exp * 0.5)
        prev = s.last_composite

        if no_trade >= cfg.no_trade_danger:
            composite = NO_TRADE_DANGER
            # No-trade should NOT pull the operator out of an existing
            # position automatically — that's the decision layer's job —
            # but the held direction is preserved so the next bar can
            # still EXIT_* if conviction also decays.
        elif s.held_direction == +1:
            # Sitting long.
            if s.bull < cfg.exit_confidence:
                composite = EXIT_BULL
                s.held_direction = 0
            else:
                composite = HOLD_BULL
        elif s.held_direction == -1:
            if s.bear < cfg.exit_confidence:
                composite = EXIT_BEAR
                s.held_direction = 0
            else:
                composite = HOLD_BEAR
        else:
            # Flat. Promote to entry only when thesis clears entry threshold
            # AND it strictly dominates the other side (avoid entering when
            # both bull and bear are elevated near vol expansion).
            if s.bull >= cfg.entry_confidence and s.bull > s.bear + 10.0:
                composite = BULL_ENTRY
                s.held_direction = +1
            elif s.bear >= cfg.entry_confidence and s.bear > s.bull + 10.0:
                composite = BEAR_ENTRY
                s.held_direction = -1
            else:
                composite = NEUTRAL_THESIS

        s.last_composite = composite

        # ── Advanced uncertainty bookkeeping (opt-in) ─────────────
        if cfg.enable_uncertainty:
            iv_label = (iv_state.state if iv_state is not None else "")
            s.iv_state_history.append(iv_label)
            # Magnitude of the largest score-state shift this bar — used
            # as a stand-in for "how much did this bar's update move us?"
            s.delta_history.append(
                abs(s.bull - (s.bull / max(eff_decay, 1e-9)))
                + abs(s.bear - (s.bear / max(eff_decay, 1e-9))))
            # Bound the history to its window so memory doesn't grow.
            cap = max(cfg.uncertainty_window * 4, 64)
            if len(s.iv_state_history) > cap:
                s.iv_state_history = s.iv_state_history[-cap:]
            if len(s.delta_history) > cap:
                s.delta_history = s.delta_history[-cap:]
        ci = self._compute_uncertainty()

        return ThesisSnapshot(
            bull_thesis_score=round(s.bull, 2),
            bear_thesis_score=round(s.bear, 2),
            vol_expansion_score=round(s.vol_exp, 2),
            liquidity_danger_score=round(s.liq_danger, 2),
            no_trade_score=round(no_trade, 2),
            held_direction=s.held_direction,
            composite_state=composite,
            just_changed=(composite != prev),
            note="; ".join(notes) if notes else "decay-only",
            bull_thesis_ci=round(ci, 2),
            bear_thesis_ci=round(ci, 2),
            vol_expansion_ci=round(ci, 2),
            liquidity_danger_ci=round(ci, 2),
            no_trade_ci=round(ci, 2),
            effective_decay=round(eff_decay, 4),
        )
