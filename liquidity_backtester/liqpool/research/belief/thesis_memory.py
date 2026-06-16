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
    """Knobs for the thesis-memory state. Defaults are the founder's numbers."""
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

    def __post_init__(self) -> None:
        if not (0.0 < self.decay_per_bar < 1.0):
            raise ValueError("decay_per_bar must be in (0,1)")
        if self.exit_confidence >= self.entry_confidence:
            raise ValueError("exit_confidence must be < entry_confidence (hysteresis)")
        if not (0.0 <= self.no_trade_danger <= 100.0):
            raise ValueError("no_trade_danger must be in [0,100]")


@dataclass
class ThesisSnapshot:
    """Per-bar output of the memory layer."""
    bull_thesis_score: float
    bear_thesis_score: float
    vol_expansion_score: float
    liquidity_danger_score: float
    no_trade_score: float
    held_direction: int                  # +1 long, -1 short, 0 flat
    composite_state: str                 # one of the BULL_ENTRY / HOLD_BULL / ... labels
    just_changed: bool                   # composite_state different from previous bar
    note: str

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
        )

    def update(self,
               *,
               iv_state: Optional[IVState] = None,
               hunt_verdict: str = "",
               trap_verdict: str = "",
               ) -> ThesisSnapshot:
        """Advance the memory by one bar.

        Inputs:
          * ``iv_state``      — the Phase-5 :class:`IVState` for this bar
          * ``hunt_verdict``  — string from ``liquidity_hunt`` (e.g.
                                ``"LIQUIDITY_HUNT_UP"`` / ``"REGIME_BREAK_DOWN"``)
          * ``trap_verdict``  — string from ``premium_divergence`` (e.g.
                                ``"STRONG_BULL_TRAP"`` / ``"BULL_TRAP"``)

        Any input may be omitted / empty — the memory simply decays.
        """
        cfg = self.cfg
        s = self._state
        # 1. Bleed all scores by the decay factor.
        s.bull *= cfg.decay_per_bar
        s.bear *= cfg.decay_per_bar
        s.vol_exp *= cfg.decay_per_bar
        s.liq_danger *= cfg.decay_per_bar

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
        )
