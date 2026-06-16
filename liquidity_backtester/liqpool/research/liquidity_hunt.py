"""Liquidity-hunt vs real-reversal classifier (stop-hunt detector).

The founder's insight (2026-06-16): a falling spot is NOT bearish on its
own. Retail sees "price down, premium down" and panic-sells. But if the
options *positioning* still defends the prior up-move — calls still
abnormally strong (positive residual anomaly), puts still abnormally weak
(negative anomaly) — then the fall is a liquidity hunt (a stop-sweep that
grabs resting orders before the move resumes), not a reversal. The
principle, stated sharply:

    During a pullback, positioning beats price. A counter-move is only
    real when the options market AGREES with it.

This module is the STATEFUL counterpart to ``premium_divergence``. Where
the trap detector reads a single bar, this one tracks a positioning
REGIME and classifies counter-moves against it:

  Phase 1 — establish the regime. Over ``regime_window`` bars the net
    directional intent (IV-cancelled, Layer 1) AND both-leg residual
    anomaly (Layer 2) must persistently agree on a direction. That is the
    founder's "confirm the move is real first."

  Phase 2 — classify the counter-move. When spot moves AGAINST the
    established regime by more than ``pullback_threshold``:
      * positioning still on-side  → LIQUIDITY_HUNT  (fake — expect resume)
      * positioning flipped        → REGIME_BREAK    (real reversal — exit)
      * positioning ambiguous      → UNCERTAIN       (wait — do not act)

  When spot moves WITH the regime and positioning agrees → TREND_INTACT.
  When no regime is established → NO_REGIME (defer to the snapshot trap
  detector in premium_divergence).

The explicit UNCERTAIN state is deliberate: "model should be every
situation proof" means having an honest "I don't know yet, wait" rather
than forcing a bad binary call on an ambiguous bar.

Everything is strictly causal — every value at bar T uses only data with
timestamp ≤ T (rolling windows look backward only). Safe to run live,
and the streaming engine (premium_divergence.StreamingDivergenceEngine)
drives it bar-by-bar on forming candles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .premium_divergence import PremiumDivergenceConfig, compute_divergence_frame


# Verdict labels.
TREND_INTACT_UP = "TREND_INTACT_UP"        # bullish regime, spot rising with it → ride
TREND_INTACT_DOWN = "TREND_INTACT_DOWN"    # bearish regime, spot falling with it → ride
LIQUIDITY_HUNT_UP = "LIQUIDITY_HUNT_UP"    # bullish regime, fake pullback → expect resume up
LIQUIDITY_HUNT_DOWN = "LIQUIDITY_HUNT_DOWN"  # bearish regime, fake bounce → expect resume down
REGIME_BREAK_DOWN = "REGIME_BREAK_DOWN"    # was bullish, real reversal down → exit longs
REGIME_BREAK_UP = "REGIME_BREAK_UP"        # was bearish, real reversal up → exit shorts
UNCERTAIN = "UNCERTAIN"                      # counter-move but positioning ambiguous → wait
NO_REGIME = "NO_REGIME"                      # no established positioning regime

HUNT_VERDICTS = (LIQUIDITY_HUNT_UP, LIQUIDITY_HUNT_DOWN)
BREAK_VERDICTS = (REGIME_BREAK_DOWN, REGIME_BREAK_UP)
ACTIONABLE_VERDICTS = HUNT_VERDICTS + BREAK_VERDICTS

# Action labels — what the operator should do with each verdict.
ACTION_RIDE = "ride"               # trend intact — hold the regime-direction position
ACTION_HOLD_THROUGH = "hold_through"  # hunt — don't panic; hold/add in regime direction
ACTION_EXIT = "exit"               # regime break — exit the regime-direction position
ACTION_NONE = "none"               # nothing actionable


@dataclass
class LiquidityHuntConfig:
    """Knobs for the regime + hunt classifier.

    ``regime_window`` is the look-back over which positioning must be
    persistent to count as an established regime (the founder's "confirm
    first"). Thresholds are in the same z / robust-σ units the underlying
    divergence frame produces, so they transfer across strikes."""
    regime_window: int = 24
    regime_threshold: float = 0.5        # min |mean net_intent_z| over the window for a regime
    require_both_leg_confirm: bool = True  # also require call/put anomaly to agree on the regime
    pullback_threshold: float = 1.0      # min |spot_move_norm| (std) to count as a counter-move
    confirm_window: int = 4              # bars to smooth positioning over for the defend/break
                                          # check — the founder's "judge cumulatively over 5-6
                                          # candles", damps intra-move 'buffer' flicker
    defend_tolerance: float = 0.5        # positioning still "on-side" if within this of zero
    break_threshold: float = 1.0         # positioning "flipped" past this against the regime
    break_grace_bars: int = 10           # a break is caught against the regime in force within
                                          # this many bars — when positioning flips fast the slow
                                          # regime metric decays to zero before the break prints,
                                          # so we carry the recent regime forward to still flag it
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.regime_window < 4:
            raise ValueError("regime_window must be >= 4")
        if self.confirm_window < 1:
            raise ValueError("confirm_window must be >= 1")


@dataclass(frozen=True)
class HuntSignal:
    """One actionable hunt / break event."""
    index: int
    ts: Any
    verdict: str
    action: str
    direction: int                  # +1 expect/resume up, -1 expect/resume down
    regime: int                     # +1 bullish, -1 bearish (the regime in force)
    spot: float
    spot_move_norm: float           # the counter-move size (signed)
    net_intent_z: float             # current positioning (Layer 1)
    call_anomaly_z: float           # current call residual anomaly (Layer 2)
    put_anomaly_z: float            # current put residual anomaly (Layer 2)
    hunt_confidence: float          # [0,1] — how strongly positioning defends the regime

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _action_for(verdict: str) -> str:
    if verdict in (TREND_INTACT_UP, TREND_INTACT_DOWN):
        return ACTION_RIDE
    if verdict in HUNT_VERDICTS:
        return ACTION_HOLD_THROUGH
    if verdict in BREAK_VERDICTS:
        return ACTION_EXIT
    return ACTION_NONE


def classify_liquidity_hunts(df: pd.DataFrame,
                             cfg: Optional[LiquidityHuntConfig] = None,
                             divergence_cfg: Optional[PremiumDivergenceConfig] = None,
                             ) -> pd.DataFrame:
    """Add regime + hunt classification columns to a divergence frame.

    ``df`` may be raw (we compute the divergence diagnostics first) or an
    already-enriched frame from :func:`compute_divergence_frame`.

    Adds columns:
      * ``regime`` — +1 bullish / −1 bearish / 0 none (established positioning)
      * ``regime_strength`` — smoothed net_intent_z over the regime window
      * ``is_counter_move`` — spot moving against the regime beyond threshold
      * ``hunt_verdict`` — one of the module's verdict labels
      * ``hunt_action`` — ride / hold_through / exit / none
      * ``hunt_confidence`` — [0,1] strength of positioning defense
    """
    cfg = cfg or LiquidityHuntConfig()
    enriched = df if "net_intent_z" in df.columns else compute_divergence_frame(
        df, divergence_cfg)
    out = enriched.copy()

    net_z = pd.to_numeric(out["net_intent_z"], errors="coerce").fillna(0.0)
    call_anom = pd.to_numeric(out["call_resid_anomaly_z"], errors="coerce").fillna(0.0)
    put_anom = pd.to_numeric(out["put_resid_anomaly_z"], errors="coerce").fillna(0.0)
    # Pressure is the accumulated residual LEVEL (Layer 1). Unlike the residual
    # anomaly (Layer 2, which measures *change* and is ~0 during a steady
    # regime), the pressure stays persistently signed while accumulation
    # continues — so it is the right backbone for confirming a regime.
    call_press = pd.to_numeric(out["call_pressure"], errors="coerce").fillna(0.0)
    put_press = pd.to_numeric(out["put_pressure"], errors="coerce").fillna(0.0)
    spot_move = pd.to_numeric(out["spot_move_norm"], errors="coerce").fillna(0.0)

    w = cfg.regime_window
    min_p = max(4, w // 2)
    regime_strength = net_z.rolling(w, min_periods=min_p).mean().fillna(0.0)
    call_press_avg = call_press.rolling(w, min_periods=min_p).mean().fillna(0.0)
    put_press_avg = put_press.rolling(w, min_periods=min_p).mean().fillna(0.0)

    bullish = regime_strength > cfg.regime_threshold
    bearish = regime_strength < -cfg.regime_threshold
    if cfg.require_both_leg_confirm:
        # Bullish regime: calls persistently strong (pressure > 0), puts
        # persistently weak (pressure < 0) — sustained LEVELS, not anomalies.
        bullish = bullish & (call_press_avg > 0.0) & (put_press_avg < 0.0)
        bearish = bearish & (call_press_avg < 0.0) & (put_press_avg > 0.0)
    regime = np.where(bullish, 1, np.where(bearish, -1, 0)).astype(int)

    # Recent regime: carry the last non-zero regime forward up to
    # ``break_grace_bars`` bars. A fast positioning flip collapses the slow
    # regime metric to zero before the break prints; the grace fill lets us
    # still flag "this was a bullish regime that just broke."
    reg_nonzero = pd.Series(regime, index=out.index).replace(0, np.nan)
    recent_regime = reg_nonzero.ffill(limit=cfg.break_grace_bars).fillna(0).astype(int).values

    # Counter-move: spot moving against the recent regime beyond the threshold.
    counter_up_regime = (recent_regime == 1) & (spot_move.values < -cfg.pullback_threshold)
    counter_dn_regime = (recent_regime == -1) & (spot_move.values > cfg.pullback_threshold)
    is_counter = counter_up_regime | counter_dn_regime

    # Positioning state relative to the regime, judged on a SHORT smoothed
    # window (the founder's "cumulatively over 5-6 candles"). Using the raw
    # single-bar anomaly here makes the verdict flicker on intra-move
    # 'buffer'; the rolling mean over confirm_window damps that.
    cw = cfg.confirm_window
    cwp = max(1, cw // 2)
    net_s = net_z.rolling(cw, min_periods=cwp).mean().fillna(0.0).values
    call_s = call_anom.rolling(cw, min_periods=cwp).mean().fillna(0.0).values
    put_s = put_anom.rolling(cw, min_periods=cwp).mean().fillna(0.0).values

    #   bullish regime defended: net intent still bullish-ish, puts not being
    #     loaded, calls not collapsing.
    bull_defended = ((net_s > -cfg.defend_tolerance)
                     & (put_s < cfg.break_threshold)
                     & (call_s > -cfg.break_threshold))
    bull_broken = ((net_s < -cfg.break_threshold)
                   | (put_s > cfg.break_threshold))
    bear_defended = ((net_s < cfg.defend_tolerance)
                     & (call_s < cfg.break_threshold)
                     & (put_s > -cfg.break_threshold))
    bear_broken = ((net_s > cfg.break_threshold)
                   | (call_s > cfg.break_threshold))

    verdict = np.full(len(out), NO_REGIME, dtype=object)
    # Trend intact: moving with the regime (not a counter-move) while in a regime.
    verdict = np.where((regime == 1) & ~is_counter, TREND_INTACT_UP, verdict)
    verdict = np.where((regime == -1) & ~is_counter, TREND_INTACT_DOWN, verdict)
    # Counter-move classification (overrides trend-intact on those bars).
    #   break is checked with priority — a real reversal must not be masked
    #   as a hunt.
    verdict = np.where(counter_up_regime & bull_defended & ~bull_broken,
                       LIQUIDITY_HUNT_UP, verdict)
    verdict = np.where(counter_dn_regime & bear_defended & ~bear_broken,
                       LIQUIDITY_HUNT_DOWN, verdict)
    verdict = np.where(counter_up_regime & bull_broken, REGIME_BREAK_DOWN, verdict)
    verdict = np.where(counter_dn_regime & bear_broken, REGIME_BREAK_UP, verdict)
    # Ambiguous counter-move: neither clearly defended nor clearly broken.
    ambiguous = is_counter & ~np.isin(
        verdict, (LIQUIDITY_HUNT_UP, LIQUIDITY_HUNT_DOWN,
                  REGIME_BREAK_DOWN, REGIME_BREAK_UP))
    verdict = np.where(ambiguous, UNCERTAIN, verdict)

    # Confidence: how strongly positioning sits on the regime's side relative
    # to the size of the adverse move. Only meaningful for hunts.
    on_side = np.where(recent_regime != 0, net_z.values * recent_regime, 0.0)
    on_side = np.clip(on_side, 0.0, None)
    adverse = np.abs(spot_move.values)
    hunt_conf = on_side / (on_side + adverse + cfg.eps)
    hunt_conf = np.where(np.isin(verdict, HUNT_VERDICTS), hunt_conf, 0.0)

    out["regime"] = regime
    out["recent_regime"] = recent_regime
    out["regime_strength"] = regime_strength.values
    out["is_counter_move"] = is_counter
    out["hunt_verdict"] = verdict
    out["hunt_action"] = [_action_for(v) for v in verdict]
    out["hunt_confidence"] = np.clip(hunt_conf, 0.0, 1.0)
    return out


def hunt_signals(df: pd.DataFrame,
                 cfg: Optional[LiquidityHuntConfig] = None,
                 divergence_cfg: Optional[PremiumDivergenceConfig] = None,
                 *,
                 include_breaks: bool = True,
                 cooldown_bars: int = 0,
                 ) -> List[HuntSignal]:
    """Extract actionable hunt / break events.

    ``include_breaks`` controls whether REGIME_BREAK_* events are emitted
    alongside LIQUIDITY_HUNT_* (set False if you only want hunt entries).
    ``cooldown_bars`` collapses repeats of the same verdict within the
    window to one event.
    """
    cfg = cfg or LiquidityHuntConfig()
    enriched = df if "hunt_verdict" in df.columns else classify_liquidity_hunts(
        df, cfg, divergence_cfg)
    targets = list(HUNT_VERDICTS)
    if include_breaks:
        targets += list(BREAK_VERDICTS)
    out: List[HuntSignal] = []
    last_emit: Dict[str, int] = {}
    spot = pd.to_numeric(enriched["spot"], errors="coerce")
    for i in range(len(enriched)):
        v = enriched["hunt_verdict"].iloc[i]
        if v not in targets:
            continue
        if cooldown_bars > 0:
            prev = last_emit.get(v)
            if prev is not None and (i - prev) <= cooldown_bars:
                last_emit[v] = i
                continue
            last_emit[v] = i
        direction = 1 if v in (LIQUIDITY_HUNT_UP, REGIME_BREAK_UP) else -1
        out.append(HuntSignal(
            index=i,
            ts=enriched["ts"].iloc[i] if "ts" in enriched.columns else i,
            verdict=str(v),
            action=str(enriched["hunt_action"].iloc[i]),
            direction=direction,
            regime=int(enriched["recent_regime"].iloc[i]
                       if "recent_regime" in enriched.columns
                       else enriched["regime"].iloc[i]),
            spot=float(spot.iloc[i]),
            spot_move_norm=float(enriched["spot_move_norm"].iloc[i]),
            net_intent_z=float(enriched["net_intent_z"].iloc[i]),
            call_anomaly_z=float(enriched["call_resid_anomaly_z"].iloc[i]),
            put_anomaly_z=float(enriched["put_resid_anomaly_z"].iloc[i]),
            hunt_confidence=float(enriched["hunt_confidence"].iloc[i]),
        ))
    return out


def summarize_hunts(df: pd.DataFrame,
                    cfg: Optional[LiquidityHuntConfig] = None,
                    divergence_cfg: Optional[PremiumDivergenceConfig] = None,
                    ) -> Dict[str, Any]:
    """Per-session rollup of the hunt verdict distribution."""
    cfg = cfg or LiquidityHuntConfig()
    enriched = df if "hunt_verdict" in df.columns else classify_liquidity_hunts(
        df, cfg, divergence_cfg)
    counts = enriched["hunt_verdict"].value_counts().to_dict()
    n = int(len(enriched))
    return {
        "n_bars": n,
        "liquidity_hunt_up": int(counts.get(LIQUIDITY_HUNT_UP, 0)),
        "liquidity_hunt_down": int(counts.get(LIQUIDITY_HUNT_DOWN, 0)),
        "regime_break_down": int(counts.get(REGIME_BREAK_DOWN, 0)),
        "regime_break_up": int(counts.get(REGIME_BREAK_UP, 0)),
        "trend_intact_up": int(counts.get(TREND_INTACT_UP, 0)),
        "trend_intact_down": int(counts.get(TREND_INTACT_DOWN, 0)),
        "uncertain": int(counts.get(UNCERTAIN, 0)),
        "no_regime": int(counts.get(NO_REGIME, 0)),
    }
