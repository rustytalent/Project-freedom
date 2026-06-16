"""Premium-vs-spot divergence — the "wrong-side strength" intention detector.

Origin: the founder's live observation on 2026-06-16 (expiry day, after a
losing streak that forced extra vigilance). The pattern, in his words:
when spot moves but the option that *should* be losing value refuses to
die, the move is "corrupted" — smart money is positioning against it and
a reversal is coming.

The mechanism, made precise:

  For an option, the premium change expected from a spot move ΔS is
  ``ΔP_expected = δ · ΔS`` where δ is the option's delta (positive for
  calls, negative for puts). The *realized* premium change ΔP_actual
  differs from that by a residual:

      D = ΔP_actual − δ · ΔS

  D is the part of the premium move NOT explained by direction. It is
  driven by (a) implied-vol changes (vega) and (b) order-flow pressure —
  people actively buying/selling that specific option. The founder's
  edge is reading (b): if during an up-move the PUT residual is positive
  (puts holding up / recovering when delta says they should fall), the
  money is accumulating downside — the up-move is a bull trap.

The confound this module removes:
  A general IV spike lifts BOTH call and put residuals together (vega is
  the same sign for both). To isolate *directional* positioning we take
  the same-strike call residual minus put residual — the common-mode IV
  component cancels and what survives is net directional intent:

      net_intent = call_pressure − put_pressure

  call_pressure / put_pressure are the residuals accumulated over a few
  candles (the founder's "cumulative over 5-6 candles"). Positive
  net_intent = money building calls = bullish; negative = building puts
  = bearish.

The 2×2 verdict (spot move × net intent):

      up  + bullish  → CONFIRMED_UP    (genuine — ride it)
      up  + bearish  → BULL_TRAP       (fade — take the put)
      down + bearish → CONFIRMED_DOWN  (genuine — ride it)
      down + bullish → BEAR_TRAP       (fade — take the call)

THESIS: option order-flow leads spot at the turn. The residual on the
side that "should" be dying reveals accumulation the tape hasn't printed
yet.

KILL CONDITION: if BULL_TRAP / BEAR_TRAP signals do not precede a mean
reversal with positive expectancy after costs over an OOS window, the
divergence is noise — kill it via the harness / kill_switch like any
other hypothesis.

REGIME WHERE IT FAILS: strong one-way trend days where genuine momentum
keeps the contrarian premium bid for real reasons (fresh hedging flow,
not a trap). The CONFIRMED_* verdicts are the guard — only fade when
move and intent DISAGREE.

Everything here is strictly causal: the value at bar T uses only data
with timestamp ≤ T (rolling windows include the current bar and look
backward only). Safe to compute live, bar by bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Numerical-robustness constants. Option legs routinely flatline at a price
# floor (a deep-OTM leg pinned at ₹0.5-1.0 barely moves). Without these
# guards a near-zero change-vol denominator turns a tiny residual into an
# astronomical normalized value and the z-score explodes. These bound the
# pipeline at every division.
_RESIDUAL_NORM_CLIP = 6.0          # max |per-bar normalized residual| (std units)
_INTENT_STD_FLOOR = 0.5            # min std for the net_intent z-score denominator
_INTENT_Z_CLIP = 12.0             # max |net_intent z-score| reported / acted on

# Verdict labels.
BULL_TRAP = "BULL_TRAP"            # spot up, money building puts → fade, take put
BEAR_TRAP = "BEAR_TRAP"            # spot down, money building calls → fade, take call
CONFIRMED_UP = "CONFIRMED_UP"      # spot up, money agrees → genuine
CONFIRMED_DOWN = "CONFIRMED_DOWN"  # spot down, money agrees → genuine
NEUTRAL = "NEUTRAL"                # no actionable read


@dataclass
class PremiumDivergenceConfig:
    """Knobs for the divergence detector.

    Defaults target 1-5 minute NIFTY option bars. ``delta_window`` is the
    look-back for the model-free effective-delta estimate;
    ``accumulation_window`` is how many candles the residual pressure is
    summed over (the founder's "3-6 candles"). Thresholds are in units of
    the normalized (z-scored) pressure / spot move so they transfer
    across strikes and vol regimes."""
    delta_window: int = 20
    accumulation_window: int = 5
    move_threshold: float = 1.2       # min |normalized spot move| (std units) to call it a move
    intent_threshold: float = 1.5     # min |net_intent z-score| (std units) to act
    min_premium: float = 0.5          # ignore near-zero premiums (deep-OTM noise)
    min_leg_activity_frac: float = 0.01  # a leg's change-std must exceed this × its
                                          # premium to count as "alive" — a leg pinned
                                          # at its price floor isn't holding up, it's dead
    call_delta_bounds: Tuple[float, float] = (0.01, 1.0)
    put_delta_bounds: Tuple[float, float] = (-1.0, -0.01)
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.delta_window < 3:
            raise ValueError("delta_window must be >= 3")
        if self.accumulation_window < 1:
            raise ValueError("accumulation_window must be >= 1")


@dataclass(frozen=True)
class DivergenceSignal:
    """One actionable trap event."""
    index: int
    ts: Any
    verdict: str                      # BULL_TRAP or BEAR_TRAP
    direction: int                    # +1 expect spot up, -1 expect spot down
    leg: str                          # "put" (for bull trap) or "call" (for bear trap)
    spot: float
    spot_move_norm: float
    net_intent: float
    entry_premium: float              # contrarian-leg premium right now
    entry_zone_low: float             # recent min of that premium (stabilization band)
    entry_zone_high: float            # recent max of that premium

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _rolling_effective_delta(prem_change: pd.Series, spot_change: pd.Series,
                             window: int, bounds: Tuple[float, float],
                             eps: float) -> pd.Series:
    """Model-free effective delta = rolling Cov(ΔP, ΔS) / Var(ΔS), clipped
    to the leg's plausible delta range.

    Clipping by leg type (call ∈ [0.01, 1], put ∈ [-1, -0.01]) keeps the
    residual interpretable even when a noisy window would otherwise flip
    the sign — a put's delta is structurally negative regardless of what
    a short noisy sample says."""
    min_periods = max(3, window // 2)
    cov = prem_change.rolling(window, min_periods=min_periods).cov(spot_change)
    var = spot_change.rolling(window, min_periods=min_periods).var()
    beta = cov / (var + eps)
    beta = beta.clip(lower=bounds[0], upper=bounds[1])
    # Before enough data accrues, fall back to the midpoint of the bounds
    # (an ATM-ish delta) so early bars don't produce wild residuals.
    fallback = (bounds[0] + bounds[1]) / 2.0
    return beta.fillna(fallback)


def _accumulated_pressure(residual: pd.Series, prem_change: pd.Series,
                          accum_window: int, vol_window: int,
                          eps: float) -> pd.Series:
    """Normalize the per-bar residual by the premium's own recent change-
    volatility (so a ₹0.3 residual on a calm option counts as much as a
    ₹3 residual on a wild one), then sum over the accumulation window."""
    vol = prem_change.rolling(vol_window, min_periods=max(3, vol_window // 2)).std()
    residual_norm = residual / (vol + eps)
    # A flatlined leg has vol≈0, which would turn a tiny residual into a huge
    # normalized value. Clip per-bar so one degenerate bar can't dominate the
    # accumulated pressure — anything beyond a few std is already "extreme".
    residual_norm = residual_norm.clip(lower=-_RESIDUAL_NORM_CLIP,
                                       upper=_RESIDUAL_NORM_CLIP).fillna(0.0)
    return residual_norm.rolling(accum_window, min_periods=1).sum()


def compute_divergence_frame(df: pd.DataFrame,
                             cfg: Optional[PremiumDivergenceConfig] = None,
                             ) -> pd.DataFrame:
    """Build the full per-bar divergence diagnostic frame.

    ``df`` must have columns: ``spot``, ``call_premium``, ``put_premium``.
    An optional ``ts`` column is carried through for signal timestamps.
    Optional ``call_delta`` / ``put_delta`` columns override the model-free
    delta estimate (use these if you have a pricing-model delta).

    Returns a copy of ``df`` with added columns:
      * ``call_resid`` / ``put_resid`` — per-bar delta residuals (₹)
      * ``call_pressure`` / ``put_pressure`` — accumulated normalized residual
      * ``net_intent`` — call_pressure − put_pressure (IV-cancelled directional read)
      * ``spot_move_norm`` — normalized accumulated spot move
      * ``verdict`` — one of BULL_TRAP / BEAR_TRAP / CONFIRMED_* / NEUTRAL
      * ``signal`` — +1 (expect up), −1 (expect down), 0 (none)
    """
    cfg = cfg or PremiumDivergenceConfig()
    required = {"spot", "call_premium", "put_premium"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_divergence_frame missing columns: {sorted(missing)}")

    out = df.copy().reset_index(drop=True)
    spot = pd.to_numeric(out["spot"], errors="coerce")
    call = pd.to_numeric(out["call_premium"], errors="coerce")
    put = pd.to_numeric(out["put_premium"], errors="coerce")

    d_spot = spot.diff()
    d_call = call.diff()
    d_put = put.diff()

    # Effective deltas (or caller-supplied overrides).
    if "call_delta" in out.columns:
        call_delta = pd.to_numeric(out["call_delta"], errors="coerce").clip(
            *cfg.call_delta_bounds)
    else:
        call_delta = _rolling_effective_delta(
            d_call, d_spot, cfg.delta_window, cfg.call_delta_bounds, cfg.eps)
    if "put_delta" in out.columns:
        put_delta = pd.to_numeric(out["put_delta"], errors="coerce").clip(
            *cfg.put_delta_bounds)
    else:
        put_delta = _rolling_effective_delta(
            d_put, d_spot, cfg.delta_window, cfg.put_delta_bounds, cfg.eps)

    # Per-bar residuals: actual premium change minus delta-explained change.
    call_resid = d_call - call_delta * d_spot
    put_resid = d_put - put_delta * d_spot

    # Accumulated, vol-normalized pressure per leg.
    call_pressure = _accumulated_pressure(
        call_resid, d_call, cfg.accumulation_window, cfg.delta_window, cfg.eps)
    put_pressure = _accumulated_pressure(
        put_resid, d_put, cfg.accumulation_window, cfg.delta_window, cfg.eps)

    # Net directional intent — IV common-mode cancels in the difference.
    net_intent = call_pressure - put_pressure
    # Z-score net_intent over a longer window so the threshold is in std
    # units and transfers across strikes / vol regimes.
    long_window = cfg.delta_window * 3
    intent_std = net_intent.rolling(
        long_window, min_periods=cfg.delta_window).std()
    # Floor the denominator: when net_intent is nearly constant (both legs
    # calm or flatlined) the raw std collapses and the z-score would blow up.
    # The floor is a fixed std-unit value because the upstream residuals are
    # already standardized, so it transfers across strikes/instruments.
    intent_std = intent_std.clip(lower=_INTENT_STD_FLOOR).fillna(_INTENT_STD_FLOOR)
    net_intent_z = (net_intent / intent_std).clip(
        lower=-_INTENT_Z_CLIP, upper=_INTENT_Z_CLIP).fillna(0.0)

    # Normalized accumulated spot move over the same window.
    spot_move = spot.diff(cfg.accumulation_window)
    spot_step_vol = d_spot.rolling(
        cfg.delta_window, min_periods=max(3, cfg.delta_window // 2)).std()
    spot_move_norm = spot_move / (
        spot_step_vol * np.sqrt(cfg.accumulation_window) + cfg.eps)

    out["call_delta_eff"] = call_delta
    out["put_delta_eff"] = put_delta
    out["call_resid"] = call_resid
    out["put_resid"] = put_resid
    out["call_pressure"] = call_pressure
    out["put_pressure"] = put_pressure
    out["net_intent"] = net_intent
    out["net_intent_z"] = net_intent_z
    out["spot_move_norm"] = spot_move_norm

    # Warmup mask — the first (delta_window + accumulation_window) bars don't
    # have enough history for stable residuals / z-scores.
    warm = pd.Series(np.arange(len(out)) >= (cfg.delta_window + cfg.accumulation_window),
                     index=out.index)

    # Activity guard: a leg pinned at its price floor (deep-OTM ₹1 lottery
    # ticket) has near-zero change-vol — its "holding up" is pinning, not
    # accumulation. Require each leg's recent change-std to exceed a fraction
    # of its own premium before we read positioning from it.
    call_step_std = d_call.rolling(
        cfg.delta_window, min_periods=max(3, cfg.delta_window // 2)).std()
    put_step_std = d_put.rolling(
        cfg.delta_window, min_periods=max(3, cfg.delta_window // 2)).std()
    call_active = call_step_std > cfg.min_leg_activity_frac * call
    put_active = put_step_std > cfg.min_leg_activity_frac * put

    # Verdict + signal. A trap requires BOTH net intent disagreeing with the
    # move AND the contrarian leg's pressure actually being positive — the
    # founder's literal read is the wrong-side option *holding up*, not merely
    # the near-side weakening — AND both legs being alive (not pinned).
    valid_premium = ((call >= cfg.min_premium) & (put >= cfg.min_premium)
                     & call_active & put_active & warm)
    move_up = spot_move_norm > cfg.move_threshold
    move_dn = spot_move_norm < -cfg.move_threshold
    intent_bull = net_intent_z > cfg.intent_threshold
    intent_bear = net_intent_z < -cfg.intent_threshold
    puts_strong = put_pressure > 0.0
    calls_strong = call_pressure > 0.0

    verdict = np.full(len(out), NEUTRAL, dtype=object)
    verdict = np.where(move_up & intent_bear & puts_strong & valid_premium,
                       BULL_TRAP, verdict)
    verdict = np.where(move_dn & intent_bull & calls_strong & valid_premium,
                       BEAR_TRAP, verdict)
    verdict = np.where(move_up & intent_bull & valid_premium, CONFIRMED_UP, verdict)
    verdict = np.where(move_dn & intent_bear & valid_premium, CONFIRMED_DOWN, verdict)
    out["verdict"] = verdict

    signal = np.zeros(len(out), dtype=int)
    signal = np.where(out["verdict"].values == BULL_TRAP, -1, signal)
    signal = np.where(out["verdict"].values == BEAR_TRAP, +1, signal)
    out["signal"] = signal
    return out


def divergence_signals(df: pd.DataFrame,
                       cfg: Optional[PremiumDivergenceConfig] = None,
                       *,
                       cooldown_bars: int = 0,
                       ) -> List[DivergenceSignal]:
    """Extract the actionable trap events from a frame.

    The frame may be raw (we compute the diagnostics) or already enriched
    by :func:`compute_divergence_frame` (we detect the added columns and
    skip recompute). Returns one :class:`DivergenceSignal` per BULL_TRAP /
    BEAR_TRAP bar, with the contrarian-leg premium and its recent
    stabilization band so the operator knows the entry zone.

    ``cooldown_bars`` suppresses repeat signals of the SAME verdict within
    that many bars of the last one — a single trap event typically spans
    many consecutive bars, and for trade extraction you want one entry per
    event, not one per bar. Default 0 emits every qualifying bar.
    """
    cfg = cfg or PremiumDivergenceConfig()
    enriched = df if "verdict" in df.columns else compute_divergence_frame(df, cfg)
    out: List[DivergenceSignal] = []
    call = pd.to_numeric(enriched["call_premium"], errors="coerce")
    put = pd.to_numeric(enriched["put_premium"], errors="coerce")
    band = cfg.accumulation_window
    last_emit: Dict[str, int] = {}
    for i in range(len(enriched)):
        v = enriched["verdict"].iloc[i]
        if v not in (BULL_TRAP, BEAR_TRAP):
            continue
        if cooldown_bars > 0:
            prev = last_emit.get(v)
            if prev is not None and (i - prev) <= cooldown_bars:
                last_emit[v] = i      # extend the cluster, but don't emit
                continue
            last_emit[v] = i
        ts = enriched["ts"].iloc[i] if "ts" in enriched.columns else i
        lo_idx = max(0, i - band + 1)
        if v == BULL_TRAP:
            # Fade the up-move → take the PUT (the contrarian leg).
            leg, direction = "put", -1
            prem = float(put.iloc[i])
            window = put.iloc[lo_idx:i + 1]
        else:
            leg, direction = "call", +1
            prem = float(call.iloc[i])
            window = call.iloc[lo_idx:i + 1]
        out.append(DivergenceSignal(
            index=i,
            ts=ts,
            verdict=str(v),
            direction=direction,
            leg=leg,
            spot=float(pd.to_numeric(enriched["spot"], errors="coerce").iloc[i]),
            spot_move_norm=float(enriched["spot_move_norm"].iloc[i]),
            net_intent=float(enriched["net_intent"].iloc[i]),
            entry_premium=prem,
            entry_zone_low=float(window.min()),
            entry_zone_high=float(window.max()),
        ))
    return out


def divergence_exit_reason(df: pd.DataFrame, position_dir: int,
                           cfg: Optional[PremiumDivergenceConfig] = None,
                           ) -> Optional[str]:
    """Given an enriched frame and the direction of an open position
    (+1 long-the-reversal-up i.e. holding a call, −1 holding a put),
    return an exit reason for the LAST bar, or ``None`` to hold.

    Exit when the edge that put you in the trade stops confirming:
      * net_intent crosses back through zero against your position
        (the money that was building your side has left), or
      * the spot move has now caught up to the intent (the reversal
        already happened — ``spot_move_norm`` now agrees with your
        direction beyond the move_threshold, so the easy part is over).
    """
    cfg = cfg or PremiumDivergenceConfig()
    enriched = df if "net_intent_z" in df.columns else compute_divergence_frame(df, cfg)
    if enriched.empty:
        return None
    net_intent = float(enriched["net_intent_z"].iloc[-1])
    spot_move_norm = float(enriched["spot_move_norm"].iloc[-1])
    if position_dir == -1:           # holding a put, expecting spot DOWN
        if net_intent > 0:
            return "net_intent flipped bullish — put accumulation has left"
        if spot_move_norm < -cfg.move_threshold:
            return "spot already moved down — reversal realized, take profit"
    elif position_dir == +1:         # holding a call, expecting spot UP
        if net_intent < 0:
            return "net_intent flipped bearish — call accumulation has left"
        if spot_move_norm > cfg.move_threshold:
            return "spot already moved up — reversal realized, take profit"
    return None


def summarize_divergence(df: pd.DataFrame,
                         cfg: Optional[PremiumDivergenceConfig] = None,
                         ) -> Dict[str, Any]:
    """Compact rollup of verdict counts + signal density for a session."""
    cfg = cfg or PremiumDivergenceConfig()
    enriched = df if "verdict" in df.columns else compute_divergence_frame(df, cfg)
    counts = enriched["verdict"].value_counts().to_dict()
    n = int(len(enriched))
    n_signals = int((enriched["signal"] != 0).sum())
    return {
        "n_bars": n,
        "n_signals": n_signals,
        "signal_rate": (n_signals / n) if n else 0.0,
        "bull_traps": int(counts.get(BULL_TRAP, 0)),
        "bear_traps": int(counts.get(BEAR_TRAP, 0)),
        "confirmed_up": int(counts.get(CONFIRMED_UP, 0)),
        "confirmed_down": int(counts.get(CONFIRMED_DOWN, 0)),
        "neutral": int(counts.get(NEUTRAL, 0)),
    }
