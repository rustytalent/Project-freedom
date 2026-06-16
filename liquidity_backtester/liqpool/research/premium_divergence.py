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
#
# Tier ladder (founder's own framing):
#   Level 1 — basic divergence:           BULL_TRAP / BEAR_TRAP
#   Level 3 — anomaly + acceptance fail:  STRONG_BULL_TRAP / STRONG_BEAR_TRAP
#
# Only STRONG_* verdicts deserve real risk; the basic verdicts are kept so
# operators can see the underlying read and tune confirmation thresholds.
STRONG_BULL_TRAP = "STRONG_BULL_TRAP"  # Layer 1 + 2 + 3 agree: fade up-move, take put
STRONG_BEAR_TRAP = "STRONG_BEAR_TRAP"  # Layer 1 + 2 + 3 agree: fade down-move, take call
BULL_TRAP = "BULL_TRAP"            # Layer 1 only: spot up, money building puts → soft fade
BEAR_TRAP = "BEAR_TRAP"            # Layer 1 only: spot down, money building calls → soft fade
CONFIRMED_UP = "CONFIRMED_UP"      # spot up, money agrees → genuine
CONFIRMED_DOWN = "CONFIRMED_DOWN"  # spot down, money agrees → genuine
NEUTRAL = "NEUTRAL"                # no actionable read

# Convenience sets so callers can branch without enumerating each verdict.
TRAP_VERDICTS = (STRONG_BULL_TRAP, STRONG_BEAR_TRAP, BULL_TRAP, BEAR_TRAP)
STRONG_TRAP_VERDICTS = (STRONG_BULL_TRAP, STRONG_BEAR_TRAP)


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
    # Layer 2 — residual anomaly ("deviation of deviation"). The residual is
    # normalized against its OWN rolling distribution so the threshold is in
    # robust-σ units and transfers across strikes/instruments.
    residual_band_window: int = 60        # rolling window for normal residual band
    anomaly_threshold: float = 1.5        # min |residual_anomaly_z| (robust σ) to be "abnormal"
    # Layer 3 — fair-value rejection / acceptance failure ("touched ₹43,
    # accepted ₹44"). The contrarian leg wicked to its fair-value zone but
    # the market refused to hold it there for ``rejection_persistence_bars``.
    acceptance_window: int = 4            # rolling window for "accepted" price
    # Tolerances calibrated against the founder's ₹43/₹44 example: low wicks
    # within ~1% of fair (touch) and the accepted level holds ~2% above fair.
    # Both are fractions so they scale with strike — ₹0.5 on a ₹50 leg.
    rejection_touch_tolerance_frac: float = 0.010   # "touched fair" if low ≤ fair × (1 + tol)
    rejection_hold_tolerance_frac: float = 0.015    # "held above" if accepted ≥ fair × (1 + tol)
    rejection_persistence_bars: int = 2   # how many consecutive bars the rejection must hold
    call_delta_bounds: Tuple[float, float] = (0.01, 1.0)
    put_delta_bounds: Tuple[float, float] = (-1.0, -0.01)
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.delta_window < 3:
            raise ValueError("delta_window must be >= 3")
        if self.accumulation_window < 1:
            raise ValueError("accumulation_window must be >= 1")
        if self.residual_band_window < self.delta_window:
            raise ValueError("residual_band_window must be >= delta_window")
        if self.acceptance_window < 1:
            raise ValueError("acceptance_window must be >= 1")
        if self.rejection_persistence_bars < 1:
            raise ValueError("rejection_persistence_bars must be >= 1")


@dataclass(frozen=True)
class DivergenceSignal:
    """One actionable trap event.

    The three layered diagnostics are surfaced as fields so the operator can
    see exactly *why* a signal fired (or didn't promote to STRONG):

      Layer 1 — ``spot_move_norm`` + ``net_intent``  (basic divergence)
      Layer 2 — ``residual_anomaly_z``  (deviation of deviation on the
                                          contrarian leg, robust-σ units)
      Layer 3 — ``fair_value_rejected`` (boolean) + ``fair_price`` (the
                                          expected premium the market refused
                                          to accept)

    The ``tier`` field is the cleanest single read: ``"strong"`` only when
    all three layers agree.
    """
    index: int
    ts: Any
    verdict: str                      # one of {STRONG_,}BULL_TRAP / {STRONG_,}BEAR_TRAP
    tier: str                         # "strong" or "basic"
    direction: int                    # +1 expect spot up, -1 expect spot down
    leg: str                          # "put" (for bull trap) or "call" (for bear trap)
    spot: float
    spot_move_norm: float
    net_intent: float                 # Layer 1 — IV-cancelled directional intent (z)
    residual_anomaly_z: float         # Layer 2 — contrarian-leg residual anomaly (robust σ)
    fair_price: float                 # Layer 3 — expected premium for the contrarian leg
    fair_value_rejected: bool         # Layer 3 — touched fair then accepted higher
    entry_premium: float              # contrarian-leg premium right now
    entry_zone_low: float             # recent min of that premium (stabilization band)
    entry_zone_high: float            # recent max of that premium

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# IQR → robust σ. Ratio is 1.349 for a normal distribution; we use it to put
# the residual anomaly z-score on the same "standard deviations" scale as the
# net_intent z so thresholds across the module are comparable.
_IQR_TO_SIGMA: float = 1.349

# Cap on the residual anomaly z so a flatlined leg can't blow it up. Same
# discipline as _INTENT_Z_CLIP above.
_RESIDUAL_ANOMALY_Z_CLIP: float = 8.0


def _residual_anomaly_z(residual: pd.Series, window: int, eps: float) -> pd.Series:
    """Layer 2: residual of residual — "deviation of deviation".

    The residual itself drifts: typical residuals for a strike at this expiry,
    time-of-day, vol regime are nonzero. We measure how far the current
    residual sits from its OWN recent center, scaled by its own dispersion,
    so the answer is "is THIS residual abnormal relative to how abnormal
    residuals usually are here?" Median + IQR are used instead of mean + std
    because the residual is heavy-tailed (occasional fat spikes blow std up).
    """
    min_periods = max(8, window // 2)
    center = residual.rolling(window, min_periods=min_periods).median()
    q75 = residual.rolling(window, min_periods=min_periods).quantile(0.75)
    q25 = residual.rolling(window, min_periods=min_periods).quantile(0.25)
    robust_sigma = ((q75 - q25) / _IQR_TO_SIGMA).clip(lower=eps)
    z = (residual - center) / robust_sigma
    return z.clip(lower=-_RESIDUAL_ANOMALY_Z_CLIP,
                  upper=_RESIDUAL_ANOMALY_Z_CLIP).fillna(0.0)


def _detect_fair_value_rejection(premium: pd.Series, fair: pd.Series,
                                 acceptance_window: int,
                                 touch_tol_frac: float,
                                 hold_tol_frac: float,
                                 persistence_bars: int,
                                 ) -> pd.Series:
    """Layer 3: the founder's ₹43/₹44 pattern. The premium wicked down to
    its fair-value zone (``recent_low ≤ fair × (1 + touch_tol)``) but the
    market refuses to hold it there (``accepted_price ≥ fair × (1 + hold_tol)``)
    for ``persistence_bars`` consecutive bars.

    ``accepted_price`` is the rolling MEDIAN over the acceptance window —
    robust to a single wick (the founder's whole point: don't be fooled by
    a wick to ₹43 if the market accepts ₹44).
    """
    min_periods = max(2, acceptance_window // 2)
    recent_low = premium.rolling(acceptance_window, min_periods=min_periods).min()
    accepted = premium.rolling(acceptance_window, min_periods=min_periods).median()
    touched = recent_low <= fair * (1.0 + float(touch_tol_frac))
    held_above = accepted >= fair * (1.0 + float(hold_tol_frac))
    rejection_bar = touched & held_above
    # Persistence: require N consecutive rejection bars. Rolling sum and
    # compare to N is the cheapest way to express "every bar in the window
    # said yes."
    persistence = rejection_bar.rolling(persistence_bars, min_periods=persistence_bars).sum()
    return (persistence >= persistence_bars).fillna(False)


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

    # Layer 2 — residual anomaly per leg ("deviation of deviation").
    call_resid_anomaly_z = _residual_anomaly_z(
        call_resid, cfg.residual_band_window, cfg.eps)
    put_resid_anomaly_z = _residual_anomaly_z(
        put_resid, cfg.residual_band_window, cfg.eps)

    # Layer 3 — fair value per leg + rejection detector. The one-step-ahead
    # fair price IS the previous premium plus the delta-explained move; the
    # actual-minus-fair gap equals the per-bar residual we already computed.
    fair_call = call - call_resid
    fair_put = put - put_resid
    call_rejected = _detect_fair_value_rejection(
        call, fair_call,
        acceptance_window=cfg.acceptance_window,
        touch_tol_frac=cfg.rejection_touch_tolerance_frac,
        hold_tol_frac=cfg.rejection_hold_tolerance_frac,
        persistence_bars=cfg.rejection_persistence_bars,
    )
    put_rejected = _detect_fair_value_rejection(
        put, fair_put,
        acceptance_window=cfg.acceptance_window,
        touch_tol_frac=cfg.rejection_touch_tolerance_frac,
        hold_tol_frac=cfg.rejection_hold_tolerance_frac,
        persistence_bars=cfg.rejection_persistence_bars,
    )

    out["call_delta_eff"] = call_delta
    out["put_delta_eff"] = put_delta
    out["call_resid"] = call_resid
    out["put_resid"] = put_resid
    out["call_pressure"] = call_pressure
    out["put_pressure"] = put_pressure
    out["net_intent"] = net_intent
    out["net_intent_z"] = net_intent_z
    out["spot_move_norm"] = spot_move_norm
    out["call_resid_anomaly_z"] = call_resid_anomaly_z
    out["put_resid_anomaly_z"] = put_resid_anomaly_z
    out["fair_call"] = fair_call
    out["fair_put"] = fair_put
    out["call_fair_rejected"] = call_rejected
    out["put_fair_rejected"] = put_rejected

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

    # Verdict ladder.
    #   basic trap (Layer 1) — net intent disagrees with the move AND the
    #     contrarian leg is alive AND its accumulated pressure is positive.
    #   STRONG trap — basic trap AND contrarian leg residual anomaly clears
    #     ``anomaly_threshold`` (Layer 2) AND that leg has fair-value
    #     rejection sustained for ``rejection_persistence_bars`` (Layer 3).
    # Both legs must be alive (not pinned) for any trap to fire.
    valid_premium = ((call >= cfg.min_premium) & (put >= cfg.min_premium)
                     & call_active & put_active & warm)
    move_up = spot_move_norm > cfg.move_threshold
    move_dn = spot_move_norm < -cfg.move_threshold
    intent_bull = net_intent_z > cfg.intent_threshold
    intent_bear = net_intent_z < -cfg.intent_threshold
    puts_strong = put_pressure > 0.0
    calls_strong = call_pressure > 0.0

    # Layer 2 — the contrarian leg's residual must be ABNORMALLY large on
    # the side that signals the trap. For a bull trap (spot up + bearish
    # intent), puts are the contrarian leg, so puts' residual_anomaly_z
    # must be positive — puts are STRONGER than their own normal residual.
    puts_abnormally_strong = put_resid_anomaly_z > cfg.anomaly_threshold
    calls_abnormally_strong = call_resid_anomaly_z > cfg.anomaly_threshold

    basic_bull_trap = (move_up & intent_bear & puts_strong & valid_premium)
    basic_bear_trap = (move_dn & intent_bull & calls_strong & valid_premium)
    strong_bull_trap = basic_bull_trap & puts_abnormally_strong & put_rejected
    strong_bear_trap = basic_bear_trap & calls_abnormally_strong & call_rejected

    verdict = np.full(len(out), NEUTRAL, dtype=object)
    # Emit basic verdicts first, then promote to STRONG where Layer 2 + 3
    # agree. The order matters: np.where chains the most specific last.
    verdict = np.where(basic_bull_trap, BULL_TRAP, verdict)
    verdict = np.where(basic_bear_trap, BEAR_TRAP, verdict)
    verdict = np.where(move_up & intent_bull & valid_premium, CONFIRMED_UP, verdict)
    verdict = np.where(move_dn & intent_bear & valid_premium, CONFIRMED_DOWN, verdict)
    verdict = np.where(strong_bull_trap, STRONG_BULL_TRAP, verdict)
    verdict = np.where(strong_bear_trap, STRONG_BEAR_TRAP, verdict)
    out["verdict"] = verdict

    signal = np.zeros(len(out), dtype=int)
    signal = np.where(np.isin(out["verdict"].values, (BULL_TRAP, STRONG_BULL_TRAP)), -1, signal)
    signal = np.where(np.isin(out["verdict"].values, (BEAR_TRAP, STRONG_BEAR_TRAP)), +1, signal)
    out["signal"] = signal

    # Tier: "strong" only when both Layer 2 + Layer 3 agreed.
    tier = np.full(len(out), "", dtype=object)
    tier = np.where(np.isin(out["verdict"].values, TRAP_VERDICTS), "basic", tier)
    tier = np.where(np.isin(out["verdict"].values, STRONG_TRAP_VERDICTS), "strong", tier)
    out["tier"] = tier
    return out


def divergence_signals(df: pd.DataFrame,
                       cfg: Optional[PremiumDivergenceConfig] = None,
                       *,
                       cooldown_bars: int = 0,
                       strong_only: bool = False,
                       ) -> List[DivergenceSignal]:
    """Extract the actionable trap events from a frame.

    The frame may be raw (we compute the diagnostics) or already enriched
    by :func:`compute_divergence_frame` (we detect the added columns and
    skip recompute). Returns one :class:`DivergenceSignal` per BULL_TRAP /
    BEAR_TRAP / STRONG_BULL_TRAP / STRONG_BEAR_TRAP bar, carrying the
    layered diagnostics (Layer 1 spot move + intent, Layer 2 residual
    anomaly z, Layer 3 fair-value rejection + fair price).

    ``cooldown_bars`` suppresses repeat signals of the SAME verdict within
    that many bars of the last one — a single trap event typically spans
    many consecutive bars, and for trade extraction you want one entry per
    event, not one per bar. Default 0 emits every qualifying bar.

    ``strong_only=True`` filters to STRONG_BULL_TRAP / STRONG_BEAR_TRAP —
    the tier where all three layers agreed. Use this for live risk; use
    the unfiltered list for research / threshold tuning.
    """
    cfg = cfg or PremiumDivergenceConfig()
    enriched = df if "verdict" in df.columns else compute_divergence_frame(df, cfg)
    out: List[DivergenceSignal] = []
    call = pd.to_numeric(enriched["call_premium"], errors="coerce")
    put = pd.to_numeric(enriched["put_premium"], errors="coerce")
    band = cfg.accumulation_window
    target_verdicts = STRONG_TRAP_VERDICTS if strong_only else TRAP_VERDICTS

    def _cluster_key(v: str) -> str:
        return "bull" if v in (BULL_TRAP, STRONG_BULL_TRAP) else "bear"

    def _is_stronger(new_v: str, existing_v: str) -> bool:
        # Strong tier always beats basic; otherwise no replacement.
        return (new_v in STRONG_TRAP_VERDICTS and
                existing_v not in STRONG_TRAP_VERDICTS)

    # Index of the most recently emitted signal per cluster_key, so we can
    # upgrade the basic emission to STRONG when the strong tier fires later
    # within the same cooldown window.
    last_emit: Dict[str, int] = {}
    last_emit_pos: Dict[str, int] = {}     # position in `out`

    for i in range(len(enriched)):
        v = enriched["verdict"].iloc[i]
        if v not in target_verdicts:
            continue
        key = _cluster_key(v)
        if cooldown_bars > 0:
            prev = last_emit.get(key)
            if prev is not None and (i - prev) <= cooldown_bars:
                # Inside the cooldown window. Upgrade the prior emission if
                # this verdict is strictly stronger; otherwise drop it.
                if _is_stronger(v, out[last_emit_pos[key]].verdict):
                    out.pop(last_emit_pos[key])
                    # Shift any later last_emit_pos entries down by 1.
                    for k, pos in list(last_emit_pos.items()):
                        if pos > last_emit_pos[key]:
                            last_emit_pos[k] = pos - 1
                    # Fall through to emit the stronger one.
                else:
                    last_emit[key] = i      # extend the cluster
                    continue
            last_emit[key] = i
        ts = enriched["ts"].iloc[i] if "ts" in enriched.columns else i
        lo_idx = max(0, i - band + 1)
        if v in (BULL_TRAP, STRONG_BULL_TRAP):
            leg, direction = "put", -1
            prem = float(put.iloc[i])
            window = put.iloc[lo_idx:i + 1]
            anomaly_z = float(enriched["put_resid_anomaly_z"].iloc[i])
            fair_price = float(enriched["fair_put"].iloc[i])
            rejected = bool(enriched["put_fair_rejected"].iloc[i])
        else:
            leg, direction = "call", +1
            prem = float(call.iloc[i])
            window = call.iloc[lo_idx:i + 1]
            anomaly_z = float(enriched["call_resid_anomaly_z"].iloc[i])
            fair_price = float(enriched["fair_call"].iloc[i])
            rejected = bool(enriched["call_fair_rejected"].iloc[i])
        out.append(DivergenceSignal(
            index=i,
            ts=ts,
            verdict=str(v),
            tier=str(enriched["tier"].iloc[i]) if "tier" in enriched.columns else "basic",
            direction=direction,
            leg=leg,
            spot=float(pd.to_numeric(enriched["spot"], errors="coerce").iloc[i]),
            spot_move_norm=float(enriched["spot_move_norm"].iloc[i]),
            net_intent=float(enriched["net_intent_z"].iloc[i]),
            residual_anomaly_z=anomaly_z,
            fair_price=fair_price,
            fair_value_rejected=rejected,
            entry_premium=prem,
            entry_zone_low=float(window.min()),
            entry_zone_high=float(window.max()),
        ))
        last_emit_pos[key] = len(out) - 1
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
        "strong_bull_traps": int(counts.get(STRONG_BULL_TRAP, 0)),
        "strong_bear_traps": int(counts.get(STRONG_BEAR_TRAP, 0)),
        "bull_traps": int(counts.get(BULL_TRAP, 0)),
        "bear_traps": int(counts.get(BEAR_TRAP, 0)),
        "confirmed_up": int(counts.get(CONFIRMED_UP, 0)),
        "confirmed_down": int(counts.get(CONFIRMED_DOWN, 0)),
        "neutral": int(counts.get(NEUTRAL, 0)),
    }
