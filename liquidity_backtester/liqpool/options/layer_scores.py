"""Pure functions that compute the six cascade layer scores.

Each function takes a clean kwargs interface, accepts ``None`` /
``NaN`` for missing inputs, and returns a float in ``[-1, +1]``.
Sign convention: positive = bullish on the underlying.

The functions are intentionally side-agnostic. The pre-trade
decision in :mod:`executor` flips direction at decision time; the
LightGBM monotone constraints flip directional features at fit
time per side.

These outputs are exactly the keys ``apply_causal_adjustments``
expects (``macro``, ``regime``, ``pool``, ``options``, ``micro``,
``manipulation``). Callers can either pass each computed score to
the CCV builder, or let
:func:`compute_layer_scores_from_inputs` build the full dict in one
call.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

from .macro_scrape import MacroOvernightSnapshot


# ---------------------------------------------------------------------------
# Utility — Tanh-based normalisation that caps near ±1
# ---------------------------------------------------------------------------

def _normalize_tanh(value: Optional[float], scale: float) -> float:
    """Map a raw signed magnitude to ``[-1, +1]`` via tanh.

    ``scale`` is the value at which the output reaches roughly
    ``tanh(1) ≈ 0.76``. Use ``None`` or ``NaN`` for missing — returns
    ``NaN`` so downstream averaging can skip it.
    """
    if value is None:
        return math.nan
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    if scale <= 0:
        return math.nan
    return math.tanh(float(value) / scale)


def _safe_mean(values: Iterable[float]) -> float:
    """Mean of non-NaN values. Returns 0 when every value is missing."""
    arr = [v for v in values if not (
        isinstance(v, float) and math.isnan(v))]
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ---------------------------------------------------------------------------
# Layer 1 — Macro
# ---------------------------------------------------------------------------

def macro_score(snapshot: Optional[MacroOvernightSnapshot] = None,
                *,
                spx_dod_pct: Optional[float] = None,
                dji_dod_pct: Optional[float] = None,
                ndx_dod_pct: Optional[float] = None,
                sgx_nifty_gap_pct: Optional[float] = None,
                usdinr_dod_pct: Optional[float] = None,
                vix_us_dod_points: Optional[float] = None,
                nifty_yday_close_pct: Optional[float] = None,
                ) -> float:
    """Compute the macro layer score.

    Either pass a :class:`MacroOvernightSnapshot` and let this
    function unpack it, or pass the individual fields directly. When
    BOTH are given, kwargs override the snapshot per field.

    The score is the equal-weight average of the available normalised
    components. Missing components are skipped, not zeroed — silent
    layers don't damp the average. When EVERY component is missing,
    the score is exactly 0 (which closes the cascade gate downstream).

    Sign convention:
      * SPX/DJI/NDX up → positive (bullish for ex-US too)
      * VIX-US up → negative (risk-off)
      * USDINR up → negative (rupee weak; capital outflow signal)
      * SGX-Nifty (or NIFTY yday close as proxy) up → positive

    Typical scales:
      SPX/DJI/NDX  : 0.50 %  (a ±0.5% close is meaningful)
      SGX-Nifty gap: 0.30 %
      USDINR DoD   : 0.20 %
      VIX-US DoD   : 1.50 points
    """
    if snapshot is not None:
        spx_dod_pct = (spx_dod_pct if spx_dod_pct is not None
                       else snapshot.spx_dod_pct)
        dji_dod_pct = (dji_dod_pct if dji_dod_pct is not None
                       else snapshot.dji_dod_pct)
        ndx_dod_pct = (ndx_dod_pct if ndx_dod_pct is not None
                       else snapshot.ndx_dod_pct)
        vix_us_dod_points = (vix_us_dod_points
                              if vix_us_dod_points is not None
                              else snapshot.vix_us_dod_points)
        nifty_yday_close_pct = (nifty_yday_close_pct
                                 if nifty_yday_close_pct is not None
                                 else snapshot.nifty_yday_close_pct)

    components = [
        _normalize_tanh(spx_dod_pct, 0.50),
        _normalize_tanh(dji_dod_pct, 0.50),
        _normalize_tanh(ndx_dod_pct, 0.50),
        _normalize_tanh(sgx_nifty_gap_pct, 0.30),
        _normalize_tanh(nifty_yday_close_pct, 0.50),
        # Negated sign: USDINR up → bearish
        -_normalize_tanh(usdinr_dod_pct, 0.20),
        # Negated sign: VIX up → bearish
        -_normalize_tanh(vix_us_dod_points, 1.50),
    ]
    return _clip(_safe_mean(components))


# ---------------------------------------------------------------------------
# Layer 2 — Index regime
# ---------------------------------------------------------------------------

def regime_score(*,
                  p_up: Optional[float] = None,
                  path_efficiency: Optional[float] = None,
                  direction_changes_30: Optional[float] = None,
                  underlying_30m_return: Optional[float] = None,
                  vol_regime_zscore_20d: Optional[float] = None,
                  ) -> float:
    """Index regime — how cleanly the underlying is trending RIGHT NOW.

    Preferred inputs are ``p_up`` (direction-model output in [0,1])
    + ``path_efficiency`` (run vs chop predictor). When the direction
    head's output isn't on the row, fall back to the intraday
    proxy ``underlying_30m_return``.

    ``vol_regime_zscore_20d`` modulates magnitude: in extreme-vol
    regimes the regime score gets damped because mean-reversion
    risk dominates trend signals.
    """
    components: list[float] = []
    if p_up is not None and not (isinstance(p_up, float)
                                  and math.isnan(p_up)):
        # 0.5 → 0; 0 → -1; 1 → +1
        direction = 2.0 * float(p_up) - 1.0
        # Multiply by path_efficiency (in [0, 1]) when available.
        if (path_efficiency is not None
                and not (isinstance(path_efficiency, float)
                          and math.isnan(path_efficiency))):
            components.append(direction * float(path_efficiency))
        else:
            components.append(direction)
    if underlying_30m_return is not None and not (
            isinstance(underlying_30m_return, float)
            and math.isnan(underlying_30m_return)):
        # 30-min return: ~0.5% is meaningful intraday.
        components.append(_normalize_tanh(
            float(underlying_30m_return) * 100, 0.50))
    if not components:
        return 0.0
    base = _safe_mean(components)
    # Vol-regime damping
    if (vol_regime_zscore_20d is not None
            and not (isinstance(vol_regime_zscore_20d, float)
                     and math.isnan(vol_regime_zscore_20d))):
        damp = max(0.4, 1.0 - 0.2 * abs(float(vol_regime_zscore_20d)))
        base *= damp
    return _clip(base)


# ---------------------------------------------------------------------------
# Layer 3 — Structural pool
# ---------------------------------------------------------------------------

def pool_score(*,
                proximity_p_60min: Optional[float] = None,
                pool_side_from_spot: Optional[str] = None,
                pool_strength: float = 1.0,
                ) -> float:
    """Score for the strike's nearest high-conviction pool.

    ``pool_side_from_spot``:
      * "below" — pool sits below current spot. Tested-from-above is
                  a demand bounce → bullish for the underlying.
      * "above" — pool sits above current spot. Tested-from-below is
                  a supply rejection → bearish for the underlying.

    ``proximity_p_60min`` (in ``[0, 1]``) sets magnitude; multiplied
    by ``pool_strength`` (defaults to 1.0). Both inputs default to
    silent → score 0.
    """
    if proximity_p_60min is None or pool_side_from_spot is None:
        return 0.0
    if (isinstance(proximity_p_60min, float)
            and math.isnan(proximity_p_60min)):
        return 0.0
    p = max(0.0, min(1.0, float(proximity_p_60min))) * float(pool_strength)
    if pool_side_from_spot == "below":
        return _clip(p)
    if pool_side_from_spot == "above":
        return _clip(-p)
    return 0.0


# ---------------------------------------------------------------------------
# Layer 4 — Options-specific
# ---------------------------------------------------------------------------

def options_score(*,
                   iv_percentile_60d: Optional[float] = None,
                   theta_per_day_pct: Optional[float] = None,
                   dte_trading_days: Optional[float] = None,
                   vega_per_volpoint_pct: Optional[float] = None,
                   ) -> float:
    """Options-specific context for the strike at this moment.

    The score is the side-agnostic "is this strike currently a
    favourable vehicle?" signal. The pre-trade gate combines it with
    macro direction; the LightGBM monotone constraints flip
    direction-sensitive features per side at fit time.

    For BUY thesis the inputs map:
      * iv_percentile high → bad (over-priced premium); → negative
      * theta_per_day_pct high → bad (theta drag); → negative
      * dte very short → bad (theta acceleration); → negative
      * vega > 0 → good (premium responds to vol moves); → positive
    """
    components: list[float] = []
    # IV percentile in [0, 1]; centre at 0.5 → map to [-1, +1] inverted.
    if iv_percentile_60d is not None and not (
            isinstance(iv_percentile_60d, float)
            and math.isnan(iv_percentile_60d)):
        components.append(-(2.0 * float(iv_percentile_60d) - 1.0))
    # Theta as % of premium: 0.05 (5%) is heavy decay → score about -1.
    if theta_per_day_pct is not None and not (
            isinstance(theta_per_day_pct, float)
            and math.isnan(theta_per_day_pct)):
        components.append(-_normalize_tanh(
            abs(float(theta_per_day_pct)), 0.05))
    # DTE: < 2 → strong negative, > 7 → strong positive.
    if dte_trading_days is not None and not (
            isinstance(dte_trading_days, float)
            and math.isnan(dte_trading_days)):
        components.append(_normalize_tanh(
            float(dte_trading_days) - 3.0, 4.0))
    # Vega: positive when nonzero — more is better for buy.
    if vega_per_volpoint_pct is not None and not (
            isinstance(vega_per_volpoint_pct, float)
            and math.isnan(vega_per_volpoint_pct)):
        components.append(_normalize_tanh(
            float(vega_per_volpoint_pct), 0.20))
    return _clip(_safe_mean(components))


# ---------------------------------------------------------------------------
# Layer 5 — Microstructure
# ---------------------------------------------------------------------------

def micro_score(*,
                 path_efficiency_30: Optional[float] = None,
                 direction_changes_30: Optional[float] = None,
                 underlying_5m_return: Optional[float] = None,
                 ema_stack_score: Optional[float] = None,
                 avwap_dev_sigma: Optional[float] = None,
                 ) -> float:
    """Bar-level structural read. Same convention as the others —
    positive = bullish.

    ``ema_stack_score`` and ``avwap_dev_sigma`` are placeholders for
    the user's own discretionary inputs. When omitted, the score is
    computed from the existing engine outputs we already have.
    """
    components: list[float] = []
    direction = math.nan
    if underlying_5m_return is not None and not (
            isinstance(underlying_5m_return, float)
            and math.isnan(underlying_5m_return)):
        direction = _normalize_tanh(
            float(underlying_5m_return) * 100, 0.30)
    # Build a "cleanness" factor in [0, 1]: path_efficiency high +
    # direction_changes low → clean trending bar.
    pe = (float(path_efficiency_30)
          if path_efficiency_30 is not None
             and not (isinstance(path_efficiency_30, float)
                       and math.isnan(path_efficiency_30))
          else 0.5)
    dc = (float(direction_changes_30)
          if direction_changes_30 is not None
             and not (isinstance(direction_changes_30, float)
                       and math.isnan(direction_changes_30))
          else 12.0)
    chop = max(0.0, min(1.0, dc / 30.0))
    cleanness = max(0.0, min(1.0, pe * (1.0 - chop)))
    if not (isinstance(direction, float) and math.isnan(direction)):
        components.append(direction * cleanness)
    if ema_stack_score is not None and not (
            isinstance(ema_stack_score, float)
            and math.isnan(ema_stack_score)):
        components.append(_clip(float(ema_stack_score)))
    if avwap_dev_sigma is not None and not (
            isinstance(avwap_dev_sigma, float)
            and math.isnan(avwap_dev_sigma)):
        # AVWAP deviation > +1σ → bullish; < -1σ → bearish.
        components.append(_normalize_tanh(
            float(avwap_dev_sigma), 1.0))
    if not components:
        return 0.0
    return _clip(_safe_mean(components))


# ---------------------------------------------------------------------------
# Layer 6 — Manipulation
# ---------------------------------------------------------------------------

def manipulation_score(*,
                        sweep_hi_detected: bool = False,
                        sweep_lo_detected: bool = False,
                        stop_run_hi_detected: bool = False,
                        stop_run_lo_detected: bool = False,
                        cum_delta_divergence_hi: bool = False,
                        cum_delta_divergence_lo: bool = False,
                        vw_swing_hi: bool = False,
                        vw_swing_lo: bool = False,
                        ) -> float:
    """Manipulation layer score.

    Inputs are boolean flags from the Stream C detectors:

      sweep_*    — wick-only sweep above/below a swing (hunting stops)
      stop_run_* — close-past-then-reclaim (the SMC sweep-and-go)
      cum_delta_divergence_* — cumulative-delta divergence at a swing
      vw_swing_* — volume-weighted swing

    Convention: a sweep ABOVE a swing high (``sweep_hi_detected``)
    suggests institutional flow is harvesting stops on the high side
    — i.e. positioning to push DOWN. So that signal flips the
    manipulation score NEGATIVE for the underlying. Mirror for lo.
    """
    pos = 0
    neg = 0
    if sweep_lo_detected: pos += 1
    if stop_run_lo_detected: pos += 1
    if cum_delta_divergence_lo: pos += 1
    if vw_swing_lo: pos += 1
    if sweep_hi_detected: neg += 1
    if stop_run_hi_detected: neg += 1
    if cum_delta_divergence_hi: neg += 1
    if vw_swing_hi: neg += 1
    total = pos + neg
    if total == 0:
        return 0.0
    return _clip((pos - neg) / 4.0)


# ---------------------------------------------------------------------------
# One-call wrapper
# ---------------------------------------------------------------------------

def compute_layer_scores_from_inputs(
    *,
    macro_snapshot: Optional[MacroOvernightSnapshot] = None,
    p_up: Optional[float] = None,
    path_efficiency_30: Optional[float] = None,
    direction_changes_30: Optional[float] = None,
    underlying_30m_return: Optional[float] = None,
    underlying_5m_return: Optional[float] = None,
    vol_regime_zscore_20d: Optional[float] = None,
    proximity_p_60min: Optional[float] = None,
    pool_side_from_spot: Optional[str] = None,
    pool_strength: float = 1.0,
    iv_percentile_60d: Optional[float] = None,
    theta_per_day_pct: Optional[float] = None,
    dte_trading_days: Optional[float] = None,
    vega_per_volpoint_pct: Optional[float] = None,
    ema_stack_score: Optional[float] = None,
    avwap_dev_sigma: Optional[float] = None,
    sweep_hi_detected: bool = False,
    sweep_lo_detected: bool = False,
    stop_run_hi_detected: bool = False,
    stop_run_lo_detected: bool = False,
    cum_delta_divergence_hi: bool = False,
    cum_delta_divergence_lo: bool = False,
    vw_swing_hi: bool = False,
    vw_swing_lo: bool = False,
    usdinr_dod_pct: Optional[float] = None,
    sgx_nifty_gap_pct: Optional[float] = None,
) -> Dict[str, float]:
    """Build the full ``{macro, regime, pool, options, micro,
    manipulation}`` dict that
    :func:`liqpool.options.ccv.apply_causal_adjustments` consumes.

    Each layer is computed independently. Missing inputs to a layer
    don't propagate to other layers — every layer that has enough
    inputs returns a real value; layers that don't return 0.
    """
    return {
        "macro": macro_score(
            snapshot=macro_snapshot,
            usdinr_dod_pct=usdinr_dod_pct,
            sgx_nifty_gap_pct=sgx_nifty_gap_pct,
        ),
        "regime": regime_score(
            p_up=p_up,
            path_efficiency=path_efficiency_30,
            direction_changes_30=direction_changes_30,
            underlying_30m_return=underlying_30m_return,
            vol_regime_zscore_20d=vol_regime_zscore_20d,
        ),
        "pool": pool_score(
            proximity_p_60min=proximity_p_60min,
            pool_side_from_spot=pool_side_from_spot,
            pool_strength=pool_strength,
        ),
        "options": options_score(
            iv_percentile_60d=iv_percentile_60d,
            theta_per_day_pct=theta_per_day_pct,
            dte_trading_days=dte_trading_days,
            vega_per_volpoint_pct=vega_per_volpoint_pct,
        ),
        "micro": micro_score(
            path_efficiency_30=path_efficiency_30,
            direction_changes_30=direction_changes_30,
            underlying_5m_return=underlying_5m_return,
            ema_stack_score=ema_stack_score,
            avwap_dev_sigma=avwap_dev_sigma,
        ),
        "manipulation": manipulation_score(
            sweep_hi_detected=sweep_hi_detected,
            sweep_lo_detected=sweep_lo_detected,
            stop_run_hi_detected=stop_run_hi_detected,
            stop_run_lo_detected=stop_run_lo_detected,
            cum_delta_divergence_hi=cum_delta_divergence_hi,
            cum_delta_divergence_lo=cum_delta_divergence_lo,
            vw_swing_hi=vw_swing_hi,
            vw_swing_lo=vw_swing_lo,
        ),
    }
