"""Winding-zone detector (Belief Engine, Phase 7).

The founder's exact concept: inside an unfinished candle or short
segment, the price has a reference X and excursions Y+/Y-. When price
reaches an excursion and pauses, that region is a *winding zone* — a
temporary auction zone where premium decides whether the move is
preparing continuation or reversal. The four types:

  * BULLISH_WINDING_UP   — spot near upper range, CE remains strong,
                            PE remains weak, spread clean, multi-strike
                            CE agreement, pullbacks do not damage CE.
                            Read: likely continuation upward.

  * BEARISH_WINDING_DOWN — mirror at lower range. Likely continuation down.

  * BULL_TRAP_WINDING    — spot near upper range BUT CE stops expanding,
                            PE refuses to fall, spread worsens, CE
                            residual fades. Read: up move may fail;
                            possible PE scalp.

  * BEAR_TRAP_WINDING    — mirror. Possible CE scalp.

This is the founder's entry/exit subsystem for scalping: the engine can
say "we are in a winding-up zone right now, here is whether to scalp a
call or take a put."

The detector consumes a recent *price track* (high/low/last across a
short bar window) plus the current Phase-5 battlefield snapshot and the
IV state. It does NOT need OHLC candles — only the recent excursions
of the live mark and where the current price sits relative to them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional
from collections import deque

from .battlefield import (
    BattlefieldSnapshot,
    FIELD_BEARISH_AGREEMENT,
    FIELD_BULLISH_AGREEMENT,
    FIELD_QUIET,
    FIELD_SINGLE_DISTORTION,
    FIELD_VOL_EXPANSION,
)
from .iv_state import (
    IVState,
    IV_DIRECTIONAL_BEAR,
    IV_DIRECTIONAL_BULL,
    IV_DIRTY_DATA,
    IV_LIQUIDITY_DISTORTION,
)


# Winding-zone labels.
BULLISH_WINDING_UP = "BULLISH_WINDING_UP"      # continuation up likely
BEARISH_WINDING_DOWN = "BEARISH_WINDING_DOWN"  # continuation down likely
BULL_TRAP_WINDING = "BULL_TRAP_WINDING"        # up move may fail → PE scalp
BEAR_TRAP_WINDING = "BEAR_TRAP_WINDING"        # down move may fail → CE scalp
NO_WINDING = "NO_WINDING"                      # not in a winding zone


# Scalp directions returned in the WindingZone payload.
SCALP_CE = "scalp_call"
SCALP_PE = "scalp_put"
SCALP_NONE = "none"


@dataclass
class WindingConfig:
    """Knobs for the winding-zone detector.

    The defaults target 1-min mark series with a ~15-bar reference window
    — the founder's "intra-candle on a 15-min frame" framing. Lower
    ``window_bars`` to bias toward short-segment / scalp use; raise it
    for slower swings."""
    window_bars: int = 15              # recent bars defining the X / Y+ / Y- band
    upper_zone_frac: float = 0.75      # last >= mid + 0.75 * (high - mid) → upper zone
    lower_zone_frac: float = 0.75      # mirror for lower zone
    min_range_pct: float = 0.0005      # band width as fraction of last; below = no zone
    pause_zscore: float = 0.4          # second-half drift / band width ratio; below
                                        # this counts as paused at the excursion. A fast
                                        # one-way move produces ratio ≈ 0.5 across an
                                        # evenly-rising series, so the threshold must
                                        # sit clearly below that.


@dataclass(frozen=True)
class WindingZone:
    """One detector reading."""
    zone: str
    scalp_direction: str
    confidence: float                  # [0,1]
    reference_x: float
    band_upper: float
    band_lower: float
    last: float
    upper_proximity: float             # 0..1, 1 = sitting on band_upper
    lower_proximity: float             # 0..1, 1 = sitting on band_lower
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class WindingDetector:
    """Stateful detector — feed one tick at a time, get a :class:`WindingZone`.

    The detector maintains its own rolling window of recent ticks. The
    founder's mid-candle survival is automatic: every ``observe`` call
    returns a snapshot the operator can read, with no requirement that
    bars be closed."""
    cfg: WindingConfig = field(default_factory=WindingConfig)
    _ticks: Deque[float] = field(default_factory=deque, init=False)

    def reset(self) -> None:
        self._ticks.clear()

    def observe(self, last_price: float,
                *,
                battlefield: Optional[BattlefieldSnapshot] = None,
                iv_state: Optional[IVState] = None,
                spread_friendly: bool = True,
                ) -> WindingZone:
        """Push one mark observation and return the current zone reading.

        ``battlefield`` + ``iv_state`` are the Phase-5 inputs — they tell
        the detector whether CE is "remaining strong" or "stops expanding"
        across the rail, and the IV-state branch is used to qualify the
        winding type. ``spread_friendly`` is True when execution is
        clean enough to consider a scalp; a worsening spread is one of
        the founder's bull-trap-winding signatures.
        """
        cfg = self.cfg
        self._ticks.append(float(last_price))
        while len(self._ticks) > cfg.window_bars:
            self._ticks.popleft()
        if len(self._ticks) < max(4, cfg.window_bars // 3):
            return WindingZone(
                zone=NO_WINDING, scalp_direction=SCALP_NONE, confidence=0.0,
                reference_x=float(last_price), band_upper=float(last_price),
                band_lower=float(last_price), last=float(last_price),
                upper_proximity=0.5, lower_proximity=0.5,
                note="not enough history",
            )

        values = list(self._ticks)
        hi = max(values)
        lo = min(values)
        mid = 0.5 * (hi + lo)
        last = float(last_price)
        rng = hi - lo
        if rng < cfg.min_range_pct * max(abs(last), 1.0):
            return WindingZone(
                zone=NO_WINDING, scalp_direction=SCALP_NONE, confidence=0.0,
                reference_x=mid, band_upper=hi, band_lower=lo, last=last,
                upper_proximity=0.5, lower_proximity=0.5,
                note="range too tight for winding",
            )

        # Proximity to each excursion in [0,1].
        upper_proximity = max(0.0, min(1.0, (last - mid) / (hi - mid + 1e-12)))
        lower_proximity = max(0.0, min(1.0, (mid - last) / (mid - lo + 1e-12)))

        # Paused-at-excursion check: the second-half drift over the
        # window should be small relative to the band width — a fast
        # one-way move is not yet a winding zone.
        half = len(values) // 2
        second_half_drift = abs(values[-1] - values[half])
        is_paused = (second_half_drift / (rng + 1e-12)) < cfg.pause_zscore

        in_upper_zone = upper_proximity >= cfg.upper_zone_frac and is_paused
        in_lower_zone = lower_proximity >= cfg.lower_zone_frac and is_paused

        if not (in_upper_zone or in_lower_zone):
            return WindingZone(
                zone=NO_WINDING, scalp_direction=SCALP_NONE, confidence=0.0,
                reference_x=mid, band_upper=hi, band_lower=lo, last=last,
                upper_proximity=upper_proximity, lower_proximity=lower_proximity,
                note="price not paused at an excursion",
            )

        # Battlefield + IV state qualify the winding type.
        # Bullish read: CE strong & PE weak (FIELD_BULLISH_AGREEMENT) +
        # IV state directional bull = positioning supports continuation.
        # Bull-trap read: CE stops expanding (battlefield NOT bullish) AND
        # spread getting worse (caller passes spread_friendly=False).
        bf_verdict = battlefield.verdict if battlefield is not None else FIELD_QUIET
        iv = iv_state.state if iv_state is not None else None

        if in_upper_zone:
            # Continuation candidate: bull battlefield + directional bull IV +
            # clean spread.
            if (bf_verdict == FIELD_BULLISH_AGREEMENT
                    and iv == IV_DIRECTIONAL_BULL
                    and spread_friendly):
                return WindingZone(
                    zone=BULLISH_WINDING_UP, scalp_direction=SCALP_CE,
                    confidence=min(1.0, (battlefield.confidence if battlefield else 0.5)
                                        * upper_proximity),
                    reference_x=mid, band_upper=hi, band_lower=lo, last=last,
                    upper_proximity=upper_proximity,
                    lower_proximity=lower_proximity,
                    note=("upper excursion paused; battlefield + IV bull, "
                          "clean spread → continuation up likely"),
                )
            # Bull-trap candidate: at upper excursion BUT positioning
            # disagrees with the up-move (battlefield bearish / single
            # distortion / vol expansion) OR the spread is worsening.
            if (bf_verdict in (FIELD_BEARISH_AGREEMENT, FIELD_VOL_EXPANSION,
                               FIELD_SINGLE_DISTORTION)
                    or not spread_friendly
                    or iv == IV_LIQUIDITY_DISTORTION):
                return WindingZone(
                    zone=BULL_TRAP_WINDING, scalp_direction=SCALP_PE,
                    confidence=min(1.0, upper_proximity * 0.8),
                    reference_x=mid, band_upper=hi, band_lower=lo, last=last,
                    upper_proximity=upper_proximity,
                    lower_proximity=lower_proximity,
                    note=("upper excursion paused; CE not extending or "
                          "spread worsening → PE scalp candidate"),
                )
            return WindingZone(
                zone=NO_WINDING, scalp_direction=SCALP_NONE, confidence=0.0,
                reference_x=mid, band_upper=hi, band_lower=lo, last=last,
                upper_proximity=upper_proximity, lower_proximity=lower_proximity,
                note="upper excursion paused but inputs ambiguous",
            )

        # in_lower_zone
        if (bf_verdict == FIELD_BEARISH_AGREEMENT
                and iv == IV_DIRECTIONAL_BEAR
                and spread_friendly):
            return WindingZone(
                zone=BEARISH_WINDING_DOWN, scalp_direction=SCALP_PE,
                confidence=min(1.0, (battlefield.confidence if battlefield else 0.5)
                                    * lower_proximity),
                reference_x=mid, band_upper=hi, band_lower=lo, last=last,
                upper_proximity=upper_proximity, lower_proximity=lower_proximity,
                note=("lower excursion paused; battlefield + IV bear, "
                      "clean spread → continuation down likely"),
            )
        if (bf_verdict in (FIELD_BULLISH_AGREEMENT, FIELD_VOL_EXPANSION,
                           FIELD_SINGLE_DISTORTION)
                or not spread_friendly
                or iv == IV_LIQUIDITY_DISTORTION):
            return WindingZone(
                zone=BEAR_TRAP_WINDING, scalp_direction=SCALP_CE,
                confidence=min(1.0, lower_proximity * 0.8),
                reference_x=mid, band_upper=hi, band_lower=lo, last=last,
                upper_proximity=upper_proximity, lower_proximity=lower_proximity,
                note=("lower excursion paused; PE not extending or "
                      "spread worsening → CE scalp candidate"),
            )
        return WindingZone(
            zone=NO_WINDING, scalp_direction=SCALP_NONE, confidence=0.0,
            reference_x=mid, band_upper=hi, band_lower=lo, last=last,
            upper_proximity=upper_proximity, lower_proximity=lower_proximity,
            note="lower excursion paused but inputs ambiguous",
        )
