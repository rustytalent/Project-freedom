"""Live research models — six minimal, honest stubs of the brain the
founder wants on screen.

These are NOT the production models. They are the contracts:

  ReactionModel       — direction & magnitude of recent move
  ProximityModel      — distance to recent supply / demand
  LiquidityModel      — sweep + reclaim micro-pattern
  QualityModel        — trend coherence (signal/noise)
  PostReactionModel   — does the move have continuation?
  ManipulationModel   — fast spike-and-reclaim warning

Each model consumes a recent spot tick history and returns either a
``ModelSignal`` to publish or ``None`` (silent — nothing to say). The
math is deliberately simple so the contract is testable end-to-end;
heavier scientists from ``liquidity_backtester`` can be dropped in via
the same ``tick(history) -> Optional[ModelSignal]`` shape later.

The shape mirrors Codex's DecisionEvent: every signal carries
reason_codes, a confidence in [0,1], an optional zone, and an
invalidation condition. That's the only contract the publisher cares
about.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .io_decl import IOSpec, declare
from .live_publisher import ModelSignal, now_ist_hms


@dataclass(frozen=True)
class Tick:
    ts: float           # monotonic seconds
    spot: float


# ---------------------------------------------------------------------------
# 1. ReactionModel — "the market just reacted"
# ---------------------------------------------------------------------------

class ReactionModel:
    """Detects a rapid spot move in the most recent window. Emits an
    'upside reaction' or 'downside reaction' signal scaled by how many
    standard deviations the move is vs the longer history."""

    name = "reaction_model"

    def __init__(self, short: int = 5, long: int = 60,
                 sigma_threshold: float = 1.4) -> None:
        self.short, self.long, self.sigma = short, long, sigma_threshold

    def tick(self, hist: Sequence[Tick], asset: str = "NIFTY"
             ) -> Optional[ModelSignal]:
        if len(hist) < self.long:
            return None
        rec = [t.spot for t in hist[-self.short:]]
        ref = [t.spot for t in hist[-self.long:]]
        move = rec[-1] - rec[0]
        rets = [b - a for a, b in zip(ref, ref[1:])]
        sd = statistics.pstdev(rets) or 1e-9
        z = move / (sd * (self.short ** 0.5))
        if abs(z) < self.sigma:
            return None
        direction = "upside_reaction" if z > 0 else "downside_reaction"
        conf = min(0.95, abs(z) / 3.0)
        signal = ("upside reaction in progress"
                  if z > 0 else "downside reaction in progress")
        spot = rec[-1]
        zone = (spot - sd, spot + sd)
        risk = (f"fails if spot closes back below {rec[0]:.0f}"
                if z > 0 else f"fails if spot closes back above {rec[0]:.0f}")
        return ModelSignal(
            ts_ist=now_ist_hms(), asset=asset, model=self.name,
            signal=signal, confidence=round(conf, 3),
            trust_tier="LOGGED",
            zone=zone, risk=risk,
            reason_codes=[direction, f"z={z:+.2f}σ", f"move={move:+.1f}pts"],
            extras={"move_pts": round(move, 2), "z": round(z, 2)},
        )


# ---------------------------------------------------------------------------
# 2. ProximityModel — "near a level the market cares about"
# ---------------------------------------------------------------------------

class ProximityModel:
    """How close is current spot to the recent rolling high or low? When
    inside one-quarter of an ATR's distance from either, fire."""

    name = "proximity_model"

    def __init__(self, lookback: int = 120, proximity_atr: float = 0.25) -> None:
        self.lookback, self.proximity_atr = lookback, proximity_atr

    def tick(self, hist: Sequence[Tick], asset: str = "NIFTY"
             ) -> Optional[ModelSignal]:
        if len(hist) < self.lookback:
            return None
        window = [t.spot for t in hist[-self.lookback:]]
        hi, lo = max(window), min(window)
        atr = (hi - lo)
        if atr < 1e-6:
            return None
        spot = window[-1]
        d_hi = (hi - spot) / atr
        d_lo = (spot - lo) / atr
        if d_hi < self.proximity_atr:
            conf = min(0.9, 0.5 + (self.proximity_atr - d_hi) * 2)
            return ModelSignal(
                ts_ist=now_ist_hms(), asset=asset, model=self.name,
                signal=f"approaching session high {hi:.0f}",
                confidence=round(conf, 3), trust_tier="LOGGED",
                zone=(hi - 0.1 * atr, hi + 0.05 * atr),
                risk="fails on clean break above with sustained volume",
                reason_codes=["near_high", "supply_zone"],
                extras={"distance_atr": round(d_hi, 3), "high": round(hi, 1)},
            )
        if d_lo < self.proximity_atr:
            conf = min(0.9, 0.5 + (self.proximity_atr - d_lo) * 2)
            return ModelSignal(
                ts_ist=now_ist_hms(), asset=asset, model=self.name,
                signal=f"approaching session low {lo:.0f}",
                confidence=round(conf, 3), trust_tier="LOGGED",
                zone=(lo - 0.05 * atr, lo + 0.1 * atr),
                risk="fails on clean break below with sustained volume",
                reason_codes=["near_low", "demand_zone"],
                extras={"distance_atr": round(d_lo, 3), "low": round(lo, 1)},
            )
        return None


# ---------------------------------------------------------------------------
# 3. LiquidityModel — "sweep + fast reclaim" microstructure
# ---------------------------------------------------------------------------

class LiquidityModel:
    """Spike beyond recent extreme followed by a fast reclaim into the
    range is the classic liquidity sweep — stops got hunted, price
    came back. Standard prop-desk read."""

    name = "liquidity_model"

    def __init__(self, lookback: int = 40, spike_atr: float = 0.35,
                 reclaim_within: int = 6) -> None:
        self.lookback, self.spike_atr, self.reclaim_within = (
            lookback, spike_atr, reclaim_within)

    def tick(self, hist: Sequence[Tick], asset: str = "NIFTY"
             ) -> Optional[ModelSignal]:
        if len(hist) < self.lookback + self.reclaim_within:
            return None
        window = [t.spot for t in hist[-self.lookback:-self.reclaim_within]]
        recent = [t.spot for t in hist[-self.reclaim_within:]]
        atr = max(window) - min(window) or 1e-6
        prior_hi, prior_lo = max(window), min(window)
        peak_recent = max(recent)
        trough_recent = min(recent)
        now = recent[-1]
        # upside sweep: poked above prior_hi by > spike_atr*atr, then came back inside
        if (peak_recent - prior_hi) > self.spike_atr * atr and now < prior_hi:
            depth = (peak_recent - prior_hi) / atr
            return ModelSignal(
                ts_ist=now_ist_hms(), asset=asset, model=self.name,
                signal=f"upside liquidity sweep + reclaim "
                       f"(swept {peak_recent:.0f}, back inside {prior_hi:.0f})",
                confidence=round(min(0.92, 0.55 + depth * 0.8), 3),
                trust_tier="LOGGED",
                zone=(prior_hi - 0.1 * atr, peak_recent),
                risk="fails if spot makes new high without giving back",
                reason_codes=["liquidity_sweep", "fast_reclaim", "stop_hunt"],
                extras={"sweep_depth_atr": round(depth, 3)},
            )
        if (prior_lo - trough_recent) > self.spike_atr * atr and now > prior_lo:
            depth = (prior_lo - trough_recent) / atr
            return ModelSignal(
                ts_ist=now_ist_hms(), asset=asset, model=self.name,
                signal=f"downside liquidity sweep + reclaim "
                       f"(swept {trough_recent:.0f}, back inside {prior_lo:.0f})",
                confidence=round(min(0.92, 0.55 + depth * 0.8), 3),
                trust_tier="LOGGED",
                zone=(trough_recent, prior_lo + 0.1 * atr),
                risk="fails if spot makes new low without giving back",
                reason_codes=["liquidity_sweep", "fast_reclaim", "stop_hunt"],
                extras={"sweep_depth_atr": round(depth, 3)},
            )
        return None


# ---------------------------------------------------------------------------
# 4. QualityModel — trend coherence
# ---------------------------------------------------------------------------

class QualityModel:
    """Is the move coherent (signal) or noisy (chop)? Ratio of net move
    over the lookback to the path length is the classic 'efficiency
    ratio' (Kaufman 1995)."""

    name = "quality_model"

    def __init__(self, lookback: int = 30) -> None:
        self.lookback = lookback

    def tick(self, hist: Sequence[Tick], asset: str = "NIFTY"
             ) -> Optional[ModelSignal]:
        if len(hist) < self.lookback:
            return None
        window = [t.spot for t in hist[-self.lookback:]]
        net = abs(window[-1] - window[0])
        path = sum(abs(b - a) for a, b in zip(window, window[1:])) or 1e-9
        er = net / path
        if er > 0.6:
            grade, conf = "HIGH (trending)", 0.78
        elif er > 0.35:
            grade, conf = "MEDIUM", 0.6
        else:
            grade, conf = "LOW (chop)", 0.55
        return ModelSignal(
            ts_ist=now_ist_hms(), asset=asset, model=self.name,
            signal=f"trend quality {grade}",
            confidence=conf, trust_tier="LOGGED",
            zone=None,
            risk="quality flips as range expands",
            reason_codes=["efficiency_ratio", grade.split()[0].lower()],
            extras={"efficiency_ratio": round(er, 3)},
        )


# ---------------------------------------------------------------------------
# 5. PostReactionModel — continuation read
# ---------------------------------------------------------------------------

class PostReactionModel:
    """After a reaction, what's the continuation risk? We measure how
    much of the reaction is retraced in the last short window."""

    name = "post_reaction_model"

    def __init__(self, reaction: int = 10, follow: int = 5) -> None:
        self.reaction, self.follow = reaction, follow

    def tick(self, hist: Sequence[Tick], asset: str = "NIFTY"
             ) -> Optional[ModelSignal]:
        if len(hist) < self.reaction + self.follow:
            return None
        seg = [t.spot for t in hist[-(self.reaction + self.follow):]]
        reaction_move = seg[self.reaction] - seg[0]
        follow_move = seg[-1] - seg[self.reaction]
        if abs(reaction_move) < 1e-6:
            return None
        retrace = -follow_move / reaction_move    # +ve means giving back
        if retrace > 0.5:
            return ModelSignal(
                ts_ist=now_ist_hms(), asset=asset, model=self.name,
                signal="reaction failing — over half retraced",
                confidence=round(min(0.9, 0.5 + retrace * 0.4), 3),
                trust_tier="LOGGED",
                risk="confirms only if next push fails too",
                reason_codes=["retrace_high", "continuation_doubt"],
                extras={"retrace_ratio": round(retrace, 3)},
            )
        if retrace < -0.3:
            return ModelSignal(
                ts_ist=now_ist_hms(), asset=asset, model=self.name,
                signal="reaction extending — continuation likely",
                confidence=round(min(0.88, 0.55 + (-retrace) * 0.4), 3),
                trust_tier="LOGGED",
                risk="watch for blow-off rejection",
                reason_codes=["continuation", "trend_follow"],
                extras={"retrace_ratio": round(retrace, 3)},
            )
        return None


# ---------------------------------------------------------------------------
# 6. ManipulationModel — fast spike-and-reclaim warning
# ---------------------------------------------------------------------------

class ManipulationModel:
    """The microstructure stop-hunt: ultra-fast spike well outside recent
    volatility, immediately followed by reclaim. Different from
    LiquidityModel: this fires in the SAME bar window, not after."""

    name = "manipulation_model"

    def __init__(self, lookback: int = 20, spike_z: float = 2.5) -> None:
        self.lookback, self.spike_z = lookback, spike_z

    def tick(self, hist: Sequence[Tick], asset: str = "NIFTY"
             ) -> Optional[ModelSignal]:
        if len(hist) < self.lookback + 3:
            return None
        window = [t.spot for t in hist[-self.lookback:]]
        rets = [b - a for a, b in zip(window, window[1:])]
        sd = statistics.pstdev(rets) or 1e-9
        last3 = window[-3:]
        max_dev = max(abs(p - window[-4]) for p in last3) / sd
        end_dev = abs(last3[-1] - window[-4]) / sd
        if max_dev > self.spike_z and end_dev < self.spike_z * 0.4:
            direction = "upside" if max(last3) > window[-4] else "downside"
            return ModelSignal(
                ts_ist=now_ist_hms(), asset=asset, model=self.name,
                signal=f"fast {direction} spike + reclaim — "
                       f"microstructure manipulation warning",
                confidence=round(min(0.9, 0.55 + (max_dev - self.spike_z) * 0.15), 3),
                trust_tier="LOGGED",
                risk="benign if move is genuine and holds",
                reason_codes=["fast_spike", "reclaim", "stop_hunt_risk"],
                extras={"peak_z": round(max_dev, 2),
                        "current_z": round(end_dev, 2)},
            )
        return None


# ---------------------------------------------------------------------------
# Pool — drives all six on each tick
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    ReactionModel(), ProximityModel(), LiquidityModel(),
    QualityModel(), PostReactionModel(), ManipulationModel(),
]


class LiveModelPool:
    """Drives every model on a tick, deduped via the publisher."""

    def __init__(self, models=None) -> None:
        self.models = list(models) if models is not None else list(DEFAULT_MODELS)

    def run(self, history: Sequence[Tick], asset: str = "NIFTY"
            ) -> List[ModelSignal]:
        out: List[ModelSignal] = []
        for m in self.models:
            try:
                sig = m.tick(history, asset=asset)
            except Exception:
                continue
            if sig is not None:
                out.append(sig)
        return out


declare(IOSpec(
    module="sentinel.live_models",
    purpose="six minimal live research models (reaction, proximity, "
            "liquidity, quality, post-reaction, manipulation) — contract "
            "stubs that publish to LivePublisher; heavier models from "
            "liquidity_backtester drop in via the same tick(history) shape",
    inputs=["Sequence[Tick] (recent spot history)"],
    outputs=["list[ModelSignal] published to LivePublisher per tick"],
    consumes_from=["sentinel.kite_client (spot ticks)"],
    produces_for=["sentinel.live_publisher"],
    tier="LOGGED",
))
