"""Opaque, non-invertible public scoring layer for the analytics feed.

This module is the IP boundary between the internal models and what a customer
ever sees. Internal estimates (reachability probability, directional probability,
reliability) are *real* signals, but they are deliberately transformed into
opaque, rank-based, watermarked tokens before they leave the server:

  * ``G`` — an integer 0..100 *rank* of an internal composite within a
    cross-section (one symbol's active levels per side, on one day). It is not a
    probability, has no units, and cannot be regressed back to the inputs.
  * ``D`` — an opaque directional token from a small abstract alphabet, bucketed
    with per-customer-jittered thresholds. It conveys directional context without
    exposing the underlying probability or the thresholds used to derive it.

Two further protections live here:

  * A per-customer *monotone* re-parameterisation (a hidden gamma curve) plus a
    deterministic canary dither. Two customers see rank-consistent but
    numerically different feeds, so a leaked feed is attributable, and neither
    customer can recover the underlying percentile/probability.
  * :func:`public_record` constructs the outbound dict from an explicit
    allow-list of neutral keys. Internal field names and values can therefore
    never leak by accident — the serializer simply has no path to emit them.

Nothing here references buy/sell/hold, entry/exit, targets, stops, or price
targets. The output is non-recommendatory market-structure analytics.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Public, non-recommendatory disclaimer carried on every record / feed.
# Kept as a module constant so the compliance linter and tests can exempt it
# (it intentionally contains negated banned words like "buy/sell/hold").
# ---------------------------------------------------------------------------
INTERPRETATION_NOTE = (
    "This output is non-recommendatory market analytics for informational and "
    "research use only. It is not investment advice, not a buy/sell/hold "
    "recommendation, not a price target, and not an instruction to trade. Users "
    "are solely responsible for independent validation and decisions. Historical "
    "statistics do not guarantee future outcomes."
)

ANALYTICS_TYPE = "market_structure_observation"
FEED_VERSION = "g-feed/1"

# Abstract, sign-bearing but probability-free directional alphabet. The product's
# value is directional *context*; what stays hidden is the probability and the
# (per-customer-jittered) thresholds used to bucket it.
DIRECTION_TOKENS: tuple[str, str, str] = ("+", "=", "-")

# Neutral keys that are the ONLY thing public_record will ever emit.
PUBLIC_KEYS: tuple[str, ...] = (
    "analytics_type",
    "instrument",
    "G",
    "D",
    "level_zone",
    "scope",
    "as_of",
    "feed_version",
    "interpretation_note",
)

# Internal vocabulary that must NEVER appear in a public payload's keys or
# values (the disclaimer field is exempt). Used by tests as a leak tripwire.
FORBIDDEN_PUBLIC_TOKENS: tuple[str, ...] = (
    "p_touch", "p_up", "p_dir", "p_direction", "probability", "prob",
    "reachability", "direction", "q_score", "quality", "reliability",
    "entry", "exit", "stop", "stoploss", "target", "buy", "sell", "hold",
    "long", "short", "signal", "recommendation", "percentile", "rank",
)


@dataclass(frozen=True)
class InternalLevel:
    """Server-side, never serialized. The raw analytics for one level."""
    symbol: str
    side: str            # "above" | "below" (used only to group/sign internally)
    level_low: float
    level_high: float
    level_mid: float
    p_touch: float       # reachability probability in [0,1]
    p_up: float | None   # market up-probability in [0,1] (direction model)
    q: float             # reliability/quality in [0,1]
    as_of: str
    scope: str = "s0"    # opaque scope code (abstracts the proximity horizon)


@dataclass(frozen=True)
class BlendWeights:
    """Internal composite weights. The *production* recipe is private config;
    these defaults exist so the module is runnable/testable. The recipe is not
    the moat — the rank+quantize+watermark below is."""
    w_touch: float = 1.40
    w_dir: float = 0.55
    w_quality: float = 0.45


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def blend_internal(levels: Sequence[InternalLevel],
                   weights: BlendWeights = BlendWeights()) -> np.ndarray:
    """Monotone blend of the internal signals into a single latent score.

    Monotone increasing in p_touch, in directional confidence, and in quality —
    so a higher blended value always means a "stronger" level. The absolute
    value is meaningless and never leaves the server; only its cross-sectional
    rank does.
    """
    if not levels:
        return np.zeros(0, dtype=float)
    p_touch = np.array([float(l.p_touch) for l in levels], dtype=float)
    q = np.array([float(l.q) for l in levels], dtype=float)
    # Directional *confidence toward the level*, regardless of side.
    dir_conf = np.array([_dir_conf_toward_level(l) for l in levels], dtype=float)
    z = (weights.w_touch * _logit(p_touch)
         + weights.w_dir * _logit(dir_conf)
         + weights.w_quality * _logit(q))
    return _sigmoid(z)


def _dir_conf_toward_level(level: InternalLevel) -> float:
    """Probability price travels toward this level, in [0,1]. Hidden internally."""
    if level.p_up is None:
        return 0.5
    p_up = float(level.p_up)
    return p_up if level.side == "above" else (1.0 - p_up)


# ---------------------------------------------------------------------------
# Per-customer watermark parameters (deterministic, hidden from the customer).
# ---------------------------------------------------------------------------
def _customer_seed(customer_id: str, day: str, salt: str = "g-feed") -> int:
    h = hashlib.sha256(f"{salt}|{customer_id}|{day}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _customer_gamma(customer_id: str, day: str) -> float:
    """A hidden monotone re-parameterisation exponent in ~[0.90, 1.11].

    Applied as G = 100 * percentile**gamma. Monotone (rank-preserving), so the
    ordering the customer relies on is intact, but the customer cannot recover
    the underlying percentile without gamma — and gamma differs per customer,
    which both prevents cross-customer differencing and fingerprints the feed.
    """
    rng = np.random.default_rng(_customer_seed(customer_id, day, "gamma"))
    return float(0.90 + 0.21 * rng.random())


def _customer_dir_delta(customer_id: str, day: str) -> float:
    """Per-customer neutral-band half-width for the D token, ~[0.07, 0.13]."""
    rng = np.random.default_rng(_customer_seed(customer_id, day, "dir"))
    return float(0.07 + 0.06 * rng.random())


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Average-rank percentile in [0,1] (ties share the mean rank)."""
    n = values.size
    if n == 0:
        return values
    if n == 1:
        return np.array([0.5])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    # average ranks for ties
    sorted_vals = values[order]
    i = 0
    avg_rank = np.empty(n, dtype=float)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            avg_rank[k] = avg
        i = j + 1
    ranks[order] = avg_rank
    return ranks / (n - 1)


def g_scores(levels: Sequence[InternalLevel], customer_id: str, day: str,
             weights: BlendWeights = BlendWeights()) -> list[int]:
    """Compute opaque integer G in [0,100] for a cross-section of levels.

    Levels should already be a single comparable cross-section (e.g. one
    symbol's active levels on one day). Ranking is done per side so that
    above/below pools are ranked among their peers.
    """
    n = len(levels)
    if n == 0:
        return []
    blended = blend_internal(levels, weights)
    gamma = _customer_gamma(customer_id, day)
    dither_rng = np.random.default_rng(_customer_seed(customer_id, day, "dither"))

    out = [0] * n
    for side in ("above", "below"):
        idx = [i for i, l in enumerate(levels) if l.side == side]
        if not idx:
            continue
        pct = _percentile_ranks(blended[idx])
        for local, i in enumerate(idx):
            g = 100.0 * (pct[local] ** gamma)
            # +-1 canary dither: keeps values within 1 of the monotone curve,
            # fingerprints the feed, and blocks exact reconstruction.
            g += float(dither_rng.integers(-1, 2))
            out[i] = int(np.clip(round(g), 0, 100))
    return out


def d_tokens(levels: Sequence[InternalLevel], customer_id: str, day: str,
             alphabet: tuple[str, str, str] = DIRECTION_TOKENS) -> list[str]:
    """Opaque directional tokens with per-customer-jittered neutral band."""
    delta = _customer_dir_delta(customer_id, day)
    toks = []
    for l in levels:
        if l.p_up is None:
            toks.append(alphabet[1])
            continue
        s = float(l.p_up) - 0.5      # market up-lean
        if s > delta:
            toks.append(alphabet[0])
        elif s < -delta:
            toks.append(alphabet[2])
        else:
            toks.append(alphabet[1])
    return toks


def public_record(level: InternalLevel, g: int, d: str) -> dict:
    """Build the outbound record from an explicit neutral allow-list.

    Internal fields have no path into this dict, so they cannot leak. Price-band
    geometry (level_zone) is public chart information, not IP.
    """
    return {
        "analytics_type": ANALYTICS_TYPE,
        "instrument": level.symbol,
        "G": int(g),
        "D": str(d),
        "level_zone": {
            "low": round(float(level.level_low), 4),
            "high": round(float(level.level_high), 4),
            "mid": round(float(level.level_mid), 4),
        },
        "scope": str(level.scope),
        "as_of": str(level.as_of),
        "feed_version": FEED_VERSION,
        "interpretation_note": INTERPRETATION_NOTE,
    }


def score_levels(levels: Sequence[InternalLevel], customer_id: str, day: str,
                 weights: BlendWeights = BlendWeights()) -> list[dict]:
    """End-to-end: internal cross-section -> list of opaque public records."""
    if not levels:
        return []
    gs = g_scores(levels, customer_id, day, weights)
    ds = d_tokens(levels, customer_id, day)
    return [public_record(l, g, d) for l, g, d in zip(levels, gs, ds)]
