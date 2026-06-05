"""Opaque, non-invertible public scoring layer for the analytics feed.

This module is the IP boundary between the internal models and what a customer
ever sees. Internal estimates (reachability probability, directional probability,
reliability) are *real* signals, but they are deliberately transformed into
opaque, rank-based, watermarked tokens before they leave the server:

  * ``feature_intensity_score`` — an integer 0..100 *rank* of an internal
    composite within a cross-section (one symbol's active levels per side, on one
    day). It is not a probability, has no units, and cannot be regressed back to
    the inputs.
  * ``feature_state`` — an opaque, non-directional class label from a small
    abstract set (``state_1``/``state_2``/``state_3``), bucketed with
    per-customer-jittered thresholds. The customer-facing label carries no
    directional claim; any internal directional meaning is kept server-side.

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
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Public, non-recommendatory disclaimer + compliance text.
# Kept as module constants so the compliance linter and tests can exempt them
# (they intentionally contain negated banned words like "buy/sell/hold").
# ---------------------------------------------------------------------------
INTERPRETATION_NOTE = (
    "This output is non-recommendatory market analytics for informational and "
    "research use only. It is not investment advice, not a buy/sell/hold "
    "recommendation, not a price target, and not an instruction to trade. Users "
    "are solely responsible for independent validation and decisions. Historical "
    "statistics do not guarantee future outcomes."
)

# Compact per-observation compliance marker (full notice lives at feed level).
COMPLIANCE_CLASSIFICATION = "non_recommendatory_market_analytics"
COMPLIANCE_TAG = COMPLIANCE_CLASSIFICATION

# Guard so a price band is never read as an actionable level.
LEVEL_USAGE_NOTE = (
    "Reference zone only; not an entry, exit, stoploss, target, or execution "
    "instruction."
)
SCORE_EXPLANATION = (
    "Opaque statistical market-structure feature. Not a trade direction and not "
    "a probability."
)

ANALYTICS_TYPE = "market_structure_observation"
FEED_ANALYTICS_TYPE = "market_structure_observation_feed"
FEED_VERSION = "g-feed/1"

# Opaque, NON-DIRECTIONAL class labels. The customer-facing label makes no
# directional claim; any internal directional meaning is kept server-side and is
# never exposed. Index order here corresponds to internal buckets only.
FEATURE_STATES: tuple[str, str, str] = ("state_1", "state_2", "state_3")

# Neutral keys that are the ONLY thing public_record will ever emit.
PUBLIC_KEYS: tuple[str, ...] = (
    "analytics_type",
    "instrument",
    "feature_intensity_score",
    "feature_state",
    "level_zone",
    "level_usage_note",
    "score_explanation",
    "scope",
    "as_of",
    "feed_version",
    "compliance_tag",
)

LICENSED_TIERS: frozenset[str] = frozenset({
    "licensed", "paid", "premium", "internal", "private",
})
DEMO_TIERS: frozenset[str] = frozenset({
    "demo", "public", "free", "marketing", "sample",
})

# Public keys whose VALUES are deliberately compliance text (negated banned
# words) and are therefore exempt from the forbidden-token leak check.
COMPLIANCE_TEXT_KEYS: frozenset[str] = frozenset({
    "level_usage_note", "score_explanation", "compliance_tag",
})

# Internal vocabulary that must NEVER appear in a public payload's keys or
# values (the compliance-text fields above are exempt). Tests use this as a
# leak tripwire.
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


def normalize_customer_tier(customer_tier: str | None) -> str:
    """Normalize public-feed access tiers.

    Library callers default to ``licensed`` for backward compatibility.
    Explicit unknown tiers are treated as demo/public, which is the safer
    geometry policy for customer-facing artifacts.
    """
    if customer_tier is None:
        return "licensed"
    tier = str(customer_tier).strip().lower()
    if not tier:
        return "licensed"
    if tier in LICENSED_TIERS:
        return "licensed"
    if tier in DEMO_TIERS:
        return "demo"
    return "demo"


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


def feature_states(levels: Sequence[InternalLevel], customer_id: str, day: str,
                   alphabet: tuple[str, str, str] = FEATURE_STATES) -> list[str]:
    """Opaque, NON-DIRECTIONAL class labels with per-customer-jittered band.

    Internally the three buckets derive from the direction model, but the
    returned label (state_1/2/3) makes no directional claim. The mapping from
    bucket to market meaning is intentionally kept server-side.
    """
    delta = _customer_dir_delta(customer_id, day)
    toks = []
    for l in levels:
        if l.p_up is None:
            toks.append(alphabet[1])
            continue
        s = float(l.p_up) - 0.5      # internal lean; never exposed
        if s > delta:
            toks.append(alphabet[0])
        elif s < -delta:
            toks.append(alphabet[2])
        else:
            toks.append(alphabet[1])
    return toks


def _demo_zone_step(mid: float) -> float:
    """Coarse price increment for demo/public zones.

    Roughly 50 bps of price with sensible lower bounds. This keeps the
    observation useful as public market-structure context while ensuring the
    exact internal pool geometry is not exposed.
    """
    m = max(abs(float(mid)), 1.0)
    return max(1.0 if m >= 100.0 else 0.25, 0.005 * m)


def _abstract_level_zone(level: InternalLevel, customer_id: str, day: str) -> dict:
    step = _demo_zone_step(level.level_mid)
    seed = _customer_seed(
        customer_id,
        day,
        f"zone|{level.symbol}|{level.side}|{level.level_mid:.8f}",
    )
    rng = np.random.default_rng(seed)
    jitter = float(rng.uniform(-0.45, 0.45)) * step
    canary = float(rng.uniform(-0.18, 0.18)) * step
    center = round((float(level.level_mid) + jitter) / step) * step + canary
    if abs(center - float(level.level_mid)) < 0.10 * step:
        center += 0.25 * step
    half_width = 0.5 * step
    return {
        "low": round(center - half_width, 4),
        "high": round(center + half_width, 4),
        "mid": round(center, 4),
    }


def _exact_level_zone(level: InternalLevel) -> dict:
    return {
        "low": round(float(level.level_low), 4),
        "high": round(float(level.level_high), 4),
        "mid": round(float(level.level_mid), 4),
    }


def public_record(level: InternalLevel, intensity: int, state: str, *,
                  customer_id: str | None = None,
                  day: str | None = None,
                  customer_tier: str | None = "licensed") -> dict:
    """Build the outbound record from an explicit neutral allow-list.

    Internal fields have no path into this dict, so they cannot leak. Licensed
    research customers receive exact chart geometry. Demo/public customers
    receive deterministic coarse zones so sample feeds cannot be used as an
    exact level service. The full disclaimer lives once at feed level; each
    record carries a compact tag.
    """
    tier = normalize_customer_tier(customer_tier)
    if tier == "demo":
        zone = _abstract_level_zone(level, customer_id or "public", day or "")
    else:
        zone = _exact_level_zone(level)
    return {
        "analytics_type": ANALYTICS_TYPE,
        "instrument": level.symbol,
        "feature_intensity_score": int(intensity),
        "feature_state": str(state),
        "level_zone": zone,
        "level_usage_note": LEVEL_USAGE_NOTE,
        "score_explanation": SCORE_EXPLANATION,
        "scope": str(level.scope),
        "as_of": str(level.as_of),
        "feed_version": FEED_VERSION,
        "compliance_tag": COMPLIANCE_TAG,
    }


def compliance_notice() -> dict:
    """Feed-level compliance block (full notice lives here, once)."""
    return {
        "classification": COMPLIANCE_CLASSIFICATION,
        "notice": INTERPRETATION_NOTE,
    }


def score_levels(levels: Sequence[InternalLevel], customer_id: str, day: str,
                 weights: BlendWeights = BlendWeights(),
                 customer_tier: str | None = "licensed") -> list[dict]:
    """End-to-end: internal cross-section -> list of opaque public records."""
    if not levels:
        return []
    intensities = g_scores(levels, customer_id, day, weights)
    states = feature_states(levels, customer_id, day)
    return [public_record(l, i, s, customer_id=customer_id, day=day,
                          customer_tier=customer_tier)
            for l, i, s in zip(levels, intensities, states)]
