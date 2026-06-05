"""Plain-text email renderer for the Daily Research Brief.

Takes a :class:`BriefDocument` and emits a human-readable string suitable
for the body of an email or a paste into Telegram / WhatsApp / etc.
PDF rendering is deferred to a later module once the first paying
customer specifies their preferred format.

Hard renderer rules (from ``docs/daily_brief_schema.md``):

  * The string must be readable in 5–7 minutes by a sophisticated
    trader who is not paid to read carefully.
  * No tipster vocabulary in human prose. The forbidden words list is
    enforced by :func:`_assert_no_tipster_language` at the end so
    drift can be caught in tests.
  * Sections in fixed order. ``key_zones`` is excluded from the human
    layer — JSON-only.
  * Stubbed sections render as a one-line "data pending" note rather
    than being skipped, so the customer sees the full product shape.
"""
from __future__ import annotations

import re
from typing import List

from .daily_brief import (
    AvoidEntry,
    BriefDocument,
    ConfidenceNotes,
    SectorRegime,
    WatchlistEntry,
)


# Words that imply trade instructions. The renderer's output is grep'd
# against this list in tests; any match is a contract violation.
TIPSTER_VOCABULARY: tuple = (
    "buy ", "sell ", "go long", "go short", "long this", "short this",
    "entry at", "target at", "stop loss", "stop-loss",
    "book profit", "exit at",
)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_tldr(brief: BriefDocument) -> str:
    """One-paragraph headline summarising the brief's verdict."""
    avoid = brief.avoid_list or []
    all_basket = any(a.symbol == "ALL_BASKET" for a in avoid)
    watch = brief.top_watchlist or []
    if all_basket:
        verdict = (
            "Today the model finds NO instrument across the basket with "
            "actionable conviction. The structural regime is set to "
            "stand-aside: probabilities of any monitored level being "
            "tested same-session remain low."
        )
    elif watch:
        verdict = (
            f"The model flags {len(watch)} instrument(s) with non-trivial "
            f"probability of testing a structural level today. Read the "
            f"watchlist before deciding whether the context matches your "
            f"own thesis."
        )
    else:
        verdict = "Brief generated — see sections below for the structural read."
    return f"TLDR — {verdict}"


def _render_sector_regime(s: SectorRegime) -> str:
    """One paragraph describing sector posture."""
    bits: List[str] = []
    if s.trending_up:
        bits.append(f"trending up: {', '.join(s.trending_up)}")
    if s.trending_down:
        bits.append(f"trending down: {', '.join(s.trending_down)}")
    if s.chopping:
        bits.append(f"chopping: {', '.join(s.chopping)}")
    if s.neutral:
        bits.append(f"neutral: {', '.join(s.neutral)}")
    body = "; ".join(bits) if bits else "no sector data available"
    leadership = (
        f" Leadership change vs yesterday: {', '.join(s.leadership_change_vs_yesterday)}."
        if s.leadership_change_vs_yesterday
        else ""
    )
    return f"SECTOR REGIME — {body}.{leadership}"


def _render_watchlist(entries: List[WatchlistEntry]) -> str:
    """Table form: one line per instrument."""
    if not entries:
        return "WATCHLIST — empty. No instruments cleared the proximity threshold today."
    lines = ["WATCHLIST — instruments the model flags as in-play today:"]
    for e in entries:
        avoidance = f"  [note: {e.avoidance_note}]" if e.avoidance_note else ""
        lines.append(
            f"  - {e.symbol} ({e.sector}) {e.side} bias toward "
            f"{e.key_level_type} at {e.key_level:.2f}. "
            f"P(test today) = {e.p_touch_today:.0%}; "
            f"P(test within 60min) = {e.p_touch_within_60min:.0%}. "
            f"Confidence: {e.model_confidence_bucket}.{avoidance}"
        )
    return "\n".join(lines)


def _render_avoid_list(entries: List[AvoidEntry]) -> str:
    """Bullet list."""
    if not entries:
        return "AVOID — no specific stand-aside calls today."
    lines = ["AVOID — the model recommends standing aside in these contexts:"]
    for e in entries:
        lines.append(
            f"  - {e.symbol}: {e.reason} "
            f"(confidence: {e.model_confidence})"
        )
    return "\n".join(lines)


def _render_confidence(c: ConfidenceNotes) -> str:
    """One paragraph on which model heads to trust today."""
    parts: List[str] = []
    if c.calibrated_today:
        parts.append(f"calibrated today: {', '.join(c.calibrated_today)}")
    if c.drifting_today:
        parts.append(f"drifting today: {', '.join(c.drifting_today)}")
    if not parts:
        parts.append("no model-health data available")
    body = "; ".join(parts)
    return (
        f"CONFIDENCE — {body}. "
        f"Overall brief confidence: {c.overall_brief_confidence}."
    )


def _render_pending_stub(section_name: str, stub: dict) -> str:
    reason = str(stub.get("_reason", "data unavailable"))
    return f"{section_name.upper()} — pending: {reason}."


def _render_yesterday_audit(audit: dict) -> str:
    """Render the YESTERDAY AUDIT section.

    Two render paths:
      * Pending stub (when no joined history exists) — same shape as
        ``_render_pending_stub`` so v0 customers see the contract.
      * Populated audit — prints predictions_made / resolved / per-bucket
        hit rates. When ``is_retrospective_calibration`` is True (Stream G),
        prefixes the section with a disclosure line so the reader knows
        the calibration was estimated by replay over historical bundles
        rather than collected from live brief calls.
    """
    if not isinstance(audit, dict):
        return "YESTERDAY AUDIT — pending: data unavailable."
    if audit.get("_status") == "pending":
        return _render_pending_stub("Yesterday audit", audit)
    lines: List[str] = ["YESTERDAY AUDIT —"]
    if audit.get("is_retrospective_calibration"):
        share = float(audit.get("retrospective_share", 0.0))
        lines.append(
            f"  Note: calibration estimated on retrospective replay "
            f"({share:.0%} of resolved predictions were backfilled from "
            f"historical bundles, not collected live). Treat the numbers "
            f"below as a directional read, not a live track record.")
    yb = audit.get("yesterday_brief_id", "unknown")
    made = int(audit.get("predictions_made", 0))
    resolved = int(audit.get("predictions_resolved", 0))
    lines.append(f"  brief={yb} predictions_made={made} resolved={resolved}")
    buckets = audit.get("hit_rate_by_confidence_bucket") or []
    if not buckets:
        lines.append("  no per-bucket calibration available (zero resolved).")
    else:
        lines.append("  hit rate by confidence bucket:")
        for row in buckets:
            ptype = row.get("prediction_type", "?")
            bucket = row.get("confidence_bucket", "?")
            n = int(row.get("n", 0))
            hr = float(row.get("hit_rate", 0.0))
            mp = float(row.get("mean_predicted_p", 0.0))
            ce = float(row.get("calibration_error", 0.0))
            lines.append(
                f"    - {ptype} / {bucket}: n={n}, hit_rate={hr:.0%}, "
                f"mean_p={mp:.0%}, calibration_error={ce:+.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------

def render_email(brief: BriefDocument) -> str:
    """Render the brief as a plain-text email body.

    Section order matches ``docs/daily_brief_schema.md``. The output is
    grep'd against the tipster-vocabulary list before return; any match
    raises so we surface contract drift loudly in dev/test, not silently
    on a customer's inbox.
    """
    parts: List[str] = []

    md = brief.brief_metadata
    parts.append(
        f"Daily Research Brief — {md.trading_date_ist}\n"
        f"Generated {md.generated_at_utc}  bundle={md.model_bundle_version}\n"
        f"Indexes covered: {', '.join(md.indexes_covered) or '(none in v1)'}\n"
        f"Estimated reading time: ~{md.reading_time_minutes} min"
    )

    parts.append(_render_tldr(brief))
    parts.append(_render_pending_stub("Index regime", brief.index_regime))
    parts.append(_render_pending_stub("Options suitability", brief.options_suitability))
    parts.append(_render_sector_regime(brief.sector_regime))
    parts.append(_render_watchlist(brief.top_watchlist))
    parts.append(_render_avoid_list(brief.avoid_list))
    parts.append(_render_confidence(brief.confidence_notes))
    parts.append(_render_yesterday_audit(brief.yesterday_audit))

    parts.append(
        "—\n"
        "This brief is research context, NOT trade instructions. The "
        "probabilities reflect the model's calibrated view; the decision "
        "to act on any of it remains entirely with the reader."
    )

    body = "\n\n".join(parts)
    _assert_no_tipster_language(body)
    return body


# ---------------------------------------------------------------------------
# Contract guardrail
# ---------------------------------------------------------------------------

def _assert_no_tipster_language(body: str) -> None:
    """Raise if the rendered body contains any forbidden tipster phrase.

    This is the legal + product moat. Better to fail loudly in CI than
    let "buy at 24500" land in a customer's inbox.
    """
    lowered = body.lower()
    hits: List[str] = []
    for phrase in TIPSTER_VOCABULARY:
        # Use a word-boundary check on the END of the phrase so e.g.
        # "buy " doesn't trigger on "buyer" or "buyback" — but "buy "
        # followed by anything (a price, a stock) does.
        pattern = re.escape(phrase)
        if re.search(pattern, lowered):
            hits.append(phrase.strip())
    if hits:
        raise RuntimeError(
            f"Daily brief renderer produced tipster-vocabulary "
            f"contract violation. Forbidden phrases found: {hits}. "
            f"This is a hard product contract — every match indicates "
            f"a regression in the rendering logic."
        )
