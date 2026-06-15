"""In-app help — what every panel on the cockpit means + what the
academic/methodology citations refer to.

The operator hovers a `?` on any panel and a 2-sentence tooltip
appears. Reduces support load + helps customers self-educate.

Codex's onboarding flow ALSO reads from here to render the first-
login walkthrough cards.
"""
from __future__ import annotations

from typing import Any, Dict

from .io_decl import IOSpec, declare


PANEL_HELP: Dict[str, Dict[str, str]] = {
    "spot_chart": {
        "title": "NIFTY · Live spot",
        "what": "Live NIFTY 50 index price, polled every 1-2 seconds. "
                "The amber crosshair on hover shows exact price at the "
                "cursor's time. Model-zone overlays drawn as coloured "
                "horizontal bands when a research model flags a level.",
        "how_to_read": "Green band = support model fired. Red band = "
                        "resistance / liquidity-sweep zone. Steel = "
                        "neutral magnet (e.g. opening range high).",
    },
    "constituent_board": {
        "title": "NIFTY top-10 constituent board",
        "what": "Real-time view of the 10 highest-weight NIFTY stocks. "
                "Each row shows price move + weight-adjusted contribution "
                "to the index move.",
        "how_to_read": "STRONG = aligned heavyweights + breadth confirms. "
                        "FRAGILE = narrow leadership; one heavyweight "
                        "carrying the index. MANIPULATED = >55% of move "
                        "concentrated in top 2 stocks. ROTATION = high "
                        "dispersion but flat index. CONSOLIDATION = "
                        "everything tight.",
    },
    "live_signals": {
        "title": "Live research signals",
        "what": "Each row is one model's current output: reaction, "
                "proximity, liquidity, quality, post-reaction, "
                "manipulation, plus signals from the research engine "
                "(tagged [L]).",
        "how_to_read": "Confidence bar at 0..100%. Red invalidation "
                        "line is what would make this signal wrong. "
                        "Reason codes are academic citations / "
                        "methodology names you can audit.",
    },
    "live_feed": {
        "title": "Live brief feed",
        "what": "Time-ordered stream of every signal published this "
                "session. Same data the curator/calibration loop "
                "trains on overnight — what you see is what gets "
                "graded.",
        "how_to_read": "Newest first. Each line: HH:MM:SS · MODEL_TAG · "
                        "signal text (confidence %).",
    },
    "crux": {
        "title": "Crux · live verdict",
        "what": "One operator-facing fusion of every live signal + "
                "behavioral tilt + intention contract + position state. "
                "Seven verdicts: EXIT_NOW, TRAIL_UP, HOLD, WATCH, "
                "TRADE, DO_NOT_CHASE, WAIT.",
        "how_to_read": "Hard overrides first (kill switch / intention "
                        "violated / RED tilt + losing book = EXIT_NOW). "
                        "Then directional census. Always shows the "
                        "supporting signals AND the strongest "
                        "contradiction (intellectual honesty).",
    },
    "mind": {
        "title": "Mind · behavioral engine",
        "what": "Six citation-bearing bias detectors continuously "
                "scanning your action stream. The dial is your "
                "composite Tilt Index 0-100, decaying over time.",
        "how_to_read": "GREEN <30: frontal lobe in charge. AMBER 30-60: "
                        "system 1 (Kahneman) taking over. RED 60-85: "
                        "spine refuses non-exit orders. CIRCUIT 85+: "
                        "emotional kill-switch level. Citations: "
                        "Steenbarger 2009, Lo 2017, Tversky-Kahneman "
                        "1974, Gilovich-Vallone-Tversky 1985, Shefrin-"
                        "Statman 1985, Tharp 2007.",
    },
    "intention": {
        "title": "Intention contract (Ulysses pattern)",
        "what": "Pre-market commitment device per Elster 2000. You "
                "commit to a max day-loss + flat-by time BEFORE the "
                "session. Until then, Crux refuses TRADE verdicts.",
        "how_to_read": "Once set, breaching the max-loss flips "
                        "violated=True. Spine refuses new non-exit "
                        "orders until tomorrow.",
    },
    "profit_lock": {
        "title": "Profit lock · ratchet",
        "what": "Anti-greed mechanism: the locked profit floor rises "
                "with every new session peak and never falls. When P&L "
                "touches the locked floor again, everything flattens.",
        "how_to_read": "Locked ✓ = realised the moment lock fires. "
                        "Floating ⚠ = still at risk. The point: even "
                        "if today goes wrong, the worst-case is the "
                        "locked floor.",
    },
    "trust_spine": {
        "title": "Trust spine · orchestrator",
        "what": "The four trust tiers signal sources can hold: "
                "SHADOW < LOGGED < TRUSTED < EXECUTION. Only "
                "trailing_stop and profit_lock can ever hold "
                "EXECUTION. Research signals max out at TRUSTED.",
        "how_to_read": "Per-tier counts show how many signals flowed "
                        "through each ceiling this session. Clamps = "
                        "research sources that tried to request EXECUTION "
                        "and were blocked.",
    },
    "scorecard": {
        "title": "Journeys · scorecard & mistakes",
        "what": "Every completed journey graded 0-100 on five "
                "20-point axes (reason codes, entry timing, trail "
                "discipline, exit quality, R-multiple). Mistakes "
                "scanned post-hoc.",
        "how_to_read": "Grades A>=85, B>=70, C>=55, D>=40, F<40. "
                        "Mistakes carry citations: Tharp 2007 R-multiples, "
                        "Steenbarger 2009 performance metrics, Hull 11e "
                        "invalidation rule.",
    },
    "monte_carlo": {
        "title": "Monte Carlo · scenario probe",
        "what": "1000 GBM paths (Boyle 1977) over next 5/15/30 min on "
                "your current book. Reports P(profit), P(target), "
                "P(stop), expected R-multiple distribution.",
        "how_to_read": "Tier: PRO. Light approximation (Δ + 0.5·Γ·move² "
                        "+ θ); good for short horizons on near-ATM. For "
                        "longer / OTM, use the SVI re-price in the "
                        "institutional layer.",
    },
    "premium_chart": {
        "title": "Option premium · live",
        "what": "Live premium history for the option contract with "
                "the largest qty in your book (or pin via the "
                "dropdown).",
        "how_to_read": "Green = premium up vs open; red = down. "
                        "Velocity ₹/min shown above. A flat spot with "
                        "rising premium = vol being bid; a rising spot "
                        "with stale premium = your contract isn't "
                        "responding.",
    },
    "tax_breakdown": {
        "title": "Tax-impact preview",
        "what": "Itemised STT, GST, STCG, brokerage, SEBI charges + "
                "stamp duty on any options round-trip. Tells you what "
                "actually lands in your bank account.",
        "how_to_read": "Net P&L = Gross - Total Charges. STCG only "
                        "applies on positive P&L (Section 111A; FY "
                        "2024-25 onwards at 20%).",
    },
    "session_context": {
        "title": "Session context · holidays · Muhurat",
        "what": "Today's trading status. Identifies NSE holidays, "
                "next trading day, Muhurat (Diwali special) sessions, "
                "and whether today is an expiry day (Thursday or "
                "Wednesday-shifted-from-Thursday-holiday).",
        "how_to_read": "Used by the cockpit's pre-market banner so "
                        "you never discover a holiday at 09:15. "
                        "Upcoming holidays in next 7 days listed.",
    },
}


CITATIONS: Dict[str, Dict[str, str]] = {
    "tharp_2007": {
        "ref": "Tharp, V.K. (2007) — Trade Your Way to Financial Freedom, 2e",
        "used_for": "R-multiple methodology + give-back ratio",
    },
    "steenbarger_2009": {
        "ref": "Steenbarger, B. (2009) — Enhancing Trader Performance",
        "used_for": "Performance metrics + revenge-trade pattern + "
                     "target-too-far definition",
    },
    "kahneman_2011": {
        "ref": "Kahneman, D. (2011) — Thinking, Fast and Slow",
        "used_for": "System 1 vs System 2 framing of TiltIndex bands",
    },
    "tversky_kahneman_1974": {
        "ref": "Tversky, A. & Kahneman, D. (1974) — Judgment Under "
                "Uncertainty: Heuristics & Biases",
        "used_for": "Anchoring detector (round-number stops)",
    },
    "gilovich_vallone_tversky_1985": {
        "ref": "Gilovich, T., Vallone, R. & Tversky, A. (1985) — "
                "The Hot Hand in Basketball",
        "used_for": "Hot-hand detector (size escalation after streak)",
    },
    "shefrin_statman_1985": {
        "ref": "Shefrin, H. & Statman, M. (1985) — The Disposition to "
                "Sell Winners Too Early and Ride Losers Too Long",
        "used_for": "Disposition effect detector (asymmetric holding)",
    },
    "lo_2017": {
        "ref": "Lo, A.W. (2017) — Adaptive Markets",
        "used_for": "FOMO entry detector (recency + herd bias)",
    },
    "elster_2000": {
        "ref": "Elster, J. (2000) — Ulysses Unbound",
        "used_for": "Intention contract pattern",
    },
    "hull_2017": {
        "ref": "Hull, J.C. (2017) — Options, Futures & Other "
                "Derivatives, 11e",
        "used_for": "Hull-taxonomy strategy detection + invalidation rule",
    },
    "boyle_1977": {
        "ref": "Boyle, P.P. (1977) — Options: A Monte Carlo Approach",
        "used_for": "Monte Carlo Lite GBM path generation",
    },
    "gatheral_2004": {
        "ref": "Gatheral, J. (2004) — A parsimonious arbitrage-free "
                "implied vol parameterisation (SVI)",
        "used_for": "Volatility surface fitting",
    },
    "burghardt_lane_1990": {
        "ref": "Burghardt, G. & Lane, M. (1990) — Vol cone methodology",
        "used_for": "Realized vol cone in customer console",
    },
}


def help_payload() -> Dict[str, Any]:
    return {
        "panels": PANEL_HELP,
        "citations": CITATIONS,
        "note": ("Every methodology cited is a published source so a "
                  "professional auditor can verify the math on sight."),
    }


declare(IOSpec(
    module="sentinel.help_text",
    purpose="in-app help text — short tooltips for every panel + the "
            "full academic citation list so a customer can audit what "
            "the cockpit is doing. Surfaces via /api/help; Codex's "
            "onboarding flow reads from here to render the first-login "
            "walkthrough cards",
    inputs=[],
    outputs=["help_payload() dict with PANEL_HELP + CITATIONS"],
    consumes_from=[],
    produces_for=["sentinel.server (/api/help)",
                  "Codex's onboarding flow"],
    tier="TRUSTED",
))
