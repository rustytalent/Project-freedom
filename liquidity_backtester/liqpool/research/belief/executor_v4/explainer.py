"""Operator-facing explainer — translates per-tick outputs to plain English.

The founder will read this in the cockpit. Every decision needs to make
sense in 5 seconds: what did we do, why, what would change our mind, and
what's protecting us.

Stateless functor. ``explain_tick(...)`` returns a single multi-line
string assembled from sections:
  * What just happened (HOLD, ENTER, EXIT, REFUSE)
  * Why (top reasons by component)
  * Probability web summary (top 3 scenarios + consensus)
  * MM mind read (dominant intent + guidance)
  * Fat-tail status
  * Crowd mirror status
  * Portfolio summary (P&L + open positions)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def explain_tick(intent: Dict[str, Any],
                  human_friendly: bool = True) -> str:
    """Compose a multi-line operator-facing explanation from a
    PortfolioIntent.to_dict() output.
    """
    lines: List[str] = []
    bar = int(intent.get("bar_index") or 0)
    ts = str(intent.get("ts") or "")
    summary = intent.get("portfolio_summary") or {}

    lines.append(f"━━ Tick @ bar {bar} ({ts}) ━━")
    # Action
    new_entry = intent.get("new_entry")
    closed = intent.get("closed_this_tick") or []
    refuse = intent.get("refuse_reasons") or []
    if new_entry:
        h = new_entry.get("hypothesis") or {}
        agg = new_entry.get("aggregator_decision") or {}
        lines.append(
            f"  ✓ OPENED {h.get('contract_label')} "
            f"× {h.get('size_lots')} lot @ ₹{h.get('entry_premium'):.2f}"
        )
        lines.append(
            f"    aggregator: {agg.get('decision')} "
            f"score={agg.get('final_score'):.2f}; "
            f"size×{agg.get('recommended_size_multiplier'):.2f}"
        )
    for c in closed:
        out = c.get("outcome") or {}
        lines.append(
            f"  ✗ CLOSED {c.get('position_id')} — "
            f"{out.get('exit_reason')[:80]} | realized ₹{out.get('realized_rupees'):.0f}"
        )
    if refuse and not new_entry:
        lines.append(
            "  ⊘ REFUSED entry: " + "; ".join(refuse[:2])
        )
    elif not new_entry and not closed and not refuse:
        lines.append("  · HOLD")

    # Web summary
    web = summary.get("scenario_web") or {}
    if web:
        consensus = web.get("directional_consensus", 0.0)
        tail = web.get("tail_mass", 0.0)
        chop = web.get("chop_mass", 0.0)
        dom_strat = web.get("dominant_strategy_class", "")
        lines.append(
            f"  WEB: consensus={consensus:+.2f}  tail={tail:.2f}  "
            f"chop={chop:.2f}  dom_strat={dom_strat}"
        )
        for sc in (web.get("top_scenarios") or [])[:3]:
            lines.append(
                f"    – {sc.get('name'):28s} "
                f"p={sc.get('current_probability'):.3f} "
                f"({sc.get('family')})"
            )

    # MM mind
    mm = summary.get("mm_posterior") or {}
    if mm:
        lines.append(
            f"  MM-MIND: {mm.get('dominant_intent')} "
            f"({mm.get('dominant_probability'):.0%}) — "
            f"{mm.get('operator_guidance', '')[:100]}"
        )

    # Fat-tail
    tail_obj = summary.get("fat_tail_score") or {}
    if tail_obj:
        action = tail_obj.get("recommended_action", "")
        score = tail_obj.get("tail_score", 0.0)
        lines.append(f"  FAT-TAIL: score={score:.2f} → {action}")

    # Crowd mirror
    crowd = summary.get("crowd_mirror") or {}
    if crowd and crowd.get("we_look_like_retail"):
        recs = crowd.get("recommended_diversification") or []
        lines.append(
            f"  CROWD: we look like retail ({crowd.get('retail_similarity_score'):.2f})"
        )
        for r in recs[:2]:
            lines.append(f"    → {r}")

    # Portfolio P&L
    pnl = intent.get("daily_pnl_rupees", 0.0)
    fees = intent.get("cumulative_fees_rupees", 0.0)
    lines.append(
        f"  PnL: ₹{pnl:+.0f} (fees ₹{fees:.0f}) | "
        f"open {summary.get('n_open', 0)} | "
        f"closed {summary.get('n_closed', 0)}"
    )

    return "\n".join(lines)
