"""Operator cockpit adapter — structured snapshot for the UI layer.

This module takes a per-tick ``PortfolioIntent.to_dict()`` and produces
a CockpitSnapshot — a compact, panel-oriented dict suitable for any
front-end to render. The Sentinel cockpit consumes this in Sprint 5;
the structure is also useful for terminal dashboards, replay tooling,
and post-mortem reports.

Panels included:

  * **action_card**     — what just happened this tick (HOLD / ENTER /
                          EXIT / REFUSE) plus headline reasons
  * **web_panel**       — scenario web summary: consensus, tail/chop
                          masses, dominant strategy class, top 5 active
                          scenarios with probability + family
  * **mm_panel**        — market-maker mind: dominant intent + prob +
                          implied bias + vol view + operator guidance
  * **fat_tail_dial**   — single tail_score with action band
  * **crowd_panel**     — we_look_like_retail flag + density gauges +
                          diversification recommendations
  * **risk_panel**      — portfolio gauges: net delta, gross exposure,
                          drawdown_r, cluster breakdown, kill switches
  * **patterns_panel**  — manipulation patterns currently active
  * **hedge_panel**     — pending hedge proposal (if any)
  * **positions_panel** — open positions with R, P&L, hypothesis links
  * **pnl_panel**       — daily P&L, fees, cumulative
  * **explainer**       — the human-readable multi-line text
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .explainer import explain_tick


# Action-card actions.
ACT_HOLD = "HOLD"
ACT_ENTER = "ENTER"
ACT_EXIT = "EXIT"
ACT_REFUSE = "REFUSE"


@dataclass
class CockpitSnapshot:
    """Per-tick cockpit-shaped view of the executor state."""
    ts: str
    bar_index: int
    action_card: Dict[str, Any]
    web_panel: Dict[str, Any]
    mm_panel: Dict[str, Any]
    fat_tail_dial: Dict[str, Any]
    crowd_panel: Dict[str, Any]
    risk_panel: Dict[str, Any]
    patterns_panel: Dict[str, Any]
    hedge_panel: Dict[str, Any]
    positions_panel: Dict[str, Any]
    pnl_panel: Dict[str, Any]
    explainer_text: str
    # Founder-requested live panels (2026-06-22)
    capital_panel: Dict[str, Any] = field(default_factory=dict)
    trades_panel: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def build_cockpit_snapshot(intent_dict: Dict[str, Any]) -> CockpitSnapshot:
    """Transform a PortfolioIntent.to_dict() into a panel-oriented cockpit view.

    The input is whatever ``PortfolioManager.evaluate(...).to_dict()``
    produced. The output is a compact, UI-friendly dict.
    """
    summary = intent_dict.get("portfolio_summary") or {}
    new_entry = intent_dict.get("new_entry")
    closed = intent_dict.get("closed_this_tick") or []
    refuse = intent_dict.get("refuse_reasons") or []

    # Action card
    if new_entry:
        h = new_entry.get("hypothesis") or {}
        ad = new_entry.get("aggregator_decision") or {}
        action_card = {
            "kind": ACT_ENTER,
            "headline": f"OPEN {h.get('contract_label')} × {h.get('size_lots')} lot",
            "premium": h.get("entry_premium"),
            "size_lots": h.get("size_lots"),
            "stop_premium": h.get("stop_premium"),
            "target_premium": h.get("target_premium"),
            "aggregator_decision": ad.get("decision"),
            "aggregator_score": ad.get("final_score"),
            "size_multiplier": ad.get("recommended_size_multiplier"),
            "rupees_at_risk": h.get("rupees_at_risk"),
            "conviction_label": (new_entry.get("conviction") or {}).get("label"),
            "conviction_signed": (new_entry.get("conviction") or {}).get("signed_value"),
        }
    elif closed:
        # Take the first closure as the headline.
        c0 = closed[0]
        out = c0.get("outcome") or {}
        action_card = {
            "kind": ACT_EXIT,
            "headline": (f"EXIT {c0.get('position_id')[:8]} — "
                          f"₹{out.get('realized_rupees'):.0f}"),
            "exit_reason": out.get("exit_reason"),
            "realized_rupees": out.get("realized_rupees"),
            "realized_r": out.get("realized_r"),
            "bars_held": out.get("bars_held"),
            "severity": out.get("exit_severity"),
        }
    elif refuse:
        action_card = {
            "kind": ACT_REFUSE,
            "headline": f"REFUSE: {refuse[0][:80]}",
            "all_reasons": list(refuse),
        }
    else:
        action_card = {
            "kind": ACT_HOLD,
            "headline": "HOLD",
        }

    # Web panel
    web = summary.get("scenario_web") or {}
    web_panel = {
        "n_active": web.get("n_active", 0),
        "directional_consensus": web.get("directional_consensus", 0.0),
        "tail_mass": web.get("tail_mass", 0.0),
        "chop_mass": web.get("chop_mass", 0.0),
        "manipulation_mass": web.get("manipulation_mass", 0.0),
        "dominant_strategy_class": web.get("dominant_strategy_class", ""),
        "top_scenarios": [
            {
                "name": sc.get("name"),
                "family": sc.get("family"),
                "probability": sc.get("current_probability"),
                "direction": sc.get("implied_direction"),
                "horizon_bars": sc.get("implied_horizon_bars"),
            }
            for sc in (web.get("top_scenarios") or [])[:5]
        ],
        "notes": web.get("notes") or [],
    }

    # MM panel
    mm = summary.get("mm_posterior") or {}
    mm_panel = {
        "dominant_intent": mm.get("dominant_intent", "neutral_inventory"),
        "dominant_probability": mm.get("dominant_probability", 0.0),
        "implied_bias": mm.get("implied_bias", 0),
        "implied_volatility_view": mm.get("implied_volatility_view", "neutral"),
        "confidence": mm.get("confidence", 0.0),
        "operator_guidance": mm.get("operator_guidance", ""),
        "active_patterns": mm.get("active_patterns") or [],
    }

    # Fat-tail dial
    tail = summary.get("fat_tail_score") or {}
    fat_tail_dial = {
        "tail_score": tail.get("tail_score", 0.0),
        "action": tail.get("recommended_action", "NORMAL"),
        "components": tail.get("components") or {},
        "notes": tail.get("notes") or [],
    }

    # Crowd panel
    crowd = summary.get("crowd_mirror") or {}
    crowd_panel = {
        "we_look_like_retail": crowd.get("we_look_like_retail", False),
        "retail_similarity_score": crowd.get("retail_similarity_score", 0.0),
        "crowd_density_long_atm_ce": crowd.get("crowd_density_long_atm_ce", 0.0),
        "crowd_density_long_atm_pe": crowd.get("crowd_density_long_atm_pe", 0.0),
        "mm_likely_target_us": crowd.get("mm_likely_target_us", False),
        "recommended_diversification": crowd.get("recommended_diversification") or [],
    }

    # Risk panel
    risk = summary.get("portfolio_risk") or {}
    risk_panel = {
        "total_premium_at_risk_rupees": risk.get("total_premium_at_risk_rupees", 0.0),
        "net_directional_exposure_lots": risk.get("net_directional_exposure_lots", 0.0),
        "gross_exposure_rupees": risk.get("gross_exposure_rupees", 0.0),
        "portfolio_drawdown_r": risk.get("portfolio_drawdown_r", 0.0),
        "net_delta": risk.get("net_delta", 0.0),
        "net_vega": risk.get("net_vega", 0.0),
        "net_theta": risk.get("net_theta", 0.0),
        "net_gamma": risk.get("net_gamma", 0.0),
        "most_dangerous_position_id": risk.get("most_dangerous_position_id", ""),
        "most_dangerous_position_risk_fraction": risk.get(
            "most_dangerous_position_risk_fraction", 0.0),
        "correlated_clusters": risk.get("correlated_clusters") or [],
        "kill_switches": risk.get("kill_switches") or [],
    }

    # Patterns panel
    patterns = summary.get("active_patterns") or []
    patterns_panel = {
        "n_active": len(patterns),
        "patterns": [
            {
                "name": p.get("pattern_name"),
                "confidence": p.get("confidence"),
                "implied_mm_intent": p.get("implied_mm_intent"),
                "suggested_trade": p.get("suggested_trade"),
                "suggested_avoid": p.get("suggested_avoid"),
                "evidence": p.get("evidence") or [],
            }
            for p in patterns
        ],
    }

    # Hedge panel
    hedge = summary.get("hedge_proposal") or {}
    hedge_panel = {
        "proposed": hedge.get("proposed", False),
        "proposals": hedge.get("proposals") or [],
        "reasons": hedge.get("reasons") or [],
        "total_premium_rupees": hedge.get("total_premium_rupees", 0.0),
    }

    # Positions panel
    open_positions = summary.get("open_positions") or []
    positions_panel = {
        "n_open": summary.get("n_open", 0),
        "open": list(open_positions),
        "n_closed": summary.get("n_closed", 0),
    }

    # P&L panel
    pnl_panel = {
        "daily_pnl_rupees": intent_dict.get("daily_pnl_rupees", 0.0),
        "cumulative_fees_rupees": intent_dict.get("cumulative_fees_rupees", 0.0),
    }
    ledger_summary = summary.get("ledger_summary") or {}
    if ledger_summary:
        pnl_panel["win_rate"] = ledger_summary.get("win_rate")
        pnl_panel["best_trade_rupees"] = ledger_summary.get("best_trade_rupees")
        pnl_panel["worst_trade_rupees"] = ledger_summary.get("worst_trade_rupees")
        pnl_panel["total_realized_rupees"] = ledger_summary.get("total_realized_rupees")

    # Live Capital panel (broker-sourced; founder requested 2026-06-22).
    capital = intent_dict.get("broker_capital") or {}
    capital_panel = {
        "starting_capital_rupees": capital.get("starting_capital_rupees", 0.0),
        "available_rupees": capital.get("available_rupees", 0.0),
        "used_margin_rupees": capital.get("used_margin_rupees", 0.0),
        "current_total_rupees": capital.get("current_total_rupees", 0.0),
        "daily_pnl_rupees": intent_dict.get("daily_pnl_rupees", 0.0),
    }

    # Live Trades tape (newest first) — opens + closes from the manager.
    recent = summary.get("recent_trades") or []
    trades_panel = {
        "n_recent": len(recent),
        "trades": list(recent)[:15],
    }

    explainer_text = explain_tick(intent_dict)

    return CockpitSnapshot(
        ts=str(intent_dict.get("ts") or ""),
        bar_index=int(intent_dict.get("bar_index") or 0),
        action_card=action_card,
        web_panel=web_panel,
        mm_panel=mm_panel,
        fat_tail_dial=fat_tail_dial,
        crowd_panel=crowd_panel,
        risk_panel=risk_panel,
        patterns_panel=patterns_panel,
        hedge_panel=hedge_panel,
        positions_panel=positions_panel,
        pnl_panel=pnl_panel,
        explainer_text=explainer_text,
        capital_panel=capital_panel,
        trades_panel=trades_panel,
    )
