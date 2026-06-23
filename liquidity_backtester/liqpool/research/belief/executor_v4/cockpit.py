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
    # Workaround panels (2026-06-22)
    latency_panel: Dict[str, Any] = field(default_factory=dict)
    attribution_panel: Dict[str, Any] = field(default_factory=dict)
    # Cockpit restoration (founder 2026-06-22): bring back the dev-of-dev
    # surface + calibrator transparency + exit reasoning + projection
    # divergence + multi-timeframe per-tf breakdown.
    substrate_panel: Dict[str, Any] = field(default_factory=dict)
    dod_heatmap_panel: Dict[str, Any] = field(default_factory=dict)
    mtf_panel: Dict[str, Any] = field(default_factory=dict)
    calibration_panel: Dict[str, Any] = field(default_factory=dict)
    weight_evolution_panel: Dict[str, Any] = field(default_factory=dict)
    exit_decision_panel: Dict[str, Any] = field(default_factory=dict)
    projection_panel: Dict[str, Any] = field(default_factory=dict)
    regime_history_panel: Dict[str, Any] = field(default_factory=dict)
    bootstrap_panel: Dict[str, Any] = field(default_factory=dict)
    # Tier-2: Belief Rehearsal Ensemble — kNN off-policy evaluation.
    rehearsal_panel: Dict[str, Any] = field(default_factory=dict)
    # Tier-2: Multi-leg structures (iron condor, jade lizard, ...).
    multi_leg_panel: Dict[str, Any] = field(default_factory=dict)
    # Tier-2 part 3: Contextual learner panel.
    contextual_learner_panel: Dict[str, Any] = field(default_factory=dict)
    # Tier-3: BeliefWebV2 — confidence intervals + causal graph +
    # lifecycle phases + information gain.
    belief_web_v2_panel: Dict[str, Any] = field(default_factory=dict)
    # Memory diagnostics (RAM-leak triage 2026-06-22).
    memory_panel: Dict[str, Any] = field(default_factory=dict)
    # ManipulationV2 — Tier-3 authoritative manipulation engine.
    manipulation_v2_panel: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _build_manipulation_v2_panel(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Project manipulation_v2 summary into the cockpit panel shape."""
    mv2 = summary.get("manipulation_v2") or {}
    last = mv2.get("last_intent") or {}
    calib = mv2.get("calibrator") or {}
    return {
        "fire_count": last.get("fire_count", 0),
        "direction": last.get("direction", 0),
        "confidence": last.get("confidence", 0.0),
        "regime": last.get("regime", "unknown"),
        "gamma_regime": last.get("gamma_regime", "unknown"),
        "horizon_bars": last.get("horizon_bars", 0),
        "targeted_strike": last.get("targeted_strike"),
        "composite_score": last.get("composite_score", 0.0),
        "per_detector": dict(last.get("per_detector") or {}),
        "notes": list(last.get("notes") or []),
        "operator_override": last.get("operator_override"),
        "calibrator_weights": dict(calib.get("weights") or {}),
        "calibrator_per_detector": dict(calib.get("per_detector") or {}),
        "n_recent_outcomes": calib.get("n_recent_outcomes", 0),
        "n_snapshots_held": calib.get("snapshots_held", 0),
    }


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
        "directional_consensus_horizon_weighted": web.get(
            "directional_consensus_horizon_weighted", 0.0),
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
        # Founder ask 2026-06-22: ALL scenarios for the pulsing-web view.
        "all_scenarios": [
            {
                "name": sc.get("name"),
                "family": sc.get("family"),
                "probability": sc.get("current_probability"),
                "direction": sc.get("implied_direction"),
                "horizon_bars": sc.get("implied_horizon_bars"),
                "recent_probabilities": sc.get("recent_probabilities") or [],
            }
            for sc in (web.get("all_scenarios") or [])
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

    # Latency panel (workaround — founder 2026-06-22).
    lat = intent_dict.get("latency_summary") or {}
    latency_panel = {
        "n_samples": lat.get("n_samples", 0),
        "total_ms_p50": (lat.get("total_ms") or {}).get("p50_ms"),
        "total_ms_p95": (lat.get("total_ms") or {}).get("p95_ms"),
        "total_ms_max": (lat.get("total_ms") or {}).get("max_ms"),
        "manager_ms_p95": (lat.get("manager_evaluate") or {}).get("p95_ms"),
        "broker_ms_p95": (lat.get("broker_routing") or {}).get("p95_ms"),
        "persist_ms_p95": (lat.get("persistence") or {}).get("p95_ms"),
    }

    # Per-strategy attribution (workaround C).
    attribution = summary.get("strategy_attribution") or {}
    attribution_panel = {
        "n_strategies_seen": attribution.get("n_strategies_seen", 0),
        "rows": (attribution.get("rows") or [])[:12],
        "slippage_tracker": summary.get("slippage_tracker") or {},
    }

    # ── Restoration panels (founder 2026-06-22) ──────────────────────
    # Substrate panel — the dev-of-dev family the cockpit was dropping.
    rich = summary.get("rich_context") or {}
    substrate_panel = {
        "regime_stability_index": rich.get("regime_stability_index", 0.0),
        "epicenter_label": rich.get("epicenter_label", ""),
        "epicenter_level": rich.get("epicenter_level", 0),
        "epicenter_migration_distance": rich.get(
            "epicenter_migration_distance", 0.0),
        "thesis_velocity_dominant_side": rich.get(
            "thesis_velocity_dominant_side", "flat"),
        "ce_signed_z_velocity": rich.get("ce_signed_z_velocity", 0.0),
        "ce_signed_z_acceleration": rich.get("ce_signed_z_acceleration", 0.0),
        "pe_signed_z_velocity": rich.get("pe_signed_z_velocity", 0.0),
        "pe_signed_z_acceleration": rich.get("pe_signed_z_acceleration", 0.0),
        "net_intent_velocity": rich.get("net_intent_velocity", 0.0),
        "net_intent_acceleration": rich.get("net_intent_acceleration", 0.0),
        "thesis_bull_velocity": rich.get("thesis_bull_velocity", 0.0),
        "thesis_bear_velocity": rich.get("thesis_bear_velocity", 0.0),
        "dispersion_velocity": rich.get("dispersion_velocity", 0.0),
    }
    # Dev-of-dev heatmap panel — slot labels + dod_z values for SVG render.
    dod_heatmap_panel = {
        "values": list(rich.get("dod_heatmap") or []),
        "labels": list(rich.get("dod_heatmap_labels") or []),
    }

    # Multi-timeframe alignment panel.
    mtf = summary.get("mtf_alignment") or {}
    mtf_panel = {
        "alignment_ok_long": mtf.get("alignment_ok_long", False),
        "alignment_score_long": mtf.get("alignment_score_long", 0.0),
        "alignment_ok_short": mtf.get("alignment_ok_short", False),
        "alignment_score_short": mtf.get("alignment_score_short", 0.0),
        "confirmation_count_long": mtf.get("confirmation_count_long", 0),
        "confirmation_count_short": mtf.get("confirmation_count_short", 0),
        "per_timeframe": dict(mtf.get("per_timeframe") or {}),
    }

    # Live calibration panel — what the calibrator did at the last close.
    calib = summary.get("live_calibration") or {}
    calibration_panel = {
        "has_event": bool(calib),
        "proposed": calib.get("proposed", False),
        "applied": calib.get("applied", False),
        "paused": calib.get("paused", False),
        "rejected_reason": calib.get("rejected_reason", ""),
        "pre_weights": dict(calib.get("pre_weights") or {}),
        "post_weights": dict(calib.get("post_weights") or {}),
        "deltas": dict(calib.get("deltas") or {}),
        "train_loss": calib.get("train_loss", 0.0),
        "val_loss": calib.get("val_loss", 0.0),
        "notes": list(calib.get("notes") or []),
    }

    # Weight evolution panel — trends + warnings + adaptability.
    wevo = summary.get("weight_evolution") or {}
    weight_evolution_panel = {
        "calibrator_paused": wevo.get("calibrator_paused", False),
        "n_snapshots": wevo.get("n_snapshots", 0),
        "adaptability_index": wevo.get("adaptability_index", 0.0),
        "trend_per_weight": dict(wevo.get("trend_per_weight") or {}),
        "most_drifting_weight": wevo.get("most_drifting_weight", ""),
        "coordinated_drift_score": wevo.get("coordinated_drift_score", 0.0),
        "val_loss_trend": wevo.get("val_loss_trend", 0.0),
        "warnings": list(wevo.get("warnings") or []),
    }

    # Adaptive-exit decision panel — portfolio-level exit coordinator
    # outputs the per-position list, plus cluster sizing + which side is
    # winning + which positions had their modifications approved.
    aex = summary.get("adaptive_exit") or {}
    exit_decision_panel = {
        "has_decision": bool(aex),
        "per_position": list(aex.get("per_position") or []),
        "cluster_bullish_count": aex.get("cluster_bullish_count", 0),
        "cluster_bearish_count": aex.get("cluster_bearish_count", 0),
        "weakest_thesis_position_id": aex.get(
            "weakest_thesis_position_id", ""),
        "modifications_approved_this_tick": list(
            aex.get("modifications_approved_this_tick") or []),
        "modifications_deferred_this_tick": list(
            aex.get("modifications_deferred_this_tick") or []),
        "regime_winning_side": aex.get("regime_winning_side", 0),
        "notes": list(aex.get("notes") or []),
    }

    # Projection panel — calibration tape summary (target/stop/neither
    # resolution rate from prior trades). The mean_realized_r tells us
    # whether the projection is, on average, calling the right shots.
    proj = summary.get("projection_summary") or {}
    projection_panel = {
        "tape_size": proj.get("tape_size", 0),
        "n_target_hits": proj.get("n_target_hits", 0),
        "n_stop_hits": proj.get("n_stop_hits", 0),
        "n_neither": proj.get("n_neither", 0),
        "mean_realized_r": proj.get("mean_realized_r", 0.0),
    }

    # Regime history panel — per-family win rates from yesterday onwards.
    regime_hist = summary.get("regime_win_history") or {}
    regime_history_panel = {
        "days_loaded": regime_hist.get("days_loaded", 0),
        "by_family": dict(regime_hist.get("by_family") or {}),
        "today_dominant_family": regime_hist.get("today_dominant_family", ""),
        "today_confidence_adjustment": regime_hist.get(
            "today_confidence_adjustment", 0.0),
        "notes": list(regime_hist.get("notes") or []),
    }

    # Bootstrap panel — what the calibrator inherited from prior days.
    boot = summary.get("calibrator_bootstrap") or {}
    bootstrap_panel = {
        "ran": boot.get("ran", False),
        "days_loaded": boot.get("days_loaded", 0),
        "observations_replayed": boot.get("observations_replayed", 0),
        "updates_applied": boot.get("updates_applied", 0),
        "seeded_weights": dict(boot.get("seeded_weights") or {}),
        "notes": list(boot.get("notes") or []),
    }

    # Belief Web v2 panel (Tier-3): confidence intervals + causal graph
    # + lifecycle phases + information gain + predicted resolutions.
    bw2 = summary.get("belief_web_v2") or {}
    belief_web_v2_panel = {
        "ready": bw2.get("ready", False),
        "bar_index": bw2.get("bar_index", 0),
        "n_active": bw2.get("n_active", 0),
        "scenarios_with_ci": list(bw2.get("scenarios_with_ci") or [])[:10],
        "causal_graph": dict(bw2.get("causal_graph") or {}),
        "conditional_table": dict(bw2.get("conditional_table") or {}),
        "markov_table": dict(bw2.get("markov_table") or {}),
        "lifecycle_distribution": dict(
            bw2.get("lifecycle_distribution") or {}),
        "most_informative_this_tick": list(
            bw2.get("most_informative_this_tick") or []),
        "total_information_gain": bw2.get("total_information_gain", 0.0),
        "coherence_score": bw2.get("coherence_score", 0.0),
        "surprise_score": bw2.get("surprise_score", 0.0),
        "predicted_resolutions": list(
            bw2.get("predicted_resolutions") or [])[:8],
        "propagation_deltas": dict(bw2.get("propagation_deltas") or {}),
        "resolution_memory": dict(bw2.get("resolution_memory") or {}),
        "notes": list(bw2.get("notes") or []),
    }

    # Contextual learner panel (Tier-2 part 3): per-regime conditional
    # weights, Shapley attributions per recent closure, recency-weighted
    # effective sample counts.
    ctx = summary.get("contextual_learner") or {}
    contextual_learner_panel = {
        "enabled": ctx.get("enabled", False),
        "current_family": ctx.get("current_family", "unknown"),
        "applied_weights": dict(ctx.get("applied_weights") or {}),
        "global_weights": dict(ctx.get("global_weights") or {}),
        "per_family_weights": dict(ctx.get("per_family_weights") or {}),
        "per_family_n_samples": dict(ctx.get("per_family_n_samples") or {}),
        "per_family_effective_n": dict(
            ctx.get("per_family_effective_n") or {}),
        "recent_closure_reports": list(
            ctx.get("recent_closure_reports") or []),
        "min_samples_to_specialise": ctx.get(
            "min_samples_to_specialise", 8),
        "recency_half_life_days": ctx.get("recency_half_life_days", 4.0),
        "last_closure_report": ctx.get("last_closure_report") or {},
    }

    # Multi-leg bundles panel (Tier-2).
    ml = summary.get("multi_leg_bundles") or {}
    last_outcome = summary.get("last_bundle_outcome") or {}
    multi_leg_panel = {
        "n_open_bundles": ml.get("n_open_bundles", 0),
        "n_closed_bundles": ml.get("n_closed_bundles", 0),
        "open": list(ml.get("open") or []),
        "closed_recent": list(ml.get("closed_recent") or []),
        "wins": ml.get("wins", 0),
        "losses": ml.get("losses", 0),
        "total_realised_rupees": ml.get("total_realised_rupees", 0.0),
        "win_rate": ml.get("win_rate", 0.0),
        "last_bundle_outcome": dict(last_outcome) if last_outcome else {},
    }

    # Rehearsal panel (Tier-2): conditional-kNN off-policy evaluation.
    # The most-recent rehearsal decision with per-perturbation outcome
    # distribution + feature weights so the operator can SEE which
    # features the ensemble considers important right now.
    rh = summary.get("rehearsal_ensemble") or {}
    last_dec = rh.get("last_decision") or {}
    rehearsal_panel = {
        "booted": rh.get("booted", False),
        "n_observations_in_ring": rh.get("n_observations_in_ring", 0),
        "history_days_loaded": rh.get("history_days_loaded", 0),
        "feature_weights": dict(rh.get("feature_weights") or {}),
        "has_decision": bool(last_dec),
        "ran": last_dec.get("ran", False),
        "rehearsal_score": last_dec.get("rehearsal_score", 0.50),
        "confidence": last_dec.get("confidence", 0.0),
        "n_analogues_total": last_dec.get("n_analogues_total", 0),
        "mean_similarity": last_dec.get("mean_similarity", 0.0),
        "recommended_action": last_dec.get("recommended_action", ""),
        "best_perturbation_name": last_dec.get(
            "best_perturbation_name", ""),
        "current_perturbation_name": last_dec.get(
            "current_perturbation_name", "as_proposed"),
        "per_perturbation": list(last_dec.get("per_perturbation") or []),
        "notes": list(last_dec.get("notes") or []),
        "deferral_reason": last_dec.get("deferral_reason", ""),
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
        latency_panel=latency_panel,
        attribution_panel=attribution_panel,
        substrate_panel=substrate_panel,
        dod_heatmap_panel=dod_heatmap_panel,
        mtf_panel=mtf_panel,
        calibration_panel=calibration_panel,
        weight_evolution_panel=weight_evolution_panel,
        exit_decision_panel=exit_decision_panel,
        projection_panel=projection_panel,
        regime_history_panel=regime_history_panel,
        bootstrap_panel=bootstrap_panel,
        rehearsal_panel=rehearsal_panel,
        multi_leg_panel=multi_leg_panel,
        contextual_learner_panel=contextual_learner_panel,
        belief_web_v2_panel=belief_web_v2_panel,
        memory_panel=dict(intent_dict.get("memory_summary") or {}),
        manipulation_v2_panel=_build_manipulation_v2_panel(summary),
    )
