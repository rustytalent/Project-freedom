"""PortfolioManager — the v4 executor's orchestrator (Sprint 1).

This is the entry point. It wires together the Sprint-1 foundation
modules (substrate, multi-timeframe memory, flow memory, economics,
hypothesis, ledger) into a single per-tick decision flow.

Per-tick algorithm (high-level):
  1. Substrate.augment(snapshot) → RichContext
  2. MultiTimeframeMemory.observe(snapshot)
  3. FlowMemory.observe(snapshot) → FlowEvent
  4. For each open position:
       - update ledger BarRecord
       - check hard exits (premium stop / spread danger / etc.)
       - check soft exits (after min hold)
       - check exit-accelerator (cut losers fast)
  5. Portfolio kill-switches (daily bleed, max-open-positions, etc.)
  6. If decision says ENTER:
       - multi-timeframe alignment veto
       - temporal contradiction check (from flow memory)
       - economics EV gate (fees + slippage + edge floor)
       - antithesis estimate from flow memory
       - if all pass → build PositionHypothesis → open ledger
  7. Emit PortfolioIntent (per-position intents + portfolio summary)

This is Sprint 1's scope. Sprint 2 swaps in scenario_web for richer
forward-projection. Sprint 3 adds manipulation patterns + MM-mind.
Sprint 4 adds the strategy library + hedge layer. Sprint 5 polishes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .aggregator import (
    AGGR_ACCEPT,
    AGGR_REFUSE,
    AGGR_SIZE_DOWN,
    AggregatorConfig,
    DecisionAggregator,
)
from .counterfactual import (
    CounterfactualConfig,
    CounterfactualGenerator,
    CounterfactualPlan,
    SEVERITY_HARD,
    SEVERITY_SOFT,
)
from .exit_engine import (
    AdaptiveExitEngine,
    AdaptiveExitEngineConfig,
    MarketState,
    PortfolioExitManager,
    PortfolioExitManagerConfig,
    build_exit_thesis,
)
from .critic import (
    AdversarialCritic,
    CriticConfig,
    CritiqueResult,
)
from .crowd_mirror import (
    CrowdMirror,
    CrowdMirrorConfig,
    CrowdMirrorReport,
)
from .fat_tail_amplifier import (
    FatTailAmplifier,
    FatTailAmplifierConfig,
    FatTailScore,
    TAIL_ACTION_REFUSE,
)
from .manipulation_patterns import (
    ManipulationBoard,
    ManipulationBoardConfig,
    PatternMatch,
)
from .hedge import (
    HedgeProposal,
    HedgeProposalConfig,
    HedgeProposer,
)
from .market_maker_mind import (
    MMPosterior,
    MarketMakerMind,
    MarketMakerMindConfig,
)
from .risk import (
    PortfolioRiskConfig,
    PortfolioRiskLayer,
    PortfolioRiskReport,
)
from .conviction import ConvictionScore, compute_conviction
from .economics import (
    EVDecision,
    ExecutionEconomicsConfig,
    ExitAcceleration,
    daily_bleed_blocker,
    expected_value_after_costs,
    fees_for_round_trip,
    estimate_slippage,
    minimum_profitable_premium_delta,
    should_accelerate_exit,
)
from .flow_memory import FlowEvent, FlowMemory, FlowMemoryConfig
from .hypothesis import (
    PositionHypothesis,
    build_hypothesis_from_snapshot,
)
from .ledger import (
    BarRecord,
    LedgerOutcome,
    LedgerStore,
    PositionLedger,
)
from .memory import MultiTimeframeMemory, TimeframeLevel
from .projection import (
    ForwardProjection,
    ProjectionConfig,
    ProjectionRecord,
)
from .scenario_web import (
    ScenarioWeb,
    ScenarioWebConfig,
    STRAT_LONG_CE,
    STRAT_LONG_PE,
    WebSnapshot,
)
from .substrate import RichContext, SubstrateConfig, SubstrateState


# Intent kinds emitted by the manager (per-position).
INTENT_OPEN_LONG = "OPEN_LONG"
INTENT_OPEN_SHORT = "OPEN_SHORT"
INTENT_OPEN_SCALP_CALL = "OPEN_SCALP_CALL"
INTENT_OPEN_SCALP_PUT = "OPEN_SCALP_PUT"
INTENT_HOLD = "HOLD"
INTENT_EXIT = "EXIT"
INTENT_REFUSE = "REFUSE"
INTENT_FLAT = "FLAT"

PROFILE_SCALP = "SCALP"
PROFILE_INTRADAY = "INTRADAY"

ENTRY_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "SCALP_CALL", "SCALP_PUT"}
SCALP_ENTRY_ACTIONS = {"SCALP_CALL", "SCALP_PUT"}
DIRTY_IV_STATES = {"dirty_data", "liquidity_distortion", "common_shock"}
DANGEROUS_BATTLEFIELD = {"single_distortion", "vol_expansion"}
SPREAD_DANGEROUS = "dangerous"
DIRTY_MARK_SOURCES = {"invalid", "last_valid", "ltp"}


@dataclass
class PortfolioManagerConfig:
    """Top-level knobs for the portfolio manager.

    Composes the per-module configs. Defaults are founder-calibrated for
    NIFTY weekly options with the operator's risk budget of ₹500-1000/trade
    and daily bleed floor of ₹5000.
    """
    economics: ExecutionEconomicsConfig = field(default_factory=ExecutionEconomicsConfig)
    substrate: SubstrateConfig = field(default_factory=SubstrateConfig)
    flow_memory: FlowMemoryConfig = field(default_factory=FlowMemoryConfig)
    scenario_web: ScenarioWebConfig = field(default_factory=ScenarioWebConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    counterfactual: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    # Sprint 3 layers
    manipulation_board: ManipulationBoardConfig = field(
        default_factory=ManipulationBoardConfig)
    market_maker_mind: MarketMakerMindConfig = field(
        default_factory=MarketMakerMindConfig)
    fat_tail_amplifier: FatTailAmplifierConfig = field(
        default_factory=FatTailAmplifierConfig)
    crowd_mirror: CrowdMirrorConfig = field(default_factory=CrowdMirrorConfig)
    # Sprint 4 layers
    portfolio_risk: PortfolioRiskConfig = field(
        default_factory=PortfolioRiskConfig)
    hedge: HedgeProposalConfig = field(default_factory=HedgeProposalConfig)
    # Adaptive Exit Quote Engine
    exit_engine: AdaptiveExitEngineConfig = field(
        default_factory=AdaptiveExitEngineConfig)
    portfolio_exit: PortfolioExitManagerConfig = field(
        default_factory=PortfolioExitManagerConfig)
    min_warm_bars: int = 80
    # Dead-market guard (founder hot-fix 2026-06-22 LIVE): when the
    # scenario web reads strongly chop/manipulation-dominant, REFUSE
    # directional entries. Multi-leg neutral strategies (iron condor,
    # jade lizard, butterfly) are in the library but not yet routed
    # through the broker (that's a separate change). Until they are,
    # taking no trade in chop is the right move.
    dead_market_refuse_directional: bool = True
    dead_market_chop_threshold: float = 0.50
    dead_market_manipulation_threshold: float = 0.40
    dead_market_low_consensus_threshold: float = 0.10
    dead_market_tail_threshold: float = 0.30
    # Decision gates
    min_entry_confidence: float = 0.55   # founder hot-fix 2026-06-22 LIVE (was 0.66)
    min_scalp_confidence: float = 0.62   # founder hot-fix 2026-06-22 LIVE (was 0.72)
    min_clean_mark_fraction: float = 0.85
    min_spread_friendliness: float = 0.55
    max_abnormal_slot_fraction: float = 0.40
    max_no_trade_score: float = 18.0
    # Targets / stops (in PREMIUM space; R = 1.0 = the stop)
    scalp_target_r: float = 1.20
    intraday_target_r: float = 1.80
    scalp_max_bars: int = 24
    intraday_max_bars: int = 90
    min_hold_bars: int = 2
    trail_after_r: float = 0.80
    profit_lock_start_r: float = 0.90
    profit_lock_giveback_r: float = 0.45
    confidence_giveback: float = 0.30
    # Sizing
    max_lots: int = 4
    # Cooldowns
    cooldown_bars_after_entry: int = 8
    cooldown_bars_after_exit: int = 6


@dataclass(frozen=True)
class PositionIntent:
    """Per-position intent emitted this tick."""
    position_id: str
    intent: str
    contract_side: str
    contract_label: str
    strike_price: float
    direction: int
    current_r: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["notes"] = list(self.notes)
        return d


@dataclass(frozen=True)
class PortfolioIntent:
    """Per-tick portfolio-level output."""
    ts: Any
    bar_index: int
    new_entry: Optional[Dict[str, Any]]              # full hypothesis if a new position opened
    position_updates: List[Dict[str, Any]]            # per-position intent dicts
    closed_this_tick: List[Dict[str, Any]]             # ledger outcomes for any closures
    refuse_reasons: List[str]                          # why we refused an engine ENTER if applicable
    portfolio_summary: Dict[str, Any]
    daily_pnl_rupees: float
    cumulative_fees_rupees: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": str(self.ts),
            "bar_index": self.bar_index,
            "new_entry": self.new_entry,
            "position_updates": list(self.position_updates),
            "closed_this_tick": list(self.closed_this_tick),
            "refuse_reasons": list(self.refuse_reasons),
            "portfolio_summary": dict(self.portfolio_summary),
            "daily_pnl_rupees": self.daily_pnl_rupees,
            "cumulative_fees_rupees": self.cumulative_fees_rupees,
            "notes": list(self.notes),
        }


@dataclass
class _OpenPositionState:
    """Mutable state per open position (kept in the manager, parallel to
    the ledger which is the immutable audit record)."""
    hypothesis: PositionHypothesis
    entry_bar: int
    last_bar: int
    best_r: float = 0.0
    worst_r: float = 0.0
    high_water_confidence: float = 0.0
    low_water_confidence: float = 1.0
    last_premium: float = 0.0


class PortfolioManager:
    """The Sprint-1 portfolio-aware executor.

    Wires substrate + multi-tf memory + flow memory + economics + hypothesis
    + ledger together. Single entry point: ``evaluate(snapshot, held_quotes)``.

    ``held_quotes`` is a dict {position_id: HeldContractRead} from the
    Sprint-1 caller (typically the live runner). For pre-open the manager
    pulls premium from the snapshot's slot_readings for the entry-target
    contract.
    """

    def __init__(self, cfg: Optional[PortfolioManagerConfig] = None) -> None:
        self.cfg = cfg or PortfolioManagerConfig()
        self.substrate_state = SubstrateState(self.cfg.substrate)
        self.mtf = MultiTimeframeMemory()
        self.flow = FlowMemory(self.cfg.flow_memory)
        # Sprint 2 layers
        self.web = ScenarioWeb(self.cfg.scenario_web)
        self.critic = AdversarialCritic(self.cfg.critic)
        self.counterfactual = CounterfactualGenerator(self.cfg.counterfactual)
        self.projection = ForwardProjection(self.cfg.projection)
        self.aggregator = DecisionAggregator(self.cfg.aggregator)
        # Live weight evolution memory + calibrator (founder Sun 2026-06-22)
        from .weight_evolution import (
            WeightEvolutionMemory, WeightEvolutionMemoryConfig,
            WeightEvolutionAnalyzer,
        )
        from .live_calibrator import LiveCalibrator, LiveCalibratorConfig
        self.weight_memory = WeightEvolutionMemory(
            WeightEvolutionMemoryConfig())
        self.weight_analyzer = WeightEvolutionAnalyzer()
        self.live_calibrator = LiveCalibrator(
            aggregator_cfg=self.cfg.aggregator,
            memory=self.weight_memory,
            calibrator_cfg=LiveCalibratorConfig(enabled=True),
        )
        self._last_calibration: Optional[Dict[str, Any]] = None
        # Recent-trades tape (last N opens + closes, newest first) — feeds
        # the cockpit's Live Trades panel.
        from collections import deque as _deque
        self._recent_trades: _deque = _deque(maxlen=30)
        # Slippage feedback (workaround A): observed slippage feeds the
        # EV gate so it stops underestimating real execution cost.
        from .monday_bonuses import SlippageTracker
        self.slippage_tracker = SlippageTracker()
        # Per-strategy attribution memory (workaround C): rolling
        # realized-rupees + win counts per strategy name. Built from
        # closed-position outcomes; surfaced in portfolio_summary.
        from collections import defaultdict as _ddict
        self._strategy_stats: dict = _ddict(
            lambda: {"n_trades": 0, "wins": 0, "losses": 0,
                      "total_realized_rupees": 0.0, "total_fees_rupees": 0.0,
                      "best_trade": 0.0, "worst_trade": 0.0,
                      "avg_realized_r": 0.0, "running_r_sum": 0.0})
        # Sprint 3 layers
        self.manipulation_board = ManipulationBoard(self.cfg.manipulation_board)
        self.mm_mind = MarketMakerMind(self.cfg.market_maker_mind)
        self.fat_tail_amp = FatTailAmplifier(self.cfg.fat_tail_amplifier)
        self.crowd_mirror = CrowdMirror(self.cfg.crowd_mirror)
        self._last_mm_posterior: Optional[MMPosterior] = None
        self._last_tail_score: Optional[FatTailScore] = None
        self._last_crowd_report: Optional[CrowdMirrorReport] = None
        self._last_patterns: List[PatternMatch] = []
        # Sprint 4 layers
        self.portfolio_risk = PortfolioRiskLayer(self.cfg.portfolio_risk)
        self.hedge_proposer = HedgeProposer(self.cfg.hedge)
        self._last_risk_report: Optional[PortfolioRiskReport] = None
        self._last_hedge_proposal: Optional[HedgeProposal] = None
        # Ledger + state
        self.ledger_store = LedgerStore()
        self._open_states: Dict[str, _OpenPositionState] = {}
        self.cooldown_until_bar: int = -1
        self.daily_pnl_rupees: float = 0.0
        self.cumulative_fees_rupees: float = 0.0
        # Snapshot caches: latest web/critic for ledger persistence
        self._last_web_snapshot: Optional[WebSnapshot] = None
        # Per-position counterfactual plans (position_id → plan).
        self._counterfactual_plans: Dict[str, CounterfactualPlan] = {}
        # Adaptive Exit Quote Engine — coordinates per-position exit
        # ghost prices, modification budget (Zerodha 25-mod cap), and
        # cross-position priority.
        self.portfolio_exit = PortfolioExitManager(self.cfg.portfolio_exit)
        self._last_exit_decision: Optional[Dict[str, Any]] = None

    # ── public API ──────────────────────────────────────────────────────

    def evaluate(self,
                 snapshot: Dict[str, Any],
                 *,
                 held_quotes: Optional[Dict[str, Any]] = None,
                 ) -> PortfolioIntent:
        """One belief snapshot → one portfolio decision tick."""
        cfg = self.cfg
        bar_index = int(_num(snapshot.get("bars_seen"), 0))
        ts = snapshot.get("ts")

        # 1. Information substrate update
        rich = self.substrate_state.observe(snapshot)
        # 2. Multi-timeframe memory
        self.mtf.observe(snapshot)
        # 3. Flow memory
        flow_event = self.flow.observe(snapshot)
        # 3a. Scenario web — live multi-scenario tracking (Sprint 2)
        mtf_views_dict = {k: v.to_dict() for k, v
                           in self.mtf.all_views().items()}
        web_snap = self.web.observe(snapshot, rich, flow_event, mtf_views_dict)
        self._last_web_snapshot = web_snap

        # 3b. Sprint 3 — manipulation board + MM mind + fat-tail + crowd mirror.
        patterns = self.manipulation_board.scan(
            flow_memory=self.flow, rich_context=rich,
            snapshot=snapshot, near_expiry=False,
        )
        self._last_patterns = patterns
        mm_posterior = self.mm_mind.infer(patterns=patterns,
                                            flow_event=flow_event)
        self._last_mm_posterior = mm_posterior
        crowd_report = self.crowd_mirror.inspect(
            [s.hypothesis for s in self._open_states.values()],
        )
        self._last_crowd_report = crowd_report
        tail_score = self.fat_tail_amp.amplify(
            patterns=patterns, mm_posterior=mm_posterior,
            flow_event=flow_event, rich_context=rich,
            iv_state=str(_map(snapshot.get("iv_state")).get("state") or ""),
            crowd_density=crowd_report.retail_similarity_score,
        )
        self._last_tail_score = tail_score

        notes: List[str] = []
        closed_this_tick: List[Dict[str, Any]] = []
        position_updates: List[Dict[str, Any]] = []
        current_r_by_position: Dict[str, float] = {}

        # 4. Update + check every open position
        for pos_id, state in list(self._open_states.items()):
            held = (held_quotes or {}).get(pos_id)
            held_premium = self._resolve_held_premium(state, snapshot, held)

            current_r = self._position_r(state, held_premium)
            current_r_by_position[pos_id] = current_r
            state.best_r = max(state.best_r, current_r)
            state.worst_r = min(state.worst_r, current_r)
            confidence_now = _num(_map(snapshot.get("decision")).get("confidence"))
            state.high_water_confidence = max(state.high_water_confidence,
                                                confidence_now)
            state.low_water_confidence = min(state.low_water_confidence,
                                              confidence_now)
            if held_premium is not None:
                state.last_premium = held_premium

            # Append a BarRecord to the ledger.
            ledger = self.ledger_store.find_open(pos_id)
            if ledger is not None:
                self._append_bar_record(
                    ledger=ledger, state=state, snapshot=snapshot,
                    held_premium=held_premium, current_r=current_r,
                    confidence=confidence_now, held_read=held,
                    bar_index=bar_index,
                )

            # Check exit reasons.
            exit_reasons, severity = self._exit_check(
                state=state, snapshot=snapshot, held_read=held,
                held_premium=held_premium, current_r=current_r,
                confidence=confidence_now, bar_index=bar_index,
            )
            if exit_reasons:
                # Close.
                outcome = self._close_position(
                    pos_id=pos_id, state=state, snapshot=snapshot,
                    held_premium=held_premium, held_read=held,
                    current_r=current_r, exit_reasons=exit_reasons,
                    severity=severity, bar_index=bar_index,
                )
                if outcome is not None:
                    closed_this_tick.append({
                        "position_id": pos_id,
                        "outcome": outcome.to_dict(),
                    })
                position_updates.append(PositionIntent(
                    position_id=pos_id,
                    intent=INTENT_EXIT,
                    contract_side=state.hypothesis.contract_side,
                    contract_label=state.hypothesis.contract_label,
                    strike_price=state.hypothesis.strike_price,
                    direction=state.hypothesis.direction,
                    current_r=round(current_r, 3),
                    notes=exit_reasons,
                ).to_dict())
                state.last_bar = bar_index
                continue

            position_updates.append(PositionIntent(
                position_id=pos_id, intent=INTENT_HOLD,
                contract_side=state.hypothesis.contract_side,
                contract_label=state.hypothesis.contract_label,
                strike_price=state.hypothesis.strike_price,
                direction=state.hypothesis.direction,
                current_r=round(current_r, 3),
                notes=[f"best_r={state.best_r:.2f}, worst_r={state.worst_r:.2f}"],
            ).to_dict())
            state.last_bar = bar_index

        # 4a. Sprint 4 portfolio risk aggregation.
        risk_report = self.portfolio_risk.compute(
            open_states=self._open_states,
            current_r_by_position=current_r_by_position,
            greeks_by_position=None,    # populated once strategy_library wires legs
            lot_size=self.cfg.economics.lot_size,
        )
        self._last_risk_report = risk_report

        # 4b. Sprint 4 hedge proposer (Sprint 4 layer).
        strike_lookup = self._build_strike_lookup(snapshot)
        self._last_hedge_proposal = self.hedge_proposer.propose(
            fat_tail_action=(tail_score.recommended_action
                              if tail_score else "NORMAL"),
            net_delta_lots=risk_report.net_directional_exposure_lots,
            open_positions=[s.hypothesis for s in self._open_states.values()],
            strike_lookup=strike_lookup,
            lot_size=self.cfg.economics.lot_size,
        )

        # 4c. Adaptive Exit Quote Engine — for every open position,
        # update ghost exit + decide whether to spend a Zerodha mod.
        try:
            market_by_position = self._build_exit_markets(
                snapshot=snapshot, held_quotes=held_quotes,
                current_r_by_position=current_r_by_position,
                bar_index=bar_index,
            )
            if market_by_position:
                regime_side = int(_num(_map(snapshot.get("iv_state")).get(
                    "direction"), 0))
                exit_decision = self.portfolio_exit.evaluate(
                    market_by_position,
                    regime_winning_side=regime_side,
                )
                self._last_exit_decision = exit_decision.to_dict()
            else:
                self._last_exit_decision = None
        except Exception:
            # Exit engine is opt-in; never break the trading loop.
            self._last_exit_decision = None

        # 5. Portfolio kill-switches.
        refuse_reasons: List[str] = []
        decision = _map(snapshot.get("decision"))
        action = str(decision.get("action") or "")
        bleed_block = daily_bleed_blocker(self.daily_pnl_rupees, self.cfg.economics)
        if bleed_block:
            refuse_reasons.append(bleed_block)
        # Structural caps first — reported before downstream risk checks.
        if (action in ENTRY_ACTIONS
                and len(self._open_states) >= self.cfg.economics.max_open_positions):
            refuse_reasons.append(
                f"max open positions ({self.cfg.economics.max_open_positions}) reached"
            )
        # Sprint-3 fat-tail kill switch.
        if (self._last_tail_score is not None
                and self._last_tail_score.recommended_action == TAIL_ACTION_REFUSE):
            refuse_reasons.append(
                f"fat-tail amp REFUSE (tail_score="
                f"{self._last_tail_score.tail_score:.2f}): "
                + "; ".join(self._last_tail_score.notes[:2])
            )
        # Sprint-4 portfolio risk kill switches.
        for k in risk_report.kill_switches:
            refuse_reasons.append(f"portfolio risk: {k}")

        # 6. Maybe open a new entry.
        new_entry: Optional[Dict[str, Any]] = None
        if (not refuse_reasons
                and action in ENTRY_ACTIONS
                and bool(snapshot.get("is_warm"))
                and bar_index >= self.cooldown_until_bar
                and len(self._open_states) < self.cfg.economics.max_open_positions):

            new_entry_intent, additional_refuse = self._consider_entry(
                snapshot=snapshot, rich=rich,
                flow_event=flow_event, web_snap=web_snap,
                bar_index=bar_index, ts=ts,
            )
            if new_entry_intent is not None:
                new_entry = new_entry_intent
            else:
                refuse_reasons.extend(additional_refuse)
        elif action in ENTRY_ACTIONS and not snapshot.get("is_warm"):
            refuse_reasons.append("warmup incomplete")
        elif action in ENTRY_ACTIONS and bar_index < self.cooldown_until_bar:
            refuse_reasons.append(f"cooldown until bar {self.cooldown_until_bar}")

        return PortfolioIntent(
            ts=ts, bar_index=bar_index,
            new_entry=new_entry,
            position_updates=position_updates,
            closed_this_tick=closed_this_tick,
            refuse_reasons=refuse_reasons,
            portfolio_summary=self._portfolio_summary(),
            daily_pnl_rupees=round(self.daily_pnl_rupees, 2),
            cumulative_fees_rupees=round(self.cumulative_fees_rupees, 2),
            notes=notes,
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _consider_entry(self,
                         snapshot: Dict[str, Any],
                         rich: RichContext,
                         flow_event: FlowEvent,
                         web_snap: WebSnapshot,
                         bar_index: int,
                         ts: Any,
                         ) -> tuple[Optional[Dict[str, Any]], List[str]]:
        """Multi-gate entry pipeline. Returns (entry_dict, refuse_reasons)."""
        cfg = self.cfg
        decision = _map(snapshot.get("decision"))
        iv = _map(snapshot.get("iv_state"))
        bf = _map(snapshot.get("battlefield"))
        thesis = _map(snapshot.get("thesis"))

        action = str(decision.get("action") or "")
        direction = int(_num(decision.get("direction"), 0))
        confidence = _clamp(_num(decision.get("confidence")))

        refuse: List[str] = []

        # Gate A: data quality.
        if str(iv.get("state") or "") in DIRTY_IV_STATES:
            refuse.append(f"IV state {iv.get('state')} unsafe")
        if str(bf.get("verdict") or "") in DANGEROUS_BATTLEFIELD:
            refuse.append(f"battlefield {bf.get('verdict')} dangerous")
        if _num(iv.get("clean_mark_fraction"), 1.0) < cfg.min_clean_mark_fraction:
            refuse.append(
                f"clean marks {iv.get('clean_mark_fraction'):.2f} "
                f"< {cfg.min_clean_mark_fraction:.2f}")
        if _num(thesis.get("no_trade_score")) > cfg.max_no_trade_score:
            refuse.append(
                f"no_trade_score {thesis.get('no_trade_score'):.1f} too high"
            )

        # Gate A2 (HOT FIX 2026-06-22): dead-market guard. When the
        # scenario web reads chop/manipulation-dominant, taking a
        # directional trade is the textbook way to bleed fees. Multi-leg
        # neutral strategies (iron condor, jade lizard, butterfly) are
        # in the library but not yet routed; until they are, refusing
        # is the right move in this regime.
        if cfg.dead_market_refuse_directional:
            chop_mass = float(getattr(web_snap, "chop_mass", 0.0))
            manip_mass = float(getattr(web_snap, "manipulation_mass", 0.0))
            tail_mass = float(getattr(web_snap, "tail_mass", 0.0))
            hw_consensus = float(getattr(
                web_snap, "directional_consensus_horizon_weighted", 0.0))
            chop_dominant = chop_mass >= cfg.dead_market_chop_threshold
            manip_dominant = manip_mass >= cfg.dead_market_manipulation_threshold
            tail_with_no_signal = (
                tail_mass >= cfg.dead_market_tail_threshold
                and abs(hw_consensus) < cfg.dead_market_low_consensus_threshold)
            if chop_dominant:
                refuse.append(
                    f"dead market: chop_mass {chop_mass:.2f} ≥ "
                    f"{cfg.dead_market_chop_threshold:.2f} — refusing directional "
                    "(neutral structures not yet routed)")
            elif manip_dominant:
                refuse.append(
                    f"dead market: manipulation_mass {manip_mass:.2f} ≥ "
                    f"{cfg.dead_market_manipulation_threshold:.2f} — refusing "
                    "directional (MM is hunting; wait for resolution)")
            elif tail_with_no_signal:
                refuse.append(
                    f"dead market: tail_mass {tail_mass:.2f} elevated AND "
                    f"horizon-weighted consensus {hw_consensus:+.2f} near zero — "
                    "no edge to direct")

        # Gate B: confidence.
        profile = (PROFILE_SCALP if action in SCALP_ENTRY_ACTIONS
                   else PROFILE_INTRADAY)
        conf_floor = (cfg.min_scalp_confidence if profile == PROFILE_SCALP
                      else cfg.min_entry_confidence)
        if confidence < conf_floor:
            refuse.append(
                f"confidence {confidence:.2f} below floor {conf_floor:.2f}"
            )

        # Gate C: multi-timeframe alignment.
        mtf_alignment = self.mtf.alignment(direction)
        if not mtf_alignment["alignment_ok"]:
            refuse.append(
                f"multi-timeframe alignment failed (L1 alone is retail) — "
                f"score={mtf_alignment['alignment_score']:.2f}"
            )

        # Gate D: temporal contradictions from flow memory.
        contradictions = self._gather_contradictions(direction)

        # Gate D': Sprint-2 adversarial critic — replaces placeholder antithesis.
        critique = self.critic.critique(
            proposed_direction=direction,
            proposed_profile=(PROFILE_SCALP if action in SCALP_ENTRY_ACTIONS
                              else PROFILE_INTRADAY),
            snapshot=snapshot, rich_context=rich, flow_event=flow_event,
            flow_memory=self.flow, web_snapshot=web_snap,
            mtf_alignment=mtf_alignment,
        )
        antithesis_score = critique.antithesis_score
        antithesis_notes = list(critique.antithesis_reasons)

        if critique.recommended_action == "REFUSE":
            refuse.append(
                f"critic REFUSE (antithesis={antithesis_score:.2f}): "
                + "; ".join(critique.antithesis_reasons[:2])
            )

        if refuse:
            return None, refuse

        # Gate E: select strike + premium.
        strike = _map(decision.get("strike"))
        contract_side = str(strike.get("side")
                              or ("CE" if direction > 0 else "PE"))
        contract_level = int(_num(strike.get("level"), 0))
        contract_label = str(strike.get("label") or f"{contract_side}_ATM")
        strike_price, entry_premium, entry_dod_z, entry_friend, entry_sstate = \
            self._lookup_target_slot(snapshot, contract_side, contract_level)
        if entry_premium is None:
            refuse.append(
                f"no clean mark for chosen contract {contract_label}"
            )
            return None, refuse
        if entry_sstate == SPREAD_DANGEROUS:
            refuse.append(
                f"chosen contract {contract_label} spread state dangerous"
            )
            return None, refuse

        # Gate F: economics — fees + slippage + EV.
        profile_target = (cfg.scalp_target_r if profile == PROFILE_SCALP
                          else cfg.intraday_target_r)
        premium_stop_pct = (cfg.economics.exit_accel_loss_pct
                            * 2.0 if profile == PROFILE_SCALP else 0.50)
        # stop ≡ entry × (1 - stop_pct) for long, (1 + stop_pct) for short
        if direction > 0:
            stop_premium = entry_premium * (1.0 - premium_stop_pct)
            target_premium = entry_premium * (1.0 + premium_stop_pct * profile_target)
        else:
            stop_premium = entry_premium * (1.0 + premium_stop_pct)
            target_premium = entry_premium * (1.0 - premium_stop_pct * profile_target)

        # Continuous conviction (sign + magnitude) — strong signals size
        # up, weak ones size down. Additive: does NOT touch `direction`.
        # HOT FIX 2026-06-22 LIVE: prefer the horizon-weighted consensus
        # (drops sub-0.15 short-term noise, weights survivors by horizon).
        # Falls back to the raw consensus only when nothing passes the
        # noise floor (so a totally quiet web doesn't read as bearish).
        _raw_consensus = float(getattr(web_snap, "directional_consensus", 0.0))
        _hw_consensus = float(getattr(
            web_snap, "directional_consensus_horizon_weighted", 0.0))
        _consensus_for_conviction = (
            _hw_consensus if abs(_hw_consensus) > 1e-6 else _raw_consensus)
        conviction = compute_conviction(
            direction=direction,
            confidence=confidence,
            web_directional_consensus=_consensus_for_conviction,
            mtf_alignment_score=float(mtf_alignment["alignment_score"]),
            antithesis_score=antithesis_score,
        )

        # Provisional sizing — base buckets, then scaled by conviction.
        base_lots = self._size_lots(confidence, antithesis_score,
                                       mtf_alignment["alignment_score"])
        # Confidence-gated sizer (workaround B): scale further by a
        # strategy-specific multiplier derived from the rolling
        # attribution. New strategies start at 1.0 (no haircut, but no
        # bonus either). Strategies with a positive realized track record
        # earn a bonus up to 1.25×; losing strategies are throttled to
        # 0.4× until they either earn back the right or get pruned.
        _action = str(decision.get("action") or "")
        strat_multiplier = self._strategy_confidence_multiplier(_action)
        size_lots = max(1, int(round(base_lots
                                       * conviction.size_multiplier
                                       * strat_multiplier)))

        # Probabilities — Sprint-1 placeholder uses confidence + alignment.
        # Sprint 2's scenario_web replaces this with proper Bayesian priors.
        prob_target = max(0.25, min(0.75,
                                       0.55 * confidence + 0.25
                                       * mtf_alignment["alignment_score"]))
        prob_stop = 1.0 - prob_target

        ev = expected_value_after_costs(
            entry_premium=entry_premium, expected_exit_premium=target_premium,
            prob_target=prob_target, prob_stop=prob_stop,
            stop_premium=stop_premium, lots=size_lots,
            is_long=direction > 0, friendliness=entry_friend,
            spread_state=entry_sstate, cfg=cfg.economics,
        )
        if not ev.approve:
            refuse.extend(ev.reasons)
            return None, refuse

        # Gate G: Sprint-2 forward projection — empirical distribution given context.
        bf_verdict = str(bf.get("verdict") or "")
        iv_state_str = str(iv.get("state") or "")
        winding_zone = str(_map(snapshot.get("winding")).get("zone") or "NO_WINDING")
        thesis_state = str(thesis.get("composite_state") or "")
        projection = self.projection.project(
            thesis_state=thesis_state, iv_state=iv_state_str,
            battlefield_verdict=bf_verdict, winding_zone=winding_zone,
            direction=direction,
            regime_stability=rich.regime_stability_index,
            target_r=profile_target, stop_r=1.0,
        )

        proposed_strategy_class = (STRAT_LONG_CE if direction > 0
                                    else STRAT_LONG_PE)

        # Gate H: Sprint-2 decision aggregator — combines everything.
        agg = self.aggregator.decide(
            base_confidence=confidence,
            mtf_alignment=mtf_alignment,
            projection=projection,
            critique=critique,
            ev_decision=ev,
            web_snapshot=web_snap,
            proposed_strategy_class=proposed_strategy_class,
            proposed_direction=direction,
            open_positions_count=len(self._open_states),
            max_open_positions=cfg.economics.max_open_positions,
            daily_pnl_rupees=self.daily_pnl_rupees,
            max_daily_bleed=cfg.economics.max_daily_bleed,
        )
        if agg.decision == AGGR_REFUSE:
            refuse.extend(agg.refuse_reasons[:3])
            return None, refuse
        size_multiplier = agg.recommended_size_multiplier
        if size_multiplier < 1.0:
            new_lots = max(1, int(round(size_lots * size_multiplier)))
            if new_lots != size_lots:
                size_lots = new_lots

        # Gate I: counterfactual — generate position-specific kill plan.
        cf_plan = self.counterfactual.generate(
            proposed_direction=direction,
            proposed_profile=profile,
            proposed_strategy_class=proposed_strategy_class,
            snapshot=snapshot, rich_context=rich,
            flow_event=flow_event, web_snapshot=web_snap,
        )

        # Hard cap: rupees-at-risk per trade.
        rupees_at_risk = abs(stop_premium - entry_premium) * cfg.economics.lot_size * size_lots
        if rupees_at_risk > cfg.economics.max_loss_per_trade * 1.5:
            # Try reducing lots.
            shrunk_lots = max(1, int(cfg.economics.max_loss_per_trade
                                       / max(0.01, abs(stop_premium - entry_premium)
                                              * cfg.economics.lot_size)))
            if shrunk_lots < 1:
                refuse.append(
                    f"rupees at risk ₹{rupees_at_risk:.0f} > budget "
                    f"₹{cfg.economics.max_loss_per_trade:.0f}; cannot size down"
                )
                return None, refuse
            size_lots = shrunk_lots
            rupees_at_risk = abs(stop_premium - entry_premium) * cfg.economics.lot_size * size_lots

        rupees_at_target = abs(target_premium - entry_premium) * cfg.economics.lot_size * size_lots
        # Workaround A: feed observed slippage (bps) into the gate when
        # we have enough live samples. Falls back to the theoretical
        # number transparently when the tracker is empty.
        slip_summary = self.slippage_tracker.rolling_summary()
        slip_n = int(slip_summary.get("n_records") or 0)
        slip_mean = (float(slip_summary.get("mean_bps") or 0.0)
                      if slip_n > 0 else None)
        slippage_est = estimate_slippage(
            lots=size_lots, friendliness=entry_friend,
            spread_state=entry_sstate, cfg=cfg.economics,
            realized_mean_bps=slip_mean,
            realized_n_samples=slip_n,
            reference_premium=entry_premium,
        )
        fees_est = fees_for_round_trip(
            entry_premium=entry_premium, exit_premium=target_premium,
            lots=size_lots, is_long=direction > 0, cfg=cfg.economics,
        ).total
        min_delta = minimum_profitable_premium_delta(
            entry_premium=entry_premium, lots=size_lots,
            is_long=direction > 0, friendliness=entry_friend,
            spread_state=entry_sstate, cfg=cfg.economics,
        )

        # All gates passed — build hypothesis, open ledger, register state.
        hypothesis = build_hypothesis_from_snapshot(
            snapshot=snapshot,
            rich_context_dict=rich.to_dict(),
            mtf_alignment=mtf_alignment,
            contradicting_recent=contradictions,
            direction=direction,
            profile=profile,
            contract_side=contract_side,
            contract_label=contract_label,
            contract_level=contract_level,
            strike_price=strike_price,
            entry_premium=entry_premium,
            entry_spot=_num(snapshot.get("spot")),
            entry_dod_z=entry_dod_z,
            entry_friendliness=entry_friend,
            entry_spread_state=entry_sstate,
            size_fraction=size_lots / cfg.max_lots,
            size_lots=size_lots,
            rupees_at_risk=rupees_at_risk,
            rupees_at_target=rupees_at_target,
            stop_premium=stop_premium,
            target_premium=target_premium,
            trail_after_r=cfg.trail_after_r,
            max_hold_bars=(cfg.scalp_max_bars if profile == PROFILE_SCALP
                            else cfg.intraday_max_bars),
            premium_stop_pct=premium_stop_pct,
            fee_estimate_rupees=fees_est,
            slippage_estimate_rupees=slippage_est,
            expected_value_rupees=ev.expected_profit_rupees,
            expected_edge_multiple=ev.edge_multiple,
            minimum_profitable_premium_delta=min_delta,
            antithesis_score=antithesis_score,
            antithesis_notes=antithesis_notes,
            bar_index=bar_index,
        )

        ledger = PositionLedger(hypothesis=hypothesis)
        self.ledger_store.open(ledger)
        self._open_states[hypothesis.position_id] = _OpenPositionState(
            hypothesis=hypothesis,
            entry_bar=bar_index,
            last_bar=bar_index,
            best_r=0.0,
            worst_r=0.0,
            high_water_confidence=confidence,
            low_water_confidence=confidence,
            last_premium=entry_premium,
        )
        # Attach the counterfactual plan for per-bar kill checks.
        self._counterfactual_plans[hypothesis.position_id] = cf_plan
        # Register the per-position Adaptive Exit Engine — manages ghost
        # exit price and the Zerodha 25-mod budget for this position.
        try:
            exit_thesis = build_exit_thesis(
                position_id=hypothesis.position_id,
                direction=direction,
                entry_premium=entry_premium,
                target_premium=target_premium,
                stop_premium=stop_premium,
                thesis_confidence=confidence,
                profile=profile,
            )
            exit_engine_instance = AdaptiveExitEngine(
                exit_thesis=exit_thesis,
                cfg=cfg.exit_engine,
            )
            self.portfolio_exit.register(hypothesis.position_id,
                                            exit_engine_instance)
        except Exception:
            # Adaptive exit is opt-in best-effort; never block entry.
            pass
        self.cooldown_until_bar = bar_index + cfg.cooldown_bars_after_entry

        # Record on the recent-trades tape (for the UI's Live Trades panel).
        import time as _time
        self._recent_trades.appendleft({
            "kind": "OPEN", "ts": _time.time(),
            "bar_index": bar_index,
            "position_id": hypothesis.position_id,
            "contract_label": hypothesis.contract_label,
            "direction": hypothesis.direction,
            "size_lots": hypothesis.size_lots,
            "entry_premium": hypothesis.entry_premium,
            "strategy": hypothesis.trigger_action,
        })

        return {
            "intent": (INTENT_OPEN_LONG if direction > 0 and profile == PROFILE_INTRADAY
                        else INTENT_OPEN_SHORT if direction < 0 and profile == PROFILE_INTRADAY
                        else INTENT_OPEN_SCALP_CALL if direction > 0
                        else INTENT_OPEN_SCALP_PUT),
            "hypothesis": hypothesis.to_dict(),
            "ev_decision": ev.to_dict(),
            "critique": critique.to_dict(),
            "aggregator_decision": agg.to_dict(),
            "projection": projection.to_dict(),
            "counterfactual_plan": cf_plan.to_dict(),
            "web_snapshot": web_snap.to_dict(),
            "active_patterns": [p.to_dict() for p in self._last_patterns],
            "mm_posterior": (self._last_mm_posterior.to_dict()
                              if self._last_mm_posterior else None),
            "fat_tail_score": (self._last_tail_score.to_dict()
                                if self._last_tail_score else None),
            "crowd_mirror": (self._last_crowd_report.to_dict()
                              if self._last_crowd_report else None),
            "portfolio_risk": (self._last_risk_report.to_dict()
                                if self._last_risk_report else None),
            "hedge_proposal": (self._last_hedge_proposal.to_dict()
                                if self._last_hedge_proposal else None),
            "adaptive_exit": self._last_exit_decision,
            "conviction": conviction.to_dict(),
        }, []

    def _gather_contradictions(self, direction: int) -> List[str]:
        """Look at recent flow_memory events for signals contradicting the
        proposed direction."""
        recent = self.flow.within_seconds(1200.0)  # 20 minutes
        if not recent:
            return []
        contradictions: List[str] = []
        recent_count = len(recent)
        # Count opposite trap/regime-break verdicts.
        opposite_trap = 0
        opposite_thesis = 0
        opposite_winding = 0
        for ev in recent:
            if direction > 0:
                if "BULL_TRAP_WINDING" in ev.winding_zone:
                    opposite_winding += 1
                if "BEAR" in ev.thesis_state:
                    opposite_thesis += 1
            else:
                if "BEAR_TRAP_WINDING" in ev.winding_zone:
                    opposite_winding += 1
                if "BULL" in ev.thesis_state:
                    opposite_thesis += 1
        if opposite_winding >= 3:
            contradictions.append(
                f"opposite-direction winding ({opposite_winding} bars in last 20 min)"
            )
        if opposite_thesis / max(1, recent_count) > 0.25:
            contradictions.append(
                f"opposite-direction thesis dominated "
                f"{opposite_thesis}/{recent_count} of last 20 minutes"
            )
        # Surprise-then-revert patterns.
        n_surp_rev = (len(self.flow.find_pattern("surprise_up_then_revert"))
                       + len(self.flow.find_pattern("surprise_down_then_revert")))
        if n_surp_rev >= 2:
            contradictions.append(
                f"{n_surp_rev} surprise-then-revert events recently — "
                f"chop / manipulation risk elevated"
            )
        # Thesis oscillation.
        n_osc = len(self.flow.find_pattern("thesis_oscillation"))
        if n_osc >= 1:
            contradictions.append("thesis oscillation detected — regime indecision")
        return contradictions

    def _estimate_antithesis(self, direction: int, flow_event: FlowEvent,
                              rich: RichContext,
                              contradictions: List[str]) -> tuple[float, List[str]]:
        """Sprint-1 antithesis score: combines contradictions count, regime
        stability, dispersion, and held-side acceptance run.

        Sprint 2's critic.py will replace this with a richer analysis.
        """
        score = 0.0
        notes: List[str] = []

        if contradictions:
            score += 0.10 * len(contradictions)
            notes.append(f"{len(contradictions)} contradiction(s) in recent memory")

        # Regime stability — if regime is unstable, antithesis up.
        if rich.regime_stability_index < 0.40:
            score += 0.20
            notes.append(
                f"regime unstable (stability={rich.regime_stability_index:.2f})"
            )

        # Dispersion spike — single-strike pressure → fragile read.
        if rich.dispersion_velocity > 0.10:
            score += 0.15
            notes.append(
                f"CE dispersion rising (velocity={rich.dispersion_velocity:.2f}) "
                f"— single-strike distortion risk"
            )

        # Epicenter migrating — instability in where pressure sits.
        if rich.epicenter_migration_distance >= 3:
            score += 0.10
            notes.append(
                f"epicenter migrating ({rich.epicenter_migration_distance:.0f} strikes)"
            )

        # If proposing long but PE rail is being defended (acceptance defended on PEs) — suspicious.
        # (Use flow event's snapshot.)
        if direction > 0 and flow_event.pe_defended_fraction >= 0.30:
            score += 0.20
            notes.append(
                f"PE rail defended at {flow_event.pe_defended_fraction:.0%} — "
                f"hidden downside positioning"
            )
        if direction < 0 and flow_event.ce_defended_fraction >= 0.30:
            score += 0.20
            notes.append(
                f"CE rail defended at {flow_event.ce_defended_fraction:.0%} — "
                f"hidden upside positioning"
            )

        # Thesis oscillation.
        n_osc = len(self.flow.find_pattern("thesis_oscillation"))
        if n_osc >= 1:
            score += 0.10
            notes.append("thesis oscillation in recent history")

        return min(1.0, score), notes

    def _build_exit_markets(self, *, snapshot: Dict[str, Any],
                                held_quotes: Optional[Dict[str, Any]],
                                current_r_by_position: Dict[str, float],
                                bar_index: int,
                                ) -> Dict[str, MarketState]:
        """For each open position, construct a MarketState the exit
        engine can consume. Held premium comes from held_quotes if
        present, otherwise from the snapshot's slot_readings for the
        position's contract."""
        out: Dict[str, MarketState] = {}
        tail_action = (self._last_tail_score.recommended_action
                        if self._last_tail_score is not None else "NORMAL")
        portfolio_pressure = bool(
            self._last_risk_report is not None
            and self._last_risk_report.kill_switches
        )
        for pos_id, state in self._open_states.items():
            h = state.hypothesis
            held = (held_quotes or {}).get(pos_id)
            held_premium = self._resolve_held_premium(state, snapshot, held)
            if held_premium is None or not math.isfinite(held_premium) or held_premium <= 0:
                continue
            bid = float(getattr(held, "bid_price", 0.0) or 0.0)
            ask = float(getattr(held, "ask_price", 0.0) or 0.0)
            friend = float(getattr(held, "friendliness",
                                       h.entry_friendliness) or 1.0)
            confidence_now = _num(_map(snapshot.get("decision"))
                                     .get("confidence"))
            decay = max(0.0, state.high_water_confidence - confidence_now)
            current_r = current_r_by_position.get(pos_id, 0.0)
            # Heuristic micro-volatility: 0.5% of held premium.
            micro_vol = max(0.20, abs(held_premium) * 0.005)
            out[pos_id] = MarketState(
                bar_index=bar_index,
                ts=snapshot.get("ts"),
                held_premium=float(held_premium),
                held_bid=bid, held_ask=ask,
                spread=max(0.0, ask - bid) if ask > 0 and bid > 0 else 0.0,
                friendliness=friend,
                thesis_intact=bool(snapshot.get("is_warm")),
                thesis_confidence_decay=decay,
                current_r=current_r,
                bars_held=max(0, bar_index - state.entry_bar),
                fat_tail_action=tail_action,
                portfolio_under_pressure=portfolio_pressure,
                micro_volatility=micro_vol,
            )
        return out

    def _build_strike_lookup(self, snapshot: Dict[str, Any]) -> Dict:
        """Map (side, level) → (strike, premium, friendliness, spread_state)
        from this tick's slot_readings."""
        out: Dict = {}
        for raw in snapshot.get("slot_readings") or []:
            slot = _map(raw)
            side = str(slot.get("option_type") or "")
            level = int(_num(slot.get("level"), 999))
            if side not in ("CE", "PE") or level == 999:
                continue
            mark = _num(slot.get("mark_price"), math.nan)
            if not math.isfinite(mark) or mark <= 0:
                continue
            strike = _num(slot.get("strike"), 0.0)
            friend = _num(slot.get("friendliness"), 1.0)
            sstate = str(slot.get("spread_state") or "clean")
            out[(side, level)] = (strike, mark, friend, sstate)
        return out

    def _lookup_target_slot(self, snapshot: Dict[str, Any],
                              contract_side: str, contract_level: int,
                              ) -> tuple[float, Optional[float], float, float, str]:
        """Pull strike_price, mark, dod_z, friendliness, spread_state for
        the target contract from the snapshot's slot_readings."""
        for raw in snapshot.get("slot_readings") or []:
            slot = _map(raw)
            if (str(slot.get("option_type") or "") == contract_side
                    and int(_num(slot.get("level"), 999)) == contract_level):
                strike = _num(slot.get("strike"), 0.0)
                mark = _num(slot.get("mark_price"), math.nan)
                if not math.isfinite(mark) or mark <= 0:
                    return strike, None, 0.0, 1.0, "clean"
                dod_z = _num(slot.get("dod_z"), 0.0)
                friend = _num(slot.get("friendliness"), 1.0)
                sstate = str(slot.get("spread_state") or "clean")
                return strike, mark, dod_z, friend, sstate
        return 0.0, None, 0.0, 1.0, "clean"

    def _size_lots(self, confidence: float, antithesis_score: float,
                    alignment_score: float) -> int:
        """Sprint-1 sizing — confidence × (1-antithesis) × alignment_score,
        bucketed to a lot count up to max_lots."""
        conviction = (
            0.50 * confidence
            + 0.30 * alignment_score
            + 0.20 * (1.0 - antithesis_score)
        )
        if conviction >= 0.80:
            return min(self.cfg.max_lots, 4)
        if conviction >= 0.70:
            return min(self.cfg.max_lots, 3)
        if conviction >= 0.55:
            return min(self.cfg.max_lots, 2)
        return 1

    def _position_r(self, state: _OpenPositionState,
                     held_premium: Optional[float]) -> float:
        h = state.hypothesis
        if (held_premium is None or not math.isfinite(held_premium)
                or h.entry_premium <= 0):
            return 0.0
        side = h.direction
        # 1R = the stop distance from entry.
        stop_distance = abs(h.stop_premium - h.entry_premium)
        if stop_distance < 1e-9:
            return 0.0
        return side * (held_premium - h.entry_premium) / stop_distance

    def _resolve_held_premium(self, state: _OpenPositionState,
                                snapshot: Dict[str, Any],
                                held_quote: Any) -> Optional[float]:
        """Prefer the explicit held_quote, fall back to slot_readings.

        HOT FIX 2026-06-22 LIVE (founder caught this with real money):
        The slot fallback used to look up by ``contract_level``. That is
        WRONG when the underlying drifts — ATM shifts, so the level-0
        slot now belongs to a DIFFERENT strike. The held position's
        strike is FIXED at h.strike_price; we must look up by STRIKE.
        Using level produced phantom thousand-rupee profits when the
        ATM moved away from the held contract.
        """
        if held_quote is not None and hasattr(held_quote, "best_price"):
            best = held_quote.best_price
            if best is not None and math.isfinite(best) and best > 0:
                return float(best)
        h = state.hypothesis
        target_side = h.contract_side
        target_strike = float(h.strike_price)
        if target_strike <= 0:
            # No strike recorded — fall back to the legacy level-based
            # lookup, but this path should never fire for a real position.
            _, mark, _, _, _ = self._lookup_target_slot(
                snapshot, target_side, h.contract_level)
            return mark
        for raw in snapshot.get("slot_readings") or []:
            slot = _map(raw)
            if str(slot.get("option_type") or "") != target_side:
                continue
            if abs(_num(slot.get("strike"), 0.0) - target_strike) > 0.01:
                continue
            mark = _num(slot.get("mark_price"), math.nan)
            if math.isfinite(mark) and mark > 0:
                return float(mark)
        return None

    def _exit_check(self, *, state: _OpenPositionState,
                     snapshot: Dict[str, Any], held_read: Any,
                     held_premium: Optional[float], current_r: float,
                     confidence: float, bar_index: int,
                     ) -> tuple[List[str], str]:
        """Returns (exit_reasons, severity). severity ∈ {"hard","soft",""}.
        Empty list means hold."""
        cfg = self.cfg
        h = state.hypothesis
        decision = _map(snapshot.get("decision"))
        iv = _map(snapshot.get("iv_state"))
        bf = _map(snapshot.get("battlefield"))
        thesis = _map(snapshot.get("thesis"))

        hard: List[str] = []
        soft: List[str] = []
        age = max(0, bar_index - state.entry_bar)

        # ── Hard exits ───────────────────────────────────────────────
        action = str(decision.get("action") or "")
        if action == "EXIT" or action.startswith("EXIT_"):
            hard.append("belief engine emitted EXIT")
        if str(iv.get("state") or "") in DIRTY_IV_STATES:
            hard.append(f"IV state {iv.get('state')} unsafe")
        if str(bf.get("verdict") or "") in DANGEROUS_BATTLEFIELD:
            hard.append(f"battlefield {bf.get('verdict')} dangerous")
        if _num(thesis.get("no_trade_score")) > cfg.max_no_trade_score:
            hard.append("no-trade danger rose while holding")
        if not bool(snapshot.get("is_warm")):
            hard.append("engine went cold mid-trade")
        if current_r <= -1.0:
            hard.append(
                f"hard stop on premium R={current_r:.2f}"
            )

        # Held leg checks
        if held_read is not None:
            sstate = getattr(held_read, "spread_state", "")
            if sstate == SPREAD_DANGEROUS:
                hard.append("held contract spread turned dangerous")
            ms = getattr(held_read, "mark_source", "")
            if ms in DIRTY_MARK_SOURCES:
                hard.append(f"held mark source dirty ({ms})")

        # ── Counterfactual position-specific kill criteria (Sprint 2) ──
        cf_plan = self._counterfactual_plans.get(h.position_id)
        if cf_plan is not None and age <= cf_plan.kill_window_bars:
            cf_kills = self._evaluate_counterfactual_kills(
                cf_plan=cf_plan, hypothesis=h,
                snapshot=snapshot, held_read=held_read, age=age,
            )
            for kk in cf_kills:
                hard.append(kk)

        if hard:
            return hard, "hard"

        # ── Exit accelerator (cut losers fast even if fees bad) ──────
        if held_premium is not None:
            decay = max(0.0, state.high_water_confidence - confidence)
            accel = should_accelerate_exit(
                entry_premium=h.entry_premium,
                current_premium=held_premium,
                stop_premium=h.stop_premium,
                bars_held=age,
                lots=h.size_lots,
                is_long=h.direction > 0,
                thesis_confidence_decay=decay,
                friendliness=getattr(held_read, "friendliness", 1.0) or 1.0,
                spread_state=getattr(held_read, "spread_state", "clean") or "clean",
                cfg=cfg.economics,
            )
            if accel.accelerate:
                return [accel.reason], "hard"

        # ── Soft exits (only after min hold) ─────────────────────────
        if age < cfg.min_hold_bars:
            return [], ""

        # Acceptance flip against position
        if held_read is not None:
            acc = getattr(held_read, "acceptance", "")
            if acc == "rejected":
                soft.append(f"held leg acceptance 'rejected'")

        # Thesis flip
        thesis_state = str(thesis.get("composite_state") or "")
        if h.direction > 0 and "BEAR" in thesis_state:
            soft.append(f"thesis flipped against long: {thesis_state}")
        if h.direction < 0 and "BULL" in thesis_state:
            soft.append(f"thesis flipped against short: {thesis_state}")

        # Confidence giveback — HOT FIX 2026-06-22 LIVE: founder observed
        # this exit alone (no thesis flip) was killing trades within 1-3
        # bars of entry because confidence is noisy. Now requires that
        # the thesis composite ALSO no longer supports the held side.
        thesis_supports_held = (
            (h.direction > 0 and "BULL" in thesis_state)
            or (h.direction < 0 and "BEAR" in thesis_state))
        if (not thesis_supports_held
                and state.high_water_confidence - confidence >= cfg.confidence_giveback
                and confidence < cfg.min_entry_confidence):
            soft.append(
                f"confidence giveback {state.high_water_confidence - confidence:.2f}"
            )

        # Profit-lock giveback
        if (state.best_r >= cfg.profit_lock_start_r
                and state.best_r - current_r >= cfg.profit_lock_giveback_r):
            soft.append(
                f"profit lock giveback (best={state.best_r:.2f}R, current={current_r:.2f}R)"
            )

        # Target reached
        target = (cfg.scalp_target_r if h.profile == PROFILE_SCALP
                   else cfg.intraday_target_r)
        if current_r >= target:
            soft.append(
                f"target reached current_r={current_r:.2f} ≥ target={target:.2f}"
            )

        # Max hold
        max_hold = (cfg.scalp_max_bars if h.profile == PROFILE_SCALP
                     else cfg.intraday_max_bars)
        if age >= max_hold:
            soft.append(f"max hold bars reached age={age}")

        if soft:
            return soft, "soft"
        return [], ""

    def _evaluate_counterfactual_kills(self, *, cf_plan: CounterfactualPlan,
                                         hypothesis: PositionHypothesis,
                                         snapshot: Dict[str, Any],
                                         held_read: Any, age: int) -> List[str]:
        """Check each HARD-severity kill criterion against current evidence.

        Returns the names of criteria that fired this tick. Each fired
        criterion is treated as a hard exit reason by the caller.
        """
        fired: List[str] = []
        thesis_state = str(_map(snapshot.get("thesis")).get(
            "composite_state") or "")
        iv_state = str(_map(snapshot.get("iv_state")).get("state") or "")
        held_acc = (getattr(held_read, "acceptance", "")
                    if held_read is not None else "")
        # Compute current per-side defended fractions from slot_readings.
        ce_def = pe_def = ce_total = pe_total = 0
        for raw in snapshot.get("slot_readings") or []:
            slot = _map(raw)
            otype = str(slot.get("option_type") or "")
            acc = str(slot.get("acceptance") or "normal")
            if otype == "CE":
                ce_total += 1
                if acc == "defended":
                    ce_def += 1
            elif otype == "PE":
                pe_total += 1
                if acc == "defended":
                    pe_def += 1
        ce_def_frac = ce_def / ce_total if ce_total else 0.0
        pe_def_frac = pe_def / pe_total if pe_total else 0.0
        direction = hypothesis.direction

        for k in cf_plan.kill_criteria:
            if k.severity != SEVERITY_HARD:
                continue
            if age > k.monitor_window_bars:
                continue
            name = k.name
            if name == "thesis_flips_bear" and direction > 0 \
                    and "BEAR" in thesis_state:
                fired.append(f"counterfactual: {name} fired (thesis={thesis_state})")
            elif name == "thesis_flips_bull" and direction < 0 \
                    and "BULL" in thesis_state:
                fired.append(f"counterfactual: {name} fired (thesis={thesis_state})")
            elif name == "iv_state_turns_dirty" \
                    and iv_state in DIRTY_IV_STATES:
                fired.append(
                    f"counterfactual: {name} fired (iv_state={iv_state})"
                )
            elif name == "pe_defended_rises" and direction > 0 \
                    and pe_def_frac >= 0.50:
                fired.append(
                    f"counterfactual: {name} fired "
                    f"(pe_defended={pe_def_frac:.0%})"
                )
            elif name == "ce_defended_rises" and direction < 0 \
                    and ce_def_frac >= 0.50:
                fired.append(
                    f"counterfactual: {name} fired "
                    f"(ce_defended={ce_def_frac:.0%})"
                )
            elif name == "held_leg_rejected" and held_acc == "rejected":
                fired.append(f"counterfactual: {name} fired")
        return fired

    def _close_position(self, *, pos_id: str, state: _OpenPositionState,
                          snapshot: Dict[str, Any],
                          held_premium: Optional[float], held_read: Any,
                          current_r: float, exit_reasons: List[str],
                          severity: str, bar_index: int) -> Optional[LedgerOutcome]:
        h = state.hypothesis
        exit_premium = held_premium if held_premium is not None else h.entry_premium

        # Compute fees + realized P&L
        is_long = h.direction > 0
        fees = fees_for_round_trip(
            entry_premium=h.entry_premium, exit_premium=exit_premium,
            lots=h.size_lots, is_long=is_long, cfg=self.cfg.economics,
        ).total
        # Slippage realized at close: also blend in the observed
        # tracker stats so post-close attribution matches the EV gate's
        # ex-ante number.
        _slip_summary = self.slippage_tracker.rolling_summary()
        _slip_n = int(_slip_summary.get("n_records") or 0)
        _slip_mean = (float(_slip_summary.get("mean_bps") or 0.0)
                       if _slip_n > 0 else None)
        slippage = estimate_slippage(
            lots=h.size_lots,
            friendliness=getattr(held_read, "friendliness", 1.0) or 1.0,
            spread_state=getattr(held_read, "spread_state", "clean") or "clean",
            cfg=self.cfg.economics,
            realized_mean_bps=_slip_mean,
            realized_n_samples=_slip_n,
            reference_premium=exit_premium,
        )
        side = 1 if is_long else -1
        shares = h.size_lots * self.cfg.economics.lot_size
        gross = side * (exit_premium - h.entry_premium) * shares
        net = gross - fees - slippage

        self.daily_pnl_rupees += net
        self.cumulative_fees_rupees += fees

        # Validation/invalidation post-mortem
        validations_ever = set()
        invalid_ever = set()
        ledger = self.ledger_store.find_open(pos_id)
        if ledger is not None:
            for b in ledger.bar_records:
                validations_ever.update(b.validations_met)
                invalid_ever.update(b.invalidations_active)
        validation_score = (
            len(validations_ever) / max(1, len(h.validation_criteria))
        )
        hypothesis_vs_reality = max(0.0, min(1.0,
            0.5 * validation_score
            + 0.5 * (1.0 if net > 0 else 0.0)
        ))

        outcome = LedgerOutcome(
            exit_reason="; ".join(exit_reasons[:3]),
            exit_severity=severity,
            exit_premium=round(exit_premium, 4),
            exit_spot=_num(snapshot.get("spot")),
            exit_bar=bar_index,
            realized_r=round(current_r, 3),
            realized_rupees=round(net, 2),
            gross_rupees=round(gross, 2),
            fees_rupees=round(fees, 2),
            slippage_rupees=round(slippage, 2),
            bars_held=max(0, bar_index - state.entry_bar),
            validation_ever_fully_met=bool(validation_score >= 1.0),
            invalidation_count=len(invalid_ever),
            hypothesis_vs_reality_score=round(hypothesis_vs_reality, 3),
        )
        self.ledger_store.close(pos_id, outcome)
        # Record on the recent-trades tape (newest first; deque enforces cap).
        import time as _time
        self._recent_trades.appendleft({
            "kind": "CLOSE", "ts": _time.time(),
            "bar_index": bar_index,
            "position_id": pos_id,
            "contract_label": h.contract_label,
            "direction": h.direction,
            "size_lots": h.size_lots,
            "exit_premium": exit_premium,
            "realized_r": current_r,
            "realized_rupees": net,
            "exit_reason": (exit_reasons[0] if exit_reasons else ""),
        })

        # Per-strategy attribution (workaround C).
        strat = h.trigger_action or "unknown"
        stats = self._strategy_stats[strat]
        stats["n_trades"] += 1
        stats["total_realized_rupees"] += float(net)
        stats["total_fees_rupees"] += float(fees)
        stats["running_r_sum"] += float(current_r)
        if net > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["best_trade"] = max(stats["best_trade"], float(net))
        stats["worst_trade"] = min(stats["worst_trade"], float(net))
        stats["avg_realized_r"] = (
            stats["running_r_sum"] / max(1, stats["n_trades"]))

        # Feed the projection tape with the realized outcome.
        target = (self.cfg.scalp_target_r if h.profile == PROFILE_SCALP
                  else self.cfg.intraday_target_r)
        closed_via_target = current_r >= target * 0.95
        closed_via_stop = current_r <= -0.95
        self.projection.record(ProjectionRecord(
            ts=h.opened_at_ts,
            bar_index=h.opened_at_bar,
            thesis_state=h.thesis_state,
            iv_state=str(h.trigger_snapshot_summary.get("iv_state") or ""),
            battlefield_verdict=str(h.trigger_snapshot_summary.get(
                "battlefield_verdict") or ""),
            winding_zone=str(h.trigger_snapshot_summary.get(
                "winding_zone") or "NO_WINDING"),
            direction=h.direction,
            regime_stability=float(
                h.rich_context.get("regime_stability_index", 1.0)),
            bull_score=float(h.trigger_snapshot_summary.get("bull_score") or 0.0),
            bear_score=float(h.trigger_snapshot_summary.get("bear_score") or 0.0),
            net_intent_z=float(h.rich_context.get("net_intent_velocity", 0.0)),
            realized_premium_change_pct=round(
                (exit_premium - h.entry_premium) / max(0.01, h.entry_premium),
                4),
            realized_r=round(current_r, 3),
            bars_to_resolution=max(0, bar_index - state.entry_bar),
            closed_via_target=closed_via_target,
            closed_via_stop=closed_via_stop,
            closed_via_neither=not (closed_via_target or closed_via_stop),
        ))
        # Feed the live calibrator (walk-forward + safety-gated). We use
        # the hypothesis's stored aggregator components when available.
        try:
            component_scores = {
                "base_score": float(h.expected_edge_multiple
                                       if h.expected_edge_multiple else
                                       state.high_water_confidence) / 3.0,
                "mtf_alignment_score": float(
                    h.mtf_alignment.get("alignment_score", 0.0)),
                "projection_factor": 0.5,
                "fees_clearance_score": min(1.0, max(0.0,
                    (h.expected_edge_multiple - 1.0) / 1.5 * 0.5 + 0.5)),
                "portfolio_capacity_score": 0.7,
            }
            calib = self.live_calibrator.on_position_closed(
                component_scores=component_scores,
                realized_r=float(current_r),
                bar_index=bar_index,
            )
            self._last_calibration = calib.to_dict()
        except Exception:
            self._last_calibration = None
        # Forget the per-position counterfactual plan.
        self._counterfactual_plans.pop(pos_id, None)
        # Unregister the adaptive exit engine.
        try:
            self.portfolio_exit.unregister(pos_id)
        except Exception:
            pass
        del self._open_states[pos_id]
        self.cooldown_until_bar = max(
            self.cooldown_until_bar,
            bar_index + self.cfg.cooldown_bars_after_exit,
        )
        return outcome

    def _append_bar_record(self, *, ledger: PositionLedger,
                            state: _OpenPositionState, snapshot: Dict[str, Any],
                            held_premium: Optional[float], current_r: float,
                            confidence: float, held_read: Any,
                            bar_index: int) -> None:
        h = state.hypothesis
        thesis = _map(snapshot.get("thesis"))
        iv = _map(snapshot.get("iv_state"))
        bf = _map(snapshot.get("battlefield"))
        decision = _map(snapshot.get("decision"))
        winding = _map(snapshot.get("winding"))

        # Bookkeeping: compute net unrealized for cumulative tracker
        net_now = 0.0
        if held_premium is not None:
            shares = h.size_lots * self.cfg.economics.lot_size
            side = 1 if h.direction > 0 else -1
            net_now = side * (held_premium - h.entry_premium) * shares

        # Semantic checks: which validations / invalidations currently fire.
        validations_met: List[str] = []
        invalidations_active: List[str] = []
        # 1. premium ≥ 0.4R hit?
        if current_r >= 0.4:
            validations_met.append("premium rises ≥0.4R within 8 bars")
        # 2. Thesis stays in held direction?
        ts = str(thesis.get("composite_state") or "")
        if h.direction > 0 and "BULL" in ts and "HOLD" in ts:
            validations_met.append(
                "thesis composite stays HOLD_BULL for ≥5 bars")
        if h.direction < 0 and "BEAR" in ts and "HOLD" in ts:
            validations_met.append(
                "thesis composite stays HOLD_BEAR for ≥5 bars")
        # 3. Battlefield agrees
        verdict = str(bf.get("verdict") or "")
        if h.direction > 0 and verdict == "bullish_agreement":
            validations_met.append(
                "battlefield verdict remains bullish_agreement or stronger")
        if h.direction < 0 and verdict == "bearish_agreement":
            validations_met.append(
                "battlefield verdict remains bearish_agreement or stronger")
        # 4. Held leg acceptance
        acc = getattr(held_read, "acceptance", "") if held_read else ""
        if acc in ("defended", "normal", ""):
            validations_met.append(
                "held leg acceptance stays 'defended' or 'normal' (not 'rejected')"
            )
        # Invalidations active right now
        if current_r <= -1.0:
            invalidations_active.append(
                f"premium falls to stop_premium ₹{h.stop_premium:.2f} (−1R)")
        sstate = getattr(held_read, "spread_state", "") if held_read else ""
        if sstate == SPREAD_DANGEROUS:
            invalidations_active.append(
                "held contract spread state turns 'dangerous'")
        if acc == "rejected":
            invalidations_active.append("held leg acceptance flips to 'rejected'")

        ledger.append_bar(BarRecord(
            bar_index=bar_index,
            ts=snapshot.get("ts"),
            spot=_num(snapshot.get("spot")),
            held_premium=held_premium if held_premium is not None else 0.0,
            current_r=round(current_r, 4),
            best_r_so_far=round(state.best_r, 4),
            worst_r_so_far=round(state.worst_r, 4),
            cumulative_realized_pnl_rupees=round(net_now, 2),
            confidence=round(confidence, 3),
            high_water_confidence=round(state.high_water_confidence, 3),
            thesis_state=ts,
            iv_state=str(iv.get("state") or ""),
            battlefield_verdict=verdict,
            winding_zone=str(winding.get("zone") or "NO_WINDING"),
            decision_action=str(decision.get("action") or ""),
            held_spread_state=sstate,
            held_friendliness=_num(getattr(held_read, "friendliness", None)
                                    if held_read else None, 1.0),
            held_acceptance=acc,
            held_dod_z=_num(getattr(held_read, "dod_z", None)
                              if held_read else None, 0.0),
            validations_met=validations_met,
            invalidations_active=invalidations_active,
            contradictions_this_bar=[],
        ))

    def _portfolio_summary(self) -> Dict[str, Any]:
        return {
            "n_open": len(self._open_states),
            "n_closed": len(self.ledger_store.closed_positions()),
            "open_positions": [
                {
                    "position_id": pid,
                    "contract_label": s.hypothesis.contract_label,
                    "direction": s.hypothesis.direction,
                    "size_lots": s.hypothesis.size_lots,
                    "best_r": round(s.best_r, 3),
                    "worst_r": round(s.worst_r, 3),
                    "last_premium": round(s.last_premium, 2),
                } for pid, s in self._open_states.items()
            ],
            "daily_pnl_rupees": round(self.daily_pnl_rupees, 2),
            "cumulative_fees_rupees": round(self.cumulative_fees_rupees, 2),
            "ledger_summary": self.ledger_store.summary(),
            "scenario_web": (self._last_web_snapshot.to_dict()
                              if self._last_web_snapshot is not None else None),
            "projection_summary": self.projection.summary(),
            "active_patterns": [p.to_dict() for p in self._last_patterns],
            "mm_posterior": (self._last_mm_posterior.to_dict()
                              if self._last_mm_posterior else None),
            "fat_tail_score": (self._last_tail_score.to_dict()
                                if self._last_tail_score else None),
            "crowd_mirror": (self._last_crowd_report.to_dict()
                              if self._last_crowd_report else None),
            "portfolio_risk": (self._last_risk_report.to_dict()
                                if self._last_risk_report else None),
            "hedge_proposal": (self._last_hedge_proposal.to_dict()
                                if self._last_hedge_proposal else None),
            "adaptive_exit": self._last_exit_decision,
            "live_calibration": self._last_calibration,
            "weight_evolution": self._weight_evolution_summary(),
            "recent_trades": list(self._recent_trades)[:20],
            "strategy_attribution": self._strategy_attribution_summary(),
            "slippage_tracker": self.slippage_tracker.rolling_summary(),
        }

    def _strategy_confidence_multiplier(self, strategy_name: str,
                                            *, min_trades: int = 4,
                                            ) -> float:
        """Confidence-gated sizer (workaround B).

        Returns 1.0 until the strategy has at least ``min_trades`` closed
        trades. After that:
          - positive avg realized R + win rate >= 0.45 → up to 1.25×
            (rewards working strategies)
          - negative avg R OR win rate < 0.30          → 0.40×
            (throttles the bleeder until it earns back the right)
          - otherwise                                     → 1.0×
        """
        stats = self._strategy_stats.get(strategy_name)
        if stats is None or stats["n_trades"] < min_trades:
            return 1.0
        avg_r = stats["avg_realized_r"]
        wr = stats["wins"] / max(1, stats["n_trades"])
        if avg_r > 0.05 and wr >= 0.45:
            # Map avg_r ∈ [0.05, 0.50] → [1.0, 1.25].
            boost = min(1.25, 1.0 + (avg_r - 0.05) * 0.55)
            return float(boost)
        if avg_r < -0.05 or wr < 0.30:
            return 0.40
        return 1.0

    def _strategy_attribution_summary(self) -> Dict[str, Any]:
        """Surface per-strategy track record for the cockpit + post-mortem."""
        rows = []
        for name, s in self._strategy_stats.items():
            n = s["n_trades"]
            if n == 0:
                continue
            rows.append({
                "strategy": name,
                "n_trades": n,
                "wins": s["wins"],
                "losses": s["losses"],
                "win_rate": round(s["wins"] / max(1, n), 3),
                "total_realized_rupees": round(s["total_realized_rupees"], 2),
                "total_fees_rupees": round(s["total_fees_rupees"], 2),
                "avg_realized_r": round(s["avg_realized_r"], 3),
                "best_trade_rupees": round(s["best_trade"], 2),
                "worst_trade_rupees": round(s["worst_trade"], 2),
                "size_multiplier_now": self._strategy_confidence_multiplier(
                    name),
            })
        # Sort by total realized rupees descending so the cockpit shows
        # the most-impactful first.
        rows.sort(key=lambda r: r["total_realized_rupees"], reverse=True)
        return {"n_strategies_seen": len(rows), "rows": rows}

    def _weight_evolution_summary(self) -> Dict[str, Any]:
        try:
            analysis = self.weight_analyzer.analyze(self.weight_memory)
            return {
                "calibrator_paused": self.live_calibrator.paused,
                "n_snapshots": analysis.n_snapshots,
                "adaptability_index": analysis.adaptability_index,
                "trend_per_weight": analysis.trend_per_weight,
                "most_drifting_weight": analysis.most_drifting_weight,
                "coordinated_drift_score": analysis.coordinated_drift_score,
                "val_loss_trend": analysis.val_loss_trend,
                "warnings": analysis.warnings,
            }
        except Exception:
            return {"calibrator_paused": self.live_calibrator.paused,
                     "n_snapshots": 0}

    def reset_daily(self) -> None:
        """Reset daily counters at session start."""
        self.daily_pnl_rupees = 0.0
        self.cumulative_fees_rupees = 0.0


def _map(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
