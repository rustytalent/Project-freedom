"""Execution economics — fees-FIRST EV gate.

The founder's #1 concrete defense against bleeding capital: every trade
must clear (fees + slippage + minimum-edge margin) BEFORE it fires. A
trade that the engine wants but the economics layer refuses is a saved
loss — and saved losses compound.

Critical rule (founder, 2026-06-19):
  "When you know the loser is making, you have to cut it. Even if the
   fees will incur. A bigger loser will incur ~10x more fees."

So this module computes TWO things:
  1. Pre-entry EV gate: is the trade worth taking given fees + expected
     slippage + the engine's confidence?
  2. In-position exit-acceleration: when a loss is forming, the fees-to-
     loss ratio should NOT delay the exit. The accelerator says "yes, exit
     now even if fees eat a chunk — the bigger loss tomorrow is worse."

Fees breakdown (Zerodha NIFTY weekly options, 2026 rates):
  * Brokerage:         flat ₹20 per executed order (₹40 round-trip)
  * STT:               0.0625% on SELL-side premium notional (only sell)
  * Exchange charges:  ~0.0035% on premium turnover (both legs)
  * SEBI charges:      0.0001% on premium turnover (both legs)
  * Stamp duty:        0.003% on BUY-side notional (only buy)
  * GST:               18% on (brokerage + exchange + SEBI charges)

For NIFTY lot=65 (per the founder's confirmation 2026-06-19), this comes
out to roughly ₹80-110 per round-trip per lot at ATM-ish premiums — the
founder's stated ~₹100/lot figure.

Risk budget defaults: per-trade max loss ₹1000, daily bleed floor ₹5000
(half of the founder's stated "₹10k = doomed" threshold so we stop bleeding
well before the threshold actually hits). All operator-configurable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────
# Constants — calibrated to the founder's reality (2026-06-19)
# ─────────────────────────────────────────────────────────────────

NIFTY_LOT_SIZE: int = 65                    # founder-confirmed (was 75, now 65)
NIFTY_MAX_LOSS_PER_TRADE: float = 1000.0    # founder-confirmed (₹500-1000 budget)
NIFTY_MAX_DAILY_BLEED: float = 5000.0       # half of founder's "₹10k = doomed" floor

# Zerodha rate card 2026 (NIFTY F&O — options).
ZERODHA_BROKERAGE_FLAT_PER_LEG: float = 20.0     # ₹20 entry + ₹20 exit
ZERODHA_STT_PCT_SELL: float = 0.000625           # 0.0625% on sell premium notional
ZERODHA_EXCHANGE_PCT: float = 0.0000345          # ~0.00345% on premium turnover (both legs)
ZERODHA_SEBI_PCT: float = 0.000001               # 0.0001% on premium turnover
ZERODHA_STAMP_DUTY_BUY_PCT: float = 0.00003      # 0.003% on buy notional
ZERODHA_GST_PCT: float = 0.18                    # 18% on (brokerage + exch + sebi)

# Slippage assumption defaults — calibrated to NIFTY weekly ATM:
# Tight spread → ~₹0.30 per leg per lot effective slippage; wider → ₹1.00+.
DEFAULT_SLIPPAGE_PER_LEG: float = 0.50           # rupees per share (lot_size shares per lot)

# Edge margin — minimum expected profit AFTER fees + slippage to bother
# entering at all. Without this we trade the 50/50s and lose to fees.
DEFAULT_MIN_EDGE_MULTIPLE: float = 1.5           # require E[profit] ≥ 1.5x cost


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionEconomicsConfig:
    """Knobs for the economics gate.

    Everything is operator-configurable; the defaults are founder-calibrated
    for NIFTY weekly options on Zerodha. Change any of these to match your
    actual broker fee structure.
    """
    # Lot & risk budgets
    lot_size: int = NIFTY_LOT_SIZE
    max_loss_per_trade: float = NIFTY_MAX_LOSS_PER_TRADE
    max_daily_bleed: float = NIFTY_MAX_DAILY_BLEED
    max_open_positions: int = 4

    # Fees
    brokerage_flat_per_leg: float = ZERODHA_BROKERAGE_FLAT_PER_LEG
    stt_pct_sell: float = ZERODHA_STT_PCT_SELL
    exchange_pct: float = ZERODHA_EXCHANGE_PCT
    sebi_pct: float = ZERODHA_SEBI_PCT
    stamp_duty_buy_pct: float = ZERODHA_STAMP_DUTY_BUY_PCT
    gst_pct: float = ZERODHA_GST_PCT

    # Slippage
    default_slippage_per_leg: float = DEFAULT_SLIPPAGE_PER_LEG

    # Edge required after costs
    min_edge_multiple: float = DEFAULT_MIN_EDGE_MULTIPLE

    # Exit acceleration thresholds — when a loss is forming, the
    # fees-to-loss ratio should NOT cause us to delay an exit. These
    # parameters define "loss is forming" precisely.
    exit_accel_loss_pct: float = 0.20      # premium down ≥20% from entry → consider acceleration
    exit_accel_min_age_bars: int = 2       # only after min hold
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.lot_size <= 0:
            raise ValueError("lot_size must be > 0")
        if self.max_loss_per_trade <= 0:
            raise ValueError("max_loss_per_trade must be > 0")
        if self.min_edge_multiple < 1.0:
            raise ValueError("min_edge_multiple must be >= 1.0 — otherwise you're trading fees-negative EV")


# ─────────────────────────────────────────────────────────────────
# Fee computation
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeeBreakdown:
    """Per-trade fee breakdown. Every component visible so the operator can
    audit and the executor's ledger can serialize the full economic record."""
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp_duty: float
    gst: float
    total: float
    notional_buy: float
    notional_sell: float
    lots: int
    lot_size: int

    @property
    def per_lot(self) -> float:
        return self.total / max(1, self.lots)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def fees_for_round_trip(*,
                        entry_premium: float,
                        exit_premium: float,
                        lots: int,
                        is_long: bool,
                        cfg: Optional[ExecutionEconomicsConfig] = None,
                        ) -> FeeBreakdown:
    """Compute full fees for one round-trip on a NIFTY weekly option.

    For a LONG: buy at entry_premium, sell at exit_premium.
    For a SHORT: sell at entry_premium, buy at exit_premium.

    All percentages applied to *premium notional* (premium × lot_size × lots),
    not strike notional. STT applies only to the sell side. Stamp duty only
    to the buy side. Exchange + SEBI charge applies on both legs.
    """
    cfg = cfg or ExecutionEconomicsConfig()
    lots = max(1, int(lots))
    shares = lots * cfg.lot_size

    if is_long:
        buy_notional = float(entry_premium) * shares
        sell_notional = float(exit_premium) * shares
    else:
        sell_notional = float(entry_premium) * shares
        buy_notional = float(exit_premium) * shares

    brokerage = 2 * cfg.brokerage_flat_per_leg          # both legs flat
    stt = cfg.stt_pct_sell * sell_notional
    exchange = cfg.exchange_pct * (buy_notional + sell_notional)
    sebi = cfg.sebi_pct * (buy_notional + sell_notional)
    stamp_duty = cfg.stamp_duty_buy_pct * buy_notional
    gst = cfg.gst_pct * (brokerage + exchange + sebi)

    total = brokerage + stt + exchange + sebi + stamp_duty + gst
    return FeeBreakdown(
        brokerage=brokerage, stt=stt, exchange=exchange, sebi=sebi,
        stamp_duty=stamp_duty, gst=gst, total=total,
        notional_buy=buy_notional, notional_sell=sell_notional,
        lots=lots, lot_size=cfg.lot_size,
    )


def estimate_slippage(*,
                      lots: int,
                      friendliness: float = 1.0,
                      spread_state: str = "clean",
                      cfg: Optional[ExecutionEconomicsConfig] = None,
                      ) -> float:
    """Estimate slippage cost in rupees for a round-trip.

    Worse friendliness / spread state → linearly higher slippage.
    """
    cfg = cfg or ExecutionEconomicsConfig()
    base = cfg.default_slippage_per_leg
    # Friendliness 1.0 → base; 0.5 → 2x base; 0.0 → 4x base.
    friendliness = max(0.05, min(1.0, float(friendliness)))
    mult_friend = 1.0 / (friendliness * friendliness + 0.05)
    mult_state = 1.0
    if spread_state == "widening":
        mult_state = 1.6
    elif spread_state == "dangerous":
        mult_state = 3.0
    per_leg = base * mult_friend * mult_state
    return 2.0 * per_leg * lots * cfg.lot_size


def minimum_profitable_premium_delta(*,
                                     entry_premium: float,
                                     lots: int,
                                     is_long: bool,
                                     friendliness: float = 1.0,
                                     spread_state: str = "clean",
                                     cfg: Optional[ExecutionEconomicsConfig] = None,
                                     ) -> float:
    """The minimum premium-move (in rupees per share) that the position
    must clear before it earns a single rupee profit — i.e., the fees +
    slippage floor expressed as a per-share premium move.

    Used as the founder's "fees floor": every entry must have an expected
    profit > this × min_edge_multiple before being approved.
    """
    cfg = cfg or ExecutionEconomicsConfig()
    # Estimate fees using a 0% premium move (entry = exit) as the cost baseline
    fees_at_zero = fees_for_round_trip(
        entry_premium=entry_premium, exit_premium=entry_premium,
        lots=lots, is_long=is_long, cfg=cfg,
    ).total
    slip = estimate_slippage(lots=lots, friendliness=friendliness,
                             spread_state=spread_state, cfg=cfg)
    total_cost = fees_at_zero + slip
    shares = lots * cfg.lot_size
    return total_cost / max(1.0, shares)


# ─────────────────────────────────────────────────────────────────
# EV gate
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EVDecision:
    """Output of the EV gate."""
    approve: bool
    expected_profit_rupees: float
    fees_rupees: float
    slippage_rupees: float
    minimum_profitable_delta: float       # per-share premium move needed to break even
    edge_multiple: float                   # E[profit_after_costs] / total_costs
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def expected_value_after_costs(*,
                               entry_premium: float,
                               expected_exit_premium: float,
                               prob_target: float,
                               prob_stop: float,
                               stop_premium: float,
                               lots: int,
                               is_long: bool,
                               friendliness: float = 1.0,
                               spread_state: str = "clean",
                               cfg: Optional[ExecutionEconomicsConfig] = None,
                               ) -> EVDecision:
    """Probabilistic EV gate.

    E[profit] = p_target × (exit_premium - entry_premium) × shares
              + p_stop × (stop_premium - entry_premium) × shares
              - fees(target) × p_target - fees(stop) × p_stop
              - slippage
    For shorts, the side is flipped.
    """
    cfg = cfg or ExecutionEconomicsConfig()
    shares = lots * cfg.lot_size
    side = 1 if is_long else -1

    target_pnl_per_share = side * (expected_exit_premium - entry_premium)
    stop_pnl_per_share = side * (stop_premium - entry_premium)
    target_gross = target_pnl_per_share * shares
    stop_gross = stop_pnl_per_share * shares

    fees_at_target = fees_for_round_trip(
        entry_premium=entry_premium, exit_premium=expected_exit_premium,
        lots=lots, is_long=is_long, cfg=cfg).total
    fees_at_stop = fees_for_round_trip(
        entry_premium=entry_premium, exit_premium=stop_premium,
        lots=lots, is_long=is_long, cfg=cfg).total
    slippage = estimate_slippage(lots=lots, friendliness=friendliness,
                                  spread_state=spread_state, cfg=cfg)

    expected_profit = (
        prob_target * (target_gross - fees_at_target)
        + prob_stop * (stop_gross - fees_at_stop)
        - slippage
    )
    avg_fees = prob_target * fees_at_target + prob_stop * fees_at_stop
    total_cost = avg_fees + slippage
    edge_mult = (expected_profit / max(cfg.eps, total_cost)) if total_cost > 0 else 0.0

    reasons: List[str] = []
    approve = True
    if expected_profit <= 0:
        approve = False
        reasons.append(f"E[profit] = ₹{expected_profit:.0f} ≤ 0; pure fees burn")
    elif edge_mult < cfg.min_edge_multiple:
        approve = False
        reasons.append(
            f"edge multiple {edge_mult:.2f} below floor {cfg.min_edge_multiple:.2f} — "
            f"E[profit]=₹{expected_profit:.0f} but costs=₹{total_cost:.0f}"
        )

    min_delta = minimum_profitable_premium_delta(
        entry_premium=entry_premium, lots=lots, is_long=is_long,
        friendliness=friendliness, spread_state=spread_state, cfg=cfg,
    )

    return EVDecision(
        approve=approve,
        expected_profit_rupees=round(expected_profit, 2),
        fees_rupees=round(avg_fees, 2),
        slippage_rupees=round(slippage, 2),
        minimum_profitable_delta=round(min_delta, 4),
        edge_multiple=round(edge_mult, 3),
        reasons=reasons,
    )


# ─────────────────────────────────────────────────────────────────
# Exit acceleration — cut losers fast even if fees ratio is bad
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExitAcceleration:
    """Output of the exit accelerator."""
    accelerate: bool
    current_loss_pct: float                # how far premium fell (signed)
    projected_loss_if_held_rupees: float   # honest estimate if we hold to stop
    fees_to_cut_now_rupees: float          # what cutting now costs in fees
    fees_to_cut_at_stop_rupees: float      # what cutting at stop would cost
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def should_accelerate_exit(*,
                           entry_premium: float,
                           current_premium: float,
                           stop_premium: float,
                           bars_held: int,
                           lots: int,
                           is_long: bool,
                           thesis_confidence_decay: float = 0.0,
                           friendliness: float = 1.0,
                           spread_state: str = "clean",
                           cfg: Optional[ExecutionEconomicsConfig] = None,
                           ) -> ExitAcceleration:
    """Should we cut RIGHT NOW even though the position hasn't hit its stop?

    The founder's rule: "when the loser is making, cut it. Even if the
    fees will incur. A bigger loser will incur ~10x more fees."

    Triggers acceleration when ALL of:
      * position has aged past min hold,
      * premium has dropped past loss_pct threshold,
      * thesis confidence has decayed materially,
      * the loss-if-held projection exceeds the cost-to-exit-now.
    """
    cfg = cfg or ExecutionEconomicsConfig()
    side = 1 if is_long else -1
    loss_pct = side * (current_premium - entry_premium) / max(cfg.eps, entry_premium)
    # loss_pct is negative when we're underwater.
    underwater_pct = -loss_pct   # positive = how much we're down

    fees_now = fees_for_round_trip(
        entry_premium=entry_premium, exit_premium=current_premium,
        lots=lots, is_long=is_long, cfg=cfg).total
    fees_at_stop = fees_for_round_trip(
        entry_premium=entry_premium, exit_premium=stop_premium,
        lots=lots, is_long=is_long, cfg=cfg).total

    shares = lots * cfg.lot_size
    loss_at_stop_rupees = side * (stop_premium - entry_premium) * shares
    # loss_at_stop_rupees is negative for a long going to stop.
    projected_loss = -loss_at_stop_rupees + fees_at_stop  # how much we'd lose if held to stop
    cut_now_loss = -((current_premium - entry_premium) * side * shares) + fees_now

    reason = ""
    accelerate = False
    if bars_held < cfg.exit_accel_min_age_bars:
        reason = f"too early — bars_held={bars_held} < min={cfg.exit_accel_min_age_bars}"
    elif underwater_pct < cfg.exit_accel_loss_pct:
        reason = f"not deep enough — underwater {underwater_pct:.1%} < {cfg.exit_accel_loss_pct:.0%}"
    elif thesis_confidence_decay < 0.15:
        reason = f"thesis still mostly intact (decay {thesis_confidence_decay:.2f})"
    elif projected_loss <= cut_now_loss:
        reason = (f"projected stop loss ₹{projected_loss:.0f} not worse than "
                  f"cut-now loss ₹{cut_now_loss:.0f}")
    else:
        accelerate = True
        reason = (
            f"loss forming — premium down {underwater_pct:.0%}, thesis decayed "
            f"{thesis_confidence_decay:.2f}; cutting now costs ₹{cut_now_loss:.0f} vs "
            f"₹{projected_loss:.0f} if held to stop. Cut now."
        )
    return ExitAcceleration(
        accelerate=accelerate,
        current_loss_pct=round(loss_pct, 4),
        projected_loss_if_held_rupees=round(projected_loss, 2),
        fees_to_cut_now_rupees=round(fees_now, 2),
        fees_to_cut_at_stop_rupees=round(fees_at_stop, 2),
        reason=reason,
    )


def daily_bleed_blocker(daily_pnl_rupees: float,
                        cfg: Optional[ExecutionEconomicsConfig] = None,
                        ) -> Optional[str]:
    """The hard portfolio kill — if today's net P&L is below the bleed
    threshold, refuse ALL new entries until tomorrow."""
    cfg = cfg or ExecutionEconomicsConfig()
    if daily_pnl_rupees <= -cfg.max_daily_bleed:
        return (f"daily bleed limit hit: net P&L ₹{daily_pnl_rupees:.0f} ≤ "
                f"-₹{cfg.max_daily_bleed:.0f}; refusing new entries today")
    return None
