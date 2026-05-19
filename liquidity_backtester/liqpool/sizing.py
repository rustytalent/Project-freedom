"""Track 5: position sizing with risk budget, sector limits, and confidence scaling.

A predicted P(respect) of 65% doesn't tell you HOW MUCH to risk. This module wraps the model
output with the three layers of risk protection from the trading plan:
  1. Per-trade risk cap (e.g. 1% of capital)
  2. Concurrent-position and per-sector concentration caps
  3. Daily-loss circuit breaker (stop trading if down -X% today)

Plus a confidence multiplier that scales size with EV — TRADE_HIGH_CONFIDENCE gets full size,
TRADE_CAUTIOUS gets half, DIR_FIGHT pools get further reduced.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SizingConfig:
    risk_per_trade_pct: float = 1.0
    """Max % of total account risked on any single trade. 1.0 = 1% per trade."""

    max_concurrent_positions: int = 4
    """Hard cap on how many positions can be open simultaneously."""

    max_sector_concentration_pct: float = 40.0
    """Hard cap on MARGIN committed per sector (% of account capital). Margin = notional /
    leverage. With 5x MIS leverage, 40% margin cap → 200% notional exposure per sector. This
    is what you intuitively mean by "40% in banking" when using leverage."""

    max_total_margin_pct: float = 85.0
    """Total margin used across all positions can't exceed X% of capital. Leaves buffer for
    market moves and fees so you don't get an auto-square-off on a wick."""

    daily_loss_cap_pct: float = 3.0
    """If today's realised P&L drops below -X%, sizing returns 0 for all new trades."""

    high_conf_size_mult: float = 1.0
    """Size multiplier for TRADE_HIGH_CONFIDENCE setups."""

    cautious_size_mult: float = 0.5
    """Size multiplier for TRADE_CAUTIOUS setups (lower EV but still tradeable)."""

    dir_fight_size_mult: float = 0.7
    """Extra penalty when the setup is against the direction model's prediction."""

    leverage: float = 5.0
    """MIS leverage on Indian intraday. Notional = shares × price; margin = notional / leverage."""


@dataclass
class SizingDecision:
    shares: int
    notional_inr: float
    margin_required_inr: float
    risk_inr: float
    confidence_multiplier: float
    blocked: bool = False
    block_reason: Optional[str] = None


@dataclass
class AccountState:
    """Snapshot of the trader's account passed in by the live runner."""
    capital_inr: float
    realised_pnl_today_inr: float = 0.0
    open_positions: List[Dict] = field(default_factory=list)
    # Each open position: {symbol, sector, notional_inr, side, ...}


# ---------------------------------------------------------------------------
# Sizing logic
# ---------------------------------------------------------------------------

def _confidence_multiplier(cfg: SizingConfig, verdict: str, dir_tag: str) -> float:
    """Map verdict + direction tag to a single multiplier."""
    if verdict == "TRADE_HIGH_CONFIDENCE":
        base = cfg.high_conf_size_mult
    elif verdict == "TRADE_CAUTIOUS":
        base = cfg.cautious_size_mult
    else:
        # WATCH / NO_TRADE / unknown verdict — don't size up.
        return 0.0

    if dir_tag == "DIR_FIGHT":
        base *= cfg.dir_fight_size_mult
    # DIR_ALIGN / DIR_NEUTRAL / DIR_NA leave the multiplier at base.
    return float(base)


def compute_shares(account_capital: float, risk_per_trade_pct: float,
                    stop_distance_inr: float) -> int:
    """Compute risk-budget shares: shares = (capital × pct) / stop_distance. Confidence
    multiplier is NOT applied here — it's a final post-cap multiplier (see size_setup)."""
    if stop_distance_inr <= 0:
        return 0
    risk_budget = account_capital * (risk_per_trade_pct / 100.0)
    return max(0, int(risk_budget / stop_distance_inr))


def size_setup(setup: Dict, cfg: SizingConfig, account: AccountState,
                verdict: str = "TRADE_HIGH_CONFIDENCE") -> SizingDecision:
    """Decide position size for ONE setup.

    `setup` must contain: symbol, side, entry, stop, sector, dir_tag.
    Returns a SizingDecision with shares + reason if blocked.
    """
    entry = float(setup["entry"])
    stop = float(setup["stop"])
    side = setup["side"]                       # "above" (sell) or "below" (buy)
    sector = setup.get("sector", "OTHER")
    dir_tag = setup.get("dir_tag", "DIR_NEUTRAL")
    sym = setup["symbol"]

    # Stop distance is the absolute distance in INR per share.
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return SizingDecision(0, 0.0, 0.0, 0.0, 0.0, blocked=True,
                              block_reason="zero stop distance")

    # 1. Daily-loss circuit
    loss_pct = account.realised_pnl_today_inr / max(account.capital_inr, 1.0) * 100.0
    if loss_pct <= -cfg.daily_loss_cap_pct:
        return SizingDecision(0, 0.0, 0.0, 0.0, 0.0, blocked=True,
                              block_reason=f"daily loss cap hit ({loss_pct:.1f}% <= "
                                            f"-{cfg.daily_loss_cap_pct:.1f}%)")

    # 2. Concurrent-position cap
    if len(account.open_positions) >= cfg.max_concurrent_positions:
        return SizingDecision(0, 0.0, 0.0, 0.0, 0.0, blocked=True,
                              block_reason=f"already {len(account.open_positions)} positions "
                                            f"open (cap {cfg.max_concurrent_positions})")

    # 3. Sector concentration cap
    sector_notional = sum(p["notional_inr"] for p in account.open_positions
                           if p.get("sector") == sector)

    # 4. Confidence multiplier
    cmult = _confidence_multiplier(cfg, verdict, dir_tag)
    if cmult <= 0:
        return SizingDecision(0, 0.0, 0.0, 0.0, 0.0, blocked=True,
                              block_reason=f"verdict {verdict!r} not tradeable")

    # 5. Compute risk-budget shares (NO cmult yet — applied at the end so it consistently
    # reduces size even when a hard cap is binding)
    shares = compute_shares(account.capital_inr, cfg.risk_per_trade_pct, stop_dist)
    if shares <= 0:
        return SizingDecision(0, 0.0, 0.0, 0.0, 0.0, blocked=True,
                              block_reason="computed 0 shares (stop too wide?)")

    notional = shares * entry
    leverage = max(cfg.leverage, 1.0)
    margin = notional / leverage

    # 6. Sector concentration post-trade check — capped on MARGIN, not raw notional. With 5x
    # leverage, 40% margin cap = 200% notional exposure per sector, which is what users
    # actually want when they say "40% per sector".
    sector_margin = sum(p["notional_inr"] / leverage for p in account.open_positions
                        if p.get("sector") == sector)
    sector_margin_cap = account.capital_inr * (cfg.max_sector_concentration_pct / 100.0)
    if sector_margin + margin > sector_margin_cap:
        margin_budget_left = max(0.0, sector_margin_cap - sector_margin)
        if margin_budget_left <= 0:
            return SizingDecision(0, 0.0, 0.0, 0.0, cmult, blocked=True,
                                  block_reason=f"sector {sector} at margin cap "
                                                f"({cfg.max_sector_concentration_pct:.0f}%)")
        max_notional_for_sector = margin_budget_left * leverage
        shares = int(max_notional_for_sector / entry)
        if shares <= 0:
            return SizingDecision(0, 0.0, 0.0, 0.0, cmult, blocked=True,
                                  block_reason=f"sector {sector} margin budget exhausted")
        notional = shares * entry
        margin = notional / leverage

    # 7. Total-margin check — don't blow through the account's free margin.
    total_margin = sum(p["notional_inr"] / leverage for p in account.open_positions)
    total_margin_cap = account.capital_inr * (cfg.max_total_margin_pct / 100.0)
    if total_margin + margin > total_margin_cap:
        margin_budget_left = max(0.0, total_margin_cap - total_margin)
        if margin_budget_left <= 0:
            return SizingDecision(0, 0.0, 0.0, 0.0, cmult, blocked=True,
                                  block_reason=f"total margin cap hit "
                                                f"({cfg.max_total_margin_pct:.0f}%)")
        max_notional = margin_budget_left * leverage
        shares = int(max_notional / entry)
        if shares <= 0:
            return SizingDecision(0, 0.0, 0.0, 0.0, cmult, blocked=True,
                                  block_reason="free margin exhausted")
        notional = shares * entry
        margin = notional / leverage

    # 8. Apply confidence multiplier at the END — guarantees DIR_FIGHT / TRADE_CAUTIOUS trades
    # are smaller even when a hard cap was binding before this point.
    if cmult < 1.0:
        shares = max(0, int(shares * cmult))
        if shares <= 0:
            return SizingDecision(0, 0.0, 0.0, 0.0, cmult, blocked=True,
                                  block_reason=f"confidence multiplier {cmult:.2f} reduced "
                                                f"shares to 0")
        notional = shares * entry
        margin = notional / leverage

    risk_inr = shares * stop_dist
    return SizingDecision(
        shares=shares,
        notional_inr=float(notional),
        margin_required_inr=float(margin),
        risk_inr=float(risk_inr),
        confidence_multiplier=cmult,
    )


def format_sizing_line(setup: Dict, decision: SizingDecision) -> str:
    """One-line summary for the live runner."""
    sym = setup["symbol"]
    if decision.blocked:
        return f"  [{sym}] BLOCKED: {decision.block_reason}"
    side = "BUY" if setup["side"] == "below" else "SELL"
    return (f"  [{sym}] {side} {decision.shares} shares  "
            f"notional ₹{decision.notional_inr:,.0f}  "
            f"margin ₹{decision.margin_required_inr:,.0f}  "
            f"risk ₹{decision.risk_inr:,.0f}  "
            f"(conf×{decision.confidence_multiplier:.2f})")
