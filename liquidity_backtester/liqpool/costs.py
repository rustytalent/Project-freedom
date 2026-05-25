"""Indian equity execution friction models.

The live decision layer should not call a setup tradeable unless the expected
move survives brokerage, taxes, exchange charges, stamp duty, and slippage.
This module keeps those assumptions explicit and serialisable in reports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class ZerodhaEquityCostConfig:
    """Current Zerodha equity cash-market charge assumptions.

    Source: Zerodha charges page, checked 2026-05-25.
    Percent fields are decimals: 0.00025 = 0.025%.
    """

    product: str = "intraday"
    brokerage_rate: float = 0.0003
    brokerage_cap_per_order: float = 20.0
    stt_intraday_sell: float = 0.00025
    stt_delivery_buy_sell: float = 0.001
    exchange_txn_rate: float = 0.0000307
    sebi_rate: float = 0.000001
    stamp_intraday_buy: float = 0.00003
    stamp_delivery_buy: float = 0.00015
    gst_rate: float = 0.18
    dp_charge_delivery_sell: float = 15.34
    slippage_bps_per_side: float = 1.0
    source_url: str = "https://zerodha.com/charges/"
    source_checked: str = "2026-05-25"

    def to_dict(self) -> Dict:
        return asdict(self)


def _brokerage(turnover: float, cfg: ZerodhaEquityCostConfig) -> float:
    if cfg.product == "delivery":
        return 0.0
    return min(cfg.brokerage_cap_per_order, turnover * cfg.brokerage_rate)


def estimate_round_trip_charges(
    entry_price: float,
    exit_price: float,
    quantity: int = 1,
    cfg: ZerodhaEquityCostConfig | None = None,
) -> Dict[str, float]:
    """Estimate all-in round-trip charges for one equity trade.

    The function is direction-agnostic. For intraday, STT is charged on the sell
    leg; for delivery, STT is charged on both legs and DP is charged once on sell.
    Slippage is modelled separately as a bps cost on both entry and exit turnover.
    """
    cfg = cfg or ZerodhaEquityCostConfig()
    quantity = max(int(quantity), 1)
    buy_turnover = abs(float(entry_price)) * quantity
    sell_turnover = abs(float(exit_price)) * quantity
    turnover = buy_turnover + sell_turnover

    brokerage = _brokerage(buy_turnover, cfg) + _brokerage(sell_turnover, cfg)
    if cfg.product == "delivery":
        stt = (buy_turnover + sell_turnover) * cfg.stt_delivery_buy_sell
        stamp = buy_turnover * cfg.stamp_delivery_buy
        dp = cfg.dp_charge_delivery_sell
    else:
        stt = sell_turnover * cfg.stt_intraday_sell
        stamp = buy_turnover * cfg.stamp_intraday_buy
        dp = 0.0
    exchange_txn = turnover * cfg.exchange_txn_rate
    sebi = turnover * cfg.sebi_rate
    gst = (brokerage + exchange_txn + sebi) * cfg.gst_rate
    slippage = turnover * (cfg.slippage_bps_per_side / 10000.0)
    total = brokerage + stt + exchange_txn + sebi + stamp + gst + dp + slippage

    return {
        "quantity": float(quantity),
        "turnover": float(turnover),
        "brokerage": float(brokerage),
        "stt": float(stt),
        "exchange_txn": float(exchange_txn),
        "sebi": float(sebi),
        "stamp": float(stamp),
        "gst": float(gst),
        "dp": float(dp),
        "slippage": float(slippage),
        "total_cost": float(total),
        "cost_per_share": float(total / quantity),
        "cost_bps_turnover": float(total / max(turnover, 1e-9) * 10000.0),
    }


def trade_levels(pool_low: float, pool_high: float, atr_value: float, side: str) -> Dict[str, float]:
    """Return default entry/stop/target levels used by the live plan.

    side is the pool location relative to current price:
    - "below": buy support/liquidity below current price
    - "above": sell resistance/liquidity above current price
    """
    atr_value = max(float(atr_value), 1e-9)
    if side == "below":
        entry = float(pool_high)
        stop = float(pool_low) - 0.5 * atr_value
        target = float(pool_high) + 2.0 * atr_value
    else:
        entry = float(pool_low)
        stop = float(pool_high) + 0.5 * atr_value
        target = float(pool_low) - 2.0 * atr_value
    return {"entry": entry, "stop": stop, "target": target}


def expected_trade_value(
    *,
    side: str,
    entry: float,
    stop: float,
    target: float,
    p_touch: float,
    p_reaction: float,
    quantity: int = 1,
    cfg: ZerodhaEquityCostConfig | None = None,
) -> Dict[str, float]:
    """Cost-adjusted expectancy for a confirmation-style trade.

    P_touch controls whether the alert is likely to become actionable. P_reaction
    controls whether the post-touch confirmation historically worked. The returned
    p_trade is intentionally not just P_touch: it is P_touch * P_reaction.
    """
    p_touch = min(max(float(p_touch), 0.0), 1.0)
    p_reaction = min(max(float(p_reaction), 0.0), 1.0)
    quantity = max(int(quantity), 1)

    if side == "below":
        direction = "UP"
        direction_sign = 1
        win_per_share = max(0.0, float(target) - float(entry))
        loss_per_share = max(0.0, float(entry) - float(stop))
        win_exit = target
        loss_exit = stop
    else:
        direction = "DOWN"
        direction_sign = -1
        win_per_share = max(0.0, float(entry) - float(target))
        loss_per_share = max(0.0, float(stop) - float(entry))
        win_exit = target
        loss_exit = stop

    win_costs = estimate_round_trip_charges(entry, win_exit, quantity, cfg)
    loss_costs = estimate_round_trip_charges(entry, loss_exit, quantity, cfg)
    expected_cost_per_share = (
        p_reaction * win_costs["cost_per_share"]
        + (1.0 - p_reaction) * loss_costs["cost_per_share"]
    )
    gross_conditional = p_reaction * win_per_share - (1.0 - p_reaction) * loss_per_share
    net_conditional = gross_conditional - expected_cost_per_share
    risk_per_share = max(loss_per_share, 1e-9)
    net_expectancy_r = float((p_touch * net_conditional) / risk_per_share)

    return {
        "direction": direction,
        "direction_sign": direction_sign,
        "p_touch": float(p_touch),
        "p_reaction": float(p_reaction),
        "p_trade": float(p_touch * p_reaction),
        "gross_expectancy_per_share": float(p_touch * gross_conditional),
        "net_expectancy_per_share": float(p_touch * net_conditional),
        "net_expectancy_r": net_expectancy_r,
        "directional_net_expectancy_r": float(abs(net_expectancy_r) * direction_sign),
        "conditional_net_expectancy_per_share": float(net_conditional),
        "win_per_share": float(win_per_share),
        "loss_per_share": float(loss_per_share),
        "expected_cost_per_share": float(expected_cost_per_share),
        "win_cost_per_share": float(win_costs["cost_per_share"]),
        "loss_cost_per_share": float(loss_costs["cost_per_share"]),
    }
