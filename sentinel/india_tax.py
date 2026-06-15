"""Indian-market-specific differentiation: tax-impact + Muhurat calendar.

Two surfaces no Western trading platform has, both core to Indian
options trading:

  * Tax-impact preview: every trade has STT (Securities Transaction
    Tax), STCG/LTCG (capital gains), brokerage, GST, SEBI charges,
    stamp duty. Every customer asks "what's my real take-home P&L?"
    For options, STT is calculated on SELL side at 0.0625% of
    premium (intrinsic value for exercised options); STCG flat 20%
    for FY 2025-26. We compute the post-tax figure honestly.

  * Muhurat / festival calendar: Diwali Muhurat trading (1-hour
    special session on Diwali evening); Republic Day, Holi, Eid
    holidays; expiry-week quirks (Thursday weekly expiry shifts to
    Wednesday when Thursday is a holiday). The cockpit surfaces a
    "next session" chip + Muhurat banner so the operator never
    discovers a holiday at 09:15.

Methodology citations (so a CA can audit the math):
  * STT: Finance Act 2024 Schedule 2; effective 01-Apr-2024
  * STCG 20%: Finance Act 2024 §111A as amended; effective FY2024-25
  * SEBI turnover charge: SEBI circular SMD-1/23068/93
  * GST 18% on brokerage: CGST Act 2017 §9
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .io_decl import IOSpec, declare

IST = timezone(timedelta(hours=5, minutes=30))


# ─────────────────────────────────────────────────────────────────
# Tax-impact calculator
# ─────────────────────────────────────────────────────────────────

# Defaults — overridable per call when Codex's pricing-table changes
DEFAULT_BROKERAGE_RUPEES = 20.0        # Zerodha flat ₹20 per order
STT_OPTIONS_SELL_PCT = 0.000625        # 0.0625% on premium, sell-side only
STT_OPTIONS_EXERCISE_PCT = 0.00125     # 0.125% on intrinsic at exercise
EXCHANGE_TXN_OPTIONS_PCT = 0.000503    # NSE F&O turnover charge
SEBI_TURNOVER_PCT = 0.0000010          # SEBI charge — 1 paisa per lakh
GST_PCT = 0.18                          # 18% on (brokerage + exchange + SEBI)
STAMP_DUTY_BUY_PCT = 0.00003           # 0.003% on buy-side premium
STCG_PCT = 0.20                         # Section 111A FY2024-25 onwards


@dataclass
class TaxBreakdown:
    """Itemised cost of one round-trip options trade. Each field is
    rupees (positive = a cost). ``net_pnl`` = gross P&L minus everything."""
    gross_pnl: float
    brokerage: float
    stt: float
    exchange_txn: float
    sebi_charge: float
    stamp_duty: float
    gst: float
    capital_gains_tax: float
    total_charges: float
    net_pnl: float

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


def options_trade_tax_impact(
    buy_premium: float, sell_premium: float, quantity: int = 0,
    lots: int = 1, lot_size: int = 75,
    brokerage_per_leg: float = DEFAULT_BROKERAGE_RUPEES,
    apply_capital_gains: bool = True,
) -> TaxBreakdown:
    """One round-trip options trade: buy at ``buy_premium``, sell at
    ``sell_premium``, ``quantity`` total units (or pass lots × lot_size).

    The convention: brokerage is per LEG (₹20 × 2 = ₹40 round-trip).
    Capital gains tax only applies when ``gross_pnl > 0`` AND
    ``apply_capital_gains`` is set; if the operator's annual options
    P&L is treated as F&O business income (Section 44AD), this should
    be False and the operator handles it at year-end.
    """
    qty = max(quantity, lots * lot_size)
    buy_turnover = buy_premium * qty
    sell_turnover = sell_premium * qty
    gross_pnl = (sell_premium - buy_premium) * qty
    # STT on SELL side only for options (not on buy)
    stt = sell_turnover * STT_OPTIONS_SELL_PCT
    # Exchange + SEBI charged on both legs
    total_turnover = buy_turnover + sell_turnover
    exchange_txn = total_turnover * EXCHANGE_TXN_OPTIONS_PCT
    sebi_charge = total_turnover * SEBI_TURNOVER_PCT
    # Stamp duty on buy side only
    stamp_duty = buy_turnover * STAMP_DUTY_BUY_PCT
    # Brokerage both legs
    brokerage = brokerage_per_leg * 2
    # GST on brokerage + exchange + SEBI (not on STT / stamp / CGT)
    gst = (brokerage + exchange_txn + sebi_charge) * GST_PCT
    # Capital gains tax on the gross P&L if positive
    cgt = 0.0
    if apply_capital_gains and gross_pnl > 0:
        # CGT is computed on (gross - all transaction charges) — the
        # CBDT treats brokerage + STT + GST as deductible expenses.
        deductible = (brokerage + stt + exchange_txn + sebi_charge
                      + stamp_duty + gst)
        taxable = max(0.0, gross_pnl - deductible)
        cgt = taxable * STCG_PCT
    total_charges = (brokerage + stt + exchange_txn + sebi_charge
                     + stamp_duty + gst + cgt)
    net_pnl = gross_pnl - total_charges
    return TaxBreakdown(
        gross_pnl=round(gross_pnl, 2),
        brokerage=round(brokerage, 2),
        stt=round(stt, 2),
        exchange_txn=round(exchange_txn, 2),
        sebi_charge=round(sebi_charge, 2),
        stamp_duty=round(stamp_duty, 2),
        gst=round(gst, 2),
        capital_gains_tax=round(cgt, 2),
        total_charges=round(total_charges, 2),
        net_pnl=round(net_pnl, 2),
    )


# ─────────────────────────────────────────────────────────────────
# NSE F&O holiday calendar — 2026 published list + recurring patterns
# ─────────────────────────────────────────────────────────────────

# Format: ISO date → label. Update yearly from NSE's published list.
NSE_HOLIDAYS_2026: Dict[str, str] = {
    "2026-01-26": "Republic Day",
    "2026-03-03": "Holi",
    "2026-03-17": "Eid-ul-Fitr",
    "2026-04-03": "Good Friday",
    "2026-05-01": "Maharashtra Day",
    "2026-05-24": "Eid-ul-Azha",
    "2026-08-15": "Independence Day",
    "2026-08-31": "Ganesh Chaturthi",
    "2026-10-02": "Gandhi Jayanti",
    "2026-10-20": "Diwali · Lakshmi Pujan",
    "2026-10-21": "Diwali · Bali Pratipada",
    "2026-12-25": "Christmas",
}

# Muhurat trading — special 60-min session on Diwali evening
# 2026: 18:30 - 19:30 IST on Diwali Lakshmi Pujan day
MUHURAT_SESSIONS_2026: Dict[str, Dict[str, str]] = {
    "2026-10-20": {
        "label": "Diwali Muhurat Trading",
        "session_open_ist": "18:30",
        "session_close_ist": "19:30",
        "notes": "Auspicious 60-min session. Low volume by tradition.",
    },
}


def is_nse_holiday(d: date | datetime | str) -> Optional[str]:
    """Returns the holiday label if NSE is closed; None otherwise."""
    if isinstance(d, (datetime,)):
        d = d.date()
    if isinstance(d, date):
        key = d.isoformat()
    else:
        key = str(d)
    return NSE_HOLIDAYS_2026.get(key)


def is_weekend(d: date | datetime | str) -> bool:
    if isinstance(d, str):
        d = datetime.fromisoformat(d).date()
    if isinstance(d, datetime):
        d = d.date()
    return d.weekday() >= 5             # Sat=5, Sun=6


def is_nse_open(d: date | datetime | str) -> bool:
    return not is_weekend(d) and is_nse_holiday(d) is None


def next_trading_day(d: date | datetime | str) -> date:
    """Walk forward day-by-day until we find a non-holiday non-weekend."""
    if isinstance(d, str):
        cur = datetime.fromisoformat(d).date()
    elif isinstance(d, datetime):
        cur = d.date()
    else:
        cur = d
    cur = cur + timedelta(days=1)
    for _ in range(15):                  # max 15 days lookahead (Diwali week)
        if is_nse_open(cur):
            return cur
        cur = cur + timedelta(days=1)
    return cur


def session_context(d: Optional[date | datetime | str] = None) -> Dict[str, Any]:
    """Everything the cockpit's pre-market banner needs to know:
      * is today a trading day, or which is next
      * is today a Muhurat day
      * is today on or near an expiry day (Thursday or Wednesday-on-
        Thursday-holiday)
      * any nearby holiday so the operator plans size
    """
    if d is None:
        cur = datetime.now(IST).date()
    elif isinstance(d, str):
        cur = datetime.fromisoformat(d).date()
    elif isinstance(d, datetime):
        cur = d.date()
    else:
        cur = d
    today_holiday = is_nse_holiday(cur)
    today_weekend = is_weekend(cur)
    today_open = today_holiday is None and not today_weekend
    nxt = cur if today_open else next_trading_day(cur)
    muhurat = MUHURAT_SESSIONS_2026.get(cur.isoformat())
    # Is this an expiry day? Default Thursday weekly; if Thursday is a
    # holiday this week, expiry shifts to Wednesday.
    expiry_today = False
    if today_open:
        if cur.weekday() == 3:               # Thursday
            expiry_today = True
        elif cur.weekday() == 2:             # Wednesday — check if Thur is holiday
            thur = cur + timedelta(days=1)
            if is_nse_holiday(thur) is not None:
                expiry_today = True
    # Holiday in the next 7 days
    upcoming = []
    for i in range(1, 8):
        check = cur + timedelta(days=i)
        if (lbl := is_nse_holiday(check)):
            upcoming.append({"date": check.isoformat(), "label": lbl})
    return {
        "today_iso": cur.isoformat(),
        "today_open": today_open,
        "today_holiday": today_holiday,
        "today_weekend": today_weekend,
        "next_trading_day": nxt.isoformat(),
        "expiry_today": expiry_today,
        "muhurat_session": muhurat,
        "upcoming_holidays": upcoming,
    }


declare(IOSpec(
    module="sentinel.india_tax",
    purpose="Indian-market specific differentiation — tax-impact "
            "calculator (STT, GST, STCG, brokerage, SEBI, stamp duty) "
            "with citation-bearing rates from the Finance Act 2024; "
            "NSE F&O holiday + Muhurat calendar with expiry-day shift "
            "logic (Thursday→Wednesday when Thursday is a holiday). "
            "Surfaces no Western platform has",
    inputs=["buy_premium, sell_premium, quantity for tax impact",
            "date (or current IST) for session context"],
    outputs=["TaxBreakdown (itemised charges + net post-tax P&L)",
             "session_context dict (today open / next day / expiry "
             "today / Muhurat / upcoming holidays)"],
    consumes_from=[],
    produces_for=["sentinel.server (auditor + cockpit pre-market banner)"],
    tier="TRUSTED",
))
