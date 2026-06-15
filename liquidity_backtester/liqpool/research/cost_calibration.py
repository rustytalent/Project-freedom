"""Cost calibration — reverse-engineer the REAL Zerodha charges from
trade confirmations and fit our cost model to them.

The Profitability Doctrine (and the Creative Playbook §B) flagged
that the default `CostsModel` is GUESSED — STT rate from the Finance
Act, slippage assumed flat at 2 ticks, brokerage assumed flat ₹20,
GST applied to a base that "should" be brokerage + exchange + SEBI.

Reality: Zerodha tweaks fees, slippage varies wildly by liquidity
bucket, and edge cases (deep ITM exercise) blow up the STT model.

This module lets us replace assumptions with measurements:

  1. Operator dumps their real trade history (Kite's "Charges"
     breakdown is downloadable as CSV or accessible via the contract
     note PDFs Zerodha emails).
  2. ``fit_from_confirmations(rows)`` runs a constrained regression
     over the components (each one independently checked) and returns
     a ``CalibratedCostsModel``.
  3. The harness re-runs every hypothesis with the calibrated model.
     Hypotheses that survived assumed costs but die under real costs
     get killed honestly.

Two outputs:
  * The fitted CostsModel (drop-in replacement).
  * A CalibrationReport that says "your STT coefficient matched the
    Finance Act exactly" or "your brokerage was 60% lower than
    assumed, you're on a discount tier."

The discipline this enables: never run a backtest on assumptions
when measurements are available.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from liqpool.research.harness import CostsModel

LOG = logging.getLogger("liqpool.research.cost_calibration")


@dataclass
class TradeConfirmation:
    """One round-trip trade with the actual charges Zerodha applied.

    Fields with ``actual_*`` are READ from the contract note PDF or
    the Console CSV — they are GROUND TRUTH that the model should
    match. The rest are inputs.
    """
    # Trade inputs
    buy_premium: float
    sell_premium: float
    qty: int
    instrument_kind: str = "options"          # options | futures | equity
    buy_ts_utc: Optional[str] = None
    sell_ts_utc: Optional[str] = None

    # Charges as Zerodha actually billed (paise-precision in source PDF).
    # Any field None means "we don't have ground truth for this leg's
    # component" — the fitter ignores nones.
    actual_brokerage_rupees: Optional[float] = None     # both legs combined
    actual_stt_rupees: Optional[float] = None
    actual_exchange_txn_rupees: Optional[float] = None
    actual_sebi_charge_rupees: Optional[float] = None
    actual_stamp_duty_rupees: Optional[float] = None
    actual_gst_rupees: Optional[float] = None
    actual_total_rupees: Optional[float] = None         # bottom line

    note: str = ""

    def buy_turnover(self) -> float:
        return self.buy_premium * self.qty

    def sell_turnover(self) -> float:
        return self.sell_premium * self.qty

    def total_turnover(self) -> float:
        return self.buy_turnover() + self.sell_turnover()


@dataclass
class CalibrationFinding:
    """One per cost-component, plus an overall finding."""
    component: str                       # brokerage / stt / exchange / sebi / stamp / gst / total
    assumed: float                       # from the default CostsModel
    measured: float                      # from the confirmations
    relative_error_pct: float            # (measured - assumed) / assumed * 100
    n_samples: int
    note: str = ""

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationReport:
    n_confirmations: int
    findings: List[CalibrationFinding] = field(default_factory=list)
    calibrated_model: Optional[CostsModel] = None
    warnings: List[str] = field(default_factory=list)
    overall_error_pct: float = 0.0
    overall_decision: str = "evaluate"  # "use_calibrated" | "model_already_accurate" | "needs_more_data"

    # Decision thresholds
    MIN_CONFIRMATIONS_TO_TRUST = 10
    USE_CALIBRATED_IF_ERROR_OVER = 5.0     # %
    MODEL_OK_IF_ERROR_UNDER = 2.0           # %

    def to_row(self) -> Dict[str, Any]:
        return {
            "n_confirmations": self.n_confirmations,
            "findings": [f.to_row() for f in self.findings],
            "calibrated_model": (asdict(self.calibrated_model)
                                  if self.calibrated_model else None),
            "warnings": self.warnings,
            "overall_error_pct": self.overall_error_pct,
            "overall_decision": self.overall_decision,
        }

    def auto_decide(self) -> None:
        if self.n_confirmations < self.MIN_CONFIRMATIONS_TO_TRUST:
            self.overall_decision = "needs_more_data"
            return
        if abs(self.overall_error_pct) > self.USE_CALIBRATED_IF_ERROR_OVER:
            self.overall_decision = "use_calibrated"
        elif abs(self.overall_error_pct) < self.MODEL_OK_IF_ERROR_UNDER:
            self.overall_decision = "model_already_accurate"
        else:
            self.overall_decision = "evaluate"


# ─────────────────────────────────────────────────────────────────
# Per-component calibrators
# ─────────────────────────────────────────────────────────────────

def _fit_proportional(samples: Sequence[float], bases: Sequence[float]
                       ) -> Optional[float]:
    """Solve charge = rate × base for the rate, via least squares.
    Returns the rate (e.g. 0.000625 for STT). Empty / degenerate
    input returns None."""
    if not samples or not bases or len(samples) != len(bases):
        return None
    paired = [(s, b) for s, b in zip(samples, bases)
              if b is not None and s is not None and b > 0]
    if len(paired) < 3:
        return None
    # Closed-form least squares for y = β·x, β = Σ(xy)/Σ(x²)
    sx2 = sum(b * b for _, b in paired)
    sxy = sum(s * b for s, b in paired)
    return sxy / sx2 if sx2 > 0 else None


def _fit_constant_per_leg(samples: Sequence[float],
                           n_legs_per_trade: int = 2) -> Optional[float]:
    """Brokerage is flat per leg. ``samples`` is the brokerage on each
    round-trip; divide by 2 and take the median for robustness."""
    if not samples:
        return None
    per_leg = [s / n_legs_per_trade for s in samples if s is not None]
    if not per_leg:
        return None
    return float(statistics.median(per_leg))


def _fit_stt_options_sell_pct(rows: Sequence[TradeConfirmation]
                                ) -> Optional[float]:
    samples, bases = [], []
    for r in rows:
        if r.actual_stt_rupees is None or r.instrument_kind != "options":
            continue
        samples.append(r.actual_stt_rupees)
        bases.append(r.sell_turnover())
    return _fit_proportional(samples, bases)


def _fit_exchange_pct(rows: Sequence[TradeConfirmation]
                       ) -> Optional[float]:
    samples, bases = [], []
    for r in rows:
        if r.actual_exchange_txn_rupees is None:
            continue
        samples.append(r.actual_exchange_txn_rupees)
        bases.append(r.total_turnover())
    return _fit_proportional(samples, bases)


def _fit_sebi_pct(rows: Sequence[TradeConfirmation]
                   ) -> Optional[float]:
    samples, bases = [], []
    for r in rows:
        if r.actual_sebi_charge_rupees is None:
            continue
        samples.append(r.actual_sebi_charge_rupees)
        bases.append(r.total_turnover())
    return _fit_proportional(samples, bases)


def _fit_stamp_pct(rows: Sequence[TradeConfirmation]
                    ) -> Optional[float]:
    samples, bases = [], []
    for r in rows:
        if r.actual_stamp_duty_rupees is None:
            continue
        samples.append(r.actual_stamp_duty_rupees)
        bases.append(r.buy_turnover())
    return _fit_proportional(samples, bases)


def _fit_gst_pct(rows: Sequence[TradeConfirmation],
                  brokerage_per_leg: float,
                  exchange_pct: float, sebi_pct: float) -> Optional[float]:
    """GST = 18% of (brokerage + exchange + SEBI). Fit against measured
    GST to verify the base is right."""
    samples, bases = [], []
    for r in rows:
        if r.actual_gst_rupees is None:
            continue
        expected_base = (brokerage_per_leg * 2
                         + r.total_turnover() * exchange_pct
                         + r.total_turnover() * sebi_pct)
        if expected_base <= 0:
            continue
        samples.append(r.actual_gst_rupees)
        bases.append(expected_base)
    return _fit_proportional(samples, bases)


# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

def fit_from_confirmations(
    confirmations: Sequence[TradeConfirmation],
    baseline: Optional[CostsModel] = None,
) -> CalibrationReport:
    """The single call that turns N real trade confirmations into a
    fitted ``CostsModel`` + a report on what changed.

    Logic:
      * For each component, fit the rate (or per-leg constant).
      * If we have ≥ 3 samples for that component, take the fit.
      * Otherwise, fall back to the baseline value.
      * Compare each fitted value to the baseline assumption and record
        the relative error.
      * Overall error: median |relative_error_pct| across components.
    """
    base = baseline or CostsModel()
    if not confirmations:
        return CalibrationReport(n_confirmations=0,
                                  warnings=["no confirmations provided"],
                                  calibrated_model=base)
    report = CalibrationReport(n_confirmations=len(confirmations),
                                 calibrated_model=base)

    # Brokerage — median of (actual_brokerage / 2)
    brokerage_samples = [c.actual_brokerage_rupees for c in confirmations
                         if c.actual_brokerage_rupees is not None]
    brokerage = _fit_constant_per_leg(brokerage_samples)
    if brokerage is None:
        brokerage = base.brokerage_per_leg_rupees
        report.warnings.append("brokerage: insufficient samples, using baseline")
    report.findings.append(_finding(
        "brokerage_per_leg_rupees", base.brokerage_per_leg_rupees,
        brokerage, len(brokerage_samples)))

    # STT options sell
    stt = _fit_stt_options_sell_pct(confirmations)
    if stt is None:
        stt = base.stt_options_sell_pct
        report.warnings.append("stt: insufficient samples, using baseline")
    report.findings.append(_finding(
        "stt_options_sell_pct", base.stt_options_sell_pct, stt,
        sum(1 for c in confirmations if c.actual_stt_rupees is not None)))

    # Exchange turnover
    exch = _fit_exchange_pct(confirmations)
    if exch is None:
        exch = base.exchange_txn_pct
        report.warnings.append("exchange_txn: insufficient samples, using baseline")
    report.findings.append(_finding(
        "exchange_txn_pct", base.exchange_txn_pct, exch,
        sum(1 for c in confirmations if c.actual_exchange_txn_rupees is not None)))

    # SEBI
    sebi = _fit_sebi_pct(confirmations)
    if sebi is None:
        sebi = base.sebi_charge_pct
        report.warnings.append("sebi: insufficient samples, using baseline")
    report.findings.append(_finding(
        "sebi_charge_pct", base.sebi_charge_pct, sebi,
        sum(1 for c in confirmations if c.actual_sebi_charge_rupees is not None)))

    # Stamp duty (buy-side)
    stamp = _fit_stamp_pct(confirmations)
    if stamp is None:
        stamp = base.stamp_duty_buy_pct
        report.warnings.append("stamp_duty: insufficient samples, using baseline")
    report.findings.append(_finding(
        "stamp_duty_buy_pct", base.stamp_duty_buy_pct, stamp,
        sum(1 for c in confirmations if c.actual_stamp_duty_rupees is not None)))

    # GST — fit against the assembled expected base
    gst = _fit_gst_pct(confirmations, brokerage, exch, sebi)
    if gst is None:
        gst = base.gst_pct
        report.warnings.append("gst: insufficient samples, using baseline")
    report.findings.append(_finding(
        "gst_pct", base.gst_pct, gst,
        sum(1 for c in confirmations if c.actual_gst_rupees is not None)))

    # Build the calibrated model
    calibrated = CostsModel(
        brokerage_per_leg_rupees=round(brokerage, 4),
        stt_options_sell_pct=round(stt, 8),
        exchange_txn_pct=round(exch, 8),
        sebi_charge_pct=round(sebi, 10),
        stamp_duty_buy_pct=round(stamp, 8),
        gst_pct=round(gst, 5),
        slippage_ticks_per_leg=base.slippage_ticks_per_leg,
        tick_size_rupees=base.tick_size_rupees,
    )
    report.calibrated_model = calibrated

    # Overall error = median |rel_err| across findings with samples
    errs = [abs(f.relative_error_pct) for f in report.findings
            if f.n_samples > 0]
    report.overall_error_pct = round(statistics.median(errs), 2) if errs else 0
    report.auto_decide()
    return report


def _finding(name: str, assumed: float, measured: float,
             n: int) -> CalibrationFinding:
    rel = 0.0 if assumed == 0 else (measured - assumed) / assumed * 100
    note = ""
    if n < 3:
        note = f"only {n} samples — fit unreliable"
    elif abs(rel) > 25:
        note = f"large divergence ({rel:.1f}%) — verify input data"
    return CalibrationFinding(
        component=name, assumed=round(assumed, 8),
        measured=round(measured, 8),
        relative_error_pct=round(rel, 2),
        n_samples=n, note=note)


# ─────────────────────────────────────────────────────────────────
# Stress test — does a hypothesis survive cost-model uncertainty?
# ─────────────────────────────────────────────────────────────────

@dataclass
class StressResult:
    """Result of running a hypothesis under adversarial cost scenarios."""
    scenarios: Dict[str, Any] = field(default_factory=dict)
    decision: str = "evaluate"          # robust | sensitive | fragile

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


def stress_test_costs(harness_runner, spec, bars,
                       baseline: Optional[CostsModel] = None,
                       slippage_ticks_to_test: Sequence[int] = (1, 2, 3, 5),
                       ) -> StressResult:
    """Run the same hypothesis with progressively pessimistic cost
    assumptions. A hypothesis is ROBUST if it survives 3-tick
    slippage; SENSITIVE if it dies between 2 and 3; FRAGILE if it
    dies at 1.

    ``harness_runner`` is a callable ``(spec, bars, costs_model)
    -> HypothesisReport`` so the function works with any harness
    wrapping (regular, walk-forward, etc.).
    """
    base = baseline or CostsModel()
    sr = StressResult()
    survival = {}
    for slip in slippage_ticks_to_test:
        cm = CostsModel(
            brokerage_per_leg_rupees=base.brokerage_per_leg_rupees,
            stt_options_sell_pct=base.stt_options_sell_pct,
            exchange_txn_pct=base.exchange_txn_pct,
            sebi_charge_pct=base.sebi_charge_pct,
            stamp_duty_buy_pct=base.stamp_duty_buy_pct,
            gst_pct=base.gst_pct,
            slippage_ticks_per_leg=int(slip),
            tick_size_rupees=base.tick_size_rupees,
        )
        try:
            report = harness_runner(spec, bars, cm)
            survival[f"slip_{slip}t"] = {
                "sharpe": report.overall.sharpe,
                "net_pnl": report.overall.net_pnl,
                "decision": report.decision,
                "n_trades": report.overall.n_trades,
            }
        except Exception as exc:
            survival[f"slip_{slip}t"] = {"error": str(exc)}
    sr.scenarios = survival
    # Decision
    kept_at = []
    for slip in slippage_ticks_to_test:
        key = f"slip_{slip}t"
        decision = (survival.get(key) or {}).get("decision")
        if decision == "keep":
            kept_at.append(slip)
    if 3 in kept_at or 5 in kept_at:
        sr.decision = "robust"
    elif 2 in kept_at:
        sr.decision = "sensitive"
    elif 1 in kept_at:
        sr.decision = "fragile"
    else:
        sr.decision = "dies"
    return sr
