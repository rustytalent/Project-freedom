"""Central tunable-parameter registry — the magic-number closure.

The detector/model audits found dozens of numeric thresholds scattered
through the codebase with no documented rationale and no sensitivity
validation. This registry closes that problem STRUCTURALLY:

  1. Every tunable lives here with: value, where it's used, why this
     value, validation status, and a sweep range for the sensitivity
     harness.
  2. ``analysis/run_param_sensitivity.py`` sweeps any registry entry
     against a bundle and reports the impact, so "unvalidated" entries
     can be promoted to "validated" with data instead of vibes.
  3. ``tests/test_tuning_registry.py`` pins each registry value to the
     actual default in the code, so the documentation can never drift
     from reality. Changing a default without updating the registry
     (or vice versa) fails CI.

Validation statuses:
  bootstrap    — chosen to get v1 working; no evidence either way
  unvalidated  — inherited/conventional; sensitivity sweep pending
  validated    — sweep run; value sits in a stable plateau
  data-derived — computed from data (not really a free parameter)

The registry deliberately does NOT replace the call-site defaults —
the engine must work without importing this module. The registry is
the documentation + validation layer; the test suite enforces
agreement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Tunable:
    name: str
    value: float
    used_in: str               # "module.symbol" or "module:approx-line"
    rationale: str
    status: str                # bootstrap | unvalidated | validated | data-derived
    sweep_range: Optional[Tuple[float, float]] = None
    sweep_steps: int = 5


# ---------------------------------------------------------------------------
# The registry. Grouped by subsystem. Every entry's `value` is pinned
# against the real code default by tests/test_tuning_registry.py.
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Tunable] = {t.name: t for t in [

    # -- detectors: sweep ------------------------------------------------
    Tunable(
        name="sweep_min_atr", value=0.20,
        used_in="liqpool/detectors/sweep.py (params fallback)",
        rationale=(
            "Pierce must exceed 0.20 ATR to count as a sweep rather than "
            "noise. Chosen at detector-batch v1 by eyeballing known sweep "
            "patterns on HDFCBANK 5m charts; never swept."
        ),
        status="unvalidated", sweep_range=(0.10, 0.40),
    ),
    Tunable(
        name="sweep_reclaim_window", value=12,
        used_in="liqpool/detectors/sweep.py (params fallback)",
        rationale=(
            "Price must reclaim the swept level within 12 bars (1 hour on "
            "5m) for the sweep+reclaim signature. Hour-scale chosen to "
            "match the intraday MIS horizon; never swept."
        ),
        status="unvalidated", sweep_range=(6, 24),
    ),
    Tunable(
        name="sweep_strength_cap", value=3.0,
        used_in="liqpool/detectors/sweep.py (params fallback)",
        rationale=(
            "Contributor strength cap so one violent sweep can't dominate "
            "pool scoring. 3.0 = 3x a baseline contributor. Arbitrary."
        ),
        status="bootstrap", sweep_range=(1.5, 5.0),
    ),
    Tunable(
        name="stop_run_close_atr", value=0.15,
        used_in="liqpool/detectors/sweep.py (params fallback)",
        rationale=(
            "Close-beyond-level minimum for the stop-run pattern. Smaller "
            "than sweep_min_atr because the stop-run definition also "
            "requires the reclaim leg. Never swept."
        ),
        status="unvalidated", sweep_range=(0.05, 0.30),
    ),

    # -- detectors: imbalance --------------------------------------------
    Tunable(
        name="imbalance_min_run", value=3,
        used_in="liqpool/detectors/imbalance.py (params fallback)",
        rationale="Minimum consecutive displacing bars. 3 bars = 15 min of "
                  "one-sided pressure on 5m. Conventional, never swept.",
        status="unvalidated", sweep_range=(2, 5),
    ),
    Tunable(
        name="imbalance_max_run", value=7,
        used_in="liqpool/detectors/imbalance.py (params fallback)",
        rationale="Run cap — beyond ~35 min the move is a trend leg, not an "
                  "imbalance event. Conventional, never swept.",
        status="unvalidated", sweep_range=(5, 12),
    ),
    Tunable(
        name="imbalance_min_atr", value=0.50,
        used_in="liqpool/detectors/imbalance.py (params fallback)",
        rationale="Cumulative displacement must exceed half an ATR. The "
                  "audit flagged this as the highest-impact unvalidated "
                  "threshold in the detector batch.",
        status="unvalidated", sweep_range=(0.25, 1.00),
    ),

    # -- detectors: volume -------------------------------------------------
    Tunable(
        name="vw_swing_window", value=30,
        used_in="liqpool/detectors/volume.py (params fallback)",
        rationale="Rolling window for mean volume baseline (2.5 hours on "
                  "5m). Never swept.",
        status="unvalidated", sweep_range=(15, 60),
    ),
    Tunable(
        name="vw_swing_multiplier", value=1.8,
        used_in="liqpool/detectors/volume.py (params fallback)",
        rationale=(
            "Swing volume must exceed 1.8x the rolling mean. The audit "
            "specifically asked 'why 1.8 and not 2.0?' — there is no "
            "documented answer. Sweep priority: high."
        ),
        status="unvalidated", sweep_range=(1.2, 3.0),
    ),
    Tunable(
        name="cum_delta_lookback", value=50,
        used_in="liqpool/detectors/volume.py (params fallback)",
        rationale="Prior-swing search window for divergence (bars). The "
                  "audit suggested ATR-relative lookback instead of bar "
                  "count; that's a structural change, deferred.",
        status="unvalidated", sweep_range=(25, 100),
    ),

    # -- ml_model: calibration ----------------------------------------------
    Tunable(
        name="bucket_shrinkage_prior_strength", value=20.0,
        used_in="liqpool/ml_model.py (n/(n+20) pull) + "
                "liqpool/calibration/empirical_bayes.py (fallback)",
        rationale=(
            "Legacy Wilson prior strength. SUPERSEDED by the Stream N "
            "beta-binomial empirical Bayes which fits this from the data; "
            "20.0 remains only as the fallback when <2 buckets exist. "
            "Adopting BetaBinomialBucketShrinker in ml_model.py retires "
            "this constant from the hot path."
        ),
        status="data-derived", sweep_range=None,
    ),
    Tunable(
        name="probability_clip_low", value=0.02,
        used_in="liqpool/ml_model.py predict paths (np.clip 0.02..0.98)",
        rationale=(
            "Log-loss singularity guard. The audit suggested replacing "
            "hard clips with temperature scaling (now available in Stream "
            "N); until adopted, 2% is the floor on any published "
            "probability."
        ),
        status="bootstrap", sweep_range=(0.005, 0.05),
    ),

    # -- options executor ------------------------------------------------
    Tunable(
        name="force_entry_threshold", value=0.30,
        used_in="liqpool/options/executor.py FORCE_ENTRY_THRESHOLD",
        rationale=(
            "Bootstrap |LCS| gate for ENTER (executor doc §3.5). "
            "Explicitly temporary: replaced by the learned force-entry "
            "classifier after >= 200 live trades. The M.4 regret model "
            "measures whether this is too tight in the meantime."
        ),
        status="bootstrap", sweep_range=(0.20, 0.45),
    ),
    Tunable(
        name="wait_threshold", value=0.10,
        used_in="liqpool/options/executor.py WAIT_THRESHOLD",
        rationale="Lower LCS bound of the WAIT band. Below it, SKIP. "
                  "Bootstrap pair to force_entry_threshold.",
        status="bootstrap", sweep_range=(0.05, 0.20),
    ),
    Tunable(
        name="macro_gate_threshold", value=0.10,
        used_in="liqpool/options/ccv.py MACRO_GATE_THRESHOLD",
        rationale=(
            "|macro| below this closes the entire cascade (no possibility "
            "space). The Stream L macro_gate_closed shadow events + "
            "resolver measure the realised cost of this caution — the "
            "first parameter that becomes data-validatable from shadow "
            "data alone."
        ),
        status="bootstrap", sweep_range=(0.05, 0.20),
    ),
    Tunable(
        name="vix_kill_spike", value=3.0,
        used_in="liqpool/options/executor.py VIX_KILL_SPIKE",
        rationale="Intra-bar VIX jump (points) that kills any open trade. "
                  "Conservative round number; rare enough that validation "
                  "needs years of data. Accepted as a safety rail.",
        status="bootstrap", sweep_range=None,
    ),

    # -- execution v3 -----------------------------------------------------
    Tunable(
        name="impact_coefficient_bps", value=1.5,
        used_in="liqpool/execution_simulator_v3.py ExecutionV3Config",
        rationale=(
            "Sqrt-impact coefficient. 1.5 bps per sqrt(notional/depth) is "
            "the conventional Almgren-Chriss ballpark for liquid equities; "
            "needs calibration against real fills before being trusted. "
            "Impact disabled below the 200k threshold regardless."
        ),
        status="unvalidated", sweep_range=(0.5, 4.0),
    ),
    Tunable(
        name="arsenal_min_target_to_cost_ratio", value=3.0,
        used_in="liqpool/arsenal/evaluator.py EvaluatorConfig",
        rationale=(
            "Best-case target reward must clear 3x round-trip cost. "
            "Encodes the founder's sizing instinct from the HDFC finding. "
            "The below_target_to_cost shadow events measure its regret "
            "rate directly."
        ),
        status="unvalidated", sweep_range=(1.5, 5.0),
    ),

    # -- triple barrier ----------------------------------------------------
    Tunable(
        name="min_risk_fraction_of_atr", value=0.10,
        used_in="liqpool/execution_backtest.py MIN_RISK_FRACTION_OF_ATR",
        rationale="Floor on risk-per-share as a fraction of ATR so "
                  "degenerate near-zero-risk entries are rejected (V1) "
                  "rather than producing fantasy infinite-R trades.",
        status="bootstrap", sweep_range=(0.05, 0.25),
    ),
]}


def get(name: str) -> Tunable:
    return REGISTRY[name]


def by_status(status: str) -> List[Tunable]:
    return [t for t in REGISTRY.values() if t.status == status]


def sweepable() -> List[Tunable]:
    """Entries the sensitivity harness can sweep."""
    return [t for t in REGISTRY.values() if t.sweep_range is not None]


def summary_frame():
    """Registry as a DataFrame for reports / the audit."""
    import pandas as pd
    return pd.DataFrame([{
        "name": t.name, "value": t.value, "status": t.status,
        "used_in": t.used_in,
        "sweep_lo": t.sweep_range[0] if t.sweep_range else None,
        "sweep_hi": t.sweep_range[1] if t.sweep_range else None,
    } for t in REGISTRY.values()])
