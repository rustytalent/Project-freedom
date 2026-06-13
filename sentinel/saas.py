"""SaaS tier gating — the founder's "nerf strategy" mechanized.

Every institutional / customer-facing function carries a tier tag
(RETAIL, PRO, QUANT). This module turns those tags into runtime gates so
the server can return 402 Payment Required for features the caller's
plan doesn't include. Without this, the SaaS plans are policy only —
the code wouldn't enforce them.

Three customer plans (the founder's pricing):

  RETAIL   ₹  — payoff curve, basic Greeks, vol risk premium, historical
                VaR, one strategy build per session
  PRO      ₹₹ — + SVI fair value, vol cone, Parkinson/GK RV, Cornish-
                Fisher VaR, Expected Shortfall, Crux scores, stress
                tests, unlimited strategy builds, equity contextual layer
  QUANT    ₹₹₹ — + Yang-Zhang RV, raw SVI parameters, full chain export,
                live ledger feed, API access

Plus FOUNDER which gates nothing (the user's own untouchable view).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Set

from .io_decl import IOSpec, declare


# Tier names — mirror sentinel.institutional's RETAIL/PRO/QUANT
RETAIL = "RETAIL"
PRO = "PRO"
QUANT = "QUANT"
FOUNDER = "FOUNDER"

# Plan tier strength (higher = more powerful)
_TIER_LEVEL: Dict[str, int] = {
    RETAIL: 1,
    PRO: 2,
    QUANT: 3,
    FOUNDER: 99,
}

# Feature catalog — every paid surface this layer mentions exists in code
# above; the catalog is the contract between code and pricing page.
FEATURE_CATALOG: Dict[str, str] = {
    # Institutional (sentinel.institutional)
    "rv.close_to_close":      RETAIL,
    "rv.parkinson":           PRO,
    "rv.garman_klass":        PRO,
    "rv.yang_zhang":          QUANT,
    "vol_cone":               PRO,
    "vol_cone_percentile":    RETAIL,
    "vol_risk_premium":       RETAIL,
    "svi.fit":                QUANT,
    "svi.fair_value":         PRO,
    "svi.skew_25d":           PRO,
    "var.historical":         RETAIL,
    "var.cornish_fisher":     PRO,
    "expected_shortfall":     PRO,
    "greek_exposures":        RETAIL,
    "crux.liquidity":         PRO,
    "crux.slippage":          PRO,
    "roll_curve":             PRO,
    # Stress (sentinel.stress)
    "stress.single_scenario": PRO,
    "stress.full_matrix":     PRO,
    # Auditor (sentinel.auditor)
    "auditor.payoff_curve":   RETAIL,
    "auditor.named_detect":   RETAIL,
    "auditor.greek_book":     PRO,
    "auditor.risk_flags":     PRO,
    # Strategy builder (sentinel.strategy_builder)
    "builder.one_per_session": RETAIL,
    "builder.unlimited":       PRO,
    # Equity contextual layer (sentinel.equity_layer)
    "equity.divergence_scalar": RETAIL,
    "equity.full_context":      PRO,
    # Live ledger feed (export)
    "ledger.full_export":       QUANT,
    "api.programmatic":         QUANT,
}


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    feature: str
    required_tier: str
    your_tier: str
    reason: str


def required_tier(feature: str) -> str:
    """Tier needed to access ``feature``. Unknown features default to PRO
    (safer than RETAIL — a missing tag should not silently expose code)."""
    return FEATURE_CATALOG.get(feature, PRO)


def gate(plan: str, feature: str) -> GateResult:
    """The single check call. Returns whether ``plan`` can access
    ``feature`` and why. FOUNDER bypasses every gate."""
    plan = plan.upper()
    if plan == FOUNDER:
        return GateResult(allowed=True, feature=feature,
                          required_tier="*", your_tier=FOUNDER,
                          reason="founder bypass")
    needed = required_tier(feature)
    if plan not in _TIER_LEVEL:
        return GateResult(allowed=False, feature=feature,
                          required_tier=needed, your_tier=plan,
                          reason=f"unknown plan: {plan}")
    if _TIER_LEVEL[plan] >= _TIER_LEVEL[needed]:
        return GateResult(allowed=True, feature=feature,
                          required_tier=needed, your_tier=plan,
                          reason="plan covers required tier")
    return GateResult(allowed=False, feature=feature,
                      required_tier=needed, your_tier=plan,
                      reason=f"upgrade to {needed} or higher")


def features_for(plan: str) -> Set[str]:
    """Every feature ``plan`` is entitled to."""
    return {f for f in FEATURE_CATALOG if gate(plan, f).allowed}


def plan_summary(plan: str) -> Dict[str, object]:
    """Short summary suitable for the customer-facing /me/plan endpoint."""
    feats = sorted(features_for(plan))
    return {
        "plan": plan.upper(),
        "level": _TIER_LEVEL.get(plan.upper(), 0),
        "feature_count": len(feats),
        "features": feats,
    }


declare(IOSpec(
    module="sentinel.saas",
    purpose="SaaS tier gating — turns RETAIL/PRO/QUANT/FOUNDER plan tags "
            "into runtime gates so server endpoints can return 402 for "
            "features outside the caller's plan",
    inputs=["plan name (RETAIL|PRO|QUANT|FOUNDER)", "feature key"],
    outputs=["GateResult (allowed bool + reason + required/your tier)",
             "features_for(plan): full entitlement set"],
    consumes_from=[],
    produces_for=["sentinel.server (entitlement check on every paid endpoint)"],
    tier="TRUSTED",
))
