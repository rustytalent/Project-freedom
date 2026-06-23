"""Multi-leg structures for executor v4.

Founder 2026-06-22 Tier-2 brief: "Today's market is dead chop. Iron
condor / jade lizard / butterfly are what we should be trading. The
StrategySelector library has them but they're not routed. Wire it."

This package provides the data shapes and constructors that let the
manager build and submit a structured option position (iron condor,
jade lizard, butterfly, strangle, vertical spread) atomically. The
broker adapter is extended with ``place_multi_leg_bundle`` which fans
out the leg orders and reconciles partial fills: if any leg can't be
filled, the legs that already filled are immediately reversed so we
never end up with an unintended directional exposure.

What lives where:

  * ``leg_spec``        — the LegSpec dataclass (one option-leg in a bundle)
  * ``bundle``          — the MultiLegBundle (a group of legs + metadata)
  * ``structures``      — constructors for iron_condor, jade_lizard,
                            butterfly, strangle, vertical_spread, taking
                            current spot + strike-grid + IV state as input
  * ``reconciliation``  — partial-fill rollback logic
  * ``bundle_ledger``   — tracks open bundles + combined P&L
"""
from .leg_spec import LegSpec, LEG_LONG, LEG_SHORT, OPT_CE, OPT_PE
from .bundle import (
    MultiLegBundle, BundleState,
    BUNDLE_PROPOSED, BUNDLE_OPEN, BUNDLE_CLOSING, BUNDLE_CLOSED,
    BUNDLE_REJECTED, BUNDLE_ROLLED_BACK,
)
from .structures import (
    StructureConstructor,
    IronCondor,
    JadeLizard,
    Butterfly,
    Strangle,
    VerticalSpread,
    STRUCTURE_IRON_CONDOR,
    STRUCTURE_JADE_LIZARD,
    STRUCTURE_BUTTERFLY,
    STRUCTURE_STRANGLE,
    STRUCTURE_VERTICAL_SPREAD,
)
from .reconciliation import (
    reconcile_partial_fills,
    BundleReconciliationOutcome,
)
from .bundle_ledger import BundleLedger, BundleLedgerEntry

__all__ = [
    "LegSpec", "LEG_LONG", "LEG_SHORT", "OPT_CE", "OPT_PE",
    "MultiLegBundle", "BundleState",
    "BUNDLE_PROPOSED", "BUNDLE_OPEN", "BUNDLE_CLOSING",
    "BUNDLE_CLOSED", "BUNDLE_REJECTED", "BUNDLE_ROLLED_BACK",
    "StructureConstructor", "IronCondor", "JadeLizard", "Butterfly",
    "Strangle", "VerticalSpread",
    "STRUCTURE_IRON_CONDOR", "STRUCTURE_JADE_LIZARD",
    "STRUCTURE_BUTTERFLY", "STRUCTURE_STRANGLE",
    "STRUCTURE_VERTICAL_SPREAD",
    "reconcile_partial_fills", "BundleReconciliationOutcome",
    "BundleLedger", "BundleLedgerEntry",
]
