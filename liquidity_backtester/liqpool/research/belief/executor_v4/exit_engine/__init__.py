"""Adaptive Exit Quote Engine — the layer that turns losing trades less-losing.

Founder's mandate (2026-06-21): "If we can sell something one rupee
higher, we can. If we see conditions are not meeting up and our desired
exit will not be reached, we just reduce our selling target."

But Zerodha caps order modifications at 25 per order. So the engine
treats modifications like ammunition — model thinks every tick, broker
order moves only when expected value justifies spending one mod.

Architecture (the founder + ChatGPT alignment):

  Signal/Entry Thesis
        ↓
  TradeHypothesis        ── already shipped (hypothesis.py)
        ↓
  ExitThesis             ── this module (exit_engine/thesis.py)
        ↓
  Adaptive Exit Engine   ── per-position state + decision (engine.py)
        ├── LocalHighTracker (local_highs.py) — peak detection + revisit
        ├── ExitMode decider (modes.py) — 5 modes: HARVEST/SHADE/CHASE/DE_RISK/KILL
        ├── ModificationBudget (budget.py) — 4-bucket allocation
        └── ModificationGate (gate.py) — should we spend a mod?
        ↓
  Portfolio Exit Manager (portfolio_manager.py) — priority across positions
        ↓
  Broker Adapter         ── already shipped (broker/)
"""
from __future__ import annotations

from .budget import (
    ModificationBucket,
    ModificationBudget,
    ModificationBudgetConfig,
    ModificationRequest,
)
from .engine import (
    AdaptiveExitEngine,
    AdaptiveExitEngineConfig,
    ExitDecision,
    MarketState,
)
from .gate import (
    ModificationGate,
    ModificationGateConfig,
    ModificationGateDecision,
)
from .local_highs import (
    LocalExtreme,
    LocalHighTracker,
    LocalHighTrackerConfig,
    RevisitForecast,
)
from .modes import (
    EXIT_MODE_CHASE_FILL,
    EXIT_MODE_DE_RISK,
    EXIT_MODE_HARVEST,
    EXIT_MODE_KILL,
    EXIT_MODE_SHADE,
    ExitMode,
    ExitModeDecision,
    decide_exit_mode,
)
from .portfolio_manager import (
    PortfolioExitDecision,
    PortfolioExitManager,
    PortfolioExitManagerConfig,
)
from .thesis import (
    ExitThesis,
    ExitThesisConfig,
    build_exit_thesis,
)

__all__ = [
    "AdaptiveExitEngine",
    "AdaptiveExitEngineConfig",
    "EXIT_MODE_CHASE_FILL",
    "EXIT_MODE_DE_RISK",
    "EXIT_MODE_HARVEST",
    "EXIT_MODE_KILL",
    "EXIT_MODE_SHADE",
    "ExitDecision",
    "ExitMode",
    "ExitModeDecision",
    "ExitThesis",
    "ExitThesisConfig",
    "LocalExtreme",
    "LocalHighTracker",
    "LocalHighTrackerConfig",
    "MarketState",
    "ModificationBucket",
    "ModificationBudget",
    "ModificationBudgetConfig",
    "ModificationGate",
    "ModificationGateConfig",
    "ModificationGateDecision",
    "ModificationRequest",
    "PortfolioExitDecision",
    "PortfolioExitManager",
    "PortfolioExitManagerConfig",
    "RevisitForecast",
    "build_exit_thesis",
    "decide_exit_mode",
]
