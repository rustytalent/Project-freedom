"""Research-context bridge — load the liquidity_backtester's latest
artifacts and convert them into ``MarketSnapshot.context`` fields the
Sentinel scientists can read.

The flaw Codex flagged (Problem S3 in the unified report): scientists
have fields for proximity / direction-probability / reaction-probability
/ key-zones / sector-regime / model-health, but no one fills them.
Without that, scientists generate generic option hypotheses instead of
*context-aware* ones.

This module bridges the gap, without taking a hard dependency on the
research codebase (which lives at /home/user/Project-freedom/
liquidity_backtester). If the artifacts aren't on disk, we return an
empty pack so the scientists fall back gracefully to chart-only
reasoning. Tier: TRUSTED. Sentinel SCORE_TIER: PRO.

The contract is intentionally narrow — the bridge speaks one struct
(``ResearchContextPack``). When the research-side schema evolves,
update the loader; do NOT scatter raw research-payload reads across
Sentinel.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .io_decl import IOSpec, declare


# Where the research repo writes its daily artifacts. Override per env.
DEFAULT_RESEARCH_ROOT = Path("/home/user/Project-freedom/liquidity_backtester/runs")


@dataclass
class ZoneLevel:
    label: str                      # human label, e.g. "PDH" / "VAL"
    price: float
    side: str                       # "support" | "resistance" | "magnet"
    horizon_minutes: int             # over what horizon it's expected to matter
    reach_probability: float = 0.0  # from the proximity model


@dataclass
class ResearchContextPack:
    """The slim handshake struct between the research brain and
    Sentinel. Every field is optional so a partial pack still helps."""
    session_date: str = ""
    source_run_id: str = ""
    direction_probability: Optional[float] = None      # P(NIFTY closes up)
    reaction_probability: Optional[float] = None
    proximity_h12: Optional[float] = None              # P(reach key zone within 12 bars)
    proximity_h36: Optional[float] = None
    proximity_h60: Optional[float] = None
    sector_regime: Optional[str] = None                 # BANK_HOT / IT_HOT / SMALLCAP_RIP / RISK_OFF
    avoid_flags: List[str] = field(default_factory=list)
    model_health: Dict[str, Any] = field(default_factory=dict)
    key_zones: List[ZoneLevel] = field(default_factory=list)
    notes: str = ""

    @property
    def is_empty(self) -> bool:
        return (self.direction_probability is None
                and self.reaction_probability is None
                and not self.key_zones)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_date": self.session_date,
            "source_run_id": self.source_run_id,
            "direction_probability": self.direction_probability,
            "reaction_probability": self.reaction_probability,
            "proximity_h12": self.proximity_h12,
            "proximity_h36": self.proximity_h36,
            "proximity_h60": self.proximity_h60,
            "sector_regime": self.sector_regime,
            "avoid_flags": list(self.avoid_flags),
            "model_health": dict(self.model_health),
            "key_zones": [vars(z) for z in self.key_zones],
            "notes": self.notes,
        }


def load_research_pack(session_date: Optional[str] = None,
                        root: Optional[Path] = None
                        ) -> ResearchContextPack:
    """Locate the latest daily brief / model artifacts under ``root`` (or
    DEFAULT_RESEARCH_ROOT) and return a ResearchContextPack.

    Best-effort: if no artifacts exist, returns an empty pack — the
    caller MUST handle the empty case (scientists fall back to
    chart-only reasoning). This is intentional so Sentinel runs even
    when the research repo hasn't produced today's brief yet.

    Layout the loader knows how to read (any one is enough):

      <root>/<session>/daily_brief.json
      <root>/<session>/research_context.json
      <root>/latest/daily_brief.json
    """
    root = Path(root or DEFAULT_RESEARCH_ROOT)
    session = session_date or datetime.utcnow().strftime("%Y-%m-%d")
    candidates = [
        root / session / "research_context.json",
        root / session / "daily_brief.json",
        root / "latest" / "research_context.json",
        root / "latest" / "daily_brief.json",
    ]
    payload: Optional[Dict[str, Any]] = None
    for c in candidates:
        if c.exists():
            try:
                payload = json.loads(c.read_text())
                break
            except Exception:
                continue
    if not payload:
        return ResearchContextPack(session_date=session)
    return _payload_to_pack(payload, session)


def _payload_to_pack(payload: Mapping[str, Any], session: str) -> ResearchContextPack:
    """Best-effort coercion of various research-side schemas into our pack.

    Supported keys (any of these recognised — the research repo can
    evolve and we adapt, not the other way around):
      * ``direction_probability`` / ``direction.up``
      * ``reaction_probability`` / ``reaction.fire``
      * ``proximity.h12`` / ``proximity_h12``
      * ``sector_regime`` / ``regime`` / ``sector.label``
      * ``avoid`` (list)
      * ``model_health`` (dict)
      * ``zones`` / ``key_zones`` (list of dicts)
    """
    dirp = (payload.get("direction_probability")
            or (payload.get("direction") or {}).get("up"))
    react = (payload.get("reaction_probability")
             or (payload.get("reaction") or {}).get("fire"))
    prox = payload.get("proximity") or {}
    sector = (payload.get("sector_regime") or payload.get("regime")
              or (payload.get("sector") or {}).get("label"))
    avoid = list(payload.get("avoid") or payload.get("avoid_flags") or [])
    zones_in = payload.get("zones") or payload.get("key_zones") or []
    zones: List[ZoneLevel] = []
    for z in zones_in:
        try:
            zones.append(ZoneLevel(
                label=str(z.get("label", "?")),
                price=float(z.get("price") or z.get("level") or 0),
                side=str(z.get("side", "magnet")),
                horizon_minutes=int(z.get("horizon_minutes")
                                     or z.get("horizon", 60)),
                reach_probability=float(z.get("reach_probability")
                                          or z.get("p_reach") or 0.0),
            ))
        except Exception:
            continue
    return ResearchContextPack(
        session_date=session,
        source_run_id=str(payload.get("run_id") or payload.get("source_run_id") or ""),
        direction_probability=_maybe_float(dirp),
        reaction_probability=_maybe_float(react),
        proximity_h12=_maybe_float(prox.get("h12") or payload.get("proximity_h12")),
        proximity_h36=_maybe_float(prox.get("h36") or payload.get("proximity_h36")),
        proximity_h60=_maybe_float(prox.get("h60") or payload.get("proximity_h60")),
        sector_regime=str(sector) if sector else None,
        avoid_flags=avoid,
        model_health=dict(payload.get("model_health") or {}),
        key_zones=zones,
        notes=str(payload.get("notes") or ""),
    )


def _maybe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def merge_into_snapshot_context(pack: ResearchContextPack,
                                  snapshot_context: Dict[str, Any]
                                  ) -> Dict[str, Any]:
    """Mutates and returns ``snapshot_context`` populated from the pack.
    The contract: this is the ONE place research fields land in the
    scientist-visible context. Sentinel scientists should never read
    research artifacts directly."""
    if pack.direction_probability is not None:
        snapshot_context["model_signal"] = pack.direction_probability
    if pack.reaction_probability is not None:
        snapshot_context["reaction_model"] = pack.reaction_probability
    if pack.sector_regime:
        snapshot_context["sector_regime"] = pack.sector_regime
    if pack.avoid_flags:
        snapshot_context["avoid_flags"] = list(pack.avoid_flags)
    if pack.key_zones:
        snapshot_context["key_zones"] = [vars(z) for z in pack.key_zones]
    if pack.proximity_h12 is not None:
        snapshot_context["proximity_h12"] = pack.proximity_h12
    snapshot_context["research_pack_id"] = pack.source_run_id
    return snapshot_context


declare(IOSpec(
    module="sentinel.liqpool_bridge",
    purpose="loader for liquidity_backtester ResearchContextPack — "
            "direction/reaction probabilities, proximity, sector regime, "
            "avoid flags, key zones — merged into MarketSnapshot.context so "
            "scientists think with the real research brain",
    inputs=["file:<research_root>/<session>/research_context.json",
            "file:<research_root>/<session>/daily_brief.json",
            "file:<research_root>/latest/*"],
    outputs=["ResearchContextPack",
             "merge_into_snapshot_context populates scientist-visible fields"],
    consumes_from=["liquidity_backtester (external)"],
    produces_for=["sentinel.scientists (via MarketSnapshot.context)"],
    tier="TRUSTED",
))
