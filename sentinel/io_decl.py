"""Module IO declarations — the self-describing wiring contract.

The founder's instruction: every module declares its inputs and outputs
so we can mechanically build the connection map. This module holds the
registry; every backend module imports ``declare`` at import time to
register itself.

Address syntax (deliberately uniform):
    module.SYMBOL            -> a callable / class / dataclass exported
    file:<path>              -> a filesystem artifact (ledger, parquet, JSON)
    queue:<name>             -> an in-process channel (we don't have one
                                 in v1 but the syntax reserves room for it)
    signal:<source>:<kind>   -> a routed orchestration Signal

A module's declaration is the canonical answer to "where does my food
come from, and where does my excretion go." The map renderer reads
the registry and prints the digestion graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class IOSpec:
    module: str                                       # "sentinel.scientists"
    purpose: str                                      # one-line role
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    consumes_from: List[str] = field(default_factory=list)   # producer modules
    produces_for: List[str] = field(default_factory=list)    # consumer modules
    tier: str = "SHADOW"                              # trust tier of outputs


REGISTRY: Dict[str, IOSpec] = {}


def declare(spec: IOSpec) -> None:
    """Module-level call. Replaces any prior declaration so reloads
    during dev don't multiply rows."""
    REGISTRY[spec.module] = spec


def render_map() -> str:
    """Plain-text adjacency map of the organism. Used by the artifact
    generator and the operator's mental model."""
    lines: List[str] = [
        "SENTINEL CONNECTION MAP",
        "=" * 60,
        f"{len(REGISTRY)} modules registered",
        "",
    ]
    for name in sorted(REGISTRY):
        s = REGISTRY[name]
        lines.append(f"[{s.tier}] {s.module}")
        lines.append(f"    purpose : {s.purpose}")
        if s.inputs:
            for inp in s.inputs:
                lines.append(f"    in   <- {inp}")
        if s.outputs:
            for outp in s.outputs:
                lines.append(f"    out  -> {outp}")
        if s.consumes_from:
            lines.append(f"    eats   from: {', '.join(sorted(s.consumes_from))}")
        if s.produces_for:
            lines.append(f"    feeds    to: {', '.join(sorted(s.produces_for))}")
        lines.append("")
    return "\n".join(lines)


def render_dot() -> str:
    """Graphviz DOT form of the same graph, for when the operator wants
    an image. Edges are producer -> consumer."""
    edges: List[str] = []
    for s in REGISTRY.values():
        for consumer in s.produces_for:
            edges.append(f'  "{s.module}" -> "{consumer}";')
    nodes: List[str] = []
    for name in sorted(REGISTRY):
        nodes.append(f'  "{name}" [label="{name}\\n{REGISTRY[name].tier}"];')
    return "digraph Sentinel {\n  rankdir=LR;\n" + "\n".join(nodes) + "\n" + \
        "\n".join(edges) + "\n}\n"
