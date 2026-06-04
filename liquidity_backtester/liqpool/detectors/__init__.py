"""Extended detector library — augments the original ``liqpool.features``
detectors with the manipulation/sweep patterns the audit + brain-dump
identified.

Each detector emits the same ``LevelCandidate`` shape as the original
detectors so it slots into ``detect_all`` and the existing pool
builder without any downstream changes.

Causality rule (NON-NEGOTIABLE): every detector emits ``known_at`` =
the close timestamp of the bar at which a real-time trader could first
observe the pattern. Detectors that need N future bars for
confirmation report ``known_at = ts(i + N) + period``, never earlier.
"""
from .sweep import liquidity_sweeps, stop_run_reclaims

__all__ = ["liquidity_sweeps", "stop_run_reclaims"]
