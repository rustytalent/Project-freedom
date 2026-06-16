"""Decay-aware kill-switch per hypothesis (idea #28).

A live hypothesis is not the same hypothesis it was in backtest. Markets
adapt, retail flow changes, the edge that survived a year of paper data
gets arb'd out. Without an automatic decay detector you keep paying the
cost-per-trade on a dead strategy until the bankroll bleeds out.

This module ships the discipline. Per hypothesis it tracks:

  * the *original* in-sample Sharpe (what we trusted at deploy time), and
  * the *rolling-window* Sharpe over the most recent N trades.

The verdict is one of:

  * ``ALIVE`` — rolling Sharpe is healthy AND not materially below
    original. Keep allocating.
  * ``WATCHING`` — rolling Sharpe has dropped, but not below the absolute
    floor; deploy at reduced size, do not kill yet.
  * ``KILLED`` — rolling Sharpe collapsed below the floor OR is a small
    fraction of the original Sharpe. Stop allocating. Resurrect requires
    a clean recovery window of fresh trades (counted post-kill so the
    decision isn't reversed by the same window that killed it).

What this module is NOT:
  * A regime classifier. It does not know *why* a strategy decayed.
    The kill verdict says "the data says this isn't working any more";
    the operator decides whether to re-enable manually.
  * A bet sizer. Kill / alive is a binary input to the allocator (idea
    #26); sizing is downstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Below this many recent trades, the rolling Sharpe is too noisy to act on.
MIN_TRADES_FOR_VERDICT: int = 20

# Rolling Sharpe below this is the "this isn't working" floor irrespective
# of what the original was. A strategy whose recent edge is < 0.30 Sharpe
# can't pay for its costs at any sensible bet size.
DEFAULT_ABSOLUTE_SHARPE_FLOOR: float = 0.30

# Rolling Sharpe / original Sharpe ratio below this is the "the edge
# decayed past usefulness" floor. 0.40 = 60% of the original edge has
# vanished. Above 0.70 we treat as healthy; in between is WATCHING.
DEFAULT_DECAY_KILL_RATIO: float = 0.40
DEFAULT_DECAY_WATCH_RATIO: float = 0.70

# Number of fresh trades required after a kill to even consider
# resurrection. Same window size we use for the kill check, so a
# clean window is statistically comparable.
DEFAULT_RESURRECT_TRADES: int = 30


@dataclass
class HypothesisHistory:
    """Per-hypothesis trade record the kill-switch consumes.

    The ``r_multiples`` array is the only numeric the verdict needs;
    ``timestamps`` is preserved so callers can window by recent days
    instead of by recent trade count when that's a better fit.
    """
    name: str
    original_sharpe: float
    r_multiples: np.ndarray = field(default_factory=lambda: np.empty(0))
    timestamps: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        arr = np.asarray(self.r_multiples, dtype=float)
        mask = np.isfinite(arr)
        if self.timestamps and len(self.timestamps) == len(arr):
            # Filter timestamps alongside r_multiples so the two stay aligned.
            self.timestamps = [ts for ts, keep in zip(self.timestamps, mask) if keep]
        self.r_multiples = arr[mask]   # drop NaNs / Infs

    @classmethod
    def from_report(cls, report: Any) -> "HypothesisHistory":
        """Build from a ``HypothesisReport`` (research/harness.py). Reads
        the per-trade R-multiples from ``report.trades`` (list of dicts
        produced by ``HypothesisReport.to_row``)."""
        rs: List[float] = []
        ts: List[Any] = []
        for t in (getattr(report, "trades", None) or []):
            r = t.get("r_multiple") if isinstance(t, dict) else getattr(t, "r_multiple", None)
            if r is None:
                continue
            rs.append(float(r))
            ts.append(t.get("exit_ts") if isinstance(t, dict) else getattr(t, "exit_ts", None))
        overall = getattr(report, "overall", None)
        original_sharpe = float(getattr(overall, "sharpe", 0.0) or 0.0)
        return cls(name=str(getattr(report, "name", "hypothesis")),
                    original_sharpe=original_sharpe,
                    r_multiples=np.asarray(rs, dtype=float),
                    timestamps=ts)


@dataclass
class KillVerdict:
    """Decision + diagnostics for one hypothesis."""
    name: str
    status: str           # "alive" / "watching" / "killed" / "insufficient_data"
    reason: str
    original_sharpe: float
    rolling_sharpe: float
    rolling_window: int
    ratio: float          # rolling_sharpe / original_sharpe (when original > 0)
    n_total: int
    n_recent: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _sharpe(r: np.ndarray) -> float:
    """Per-trade Sharpe ratio (no time scaling) — units are R per trade."""
    if r.size < 2:
        return 0.0
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    return mean / std if std > 1e-12 else 0.0


def verdict_for_hypothesis(history: HypothesisHistory,
                            *,
                            window_trades: int = 30,
                            min_trades: int = MIN_TRADES_FOR_VERDICT,
                            absolute_sharpe_floor: float = DEFAULT_ABSOLUTE_SHARPE_FLOOR,
                            decay_kill_ratio: float = DEFAULT_DECAY_KILL_RATIO,
                            decay_watch_ratio: float = DEFAULT_DECAY_WATCH_RATIO,
                            ) -> KillVerdict:
    """Compute one hypothesis's verdict.

    Discipline:
      * ``insufficient_data`` when fewer than ``min_trades`` are available.
      * ``killed`` when rolling Sharpe is below the absolute floor OR
        below ``decay_kill_ratio * original_sharpe``.
      * ``watching`` when rolling Sharpe is below ``decay_watch_ratio *
        original_sharpe`` but still above the absolute floor.
      * ``alive`` otherwise.
    """
    r = np.asarray(history.r_multiples, dtype=float)
    n_total = int(r.size)
    if n_total < min_trades:
        return KillVerdict(
            name=history.name,
            status="insufficient_data",
            reason=f"only {n_total} trades, need {min_trades}",
            original_sharpe=float(history.original_sharpe),
            rolling_sharpe=_sharpe(r),
            rolling_window=int(window_trades),
            ratio=0.0,
            n_total=n_total,
            n_recent=n_total,
        )
    window = r[-min(window_trades, n_total):]
    rolling = _sharpe(window)
    original = float(history.original_sharpe)
    ratio = rolling / original if abs(original) > 1e-9 else math.copysign(1.0, rolling) if rolling else 0.0
    if rolling < absolute_sharpe_floor:
        status = "killed"
        reason = (f"rolling Sharpe {rolling:+.2f} below absolute floor "
                  f"{absolute_sharpe_floor:+.2f}")
    elif original > 0 and rolling < decay_kill_ratio * original:
        status = "killed"
        reason = (f"rolling Sharpe {rolling:+.2f} is {ratio:.0%} of original "
                  f"{original:+.2f}; below kill ratio {decay_kill_ratio:.0%}")
    elif original > 0 and rolling < decay_watch_ratio * original:
        status = "watching"
        reason = (f"rolling Sharpe {rolling:+.2f} is {ratio:.0%} of original "
                  f"{original:+.2f}; below watch ratio {decay_watch_ratio:.0%}")
    else:
        status = "alive"
        reason = f"rolling Sharpe {rolling:+.2f} on last {len(window)} trades"
    return KillVerdict(
        name=history.name,
        status=status,
        reason=reason,
        original_sharpe=original,
        rolling_sharpe=rolling,
        rolling_window=int(len(window)),
        ratio=float(ratio),
        n_total=n_total,
        n_recent=int(len(window)),
    )


def verdicts_for_basket(histories: Sequence[HypothesisHistory],
                         **kwargs) -> Dict[str, KillVerdict]:
    """Compute verdicts for many hypotheses at once. Returns name -> verdict."""
    return {h.name: verdict_for_hypothesis(h, **kwargs) for h in histories}


def filter_alive(verdicts: Dict[str, KillVerdict],
                 *,
                 include_watching: bool = True) -> List[str]:
    """Helper for the allocator: names of hypotheses that should still
    receive capital. ``include_watching`` lets a caller deploy
    watch-status strategies at reduced size."""
    keep = {"alive"}
    if include_watching:
        keep.add("watching")
    return [name for name, v in verdicts.items() if v.status in keep]


@dataclass
class ResurrectionVerdict:
    """Decision on whether a previously killed hypothesis has recovered.

    Operator-facing: the verdict NEVER auto-resurrects in production.
    It just tells the operator "the data now looks recovered" so they
    can choose to flip the kill bit manually."""
    name: str
    eligible: bool
    reason: str
    rolling_sharpe: float
    rolling_window: int
    n_post_kill: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def check_resurrection(history: HypothesisHistory,
                        *,
                        kill_at_index: int,
                        window_trades: int = DEFAULT_RESURRECT_TRADES,
                        min_post_kill: int = DEFAULT_RESURRECT_TRADES,
                        absolute_sharpe_floor: float = DEFAULT_ABSOLUTE_SHARPE_FLOOR,
                        ) -> ResurrectionVerdict:
    """Test whether the trades AFTER ``kill_at_index`` show recovered edge.

    ``kill_at_index`` is the position in ``history.r_multiples`` at which
    the kill was applied — only trades that came in after that index
    count toward resurrection. This avoids the trap of "the same window
    that killed it also resurrects it because the floor moved."
    """
    r = np.asarray(history.r_multiples, dtype=float)
    if kill_at_index < 0 or kill_at_index > len(r):
        return ResurrectionVerdict(
            name=history.name, eligible=False,
            reason="invalid kill index",
            rolling_sharpe=0.0, rolling_window=0, n_post_kill=0,
        )
    post = r[kill_at_index:]
    n_post = int(post.size)
    if n_post < min_post_kill:
        return ResurrectionVerdict(
            name=history.name, eligible=False,
            reason=f"only {n_post} post-kill trades, need {min_post_kill}",
            rolling_sharpe=_sharpe(post),
            rolling_window=int(min(window_trades, n_post)),
            n_post_kill=n_post,
        )
    window = post[-window_trades:]
    rolling = _sharpe(window)
    if rolling < absolute_sharpe_floor:
        return ResurrectionVerdict(
            name=history.name, eligible=False,
            reason=(f"post-kill rolling Sharpe {rolling:+.2f} still below "
                    f"absolute floor {absolute_sharpe_floor:+.2f}"),
            rolling_sharpe=rolling,
            rolling_window=int(len(window)),
            n_post_kill=n_post,
        )
    return ResurrectionVerdict(
        name=history.name, eligible=True,
        reason=(f"post-kill rolling Sharpe {rolling:+.2f} on last "
                f"{len(window)} trades cleared the floor — operator may "
                f"resurrect manually"),
        rolling_sharpe=rolling,
        rolling_window=int(len(window)),
        n_post_kill=n_post,
    )
