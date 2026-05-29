"""Raw-output ingestion engine for the opaque analytics feed.

This engine is intentionally decoupled from the model/predict code. It does NOT
import or run any model. It reads the *raw output files* that a predict run
already writes (``live_gate_decisions.csv``, ``tradeable_setups.csv``,
``watchlist.csv``, ``rejected_setups.csv``, ``track_a_pretouch_setups.csv``, or
``live_plan.json``) and converts each level row into a server-side
:class:`~liqpool.scoring.InternalLevel`.

Those internal levels are then handed to :mod:`liqpool.scoring`, which turns them
into opaque, non-invertible public records. So the data path is:

    model/predict (untouched)  ->  raw CSV/JSON  ->  THIS ENGINE  ->  scoring  ->  public feed

Column names vary slightly across the raw files, so a small alias resolver maps
whatever is present onto the fields we need (reachability, direction, quality,
price-band geometry).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from liqpool.scoring import InternalLevel

# Column aliases across the various raw output files.
_TOUCH_ALIASES = ("p_touch", "t_today", "p_touch_today")
_UP_ALIASES = ("p_up", "dir_p_up")
_DIRPOOL_ALIASES = ("p_direction_to_pool",)
_Q_ALIASES = ("q", "p_respect", "q_score")
_LOW_ALIASES = ("pool_low", "level_low", "price_low")
_HIGH_ALIASES = ("pool_high", "level_high", "price_high")
_MID_ALIASES = ("pool_mid", "level_mid", "mid")
_SIDE_ALIASES = ("side",)
_SYMBOL_ALIASES = ("symbol", "instrument")

# Raw files in priority order: first one found wins as the level source.
_PRIMARY_FILES = ("live_gate_decisions.csv",)
_SPLIT_FILES = ("tradeable_setups.csv", "watchlist.csv", "rejected_setups.csv")
_FALLBACK_CSV = ("track_a_pretouch_setups.csv",)


def _first(row: dict, aliases: Sequence[str]) -> Any:
    for a in aliases:
        if a in row and row[a] is not None and row[a] != "":
            return row[a]
    return None


def _to_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _derive_p_up(row: dict, side: str) -> float | None:
    """Recover a market up-probability from whatever directional field exists.

    Prefers a real probability; otherwise derives a coarse proxy from the
    categorical direction fields the gate writes. Returns None when direction is
    genuinely unavailable.
    """
    p_up = _to_float(_first(row, _UP_ALIASES))
    if p_up is not None:
        return min(max(p_up, 0.0), 1.0)

    p_dir = _to_float(_first(row, _DIRPOOL_ALIASES))
    if p_dir is not None:
        p_dir = min(max(p_dir, 0.0), 1.0)
        return p_dir if side == "above" else (1.0 - p_dir)

    sign = _to_float(row.get("direction_sign"))
    if sign is not None and sign != 0.0:
        return min(max(0.5 + 0.15 * (1.0 if sign > 0 else -1.0), 0.0), 1.0)

    tag = row.get("dir_tag")
    if isinstance(tag, str):
        if tag == "DIR_ALIGN":
            return 0.65 if side == "above" else 0.35
        if tag == "DIR_FIGHT":
            return 0.35 if side == "above" else 0.65
        if tag == "DIR_NEUTRAL":
            return 0.5
    return None


def row_to_internal_level(row: dict, as_of: str, scope: str = "s0") -> InternalLevel | None:
    """Map one raw level row to an InternalLevel, or None if it lacks geometry."""
    symbol = _first(row, _SYMBOL_ALIASES)
    if not symbol:
        return None
    side = str(_first(row, _SIDE_ALIASES) or "above")

    low = _to_float(_first(row, _LOW_ALIASES))
    high = _to_float(_first(row, _HIGH_ALIASES))
    mid = _to_float(_first(row, _MID_ALIASES))
    if low is None and high is None and mid is None:
        return None
    if mid is None:
        mid = (low + high) / 2.0 if (low is not None and high is not None) else (low or high)
    if low is None:
        low = mid
    if high is None:
        high = mid

    return InternalLevel(
        symbol=str(symbol),
        side=side,
        level_low=float(low),
        level_high=float(high),
        level_mid=float(mid),
        p_touch=_to_float(_first(row, _TOUCH_ALIASES)) or 0.0,
        p_up=_derive_p_up(row, side),
        q=_to_float(_first(row, _Q_ALIASES)) or 0.0,
        as_of=str(as_of),
        scope=scope,
    )


def levels_from_rows(rows: Iterable[dict], as_of: str,
                     scope: str = "s0") -> list[InternalLevel]:
    out = []
    for row in rows:
        lvl = row_to_internal_level(row, as_of, scope)
        if lvl is not None:
            out.append(lvl)
    return out


def _read_csv_rows(path: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _read_live_plan_rows(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    rows: list[dict] = []
    rows.extend(data.get("tradeable_setups", []) or [])
    track_a = data.get("track_a_pretouch", {}) or {}
    rows.extend(track_a.get("setups", []) or [])
    return rows


def read_raw_levels(source: str | Path) -> list[dict]:
    """Read raw level rows from a predict-output directory or a single file.

    Directory: tries ``live_gate_decisions.csv`` first (full candidate set), then
    the split CSVs combined, then ``track_a_pretouch_setups.csv``, then
    ``live_plan.json``. File: reads that file by extension.
    """
    p = Path(source)
    if p.is_file():
        if p.suffix == ".json":
            return _read_live_plan_rows(p)
        return _read_csv_rows(p)
    if not p.is_dir():
        raise FileNotFoundError(f"raw source not found: {p}")

    for name in _PRIMARY_FILES:
        f = p / name
        if f.exists():
            return _read_csv_rows(f)

    combined: list[dict] = []
    for name in _SPLIT_FILES:
        f = p / name
        if f.exists():
            combined.extend(_read_csv_rows(f))
    if combined:
        return combined

    for name in _FALLBACK_CSV:
        f = p / name
        if f.exists():
            return _read_csv_rows(f)

    lp = p / "live_plan.json"
    if lp.exists():
        return _read_live_plan_rows(lp)

    raise FileNotFoundError(f"no recognised raw output files under {p}")


class RawFeedScorer:
    """Scorer that ingests a predict-output directory (or single raw file).

    Reads the raw rows once at construction (or lazily per call if ``cache`` is
    False) and serves opaque levels per symbol. Touches no model code.
    """

    def __init__(self, source: str | Path, as_of: str | None = None,
                 scope: str = "s0", cache: bool = True):
        self._source = source
        self._scope = scope
        self._as_of = as_of
        self._cache_rows: list[dict] | None = None
        self._cache = cache

    def _rows(self) -> list[dict]:
        if self._cache and self._cache_rows is not None:
            return self._cache_rows
        rows = read_raw_levels(self._source)
        if self._cache:
            self._cache_rows = rows
        return rows

    def internal_levels(self, symbol: str, date: str,
                        universe: str | None = None) -> list[InternalLevel]:
        as_of = self._as_of or date
        rows = [r for r in self._rows() if str(_first(r, _SYMBOL_ALIASES)) == symbol]
        return levels_from_rows(rows, as_of, self._scope)
