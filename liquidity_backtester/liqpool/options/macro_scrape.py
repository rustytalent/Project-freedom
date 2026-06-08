"""Overnight macro scrape — Yahoo Finance free public endpoint.

Closes the gap noted in the D.6 commit: today's model trains only
on warehouse-resident macro (India VIX, USDINR). The cascade's
macro layer ALSO wants overnight global indexes — SPX/DJI/NDX —
plus the SGX-Nifty futures proxy and US VIX, per
``docs/options_executor_layered_conviction.md`` §2 Layer 1.

This module fetches those from Yahoo's free public endpoint with
graceful degradation: any field that the endpoint refuses (rate
limit, 403, network failure, malformed JSON) returns ``NaN``. The
downstream macro-score function then collapses to 0 when too much
data is missing, which closes the cascade gate and the executor
SKIPs the day. This is the documented behaviour from §11 of the
amendment.

Production path when revenue allows: swap the ``MacroAdapter``
implementation for a paid feed (Polygon, Twelve Data, AlphaVantage)
by replacing one class. The score function and the featurizer
contract stay the same.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Protocol


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# What we fetch
# ---------------------------------------------------------------------------

# Yahoo Finance tickers for the overnight signals we want.
# Order matters only for stability of the result dict iteration.
MACRO_TICKERS: Dict[str, str] = {
    "spx":      "^GSPC",       # S&P 500
    "dji":      "^DJI",        # Dow Jones Industrial
    "ndx":      "^IXIC",       # NASDAQ Composite
    "vix_us":   "^VIX",        # CBOE Volatility Index
    # SGX Nifty futures don't have a stable Yahoo ticker; the
    # 5-bar ^NSEI close serves as the "Asian-session Nifty" proxy
    # for the morning. When a paid feed comes online, swap to the
    # actual SGX/GIFT Nifty futures level.
    "nifty":    "^NSEI",
}

# Browser-style UA so the public endpoint serves us. Yahoo blocks
# default Python UAs; this matches a real Chrome request.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------

@dataclass
class MacroOvernightSnapshot:
    """One morning's overnight macro values.

    Each field is the day-over-day percent change at yesterday's
    close, ``NaN`` when the underlying fetch failed. The macro-score
    function treats NaN as "missing" — averages skip them; a fully-
    missing snapshot returns score 0 (gate closed).
    """
    as_of_utc: str
    spx_dod_pct: float
    dji_dod_pct: float
    ndx_dod_pct: float
    vix_us_dod_points: float            # absolute, NOT percent
    nifty_yday_close_pct: float         # NIFTY's own DoD pct, proxy for SGX
    sources: Dict[str, str]             # ticker → "ok" | error string

    @property
    def has_any_data(self) -> bool:
        import math
        return any(not math.isnan(v) for v in (
            self.spx_dod_pct, self.dji_dod_pct, self.ndx_dod_pct,
            self.vix_us_dod_points, self.nifty_yday_close_pct,
        ))


# ---------------------------------------------------------------------------
# Adapter interface — swap to paid feed by implementing this
# ---------------------------------------------------------------------------

class MacroAdapter(Protocol):
    """Pluggable macro feed. The Yahoo adapter is the v1 default;
    paid-feed adapters land when revenue allows."""

    def fetch(self) -> MacroOvernightSnapshot: ...


# ---------------------------------------------------------------------------
# Yahoo adapter
# ---------------------------------------------------------------------------

class YahooMacroAdapter:
    """Fetches overnight values from Yahoo Finance's free public
    endpoint. Public endpoint is unauthenticated but rate-limited;
    callers should not fetch more than once per session per morning.

    Every fetch is wrapped in try/except — any error per ticker
    leaves the corresponding field as ``NaN`` and tags the source as
    the error type. The snapshot is returned regardless of how many
    tickers succeeded.
    """

    def __init__(self, timeout_seconds: float = 8.0):
        self.timeout_seconds = timeout_seconds

    def fetch(self) -> MacroOvernightSnapshot:
        import math
        from datetime import datetime, timezone

        # Build per-ticker results map.
        results: Dict[str, tuple[float, str]] = {}
        for key, ticker in MACRO_TICKERS.items():
            try:
                close_pct = self._fetch_dod_pct(ticker)
                results[key] = (close_pct, "ok")
            except Exception as exc:                              # noqa: BLE001
                LOGGER.warning(
                    "macro fetch failed for %s (%s): %s",
                    key, ticker, exc,
                )
                results[key] = (math.nan, type(exc).__name__)

        # VIX-US is reported as POINT change, not percent.
        # We computed it as percent above for consistency; convert
        # back to absolute by multiplying by the prior close.
        vix_pct = results["vix_us"][0]
        vix_abs = math.nan if math.isnan(vix_pct) else self._vix_abs_change(
            "^VIX", vix_pct)

        return MacroOvernightSnapshot(
            as_of_utc=datetime.now(timezone.utc).isoformat(),
            spx_dod_pct=results["spx"][0],
            dji_dod_pct=results["dji"][0],
            ndx_dod_pct=results["ndx"][0],
            vix_us_dod_points=vix_abs,
            nifty_yday_close_pct=results["nifty"][0],
            sources={k: status for k, (_, status) in results.items()},
        )

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _fetch_dod_pct(self, ticker: str) -> float:
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{urllib.parse.quote(ticker)}?interval=1d&range=5d"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req,
                                      timeout=self.timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return _last_dod_pct_from_chart(payload)

    def _vix_abs_change(self, ticker: str, pct: float) -> float:
        """Re-fetch VIX to recover the absolute prior-close (so we can
        emit a point change rather than a percent change — for VIX
        the methodology cares about points)."""
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{urllib.parse.quote(ticker)}?interval=1d&range=5d"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _USER_AGENT,
                     "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout_seconds
            ) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            closes = _closes_from_chart(payload)
            if len(closes) >= 2:
                return float(closes[-1] - closes[-2])
        except Exception:                                          # noqa: BLE001
            pass
        return float("nan")


# ---------------------------------------------------------------------------
# Pure helpers — independent of HTTP, fully testable
# ---------------------------------------------------------------------------

def _closes_from_chart(payload: dict) -> list[float]:
    """Pull a sorted list of close prices from Yahoo's chart payload."""
    chart = payload.get("chart", {})
    result = chart.get("result") or []
    if not result:
        return []
    r = result[0]
    indicators = r.get("indicators", {}) or {}
    quote = (indicators.get("quote") or [{}])[0]
    closes = quote.get("close", []) or []
    # Yahoo sometimes returns trailing None on a fresh day; trim.
    return [float(c) for c in closes if c is not None]


def _last_dod_pct_from_chart(payload: dict) -> float:
    closes = _closes_from_chart(payload)
    if len(closes) < 2:
        raise ValueError("not enough close bars to compute DoD")
    prev, last = closes[-2], closes[-1]
    if prev == 0:
        raise ValueError("zero previous close")
    return (last - prev) / prev * 100.0


# Bind urllib.parse lazily since the import is otherwise unused at
# module load time on platforms without it.
import urllib.parse  # noqa: E402


# ---------------------------------------------------------------------------
# Stub adapter — for tests and for environments where Yahoo is blocked
# ---------------------------------------------------------------------------

class StubMacroAdapter:
    """Returns a pre-canned snapshot. Used in tests; also a sane
    operator-time fallback when Yahoo is blacklisted from the
    environment (e.g. some VPS providers' default firewalls)."""

    def __init__(self, snapshot: MacroOvernightSnapshot):
        self.snapshot = snapshot

    def fetch(self) -> MacroOvernightSnapshot:
        return self.snapshot


def empty_snapshot(as_of_utc: str = "1970-01-01T00:00:00+00:00"
                    ) -> MacroOvernightSnapshot:
    """Snapshot with every field NaN — the macro_score function will
    return 0 on this, closing the cascade gate. Use this when Yahoo
    is unreachable and you want the executor to SKIP the day cleanly
    rather than crash."""
    import math
    return MacroOvernightSnapshot(
        as_of_utc=as_of_utc,
        spx_dod_pct=math.nan,
        dji_dod_pct=math.nan,
        ndx_dod_pct=math.nan,
        vix_us_dod_points=math.nan,
        nifty_yday_close_pct=math.nan,
        sources={k: "stub_empty" for k in MACRO_TICKERS},
    )
