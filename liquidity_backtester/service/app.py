"""Thin HTTP adapter for the opaque analytics feed.

All logic lives in :mod:`liqpool.serving` / :mod:`liqpool.scoring` /
:mod:`liqpool.ingest`; this module only does HTTP concerns: API-key auth,
per-key rate limiting, and request/response shaping. Inference / scoring runs
server-side and the response carries only opaque, allow-listed records.

Configuration (environment variables):
  * ``GFEED_API_KEYS`` — comma-separated ``key:customer_id`` pairs. If unset, a
    single ``demo-key:demo`` pair is used only outside production.
  * ``GFEED_RAW_DIR`` — predict-output directory to ingest via RawFeedScorer.
    If unset, a deterministic StubScorer is used only outside production.
  * ``GFEED_ENV`` — set to ``production`` to fail closed when required serving
    inputs are missing.
  * ``GFEED_ALLOW_DEV_DEFAULTS`` — set to 1/true/yes to explicitly allow demo
    API keys and the stub scorer in non-production environments.
  * ``GFEED_RATE_PER_MIN`` — max requests per customer per minute (default 120).

Run locally:
    uvicorn service.app:app --reload    # from the liquidity_backtester/ dir
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from liqpool.serving import RawFeedScorer, StubScorer, handle_levels_request

app = FastAPI(
    title="Market-Structure Analytics Feed",
    version="1",
    description=(
        "Non-recommendatory quantitative market-structure analytics. Outputs are "
        "opaque statistical features for independent research use only — not "
        "investment advice, not a recommendation, not an instruction to trade."
    ),
)


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _production_mode() -> bool:
    return os.environ.get("GFEED_ENV", "").strip().lower() in {"prod", "production"}


def _allow_dev_defaults() -> bool:
    return (not _production_mode()) or _is_truthy_env("GFEED_ALLOW_DEV_DEFAULTS")


def _load_api_keys() -> dict[str, str]:
    raw = os.environ.get("GFEED_API_KEYS", "").strip()
    if not raw:
        if not _allow_dev_defaults():
            raise RuntimeError(
                "GFEED_API_KEYS is required when GFEED_ENV=production"
            )
        return {"demo-key": "demo"}
    keys: dict[str, str] = {}
    for pair in raw.split(","):
        if ":" in pair:
            k, cust = pair.split(":", 1)
            keys[k.strip()] = cust.strip()
    if not keys:
        raise RuntimeError("GFEED_API_KEYS was provided but no key:customer pairs parsed")
    return keys


def _build_scorer():
    raw_dir = os.environ.get("GFEED_RAW_DIR", "").strip()
    if raw_dir:
        return RawFeedScorer(raw_dir)
    if not _allow_dev_defaults():
        raise RuntimeError(
            "GFEED_RAW_DIR is required when GFEED_ENV=production"
        )
    return StubScorer()


API_KEYS = _load_api_keys()
SCORER = _build_scorer()
RATE_PER_MIN = int(os.environ.get("GFEED_RATE_PER_MIN", "120"))

_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = Lock()


def _check_rate(customer_id: str) -> None:
    now = time.monotonic()
    with _hits_lock:
        dq = _hits[customer_id]
        while dq and now - dq[0] > 60.0:
            dq.popleft()
        if len(dq) >= RATE_PER_MIN:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        dq.append(now)


def require_customer(x_api_key: str | None = Header(default=None)) -> str:
    customer = API_KEYS.get(x_api_key or "")
    if customer is None:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    _check_rate(customer)
    return customer


@app.get("/healthz")
def healthz() -> dict:
    scorer_name = type(SCORER).__name__
    return {
        "status": "ok",
        "mode": "production" if _production_mode() else "development",
        "scorer": scorer_name,
        "using_stub_scorer": isinstance(SCORER, StubScorer),
        "using_demo_key": "demo-key" in API_KEYS,
    }


@app.get("/v1/levels")
def levels(symbol: str = Query(..., min_length=1),
           date: str = Query(...),
           customer: str = Depends(require_customer)) -> dict:
    try:
        return handle_levels_request(SCORER, symbol=symbol, date=date,
                                     customer_id=customer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
