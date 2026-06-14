# Launch readme — operating the cockpit in production

**Date**: 2026-06-14
**Branch**: `claude/liquidity-pool-backtester-1uskb`
**Test status**: Sentinel 325 / liquidity_backtester 841 / cross-codebase 53 = **1,219 tests**

This is the operator + Codex handoff for the cockpit. It covers: what
runs, what env vars are needed, what Codex's Supabase + Google Auth side
plugs into, and the launch-day checklist.

---

## 1. What runs

```
sentinel/server.py            FastAPI app + background market loop
sentinel/static/index.html    operator cockpit (the trading screen)
sentinel/static/console.html  customer console (auditor, builder, etc.)
sentinel/static/sentinel.css  the shared design system
```

Two routes that don't need auth:

```
GET  /healthz   liveness probe (always 200 if the process is up)
GET  /readyz    readiness probe (503 if loop dead / broker down / disk full)
```

Codex's LB / Vercel / Render / whatever should point health checks at
`/readyz`.

---

## 2. Environment variables

| Variable | Default | What it does |
|---|---|---|
| `SENTINEL_DEMO` | unset | `1` → DemoAccount, synthetic ticks, safe for local dev |
| `SENTINEL_JOURNAL_DIR` | `~/.sentinel` | where ledger / mind-reports / promotions land |
| `SENTINEL_TOKEN` | unset | legacy auth header (set to enable the X-Sentinel-Token gate) |
| `KITE_API_KEY` | unset | live mode only — Kite Connect |
| `KITE_ACCESS_TOKEN` | unset | live mode only — Kite Connect daily token |
| `SENTINEL_CONFIRM_REAL` | unset | `1` enables real orders (otherwise dry-run) |
| `SENTINEL_MAX_ORDERS_PER_DAY` | `200` | hard cap; the spine refuses past this |
| `SENTINEL_LEDGER_RESOLVE_MIN` | `10` | journey resolution horizon |

For Codex's auth integration (next section), add whatever Supabase env
vars his code needs (`SUPABASE_URL`, `SUPABASE_JWT_SECRET`, `GOOGLE_CLIENT_ID`).
Sentinel's own code never reads these.

---

## 3. Codex's auth integration — the ONE plug

Sentinel doesn't import Supabase. Codex's side does. The contract:

```python
# Codex's file (their repo or their edge function)
import os, jwt, requests
from sentinel.auth import AuthContext, register_verifier

SUPABASE_JWT_SECRET = os.environ["SUPABASE_JWT_SECRET"]

def supabase_verify(token: str) -> AuthContext:
    claims = jwt.decode(
        token, SUPABASE_JWT_SECRET,
        algorithms=["HS256"], audience="authenticated")
    user_id = claims["sub"]
    email = claims.get("email", "")
    # Codex looks up the plan tier in Supabase
    plan = supabase_table("users").select("plan").eq("id", user_id).single().data["plan"]
    return AuthContext(
        user_id=user_id, email=email,
        plan=plan or "RETAIL",
        supabase_jwt=token)

# Call this ONCE at app boot, before uvicorn starts serving
register_verifier(supabase_verify)
```

What happens then:

- Operator hits `/` (or `/console`).
- Frontend redirects to Google OAuth via Supabase Auth.
- Supabase returns a JWT and stores user state.
- Frontend includes `Authorization: Bearer <jwt>` on every `/api/*` call.
- Sentinel's `resolve_auth` dependency reads the header, calls
  `verify_token(...)`, which routes to Codex's `supabase_verify(...)`.
- The returned `AuthContext` exposes `.plan` — the existing
  `require("feature.x", plan)` gating works unchanged.

If Codex isn't ready yet, set `X-Sentinel-Plan: FOUNDER` in the dev
console. That's the fallback path that keeps local cockpit work
unblocked. **Do not ship this path enabled in production** — disable it
by writing your own verifier that ignores the legacy header.

---

## 4. The launch-day checklist

Operator flow on a normal trading morning:

1. **08:30 IST — pre-market brief**
   - liquidity_backtester's nightly cron generated `daily_brief.json`
     for today
   - Sentinel's `liqpool_bridge.load_research_pack(session)` reads it
     when the cockpit boots
2. **09:00 IST — operator logs in** via Google Auth → Supabase JWT
3. **09:00–09:14 IST — preflight modal**
   - Forced popup on first load of the session
   - Operator must commit: max day loss, expected regime, brief read
   - This calls `/api/preflight/ack`, which sets the Ulysses-pattern
     intention contract on the psychology engine
   - **Until acknowledged, Crux refuses TRADE verdicts**
4. **09:15 IST — market opens**
   - `liqpool/live_inference.py` starts publishing signals via JSONL
     (Codex's Kite WS layer feeds it — see §6)
   - Sentinel's `liqpool_tail.poll()` reads the JSONL every quote cycle
   - Crux composer fuses everything into one verdict
5. **14:30 IST — MIS no-new-entry**
   - Live inference auto-tags signals with `mis_no_new_entry=True`
   - Cockpit shows "MIS no-new-entry" chip; Crux refuses TRADE
6. **15:15 IST — MIS auto square-off**
   - Live inference auto-tags with `mis_squareoff=True`
   - TrailEngine / ProfitLock flatten any remaining positions
7. **15:30 IST — close**
8. **Night cron**
   - `python -m sentinel.ledger_export --session $(date +%F)` exports
     the day's decisions to JSONL
   - `python -m scripts.train_flywheel --bundle latest_bundle.pkl
      --shadow-root data/shadow --outcome-root data/outcome
      --sentinel-journal ~/.sentinel --out models/flywheel` trains
     tomorrow's flywheel hub on the COMBINED data

---

## 5. What this code does NOT do (deliberately)

1. **No order placement** beyond exit-only (trailing stop, profit
   lock). The trust spine enforces it.
2. **No imports of Supabase / `requests` / Postgres clients.** Codex's
   side handles everything network-cloud-side.
3. **No live Kite websocket feed.** The `LiveInferenceServer.on_tick`
   contract is in place; Codex (or a follow-up sprint) writes the
   producer that feeds it ticks from Kite's WS API. See §6.

---

## 6. Known-good ports & follow-ups

**Required before going live with real money:**

- Live Kite WS → `LiveInferenceServer.on_tick` producer
  (currently demo mode synthesises ticks; production needs the real
  WS subscriber)
- Codex's `register_verifier` call wired at app boot
- Real model bundle trained on real warehouse data (the demo bundle
  ships with the `_FakeReport` fixture only)
- A reverse proxy (nginx / Caddy / Render) that terminates TLS and
  rate-limits

**Nice-to-have (Wave 18+):**

- `RunManifest` in `multi_asset_run.py` entry points
- TradingView Lightweight Charts swap for the spot chart
- Mobile cockpit layout
- Per-user layout persistence

---

## 7. Run commands

```bash
# Local dev (no auth needed, synthetic ticks)
SENTINEL_DEMO=1 uvicorn sentinel.server:app --reload --port 8800

# Production (after Codex's register_verifier is wired)
uvicorn sentinel.server:app --host 0.0.0.0 --port 8800 --workers 1

# Real broker, REAL orders (sets the spine to LIVE, not DRY_RUN)
SENTINEL_CONFIRM_REAL=1 uvicorn sentinel.server:app --host 0.0.0.0 --port 8800

# Tests
python -m pytest sentinel/tests -q             # 325 tests
python -m pytest liquidity_backtester/tests -q # 841 + cross-codebase
```

---

## 8. The one thing Codex MUST NOT do without explicit sign-off

**Touch the trust spine** (`sentinel/orchestration.py`). Only the
hard-wired EXIT actors (`trailing_stop`, `profit_lock`) reach the order
path. Adding a new EXECUTION source — or relaxing the kill-switch
gate, or removing the RED-tilt block — is a behavioral safety
regression. If a feature needs to place orders, it goes through the
existing pattern: get graduated to TRUSTED by the curator first, then
the SPINE decides if it gets EXECUTION. Don't shortcut.

Everything else is fair game.
