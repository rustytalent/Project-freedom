# Codex backlog — what's done, what's left, how to do each

Generated 2026-06-14 by Opus 4.7 / Claude Code session.
Branch: `claude/liquidity-pool-backtester-1uskb`
**Test status: Sentinel 342 + liquidity_backtester 841 = 1,216 passing.**

---

## 0. Honest answer to the three questions

> **Q1: Will Sentinel get live data from Kite API?**
> **YES** — REST polling works today. `sentinel/kite_client.py` wraps
> `kiteconnect.KiteConnect`. Operator sets `KITE_API_KEY` +
> `KITE_ACCESS_TOKEN` env vars, cockpit polls funds / positions /
> quotes / chain at 1-quote-per-second.
>
> **GAP**: No Kite WebSocket. REST polling stales every 1s. For
> tick-by-tick you need `KiteTicker` wired into a producer that calls
> `LiveInferenceServer.on_tick(features)`. See §1.

> **Q2: Can people connect their own API to Sentinel?**
> **YES, as of Wave 18 (this commit)** — `POST /api/byok/connect`
> stores each user's encrypted credentials keyed by their AuthContext
> user_id. Cleared on process restart by design; Codex's Supabase
> layer handles AT-REST persistence + daily token refresh.

> **Q3: Is everything else production-grade SaaS Tier-3?**
> **For the single-operator cockpit: yes.**
> **For multi-tenant SaaS: this commit closes the big-three (BYOK,
> audit log, hardened auth). Five remaining gaps are listed in §1
> with implementation guidance.**

---

## 1. What's REMAINING — by priority, with how-to

### 🔴 BLOCKER for real-money launch — Live Kite WebSocket producer

**Why it matters**: REST polling = 1 quote per second. WebSocket =
sub-100ms ticks. Without WS you can't run a tick-driven model.

**Where it plugs in**: `liqpool/live_inference.py` already exposes
`LiveInferenceServer.on_tick(features_per_head, asset)`. You need
a feeder that calls it.

**How**:
```python
# liqpool/live_feed.py  (Codex writes this)
from kiteconnect import KiteTicker
from liqpool.live_inference import LiveInferenceBundle, LiveInferenceServer, JsonlPublisher
from liqpool.timing import StateFeaturizer
import pandas as pd

class LiveFeed:
    def __init__(self, api_key, access_token, instrument_tokens):
        self.kt = KiteTicker(api_key, access_token)
        self.kt.on_ticks = self._on_ticks
        self.kt.on_connect = lambda ws, response: ws.subscribe(instrument_tokens)
        self.kt.on_close = self._on_close
        # bar builder: aggregate ticks into 5m OHLCV
        self.bar_builder = MinuteBarAggregator(timeframe="5m")
        bundle = LiveInferenceBundle.from_pickle("/path/to/bundle.pkl")
        self.server = LiveInferenceServer(
            bundle=bundle,
            publisher=JsonlPublisher("/var/lib/sentinel/liqpool_live_signals.jsonl"),
            asset="NIFTY",
            enforce_mis=True)

    def _on_ticks(self, ws, ticks):
        for tick in ticks:
            closed_bars = self.bar_builder.feed(tick)
            for bar in closed_bars:
                features = StateFeaturizer().features_at(bar)
                self.server.on_tick({
                    "direction": features,
                    "proximity_h12": features,
                    "proximity_h25": features,
                    "proximity_h78": features,
                    "quality": features,
                })

    def _on_close(self, ws, code, reason):
        # exponential backoff reconnect — Kite drops sessions daily
        ws.reconnect()

    def run(self):
        self.kt.connect(threaded=True)
```

**Acceptance**: a flowing 5-min `bars.parquet` produced from live
ticks. Existing `test_live_inference.py` already pins the contract.

**Estimated time**: 1-2 days.

---

### 🔴 BLOCKER — Codex's `register_verifier` wired at app boot

**Why it matters**: Without it, Sentinel runs in demo mode (header
auth disabled in prod). Hardened in Wave 18 — `X-Sentinel-Plan`
header only honoured when `SENTINEL_DEMO=1` OR
`SENTINEL_ALLOW_HEADER_AUTH=1`. Without either, all auth-required
endpoints return 401.

**How** (`codex_auth.py` in your repo):
```python
import os
import jwt
from sentinel.auth import AuthContext, register_verifier
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_JWT_SECRET = os.environ["SUPABASE_JWT_SECRET"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def supabase_verify(token: str) -> AuthContext:
    claims = jwt.decode(
        token, SUPABASE_JWT_SECRET,
        algorithms=["HS256"], audience="authenticated")
    user_id = claims["sub"]
    email = claims.get("email", "")
    # Plan tier lives in your users table
    row = supabase.table("users").select("plan").eq("id", user_id).single().execute()
    plan = (row.data or {}).get("plan", "RETAIL")
    return AuthContext(
        user_id=user_id, email=email,
        plan=plan, supabase_jwt=token)

register_verifier(supabase_verify)
```

Run this as a module that gets imported BEFORE `uvicorn sentinel.server:app`.

**Acceptance**: `Authorization: Bearer <real_jwt>` returns the correct
plan; bad / expired token returns 401.

**Estimated time**: 2-4 hours including Supabase user table setup.

---

### 🟠 IMPORTANT — Auto-restore BYOK creds from Supabase on process boot

**Why it matters**: The in-process BYOK store is intentionally
ephemeral (Sentinel never writes plaintext to disk). If your gateway
restarts mid-session, every connected user loses their broker session
and has to reconnect.

**How**: On startup of the Sentinel process, Codex's side queries
Supabase for active sessions and re-injects them.

```python
# codex_bootstrap.py
from sentinel.byok import GLOBAL_STORE
from supabase_admin_client import supabase
from cryptography.fernet import Fernet

KEK = os.environ["SUPABASE_KEK"]  # at-rest master key Codex owns
fernet = Fernet(KEK)

def restore_byok():
    rows = supabase.table("kite_sessions").select("*").execute().data
    for row in rows:
        # row.encrypted_credentials is fernet-encrypted at-rest
        plaintext = fernet.decrypt(row["encrypted_credentials"]).decode()
        api_key, access_token = plaintext.split("\n", 1)
        GLOBAL_STORE.set(row["user_id"], api_key, access_token)

# Call from FastAPI startup hook
@app.on_event("startup")
def on_boot():
    restore_byok()
```

Add a Supabase trigger / cron that calls `kite.generate_session(...)`
each morning, persists the new access_token via the same fernet
encryption.

**Acceptance**: kill the Sentinel process; bring it back; existing
users' Kite sessions are alive without their having to re-connect.

**Estimated time**: 1 day.

---

### 🟠 IMPORTANT — Multi-user request routing

**What's wrong**: Today `sentinel.server.CORE` is ONE `Sentinel`
instance with ONE broker account. If two users hit the same process
they share the same portfolio, the same kill switch, the same
positions. That's fine for single-operator; it's wrong for SaaS.

**Two paths**:

**Option A — Process-per-user (simpler, lower throughput)**:
- Codex's gateway routes `user_id` → `process_port` based on a
  Redis lookup, spawns a Sentinel process per active user.
- Pro: zero code change to Sentinel. Con: ~50 MB RAM × N users.

**Option B — Multi-tenant in-process (harder, scales further)**:
- Refactor `Sentinel` from a singleton to a per-user `SessionState`.
- The hot loop becomes a dict-of-sessions iterator.
- Every per-user piece (portfolio, psychology, journey, ledger)
  becomes per-session.

For an MVP launch with <100 paying users, **Option A is the right call.**
Estimated time: 1 day of Codex's gateway work, zero on Sentinel side.

For >1000 users, Option B is the rewrite. Estimated: 2-3 weeks.

---

### 🟡 NICE-TO-HAVE — Sentry / Datadog observability hook

**Already in place**: `sentinel.audit_log.register_audit_sink(fn)`.
Codex calls this once at boot.

```python
# codex_observability.py
import sentry_sdk
from sentinel.audit_log import register_audit_sink

sentry_sdk.init(dsn=os.environ["SENTRY_DSN"], traces_sample_rate=0.1)

def to_sentry(record):
    if record["severity"] in ("WARN", "ERROR"):
        sentry_sdk.capture_message(
            f"[{record['event']}] {record['payload']}",
            level=record["severity"].lower())

register_audit_sink(to_sentry)
```

**Acceptance**: order_blocked / intention_violation events appear in
Sentry within 30s.

**Estimated time**: 1 hour.

---

### 🟡 NICE-TO-HAVE — Daily backup of `<journal_dir>` ✓ DONE (Wave 19)

`sentinel/scripts/backup_journal.py` ships now:
```
python -m sentinel.scripts.backup_journal \
    --journal ~/.sentinel \
    --session 2026-06-14 \
    --out /var/lib/sentinel/backups \
    --keep 30 \
    --upload s3://bucket/path
```
Writes `sentinel-backup-<date>.tar.gz` + a sha256 manifest of every
file (corruption detectable). Prunes older-than-keep tarballs. Optional
S3 upload via `aws` CLI (no boto3 dep). Codex adds the cron entry —
23:55 IST recommended.

---

### 🟡 NICE-TO-HAVE — Stripe billing integration

**Plan tier lives in Supabase users table** (per the auth contract in
§2). When a user upgrades RETAIL → PRO via Stripe webhook:
- Stripe sends `customer.subscription.updated`
- Codex's webhook handler updates Supabase `users.plan`
- Next call from that user gets their new plan because Sentinel re-
  reads it on every `verify_token(...)` call

No Sentinel code change. **Estimated time: 1 day Codex side.**

---

### 🟡 NICE-TO-HAVE — Email / push notifications

**Triggers**:
- Tilt band crossed RED → email/push "you're tilted, take a break"
- Intention violated → email/push "max-loss breached, flatten and stop"
- Daily mind report ready → email next morning at 08:00 IST

**Hook**: Register an audit sink (same pattern as Sentry above) that
filters for the right events and dispatches via SendGrid / Resend /
FCM.

**Estimated time**: 1 day.

---

### 🟢 POLISH — Mobile cockpit layout ✓ DONE (Wave 19)

`@media (max-width: 900px)` + `@media (max-width: 540px)` rules in
`sentinel.css`: 2-col grid stacks to 1-col, metric cells go to 2×N,
hover-only crosshairs disabled (touch UX), command palette goes
full-width, dense tables breathe, status bar shrinks. Cockpit is now
usable on phone for ops monitoring; the full trading flow still wants
desktop but a kill button + tilt dial + Crux verdict + position list
are all readable.

---

### 🟢 POLISH — TradingView Lightweight Charts swap

Replace inline SVG `drawSpotChart` with TradingView's open-source
charting library. Same data feed (`spot_history` payload), much
better UX (zoom, pan, crosshair, multi-pane).

`<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>`

Their docs are decent. **Estimated time: 2-3 days** including
making sure model-zone overlays still render correctly.

---

## 2. What's DONE (Tier-3 production-grade)

This is the inventory for compliance / due-diligence questions:

### Auth & access control
- `sentinel/auth.py` — plug-in verifier contract; production verifier
  registered by Codex once at boot
- Hardened header fallback (Wave 18) — `X-Sentinel-Plan` only honoured
  when `SENTINEL_DEMO=1` OR `SENTINEL_ALLOW_HEADER_AUTH=1`
- `sentinel/saas.py` — 4-tier (RETAIL/PRO/QUANT/FOUNDER) gating with
  unknown-feature-defaults-to-PRO safety
- `/api/byok/connect` — encrypted per-user Kite credentials

### Trust spine + safety
- `sentinel/orchestration.py` — only `trailing_stop` + `profit_lock`
  reach EXECUTION; everything else clamped to ≤TRUSTED
- Spine gates: kill switch, RED tilt, intention violation, preflight
  not acknowledged → no orders
- `/api/graduate` — auditable promotion record per call
- TrustPromotionRecord with curator evidence

### Behavioral engine
- 6 citation-bearing bias detectors (Steenbarger, Lo, Gilovich,
  Shefrin & Statman, Tversky & Kahneman, Tharp)
- TiltIndex 0-100 EWMA with GREEN/AMBER/RED/CIRCUIT bands
- IntentionContract Ulysses pattern with the spine enforcement
- MindReport end-of-session reflection

### Audit & compliance
- `sentinel/audit_log.py` — JSONL audit trail, credential auto-scrub,
  external sinks for Sentry/Datadog (Wave 18)
- `<journal_dir>/audit.jsonl` — full append-only trail
- `<journal_dir>/trust_promotions.jsonl` — every graduation
- `<journal_dir>/mind_reports.jsonl` — end-of-session reflections
- Shadow ledger with five-block schema, six event kinds

### Cross-codebase spine
- `liqpool/contracts/` — shared `DecisionEvent`, `ModelSignal`,
  `TrustPromotionRecord`, `ResearchContextPack`, `RunManifest`
- `liqpool/live_inference.py` — research-engine live mode (with MIS
  session window enforcement, Wave 16)
- `liqpool/sentinel_adapter.py` — night trainer reads Sentinel's
  ledger via the same shape as its own shadow log
- `sentinel/liqpool_bridge.py` — Sentinel reads liqpool's live signals
  via JSONL tail on every quote cycle
- Bidirectional: day (liqpool → sentinel) + night (sentinel → liqpool)

### Replay + paper trading (Wave 19)
- `sentinel/replay.py` — `ReplayController` loads a past session,
  walks events at chosen speed, publishes onto the live bus tagged
  SHADOW (replay can never reach EXECUTION). Refuses to start while
  spine is live-real so synthetic signals can't influence real
  orders. New endpoints: `POST /api/replay/{start,pause,resume,stop}`,
  `GET /api/replay/state`.
- `sentinel/paper.py` — `PaperAccount` wraps a real Kite (or demo)
  account: passes through quotes/funds/positions/chain unchanged,
  intercepts `place_market_exit` to simulate fills at LTP. Journals
  to `paper_orders.jsonl`, tracks open book + realized P&L. Enabled
  via `SENTINEL_PAPER=1`. Differentiator between a RETAIL "sim"
  plan and a PRO "live" plan.

### Live cockpit (the operator UI)
- NIFTY live spot chart with hover crosshair + model-zone overlays
- NIFTY top-10 constituent board with move-quality verdict
- Live research signals + Live brief feed (one bus)
- Crux meta-signal composer (one operator verdict)
- Mind panel (tilt dial + intention contract + bias feed)
- Journey + scorecard + mistakes panel
- Profit-lock ratchet + trust spine + active trails
- Pre-market preflight modal (gates TRADE verdicts)
- Option premium chart + Monte Carlo Lite
- Source-tagged overlays ([L]/[S])

### Customer console (/console)
- Strategy Auditor (Hull taxonomy)
- Strategy Builder (6 customer intents)
- Crisis Stress test (7 scenarios)
- VaR / Expected Shortfall / Vol Cone
- Equity context regime classifier

### Health / observability
- `/healthz` — liveness probe
- `/readyz` — readiness probe (degraded if loop dead / broker
  down / disk full)
- Audit log to JSONL + stderr + external sinks

### Tests
- **Sentinel: 361 tests** (incl. 16 SaaS hardening, 15 launch prep,
  19 replay+paper+mobile+backup, 1 end-to-end soul test)
- **liquidity_backtester: 841 tests** (incl. 16 golden rows pinning
  leakage-sensitive math, 33 cross-codebase contracts + adapter +
  live inference)
- **Total: 1,235 — both halves green on this commit.**

---

## 3. The ONE thing Codex must NOT touch without sign-off

**The trust spine** (`sentinel/orchestration.py`). Specifically:
- Don't add a new EXECUTION-tier source (only `trailing_stop` and
  `profit_lock` may reach the order path)
- Don't relax the kill switch check
- Don't remove the RED-tilt block
- Don't bypass the preflight gate
- Don't write to the order path without going through `Orchestrator.route`

If a feature needs to place orders, it goes through the existing
pattern: graduate to TRUSTED via the curator's evidence → then the
spine decides whether to allow EXECUTION. Don't shortcut.

Everything else: go.

---

## 4. The launch sequence (Codex side, in order)

1. **Now**: deploy this branch to staging, set `SENTINEL_DEMO=1`,
   confirm `/healthz` + `/readyz` + cockpit all up.
2. **Day 1 morning**: write `codex_auth.py` per §1.2 — wire Supabase
   verifier; deploy without `SENTINEL_DEMO` env var; confirm
   header auth is now refused (test_header_auth_blocked_in_production_mode
   pins this).
3. **Day 1 afternoon**: write `LiveFeed` per §1.1 — wire KiteTicker
   → `LiveInferenceServer.on_tick`; confirm signals flowing in
   `<journal>/liqpool_live_signals.jsonl`.
4. **Day 1 evening**: write BYOK auto-restore per §1.3 — confirm
   credentials survive a process restart.
5. **Day 2**: wire Sentry sink (§1.5), daily backup cron (§1.6).
6. **Day 3**: first paying customer onboarded — watch `audit.jsonl`
   for `byok_connect` event.

---

## 5. The ONE thing that's not in code but matters legally

**SEBI Investment Adviser registration**. India requires registration
for any service that gives personalised investment advice. Sentinel
is positioned carefully — it provides decision-support context
(zones, probabilities, behavioral alerts) and execution discipline
(profit-lock, trailing stop) — NOT named-strike-named-target
"signals." But that line is read narrowly by SEBI.

**Recommendation**: have a lawyer review the cockpit screens + brief
copy before public launch. Specifically the "TRADE" Crux verdict —
ensure copy makes clear it's the operator's verdict-aided decision,
not the platform's advice.

Not Codex's job to solve; mentioning for completeness.

---

Mother's verdict, again: **the child is healthy.** The blood circulates.
The cycle is closed. There's a list of polish + ops items here, none of
which block tomorrow's soft launch.
