# Sentinel — live trading copilot

A standalone program for the VPS. Watches your live Zerodha account,
computes Greeks from first principles, trails your winners, locks your
day's profit at the portfolio level, recommends dip candidates, and
scores its own suggestions — behind one professional dashboard.

**It never decides entries. It never trades on its own.** It acts only
on trails you explicitly arm, and everything else is decision support.

---

## Quick start

```bash
pip install fastapi uvicorn kiteconnect

# DEMO (no credentials, works on weekends — synthetic ticking account)
SENTINEL_DEMO=1 uvicorn sentinel.server:app --port 8800

# LIVE, dry-run (real account data, simulated orders)
export KITE_API_KEY=...  KITE_ACCESS_TOKEN=...
uvicorn sentinel.server:app --host 127.0.0.1 --port 8800

# LIVE, real orders (the only mode that sends orders)
SENTINEL_CONFIRM_REAL=1 uvicorn sentinel.server:app --host 127.0.0.1 --port 8800
```

Open http://localhost:8800 — that's the dashboard.

Set `SENTINEL_TOKEN=<secret>` to require a token on every API call
(the dashboard prompts once and remembers). **Always set this on a
VPS.** Run behind nginx/caddy with TLS if exposed beyond localhost;
prefer an SSH tunnel (`ssh -L 8800:localhost:8800 vps`) instead of
exposing at all.

## What's on the dashboard

| Panel | What it shows |
|---|---|
| Header KPIs | cash, utilised margin, live day P&L, session peak, spot |
| Positions | every leg with live IV, delta, theta/day (computed from premiums via Black-Scholes implied vol — Kite provides no Greeks), session peak P&L and drawdown-from-peak per leg, one-click trail arming with a rupee cushion |
| P&L vs spot curve | the whole book repriced across ±2% spot moves at constant IV; breakeven spots marked ◆ — "above this level profit, below this loss" |
| What-if slider | drag ±300 points; every leg shows premium-at-move and ΔP&L |
| Portfolio lock | protect TOTAL day P&L: give-back ≥ cushion from session peak → flatten everything at market |
| Active trails | live peak/last/cushion per armed trail, cancel buttons |
| Suggestions | rule-based calls (arm-trail / close / tighten / warn), each with its rule's live hit-rate from the self-scoring ledger; low-accuracy rules shown MUTED |
| Recommendations | the dip scanner: direction view (auto from spot momentum, or set UP/DOWN manually) → which options fell hardest from day high, near the money, liquid — with every score component visible |
| Pairs | CE+PE pairs on the same underlying/expiry, combined P&L, net delta, which leg is funding which |
| Activity log | every exit, arm, cancel, error with timestamps |
| KILL | cancels all trails, refuses all orders until restart |

## The self-scoring ledger (the learning seed)

Every suggestion is journalled with the premium at suggestion time.
Ten minutes later it's scored against the live premium:

- CLOSE/TIGHTEN → right if premium fell afterward
- ARM_TRAIL → right if a ≥20% give-back from the post-suggestion peak occurred

Per-rule EWMA hit-rates persist across restarts and feed back into
the display: rules under 45% accuracy are flagged MUTED (visible but
demoted). This is v1 of "the system checks itself at T+10 and
improves" — honest, simple, inspectable.

## Kite Connect limits Sentinel respects

Encoded in `kite_client.RateLimiter` (VERIFY against your plan; the
docs site blocks automated fetching):

| Endpoint class | Limit | Sentinel's usage |
|---|---|---|
| quote/ltp | 1 req/s | one batched call/sec (≤500 instruments per call) |
| orders | 10 req/s, 200/min, 3000/day | self-capped at **50/day**, 180/min guard |
| everything else | 10 req/s | positions/funds every 15s |
| access token | expires daily ~6 AM | re-login each morning (see below) |
| WebSocket | 3 conn/key, 3000 instr/conn | not used in v1 (polling); v2 upgrade |

Morning token ritual (same as the research repo):
```bash
python liquidity_backtester/examples/zerodha_login.py   # paste request_token
export KITE_ACCESS_TOKEN=<fresh token>
```

## VPS requirements

Sentinel alone is tiny: it makes ~1-2 HTTP calls/sec and holds a few
MB of state. **2 vCPU / 4 GB RAM / 20 GB disk is plenty** (₹400-800/mo).

Recommended layout:
- **Trading box (small, isolated)**: Sentinel only. Nothing else
  competes for CPU when an exit order needs to fire.
- **Research box (your existing 16-core/32GB)**: the liqpool engine,
  training, backtests. Sentinel's journals can sync here nightly to
  feed the shadow-log flywheel.

Do not run Sentinel on the research box while a 16-worker backtest is
hammering it — order latency is the one thing that must never queue.

## Safety model (read once, remember forever)

1. Dry-run default. `SENTINEL_CONFIRM_REAL=1` is the only path to
   real orders.
2. Self-capped at 50 orders/day, 180/min — far under Kite's caps.
3. Bad quotes never fire a trail and never corrupt a peak.
4. Idempotent exits — one fire per trail, ever.
5. Journals (`~/.sentinel/`) — crash recovery resumes armed trails.
6. Auto-flatten 15:14 IST — before broker MIS square-off.
7. KILL switch — instant, total, until restart.
8. Suggestions and recommendations are **decision support**. The only
   orders Sentinel ever places are exits for trails *you* armed and
   the portfolio lock *you* armed.

## Architecture

```
sentinel/
  config.py       env-driven config; multi-account-ready (accounts list)
  kite_client.py  rate limiter + KiteAccount (REST) + DemoAccount
  greeks.py       BS price, implied vol (bisection), delta/gamma/theta/vega
  portfolio.py    enriched positions, CE+PE pair detection,
                  scenario curve + breakevens, what-if engine
  trails.py       tick-driven trail engine + portfolio-level lock
  advisor.py      maximizer rules (R1-R5) + dip recommender +
                  self-scoring suggestion ledger
  server.py       FastAPI app + the live loop (quotes 1s / portfolio 15s /
                  ledger 30s / recommendations 60s)
  static/index.html   the dashboard (vanilla JS, zero build step)
  tests/          29 tests; demo mode makes the whole stack testable
```

## Roadmap (deliberate v1 omissions)

- **WebSocket ticks** (KiteTicker) — sub-second trail reaction; v1
  polls at 1/s which is adequate for rupee-cushion trails.
- **Multi-account** — config already takes a list; the dashboard
  needs an account switcher + aggregate view. This is the fund-manager
  view (two accounts on one screen).
- **liqpool bridge** — feed the research engine's trained organs
  (proximity/valuation heads, flywheel hub) into the suggestions
  panel; Sentinel's journals already produce shadow-log-shaped data
  to train on.
- **Partial exits / scale-out trails** — needs the operator to define
  the rule they actually want.
- **SaaS multi-tenancy** — auth, per-user accounts, billing. The
  product thesis; not a weekend.
