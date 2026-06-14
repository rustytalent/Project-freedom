# Sentinel — Handoff Documentation (for Codex / any agent picking this up)

**Status as of 2026-06-14 (Wave 17 — LAUNCH READY)**
**Branch**: `claude/liquidity-pool-backtester-1uskb`
**Latest pass**: Wave 17 — production-readiness pieces. New `auth.py` integration plug (Codex calls `register_verifier(supabase_verify)` once at boot; Sentinel never imports Supabase). `/healthz` + `/readyz` endpoints (LB-friendly). Preflight commitment modal + `/api/preflight/ack` endpoint that seeds the Ulysses-contract intention and **gates Crux TRADE verdicts until the operator commits**. Cockpit overlay toggles now show `[L]` or `[S]` so source provenance is one glance. `LAUNCH_README.md` documents env vars, run commands, the Codex integration contract, the launch-day operator flow, and the one thing Codex must NOT touch (the trust spine). Final end-to-end soul test (`test_launch_smoke.py`) walks the whole day cycle in one test through both halves of the project.
**Test status**: **Sentinel 326 + liquidity_backtester 841 + 33 cross-codebase = 1,200 passing — no regressions**.
**Codex green flag**: ✓ — see §3 in `LAUNCH_README.md`. The `register_verifier(fn)` plug-in is the ONE integration point; everything else (Google OAuth, Supabase tables, plan-lookup) lives in Codex's repo and never touches Sentinel internals.

---

## Older waves (archived)

**Status as of 2026-06-14 (Wave 15 — cross-codebase spine)**
**Branch**: `claude/liquidity-pool-backtester-1uskb`
**Latest pass**: Wave 15 — the founder's "Sentinel + liquidity_backtester must be tightly connected" ask. Built the missing shared contracts package + live inference path + bidirectional bridge so the two halves train each other every day/night cycle. The research engine now publishes live signals during market hours (via `liqpool/live_inference.py` → `JsonlPublisher`); Sentinel's cockpit tails that JSONL through a new `LiveSignalsTail` on every quote cycle and publishes each row as a TRUSTED `ModelSignal` on its own bus. The night trainer (`scripts/train_flywheel.py`) gained `--sentinel-journal` / `--sentinel-export` flags that feed Sentinel's exported decision ledger into the flywheel's shadow-frame pile so liqpool's seven organs (regret, trust, drift, archetypes, aging, transfer, meta_calibrator) train on the live decisions Sentinel produced.
**Test status**: **Sentinel 310 + liquidity_backtester 821 + 33 new integration = 1,164 total tests passing**
**Module count**: Sentinel 30 + new `liqpool.contracts` package (events/signals/promotions/research_context/manifest) + `liqpool.live_inference` + `liqpool.sentinel_adapter`
**Lines of code**: ~12,500 Sentinel + ~17k liquidity_backtester (+ shared contracts) + ~4,800 sentinel tests + 33 new integration tests

This document is the single source of truth for what Sentinel is, how it's
wired, what's done, and what's left. Read this before touching the code. It
exists so any agent (Codex, future Claude, or human) can resume work without
re-deriving the founder's vision from scratch.

---

## 1. What Sentinel is

Sentinel is a **standalone live trading copilot for Zerodha Kite Connect**,
targeting NIFTY/BANKNIFTY weekly options. It is NOT part of the
`liquidity_backtester` research codebase — it is a separate program that
ships to retail/pro/quant subscribers as a SaaS product.

### Three commitments that constrain every design decision

1. **Backend first.** No UI work until the backend can produce two internal
   report artifacts (`system_report.md` + `connection_map.{txt,dot}`) that
   tell the founder what is wired to what. UI is the last layer.
2. **Self-describing modules.** Every module calls `declare(IOSpec(...))` at
   import time so the wiring graph is mechanical, not hand-drawn. This is
   how we know the system is whole.
3. **Trust tiers — research never silently moves real money.** Signals are
   tagged SHADOW(0) → LOGGED(1) → TRUSTED(2) → EXECUTION(3). Only signals at
   EXECUTION tier reach an order. The Curator graduates a model upward only
   after it has earned it on the shadow ledger.

### Anti-greed psychology, mechanized

The product is built around the founder's observation that retail traders
lose because they (a) chase wins back down to losses, (b) widen stops to
avoid taking pain, and (c) re-trade revenge after losses. Sentinel
mechanizes the opposite:

- **Profit-lock ratchet** (`profit_lock.py`): locked profit rises with the
  realized peak and NEVER falls. Verified against the founder's numbers
  (peak 11500 → locked 6200, floating 1800, worst_case 6200).
- **Curator + calibration** (`curator.py` + `calibration.py`): re-run grid
  search over target/stop knobs on stored journeys; only accept a knob
  change if it improves expectancy on the actual stored data. No blind
  retraining.
- **Leakage guard** (`leakage_guard.py`): flags a hit-rate jump > 0.15 as
  SUSPICIOUS unless expectancy also improved (then EARNED).

---

## 2. Architectural pillars

### 2.1 The Shadow Ledger (the substrate AND the product)

`sentinel/shadow_ledger.py` — append-only JSONL with five blocks per event:

| Block      | What it stores                                             |
|------------|------------------------------------------------------------|
| identity   | instrument, option_type, strike, expiry, moneyness_key, spot, premium, delta |
| context    | regime, time-of-day, IV percentile, news flag, etc.        |
| hypothesis | expected move, target, stop, horizon, **mandatory reason_codes** |
| journey    | timestamped MFE/MAE/outcomes/time-to-target/time-to-stop   |
| judgment   | curator's verdict + grade + post-mortem                    |

Six event kinds:
`ACTUAL, VIRTUAL, REJECTED, ALTERNATIVE, MODEL_SUGGESTION, COUNTERFACTUAL`

**Invariant**: writing a Hypothesis without `reason_codes` raises at write
time. Every trade explains itself.

### 2.2 Moneyness-normalized identity (solves the Thursday-expiry problem)

`sentinel/moneyness.py` — keys are `(option_type, atm_offset)` rendered as
`"NIFTY:CE:+0"` / `"NIFTY:PE:-2"`. `+offset` always means *further OTM* for
both CE and PE. Behaviour observed on last week's `25000 CE` transfers to
this week's `25500 CE` because the moneyness key is the same.

This is what lets the system learn from history when strike levels reset
weekly — without it, Thursday afternoon would always be "no data."

### 2.3 Trust-tier spine

`sentinel/orchestration.py` — `Tier` enum + `Orchestrator.set_ceiling(name, tier)`
graduates a producer (scientist / model). Ungraduated EXECUTION requests are
clamped to TRUSTED. This is the wall between research and order placement.

### 2.4 The laboratory loop

```
scientists.py  (generate candidates + reason_codes)
      ↓
shadow_ledger.py  (stamp identity / context / hypothesis / journey)
      ↓
curator.py  (judge after >= 30 completed journeys → 'firm' findings)
      ↓
calibration.py  (grid-search knobs, RE-RUN stored journeys, accept only if expectancy improves)
      ↓
knobs.json  (persist tuned knobs; weights are NEVER touched here)
```

Knobs being tuned (`Knobs` dataclass):
`target_distance_mult, stop_distance_mult, entry_threshold,
holding_time_min, confidence_haircut, partial_booking_ratio,
trail_aggressiveness, strike_preference`.

Calibration grid:
- `TARGET_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`
- `STOP_GRID   = [1.0, 1.25, 1.5, 1.75, 2.0]`

### 2.5 IO self-declaration

`sentinel/io_decl.py` — every module ends with `declare(IOSpec(...))`. The
`REGISTRY` is the single source of truth for the connection map. Adding a
module without `declare(...)` means it doesn't appear on the map — that is
the enforcement mechanism.

`sentinel/reports.py` — `write_artifacts(journal, session, out_dir)`
produces:
- `system_report.md` — curator findings + self-assessment + knob state
- `connection_map.txt` — human-readable wiring
- `connection_map.dot` — Graphviz source for visual review

---

## 3. Module inventory (22 modules, all IO-declared)

| Module | Tier | Purpose |
|--------|------|---------|
| `greeks.py` | LOGGED | BS price + IV bisection + delta/gamma/theta/vega from live premiums (Kite doesn't ship Greeks) |
| `moneyness.py` | LOGGED | ATM±5 universe (22 contracts) + normalized identity keys |
| `shadow_ledger.py` | TRUSTED | Five-block append-only ledger; six event kinds; mandatory reason_codes |
| `orchestration.py` | TRUSTED | Trust-tier spine + signal routing |
| `profit_lock.py` | TRUSTED | Ratchet (locked never falls); ratio mode + buffer mode |
| `portfolio.py` | TRUSTED | Position state + net Greeks book |
| `trails.py` | TRUSTED | Trailing-stop engine wired to profit-lock |
| `advisor.py` | TRUSTED | Maximizer rules + dip recommender + self-scoring suggestion ledger |
| `kite_client.py` | EXECUTION | Kite Connect adapter with rate limits (1 q/s, 10 ord/s, 200/min, 3000/day) |
| `scientists.py` | LOGGED | 4 strategies: SecondPullback, GammaScalp, ChopMeanRev, ThetaGuard. All emit reason_codes. |
| `curator.py` | LOGGED | Per-session findings: target_distance_bias, stop_tightness_bias, atm_vs_otm, per-scientist hit-rates |
| `calibration.py` | TRUSTED | Re-run grid search; accept only if expectancy improves |
| `scenario_engine.py` | TRUSTED | `evaluate()`, `payoff_curve()`, `pnl_heatmap()`, `premium_sensitivity()`, risk waterfall |
| `leakage_guard.py` | TRUSTED | CLEAN / EARNED / SUSPICIOUS verdicts on metric jumps |
| `institutional.py` | TRUSTED | Named institutional methodologies — see §4. |
| **`stress.py`** | TRUSTED | **Wave 3** — 7 canned crisis scenarios, per-leg + portfolio P&L matrix |
| **`auditor.py`** | TRUSTED | **Wave 3** — multi-leg payoff curve + Hull-taxonomy detect + risk flags |
| **`strategy_builder.py`** | TRUSTED | **Wave 3** — 6 customer intents → legs, auto-audited |
| **`equity_layer.py`** | TRUSTED | **Wave 3** — NIFTY top-10 regime classifier (HIDDEN_BULL etc.) |
| **`saas.py`** | TRUSTED | **Wave 3** — RETAIL/PRO/QUANT/FOUNDER tier gates over the feature catalog |
| **`liqpool_bridge.py`** | TRUSTED | **Wave 4** — loads liquidity_backtester ResearchContextPack into `MarketSnapshot.context` |
| **`ledger_export.py`** | TRUSTED | **Wave 4** — Sentinel JSONL → DecisionEvent rows for the research-side flywheel |

Plus: `reports.py` (writes artifacts), `server.py` (FastAPI), `config.py`,
`io_decl.py` (the registry mechanism itself).

---

## 4. The institutional layer (last commit, d7eb15b)

`sentinel/institutional.py` is the **ground-anchored framework layer**. Every
function carries a named methodology + citation so a prop desk can audit
output on sight — no novel scores to interpret.

### What's in it

| Function | Methodology | Tier |
|----------|-------------|------|
| `SVIParams`, `fit_svi_slice()` | Gatheral (2004) raw SVI; scipy if present, deterministic grid+Nelder-Mead fallback | PRO |
| `svi_skew_25d()` | 25-delta risk-reversal / butterfly (industry FX/equity convention) | PRO |
| `fair_value_from_svi()` | SVI IV at log-moneyness, priced under BS-Merton (1973) | PRO |
| `rv_close_to_close()` | Textbook log-return std | RETAIL |
| `rv_parkinson()` | Parkinson, J. of Business 1980 | PRO |
| `rv_garman_klass()` | Garman & Klass, J. of Business 1980 | PRO |
| `rv_yang_zhang()` | Yang & Zhang, J. of Business 2000 (overnight-gap aware) | QUANT |
| `vol_cone()` + `vol_cone_percentile()` | VIX-cone (Burghardt & Lane, J. Derivatives 1990) | PRO |
| `vol_risk_premium()` | Bakshi & Kapadia, RFS 2003 (implied − realised) | RETAIL |
| `parametric_var()` | Cornish & Fisher (1937); BIS 1996 market-risk standard | PRO |
| `historical_var()` | Non-parametric VaR (BIS standard) | RETAIL |
| `expected_shortfall()` | Basel III replaced VaR with ES@0.975 for market risk | PRO |
| `portfolio_greek_exposures()` | Position-weighted Greeks (the "Greek book") | RETAIL |
| `crux_liquidity_score()` | Amihud 2002 + Almgren-Chriss spread-cost composite | PRO |
| `crux_slippage_score()` | Almgren-Chriss (2000) optimal execution impact | PRO |
| `roll_curve()` | Erb & Harvey, FAJ 2006 roll-yield | PRO |
| `_norm_inv()` | Beasley-Springer-Moro inverse normal (no scipy dep) | — |

### Why each component matters commercially

- **SVI + fair value** answers "is this strike cheap or rich?" with the
  exact methodology a prop desk uses.
- **RV estimators** triangulate true session vol (C2C overstates, Parkinson
  trims overnight noise, GK uses OHLC, YZ handles gaps).
- **VIX-cone percentile** tells customers "vol is at 78th percentile of
  60d history" — directly actionable for vol-selling/buying decisions.
- **VaR + ES** are what regulators and risk officers ask for. Without
  them, the product cannot sell into a desk.
- **Crux Liquidity / Slippage Scores** are sellable institutional scores
  branded as Sentinel's own composites but anchored on Amihud / Almgren-
  Chriss inputs — desks understand the inputs, customers understand the
  score.
- **Roll curve** is the standard contango/backwardation readout futures
  desks consume daily.

### SaaS tiers (the founder's nerf strategy)

Every function is tagged `RETAIL` / `PRO` / `QUANT` so the SaaS layer can
gate outputs by plan:
- **Retail**: C2C RV, historical VaR, vol risk premium, Greek book.
- **Pro**: + SVI, fair value, Parkinson/GK, vol cone, CF-VaR, ES, Crux scores, roll curve.
- **Quant**: + Yang-Zhang and raw SVI parameters for in-house surface work.

---

## 5. Test inventory (98 passing)

| File | Tests | Coverage |
|------|-------|----------|
| `test_bedrock.py` | 26 | Greeks, moneyness, shadow ledger, orchestration, profit-lock, portfolio |
| `test_core.py` | varied | Trails, advisor, kite-client mock, basic flow |
| `test_wave2.py` | 15 | Scientists pool, full lab loop (scientists → ledger → curator → calibration), scenario engine, leakage guard, IO map, reports |
| `test_trails_server.py` | varied | FastAPI server auth + state endpoint |
| `test_institutional.py` | 28 | SVI round-trip, RV estimator scales, vol-cone monotonicity, CF-VaR scale on N(0,1000), ES≥VaR, Crux composite weights, roll-curve regime labels, Beasley-Springer at standard quantiles |
| `test_wave3.py` | 36 | Stress matrix (BS + Greek proxy), Hull-taxonomy detection (long call, vertical spreads, straddles, condors, butterflies, custom), uncapped-loss flagging, regime-mismatch flagging, builder for 6 intents auto-detected by auditor, NIFTY top-10 regime classifier (HIDDEN_BULL etc.), SaaS gate matrix (RETAIL blocked from PRO, QUANT entitled to all, FOUNDER bypass, unknown plan denied) |
| `test_wave4.py` | 26 | Curator fair-value misprice (rich/healthy/no-resolver/per-event stamp), liqpool_bridge (empty / payload / latest fallback / corrupt tolerant / merge into context), ledger_export (kind mapping, DecisionEvent shape, JSONL roundtrip, empty session, None-field omission), FastAPI SaaS enforcement (/me/plan, /saas/catalog, /audit RETAIL vs PRO, /stress 402 + full matrix, /var historical vs CF, /vol_cone PRO-only, /equity_context scalar vs full, /build 400 on unknown intent) |
| `test_wave5.py` | 10 | Orchestrator in the hot path — trail exit routes through spine + places order, profit-lock fire routes via spine, ungraduated source requesting EXECUTION is clamped (no order), TRUSTED source surfaces but never executes, kill switch blocks execution at the spine, /api/graduate sets ceiling + refuses EXECUTION + rejects bad tier, /api/state exposes orchestrator stats, maximizer/dip_recommender graduated at startup |
| `test_wave6.py` | 6 | Canonical ledger — suggestion mirrors into ShadowLedger as model_suggestion (scientist=rule_id, hypothesis=None), resolution stamps canonical judgment, canonical event flows through ledger_export as MODEL_PREDICTION, no-shadow path is backward compatible, transient state (`_canonical_events`/`_post_peak`) drained on resolve (leak regression), WARN mirrors but never scores |
| `test_wave7.py` | 8 | trust_tier persistence (default SHADOW, explicit round-trip, suggestion=TRUSTED, flows through ledger_export), TrustPromotionRecord direction classification + evidence filtering, /api/graduate records + persists promotion to JSONL + surfaces in state, demotion recorded |
| `test_wave8_ui.py` | 10 | both pages served (cockpit has Trust-spine panel + console link; console references its endpoints incl. equity), console endpoints work end-to-end (audit→BULL_CALL_SPREAD with greeks, build→LONG_STRADDLE, stress matrix, VaR + vol cone), RETAIL→402 upsell shape, **`/sentinel.css` design system served with text/css + linked from both pages**, **cockpit ships command palette + status bar + market-hours chip + Trust-spine survives the rebuild**, **console ships equity tab + command palette + every institutional surface named on the tab strip** |

Run:
```bash
python -m pytest sentinel/tests -q
```

---

## 6. What's still open (the tier-3/tier-4 roadmap)

### 6.1 Top-10 NIFTY equity contextual layer ✅ DONE (Wave 3)

`sentinel/equity_layer.py` ships with `top10_contextual_layer(returns, idx)`
returning regime labels: HIDDEN_BULL / HIDDEN_BEAR / BROAD_BULL /
BROAD_BEAR / TRUE_FLAT, plus a scalar `weightage_divergence()`. Canned
Q2-2026 NIFTY top-10 weights with refresh-date stamp. **Next:** wire it
into `MarketSnapshot.context` so scientists read regime directly.

### 6.2 Real crisis stress simulator ✅ DONE (Wave 3)

`sentinel/stress.py` ships with 7 canned scenarios: GFC_OCT_2008,
COVID_MAR_2020, FLASH_CRASH_2010, DEMONETISATION_2016, YES_BANK_MAR_2020,
ADANI_JAN_2023, US_DOWNGRADE_2011. Each carries (spot_shock, iv_shock,
timeframe, narrative). Two entry points: `stress_test(legs, spot,
scenario)` for one shock, `stress_test_all(legs, spot)` for the full
matrix sorted worst-first. Supports BS re-price path when strikes +
option_types provided, parametric Greek proxy otherwise.

### 6.3 Customer-facing strategy auditor ✅ DONE (Wave 3)

`sentinel/auditor.py` ships with `audit_strategy(legs, spot, regime)`
returning: detected named strategy (Hull 11e taxonomy — LONG_CALL,
BULL_CALL_SPREAD, IRON_CONDOR, IRON_BUTTERFLY, CALL_BUTTERFLY etc.),
payoff curve, max profit / max loss, linear-interpolated breakevens, net
Greeks (delegating to `institutional.portfolio_greek_exposures`), and
flags for uncapped loss / negative-theta-in-chop / short-gamma-in-
trending / vega-short-delta-neutral.

### 6.4 Multi-leg strategy builder ✅ DONE (Wave 3)

`sentinel/strategy_builder.py` ships with `build(intent, spot, chain)`
for 6 intents: PREMIUM_SELL_NEUTRAL (iron condor at ±2σ wings),
PREMIUM_SELL_INCOME (short strangle), DIRECTIONAL_BULL (bull call
spread), DIRECTIONAL_BEAR (bear put spread), VOL_BUY (long straddle),
NEUTRAL_INCOME (iron butterfly). Picks legs from the supplied chain,
falls back to closest strike when target is missing (recorded in
`missing_strikes`), then runs them through the auditor automatically so
the customer sees the verdict alongside the build.

### 6.5 Wire institutional outputs into reports + curator ✅ DONE (Wave 3 + 4)

Wave 3: `reports.generate_system_report` section 8 lists every named
methodology, stress scenarios, auditor flags, builder intents, equity-
layer regimes, and the SaaS catalog count.

Wave 4: `Curator(fair_value_resolver=..., misprice_threshold=0.05)`
emits a `fair_value_misprice_bias` finding when entries deviate
systematically from a supplied fair-value oracle (typically
`institutional.fair_value_from_svi`). The resolver is callable-based so
the curator stays decoupled from the institutional module — the wiring
lives at the caller (server / nightly job). Per-event judgment block
also carries `misprice_vs_fair_value_pct` for granular debugging.

### 6.6 SaaS tier gating ✅ DONE (Wave 3 + 4)

Wave 3: `sentinel/saas.py` ships with FEATURE_CATALOG, `gate()`,
`features_for()`, FOUNDER bypass, and a safer-PRO default for unknown
features.

Wave 4: `sentinel/server.py` enforces it. New endpoints all behind
`Depends(auth) + Depends(resolve_plan)`:

- `GET  /api/me/plan` — entitlement summary.
- `GET  /api/saas/catalog` — feature catalog with `allowed: bool` per row.
- `POST /api/audit` — strategy auditor; greek_book + risk_flags stripped
  for RETAIL, included for PRO+.
- `POST /api/build` — strategy builder; RETAIL gets one build per session
  via `builder.one_per_session`, PRO+ gets unlimited.
- `POST /api/stress` — single scenario (PRO) or full matrix (PRO).
- `POST /api/equity_context` — scalar at RETAIL, full pack at PRO.
- `POST /api/var` — historical (RETAIL), CF (PRO), ES (PRO).
- `POST /api/vol_cone` — PRO.

Plan resolves from `X-Sentinel-Plan` header (defaults to FOUNDER for the
user's own view). 402 Payment Required carries `{feature, required_tier,
your_tier, reason}` so the client can show an actionable upsell.

### 6.7 UI 🟢 INSTITUTIONAL-TIER (Wave 8 + 8b)

Backend-first commitment honoured — UI work started only after §6.5/§6.6
landed. Both pages now share `static/sentinel.css` (a deliberate
design system, served at `/sentinel.css`) modelled on Bloomberg
Terminal / FactSet / Refinitiv Eikon conventions: deep-charcoal
surface, amber/copper accents (never neon), Inter for chrome +
JetBrains Mono for every number, tabular-nums everywhere, dense panel
spacing, sticky table headers, tier-coloured chips, restrained
greens/reds.

**Cross-cutting UX primitives (both pages):**
- Top bar with brand mark, mode chip, **NSE market-hours chip** (open
  09:15–15:30 IST Mon-Fri, live-tracked), **IST clock**, and tab nav.
- **Bottom status bar** with mode, session date, ledger pending, spine
  routed + clamped counts, last-poll timestamp, and `⌘K` hint.
- **Command palette (⌘K)** with fuzzy search across actions, tabs,
  plan switches, and (cockpit) every open position by symbol; arrow-key
  navigation, ⏎ to run.
- **Keyboard shortcuts**: `⌘K` palette, `1`–`5` switch tabs on console,
  `g c` jumps to console / `g k` kill switch on cockpit, `?` help.
- **Lock cards** for 402-gated features with a one-click upgrade to
  the next plan tier.
- Skeleton loaders instead of blank "—" while data is in flight.

**`static/index.html` — operator cockpit** (rebuilt). Polls `/api/state`
every 2s. Panels: Positions (with inline trail arming), Scenario curve
(real chart with grid lines, axis labels, gradient area-fill, breakeven
diamonds, `what-if` slider), Recommendations, **Profit-lock ratchet**
(4-cell metric strip), **Trust spine** (routed-by-tier cells, source
ceilings as tier-coloured chips, recent `TrustPromotionRecord`s, a
graduate control), Active Trails, Suggestions, Pairs, Activity feed.

**`static/console.html` — customer SaaS console** (rebuilt) at
`GET /console`. Five tabs:
- *Strategy Auditor* → `/api/audit` — verdict block (named strategy in
  amber serif-weight), 3-cell max P/L + breakevens, grid-lined payoff
  chart with spot guide, Greek book table + risk-flag cards (PRO).
- *Strategy Builder* → `/api/build` — intent picker; result shows
  intent → detected name → ±1σ band, legs table + summary cells, all
  audit flags.
- *Crisis Stress* → `/api/stress` — full matrix with horizontal **bullet
  bars** scaled to worst-case loss; rows tooltipped with the scenario
  narrative.
- *Risk · VaR & Vol Cone* → `/api/var` + `/api/vol_cone` — VaR cells +
  skew/kurt readouts; vol-cone SVG with amber bars (p10-p90), median
  dots, steel rings for current.
- *Equity Context* → `/api/equity_context` — top-10 input grid, big
  regime headline, summed contribution, breadth split, full
  per-stock weight/return/contribution/residual table on PRO; scalar
  divergence + upsell on RETAIL.

**Still open on UI**: real auth/login + signed plan-token (today the
plan is a header the user picks — fine for cockpit/demo, needs a JWT
for real customers); wiring the builder to the live chain in
production; mobile polish; per-user layout persistence.

### 6.8 Codex's spine items (from the unified architecture report)

Codex's `opus_unified_sentinel_liquidity_architecture_report.md` lists a
cross-codebase queue. **Wave 4 closed the two ledger-side items**:

- **Research context bridge** ✅ **DONE**
  `sentinel/liqpool_bridge.py` reads the latest research artifacts from
  `liquidity_backtester/runs/<session>/{research_context,daily_brief}.json`
  (or `runs/latest/`), normalises into a `ResearchContextPack`
  (direction/reaction probabilities, proximity h12/h36/h60, sector
  regime, avoid flags, key zones), and `merge_into_snapshot_context()`
  populates the scientist-visible context. Tolerates corrupt payloads
  and missing files — Sentinel runs without today's brief.

- **Live-to-research adapter** ✅ **DONE**
  `sentinel/ledger_export.py` exports the Sentinel JSONL shadow ledger
  to a `DecisionEvent`-shaped JSONL/parquet under
  `<out>/sentinel_live/session=<...>/decision_events.{jsonl,parquet}`.
  Sentinel kinds map to the shared vocabulary: ACTUAL→EXECUTED,
  VIRTUAL→VIRTUAL_DECISION, REJECTED→REJECTED_CANDIDATE etc. CLI:
  `python -m sentinel.ledger_export --session 2026-06-13`. Idempotent
  by event_id so re-running is safe.

- **Orchestrator in the hot path** ✅ **DONE (Wave 5)**
  `sentinel.server` now routes every order placement through
  `orchestration.Orchestrator`. The trail engine's injected `exit_fn`
  is `Sentinel._trail_exit`, which wraps the exit as a
  `Signal(source="trailing_stop", tier=EXECUTION, kind="exit")` and
  routes it; the profit-lock fire path routes `source="profit_lock"`.
  Three sinks: `_sig_ledger` (records every routed signal — exposed at
  `/api/state.routed_signals`), `_sig_surface` (operator-facing
  TRUSTED+), `_sig_execute` (the ONE gated path to
  `place_market_exit`). `maximizer` + `dip_recommender` are graduated
  to TRUSTED at startup so they surface but can never execute. New
  `POST /api/graduate` is the curator's lever (refuses EXECUTION for
  any source — reserved for the two exit actors). Kill switch flips
  `orchestrator.allow_execution = False` so even a hard-wired actor is
  blocked at the spine. `/api/state` exposes `orchestrator.stats()`.

- **trust_tier on persisted events** ✅ **DONE (Wave 7)**
  `LedgerEvent.trust_tier` (default "SHADOW") is persisted in `to_row()`
  and flows through `ledger_export.DecisionEvent.trust_tier`. Scientist
  hypotheses stay SHADOW; canonicalised suggestions are TRUSTED. Matches
  Codex's `DecisionEvent.trust_tier`.
- **TrustPromotionRecord** ✅ **DONE (Wave 7)**
  `orchestration.TrustPromotionRecord` (Codex §5.1) — source, from/to
  tier, decision (promoted/demoted/held), ts, and curator evidence
  (evidence_window, n, hit_rate, expectancy, drawdown, calibration_error,
  leakage_status). `/api/graduate` builds one per call, persists to
  `<journal>/trust_promotions.jsonl`, and surfaces recent ones in
  `/api/state.promotions`. The endpoint accepts optional evidence fields.

**Still open from Codex's queue** (cross-repo — need both codebases):

- **Shared event spine schema** — formalise `DecisionEvent` /
  `ExecutionIntent` / `OutcomeEvent` as a versioned package both
  codebases depend on (currently each side has its own dataclasses
  that agree by convention; `ledger_export.DecisionEvent` is the
  Sentinel-side shape and already matches Codex's proposed fields).
- **Canonical ledger** ✅ **DONE (Wave 6)**
  `SuggestionLedger(shadow_ledger=..., session=...)` mirrors each
  suggestion into the `ShadowLedger` as a `model_suggestion` event
  (`scientist=rule_id`, contract identity, `hypothesis=None` — an
  advisory predicts no target/stop, so no fabricated price hypothesis).
  Resolution stamps the canonical event's judgment + completes its
  journey, so the curator / `ledger_export` / flywheel see the outcome.
  The `suggestions.jsonl` it still writes is now a backward-compatible
  projection, not the system of record. `server.py` constructs one
  shared `ShadowLedger(cfg.journal_dir)` and passes it in. Backward
  compatible: omit `shadow_ledger` and it behaves exactly as before.
- **liquidity_backtester side**: `ShadowLogger` + `FlywheelHub` wired
  into the standard Daily Brief run; `multi_asset_run.py` broken up;
  artifact-health gate enforced.

---

## 7. Reading order for a new agent

If you've just been handed this codebase:

1. `sentinel/docs/SENTINEL_ARCHITECTURE.md` — founder's spec, the why.
2. **This file** — what's done, what's left, how the pieces wire.
3. `sentinel/io_decl.py` (~80 lines) — how self-declaration works.
4. `sentinel/shadow_ledger.py` — the substrate; every other module reads
   or writes to it.
5. `sentinel/orchestration.py` — the trust-tier wall.
6. `sentinel/scientists.py` → `curator.py` → `calibration.py` — the
   laboratory loop end-to-end.
7. `sentinel/institutional.py` — the named-methodology pack.
8. `sentinel/tests/test_wave2.py` — best single file to see the loop
   running end-to-end.

Then write code.

---

## 8. Non-negotiables (do not violate)

1. **No `reason_codes` → no write to ledger.** The check lives in
   `shadow_ledger.write()`. Don't bypass it.
2. **No blind retraining of weights inside `calibration.py`.** Calibration
   tunes knobs. Weight retraining is a deliberate overnight job.
3. **No EXECUTION-tier signal from an ungraduated producer.** The
   `Orchestrator` clamps it; don't add a backdoor.
4. **No new module without `declare(IOSpec(...))`.** If it isn't on the
   map, it doesn't exist.
5. **No UI work before §6.5 is done.** Backend first; this is the founder's
   explicit ordering.
6. **No scipy as a hard dependency.** Everything has a deterministic
   fallback (see SVI fit, inverse normal). Keep this property.
7. **No silent compute on real money.** If the live trading loop calls a
   new function, that function's outputs must be loggable to the shadow
   ledger.

---

## 9. Local dev quickstart

```bash
# Run the full sentinel suite
python -m pytest sentinel/tests -q

# Write the two report artifacts
python -c "
from sentinel.reports import write_artifacts
from pathlib import Path
write_artifacts(Path('/tmp/journal'), '2026-06-13', Path('/tmp/sentinel_reports'))
"

# Start the FastAPI server (dev)
uvicorn sentinel.server:app --reload --port 8000

# Render the connection map as a PNG
python -c "
from sentinel.reports import connection_map
Path('/tmp/conn.dot').write_text(connection_map()['dot'])
"
dot -Tpng /tmp/conn.dot -o /tmp/conn.png
```

---

## 10. Where to ask questions

- Founder's intent on a design call: re-read
  `sentinel/docs/SENTINEL_ARCHITECTURE.md`.
- Wiring: render the connection map (`reports.connection_map()`).
- Test failures: `pytest -q --tb=short` then read the failing test name —
  every test name describes the invariant being checked.
- Anything unclear about WHY a module exists: that module's top docstring
  carries the founder's reasoning verbatim.
