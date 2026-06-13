# Sentinel — Handoff Documentation (for Codex / any agent picking this up)

**Status as of 2026-06-13**
**Branch**: `claude/liquidity-pool-backtester-1uskb`
**Last commit**: `d7eb15b` — institutional analytics layer (SVI, VaR/ES, RV cone, Crux scores)
**Test status**: 98 passed (sentinel suite)
**Lines of code**: ~5,000 backend + ~1,300 tests

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

## 3. Module inventory (15 modules, all IO-declared)

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
| **`institutional.py`** | TRUSTED | **NEW — see §4. Named institutional methodologies.** |

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

Run:
```bash
python -m pytest sentinel/tests -q
```

---

## 6. What's still open (the tier-3/tier-4 roadmap)

These are the remaining items from the founder's full vision. Order is
suggested but not strict — pick the one with the highest commercial pull
for your sprint.

### 6.1 Top-10 NIFTY equity contextual layer

**Problem**: NIFTY moves because its top constituents move. RELIANCE,
HDFCBANK, ICICIBANK, INFY, TCS, etc. carry ~50% of the index weight. When
the index is flat but RELIANCE is +2%, the index is masking a bull move.

**Build**:
- `sentinel/equity_layer.py` with `top10_weightage_divergence()` returning
  contribution-to-NIFTY per stock vs raw price move per stock.
- Wire as a context flag the scientists can consult ("hidden bull" /
  "hidden bear" / "true flat").

### 6.2 Real crisis stress simulator

**Problem**: VaR/ES are statistical; a desk also wants "what happened to a
position like this on Demonetisation day?"

**Build**:
- `sentinel/stress.py` with canned scenarios for 2008 GFC, COVID Mar-2020,
  Flash Crash, Demonetisation Nov-2016, Yes Bank Mar-2020, Adani Jan-2023.
- Each scenario is a (spot_shock_pct, iv_shock_abs, timeframe) triple.
- `stress_test(legs, scenario_name)` returns the P&L decomposition under
  that shock.

### 6.3 Customer-facing strategy auditor

**Problem**: Customers paste their own option positions and want a
verdict: "is this a sound strategy?" Today the scenario engine can rate
ONE leg; multi-leg auditing is missing.

**Build**:
- `sentinel/auditor.py` with `audit_strategy(legs)` returning:
  - max profit / max loss / breakeven points
  - net delta/gamma/theta/vega
  - margin estimate
  - named-strategy detection (iron condor, bull call spread, etc.)
  - flags ("naked short call — uncapped loss", "negative theta in chop")

### 6.4 Multi-leg strategy builder

**Problem**: Customers want to ask "build me a delta-neutral premium-seller
for this expiry."

**Build**:
- `sentinel/strategy_builder.py` with `build(intent, chain, constraints)`
  picking legs to satisfy the intent. Intents: PREMIUM_SELL,
  DIRECTIONAL_BULL, DIRECTIONAL_BEAR, VOL_BUY, NEUTRAL_INCOME.

### 6.5 Wire institutional outputs into curator + reports

**Currently**: `institutional.py` ships, but `curator.py` doesn't yet
consult fair-value-vs-market when judging journeys, and
`reports.write_artifacts` doesn't surface RV cone / VaR / Crux scores in
`system_report.md`.

**Build**:
- Add to `CuratorReport`: `fair_value_misprice_pct` per journey, flagged
  when the actual entry was >5% above fair value.
- Add to `system_report.md`: a "Risk dashboard" section showing latest VaR,
  ES, vol-cone percentile, Crux liquidity for the watched book.

### 6.6 SaaS tier gating

**Currently**: every institutional function carries a RETAIL/PRO/QUANT tag
but nothing reads the tag.

**Build**:
- `sentinel/saas.py` with `gate(plan, function_name)` returning True/False.
- `server.py` endpoints check the calling user's plan before invoking
  PRO/QUANT methods; return 402 Payment Required otherwise.

### 6.7 UI (last)

When all the above is done: build the UI in `static/` against the FastAPI
endpoints. The architecture commitment is "backend first" — do not start
UI work while §6.1–6.6 are open.

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
