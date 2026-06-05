# MASTER PLAN

> The canonical operating document for the entire project. Both Opus and
> Codex read this first on every session. This supersedes earlier
> roadmap/coordination/system-overview docs as the single source of truth
> for *what we're doing, why, in parallel, and where we deviated from the
> original plan*.
>
> The earlier docs (`COMPANY_MAP.md`, `COORDINATION.md`, `RESEARCH_ROADMAP.md`,
> `SYSTEM_OVERVIEW.md`) remain — they are reference material this document
> points at. But the **plan lives here.**
>
> Maintenance rule: update §5 (Deviation Log) on every commit. Update §3
> (Streams) when a stream's status changes. Update §2 (Layer Matrix) when
> a layer's contents change. Never delete; mark stale entries as
> SUPERSEDED with a pointer to the replacement.

---

## §0. Orientation — read this if you're a fresh session

You are working on a market-intelligence operating system for NSE Indian
equities, indexes, and options. The technical infrastructure is tier-3
quant-research-grade. The commercial product (Daily Research Brief) is
days from first pilot customers.

**The single most important thing to internalize about this project's
state:** we are NOT building sequentially anymore. There are 8 parallel
work streams (§3), each scoped to one or more layers (§2), each with its
own owner, dependencies, and acceptance criteria. Read §3 to find the
stream you're picking up.

**The single most important methodological discipline:** every result
that surprises us produces a *deviation* from the original plan. Those
deviations are where the breakthroughs happen — but they're also where
we lose track. So §5 logs every deviation as a tree-node with cause,
finding, impact. If you change direction, you log it there or it doesn't
count.

**Who owns what:**
- **Opus** (Claude) — strategy, design, methodology rulings, small
  self-contained code, documentation. Operates in cloud sessions.
- **Codex** — primary implementer, retrains, long-running VPS sweeps,
  multi-file refactors. Operates on a separate machine with VPS access.
- **User** — architect, customer-channel owner, final decision on
  product/strategy/pricing tradeoffs, dispatcher.

**Current branch-hygiene status:** SUPERSEDED 2026-06-05. The former
Stream B chokepoint is resolved on the remote branch. Codex's four
audit-patch batches were rebased and pushed as `52c88df`, `ce1ed40`,
`a876351`, and `ab7d57d`; local `HEAD` and
`origin/claude/liquidity-pool-backtester-1uskb` both point at
`ab7d57d`. Work that was blocked only by missing audit patches is now
unblocked, but implementation/training acceptance criteria still apply.

---

## §1. The Architecture (one-page summary)

### Two axes

**Axis 1 — The Stack (necessity chain, bottom → top):**

```
L0  Data
L1  Features
L2  Models
L3  Signals / Alpha
L4  Execution & Validation
L5  Audit
L6  Products
L7  Customers
```

Each layer was born because the layer below couldn't answer the
question alone. Infrastructure is the orthogonal spine that holds
L0–L5 upright (training pipeline, parallel workers, bundle persistence,
report generation).

**Axis 2 — The Horizon / Instrument (same engine, different (L, T)):**

| Horizon | Question | Customer | State |
|---|---|---|---|
| Swing (days–weeks) | "Will price reach L by end-of-week?" | Positional, option-buyers | Engine ready; presentation not built |
| Intraday MIS (minutes–hours) | "Will price reach L same-session?" | Day-traders | Engine + brief built |
| Index options (T = expiry, L = strike) | "Will Nifty/BankNifty test strike K before expiry?" | Our pilot customers | Engine ready; level-to-strike translator built; Greeks dataset ready |

### The Eight Parallel Streams (§3 has the detail)

| Stream | What | Layers | Owner | Status |
|---|---|---|---|---|
| **A** | Rupee-sizing experiment | L4 | Codex | unblocked, queued |
| **B** | Audit-patch branch hygiene | L0–L5 | Codex + user | ✅ RESOLVED — pushed/reconciled at `ab7d57d` |
| **C** | Detector batch v1 (liquidity sweep, stop-run-reclaim, etc.) | L1 | Opus | ✅ done; importance verification on next retrain |
| **D** | Options vertical slice (Greeks features → model → brief) | L1+L2+L6 | Opus design, Codex training | design done; training now unblocked/queued |
| **E** | Live broker hardening (order state machine) | L4 | Codex | partially hardened; full order lifecycle still queued |
| **F** | Swing vertical slice (proximity at 5d/10d/20d + swing brief) | L2+L6 | Opus design, Codex training | design done; training now unblocked/queued |
| **G** | Yesterday-Audit retrospective flag | L5+L6 | Opus | ✅ done |
| **H** | Distribution + first 3 customers | L7 | User | unblocked, independent |

---

## §2. Layer × Status Matrix

Detailed inventory of every idea ever raised, mapped to its layer, with
current status. This is the truth table for "what exists, what's queued,
what was declined."

### L0 — Data

| Item | Status | Source | Stream |
|---|---|---|---|
| 5 indexes spot OHLCV 3y (NIFTY50, BANKNIFTY, FINNIFTY, MIDCPNIFTY, INDIAVIX as 5th) | ✅ in Kite warehouse | this week | done |
| 25 equities 1-min 3y | ✅ warehouse | this week | done |
| India VIX 3y intraday + 5y daily | ✅ warehouse | this week | done |
| Active futures + options 5-min + OI | ✅ warehouse | this week | done |
| Bhavcopy EOD 3y all strikes | ✅ warehouse (since 2024-01-01) | this week | done |
| Macro / risk-free rate daily | ✅ warehouse (7% flat for v1) | this week | done |
| Live tick + 5-level depth WebSocket capture | scoped, not running | brain-dump moat | future |
| F&O expiry calendar with NSE holiday shift | audit P0 flag | audit | C/B |
| Corporate-action / split-adjustment metadata | audit P0 flag | audit | C |
| Session-anchored higher-TF resampling (60m/180m) | audit P0 flag | audit | C |
| Sectoral indices feed (Nifty Auto/Bank/IT/FMCG/Pharma) | mentioned, deferred | brain-dump | future |
| RBI MPC/policy calendar scrape | mentioned, deferred | brain-dump | future |

### L1 — Features

| Item | Status | Notes |
|---|---|---|
| Base 17 features (returns, momentum, z-score, ATR, pull) | ✅ shipped | original |
| 5 MTF session-context features | ✅ shipped | this week |
| 9 AVWAP/FRVP volume features | ✅ shipped | this week |
| 4 expiry-cycle features | ✅ shipped | this week |
| 7 path/shape features (trap, efficiency, vol regime) | ✅ shipped | this week |
| **Liquidity-sweep detector** | scoped T0.3 | Stream C |
| **Stop-run + reclaim pattern detector** | scoped T0.3 + T1.1 | Stream C |
| **Multi-bar imbalance detector** | scoped T0.3 | Stream C |
| **Premium/discount mid (50% retracement of impulse)** | scoped T0.3 | Stream C |
| **Volume-weighted swing levels** | scoped T0.3 | Stream C |
| **Cumulative-delta-divergence proxy** | scoped T0.3 | Stream C |
| **Asian/overnight range markers** | scoped T0.3 | Stream C |
| **Sector context per stock** | partial in sectors.py | Stream C+F |
| **Cross-asset correlation features** | brain-dump, not built | future |
| **Daily Greeks → intraday lag feature** | warehouse helper ready | Stream D |
| **Continuous (vs snapshot) every bar** | T1.4, cost-dependent | future |
| **Path-along-route features** (proximity at each pool on the journey) | T1.1, scoped | Stream D parallel |
| **Audit-flagged: no-bfill warmups** | warehouse helper exists, not threaded everywhere | piecemeal |

### L2 — Models

| Item | Status | AUC / metric |
|---|---|---|
| Q (SectorMoE) | ✅ trained | val AUC 0.541 — demoted to context |
| Direction (60-bar) | ✅ trained | OOS AUC 0.567, top-quartile 58.8% |
| Proximity ×3 (h=12/36/60) | ✅ trained | AUC 0.92-0.95 overall; 0.73-0.76 at 0-1 ATR |
| Reaction × 3 sub-targets | ✅ trained | strict 0.78, reclaim 0.73, break 0.78 |
| R1 policy return regressor | ✅ trained | spearman 0.16-0.32 across 4 modes |
| Time-decay sample weighting | ✅ shipped (opt-in) | Config default flip pending |
| **Q multi-class target** (5-class vs binary) | T1.3 scoped; unblocked, not prioritized | future |
| **Per-factor Q sub-models** (EQHL/FVG/OB/REJ) | T1.3 scoped; unblocked, not prioritized | future |
| **Manipulation-aware direction model** | T1.1 scoped | Stream D |
| **Sub-alpha library** (momentum.with_volume vs .fake_breakout) | T1.2 scoped | future |
| **OptionsExpectedReturnModel** | design ✅ in `6f9ffd8`; training/wiring not built | Stream D |
| **Swing-horizon proximity** (5d/10d/20d) | design ✅ in `6f9ffd8`; training not built | Stream F |
| **Q on V2 notional labels** | ✅ pushed as `ce1ed40`; retrain verification still required | Stream B resolved |

### L3 — Signals / Alpha

| Item | Status |
|---|---|
| pool_reach, mean_reversion, momentum (baselines) | ✅ |
| quality_filtered_pool, direction_confirmed_pool | ✅ |
| proximity_journey + proximity_journey_baseline (attribution) | ✅ |
| distance_5_8_journey, proximity_direction_soft, opening_range_to_pool, sector_rotation_journey | ✅ in research_registry (sparse) |
| Avoidance alpha as first-class | partial in brief | Stream G |
| **Options scoring alpha** (uses OptionsExpectedReturnModel) | design ✅; implementation not built | Stream D |
| **Swing alphas** | design ✅; implementation not built | Stream F |
| **Path-dependent alpha** | T1.1 | Stream D parallel |
| **Aqua-regia mixer** (regime-conditional alpha blend) | T2.1, post-edge | future |

### L4 — Execution & Validation

| Item | Status |
|---|---|
| V1 + V2 simulators | ✅ |
| `min_target_to_cost_ratio` filter (3×) | ✅ |
| Geometry sweep (post-touch declined) | ✅ ran, declined |
| Sweep_reclaim mode | ✅ shipped, declined |
| **`RupeeTargetExecutionConfig`** (₹600 floor, variable qty) | new today, NOT built | Stream A |
| **Scale-out execution layer** (25/50/75 partial profit, options) | new today, NOT built | Stream D |
| **Live broker order state machine** | partial; audit P1 | Stream E |
| **Live broker fail-closed** | ✅ pushed as `52c88df`; full state machine still open | Stream E |
| **Live default-quantity required** | ✅ pushed as `52c88df` | Stream E |
| **V2 ≡ Arsenal evaluator equivalence proof** | audit gap | future test commit |
| **1-minute replay for triple-barrier** | audit P0 | future |
| **Slippage extraction as dataset** | T4.1 moat | future |

### L5 — Audit

| Item | Status |
|---|---|
| OOS prediction audit, calibration, distance buckets | ✅ |
| Phase 3C leakage replay (noise-reduced) | ✅ |
| 4 null tests | ✅ |
| Outcome log writer (1260 rows logged) | ✅ |
| Outcome log atomic-write + dedup | ✅ pushed as `52c88df` | Stream B resolved |
| Backfill ordering fix | ✅ pushed as `ce1ed40` | Stream B resolved |
| Drift monitor for h=12/36/60 | ✅ pushed as `52c88df` | Stream B resolved |
| **`is_retrospective` flag on backfilled predictions** | ✅ shipped in `8c4f210` | Stream G done |
| **Null tests on same evaluation slice as final claim** | audit P1 | future |
| **DSR in reports** | underused | future |
| **Moment null** | spec'd, not run | future |

### L6 — Products

| Item | Status |
|---|---|
| Daily Research Brief (equity-intraday v1) | ✅ shipped, integrated |
| Sample brief artifact | ✅ shipped |
| Strategy Diagnosis spec | ✅ shipped |
| 5-index options-suitability section | ✅ skeleton; needs Greeks-wire | Stream D |
| **Daily brief with retrospective flag** | ✅ renderer disclosure shipped in `8c4f210` | Stream G done |
| **Swing brief schema + generator** | schema specced in `6f9ffd8`; generator NOT built | Stream F |
| **Options brief tiers (T1/T2/T3/T4)** | NOT specced | future |
| **Hedging brief / portfolio overlay** | tier 3 | future |
| **Newsletter (free distribution)** | NOT built | Stream H |
| **Educational content** | NOT built | future |
| **Strategy X-Ray as delivered product** | spec'd, NOT built | future |

### L7 — Customers

| Item | Status |
|---|---|
| 3 hand-picked pilot users | identified by user | Stream H |
| First-week free, then charge | decided | H |
| PDF + email delivery | decided | small commit pending |
| Web dashboard if <₹300/mo hosting | conditional | H |
| Tiered pricing T1–T4 | mentioned | H |
| Customer outcome tracking (did THEY profit) | flywheel, NOT built | future schema |

---

## §3. The Parallel Streams (the work board)

Each stream is independently dispatchable. Owners are listed.
Dependencies are explicit. The "during chokepoint?" column records
whether the stream was safe during the former Stream B branch-hygiene
blocker; Stream B itself is now resolved.

### Stream A — Rupee-sizing experiment

**Layers**: L4
**Owner**: Codex (VPS, uses existing bundle)
**Depends on**: nothing (uses existing 710362b bundle as-is)
**During chokepoint?**: YES — independent of audit patches
**Status**: queued, prompt drafted, ready to dispatch
**Scope**:
- Implement `RupeeTargetExecutionConfig` in V2 simulator:
  `required_reward_inr=600`, `min_per_share_move=6`, `max_notional=200000`,
  `min_notional=30000`, `stop_ratio=0.5`.
- Rerun touched-pool population on the existing bundle under this
  sizing rule (replaces ATR-symmetric geometry).
- Per-factor breakdown. p-values vs zero.

**Acceptance**: any cell crosses zero with p < 0.05 on n ≥ 200 →
post-touch resurrects at small-N selective scale. Otherwise → post-touch
definitively closed under user's actual operating discipline.

**ETA**: 30 min wall-clock.

**Why this matters**: the entire post-touch verdict was given under
ATR-symmetric geometry — not the user's actual rupee-floor discipline.
This is the one experiment that could legitimately reopen post-touch.

---

### Stream B — Audit-patch branch hygiene

**Layers**: L0–L5 (spans the stack)
**Owner**: Codex + user
**Depends on**: nothing — former laptop/remote chokepoint resolved
**During chokepoint?**: n/a
**Status**: ✅ RESOLVED — Codex audit patches are on the remote branch
**Scope**:
- Confirm Codex's 4 rebased commits (`52c88df`, `ce1ed40`, `a876351`,
  `ab7d57d`) are pushed to `claude/liquidity-pool-backtester-1uskb`.
- Ensure Opus commits (`c488483`, `0b27e53`, `843b400`, `5d01d81`,
  `8c4f210`, `ee92848`, `6f9ffd8`, `64ab9b5`) are reconciled with them.
- Single branch, all patches in effect.

**Acceptance**: `git log --oneline -12` on the branch shows both agents'
commits, local `HEAD` equals `origin/claude/liquidity-pool-backtester-1uskb`,
and there are no tracked-file conflicts. Verified 2026-06-05 at
`ab7d57d`.

**What was unblocked by it**:
- Retrains using V2-notional policy labels (`ce1ed40`).
- Drift monitor fixes and outcome-log hardening (`52c88df`).
- Backfill ordering fix (`ce1ed40`).
- Live broker fail-closed/default-quantity hardening (`52c88df`).
- Geometry-sweep resume (`ab7d57d`).
- Stream D / E / F code work that touches L2/L4/L5.

**ETA**: done.

---

### Stream C — Detector batch v1

**Layers**: L1
**Owner**: Opus
**Depends on**: nothing (pure feature additions, no model retraining
needed to ship them; importance verification happens on the next retrain
after Stream B's reconciled branch)
**During chokepoint?**: YES — fully independent
**Status**: all 3 commits ✅ DONE; feature-importance verification
deferred to next post-B retrain
**Scope (commit 1)** ✅ `5d01d81`:
- Liquidity-sweep detector (high pierced + reclaimed within K bars).
- Stop-run + reclaim pattern (the SMC manipulation signature).
- One file: `liqpool/detectors/sweep.py`.
- Added to `Pool.contributors` factor families.
- Tests pin: detection on a known pattern, non-detection on noise,
  causality under truncation.

**Scope (commit 2)** ✅ `ee92848`:
- Multi-bar imbalance + premium/discount midpoint
  (`liqpool/detectors/imbalance.py`).

**Scope (commit 3)** ✅ `ee92848`:
- Volume-weighted swing + cumulative-delta-proxy
  (`liqpool/detectors/volume.py`).

**Acceptance per commit**: tests pass, lint clean, feature-importance
verification deferred to next post-B retrain.

**ETA**: 1 commit per session, 3 commits total. — DONE.

---

### Stream D — Options vertical slice (Greeks → model → brief)

**Layers**: L1 + L2 + L6
**Owner**: Opus (design); Codex (training, post-B)
**Depends on**:
- Warehouse Greeks parquet (✅ have it)
- Stream B resolved; training now depends on Codex/VPS scheduling
**During chokepoint?**: design portions YES, training portions NO
**Status**: design ✅ DONE (`docs/options_expected_return_model_spec.md`);
training unblocked/queued
**Scope (this session, design only)**:
- `OptionsExpectedReturnModel` specification: inputs (proximity h=12/36/60,
  direction, path_efficiency, vol_regime_zscore_20d, distance-to-strike,
  days-to-expiry, IV percentile, side), output (expected return per
  ATM weekly option premium under (entry, side, time-of-day)).
- Daily-Greeks-to-intraday-equity feature pipeline using
  `align_daily_to_intraday` (the lagged join I already shipped).
- Brief's `options_suitability` section schema update — populated
  Greeks-aware output.

**Scope (post-B, training)**:
- Train the head on the EOD options + spot + risk-free join.
- Wire the brief.

**Acceptance**: design committed as spec doc this session. Training
commit lands in a post-Stream-B training session.

**ETA**: design 1 commit this session; training 2-3 commits post-B.

---

### Stream E — Live broker hardening

**Layers**: L4
**Owner**: Codex
**Depends on**: Stream B resolved; remaining work is implementation
and reconciliation coverage
**During chokepoint?**: n/a
**Status**: partial — fail-closed/default-quantity landed in `52c88df`;
full order lifecycle still queued
**Scope**:
- Order state machine (entry submitted → filled → SL/target placed →
  monitor → reconcile).
- Fail-closed on broker read errors in live mode (test mode keeps the
  empty-fallback).
- Default-quantity required for auto-confirm.
- Reconciliation tests.

**Acceptance**: live runner can't place orders without explicit sizing
intent; broker read failures fail-closed; orphan-order tests pass.

**ETA**: 2-3 commits post-B.

---

### Stream F — Swing vertical slice

**Layers**: L2 + L6
**Owner**: Opus (design); Codex (training, post-B)
**Depends on**: Stream B resolved; training now depends on Codex/VPS scheduling
**During chokepoint?**: n/a
**Status**: design ✅ DONE (`docs/swing_vertical_spec.md`); training
unblocked/queued
**Scope (this session, design only)**:
- Swing horizons (5d, 10d, 20d) added to proximity model config.
- Swing brief schema (different from intraday: longer T, no MIS cap,
  multi-day persistence per pool).
- Swing-specific reaction labels (no same-session cap).

**Scope (post-B, training)**:
- Train proximity at swing horizons.
- Generate swing brief.
- Backfill swing outcome log.

**Acceptance**: design doc committed this session. Training landed
post-B with a swing-brief sample artifact.

**ETA**: design 1 commit this session; training 2-3 commits post-B.

**Why this matters**: cost arithmetic on swing (delivery, 0 brokerage,
larger moves per trade) is structurally favorable. This is the most
likely product line to land profitable without further alpha changes.

---

### Stream G — Yesterday-Audit retrospective flag

**Layers**: L5 + L6
**Owner**: Opus
**Depends on**: nothing (small additive change)
**During chokepoint?**: YES — independent
**Status**: ✅ DONE (`8c4f210`)
**Scope**:
- `PredictionRecord.is_retrospective: bool = False` field.
- Backfill script sets `True` on retrospective replays.
- Brief renderer surfaces the flag: "Yesterday audit (calibration
  estimated on retrospective replay)" when >50% of predictions are
  retrospective.
- Test pins the flag end-to-end.

**Acceptance**: 1 commit, test passing. — DONE.

**ETA**: 30 min in a fresh session. — DONE.

**Side effect**: the renderer's previous unconditional pending-stub
rendering of YESTERDAY AUDIT is replaced with a real per-bucket
hit-rate table (still pending-stubs when no joined log exists). Means
the section will start showing real numbers as soon as the backfill
writer is run against a real bundle, with the retrospective disclosure
line until live outcomes accumulate.

---

### Stream H — Distribution + first 3 customers

**Layers**: L7
**Owner**: User (Opus + Codex cannot do this work)
**Depends on**: nothing (independent of engineering streams)
**During chokepoint?**: YES — totally independent
**Status**: ongoing, user is on it
**Scope**:
- 3 hand-picked pilot users from existing pool (options-focused).
- DM/email outreach with sample brief attached.
- First brief delivery commitment with date.
- Newsletter draft for organic distribution.
- Pricing communication.

**Acceptance**: 3 customers say yes; first brief send date locked.

**ETA**: user-dependent.

---

## §4. The Decision Tree

The original plan as a tree. Each branch annotated with status. When we
deviate (which happens often and is generally good), §5 logs the
deviation node with cause/finding/impact.

```
ROOT — Build a market-intelligence operating system
│
├── [Layer 0-5] Build the engine + audit machinery
│   ├── [L0] Data warehouse                                       ✅ DONE
│   ├── [L1] 42 base features                                     ✅ DONE
│   ├── [L1] Detector upgrade batch                               🔄 Stream C
│   ├── [L2] 5 trained models (Q/Dir/Prox/React/R1)               ✅ DONE
│   ├── [L2] Time-decay sample weighting                          ✅ shipped, default flip pending
│   ├── [L2] Manipulation-aware direction model                   🔮 Stream D / future
│   ├── [L2] Sub-alpha library                                    🔮 future
│   ├── [L3] Alpha registries (5 production default; 4 sparse research) ✅ DONE
│   ├── [L4] V2 simulator + cost realism                          ✅ DONE
│   ├── [L4] RupeeTargetExecutionConfig                           🔄 Stream A — TODAY'S INSIGHT
│   ├── [L4] Scale-out partial-profit (for options)               🔮 Stream D
│   ├── [L4] Live broker hardening                                🔄 Stream E
│   ├── [L5] 4 null tests + calibration + outcome log             ✅ DONE
│   └── [L5] Retrospective flag                                   🔄 Stream G
│
├── [Layer 6] Build the products
│   ├── [L6] Daily Brief equity-intraday v1                       ✅ DONE
│   ├── [L6] Options-suitability section (Greeks-wired)           🔄 Stream D
│   ├── [L6] Swing brief                                          🔄 Stream F
│   ├── [L6] Strategy Diagnosis as paid deliverable               🔮 spec'd, future
│   ├── [L6] Newsletter (free distribution)                       🔄 Stream H
│   └── [L6] Hedging brief / portfolio overlay                    🔮 tier-3 future
│
├── [Layer 7] Distribute to customers
│   ├── 3 hand-picked pilot users                                 🔄 Stream H
│   ├── First-week free, then charge                              📅 decided
│   ├── PDF + email + maybe web dashboard                         📅 decided
│   └── Tiered pricing T1-T4                                      🔮 future
│
└── [Strategic axis] Self-trading vs research-subscription business
    ├── [Decision] Three revenue lines, sequenced                 📅 decided
    │   ├── 1. Daily brief subscription (ships first)             🔄 imminent
    │   ├── 2. Strategy Diagnosis (sells second)                  🔮 spec'd
    │   └── 3. Private alpha (sells last; harder)                 ❌ DECLINED for post-touch cash MIS
    │
    └── [Empire] Multi-layer flywheel (data + infra + media + ed) 🔮 24+ months
        ├── Slippage dataset as moat                              🔮 T4.1
        ├── Pool / liquidity-zone dataset                         🔮 T4.2
        ├── Outcome-log dataset                                   🔮 T4.3
        ├── Detector / event-marker dataset                       🔮 T4.4
        ├── Education / learning camps                            🔮 future
        ├── Newsletter / content channel                          🔄 Stream H
        ├── Infrastructure-as-a-service (X-Ray, backtest)         🔮 T3.1
        ├── Managed PMS/AIF / hedge fund                          🔮 24-36mo path
        └── Multi-broker API distribution                         🔮 24mo path
```

**Legend**:
- ✅ DONE
- 🔄 IN PROGRESS / READY TO START (a stream)
- 🔮 SCOPED / FUTURE
- 📅 DECISION RECORDED
- ❌ DECLINED with reason
- ⚠️ BLOCKED (by an explicitly named active dependency)

---

## §5. Deviation Log (chronological + tree-linked)

> Every time we deviate from the original plan because a result surprised
> us, the deviation is logged here as a node with: branch point, trigger,
> finding, plan-change, status.
>
> Deviations are not failures. They're often where the project advances.
> The discipline is to log them so we know which path we're on.

### Deviation D1 — Q model declared weak, not gate

- **Branch from**: original assumption that Q would be the primary
  ranking signal (high-quality vs low-quality pools).
- **Trigger**: 2026-06-03 retrain showed Q val AUC 0.541 — barely above
  noise. The Q-DECOMPRESS commit fixed bucket-shrinkage compression but
  revealed that the underlying signal is genuinely weak.
- **Finding**: 4 of 5 sector experts got 0% MoE weight; the global model
  dominates. Q has no useful ranking power beyond ~5% above base rate.
- **Plan-change**: Demoted Q from "primary gate" to "context only."
  Stopped using Q ≥ 0.55 as a hard gate in live plans.
  `train_sector_experts` defaulted to False (saves ~40% Q training time).
- **Downstream**: All "is the pool high-quality" reasoning now uses
  proximity + reaction model outputs as the primary rankers. Q stays in
  reports as a context tag.
- **Status**: RESOLVED, plan adjusted.

### Deviation D2 — Post-touch cash MIS self-trading empirically dead

- **Branch from**: original strategy plan ("we trade pools, ride the
  rejection, that's the alpha").
- **Trigger**: 2026-06-04 geometry-mode sweep returned
  `any_positive_EV_cell=false` across 18 cells (6 geometries × 3 modes).
  Best (2.5/2.5 / touch_confirmed) at mean_R = -0.324R; reclaim mode
  gross_R -0.052 with cost_R +0.906 (17× cost-to-gross ratio).
- **Finding**: model ranks correctly but cost arithmetic on small-ATR
  cash-equity moves consumes any gross edge. Not a model bug — a
  structural product-of-trade-size-and-cost bug.
- **Plan-change**:
  - Stop trying to make post-touch cash MIS work under current cost
    structure.
  - Roadmap item moved to "Explicitly declined" with three trigger
    conditions for re-evaluation (cost regime drop, options translation,
    larger-ATR instrument).
  - Pivot: pre-touch journey alphas + options translation + swing.
- **Downstream**: Stream A (rupee-sizing) became the experiment that
  could legitimately re-open this if user's actual operating discipline
  changes the verdict.
- **Status**: RESOLVED, pivot accepted, Stream A queued as the one
  remaining post-touch re-test.

### Deviation D3 — Brief generator had `dist_atr=1.0` placeholder

- **Branch from**: assumed proximity-prediction inputs were correct.
- **Trigger**: 2026-06-04 first Codex backfill on real bundle: 1260
  predictions, calibration_error 0.98 on the very_high bucket — model
  predicting 98% on outcomes that never happened.
- **Finding**: I had a placeholder `dist_atr=1.0` in the brief
  generator's `_gather_active_pool_predictions`. Every proximity
  prediction was generated as "this pool is 1 ATR from price" regardless
  of the real distance.
- **Plan-change**: Fixed in commit `a6aab91`. Regression test added so
  the bug can never come back silently. 1260 backfilled predictions
  marked stale.
- **Downstream**: re-backfill needed on bug-fixed + audit-patched HEAD.
  Calibration numbers in upcoming briefs will be honest.
- **Status**: RESOLVED, fix shipped, re-backfill queued on the reconciled
  post-Stream-B branch.

### Deviation D4 — MFE > MAE finding suggested geometry was wrong direction

- **Branch from**: original assumption that pool-respect strategies
  should use tight stops (0.5 ATR) and 2.0 ATR targets.
- **Trigger**: 2026-06-04 post-touch reaction quality showed avg MFE
  2.63 ATR > avg MAE 2.22 ATR. Strict-respect rate 46% but the
  favourable-excursion side averages bigger than the adverse.
- **Finding**: stops too tight to survive MAE; targets too short to
  catch MFE. Suggested testing wider stops + wider targets.
- **Plan-change**: Geometry-mode sweep run.
- **Downstream**: led to D2 (the cost arithmetic confirmation that
  wider geometry helps but not enough).
- **Status**: RESOLVED — interesting but ultimately confirmed D2.

### Deviation D5 — Rupee-sizing discipline ≠ ATR-symmetric geometry

- **Branch from**: D2's "post-touch is dead under any geometry" verdict.
- **Trigger**: 2026-06-04 (today) — user explained they personally trade
  with a ₹600 absolute-rupee target floor and scale QUANTITY (not target
  ATR) to maintain it. Our entire geometry sweep assumed ATR-multiple
  targets. We never tested user's actual operating discipline.
- **Finding**: an entire class of strategies (rupee-floor + variable-qty)
  was never simulated. Cost-realism filter ratio = 12× under user's
  sizing vs ~2× under ATR-symmetric. The whole "post-touch is dead"
  conclusion has an asterisk.
- **Plan-change**: Stream A queued. `RupeeTargetExecutionConfig` added
  to L4 layer table. If Stream A clears zero on any cell, post-touch
  resurrects at small-N selective scale.
- **Downstream**: Stream A becomes highest-signal experiment in queue;
  status of D2 ("declined") is provisional until Stream A runs.
- **Status**: OPEN, pending Stream A result.

### Deviation D6 — Architecture-first thinking adopted

- **Branch from**: my own depth-first habit ("do this then this then
  this").
- **Trigger**: 2026-06-04 (today) — user called out that I was thinking
  rigidly sequential when the layers are independent and most work can
  parallelize.
- **Finding**: 30+ work items spread across 7 layers; only ~3 are truly
  blocked by Stream B. Most can run concurrently.
- **Plan-change**: Replaced sequential queue with 8-stream parallel
  dispatch (§3). Master plan doc (this doc) created as the canonical
  operating board. Every new idea is mapped onto the layer matrix
  BEFORE proposing work.
- **Downstream**: this whole doc.
- **Status**: ADOPTED, operating mode change.

### Deviation D7 — Velocity expectation revised upward

- **Branch from**: my own habit of giving "weeks/months" estimates
  patterned on typical software timelines.
- **Trigger**: 2026-06-04 (today) — user pointed out that things I said
  would take "months" got shipped in hours today.
- **Finding**: actual project velocity is ~3-5× what I was projecting.
  10 days produced ~30 substantive commits + 10 new modules + the
  warehouse + the brief + the diagnosis spec + the audit fixes.
- **Plan-change**: §6 (Velocity & Cadence) targets 5-day / 10-day
  horizons for major deliverables, not month-scale.
- **Downstream**: stream ETAs in §3 reflect this; what would have been
  "post-revenue T1.1" might land in week 2 not month 6.
- **Status**: ADOPTED.

### Deviation D8 — Stream B branch chokepoint resolved

- **Branch from**: §0 / §3 assumption that Codex's audit-patch commits
  were local-only and unavailable on the remote branch.
- **Trigger**: 2026-06-05 branch check after Codex push/rebase. Local
  `HEAD`, `origin/claude/liquidity-pool-backtester-1uskb`, and fetched
  remote state all resolved to `ab7d57d`.
- **Finding**: the old local commit names (`06e4b83`, `34ceb5b`,
  `828d120`, `cd9f384`) are superseded by pushed rebased commits
  `52c88df`, `ce1ed40`, `a876351`, and `ab7d57d`. The branch now
  contains both Opus stream commits and Codex audit-patch commits.
- **Plan-change**: Stream B marked RESOLVED. Stream D training, Stream E
  implementation, and Stream F training are unblocked from branch-hygiene
  perspective. Their own acceptance tests/retrains still gate product
  readiness.
- **Downstream**: all future retrains/sweeps should use `ab7d57d` or a
  successor that includes it; do not cite the laptop chokepoint as an
  active blocker.
- **Status**: RESOLVED.

---

## §6. Velocity & Cadence

**Observed velocity (10-day rolling):**
- ~30 commits to the branch
- ~10 new product/library modules
- ~400 tests added (currently 405 passing)
- ~5 retrains executed by Codex on the VPS
- 4 major audit-patch batches (per Codex)
- 8 strategic deviations (D1–D8 above) logged and resolved

**Honest implications for projection:**

What I previously called:
- "Months" → realistically 2-3 weeks at observed velocity
- "Weeks" → realistically 3-5 days
- "Days" → realistically same-day at this pace
- "Hours" → realistically minutes when no dependency

**The pacing rules to keep this sustainable:**

1. **Each stream produces visible progress within 48 hours of dispatch.**
   If a stream goes silent for >48h, escalate or split.
2. **One major commit per agent per active session.** No 5-commit
   marathons that introduce regressions.
3. **Every commit appends to §5 Deviation Log if it's a course-change**,
   or to COORDINATION.md RECENTLY DECIDED if it's plan-adherent.
4. **Status of any active chokepoint is checked before EVERY dependent
   stream commit.** Don't queue dependent work blind.

**5-day target (from today):**
- Stream A result back (rupee-sizing verdict)
- Stream B unblocked + reconciled ✅ (`ab7d57d`)
- Streams C, G, H ship at least one commit each
- Stream D + F design docs landed; training queued
- One re-backfill + retrain on the bug-fixed + audit-patched branch
- One options-Greeks-wired brief sample
- 3 pilot users on Stream H (target — user-dependent)

**10-day stretch:**
- All 8 streams have shipped at least one commit
- Options brief v1 customer-shippable
- Swing brief v1 spec'd + training started
- First paying customer (week-2 onward per the free-week-1 plan)
- Track record forward outcome log at 7-10 days of live predictions

---

## §7. Operating Discipline

### How Opus operates

- Reads `AGENTS.md` → `MASTER_PLAN.md` (this doc, §3) → `COORDINATION.md`
  → other relevant docs on every session start.
- For every user input that proposes new work: map onto §2 layer matrix
  before responding.
- Treats §3 streams as parallel by default. Never says "do this then
  this" if the items are independent.
- Names the chokepoint explicitly when one exists.
- Records every plan-deviating result in §5 with cause/finding/impact.
- Ships documentation + small self-contained code commits.
- Defers long-running training / multi-file refactors to Codex.

### How Codex operates

- Same session-start read order.
- Owns: long-running training jobs, sweep runners, multi-file refactors,
  branch hygiene (Stream B), VPS deployments.
- Reports back via:
  - Commit messages with substantive details
  - Appends to COORDINATION.md RECENTLY DECIDED
  - For deviation-class results: appends a §5 deviation entry here

### How User operates

- Architect + dispatcher. Reviews master plan, approves stream priority
  changes, makes pricing/strategy/product decisions.
- Owns Stream H entirely (no engineering substitute exists for
  customer relationships).
- Reports machine availability or remote access constraints when they
  block a stream.
- Surfaces new ideas; Opus catches them in §2 layer matrix, §5 dev log,
  or §3 stream creation.

### Cross-stream coordination rules

- **Conflict avoidance**: each stream touches a stated set of files (in
  the stream's COORDINATION entry). If your work touches files outside
  your stream's set, pause and announce in COORDINATION before editing.
- **Test discipline**: every commit must end with `python -m pytest -q`
  green + `python scripts/compliance_lint.py` `enforced_failures=0`.
- **No silent backwards-incompatible changes**: schema changes, config
  defaults, public API → flag in commit message AND in §5.
- **Branch policy**: all work on `claude/liquidity-pool-backtester-1uskb`
  unless the user explicitly creates a replacement branch. Stream B has
  reconciled; this branch is the canonical working branch as of `ab7d57d`.
- **Push policy**: each agent pushes after every passing commit. Never
  hoard local commits.

---

## §8. Open Questions / Pending Decisions

> Items requiring user input. Each blocks at least one stream.

| Q | Question | Blocks |
|---|---|---|
| Q1 | Confirm Stream A `RupeeTargetExecutionConfig` parameters: ₹600 floor, ₹6/share min move, max ₹2L notional, min ₹30k, stop=50% of target. Acceptable? | Stream A |
| Q2 | SUPERSEDED: Stream B is unblocked; Codex's rebased patch commits are pushed at `ab7d57d`. | none |
| Q3 | For Stream H: first 3 pilot customer names + target send date for brief #1 | Stream H execution |
| Q4 | Web-dashboard hosting decision: GitHub Pages + Vercel free tier confirmed acceptable, or other? | Stream H + future product |
| Q5 | Options model — train on EOD-only data (current warehouse coverage Jan 2024 – Jun 2026), or wait until forward-captured intraday data accumulates? | Stream D training timing |
| Q6 | Strategy Diagnosis product — launch as paid tier alongside Daily Brief immediately, or after brief proves out? | Product timing |

---

## §9. Explicitly Declined

> Decisions to NOT do something, with reasons. Kept so future cycles
> don't re-litigate without new evidence.

- **Full Greeks engine for v1 options brief** — wait until paying
  customer asks; level-to-strike mapping suffices.
- **Mobile app** — distribution problem, not a product problem.
- **Public Discord/Telegram community** — SEBI risk, support burden.
- **Copy-trading** — outside SEBI-safe corridor.
- **Tipster-style "buy X at Y stop Z" outputs** — architectural; enforced
  at render time.
- **Post-touch cash-equity MIS self-trading at current cost structure**
  (under ATR-symmetric geometry) — empirically dead per geometry-mode
  sweep. Re-evaluation requires (a) cost regime drops, (b) move to
  options, OR (c) move to larger-ATR instrument class, OR (d) Stream A
  rupee-sizing produces a positive cell.
- **Building a full Brain/Strategy Executor organ (T2)** before any one
  alpha clears costs — combining negative-edge signals into a more
  elaborate negative-edge system is a known failure mode.
- **Heavy CPCV when walk-forward suffices** — kept as optional research
  flag; not default.

---

## §10. Maintenance Protocol

**On every commit:**
1. Update COORDINATION.md RECENTLY DECIDED with the one-line entry.
2. If the commit changes status of a stream, update §3 here.
3. If the commit was a deviation from the prior plan, append a §5 entry.
4. If the commit changes architectural facts, update §2 here.

**On every session start:**
1. Read `AGENTS.md`.
2. Read §3 (Streams) here to see what's in flight.
3. Read §5 (Deviation Log) tail for recent direction changes.
4. Read COORDINATION.md RECENTLY DECIDED tail for the last 10 commits.

**On every strategic decision:**
1. Record in COORDINATION.md "Strategic decisions" block.
2. Mirror in §4 Decision Tree as a node update.
3. If decision creates new work: §3 stream entry; §2 matrix update.

**Stream B reconciliation (completed 2026-06-05):**
1. Codex pushed the 4 audit patches as rebased commits
   `52c88df`, `ce1ed40`, `a876351`, `ab7d57d`.
2. Local `HEAD` and `origin/claude/liquidity-pool-backtester-1uskb`
   both resolved to `ab7d57d`.
3. Dependent streams (D training, E, F training) are unblocked from
   branch-hygiene perspective.
4. Remaining duty: run/record confirmation retrains and stream-specific
   acceptance tests before claiming product readiness.

---

## §11. Quick reference for fresh agents

**If you have 60 seconds:**
1. Read §0 (orientation).
2. Read §3 (streams) — find the one assigned to you.
3. Read its dependencies row. If chokepoint, stop and check §5 D6 / D7
   for recent context.

**If you have 5 minutes:**
1. Above, plus
2. §1 (architecture summary) for context.
3. §5 last 3 deviation entries for what direction the project just took.

**If you're picking up after a long absence (>3 days):**
1. Read §0, §1, §3, §4, §5 in full.
2. Read COORDINATION.md RECENTLY DECIDED tail (≥20 commits).
3. Pull the branch and verify your local matches the master.
4. THEN start work.

---

*This document is the canonical operating plan. Last updated by Opus,
2026-06-04, in response to user's request for an architecture-first
master plan. All prior sequential queues are deprecated in favor of
§3's parallel streams.*
