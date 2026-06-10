# The Engine as an Organism

> A metabolic map of the codebase: what each organ produces (excretes),
> what it consumes (eats), and the pathways that now feed organs into
> each other's betterment. Companion to `SYSTEM_OVERVIEW.md` (the static
> layer view) and `CONSOLIDATED_ENHANCEMENT_PLAN.md` (the work board).

The insight driving this view: the engine was a **food factory with no
digestion** — it excreted rich artifacts (walkforward results, the
shadow log, the outcome log, drift history) but almost nothing fed those
artifacts back into the organs that could grow from them. The flywheel
layer (Streams L + M) plus the extractors/hub (this pass) are the
digestive + circulatory systems that close the loops.

## The substrates (what flows through the body)

| Substrate | Produced by | Schema |
|---|---|---|
| **Pools + outcomes** | walkforward / tester | `wf.oos_pools`, `wf.oos_results` (PoolResult) |
| **Shadow events** | every declined gate (Stream L) | `shadow_log` parquet, 8 event kinds |
| **Outcome log** | brief + nightly resolver | predictions ⋈ resolutions, hash-chained |
| **Drift history** | each retrain's DriftMetrics | chronological metric frame |
| **Reaction paths** | tester (post-touch closes) | `(close - level)/ATR` over k bars |

## The organs and their metabolism

Each organ now has a **digestive enzyme** (extractor) that converts a
raw substrate into the organ's food, and a **delivery vein** (a hub
serving helper or a direct consumer hook) that carries the trained
organ to where it acts.

| Organ (model) | Eats (via extractor) | Excretes / acts on |
|---|---|---|
| **M.4 Regret** | shadow ⋈ resolutions | executor SKIP advisory `regret_advisory` |
| **M.5 DetectorTrust** | `detector_outcomes_from_report` | `hub.trust_weighted_score()` → pool scoring |
| **M.6 DriftImminent** | drift history | `drift.attach_imminence()` → DriftReport |
| **M.7 Archetypes** | `reaction_paths_from_report` | reaction label dimension + brief vocabulary |
| **M.8 BucketAging** | `bucket_aging_history_from_joined` | `hub.staleness_note()` → confidence notes |
| **M.9 CrossAsset** | `co_occurrences_from_report` | unified-trainer sample weighting (future) |
| **M.10 MetaCalibrator** | outcome log ⋈ | brief probability recalibration (live) |
| **K.1 Slippage** | V2 trade frame | V3 simulator slippage |
| **K.2 Fill** | V2 fill outcomes | V3 fill probability |
| **K.3 Survival** | triple-barrier outcomes | V3 sizing + position aging |

## The closed loops (cycles)

These are the metabolic cycles — output of one organ becomes input of
another, and the cycle tightens over time:

### Loop 1 — The calibration cycle (live)
```
brief publishes probability
   → outcome log resolves it next session
   → M.10 meta-calibrator fits on the residual
   → next brief's probability is recalibrated  ──┐
   ↑                                              │
   └──────────────────────────────────────────────┘
```
Closed in this pass: `generate_brief(flywheel_hub=...)` runs the
meta-calibrator on predictions before the watchlist threshold, and
discloses the applied temperature in the confidence notes.

### Loop 2 — The regret cycle (advisory)
```
executor SKIPs a trade
   → shadow log records the context
   → resolver scores the counterfactual (regret / vindicated)
   → M.4 regret estimator fits
   → next SKIP carries P(regret) advisory  ──┐
   ↑                                          │
   └────────────────────────────────────────── ┘
```
Closed advisory-only: `pre_trade_decision(ccv, regret_model=...)`
attaches `regret_advisory`. Action unchanged until Gate-2 validation.

### Loop 3 — The trust cycle
```
detectors fire → pools form → pools resolve (respect/break)
   → M.5 trust router fits per (factor, regime)
   → pool scoring weights contributors by trust  ──┐
   ↑                                                │
   └──────────────────────────────────────────────── ┘
```
Plumbed via `hub.trust_weighted_score()`; pool-scorer adoption is the
next wiring step (kept out of this pass to avoid changing live pool
scores before a sensitivity sweep).

### Loop 4 — The drift-anticipation cycle
```
each retrain emits DriftMetrics
   → history accumulates
   → M.6 imminence model fits on trajectories
   → next drift check carries P(drift soon)  ──┐
   ↑     (and LGBMX importance_drift feeds the   │
   └──────  same history automatically) ─────────┘
```
Closed: `drift.attach_imminence()`. Also fed automatically by LGBMX's
built-in `importance_drift()` — every retrain now produces the
regime-shift signal without a separate audit.

### Loop 5 — The execution-realism cycle
```
V2 simulates trades → emits realised slippage / fills / hold times
   → K.1/K.2/K.3 fit on that output
   → V3 uses learned slippage/fill/survival
   → more realistic backtests → better sizing  ──┐
   ↑                                              │
   └──────────────────────────────────────────────┘
```
Plumbed via `ExecutionV3Config`; training runs are the operator step.

## The circulatory hub

`FlywheelHub` (liqpool/flywheel/hub.py) is the bloodstream. One object
holds all seven organs. `fit_from_artifacts()` is the nightly
metabolism (run by `scripts/train_flywheel.py`); the serving helpers
(`adjust_probability`, `trust_weighted_score`, `staleness_note`) are
the veins. Every consumer accepts an **optional** hub and degrades to
today's behaviour when it's absent or an organ is unfit — the body
survives a starved organ.

This is the primitive form of the roadmap's **T2.3 cross-organ
communication protocol**: organs never import each other; they all
plug into the hub.

## What is NOT yet looped (honest gaps)

- **M.9 cross-asset** has its extractor + matrix but no consumer wired;
  the unified trainer doesn't yet downweight low-transfer sample
  sharing. Next pass.
- **M.7 archetypes** clusters and classifies but the archetype id isn't
  yet attached to the brief's key-zone block as a customer-facing
  label. Needs the operator's naming pass on `archetype_profiles()`
  first.
- **Pool-scorer trust weighting** is plumbed (`trust_weighted_score`)
  but not called from `score_pool` — deferred until a sensitivity
  sweep confirms it doesn't destabilise live pool ordering.
- All organs are **frameworks trained on synthetic test data only**.
  Real training is the operator step once substrates have accumulated.

## Nutrient flow summary

```
                     ┌─────────────────────────────────────┐
                     │            FlywheelHub               │
                     │  (bloodstream — all organs plug in)  │
                     └───▲───────────────────────────┬──────┘
        fit_from_artifacts│                           │serving veins
                          │                           ▼
   ┌──────────┬───────────┴──────┐         ┌──────────┬──────────┐
   │ bundle   │ shadow log       │         │ brief    │ executor │
   │ outcome  │ drift history    │         │ drift    │ pools    │
   └────┬─────┴──────────────────┘         └──────────┴──────────┘
        │ extractors (enzymes)
        ▼
   detector_outcomes / reaction_paths / co_occurrences /
   bucket_aging_history
```
