# Track A Pool-Level Null — Opus Decision Memo

**For:** Codex (implementer) + user
**From:** Opus (brain/debugger)
**Date:** 2026-05-31
**Status of Track A:** DO NOT advance to paper trading. One decisive test remains
before the strategic branch decision.

---

## 1. The number everyone is under-weighting

The decision-relevant number in the pool-level null report is **not** any p-value.
It is:

> **1.5× cost stress = −0.029R (direction-hard), −0.042R (direction-soft)**

At realistic costs, the strategy is **net negative**, before any pool-vs-null
argument. Whatever the edge's *source* — pool geometry, intent timing, or
sector-time state — it is **sub-cost at current strength**. This reframes the
whole debate: the question is not "is the signal real?" but "is there a *subset*
where the signal is strong enough to (a) clear costs AND (b) beat its matched
null?" **Two hurdles. Both must clear.** The pool-level null only spoke to (b),
and only in aggregate.

The user's counter-hypothesis, **even if fully correct, does not by itself make
the strategy tradeable.** It only relocates where the sub-cost signal lives.

---

## 2. What the null actually proved — read the decomposition, not the verdict label

The four nulls cleanly decompose where signal lives:

| Component manipulated | Null R | p | Read |
|---|---|---|---|
| Pool **geometry** (random matched-distance) | +0.225 vs +0.227 | **0.485** | dead-center → geometry adds **nothing** |
| **Direction** (shuffled) | +0.220 | 0.347 | direction adds **almost nothing** |
| Pool **distance** (ATR-5 fixed) | +0.236 ≥ actual | — | fixed ATR target **as good or better** |
| **Sector** (sector-neutral random) | +0.187 / **+0.129** | 0.109 / **0.030** | **sector is the one survivor** |

The geometry null sits at the ~50th percentile of its null distribution — this is
**not a marginal fail, it is a dead-center fail.** Pool *level as target* is
settled: it is not the edge. Do not re-litigate it.

The **only** component that degrades when randomized is **sector** (p=0.030 on
soft is significant at 5%). Combined with the multi-pocket reports (time-of-day
matters: midday/5-8ATR +0.722R, morning/5-8ATR +0.668R), the surviving signal
is **"sector + time-of-day + being at ~5 ATR distance in a trending intraday
window."** Not pool geometry. Not direction. Not pool-specific distance.

---

## 3. The user's counter-hypothesis: partially right, and it sharpens the question

**The user is correct on the structural point:** the null preserves the
model-selected *moment* and only randomizes the pool *object*, so it **cannot**
refute "pool detection selects high-intent moments." That hypothesis is genuinely
**untested** by this experiment. Good catch.

**The sharpening the user's framing misses:** the moment *is* pool-derived. A
candidate exists *because* a pool was detected at distance D and the
proximity/direction models fired on it. So the null is really showing: *given a
pool was detected here, the exact level doesn't matter.* That is fully consistent
with the user's own reframe — **pool as intent/state detector, not as precise
target geometry.** The null **supports** that reframe; it does not contradict it.

What remains genuinely untested is the user's actual claim: **do pool-selected
moments beat random moments?** That requires a *different* null.

---

## 4. The decisive missing experiment: the MOMENT NULL

The geometry null fixed *when* and randomized *where*. To test intent-timing you
need the opposite: **fix the "pool exists at distance D in sector S" structure and
randomize *when* the model acts.**

- Keep the structural condition "a pool exists at distance D, sector S,
  time-bucket T."
- Instead of acting at the model-selected timestamps, act at **random timestamps
  with matched (distance × sector × time-of-day) distribution** that the model did
  **not** select.
- Compare realized R of model-selected moments vs random matched moments, through
  the **same simulator and cost model**.

**Decision logic:**
- **Model-selected moments BEAT random matched moments (p<0.05)** → the user is
  right: pool detection carries intent-timing signal. Rebuild Track A as a
  **state/feature model** (pool-derived state features, ATR/learned target
  geometry — *not* pool target geometry). This is their option C.
- **They do NOT beat random matched moments** → the entire edge is "be at ~5 ATR
  in these sectors at these times," with **zero model value**. Simplify to a
  sector-time-ATR rule (option B) or shelve as sub-cost.

**Opus's prior** (state honestly): given sector is already the *only* surviving
null and geometry+direction both died, I expect the moment null to show pools add
little beyond sector-time-distance state. **But it must be run** — the user earned
a clean test of their hypothesis, not my prior.

---

## 5. Second experiment: pocket-conditional geometry null

The aggregate geometry null can hide a regime where pool geometry *does* matter.
**Re-run the existing geometry null WITHIN the strong pockets** (midday/5-8ATR,
morning/5-8ATR), not on the full candidate set.

- If real pools beat matched-random *inside* the +0.7R pocket (p<0.05) → geometry
  matters in that regime even if not overall, and that pocket is the place to
  build.
- If not → geometry is dead even in the best regime; the pocket's R is pure
  sector-time-distance.

---

## 6. Direct answers to the 6 questions

1. **Too harsh / unfair?** No — *fair but narrow*. It correctly answers "does
   exact pool geometry as target matter?" (clean no). It was over-interpreted as
   "pools are useless." Narrow the conclusion to "pool level as precise target is
   not the edge." It does NOT test intent-timing — that's the moment null's job.

2. **Invalidates detector, or only "exact pool level as target"?** Only the
   latter. Pool-as-state/feature/moment-selector is **untested, not invalidated.**
   Caveat: even if it survives, cost stress says it's sub-cost at current strength.

3. **Next diagnostic?** New **option F: the moment null** (§4) is first and
   decisive, paired with the **pocket-conditional geometry null** (§5).
   - **C** (final-stage state-feature model, ATR target) is the right *direction*
     but build it AFTER F confirms there's moment signal to model — building C
     before F risks modeling noise.
   - **B** (sector-time-ATR journey) → build as the **baseline to beat**, not the
     destination. If a pool-state model can't beat sector-time-ATR, pools add
     nothing.
   - **A** (path-quality) → useful color, secondary; won't change the branch.
   - **D** (1m replay) → **low value.** 1m fill realism shifts absolute R for
     actual *and* null equally, so the *relative* comparison that "failed" won't
     change. Skip it for this decision.

4. **How to treat pools?** Provisionally **state/intent detectors → feature
   generators** (operationally the same: pool-derived features in a final-stage
   model). **Not** precise targets (refuted). Confirm with the moment null before
   committing engineering.

5. **Does ATR-5 ≈ actual mean pivot to ATR journey strategy?** It means
   **simplify the target geometry** (pool machinery adds no target value over a
   fixed ATR rule). But ATR-5 has the **same sub-cost R**, so it is **not a
   solution, only a cleaner baseline.** Pivoting to "ATR journey" without finding
   stronger gross R just relocates the same losing trade. The real task is finding
   *what makes the high-R pockets high-R* and whether *those* survive the nulls.

6. **Correct next artifact?** ONE report, TWO experiments: **(a) moment null** +
   **(b) pocket-conditional geometry null**, with per-cell metrics: mean R + CI95,
   target-hit rate, MFE/MAE, time-to-target, tail/drawdown, and **gross R vs 1.5×
   cost-stressed R** (so the cost wall is visible per cell). **Decision gate:** is
   there any (pocket × component) cell where the real signal both (i) clears 1.5×
   cost AND (ii) beats its matched null at p<0.05? Yes → build C on that cell.
   No → pools are a sub-cost research artifact; pivot to options-data foundation
   or shelve Track A.

---

## 7. Implementation spec — task R-MOMENT-NULL (for Codex)

Build as a sibling of the existing pool-level null harness Codex already wrote
(same replay simulator, same cost model, same workers/trials interface).

### 7.1 Moment null

```
def moment_null(
    candidates,           # the model-SELECTED candidate rows (real moments)
    full_bar_universe,    # all bars per asset, to draw counterfactual moments
    *, n_trials=100, workers=12, seed=17,
    match_on=("distance_bucket", "sector", "time_bucket"),
    direction_mode="soft",
) -> MomentNullResult:
    """
    For each trial:
      For each real candidate, draw a RANDOM bar from the same asset whose
      (distance_bucket, sector, time_bucket) matches the candidate's, and that
      the model did NOT select. Build a synthetic candidate at that moment with
      the SAME distance/side/geometry rules. Replay through the same simulator.
    Returns: actual_mean_R, null_mean_R distribution, one-sided p-value
      = fraction of null trials with mean_R >= actual_mean_R.
    """
```

Matching detail: the counterfactual moment must match the candidate's
**distance-to-nearest-pool bucket, sector, and time-of-day bucket** so the only
thing varying is "did the MODEL pick this moment vs a random matched moment."
Critically, the counterfactual must use a pool that *exists at that moment* at
matched distance (or a synthetic matched-distance level), so we're testing the
model's *selection among matched moments*, not "pool vs no-pool."

### 7.2 Pocket-conditional geometry null

Re-use the existing geometry null (`random_pool_matched`) but add a
`--restrict-pocket` filter that runs it within a single pocket
(e.g. `time_bucket=midday AND distance_bucket=5-8ATR`). Output one verdict per
pocket.

### 7.3 Report: `reports/track_a_moment_and_pocket_null.md`

Per (experiment × direction_mode × pocket):
- actual mean R + CI95
- null median R + one-sided p
- target-hit rate, MFE/MAE, median bars-to-exit
- max drawdown, worst-decile R (tail)
- **gross R and 1.5× cost-stressed R** side by side
- PASS / FAIL on the two-hurdle gate (clears cost AND beats null p<0.05)

### 7.4 Acceptance criteria

- Moment null returns finite p-value; with `n_trials>=100` the null distribution
  has >=100 samples.
- Pocket-conditional null runs for at least the two strong pockets
  (midday/5-8ATR, morning/5-8ATR).
- Report explicitly states, per pocket: does the real signal clear the
  two-hurdle gate?
- Unit tests mirroring `tests/test_arsenal_null_tests.py` (matched-draw
  correctness, p-value monotonicity, seed determinism).

### 7.5 Estimated effort

1–2 days. The replay simulator + matching logic already exist in Codex's
pool-level null harness; the moment null is a re-draw of the *timestamp* axis
instead of the *level* axis.

---

## 8. What NOT to do

- **Do not** re-litigate pool geometry as target — settled (dead-center null).
- **Do not** run the 1m replay before deciding — won't change the relative
  comparison; pure cost.
- **Do not** advance to paper trading — the cost stress alone forbids it.
- **Do not** build the final-stage state model (option C) before the moment null
  confirms there is moment-selection signal to model — that risks fitting noise.
- **Do not** treat "exclude BANKING/IT" as a production rule — it's post-hoc
  (see CODEX_HANDOFF §3 Q2).

---

## 9. One-line summary

Pool **geometry** as target is dead (clean null). The surviving signal is
**sector + time + ~5 ATR distance**, and it is **sub-cost** at current strength.
The user is right that pools may still be **intent/state detectors** — but that
is **untested**, and the **moment null** is the one experiment that resolves it.
Run the moment null + pocket-conditional geometry null, judged on the
**two-hurdle gate (beat null AND clear 1.5× cost)**, before any strategic branch.
