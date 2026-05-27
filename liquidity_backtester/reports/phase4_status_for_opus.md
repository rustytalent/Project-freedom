# Phase 4 Status For Opus

Date: 2026-05-28

## Current Read

The project has shifted from "liquidity pool respect" toward a more useful thesis: journey-to-liquidity. The strongest capitalizable signal is price traveling toward a pool, not necessarily the reaction after touch.

The current equity result is not deployable yet. It is also no longer dead. We have one constrained Track A pocket that is positive under the v2 simulator, but it is still marginal under the first CPCV/component-null layer.

## What Is Solid

- Q compression is real. The old `Q >= 70%` live gate was mathematically unreachable.
- Proximity is causal at the audited feature level. Distance leakage audit found zero mismatches across 7.67M rows.
- Pool availability and contributor timing passed the replay audit.
- MTF closed-bar audit has warnings on daily/weekly aggregation, but no hard intraday closed-bar failure.
- Execution simulator v2 is stricter and more honest than v1.
- Existing post-touch execution modes remain negative under v2.
- Track A is the main active path because proximity remains strong at trade-relevant distances.

## Key Numbers

### V2 Post-Touch/Legacy Execution

Neutral fill, state-dependent slippage, 1-minute resolution:

| mode | trades | win | mean R | PF |
| --- | ---: | ---: | ---: | ---: |
| displacement_confirmed | 7,730 | 37.3% | -0.68 | 0.26 |
| touch_confirmed | 19,107 | 35.7% | -0.71 | 0.24 |
| blind_limit | 20,696 | 19.7% | -1.20 | 0.10 |
| reclaim_confirmed | 7,683 | 14.7% | -1.42 | 0.07 |

This means Track B/post-touch is not the current priority unless a later final-stage model rescues it.

### Track A Broad Sweep

Initial Track A pre-touch sweep verdict: `MARGINAL`.

Positive cells exist, but the best deployable evidence failed robustness filters. Best broad result was around `+0.222R`.

### Track A Constrained Pocket

Best research pocket:

- Long only
- Exclude BANKING and IT
- Sectors: AUTO, FMCG, PHARMA
- P_touch >= 0.75
- Direction-to-pool >= 0.65
- Distance 3-8 ATR
- Target fraction 1.0
- Stop 2.0 ATR
- Hold 60 bars, intraday bounded

Result:

- 434 trades
- Mean R: `+0.365`
- Win rate: `57.4%`
- PF: `1.91`

This is the first pocket worth hardening.

### First CPCV / Component Null Layer

Report: `reports/phase4_track_a_cpcv_nulls.md`

Verdict: `CPCV_COMPONENT_MARGINAL`

Path summary:

| scenario | paths | median n | mean R | worst R | positive paths | PF>1 paths | pass paths |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| long_all_sectors | 45 | 105 | +0.322 | -0.301 | 84.4% | 84.4% | 84.4% |
| long_ex_banking_it | 45 | 87 | +0.367 | -0.251 | 80.0% | 80.0% | 80.0% |
| banking_it_long_control | 45 | 18 | +0.117 | -0.720 | 62.2% | 62.2% | 0.0% |
| short_all_sectors_control | 45 | 9 | -0.547 | -1.432 | 11.1% | 11.1% | 0.0% |

Component controls:

- Actual long ex-BANKING/IT: `+0.365R`, PF `1.91`
- Actual long all sectors: `+0.321R`, PF `1.79`
- BANKING/IT long control: `+0.113R`, PF `1.27`, too thin
- Short control: `-0.546R`, PF `0.39`
- Shuffled-direction null median: `+0.222R`, PF `1.47`
- Shuffled-direction p(null >= actual): `0.000`

Interpretation:

- The pocket is not pure noise; it is positive across many chronological paths.
- But shuffled direction still makes money, so proximity + sector + geometry explain a lot of the edge.
- Direction adds lift, but not enough to call the strategy robust.
- BANKING/IT exclusion should remain a soft hypothesis, not a production hard rule.

## What I Would Trust Tomorrow

Use the model as a research assistant, not an auto-trader.

Most trustworthy outputs:

- Direction bias: useful as a context filter, especially when it agrees with price action.
- Proximity: strongest component. High P_touch means "watch this journey," not "enter blindly."
- Track A pocket: useful as a watchlist generator only. A long setup in AUTO/FMCG/PHARMA with high proximity, direction alignment, and 3-8 ATR distance is worth human attention.
- No-trade verdicts from legacy execution: trustworthy. The legacy post-touch modes are negative under honest execution.

Least trustworthy outputs:

- Absolute Q thresholds. Use Q percentiles, not `Q >= 70%`.
- Post-touch 90%+ reaction alerts without recalibration.
- Any setup that depends on BANKING/IT exclusion as if it were proven.
- Short Track A signals. The current short control is poor and thin.

## Non-Geometry Features

Do not add them before the next validation step.

Useful future non-geometry features:

- Gap up/down size and open location vs prior range
- Gift Nifty overnight return
- India VIX overnight change
- Sector index return and relative strength
- Market breadth
- Time-of-day/session phase
- Consecutive market up/down days
- Event/news euphoria flag

My opinion: these are likely needed, but they should be added after we know how the current pocket fails under stronger nulls. If CPCV/null failure is concentrated on gap/euphoria days, gap features become a targeted fix. If failure is uniform, new features are probably complexity on noise.

## Next Best Engineering Step

Run the full synthetic-null suite:

1. Random pools at matched spatial density
2. ATR-offset pools at k = 1, 2, 3, 5
3. Sector-neutral random pools
4. Shuffled-direction null already exists as first layer, but should be repeated against the full synthetic-null framework

Then decide:

- If Track A beats nulls and remains positive: build final-stage journey trade model with triple-barrier labels.
- If it fails nulls: do not paper trade; pivot toward options data foundation or redesign labels.

## Questions For Opus

1. Given shuffled direction median is still `+0.222R` while actual is `+0.365R`, should direction remain a hard gate or become a soft feature in a final-stage model?
2. Is excluding BANKING/IT too post-hoc, given all-sector long still performs at `+0.321R`?
3. Should the next null suite simulate random/ATR pools through the full v2 pre-touch simulator, or can we derive a valid faster approximation from existing candidate rows?
4. Should triple-barrier labels define "journey completed before adverse move" rather than "pool respected after touch"?
5. Would you prioritize final-stage model before or after synthetic random-pool/ATR nulls, given this first CPCV layer is marginal but positive?

## Current Recommendation

Do not trade live. Use daily predict only as a directional/proximity watchlist.

The project should continue as a journey-to-liquidity research engine. The current edge is in pre-touch travel toward pools, not in post-touch respect. The next proof step is null validation, not more alpha features.
