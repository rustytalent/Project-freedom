# Manipulation Alpha Factory

Date: 2026-06-15

This is the build map for turning Sentinel and the liquidity backtester into an alpha factory.

## Thesis

Every engineered market behavior is either:

- a tradeable inefficiency,
- an avoidance signal,
- a sizing constraint,
- or a context feature that helps another alpha.

The machine should not ask only "is this setup bullish or bearish?" It should ask:

> What kind of engineered state is this, have we seen it before, what happened next, what did options premium do, what did top-weight constituents do, and what is the cheapest safe way to express or avoid it?

## Core Flow

```
warehouse + live feed
  -> state feature builders
  -> manipulation atlas
  -> scientist hypotheses
  -> hypothesis miner
  -> shared validation harness
  -> Sentinel shadow ledger
  -> curator findings
  -> flywheel trust and decay
  -> capital governor
```

## Manipulation Atlas V1

The atlas should classify states such as:

| State | Meaning | Possible Action |
|---|---|---|
| index_masking | Index held by a few heavyweights while breadth is weak | Avoid chasing index direction |
| broad_sponsorship | Heavyweights and breadth agree | Directional setups can be trusted more |
| gap_distribution_trap | Gap up/down unsupported by AVWAP/breadth/premium | Fade or avoid late chase |
| dii_cushion_absorption | Sell pressure slows because local sponsorship absorbs | Watch reclaim setups |
| premium_kill_zone | Spot pinned near strike while premium rich and theta active | Avoid buying, consider sell-side only after risk check |
| avwap_defense | Price repeatedly defends anchored VWAP | Use as context for continuation or stop placement |
| stop_run_reclaim | Liquidity sweep quickly reclaims | Potential reversal alpha |
| liquidity_vacuum | Low participation, stretched move, poor confirmation | Mean reversion or no trade |

## Top-Weight Index Pressure

For each of the top-weight constituents, compute:

- weight in index
- price distance from today's AVWAP
- price distance from previous-session AVWAP
- price distance from weekly AVWAP
- percentile of current intraday position versus history
- gap direction and follow-through
- session return
- whether the stock supports or fights index direction

Aggregate into:

- `weighted_today_avwap_pressure`
- `weighted_prev_avwap_pressure`
- `weighted_historical_position`
- `concentration_score`
- `breadth_agreement`
- `index_masking_score`
- `sponsorship_score`

## Premium Fair-Value Map

For each option contract, compute:

- market premium
- model fair premium
- richness percentage
- spread percentage
- IV rank by moneyness
- theta burn per hour
- gamma opportunity
- expected spot move required to break even
- liquidity quality

Aggregate into:

- `premium_rich_score`
- `premium_cheap_score`
- `theta_danger_score`
- `gamma_opportunity_score`
- `tradeable_liquidity_score`

Cheap premium is not automatically a buy. It is useful only when movement probability, liquidity, and timing agree.

## Scientist Contract

Every scientist must emit:

- state name
- side or avoidance action
- expected path
- expected horizon
- invalidation
- confidence
- cost viability
- reason codes
- trust tier

No reason codes, no hypothesis.

## Miner Contract

The miner can generate many candidate rules, but validation owns truth. Each mined candidate must be serialized as:

- feature conditions
- side
- holding window
- stop/target geometry
- training slice
- validation slice
- net R after cost
- stress result
- null result
- verdict

## Graduation Ladder

| Tier | Meaning |
|---|---|
| rejected | failed validation or malformed |
| shadow | logged only, invisible to trading |
| paper | simulated in live time |
| tiny_live | small size only, strict governor |
| trusted | survived drift and decay checks |
| sized | eligible for allocator sizing |

## Immediate Build Queue

1. `liqpool/research/manipulation_atlas.py`
2. `liqpool/research/hypothesis_miner.py`
3. `analysis/run_profitability_miner.py`
4. Sentinel scientist bridge for manipulation states.
5. Curator report section for manipulation-state hit rates.
6. Capital governor integration after paper evidence exists.

