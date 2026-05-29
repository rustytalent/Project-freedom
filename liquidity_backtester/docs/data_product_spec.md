# Market-Structure Analytics Feed — Data Product Spec

Audience: engineers and quantitative researchers integrating the feed into their
own independent strategy research, backtesting, and risk models.

This feed is **non-recommendatory market-structure analytics**. It is a
statistical feature layer for a user-owned process. It is not investment advice,
not a recommendation, and not an instruction to trade.

## What the feed provides

For an instrument on a given day, the feed returns a set of **market-structure
observations** — one per liquidity zone currently in the structure. Each
observation carries two opaque analytics fields plus the zone's price geometry.

### `feature_intensity_score` — opaque analytics score (integer 0–100)

An opaque, unitless ordinal score. Within a day's cross-section, a higher score
corresponds to a liquidity zone that is, by the platform's internal analytics,
*more frequently reached* in historical market-structure behavior.

- It is **not** a probability and has **no units**. It is not a confidence in any
  trade.
- It is an ordering within the day's cross-section, quantized to an integer.
- It is **watermarked per customer**: two customers see consistent orderings but
  numerically distinct values. Do not compare scores across customers.
- The method is proprietary and intentionally not disclosed. Measure its
  usefulness empirically against your own outcomes (see "Validating it").

### `feature_state` — opaque, non-directional class label

An opaque class label from a small abstract set (`state_1`, `state_2`,
`state_3`). It is a statistical feature class, **not a trade direction**, not a
probability, not a price expectation, and not an instruction. Any meaning is for
you to characterize empirically against your own data; the feed asserts none.

### Other fields

| field | meaning |
|---|---|
| `analytics_type` | `market_structure_observation` (per record); feed root is `market_structure_observation_feed` |
| `instrument` | the symbol queried |
| `level_zone` | the zone's price geometry: `{low, high, mid}` |
| `level_usage_note` | reminder that the zone is a reference only, not an actionable level |
| `score_explanation` | reminder that the score is a statistical feature, not a direction |
| `scope` | opaque code for the observation's analytical scope |
| `as_of` | timestamp the observation was computed for |
| `feed_version` | feed schema version |
| `compliance_tag` | compact classification; the full notice is at the feed root in `compliance_notice` |

The feed deliberately does **not** expose probabilities, model internals,
features, position sizing, or any reference to entering/exiting positions.

## API

```
GET /v1/levels?symbol=<SYMBOL>&date=<YYYY-MM-DD>
Header: X-API-Key: <your key>
```

Response (envelope):

```json
{
  "analytics_type": "market_structure_observation_feed",
  "instrument": "HDFCBANK.NS",
  "as_of": "2026-05-29",
  "feed_version": "g-feed/1",
  "observation_count": 3,
  "observations": [
    {
      "analytics_type": "market_structure_observation",
      "instrument": "HDFCBANK.NS",
      "feature_intensity_score": 100,
      "feature_state": "state_2",
      "level_zone": {"low": 1655.0, "high": 1660.0, "mid": 1657.5},
      "level_usage_note": "Reference zone only; not an entry, exit, stoploss, target, or execution instruction.",
      "score_explanation": "Opaque statistical market-structure feature. Not a trade direction and not a probability.",
      "scope": "s0",
      "compliance_tag": "non_recommendatory_market_analytics"
    }
  ],
  "compliance_notice": {
    "classification": "non_recommendatory_market_analytics",
    "notice": "... full non-recommendatory disclaimer ..."
  }
}
```

`401` — missing/invalid key. `429` — per-key rate limit exceeded.

## Validating it (without us revealing the method)

We publish an aggregate reliability table built on out-of-sample history: e.g.,
zones in the `feature_intensity_score` 90–100 band were reached materially more
often than zones in the 0–10 band, over a stated sample size. This lets you
confirm the analytics carry information while the recipe stays proprietary. You
are encouraged to reproduce the same characterization on your own data.

## Suggested integration patterns

The feed is an **input to your own process**, never a decision:

- Use `feature_intensity_score` as a feature in a user-owned ranking/filter
  inside your research.
- Use `feature_state` as an additional opaque feature alongside your own factors.
- Use `level_zone` geometry as reference points in your own analysis.

The feed does not tell you what to do at any level. All decisions, validation,
and execution are entirely yours.

## Limitations

- Historical statistics do not guarantee future outcomes.
- `feature_intensity_score` and `feature_state` are opaque and watermarked; they
  are not directly comparable across customers and cannot be reverse-engineered
  into the underlying analytics.
- Redistribution, resale, or derivation of competing datasets from the feed is
  prohibited by the terms of service.

## Disclaimer

This output is non-recommendatory market analytics for informational and research use only. It is not investment advice, not a buy/sell/hold recommendation, not a price target, and not an instruction to trade. Users are solely responsible for independent validation and decisions. Historical statistics do not guarantee future outcomes.
