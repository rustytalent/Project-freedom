# Sample Daily Research Brief — illustrative

> This document shows the EXACT shape and content of the artifact a
> pilot customer receives in their inbox each morning. It is rendered
> from the same `BriefDocument` schema (`docs/daily_brief_schema.md`)
> via `liqpool/products/brief_renderer.py` — the tipster-vocabulary
> guardrail enforced at render time means everything below is
> regime-and-probability language, never trade instructions.
>
> The example below uses the verdict and per-asset readouts from the
> `core25_head_alpha_710362b` bundle. A real customer brief for
> 2026-06-04 would render with that day's bundle as the source.

---

```
Daily Research Brief — 2026-06-04
Generated 2026-06-04T02:30:00Z   bundle=core25_head_alpha@1e9a9de
Indexes covered: NIFTY50, BANKNIFTY, FINNIFTY, NIFTYMIDCAPSELECT, SENSEX
Estimated reading time: ~6 min

────────────────────────────────────────────────────────────────────

TLDR — Today the model finds NO instrument across the 25-stock basket
with actionable conviction for an intraday same-session move. The
structural regime is set to stand-aside: probabilities of any monitored
level being tested same-session remain below the conviction threshold
across the basket. Index-options buyers face elevated theta-danger;
sellers face moderate conditions. Yesterday's audit confirms model
calibration is within band on 73 of 79 logged predictions.

────────────────────────────────────────────────────────────────────

INDEX REGIME

  NIFTY50         vol regime normal (z=+0.34)   gap classification:
                  expected small_gap_up (0.42 ATR). Session character
                  prediction: range_day 55% / trend_day 30% / chop_day
                  15%. Trap risk score 0.18 (low).

  BANKNIFTY       vol regime normal (z=+0.11)   gap classification:
                  expected flat. Session character: range_day 60%.
                  Trap risk 0.22.

  FINNIFTY        vol regime elevated (z=+1.05) gap classification:
                  expected small_gap_down (-0.31 ATR). Session
                  character: chop_day 50%. Trap risk 0.41 (moderate).

  NIFTYMIDCAPSELECT  vol regime normal (z=-0.08)
  SENSEX          vol regime normal (z=+0.28)

────────────────────────────────────────────────────────────────────

OPTIONS SUITABILITY

  NIFTY50         directional bias mild_up; expected today range
                  392 points (1.6 ATR); movement quality prediction
                  clean_60pct_choppy_40pct. Premium regime:
                  buyers MODERATE / sellers MODERATE. Theta-danger
                  score 0.42. Strike levels in play today:
                    24500  P(test today) 78%  side: below   demand_pool
                    24700  P(test today) 41%  side: above   supply_pool

  BANKNIFTY       directional bias neutral; expected range 480 points.
                  Premium regime: buyers UNFAVOURABLE / sellers
                  FAVOURABLE. Theta-danger 0.68 (elevated chop signal).
                  Strike levels in play today:
                    52000  P(test today) 33%  side: above   supply_pool
                    51500  P(test today) 28%  side: below   demand_pool

  FINNIFTY        chop predicted; premium buyers UNFAVOURABLE.
                  No strike levels cleared the 15% conviction floor.

  NIFTYMIDCAPSELECT  premium regime UNCONVENTIONAL — limited liquidity
                     in detected pools today. Skip.

  SENSEX          premium regime MODERATE both sides. Strike levels:
                    81500  P(test today) 35%  side: above   supply_pool

────────────────────────────────────────────────────────────────────

SECTOR REGIME

  Trending up: (none).
  Trending down: FMCG.
  Chopping: AUTO, PHARMA, ENERGY.
  Neutral: BANKING, IT.

  Leadership change vs yesterday: BANKING flipped from
  trending_up (yesterday) to neutral (today); FMCG entered
  trending_down conviction.

  Money rotation reading: BANKING leadership weakening on 5-day basis
  while FMCG enters multi-day weakness. Sectoral capital appears to
  be rotating OUT of FMCG and OUT of high-conviction BANKING into a
  more cautious basket-wide neutral stance.

────────────────────────────────────────────────────────────────────

WATCHLIST — instruments the model flags as in-play today

  (none cleared the proximity threshold)

  The strongest individual-stock proximity reading in the basket was
  HCLTECH @ ₹1182, with P(test today) for its nearest active demand
  pool at 12%. The conviction threshold for the watchlist is 20%.
  All other 24 instruments were below 10%.

  This is consistent with the regime read above (range-day predicted
  on index; sector leadership weakening). The watchlist remains
  empty by design — we don't lower the threshold to find content.

────────────────────────────────────────────────────────────────────

AVOID — the model recommends standing aside in these contexts

  ALL_BASKET                no_setup_today_low_proximity_across_basket
                            confidence: HIGH
                            Max same-session touch probability across
                            all 25 instruments is 12% (HCLTECH). The
                            structural read is "stand aside today;
                            re-evaluate tomorrow's bundle".

  ALL_FMCG                  sector_trending_down_avoid_new_longs
                            confidence: MODERATE
                            FMCG sector in confirmed downtrend on
                            both 5-day and 20-day windows; new longs
                            on HINDUNILVR / ITC / NESTLEIND /
                            BRITANNIA / DABUR face structural
                            sector headwind.

  ALL_INDEX_OPTIONS_BUYERS  elevated_theta_danger_on_chop_predicted
                            confidence: MODERATE
                            Movement-quality prediction skews
                            choppy_in_range on BANKNIFTY (theta-danger
                            0.68) and FINNIFTY (0.71). Premium decay
                            is the dominant risk for buyers today
                            regardless of directional thesis.

────────────────────────────────────────────────────────────────────

CONFIDENCE NOTES

  Calibrated today:
    proximity_h12, proximity_h36, proximity_h60, direction_60bar,
    reaction_strict, reaction_break_continuation

  Drifting today:
    quality_blended (calibration error +4.7% on BANKING; demoted to
    context-only — not used as a hard gate)

  Overall brief confidence: MODERATE-TO-HIGH for proximity-derived
  sections (the spine of today's read); LOW for any Q-derived
  ranking (Q model is in known-weak regime).

────────────────────────────────────────────────────────────────────

YESTERDAY AUDIT — 2026-06-03

  Predictions made:       79
  Resolved (clean):       73
  Data gaps:               6
  Pending:                 0

  Hit rate by confidence bucket:
                       n   mean predicted   actual hit rate   error
    very_high          8        82.4%             75.0%       -7.4%
    high              19        67.8%             63.2%       -4.6%
    moderate          31        51.2%             54.8%       +3.6%
    low               15        29.1%             33.3%       +4.2%

  Avoid list validation:
    ALL_BASKET (avoid) validated      basket lost on avg -0.28 ATR
                                      across instruments yesterday
    ALL_FMCG (avoid)   validated      FMCG basket lost -0.62 ATR
                                      avg on the day

  Directional bias results:
    NIFTY50:   matched (mild_up read; index closed +0.4%)
    BANKNIFTY: matched (neutral; index closed -0.1%)
    FINNIFTY:  partial_match
    Sensex:    matched
    Midcap:    wrong (read trending up; closed -0.6%)

  Notable miss:
    HCLTECH — predicted P(test today) 22% on the 1198 supply pool;
              actual test confirmed in afternoon session. Outcome
              was a sweep-and-reclaim pattern — pool was touched but
              not respected. Our proximity prediction was CORRECT in
              direction and conviction; our post-touch reaction
              prediction (strict_respect 56%) was wrong.
              Diagnosis: post-touch reaction prediction in chop-
              flagged sessions is currently noisy. We flag chop-day
              context next morning to reduce overreliance on the
              reaction model's confidence when chop is predicted.

────────────────────────────────────────────────────────────────────

This brief is research context, NOT trade instructions. The
probabilities reflect the model's calibrated view; the decision to
act on any of it remains entirely with the reader.

If the avoid-list signal saves you one losing day this month, the
brief has paid for itself. If you find the audit section reveals a
miss you wouldn't have caught without it, the brief is doing its job.

Reply with any specific instrument you'd like included in tomorrow's
audit breakdown.

— end of brief —
```

---

## What this sample demonstrates

**The customer-facing principles in action:**

1. **TLDR first** — a sophisticated reader gets the verdict in 4
   sentences before deciding whether to read the rest.

2. **Index regime as a 2-line per-index block** — vol, gap,
   character, trap risk. Total ~30 seconds per index.

3. **Options suitability is regime-language, not action-language.**
   Notice we say "Premium regime: buyers MODERATE / sellers FAVOURABLE"
   — never "buy puts" or "sell calls". The customer maps the regime
   to their own thesis.

4. **The AVOID list is the most valuable section.** "Stand aside
   today" prevents losses, which compound the same way wins do.
   This is also the SEBI-safest content — helping someone NOT
   trade is unambiguously not advisory.

5. **CONFIDENCE NOTES expose model health honestly.** "Q model in
   known-weak regime; not used as a hard gate" is exactly the
   level of transparency that turns a research firm into a
   trusted vendor instead of a tipster.

6. **YESTERDAY AUDIT compounds.** Every brief includes the prior
   day's predictions and what happened. Three months in, the
   audit section IS the marketing. A prospective customer reading
   90 days of "here's what we said, here's what happened, here's
   where we were wrong" cannot be sold to by a competing tipster.

7. **The notable miss section.** Most services hide losses. We
   expose them with diagnoses. This is the rare-honesty mechanism
   that's the moat.

**What's NOT here (deliberately):**
- No "buy", "sell", "go long", "go short", "entry at", "stop loss",
  "target at", "book profit". The tipster guardrail in
  `brief_renderer.py` raises at render time on any of these.
- No instruction tone. Every sentence is descriptive or contextual.
- No specific position sizing. We don't know the customer's capital.
- No tier-specific features. This is the standard brief shape;
  T1/T2/T3/T4 tiers would adjust depth, not change the structural
  shape.

## Pricing implications this sample reveals

After reading this artifact:

- A discretionary trader who currently makes 20 trades/month and
  loses on 5 chop days where the avoid-list would have flagged it
  has measurable economic value. Even at ₹2,000 saved per chop-day
  loss × 5 days = ₹10,000/month value, paying ₹3,000-5,000/month
  for the brief is rational.

- An algo trader who can integrate the JSON layer programmatically
  gets a calibrated probability feed they cannot generate themselves.
  ₹10,000-25,000/month is rational.

- A small prop desk using it as a primary research input would pay
  ₹50,000-1,00,000/month and still find it cheaper than running an
  equivalent research function in-house.

Three tiers, three audiences, same brief output with progressively
richer JSON depth.

## What changes if Codex's experiments come back positive

If Task 1 (top-decile R1 filter) or Task 2 (journey alpha net R)
land positive, this same sample brief gets a NEW section called
**SELECTIVE OPPORTUNITY** that surfaces the top 1-2 trades-with-edge
identified by the live filter, framed as: "the model identifies
[symbol] at [level] as a top-decile predicted-R candidate today
under [mode]. This is research context — execution is the reader's
decision."

That's the optional upgrade path. It doesn't change the brief's
fundamental shape; it adds one section.

If both come back negative, the brief stays exactly as above —
which is fine. The avoid-list and the audit alone are worth
₹5,000-10,000/month to a serious customer.
