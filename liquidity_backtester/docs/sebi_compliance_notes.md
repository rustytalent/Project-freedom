# SEBI Compliance Notes (internal)

> **NOT LEGAL ADVICE.** This file reduces regulatory *surface area* in product
> language, fields, and outputs. It does not make the product compliant. The
> platform must be reviewed by SEBI-specialist counsel/CS before any paid public
> launch. See the open actions and warnings at the bottom.

## 1. Defense-in-depth positioning

Core framing — use this everywhere (marketing, docs, API, UI):

> **"Non-recommendatory quantitative market-structure analytics and statistical
> feature layer for independent strategy research, backtesting, and risk-model
> enhancement."**

SEBI's "research report / research services" definition is broad: it covers
buy/sell/hold opinions, **price targets, entry-exit calls, and model portfolios**,
and even "an opinion concerning securities that provides a basis for an investment
decision." So we layer defenses rather than rely on any single one:

1. No recommendation language.
2. No directional trade instruction.
3. No price targets / stop / entry / exit.
4. No model portfolios.
5. No personalized advice (capital, age, risk profile, goals).
6. No performance / profit claims.
7. Disclaimers on every output (see `liqpool/scoring.py::INTERPRETATION_NOTE`).
8. Outputs structured as **opaque statistical features**, not decisions.
9. B2B / sophisticated-user positioning.
10. Legal review marked as **required before launch**.

## 2. How the product implements this

The public feed exposes only two opaque tokens plus neutral geometry/metadata
(`liqpool/scoring.py::public_record`):

- `G` — integer 0–100 *rank* of an internal composite within a cross-section,
  quantized + per-customer watermarked. Not a probability, no units.
- `D` — opaque directional token (`+ / = / -`) with per-customer-jittered
  thresholds. Conveys directional *context*, not a probability or a call.
- `level_zone` (public chart geometry), `scope`, `as_of`, `feed_version`,
  `analytics_type`, `interpretation_note` (disclaimer).

It never emits buy/sell/hold, entry/exit, stop, target, price target,
position size, or any per-row probability. The serializer is an explicit
allow-list, enforced by `tests/test_scoring.py`.

## 3. Hard-banned language

Do not use these in user-facing copy/fields/outputs (except inside a compliance
warning list). Enforced by `scripts/compliance_lint.py`:

`buy, sell, hold, recommendation, recommended, advise, advice, call, stock call,
tip, signal, trading signal, entry, exit, entry zone, exit zone, stoploss,
stop loss, target, price target, take profit, SL, TP, long this, short this,
go long, go short, bullish call, bearish call, trade this, tradeable, best trade,
best stock, best pick, top pick, winner, sure shot, guaranteed, profit guaranteed,
high conviction, multibagger, jackpot, intraday call, option call, option tip,
model portfolio, portfolio recommendation, allocation recommendation, execute
trade, auto-trade recommendation`

## 4. Amber-zone language

Risky; only in user-facing output if explicitly approved in a LEGAL_REVIEW_ALLOWED
list: `bullish, bearish, upside, downside, directional bias, long/short bias,
expected move, likely up/down, reversal likely, breakout/breakdown likely,
opportunity, setup, confluence, conviction, rank, score for long/short,
avoid buying/selling/shorting`.

## 5. Safe replacement vocabulary

| Instead of | Use |
|---|---|
| signal | analytics event / statistical observation / feature output |
| trade setup / setup | market-structure scenario / statistical scenario |
| long/short bias | directional context metric / forward-return distribution estimate |
| buy/sell recommendation | non-recommendatory analytics output / independent research datapoint |
| target / stop / entry / exit | reference level / observed level / liquidity zone / volatility boundary (do **not** tell users to act there) |
| best stock today | highest data-quality observations / scenarios meeting statistical filters |
| profit / backtest performance | historical simulation result / model validation metric (non-guaranteed) |

## 6. Product architecture rules

- Outputs are data/feature oriented, never action oriented.
- No final "decision" fields. No fields named buy/sell/hold/signal/trade/entry/
  exit/target/stoploss.
- Rank only by neutral criteria if ranking is unavoidable: data_quality, sample
  size, volatility-regime strength, liquidity-reachability, anomaly score,
  historical observation count. Never by "profit potential" / "best trade" /
  "upside" / "long score".
- No personalization to capital/age/risk/portfolio/goals. No position sizing
  ("use 1% risk", "allocate X%").
- Do not connect analytics output directly to broker execution without separate
  legal review.

Self-test every output: *Can a retail user directly convert this into a trade
instruction? Does it sound like a stock tip? Does it make the final decision for
the user?* If yes → rewrite. *Is it a neutral feature for a user-owned strategy?*
If yes → probably okay.

## 7. CI / pre-deploy gate

Run the linter before any deploy that touches user-facing surfaces:

```bash
# fails (exit 1) on hard-banned language in the customer-facing scope
python scripts/compliance_lint.py
# full inventory (does not fail) -> regenerate the audit report
python scripts/compliance_lint.py --report docs/compliance_audit_report.md
```

`tests/test_compliance_language.py` runs the enforced check in CI. The linter
treats the model/engine code and in-browser research UI as report-only (they
legitimately model execution mechanics); only docs, the HTTP service, the feed
modules, and READMEs are enforced.

## 8. Open actions (MUST resolve before paid public launch)

- [ ] **Legal opinion** — obtain a written opinion from SEBI-specialist
  counsel/CS on whether this product is non-recommendatory in substance.
- [ ] **RA registration** — evaluate whether SEBI Research Analyst *entity*
  registration is required given outputs are security-specific and paid.
- [ ] Define contractual B2B / sophisticated-user terms (redistribution ban,
  "no advice" acknowledgement, independent-decision representation).

## 9. Warnings (do not rationalize away)

- Disclaimers do **not** override product substance. If the platform behaves
  like stock-specific investment-decision support, regulatory risk remains.
- Even probability/directional analytics may be treated as research if
  security-specific and paid.
- B2B positioning helps but does not automatically remove risk.
- Retail users increase risk. Personalization to capital/risk → IA risk.
- Broker-execution connection → risk rises sharply.
- Profit/backtest-return marketing → risk rises.
- Telegram/WhatsApp alerts that look like calls/signals → risk rises.

## 10. Remaining unresolved legal/compliance risks (for counsel)

1. Security-specific + paid analytics may still be "research".
2. The line between "opinion providing basis for a decision" and "neutral
   feature" is judgment-dependent.
3. Whether `G`/`D`, though opaque, constitute an implied opinion on securities.
4. Distribution channel risk (any future alerts product).
5. Jurisdiction of customers (resident vs non-resident; retail vs institutional).

## 11. SEBI references consulted (May 2026)

- SEBI — Guidelines for Research Analysts (Jan 2025):
  https://www.sebi.gov.in/legal/circulars/jan-2025/guidelines-for-research-analysts_90634.html
- SEBI Research Analyst FAQs (Jul 2025):
  https://www.sebi.gov.in/sebi_data/faqfiles/jul-2025/1753269723942.pdf
- "When SEBI RA Regulations do not apply" — CS Kruti Gogri:
  https://cskruti.com/when-sebi-research-analyst-regulations-do-not-apply-to-you/
- SEBI Investor — Research Analyst overview:
  https://investor.sebi.gov.in/research_analyst.html
