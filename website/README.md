# Aurora Research — Website

Customer-facing website and authenticated subscriber portal for the
market-intelligence operating system. Built per the spec in
`docs/website_codex_prompt.md` of the engine repo.

> Front of house only. The engine, models, detectors, and calibration
> architecture live in the private engine repo and are NEVER
> documented here. See **§ The moat constraint** below.

## Local development

```bash
pnpm install
pnpm dev
# → http://localhost:3000
```

Requires Node 20+. Tested on Node 22.

Other commands:

```bash
pnpm typecheck   # tsc --noEmit
pnpm lint        # next lint
pnpm test        # vitest (renderer + moat tests)
pnpm build       # next build
```

## Architecture

```
                ┌─────────────────────────┐
                │  engine (private repo)  │
                │  generates brief JSON   │
                └────────────┬────────────┘
                             │
                             │  POST /api/v1/briefs
                             │  Bearer ENGINE_INGEST_TOKEN
                             ▼
   ┌──────────────────────────────────────────────────┐
   │                  website (this repo)             │
   │                                                  │
   │  app/(marketing)  ← public surface               │
   │  app/(portal)     ← subscriber surface (auth)    │
   │  app/api          ← ingest + sanitised reads     │
   │                                                  │
   │  Supabase: subscribers, briefs, outcome_log_view │
   │  Razorpay: subscription payments                 │
   │  Resend:   transactional + brief delivery email  │
   └──────────────────────────────────────────────────┘
                             │
                             ▼
                       subscriber inbox
                       + portal browser
```

## Brief content

Brief JSON arrives via authenticated POST from the engine and is
stored in Supabase. The web view is the canonical render
(`lib/brief-render.ts`); the email is a copy. The same renderer
enforces the tipster-vocabulary guardrail at render time so a
forbidden phrase fails loudly before reaching any customer.

Sample brief content lives in `content/briefs/`. The public sample
brief is a real historical artifact with symbols category-anonymised
and specific levels replaced with `₹[subscriber-only]` tokens. The
subscriber portal in production renders the un-redacted JSON for
authenticated readers.

## § The moat constraint (READ THIS)

This codebase is the public surface of a product whose competitive
defensibility lives in a private engine. The methodology that powers
our briefs — the detectors, the model architectures, the feature
engineering, the calibration techniques, the hyperparameters — must
NEVER appear in this repo. Not in copy, not in tooltips, not in code
comments, not in API responses, not in the rendered HTML.

The `tests/moat.test.ts` test grep-scans every file in the repo
against a forbidden-name list. It runs on every CI build. Adding
internal technique names anywhere in the codebase fails the build.

Allowed surface (output, discipline, philosophy, track record):

- The brief OUTPUT shape, prose, and section names.
- The tipster-guardrail policy and discipline language.
- The outcome-log calibration dashboard (per-bucket hit rate,
  calibration error, drift flags, retrospective-share disclosure).
- The principle list: causality, no lookahead, fail-closed, audit-first.
- Product tier scope and pricing.

Forbidden surface (technique):

- Detector names from any internal feature module.
- Model class names, AUCs, calibration architecture details.
- Feature names from the internal featurizer.
- Specific hyperparameters of any kind.
- Internal repo / module / agent names.

Heuristic for any new content: if a competitor reads the line and the
only thing they can clone is the aesthetic of how research is
delivered, fine. If they can extract a technique, a feature, an
architecture, or a hyperparameter, the line gets cut.

When in doubt, leave it out.

## Environment variables

See `.env.example`. None of these may ever land in the repo with real
values. Vercel project settings hold them in production. For local
dev, copy `.env.example` to `.env.local` and fill what you need —
the site degrades gracefully with mock data when external services
are not configured.

## Deploy

Production target: Vercel.

1. Push the `website/` subtree to a separate repo (or run the
   monorepo deploy directly from this path).
2. Create a Vercel project pointing at the website root.
3. Add the env vars from `.env.example`.
4. First deploy: review the preview URL, hit every page, run
   Lighthouse.
5. Promote to production once `pnpm test`, `pnpm typecheck`, and
   `pnpm lint` are clean.

## What's stubbed vs production

| Concern                  | Status                                    |
| ------------------------ | ----------------------------------------- |
| Marketing pages          | Production-ready, real copy               |
| Sample brief render      | Production-ready, byte-aligned to engine  |
| Track record dashboard   | Working with deterministic mock data      |
| Portal pages             | Skeleton with stubbed auth and brief data |
| Engine ingest API        | Skeleton; validates bearer token          |
| Outcome-log summary API  | Returns mock data; swap for Supabase read |
| Razorpay subscribe flow  | Not wired (CTAs link to /contact)         |
| Resend email delivery    | Not wired                                 |
| Supabase auth + tier DB  | Not wired                                 |
| Sentry / Plausible       | Not wired                                 |

Each stub is marked with a `TODO` comment pointing at the integration
to wire when keys are available.
