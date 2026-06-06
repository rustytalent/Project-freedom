# Aurora Research Website

Customer-facing website and subscriber portal for the research-data
product. The public surface sells the brief, the archive, the audit
record, and account delivery. The private research engine is not
documented here.

## Local development

```bash
pnpm install
pnpm dev
# http://localhost:3000
```

Requires Node 20 or newer.

Useful checks:

```bash
pnpm typecheck
pnpm test
pnpm build
```

## Product surface

The site currently supports:

- Public marketing pages for Core Research, Live Desk, Audit Base, and
  B2B audit work.
- Public sample brief with subscriber-only levels redacted.
- Five-day signed-in preview that starts on first Google login.
- Public track record page backed by deterministic demo data until the
  live aggregate view is connected.
- Subscriber portal skeleton for today, brief archive, calibration, and
  account pages.
- Authenticated brief ingest at `POST /api/v1/briefs`.
- Razorpay order creation at `POST /api/checkout/razorpay-order`.
- Google sign-in entry at `/sign-in` when Supabase Auth is configured.

## Moat rule

This repo is public-facing. Do not place private technique names,
feature names, model class names, hyperparameters, private repo names,
or internal agent names anywhere in code, copy, tooltips, docs, sample
data, or API responses.

Allowed public language:

- Brief sections and customer-visible outcomes.
- Calibration dashboard language.
- Outcome log language.
- Product pricing, delivery, and account workflow.
- Legal and disclosure language.

Forbidden public language:

- Private research methods.
- Private model names and feature names.
- Specific validation architecture.
- Specific tuning parameters.
- Private repository or agent names.

The test suite includes a moat scan. If it fails, rewrite the public
line into customer-facing vocabulary.

## Environment variables

Copy `.env.example` to `.env.local` for local development. Production
values belong in Vercel project settings.

Required for brief ingest:

- `ENGINE_INGEST_TOKEN`
- `NEXT_PUBLIC_SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

Required for Google sign-in:

- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- Google OAuth provider enabled in Supabase Auth
- Production redirect URL set to `https://<domain>/auth/callback`

Required for Razorpay checkout:

- `RAZORPAY_KEY_ID`
- `RAZORPAY_KEY_SECRET`
- `RAZORPAY_WEBHOOK_SECRET`

Optional for delivery and operations:

- `RESEND_API_KEY`
- `SENTRY_DSN`
- `FOUNDER_EMAIL`
- `SLACK_WEBHOOK_ALERTS`

## Production setup

1. Create the Supabase project.
2. Enable Google Auth in Supabase.
3. Add the production callback URL:
   `https://<domain>/auth/callback`.
4. Create the minimum `briefs` table:

```sql
create table if not exists briefs (
  id bigserial primary key,
  brief_id text not null unique,
  trading_date_ist date not null,
  schema_version text not null,
  payload jsonb not null,
  created_at timestamptz not null default now()
);
```

5. Add row-level security policies before exposing subscriber reads.
6. Create the subscriber table with `tier = 'free_signup'`,
   `preview_started_at`, and `preview_expires_at`. First Google login
   should set a five-day preview clock and later logins should not
   reset it.
7. Create Razorpay live keys and add the keys to Vercel.
8. Create a Razorpay webhook that verifies payment events and updates
   the subscriber tier table. Keep the webhook server-side only.
9. Configure Resend for transactional email and brief delivery.
10. Deploy to Vercel and run:

```bash
pnpm test
pnpm typecheck
pnpm build
```

## Engine ingest command

Once the website is deployed, the private engine should send the
generated Daily Brief JSON to the website:

```bash
curl -X POST "https://<domain>/api/v1/briefs" \
  -H "Authorization: Bearer $ENGINE_INGEST_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @daily_brief.json
```

If Supabase is not configured, the endpoint still validates the token
and payload but returns `stored: false` with `supabase_env_missing`.

## Current status

| Concern | Status |
| --- | --- |
| Marketing pages | Production copy pass in progress |
| Sample brief render | Working with redaction |
| Track record dashboard | Demo aggregate data |
| Portal pages | Skeleton account and brief views |
| Engine ingest API | Token validation plus Supabase write |
| Razorpay checkout | Order API and checkout page wired |
| Razorpay webhook | Pending |
| Google sign-in | Entry route wired, session persistence pending |
| Resend email delivery | Pending |
| Subscriber entitlement DB | Pending |
