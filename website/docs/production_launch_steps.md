# Production Launch Steps

Use this as the operator checklist for turning the website from a
staging product into a paid subscriber product.

## 1. Supabase

1. Create a Supabase project.
2. Create the `briefs` table from `website/README.md`.
3. Create a `subscribers` table:

```sql
create table if not exists subscribers (
  id uuid primary key default gen_random_uuid(),
  email text not null unique,
  tier text not null default 'preview',
  razorpay_customer_id text,
  razorpay_subscription_id text,
  status text not null default 'inactive',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
```

4. Enable row-level security before exposing subscriber reads.
5. Enable Google Auth.
6. Add redirect URLs:
   - `http://localhost:3000/auth/callback`
   - `https://<domain>/auth/callback`

## 2. Razorpay

1. Create Razorpay live keys.
2. Add keys to Vercel:
   - `RAZORPAY_KEY_ID`
   - `RAZORPAY_KEY_SECRET`
   - `RAZORPAY_WEBHOOK_SECRET`
3. Use test mode first and complete one Daily Brief checkout.
4. Add a webhook endpoint in the website before public launch. The
   webhook should verify the signature, upsert the subscriber email,
   and activate the paid tier.

## 3. Google Sign-In

1. Add `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   to Vercel.
2. Confirm `/sign-in` sends the customer to Google.
3. Confirm `/auth/callback` returns the customer to the site.
4. Add session persistence and portal route protection before launch.

## 4. Brief Ingest

1. Add `ENGINE_INGEST_TOKEN` to Vercel.
2. Add `SUPABASE_SERVICE_ROLE_KEY` to Vercel.
3. Send a staging brief:

```bash
curl -X POST "https://<domain>/api/v1/briefs" \
  -H "Authorization: Bearer $ENGINE_INGEST_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @daily_brief.json
```

4. Confirm the row appears in Supabase.
5. Connect email delivery after storage is verified.

## 5. Public Launch Checks

Run:

```bash
pnpm test
pnpm typecheck
pnpm build
```

Then manually check:

- `/`
- `/pricing`
- `/checkout?plan=daily&cycle=monthly`
- `/sign-in`
- `/sample-brief`
- `/track-record`
- `/portal`

No public page should disclose private technique names or exact
subscriber-only levels.
