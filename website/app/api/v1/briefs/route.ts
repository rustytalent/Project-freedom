import { NextResponse } from "next/server";

// Engine ingest endpoint.
//
// The brief-generation engine POSTs a generated brief JSON to this
// route with an `Authorization: Bearer <ENGINE_INGEST_TOKEN>` header.
// We validate the bearer token against the env var, store the brief
// in Supabase, and return 202 Accepted. A separate scheduled job
// then triggers email delivery via Resend.
//
// This file is the SKELETON. Supabase write + Resend trigger land
// when the engine is ready to POST against a real deploy.

export const runtime = "edge";

type BriefIngestBody = {
  schema_version: string;
  brief_metadata: {
    brief_id: string;
    trading_date_ist: string;
  };
  // Other fields validated against the full schema at Supabase write time.
  [key: string]: unknown;
};

function unauthorized(reason: string): NextResponse {
  return NextResponse.json({ error: reason }, { status: 401 });
}

export async function POST(req: Request): Promise<NextResponse> {
  const auth = req.headers.get("authorization") ?? "";
  const expected = process.env.ENGINE_INGEST_TOKEN;
  if (!expected) {
    return NextResponse.json(
      { error: "ENGINE_INGEST_TOKEN not configured on this deploy" },
      { status: 503 },
    );
  }
  if (!auth.startsWith("Bearer ")) return unauthorized("missing_bearer");
  if (auth.slice("Bearer ".length) !== expected) {
    return unauthorized("invalid_token");
  }

  let body: BriefIngestBody;
  try {
    body = (await req.json()) as BriefIngestBody;
  } catch {
    return NextResponse.json(
      { error: "invalid_json" },
      { status: 400 },
    );
  }

  if (
    !body.schema_version ||
    !body.brief_metadata?.brief_id ||
    !body.brief_metadata?.trading_date_ist
  ) {
    return NextResponse.json(
      { error: "missing_required_fields" },
      { status: 400 },
    );
  }

  // TODO: write to Supabase, enqueue delivery, update calibration view.
  return NextResponse.json(
    {
      accepted: true,
      brief_id: body.brief_metadata.brief_id,
      trading_date_ist: body.brief_metadata.trading_date_ist,
      stored: false,            // flip to true once Supabase write is wired
      next_step: "delivery_queue_pending",
    },
    { status: 202 },
  );
}
