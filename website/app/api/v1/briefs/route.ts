import { NextResponse } from "next/server";
import {
  storeBrief,
  validateBriefPayload,
  type BriefIngestBody,
} from "@/lib/brief-store";

// Engine ingest endpoint.
//
// The brief-generation engine POSTs a generated brief JSON to this
// route with an `Authorization: Bearer <ENGINE_INGEST_TOKEN>` header.
// We validate the bearer token against the env var, store the brief
// in Supabase, and return 202 Accepted. A separate scheduled job
// then triggers email delivery via Resend.
//
// Delivery email is still handled by the scheduled sender. This
// endpoint's contract is storage: if Supabase env vars are present,
// a valid brief is persisted before we return.

export const runtime = "edge";

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

  const validation = validateBriefPayload(body);
  if (!validation.ok) {
    return NextResponse.json(
      { error: validation.error },
      { status: 400 },
    );
  }

  const storage = await storeBrief(body);
  return NextResponse.json(
    {
      accepted: true,
      brief_id: body.brief_metadata.brief_id,
      trading_date_ist: body.brief_metadata.trading_date_ist,
      stored: storage.stored,
      storage,
      next_step: storage.stored
        ? "stored_pending_delivery"
        : "configure_storage_then_retry",
    },
    { status: 202 },
  );
}
