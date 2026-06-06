import { NextResponse } from "next/server";
import { getStorage } from "@/lib/artifact-storage";
import type { AccessTier, ArtifactKind, ArtifactRecord } from "@/lib/artifacts";

// Engine ingest endpoint for research artifacts.
//
// The brief-generation engine POSTs each artifact here once it is
// produced on the VPS. Payload is JSON for simplicity (matches the
// /api/v1/briefs pattern); file content is base64-encoded inside the
// JSON body. For files larger than ~25MB switch to a presigned-URL
// upload flow.
//
// Auth: `Authorization: Bearer <ENGINE_INGEST_TOKEN>`. Same token as
// the briefs endpoint - one shared secret keeps engine config small.

export const runtime = "nodejs";
// Engine artifacts can be up to a few MB.
export const maxDuration = 30;

type UploadBody = {
  kind: ArtifactKind;
  filename: string;
  tier: AccessTier;
  // Base64-encoded artifact bytes.
  content_b64: string;
  sha256?: string;
  trading_date_ist?: string;
  description?: string;
  meta?: Record<string, unknown>;
};

function unauthorized(reason: string): NextResponse {
  return NextResponse.json({ error: reason }, { status: 401 });
}

function badRequest(reason: string): NextResponse {
  return NextResponse.json({ error: reason }, { status: 400 });
}

const VALID_KINDS: ReadonlySet<ArtifactKind> = new Set<ArtifactKind>([
  "daily_brief_pdf",
  "daily_brief_email",
  "swing_brief_pdf",
  "diagnosis_report_pdf",
  "calibration_summary_csv",
  "outcome_log_export_csv",
  "yesterday_audit_csv",
  "options_strikes_csv",
  "weekly_research_note_pdf",
]);

const VALID_TIERS: ReadonlySet<AccessTier> = new Set<AccessTier>([
  "public",
  "free_signup",
  "paid_intraday",
  "paid_multi_product",
  "paid_diagnosis",
]);

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
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
  if (auth.slice(7) !== expected) return unauthorized("invalid_token");

  let body: UploadBody;
  try {
    body = (await req.json()) as UploadBody;
  } catch {
    return badRequest("invalid_json");
  }
  if (!body.kind || !VALID_KINDS.has(body.kind)) {
    return badRequest("invalid_kind");
  }
  if (!body.tier || !VALID_TIERS.has(body.tier)) {
    return badRequest("invalid_tier");
  }
  if (!body.filename || typeof body.filename !== "string") {
    return badRequest("missing_filename");
  }
  if (!body.content_b64 || typeof body.content_b64 !== "string") {
    return badRequest("missing_content_b64");
  }

  let bytes: ArrayBuffer;
  try {
    const buf = Buffer.from(body.content_b64, "base64");
    bytes = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  } catch {
    return badRequest("invalid_base64");
  }

  const sha = await sha256Hex(bytes);
  if (body.sha256 && body.sha256.toLowerCase() !== sha) {
    return badRequest("sha256_mismatch");
  }

  const id = `${body.kind}-${sha.slice(0, 16)}`;
  const rec: ArtifactRecord = {
    id,
    kind: body.kind,
    trading_date_ist: body.trading_date_ist,
    generated_at_utc: new Date().toISOString(),
    filename: body.filename,
    bytes: bytes.byteLength,
    sha256: sha,
    tier: body.tier,
    storage_key: id,
    description: body.description,
    meta: body.meta,
  };

  const store = await getStorage();
  await store.put(rec, bytes);

  return NextResponse.json(
    {
      accepted: true,
      id,
      kind: rec.kind,
      bytes: rec.bytes,
      sha256: sha,
      tier: rec.tier,
    },
    { status: 202 },
  );
}
