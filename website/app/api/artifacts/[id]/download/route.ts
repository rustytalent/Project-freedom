import { NextResponse } from "next/server";
import { canAccess, type AccessTier } from "@/lib/artifacts";
import { getStorage } from "@/lib/artifact-storage";

// Tier-gated download.
//
// In production the reader's tier is resolved from the Supabase
// session cookie. For this stub we accept an optional header
// `x-reader-tiers` (comma-separated). Missing header  treated as
// `["free_signup"]` (the default after email-signup; conservative).
// `public` artifacts always download.

export const runtime = "nodejs";

type Params = { id: string };

function parseTiers(raw: string | null): Set<AccessTier> {
  if (!raw) return new Set<AccessTier>(["free_signup"]);
  const known = new Set<AccessTier>([
    "public",
    "free_signup",
    "paid_intraday",
    "paid_multi_product",
    "paid_diagnosis",
  ]);
  return new Set<AccessTier>(
    raw
      .split(",")
      .map((s) => s.trim())
      .filter((s): s is AccessTier => known.has(s as AccessTier)),
  );
}

const CONTENT_TYPES: Record<string, string> = {
  daily_brief_pdf: "application/pdf",
  swing_brief_pdf: "application/pdf",
  diagnosis_report_pdf: "application/pdf",
  weekly_research_note_pdf: "application/pdf",
  daily_brief_email: "text/plain; charset=utf-8",
  calibration_summary_csv: "text/csv; charset=utf-8",
  outcome_log_export_csv: "text/csv; charset=utf-8",
  yesterday_audit_csv: "text/csv; charset=utf-8",
  options_strikes_csv: "text/csv; charset=utf-8",
};

export async function GET(
  req: Request,
  ctx: { params: Promise<Params> },
): Promise<Response> {
  const { id } = await ctx.params;
  const store = await getStorage();
  const rec = await store.get(id);
  if (!rec) {
    return NextResponse.json(
      { error: "not_found", id },
      { status: 404 },
    );
  }
  const readerTiers = parseTiers(req.headers.get("x-reader-tiers"));
  if (!canAccess(readerTiers, rec.tier)) {
    return NextResponse.json(
      {
        error: "tier_insufficient",
        required: rec.tier,
        reader: Array.from(readerTiers),
        upgrade_url: "/pricing",
      },
      { status: 403 },
    );
  }

  const bytes = await store.readBytes(id);
  if (!bytes) {
    return NextResponse.json({ error: "blob_missing" }, { status: 410 });
  }
  const ctype = CONTENT_TYPES[rec.kind] ?? "application/octet-stream";
  return new Response(bytes, {
    status: 200,
    headers: {
      "Content-Type": ctype,
      "Content-Length": String(rec.bytes),
      "Content-Disposition":
        `attachment; filename="${rec.filename}"`,
      "X-Artifact-Id": rec.id,
      "X-Artifact-Kind": rec.kind,
      "Cache-Control": "private, no-store",
    },
  });
}
