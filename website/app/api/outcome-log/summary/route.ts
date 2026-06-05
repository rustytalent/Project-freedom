import { NextResponse } from "next/server";
import {
  mockDriftFlags,
  mockLatestSummary,
  mockRetrospectiveShare,
  mockTimeSeries,
} from "@/lib/outcome-log-mock";

// Public sanitised read of the outcome log.
//
// This endpoint is what the public Track Record page consumes. It
// returns ONLY aggregate per-bucket metrics — never prediction_ids,
// never specific symbols or levels. The shape matches the
// `outcome_log_public_view` Supabase view that the engine populates
// nightly.
//
// CRITICAL: this route must NEVER leak per-prediction detail. If you
// add fields, audit them against §2 of the website spec
// (docs/website_codex_prompt.md in the engine repo).

export const runtime = "edge";
// Revalidate hourly; the engine populates the view nightly so an
// hour-stale read is fine.
export const revalidate = 3600;

export async function GET(): Promise<NextResponse> {
  // TODO: replace mock with a real Supabase read once the public-view
  // table is provisioned.
  const payload = {
    as_of_utc: new Date().toISOString(),
    buckets: mockLatestSummary(),
    time_series: mockTimeSeries(),
    drift_flags: mockDriftFlags(),
    retrospective_share: mockRetrospectiveShare(),
  };
  return NextResponse.json(payload, {
    headers: {
      "Cache-Control":
        "public, s-maxage=3600, stale-while-revalidate=86400",
    },
  });
}
