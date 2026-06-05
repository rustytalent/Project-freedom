import { NextResponse } from "next/server";
import { getStorage } from "@/lib/artifact-storage";
import type { ArtifactRecord } from "@/lib/artifacts";

// Public sanitised list of the latest published artifacts.
//
// This endpoint returns metadata only (no download URL) so it can be
// consumed by the public landing page's "Latest published" ticker
// without exposing tier-gated content. Authenticated portal pages
// hit the same data source via `getStorage().list()` directly so
// they can show download buttons.

export const runtime = "nodejs";
export const revalidate = 60;

type PublicRow = Pick<
  ArtifactRecord,
  | "id"
  | "kind"
  | "trading_date_ist"
  | "generated_at_utc"
  | "tier"
  | "description"
> & { bytes: number };

export async function GET(): Promise<NextResponse> {
  const store = await getStorage();
  const all = await store.list({ limit: 20 });
  const rows: PublicRow[] = all.map((r) => ({
    id: r.id,
    kind: r.kind,
    trading_date_ist: r.trading_date_ist,
    generated_at_utc: r.generated_at_utc,
    tier: r.tier,
    description: r.description,
    bytes: r.bytes,
  }));
  return NextResponse.json(
    {
      as_of_utc: new Date().toISOString(),
      count: rows.length,
      artifacts: rows,
    },
    {
      headers: {
        "Cache-Control":
          "public, s-maxage=60, stale-while-revalidate=600",
      },
    },
  );
}
