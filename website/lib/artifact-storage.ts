// Storage adapter for artifact uploads.
//
// V1 ships with two adapters:
//   * `MemoryStorage` — module-singleton, persists for the lifetime
//     of the serverless instance. Seeded with mock artifacts so the
//     portal renders something meaningful on first deploy.
//   * `VercelBlobStorage` — production target. SKELETON; wired when
//     BLOB_READ_WRITE_TOKEN env var is set. Uses @vercel/blob.
//
// The interface is intentionally small. Storage adapters do NOT
// implement tier gating — that lives at the API/page layer, where
// the reader's identity is known.
import type { ArtifactRecord } from "./artifacts";

export interface ArtifactStorage {
  put(rec: ArtifactRecord, content: ArrayBuffer): Promise<void>;
  get(id: string): Promise<ArtifactRecord | null>;
  list(filter?: {
    kind?: ArtifactRecord["kind"];
    limit?: number;
  }): Promise<ArtifactRecord[]>;
  /** Resolve an artifact to a URL the browser can fetch.
   *  TTL is advisory; the in-memory adapter ignores it. */
  downloadUrl(id: string, ttl_seconds?: number): Promise<string | null>;
  /** Read raw bytes — used by the in-process download proxy when
   *  the adapter doesn't expose signed URLs. */
  readBytes(id: string): Promise<ArrayBuffer | null>;
}

class MemoryStorage implements ArtifactStorage {
  private records = new Map<string, ArtifactRecord>();
  private blobs = new Map<string, ArrayBuffer>();

  async put(rec: ArtifactRecord, content: ArrayBuffer): Promise<void> {
    this.records.set(rec.id, rec);
    this.blobs.set(rec.id, content);
  }

  async get(id: string): Promise<ArtifactRecord | null> {
    return this.records.get(id) ?? null;
  }

  async list(
    filter?: { kind?: ArtifactRecord["kind"]; limit?: number },
  ): Promise<ArtifactRecord[]> {
    let all = Array.from(this.records.values());
    if (filter?.kind) {
      all = all.filter((r) => r.kind === filter.kind);
    }
    // Newest first.
    all.sort((a, b) =>
      b.generated_at_utc.localeCompare(a.generated_at_utc),
    );
    return filter?.limit ? all.slice(0, filter.limit) : all;
  }

  async downloadUrl(id: string): Promise<string | null> {
    if (!this.records.has(id)) return null;
    // Memory adapter routes downloads via /api/artifacts/[id]/download
    return `/api/artifacts/${id}/download`;
  }

  async readBytes(id: string): Promise<ArrayBuffer | null> {
    return this.blobs.get(id) ?? null;
  }
}

// Module singleton. Vercel keeps the serverless instance warm across
// requests within a short window; for cold starts the seeded mock
// data ensures the portal always shows SOMETHING. Real persistence
// will land when we switch this for the Vercel Blob / Supabase
// adapter.
const memory = new MemoryStorage();
let seeded = false;

async function seedOnce(): Promise<void> {
  if (seeded) return;
  seeded = true;
  const now = new Date();
  const isoToday = now.toISOString().slice(0, 10);
  const isoYday = new Date(now.getTime() - 86400_000)
    .toISOString()
    .slice(0, 10);
  const iso2 = new Date(now.getTime() - 2 * 86400_000)
    .toISOString()
    .slice(0, 10);
  const iso3 = new Date(now.getTime() - 3 * 86400_000)
    .toISOString()
    .slice(0, 10);
  const isoWeek = new Date(now.getTime() - 7 * 86400_000)
    .toISOString()
    .slice(0, 10);
  const fixtures: Array<{
    rec: ArtifactRecord;
    body: string;
  }> = [
    {
      rec: {
        id: `demo-daily-${isoToday}`,
        kind: "daily_brief_pdf",
        trading_date_ist: isoToday,
        generated_at_utc: new Date(now.getTime() - 12 * 60 * 1000).toISOString(),
        filename: `daily-brief-${isoToday}.pdf`,
        bytes: 184_300,
        sha256: "demo-fixture",
        tier: "paid_intraday",
        storage_key: "memory",
        description:
          "Today's Daily Brief. PDF mirror of the email + portal render.",
      },
      body: `Daily Brief — ${isoToday}\n(Mock PDF fixture for first-deploy demo.)`,
    },
    {
      rec: {
        id: `demo-daily-${isoYday}`,
        kind: "daily_brief_pdf",
        trading_date_ist: isoYday,
        generated_at_utc: new Date(now.getTime() - 26 * 60 * 60 * 1000).toISOString(),
        filename: `daily-brief-${isoYday}.pdf`,
        bytes: 178_200,
        sha256: "demo-fixture",
        tier: "paid_intraday",
        storage_key: "memory",
      },
      body: `Daily Brief — ${isoYday}\n(Mock PDF fixture.)`,
    },
    {
      rec: {
        id: `demo-daily-${iso2}`,
        kind: "daily_brief_pdf",
        trading_date_ist: iso2,
        generated_at_utc: new Date(now.getTime() - 2 * 86400_000 + 30 * 60 * 1000).toISOString(),
        filename: `daily-brief-${iso2}.pdf`,
        bytes: 181_500,
        sha256: "demo-fixture",
        tier: "paid_intraday",
        storage_key: "memory",
      },
      body: `Daily Brief — ${iso2}\n(Mock.)`,
    },
    {
      rec: {
        id: `demo-daily-${iso3}`,
        kind: "daily_brief_pdf",
        trading_date_ist: iso3,
        generated_at_utc: new Date(now.getTime() - 3 * 86400_000 + 30 * 60 * 1000).toISOString(),
        filename: `daily-brief-${iso3}.pdf`,
        bytes: 179_000,
        sha256: "demo-fixture",
        tier: "paid_intraday",
        storage_key: "memory",
      },
      body: `Daily Brief — ${iso3}\n(Mock.)`,
    },
    {
      rec: {
        id: `demo-swing-${isoWeek}`,
        kind: "swing_brief_pdf",
        trading_date_ist: isoWeek,
        generated_at_utc: new Date(now.getTime() - 7 * 86400_000).toISOString(),
        filename: `swing-brief-week-of-${isoWeek}.pdf`,
        bytes: 256_700,
        sha256: "demo-fixture",
        tier: "paid_multi_product",
        storage_key: "memory",
        description:
          "Multi-day positional research for the trading week of " +
          isoWeek + ".",
      },
      body: `Swing Brief — week of ${isoWeek}\n(Mock.)`,
    },
    {
      rec: {
        id: `demo-calibration-${isoToday}`,
        kind: "calibration_summary_csv",
        generated_at_utc: new Date(now.getTime() - 90 * 60 * 1000).toISOString(),
        filename: `calibration-summary-last-90d.csv`,
        bytes: 9_840,
        sha256: "demo-fixture",
        tier: "free_signup",
        storage_key: "memory",
        description:
          "Per-bucket calibration for the last 90 trading days. " +
          "The same numbers you see on the public dashboard, in CSV.",
      },
      body:
        "trading_date_ist,prediction_type,confidence_bucket,n,hit_rate," +
        "mean_predicted_p,calibration_error,is_retrospective_share\n" +
        `${isoToday},proximity,high,152,0.71,0.72,0.01,0.35\n` +
        `${isoToday},proximity,moderate,204,0.58,0.57,-0.01,0.35\n` +
        `${isoToday},avoidance,moderate,42,0.86,0.83,-0.03,0.20\n`,
    },
    {
      rec: {
        id: `demo-yaudit-${isoYday}`,
        kind: "yesterday_audit_csv",
        trading_date_ist: isoYday,
        generated_at_utc: new Date(now.getTime() - 60 * 60 * 1000).toISOString(),
        filename: `yesterday-audit-${isoYday}.csv`,
        bytes: 2_140,
        sha256: "demo-fixture",
        tier: "free_signup",
        storage_key: "memory",
        description:
          "Per-bucket hit-rate audit for the previous IST trading session.",
      },
      body:
        "prediction_type,confidence_bucket,n,hit_rate,mean_p,calibration_error\n" +
        "proximity,very_high,2,0.50,0.84,0.34\n" +
        "proximity,high,5,0.80,0.71,-0.09\n" +
        "proximity,moderate,3,0.33,0.55,0.22\n" +
        "avoidance,moderate,2,1.00,1.00,0.00\n",
    },
  ];
  for (const { rec, body } of fixtures) {
    const enc = new TextEncoder();
    const bytes = enc.encode(body);
    // ArrayBuffer of the encoded text.
    await memory.put(rec, bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    ));
  }
}

export async function getStorage(): Promise<ArtifactStorage> {
  // TODO: when BLOB_READ_WRITE_TOKEN is present, return
  // VercelBlobStorage instead of memory + seeded fixtures.
  await seedOnce();
  return memory;
}
