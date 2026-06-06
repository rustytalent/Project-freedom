import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { POST } from "@/app/api/v1/briefs/route";
import { validateBriefPayload, type BriefIngestBody } from "@/lib/brief-store";

const ORIGINAL_ENV = { ...process.env };

const sampleBrief: BriefIngestBody = {
  schema_version: "1.0",
  brief_metadata: {
    brief_id: "BRIEF_2026_06_06",
    trading_date_ist: "2026-06-06",
  },
  sections: [],
};

function request(body: unknown, token = "secret-token"): Request {
  return new Request("https://example.test/api/v1/briefs", {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

describe("brief ingest validation", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    process.env = { ...ORIGINAL_ENV };
  });

  afterEach(() => {
    process.env = { ...ORIGINAL_ENV };
    vi.restoreAllMocks();
  });

  it("accepts the minimum brief contract", () => {
    expect(validateBriefPayload(sampleBrief)).toEqual({ ok: true });
  });

  it("rejects a payload without a trading date", () => {
    const invalid = {
      ...sampleBrief,
      brief_metadata: { brief_id: "BRIEF_2026_06_06" },
    };
    expect(validateBriefPayload(invalid)).toEqual({
      ok: false,
      error: "missing_trading_date_ist",
    });
  });

  it("fails closed when the ingest token is missing", async () => {
    delete process.env.ENGINE_INGEST_TOKEN;
    const response = await POST(request(sampleBrief));
    expect(response.status).toBe(503);
  });

  it("rejects an invalid bearer token", async () => {
    process.env.ENGINE_INGEST_TOKEN = "secret-token";
    const response = await POST(request(sampleBrief, "wrong-token"));
    expect(response.status).toBe(401);
  });

  it("accepts a valid brief but reports storage as unconfigured", async () => {
    process.env.ENGINE_INGEST_TOKEN = "secret-token";
    delete process.env.NEXT_PUBLIC_SUPABASE_URL;
    delete process.env.SUPABASE_SERVICE_ROLE_KEY;

    const response = await POST(request(sampleBrief));
    const payload = (await response.json()) as { stored: boolean };

    expect(response.status).toBe(202);
    expect(payload.stored).toBe(false);
  });

  it("writes a valid brief to Supabase when storage env is present", async () => {
    process.env.ENGINE_INGEST_TOKEN = "secret-token";
    process.env.NEXT_PUBLIC_SUPABASE_URL = "https://db.example";
    process.env.SUPABASE_SERVICE_ROLE_KEY = "service-role";

    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify([{ id: 1 }]), { status: 201 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(request(sampleBrief));
    const payload = (await response.json()) as { stored: boolean };

    expect(response.status).toBe(202);
    expect(payload.stored).toBe(true);
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe("https://db.example/rest/v1/briefs?on_conflict=brief_id");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toMatchObject({
      brief_id: sampleBrief.brief_metadata.brief_id,
      trading_date_ist: sampleBrief.brief_metadata.trading_date_ist,
      schema_version: sampleBrief.schema_version,
    });
  });
});
