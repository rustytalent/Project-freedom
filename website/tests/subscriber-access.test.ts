import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ensurePreviewSubscriber } from "@/lib/subscriber-access";

const ORIGINAL_ENV = { ...process.env };

describe("subscriber preview access", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    process.env = {
      ...ORIGINAL_ENV,
      NEXT_PUBLIC_SUPABASE_URL: "https://db.example",
      SUPABASE_SERVICE_ROLE_KEY: "service-role",
    };
  });

  afterEach(() => {
    process.env = { ...ORIGINAL_ENV };
    vi.restoreAllMocks();
  });

  it("creates a five-day preview subscriber without overwriting existing rows", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(
      new Response(JSON.stringify([{ email: "reader@example.com" }]), {
        status: 201,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await ensurePreviewSubscriber({
      email: " Reader@Example.com ",
      userId: "user-123",
    });

    expect(result).toMatchObject({ stored: true, row_count: 1 });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe("https://db.example/rest/v1/subscribers?on_conflict=email");
    expect(init.headers).toMatchObject({
      prefer: "resolution=ignore-duplicates,return=representation",
    });
    const body = JSON.parse(String(init.body)) as {
      email: string;
      user_id: string;
      tier: string;
      plan_id: string;
      status: string;
      preview_started_at: string;
      preview_expires_at: string;
    };
    expect(body).toMatchObject({
      email: "reader@example.com",
      user_id: "user-123",
      tier: "free_signup",
      plan_id: "preview",
      status: "trialing",
    });
    const started = Date.parse(body.preview_started_at);
    const expires = Date.parse(body.preview_expires_at);
    expect(expires - started).toBe(5 * 24 * 60 * 60 * 1000);
  });
});
