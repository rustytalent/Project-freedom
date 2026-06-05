import { describe, expect, it } from "vitest";
import {
  canAccess,
  type AccessTier,
  relativeTimeFromNow,
} from "@/lib/artifacts";
import { getStorage } from "@/lib/artifact-storage";

describe("canAccess tier policy", () => {
  it("public artifacts are always downloadable", () => {
    expect(canAccess(new Set(), "public")).toBe(true);
    expect(canAccess(new Set<AccessTier>(["free_signup"]), "public")).toBe(true);
  });

  it("paid_intraday subscribers see free_signup and paid_intraday", () => {
    const tiers = new Set<AccessTier>(["paid_intraday"]);
    expect(canAccess(tiers, "free_signup")).toBe(true);
    expect(canAccess(tiers, "paid_intraday")).toBe(true);
    expect(canAccess(tiers, "paid_multi_product")).toBe(false);
  });

  it("multi_product subscribers inherit intraday access", () => {
    const tiers = new Set<AccessTier>(["paid_multi_product"]);
    expect(canAccess(tiers, "paid_intraday")).toBe(true);
    expect(canAccess(tiers, "paid_multi_product")).toBe(true);
  });

  it("diagnosis access does NOT come from multi_product subscription", () => {
    // Diagnosis reports are personal; the higher-tier general
    // subscription does not entitle a customer to other readers'
    // diagnosis reports.
    const multi = new Set<AccessTier>(["paid_multi_product"]);
    expect(canAccess(multi, "paid_diagnosis")).toBe(false);
    const both = new Set<AccessTier>([
      "paid_multi_product",
      "paid_diagnosis",
    ]);
    expect(canAccess(both, "paid_diagnosis")).toBe(true);
  });

  it("unauthenticated readers can only see public", () => {
    const empty = new Set<AccessTier>();
    expect(canAccess(empty, "free_signup")).toBe(false);
    expect(canAccess(empty, "paid_intraday")).toBe(false);
    expect(canAccess(empty, "paid_multi_product")).toBe(false);
  });
});

describe("relativeTimeFromNow", () => {
  it("formats fresh timestamps as 'NNs ago'", () => {
    const t = new Date(Date.now() - 4_000).toISOString();
    expect(relativeTimeFromNow(t)).toMatch(/^\d+s ago$/);
  });

  it("formats minute-scale timestamps as 'NN min ago'", () => {
    const t = new Date(Date.now() - 12 * 60 * 1000).toISOString();
    expect(relativeTimeFromNow(t)).toBe("12 min ago");
  });

  it("formats hour-scale timestamps as 'NN h ago'", () => {
    const t = new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString();
    expect(relativeTimeFromNow(t)).toBe("3 h ago");
  });

  it("formats day-scale timestamps with singular/plural", () => {
    const t1 = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    expect(relativeTimeFromNow(t1)).toBe("1 day ago");
    const t5 = new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString();
    expect(relativeTimeFromNow(t5)).toBe("5 days ago");
  });
});

describe("MemoryStorage seeded fixtures", () => {
  it("returns at least one artifact on first call", async () => {
    const store = await getStorage();
    const list = await store.list();
    expect(list.length).toBeGreaterThan(0);
  });

  it("lists newest-first by generated_at_utc", async () => {
    const store = await getStorage();
    const list = await store.list({ limit: 5 });
    for (let i = 1; i < list.length; i++) {
      expect(list[i - 1].generated_at_utc >= list[i].generated_at_utc).toBe(
        true,
      );
    }
  });

  it("filters by kind", async () => {
    const store = await getStorage();
    const daily = await store.list({ kind: "daily_brief_pdf" });
    expect(daily.length).toBeGreaterThan(0);
    for (const r of daily) {
      expect(r.kind).toBe("daily_brief_pdf");
    }
  });

  it("downloadUrl resolves for known ids and null for unknown", async () => {
    const store = await getStorage();
    const [first] = await store.list({ limit: 1 });
    const url = await store.downloadUrl(first.id);
    expect(url).toBe(`/api/artifacts/${first.id}/download`);
    expect(await store.downloadUrl("nope-not-real")).toBe(null);
  });

  it("readBytes returns the seeded content for known ids", async () => {
    const store = await getStorage();
    const [first] = await store.list({ limit: 1 });
    const bytes = await store.readBytes(first.id);
    expect(bytes).not.toBe(null);
    expect((bytes as ArrayBuffer).byteLength).toBeGreaterThan(0);
  });
});
