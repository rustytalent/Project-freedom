import { describe, it, expect } from "vitest";
import {
  currentTradingDay, previousTradingDay, nextTradingDay,
  buildDateline, formatLong, formatShort, formatEdition,
  editionNumber,
} from "../lib/clock";

describe("clock - IST trading day math", () => {
  it("returns today if today is a weekday", () => {
    // Mon 8 Jun 2026 (a Monday) at 03:00 UTC = 08:30 IST.
    expect(currentTradingDay(new Date("2026-06-08T03:00:00Z"))).toBe("2026-06-08");
  });
  it("falls back to Friday on Saturday", () => {
    expect(currentTradingDay(new Date("2026-06-06T10:00:00Z"))).toBe("2026-06-05");
  });
  it("falls back to Friday on Sunday", () => {
    expect(currentTradingDay(new Date("2026-06-07T10:00:00Z"))).toBe("2026-06-05");
  });
  it("yesterday of Monday is the previous Friday", () => {
    expect(previousTradingDay("2026-06-08")).toBe("2026-06-05");
  });
  it("yesterday of Friday is Thursday", () => {
    expect(previousTradingDay("2026-06-05")).toBe("2026-06-04");
  });
  it("next of Friday is the next Monday", () => {
    expect(nextTradingDay("2026-06-05")).toBe("2026-06-08");
  });
  it("formats long and short dates", () => {
    expect(formatLong("2026-06-08")).toBe("08 JUN 2026");
    expect(formatShort("2026-06-08")).toBe("08 JUN");
  });
  it("edition increments only on trading days", () => {
    // From 2026-01-15 (Thu) to 2026-01-15 inclusive = 1.
    expect(editionNumber("2026-01-15")).toBe(1);
    // Thu, Fri = 2; weekend skipped; Mon = 3.
    expect(editionNumber("2026-01-19")).toBe(3);
  });
  it("buildDateline produces consistent strings", () => {
    const d = buildDateline(new Date("2026-06-08T03:00:00Z"));
    expect(d.todayKey).toBe("2026-06-08");
    expect(d.yesterdayKey).toBe("2026-06-05");
    expect(d.nextKey).toBe("2026-06-09");
    expect(d.todayLong).toBe("08 JUN 2026");
    expect(d.yesterdayShort).toBe("05 JUN");
    expect(formatEdition(d.editionNumber)).toBe(d.edition);
    expect(d.nextBriefLabel).toBe("Tomorrow 08:30 IST");
  });
});
