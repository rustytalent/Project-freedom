import { describe, expect, it } from "vitest";
import {
  TIPSTER_VOCABULARY,
  assertNoTipsterLanguage,
  renderEmail,
  type BriefDocument,
} from "@/lib/brief-render";
import sampleBrief from "@/content/briefs/sample-2026-03-14.json";

describe("tipster vocabulary guardrail", () => {
  it("raises when a forbidden phrase appears", () => {
    expect(() =>
      assertNoTipsterLanguage(
        "the model says buy reliance at 1200 with stop loss at 1180",
      ),
    ).toThrow(/buy/);
  });

  it("passes on calibrated research language", () => {
    expect(() =>
      assertNoTipsterLanguage(
        "Proximity model assigns a 70% probability to the level " +
          "being tested today. Context only; consult your own thesis.",
      ),
    ).not.toThrow();
  });

  it("covers every documented forbidden phrase", () => {
    for (const phrase of TIPSTER_VOCABULARY) {
      expect(() =>
        assertNoTipsterLanguage(`the model says ${phrase}HDFCBANK`),
      ).toThrow();
    }
  });
});

describe("renderEmail on the sample brief", () => {
  const brief = sampleBrief as unknown as BriefDocument;

  it("produces a non-trivial email body", () => {
    const text = renderEmail(brief);
    expect(text.length).toBeGreaterThan(400);
    expect(text).toContain("Daily Research Brief");
    expect(text).toContain("2026-03-14");
  });

  it("never contains tipster vocabulary on the sample fixture", () => {
    // renderEmail itself runs the guardrail; if it throws, the test
    // fails loudly. This is the production-style contract.
    const text = renderEmail(brief);
    const lowered = text.toLowerCase();
    for (const phrase of TIPSTER_VOCABULARY) {
      expect(lowered).not.toContain(phrase);
    }
  });

  it("includes the sector regime, watchlist, and yesterday audit sections", () => {
    const text = renderEmail(brief);
    expect(text).toContain("SECTOR REGIME");
    expect(text).toContain("WATCHLIST");
    expect(text).toContain("YESTERDAY AUDIT");
  });

  it("populates the yesterday audit per-bucket table", () => {
    const text = renderEmail(brief);
    expect(text).toMatch(/hit rate by confidence bucket:/);
    expect(text).toMatch(/proximity \/ high/);
  });
});

describe("retrospective audit disclosure", () => {
  it("prepends the disclosure line when the share crosses 50%", () => {
    const brief = JSON.parse(
      JSON.stringify(sampleBrief),
    ) as BriefDocument;
    // Force the retrospective state on.
    if ("hit_rate_by_confidence_bucket" in brief.yesterday_audit) {
      brief.yesterday_audit.is_retrospective_calibration = true;
      brief.yesterday_audit.retrospective_share = 0.82;
    }
    const text = renderEmail(brief);
    expect(text).toContain("retrospective replay");
    expect(text).toContain("82%");
  });

  it("omits the disclosure line when the audit is fully live", () => {
    const brief = JSON.parse(
      JSON.stringify(sampleBrief),
    ) as BriefDocument;
    if ("hit_rate_by_confidence_bucket" in brief.yesterday_audit) {
      brief.yesterday_audit.is_retrospective_calibration = false;
      brief.yesterday_audit.retrospective_share = 0.0;
    }
    const text = renderEmail(brief);
    expect(text).not.toContain("retrospective replay");
  });
});
