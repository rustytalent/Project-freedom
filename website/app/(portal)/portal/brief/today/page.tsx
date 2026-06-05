import type { Metadata } from "next";
import { SampleBriefRenderer } from "@/components/brief/SampleBriefRenderer";
import sampleBrief from "@/content/briefs/sample-2026-03-14.json";
import type { BriefDocument } from "@/lib/brief-render";

export const metadata: Metadata = {
  title: "Today's brief",
};

// In production the brief is fetched from Supabase by the
// today-in-IST trading date. For now we render the same sample
// brief used on the public marketing surface — the only
// difference being that in the portal the levels would NOT be
// redacted. The redaction tokens live in the JSON itself; in
// production the portal would read the un-redacted source while
// the marketing surface reads the redacted one.

export default function TodaysBriefPage() {
  const brief = sampleBrief as unknown as BriefDocument;
  return (
    <div className="max-w-prose mx-auto px-6 py-14">
      <header className="mb-10">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-3">
          Daily Brief &middot; published 08:30 IST
        </p>
        <p className="text-xs text-fg-subtle">
          The full subscriber view. Levels, symbols, and strike numbers
          are all visible to you in the live deployment (this preview
          page uses the public sample data).
        </p>
      </header>
      <div className="bg-bg-raised border border-border rounded-sm p-8 md:p-10">
        <SampleBriefRenderer brief={brief} />
      </div>
    </div>
  );
}
