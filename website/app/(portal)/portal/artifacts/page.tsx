import type { Metadata } from "next";
import { ArtifactList } from "@/components/portal/ArtifactList";
import { getStorage } from "@/lib/artifact-storage";
import {
  type ArtifactKind,
  type ArtifactRecord,
} from "@/lib/artifacts";

export const metadata: Metadata = {
  title: "Artifacts",
  description:
    "Latest research artifacts published by the engine. Daily briefs, weekly notes, calibration exports.",
};

// Keep this dynamic so newly-uploaded artifacts appear immediately.
export const dynamic = "force-dynamic";

const SECTION_ORDER: Array<{
  heading: string;
  kinds: ArtifactKind[];
  empty: string;
}> = [
  {
    heading: "Daily Brief",
    kinds: ["daily_brief_pdf", "daily_brief_email"],
    empty:
      "No Daily Brief artifacts yet. Today's brief will appear here by 08:30 IST.",
  },
  {
    heading: "Live Desk notes",
    kinds: ["swing_brief_pdf"],
    empty:
      "No Live Desk note artifacts yet. Monday pre-open is the next scheduled publication.",
  },
  {
    heading: "Calibration & audit exports",
    kinds: [
      "calibration_summary_csv",
      "yesterday_audit_csv",
      "outcome_log_export_csv",
    ],
    empty: "Calibration exports refresh nightly.",
  },
  {
    heading: "Options",
    kinds: [
      "options_strikes_csv",
      "options_executor_calls_csv",
      "options_executor_audit_csv",
    ],
    empty: "No options exports for this date.",
  },
  {
    heading: "Research notes",
    kinds: ["weekly_research_note_pdf"],
    empty: "Weekly research notes appear here on Monday mornings.",
  },
  {
    heading: "Audit reports",
    kinds: ["diagnosis_report_pdf"],
    empty:
      "Audit reports appear here only for the customer account they belong to, once published.",
  },
];

export default async function PortalArtifactsPage() {
  const store = await getStorage();
  const all = await store.list({ limit: 200 });
  const byKind: Map<ArtifactKind, ArtifactRecord[]> = new Map();
  for (const r of all) {
    if (!byKind.has(r.kind)) byKind.set(r.kind, []);
    byKind.get(r.kind)!.push(r);
  }

  const latest = all[0];

  return (
    <div className="max-w-dash mx-auto px-6 py-14">
      <header className="mb-10 max-w-prose">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-3">
          Artifacts
        </p>
        <h1 className="font-serif text-3xl md:text-4xl text-fg">
          Every artifact the engine publishes, in one place.
        </h1>
        <p className="mt-4 text-fg-muted leading-relaxed">
          Daily Brief PDFs, options exports, index notes, calibration
          exports, and audit reports - uploaded automatically the
          moment the engine finishes generating them. Tier-gated; you
          see what your subscription includes.
        </p>
        {latest && (
          <p className="mt-4 text-xs text-fg-subtle font-mono">
            Latest publication:{" "}
            <span className="text-fg">
              {new Date(latest.generated_at_utc).toLocaleString("en-IN", {
                timeZone: "Asia/Kolkata",
                hour12: false,
              })}{" "}
              IST
            </span>
          </p>
        )}
      </header>

      <div className="space-y-14">
        {SECTION_ORDER.map((section) => {
          const records: ArtifactRecord[] = [];
          for (const k of section.kinds) {
            const rs = byKind.get(k);
            if (rs) records.push(...rs);
          }
          records.sort((a, b) =>
            b.generated_at_utc.localeCompare(a.generated_at_utc),
          );
          return (
            <section key={section.heading}>
              <div className="flex items-baseline justify-between mb-4">
                <h2 className="font-serif text-2xl text-fg">
                  {section.heading}
                </h2>
                {records.length > 0 && (
                  <p className="text-xs text-fg-subtle tabnum">
                    {records.length} file{records.length === 1 ? "" : "s"}
                  </p>
                )}
              </div>
              <ArtifactList records={records} emptyMessage={section.empty} />
            </section>
          );
        })}
      </div>
    </div>
  );
}
