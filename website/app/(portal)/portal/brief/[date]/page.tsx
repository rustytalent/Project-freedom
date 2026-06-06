import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SampleBriefRenderer } from "@/components/brief/SampleBriefRenderer";
import sampleBrief from "@/content/briefs/sample-2026-03-14.json";
import type { BriefDocument } from "@/lib/brief-render";

type Props = {
  params: Promise<{ date: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { date } = await params;
  return {
    title: `Brief - ${date}`,
  };
}

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export default async function ArchivedBriefPage({ params }: Props) {
  const { date } = await params;
  if (!DATE_RE.test(date)) notFound();
  // Production: fetch brief JSON for this trading date from Supabase.
  // For now: render the sample for every archive date.
  const brief = sampleBrief as unknown as BriefDocument;
  return (
    <div className="max-w-prose mx-auto px-6 py-14">
      <header className="mb-10">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-3">
          Archive &middot; {date}
        </p>
      </header>
      <div className="bg-bg-raised border border-border rounded-sm p-8 md:p-10">
        <SampleBriefRenderer brief={brief} />
      </div>
    </div>
  );
}
