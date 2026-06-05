import type { Metadata } from "next";
import Link from "next/link";
import { SampleBriefRenderer } from "@/components/brief/SampleBriefRenderer";
import { LinkButton } from "@/components/ui/button";
import sampleBrief from "@/content/briefs/sample-2026-03-14.json";
import type { BriefDocument } from "@/lib/brief-render";

export const metadata: Metadata = {
  title: "Sample brief",
  description:
    "A real historical Daily Brief, redacted for public view. Real shape, real prose, levels and symbols masked.",
};

export default function SampleBriefPage() {
  // Cast: the JSON file is type-checked at build time against the
  // tsconfig resolveJsonModule path. At runtime we trust the shape.
  const brief = sampleBrief as unknown as BriefDocument;
  return (
    <article className="max-w-prose mx-auto px-6 py-20">
      <header className="mb-12">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
          Sample · published historically · public view
        </p>
        <h1 className="font-serif text-3xl md:text-4xl leading-tight text-fg">
          One real brief, redacted for public view.
        </h1>
        <p className="mt-4 text-fg-muted leading-relaxed">
          The shape, prose, and section order below match what a paying
          subscriber receives. Specific symbol names and price levels
          are replaced with category labels and{" "}
          <span className="font-mono">₹[subscriber-only]</span> tokens
          so the actionable content is preserved for paying readers,
          while you see what every brief looks like.
        </p>
        <p className="mt-2 text-xs text-fg-subtle">
          The Yesterday Audit numbers are aggregate per-bucket
          statistics — those need no redaction.
        </p>
      </header>

      <div className="bg-bg-raised border border-border rounded-sm p-8 md:p-10">
        <SampleBriefRenderer brief={brief} />
      </div>

      <div className="mt-12 pt-8 border-t border-border flex flex-wrap gap-3">
        <LinkButton href="/pricing" variant="primary">
          See pricing
        </LinkButton>
        <LinkButton href="/portal" variant="secondary">
          Subscribe
        </LinkButton>
        <Link
          href="/philosophy"
          className="ml-2 text-sm text-fg-muted hover:text-fg transition-colors self-center"
        >
          Read the philosophy →
        </Link>
      </div>
    </article>
  );
}
