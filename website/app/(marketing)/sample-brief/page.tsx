import type { Metadata } from "next";
import Link from "next/link";
import { SampleBriefRenderer } from "@/components/brief/SampleBriefRenderer";
import { LinkButton } from "@/components/ui/button";
import { PageHeader } from "@/components/marketing/PageHeader";
import { Reveal } from "@/components/ui/Reveal";
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
    <>
      <PageHeader
        section="Sample brief"
        editionTag="14 MAR 2026"
        dateline="PUBLIC VIEW · REDACTED"
        status={
          <span className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-fg-muted">
            <span aria-hidden="true">·</span>
            <span>Six-minute read</span>
          </span>
        }
        title={
          <>
            One real brief,{" "}
            <span className="text-warm">redacted</span> for public view.
          </>
        }
        lead={
          <p>
            The shape, prose, and section order below match what a
            paying subscriber receives. Specific symbol names and price
            levels are replaced with category labels and{" "}
            <span className="font-mono text-warm-glow">
              ₹[subscriber-only]
            </span>{" "}
            tokens so the actionable content is preserved for paying
            readers, while you see exactly what every brief looks like.
          </p>
        }
        meta="The Yesterday Audit numbers are aggregate per-bucket statistics — those need no redaction"
      />

      <div className="max-w-prose mx-auto px-6 pb-24">
        <Reveal>
          {/* The brief is rendered inside a warm-glow card matching
              the hero artifact treatment - signals "this is the
              product, not a marketing approximation of the product". */}
          <div className="hero-artifact bg-bg-raised border border-border rounded-sm p-8 md:p-10 relative overflow-hidden">
            <div className="absolute inset-0 panel-grid opacity-30 pointer-events-none" />
            <div className="relative">
              <SampleBriefRenderer brief={brief} />
            </div>
          </div>
        </Reveal>

        <Reveal>
          {/* CTA footer - mirrors the home final-CTA polish so the
              read ends with the same warmth, not a flat link bar. */}
          <div className="mt-16 pt-12 border-t border-border">
            <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-warm-glow mb-5">
              Next morning · same time · in your inbox
            </p>
            <h2 className="font-serif text-2xl md:text-3xl leading-snug text-fg max-w-md">
              Subscribe and get the{" "}
              <span className="text-warm">unredacted</span> version of
              this brief, every NSE morning.
            </h2>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <LinkButton href="/pricing" variant="primary">
                See pricing
              </LinkButton>
              <LinkButton
                href="/checkout?plan=daily&cycle=monthly"
                variant="secondary"
              >
                Subscribe
              </LinkButton>
              <Link
                href="/philosophy"
                className="ml-1 text-sm text-fg-muted hover:text-fg transition-colors self-center underline underline-offset-4 decoration-fg-subtle"
              >
                Read the philosophy first
              </Link>
            </div>
            <ul className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-2 font-mono text-[11px] uppercase tracking-[0.16em]">
              <li className="flex items-center gap-2 text-warm-glow">
                <span aria-hidden="true" className="text-warm">·</span>
                <span>Five-day preview</span>
              </li>
              <li className="flex items-center gap-2 text-warm-glow">
                <span aria-hidden="true" className="text-warm">·</span>
                <span>Google sign-in</span>
              </li>
              <li className="flex items-center gap-2 text-warm-glow">
                <span aria-hidden="true" className="text-warm">·</span>
                <span>No card required</span>
              </li>
            </ul>
          </div>
        </Reveal>
      </div>
    </>
  );
}
