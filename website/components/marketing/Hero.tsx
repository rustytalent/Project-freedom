import { Suspense } from "react";
import Link from "next/link";
import { brand } from "@/lib/brand";
import { LinkButton } from "@/components/ui/button";
import { LatestPublishedTicker } from "./LatestPublishedTicker";
import { LivePulse } from "./LivePulse";
import { HeroDashboardPreview } from "./HeroDashboardPreview";

export function Hero() {
  return (
    <section className="relative grid-backdrop overflow-hidden">
      <div className="absolute inset-0 hero-glow pointer-events-none" aria-hidden="true" />
      <div className="relative max-w-dash mx-auto px-6 pt-16 md:pt-20 pb-24">
        {/* Editorial top strip - establishes "this is a published
            research product" before the headline lands. */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-10 pb-4 border-b border-border/60 fade-in-up">
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-fg-muted">
            Research desk
          </p>
          <span className="hidden sm:inline-block w-px h-3 bg-border" />
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-fg-subtle">
            Edition 06 · 06 JUN 2026 · IST
          </p>
          <span className="hidden sm:inline-block w-px h-3 bg-border" />
          <Suspense fallback={<LivePulse label="Loading status…" />}>
            <LatestPublishedTicker />
          </Suspense>
        </div>

        <div className="grid gap-14 lg:grid-cols-[1.15fr_1fr] lg:gap-16 items-start fade-in-up">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-accent mb-6">
              Research, not signals.
            </p>
            <h1 className="font-serif text-5xl md:text-6xl lg:text-[64px] leading-[1.03] tracking-tight text-fg">
              {brand.tagline}
            </h1>
            <p className="text-lg text-fg-muted mt-7 max-w-xl leading-relaxed">
              {brand.subTagline}
            </p>
            <div className="mt-10 flex flex-wrap items-center gap-3">
              <LinkButton href="/sample-brief" variant="primary">
                View sample brief
              </LinkButton>
              <LinkButton href="/track-record" variant="secondary">
                See track record
              </LinkButton>
            </div>
            <p className="mt-5 text-sm text-fg-subtle">
              Or read{" "}
              <Link
                href="/philosophy"
                className="text-fg-muted hover:text-fg transition-colors underline underline-offset-4 decoration-fg-subtle"
              >
                the philosophy
              </Link>{" "}
              first — six minutes, no signup.
            </p>
          </div>
          <div className="lg:pt-2">
            <HeroDashboardPreview />
            <p className="mt-3 font-mono text-[10px] uppercase tracking-[0.18em] text-fg-subtle text-right">
              Live preview · redacted public view
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
