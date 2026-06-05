import { Suspense } from "react";
import Link from "next/link";
import { brand } from "@/lib/brand";
import { LinkButton } from "@/components/ui/button";
import { LatestPublishedTicker } from "./LatestPublishedTicker";
import { LivePulse } from "./LivePulse";

export function Hero() {
  return (
    <section className="relative grid-backdrop overflow-hidden">
      <div className="absolute inset-0 hero-glow pointer-events-none" aria-hidden="true" />
      <div className="relative max-w-dash mx-auto px-6 pt-24 pb-24 fade-in-up">
        <div className="flex items-center gap-3 mb-8">
          <p className="text-xs uppercase tracking-[0.18em] text-accent">
            Research, not signals.
          </p>
          <span className="hidden sm:inline-block w-px h-3 bg-border" />
          <Suspense fallback={<LivePulse label="Loading status…" />}>
            <LatestPublishedTicker />
          </Suspense>
        </div>
        <h1 className="font-serif text-5xl md:text-6xl leading-[1.05] tracking-tight text-fg max-w-3xl">
          {brand.tagline}
        </h1>
        <p className="text-lg text-fg-muted mt-6 max-w-xl leading-relaxed">
          {brand.subTagline}
        </p>
        <div className="mt-10 flex flex-wrap items-center gap-3">
          <LinkButton href="/sample-brief" variant="primary">
            View sample brief
          </LinkButton>
          <LinkButton href="/track-record" variant="secondary">
            See track record
          </LinkButton>
          <Link
            href="/philosophy"
            className="ml-2 text-sm text-fg-muted hover:text-fg transition-colors underline-offset-4"
          >
            Read our philosophy →
          </Link>
        </div>
      </div>
    </section>
  );
}
