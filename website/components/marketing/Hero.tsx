import { Suspense } from "react";
import Link from "next/link";
import { brand } from "@/lib/brand";
import { LinkButton } from "@/components/ui/button";
import { LatestPublishedTicker } from "./LatestPublishedTicker";
import { LivePulse } from "./LivePulse";
import { HeroDashboardPreview } from "./HeroDashboardPreview";

/**
 * The headline.
 *
 * Visually anchored on the word "Calibrated" - the differentiator
 * we want the eye to lock onto in the first second. The rest of the
 * sentence reads in the off-white text colour, so the focal word
 * stands out as a single warm note in a cool composition.
 */
function Headline() {
  const t = brand.tagline;
  const focal = "Calibrated";
  const rest = t.startsWith(focal) ? t.slice(focal.length) : t;
  return (
    <h1 className="font-serif text-5xl md:text-6xl lg:text-[68px] leading-[1.02] tracking-[-0.02em] text-fg">
      <span className="text-warm">{focal}</span>
      {rest}
    </h1>
  );
}

export function Hero() {
  return (
    <section className="relative grid-backdrop overflow-hidden">
      <div className="absolute inset-0 hero-glow pointer-events-none" aria-hidden="true" />
      <div className="relative max-w-dash mx-auto px-6 pt-16 md:pt-24 pb-28 md:pb-32">
        {/* Editorial top strip. Mono, dense, dated. Frames the page
            as a published artifact, not a marketing landing. */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-12 pb-5 border-b border-border/60 fade-in-up">
          <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-fg-muted">
            Research desk
          </span>
          <span className="hidden sm:inline-block w-px h-3 bg-border" />
          <span
            className="font-mono text-[11px] uppercase tracking-[0.18em] text-warm-glow px-2 py-0.5 border border-warm/40 rounded-sm"
            aria-label="Edition number"
          >
            Edition #006
          </span>
          <span className="hidden sm:inline-block w-px h-3 bg-border" />
          <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-fg-subtle">
            06 JUN 2026 · IST
          </span>
          <span className="hidden sm:inline-block w-px h-3 bg-border" />
          <Suspense fallback={<LivePulse label="Loading status…" />}>
            <LatestPublishedTicker />
          </Suspense>
        </div>

        <div className="grid gap-14 lg:grid-cols-[1.12fr_1fr] lg:gap-20 items-start fade-in-up">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-accent mb-7">
              Research, not signals.
            </p>
            <Headline />
            <p className="text-lg text-fg-muted mt-8 max-w-xl leading-relaxed">
              {brand.subTagline}
            </p>
            <div className="mt-12 flex flex-wrap items-center gap-3">
              <LinkButton href="/sample-brief" variant="primary">
                View sample brief
              </LinkButton>
              <LinkButton href="/track-record" variant="secondary">
                See track record
              </LinkButton>
            </div>
            {/* Friction-reducing microcopy. Warm bronze in mono so the
                eye registers it as a precise commitment, not marketing
                fluff. Sits directly below the CTAs - first thing the
                visitor reads after deciding to act. */}
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
          <div className="lg:pt-2 hero-artifact-wrap">
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
