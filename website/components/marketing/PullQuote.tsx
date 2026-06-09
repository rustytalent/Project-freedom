import { Reveal } from "@/components/ui/Reveal";

/**
 * PullQuote: a full-bleed editorial moment between dense sections.
 *
 * Big serif italic line, attribution in mono caps, a warm vertical
 * rule on the left. Modelled on magazine spread pull-quotes - a
 * breath between data-heavy sections, a chance for the brand voice
 * to land alone.
 *
 * Pure server component, fades up on scroll via Reveal.
 */
export function PullQuote({
  quote,
  attribution,
}: {
  quote: string;
  attribution: string;
}) {
  return (
    <section className="relative overflow-hidden">
      <div className="absolute inset-0 hero-glow opacity-40 pointer-events-none" aria-hidden="true" />
      <div className="relative max-w-dash mx-auto px-6 py-24 md:py-28">
        <Reveal>
          <figure className="max-w-3xl mx-auto pl-8 md:pl-10 border-l-2 border-warm/50">
            <blockquote className="font-serif italic text-3xl md:text-[40px] leading-[1.15] tracking-[-0.01em] text-fg">
              <span aria-hidden="true" className="text-warm mr-1">&ldquo;</span>
              {quote}
              <span aria-hidden="true" className="text-warm ml-0.5">&rdquo;</span>
            </blockquote>
            <figcaption className="mt-7 font-mono text-[11px] uppercase tracking-[0.22em] text-fg-subtle">
              &mdash; {attribution}
            </figcaption>
          </figure>
        </Reveal>
      </div>
    </section>
  );
}
