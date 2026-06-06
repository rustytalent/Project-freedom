import type { Metadata } from "next";
import { LinkButton } from "@/components/ui/button";
import { brand } from "@/lib/brand";

export const metadata: Metadata = {
  title: "About",
  description: `Why ${brand.name} exists.`,
};

export default function AboutPage() {
  return (
    <article className="max-w-prose mx-auto px-6 py-20">
      <header className="mb-12">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
          About
        </p>
        <h1 className="font-serif text-4xl md:text-5xl leading-tight text-fg">
          Why {brand.name} exists.
        </h1>
      </header>

      <section className="text-fg-muted leading-relaxed space-y-6">
        <p>
          The Indian markets have a vacuum where calibrated research
          should be. The well-funded retail flow is served by signal
          services that don&rsquo;t audit themselves and tipsters who
          can&rsquo;t describe their own discipline. The serious end
          of the same flow is served by sell-side desks built for
          institutional clients, not for the trader running their own
          book on their own conviction.
        </p>
        <p>
          {brand.name} is built for that gap. A research operation,
          built with the same engineering discipline a small quant
          fund would build for its own internal use, packaged into
          products the working trader can read in six minutes.
        </p>
        <p>
          We don&rsquo;t publish signals. We don&rsquo;t hold
          ourselves out as SEBI-registered advisors. We don&rsquo;t
          take performance fees. We publish calibrated context - and
          we audit ourselves in public.
        </p>
        <p>
          If the philosophy resonates, the{" "}
          <a
            href="/sample-brief"
            className="text-accent underline underline-offset-4"
          >
            sample brief
          </a>{" "}
          will give you the shape and the prose. The{" "}
          <a
            href="/track-record"
            className="text-accent underline underline-offset-4"
          >
            track record
          </a>{" "}
          will give you the calibration evidence. The rest is your
          call.
        </p>
      </section>

      <section className="mt-16 pt-12 border-t border-border">
        <h2 className="font-serif text-2xl text-fg mb-4">Contact</h2>
        <p className="text-fg-muted leading-relaxed">
          Founder + brand only for now. Reach us at{" "}
          <a
            href={`mailto:${brand.contact.email}`}
            className="text-accent underline underline-offset-4"
          >
            {brand.contact.email}
          </a>{" "}
          for product questions, partnerships, or to book a 30-minute
          intake call for Audit Base.
        </p>
      </section>

      <div className="mt-16 pt-12 border-t border-border flex flex-wrap gap-3">
        <LinkButton href="/pricing" variant="primary">
          See pricing
        </LinkButton>
        <LinkButton href="/contact" variant="secondary">
          Talk to us
        </LinkButton>
      </div>
    </article>
  );
}
