import type { Metadata } from "next";
import { brand } from "@/lib/brand";

export const metadata: Metadata = {
  title: "Philosophy",
  description:
    `What ${brand.name} believes and what we refuse to do. ` +
    "Research, not signals. Calibration, not promises.",
};

export default function PhilosophyPage() {
  return (
    <article className="max-w-prose mx-auto px-6 py-20">
      <header className="mb-12">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
          The operating principles
        </p>
        <h1 className="font-serif text-4xl md:text-5xl leading-tight text-fg">
          What we believe, and what we refuse to do.
        </h1>
        <p className="mt-6 text-fg-muted leading-relaxed">
          This is the longest page on the site, and the most important
          one. If you only read one thing before subscribing, read this.
        </p>
      </header>

      <Section
        eyebrow="01"
        title="Research, not signals."
        body={
          <>
            <p>
              The Indian markets are flooded with services that tell you
              what to do. They send instructions, show perfect-looking
              charts, and charge for urgency. That is not the product
              we are building.
            </p>
            <p>
              {brand.name} publishes research. A brief is a structured
              read of where the market sits today, which structural
              levels are in play, which sectors are leading, where the
              research stack has actionable conviction, and where it
              doesn&rsquo;t. The decision to act on any of it remains
              entirely with the reader.
            </p>
            <p>
              We believe the right unit of subscription is{" "}
              <em>context</em>, not <em>commands</em>. Anyone trading
              their own money deserves to make their own decisions, on
              their own thesis, with the best available context. That
              is what we sell.
            </p>
          </>
        }
      />

      <Section
        eyebrow="02"
        title="Probabilities, not certainties."
        body={
          <>
            <p>
              Every claim in every brief carries a number. A 62% chance
              that a particular structural level is tested today is not
              a prediction that it <em>will</em> be tested. It is the
              system&rsquo;s calibrated estimate of how often the
              setup&rsquo;s historical analogues did.
            </p>
            <p>
              Sometimes the level holds. Sometimes it doesn&rsquo;t. The
              brief tells you which sometimes is more likely, and by
              how much. The rest is your edge to find.
            </p>
          </>
        }
      />

      <Section
        eyebrow="03"
        title="Calibration, not promises."
        body={
          <>
            <p>
              A 70% probability we publish should resolve in our favour
              roughly 70 times out of 100. If it doesn&rsquo;t, the gap
              shows up on our public track-record dashboard as
              calibration error.
            </p>
            <p>
              When a particular probability bucket drifts beyond
              a tolerance, we flag it as <em>drifting</em> in the brief
              itself. You see, in real time, which parts of the read
              deserve more caution.
            </p>
            <p>
              We do not advertise cherry-picked weeks. We show the
              per-bucket hit rate of every prediction we have published
              through the product.
            </p>
          </>
        }
      />

      <Section
        eyebrow="04"
        title="Audit, not aspiration."
        body={
          <>
            <p>
              Every brief ends with an audit of the previous
              brief&rsquo;s calls. Per confidence bucket, per prediction
              type. The numbers are aggregated nightly and are visible
              to anyone who visits our track record page - subscriber
              or not.
            </p>
            <p>
              When the audit window contains predictions that were{" "}
              <em>replayed</em> from historical bundles rather than
              collected live, we say so explicitly. Retrospective
              replay is useful for filling in calibration before our
              live history has accumulated, but it is not the same
              evidence - and the disclosure line keeps the difference
              honest.
            </p>
          </>
        }
      />

      <Section
        eyebrow="05"
        title="Time order, never hindsight."
        body={
          <>
            <p>
              Every number in every brief was knowable in real time at
              the moment we claim to know it. If a data point would not
              have existed before publication, it does not belong in the
              brief.
            </p>
            <p>
              This is why the product separates live evidence from
              retrospective replay and labels both clearly. The reader
              should never have to guess what was known when.
            </p>
          </>
        }
      />

      <Section
        eyebrow="06"
        title="What we refuse to do."
        body={
          <>
            <p>
              We refuse instruction-style language in customer research.
              The brief can tell you where the market looks stretched,
              where attention is warranted, and where caution is higher.
              It cannot make the decision for you.
            </p>
            <p>We also refuse, structurally:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li>
                Performance-fee structures. We charge a flat
                subscription. Our incentives stay aligned with research
                quality, never with how aggressively you trade.
              </li>
              <li>
                Urgency-based marketing. No countdown timers, no
                &ldquo;limited slots&rdquo;, no &ldquo;closing
                tonight&rdquo;.
              </li>
              <li>
                Marketing the model&rsquo;s individual best week. We
                publish the full distribution; you can find the best
                and worst weeks yourself.
              </li>
              <li>
                Holding ourselves out as SEBI-registered investment
                advisors. We are not. Our content is research context,
                not investment advice, and the disclosure runs on every
                page.
              </li>
            </ul>
          </>
        }
      />

      <Section
        eyebrow="07"
        title="What that means for the reader."
        body={
          <>
            <p>
              {brand.name} is built for traders who already know what
              they are doing - and want better context to do it with.
              If you want signals, this is not the right service. If
              you want a calibrated reading of structural risk every
              morning, before NSE open, audited the next morning, this
              is exactly the right service.
            </p>
            <p>
              Read the sample brief. Read the track record. Make your
              own call.
            </p>
          </>
        }
      />
    </article>
  );
}

function Section({
  eyebrow,
  title,
  body,
}: {
  eyebrow: string;
  title: string;
  body: React.ReactNode;
}) {
  return (
    <section className="mt-16 pt-12 border-t border-border first-of-type:mt-12 first-of-type:pt-0 first-of-type:border-0">
      <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
        {eyebrow}
      </p>
      <h2 className="font-serif text-2xl md:text-3xl leading-tight text-fg mb-6">
        {title}
      </h2>
      <div className="text-fg-muted leading-relaxed space-y-4">{body}</div>
    </section>
  );
}
