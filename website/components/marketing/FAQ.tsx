import { Reveal } from "@/components/ui/Reveal";

const faqs = [
  {
    q: "Is this a tip service or a signal service?",
    a: "Neither. The Daily Brief is a research artifact — sector regime, " +
       "structural levels in play, reaction quality, avoidance flags. " +
       "Every claim carries a calibrated probability. The decision to " +
       "act remains with the reader.",
  },
  {
    q: "Are you SEBI-registered investment advisors?",
    a: "No. We are a research publisher, not a registered investment " +
       "advisor. The brief is research context only, not investment " +
       "advice. The disclosures page is explicit about this and every " +
       "page footer carries the short form.",
  },
  {
    q: "What does \"calibrated\" actually mean here?",
    a: "When the brief publishes a 70% probability, that estimate is " +
       "checked against the actual outcome rate on a holdout window the " +
       "model never saw during fitting. The dashboard publishes the gap " +
       "(calibration error) per confidence bucket. If a head drifts " +
       "out of tolerance, we flag it until it recovers.",
  },
  {
    q: "What's the delivery time?",
    a: "Briefs are published by 08:30 IST on NSE trading days. Email " +
       "plus the subscriber portal. The portal carries the full archive.",
  },
  {
    q: "Can I see a real brief before paying?",
    a: "Yes — the sample brief is a real published artifact, redacted " +
       "for public view. Subscriber-only levels are masked but the " +
       "shape, sections, and language are intact. Google sign-in also " +
       "starts an automatic five-day preview.",
  },
  {
    q: "What if my preview expires and I don't subscribe?",
    a: "Nothing. Your account stays. The preview is a single five-day " +
       "window — when it expires, the portal goes back to the public " +
       "marketing surface. No card charged, no follow-up sequence.",
  },
  {
    q: "Do you ever recommend specific entry/exit levels?",
    a: "Subscriber briefs publish exact structural levels and reaction " +
       "context. They are not entry/exit instructions — they are the " +
       "context against which the reader's own decision is made. The " +
       "brief never tells the reader what size, when to enter, or " +
       "when to exit.",
  },
  {
    q: "Can I cancel anytime?",
    a: "Monthly subscriptions cancel any time from the portal. Annual " +
       "subscriptions are refundable pro rata within the first fourteen " +
       "days after billing.",
  },
] as const;

export function FAQ() {
  return (
    <section>
      <div className="max-w-dash mx-auto px-6 py-24 md:py-28 grid gap-12 md:grid-cols-[1fr_2fr]">
        <Reveal>
          <div className="md:sticky md:top-28">
            <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
              Common questions
            </p>
            <h2 className="font-serif text-3xl md:text-4xl leading-tight text-fg">
              Direct answers to the questions we get most.
            </h2>
            <p className="mt-5 text-sm text-fg-muted leading-relaxed max-w-sm">
              If your question isn&rsquo;t here, the contact page reaches
              the founder directly.
            </p>
          </div>
        </Reveal>
        <Reveal delay={120}>
          <dl className="divide-y divide-border border-t border-border">
            {faqs.map((f) => (
              <details
                key={f.q}
                className="group py-5 [&_summary]:list-none"
              >
                <summary className="flex cursor-pointer items-start justify-between gap-6">
                  <span className="font-serif text-lg text-fg leading-snug">
                    {f.q}
                  </span>
                  <span
                    aria-hidden="true"
                    className="mt-1 inline-block font-mono text-xs text-fg-subtle transition-transform group-open:rotate-45"
                  >
                    +
                  </span>
                </summary>
                <p className="mt-4 text-sm text-fg-muted leading-relaxed max-w-2xl">
                  {f.a}
                </p>
              </details>
            ))}
          </dl>
        </Reveal>
      </div>
    </section>
  );
}
