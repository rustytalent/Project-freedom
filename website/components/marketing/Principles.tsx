import Link from "next/link";
import { Reveal } from "@/components/ui/Reveal";

const principles = [
  {
    n: "01",
    title: "Research, not signals.",
    body:
      "We sell context — sector regime, structural levels, reaction " +
      "quality, avoidance flags. We don't sell instructions. The decision " +
      "to act remains with the reader.",
  },
  {
    n: "02",
    title: "Probabilities, not certainties.",
    body:
      "Every claim carries a number. A 62% touch probability is the " +
      "system's calibrated estimate of how often historical analogues " +
      "did, not a prediction it will.",
  },
  {
    n: "03",
    title: "Calibration, not promises.",
    body:
      "70% should resolve 70 times out of 100. When it doesn't, the gap " +
      "shows up as calibration error and the head is flagged drifting " +
      "until it returns to tolerance.",
  },
  {
    n: "04",
    title: "Audit, in public.",
    body:
      "Every brief opens by auditing yesterday's. Hit rate per " +
      "confidence bucket, calibration error per prediction type, drift " +
      "flags before they degrade the next read.",
  },
] as const;

export function Principles() {
  return (
    <section className="border-t border-border bg-bg-raised/40">
      <div className="max-w-dash mx-auto px-6 py-24 md:py-28">
        <Reveal>
          <div className="mb-14 flex flex-wrap items-end justify-between gap-6">
            <div className="max-w-2xl">
              <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
                How we publish
              </p>
              <h2 className="font-serif text-3xl md:text-4xl leading-tight text-fg">
                Four commitments, audited every session.
              </h2>
            </div>
            <Link
              href="/philosophy"
              className="text-sm text-fg-muted hover:text-fg transition-colors underline underline-offset-4 decoration-fg-subtle"
            >
              Read the full philosophy &rarr;
            </Link>
          </div>
        </Reveal>
        <div className="grid gap-px bg-border md:grid-cols-2 border border-border">
          {principles.map((p, i) => (
            <Reveal key={p.n} delay={i * 90}>
              <div className="h-full bg-bg p-8 md:p-10 transition-colors hover:bg-bg-subtle/60">
                <div className="flex items-baseline gap-4 mb-4">
                  <span className="font-mono text-xs text-fg-subtle">
                    {p.n}
                  </span>
                  <span className="h-px flex-1 bg-border" />
                </div>
                <h3 className="font-serif text-xl md:text-2xl text-fg leading-snug">
                  {p.title}
                </h3>
                <p className="mt-4 text-sm text-fg-muted leading-relaxed max-w-md">
                  {p.body}
                </p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
