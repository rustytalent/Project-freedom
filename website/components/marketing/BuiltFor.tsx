import { Reveal } from "@/components/ui/Reveal";

const personas = [
  {
    tag: "The independent trader",
    title: "Running your own book, on your own thesis.",
    body:
      "You're past the tipster phase. You want pre-market context that " +
      "respects your judgement, not a buy call you have to second-guess. " +
      "The brief tells you where the structural levels are, where the " +
      "regime is leaning, and where to stand aside — then it gets out of " +
      "the way.",
  },
  {
    tag: "The desk analyst",
    title: "On the morning huddle, defending a read.",
    body:
      "You want a sober second opinion before you walk into the room. " +
      "Calibrated probabilities, redacted sector regime, and an honest " +
      "audit of yesterday's calls give you something to anchor or argue " +
      "against — not a black-box signal you can't explain.",
  },
  {
    tag: "The fund PM",
    title: "Allocating capital across desks.",
    body:
      "You don't need another data feed. You need a research partner " +
      "whose calibration error you can verify in public, whose drift is " +
      "flagged before the loss is taken, and whose discipline you can " +
      "show to your IC.",
  },
] as const;

export function BuiltFor() {
  return (
    <section className="border-t border-border">
      <div className="max-w-dash mx-auto px-6 py-20">
        <Reveal>
          <div className="max-w-2xl mb-14">
            <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
              Built for
            </p>
            <h2 className="font-serif text-3xl md:text-4xl leading-tight text-fg">
              Three readers. One discipline.
            </h2>
            <p className="mt-5 text-fg-muted leading-relaxed">
              We don&rsquo;t segment by tier or asset class. We segment by
              how you use the brief. If you recognise yourself in one of
              these, the product is built for you.
            </p>
          </div>
        </Reveal>
        <div className="grid gap-px bg-border md:grid-cols-3 border border-border">
          {personas.map((p, i) => (
            <Reveal key={p.tag} delay={i * 110}>
              <article className="h-full bg-bg p-8 md:p-10 transition-colors hover:bg-bg-subtle/60">
                <p className="font-mono text-xs text-accent tracking-wider mb-6">
                  0{i + 1} &middot; {p.tag}
                </p>
                <h3 className="font-serif text-xl md:text-2xl text-fg leading-snug">
                  {p.title}
                </h3>
                <p className="mt-5 text-sm text-fg-muted leading-relaxed">
                  {p.body}
                </p>
              </article>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
