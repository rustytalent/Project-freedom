import { Reveal } from "@/components/ui/Reveal";

/**
 * ProcessTimeline: "How a brief comes together" - a five-step
 * horizontal stepper that gives the visitor a glimpse of the
 * engine room without leaking technique. Stage labels are exactly
 * the publicly named pipeline stages from the engine spec; nothing
 * here exposes private agent names, model classes, or feature
 * engineering choices.
 *
 * On desktop: horizontal row with connecting hairlines.
 * On mobile: vertical column.
 *
 * Each step renders a small server-only SVG icon, a 2-digit number
 * in mono, a label, and a one-sentence description.
 */

type Step = {
  n: string;
  label: string;
  body: string;
  icon: "ingest" | "weave" | "calibrate" | "audit" | "publish";
};

const steps: Step[] = [
  {
    n: "01",
    label: "Ingest",
    body: "Overnight feeds and prior-session prints arrive.",
    icon: "ingest",
  },
  {
    n: "02",
    label: "Weave",
    body: "Sector regime, structural levels, and reaction context are stitched.",
    icon: "weave",
  },
  {
    n: "03",
    label: "Calibrate",
    body: "Every probability is checked against a hold-out window.",
    icon: "calibrate",
  },
  {
    n: "04",
    label: "Audit",
    body: "Yesterday's brief is graded against actual outcomes.",
    icon: "audit",
  },
  {
    n: "05",
    label: "Publish",
    body: "Brief lands in the inbox before 08:30 IST.",
    icon: "publish",
  },
];

function Icon({ kind }: { kind: Step["icon"] }) {
  // 28x28 viewBox, stroke-based. currentColor inherits warm bronze
  // from the wrapper so each icon picks up the warm accent.
  const common = {
    width: 28,
    height: 28,
    viewBox: "0 0 28 28",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.25,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };
  switch (kind) {
    case "ingest":
      // Downward arrows into a tray - feeds flowing in.
      return (
        <svg {...common}>
          <path d="M6 10 L6 16 M3 13 L6 16 L9 13" />
          <path d="M14 8 L14 16 M11 13 L14 16 L17 13" />
          <path d="M22 11 L22 16 M19 13 L22 16 L25 13" />
          <path d="M3 20 L25 20" />
        </svg>
      );
    case "weave":
      // Crossing diagonals over a bar - signals stitching together.
      return (
        <svg {...common}>
          <path d="M4 6 L24 22" />
          <path d="M24 6 L4 22" />
          <circle cx="14" cy="14" r="2.5" fill="currentColor" />
        </svg>
      );
    case "calibrate":
      // Reticle target - exact identity match to the brand mark.
      return (
        <svg {...common}>
          <circle cx="14" cy="14" r="6.5" />
          <circle cx="14" cy="14" r="2.5" fill="currentColor" />
          <path d="M14 2 L14 6 M14 22 L14 26 M2 14 L6 14 M22 14 L26 14" />
        </svg>
      );
    case "audit":
      // Tick over a list - graded outcomes.
      return (
        <svg {...common}>
          <path d="M4 8 L11 8" />
          <path d="M4 14 L11 14" />
          <path d="M4 20 L11 20" />
          <path d="M16 13 L19 17 L25 9" />
        </svg>
      );
    case "publish":
      // Outward arrows from a centred point - sending.
      return (
        <svg {...common}>
          <circle cx="14" cy="14" r="3" fill="currentColor" />
          <path d="M14 7 L14 4" />
          <path d="M14 24 L14 21" />
          <path d="M7 14 L4 14" />
          <path d="M24 14 L21 14" />
          <path d="M9 9 L7 7" />
          <path d="M19 19 L21 21" />
          <path d="M9 19 L7 21" />
          <path d="M19 9 L21 7" />
        </svg>
      );
  }
}

export function ProcessTimeline() {
  return (
    <section className="relative">
      <div className="max-w-dash mx-auto px-6 py-24 md:py-28">
        <Reveal>
          <div className="max-w-2xl mb-14">
            <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
              How a brief comes together
            </p>
            <h2 className="font-serif text-3xl md:text-4xl leading-tight text-fg">
              Five stages.{" "}
              <span className="text-warm">Every morning.</span>{" "}
              Before the bell.
            </h2>
            <p className="mt-5 text-fg-muted leading-relaxed">
              We don&rsquo;t hide the pipeline. We hide the technique.
              The stages below run in this order every NSE session
              &mdash; the same shape, the same gates, the same audit
              at the end. Subscribers see the output; the engine&rsquo;s
              internals stay inside the moat.
            </p>
          </div>
        </Reveal>

        <ol className="relative grid gap-px bg-border border border-border md:grid-cols-5">
          {steps.map((s, i) => (
            <Reveal key={s.n} delay={i * 110}>
              <li className="relative h-full bg-bg p-6 md:p-7 transition-colors hover:bg-bg-subtle/60 group">
                <div className="flex items-start justify-between mb-5">
                  <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-fg-subtle">
                    {s.n}
                  </span>
                  <span className="text-warm/80 group-hover:text-warm-glow transition-colors">
                    <Icon kind={s.icon} />
                  </span>
                </div>
                <p className="font-serif text-lg md:text-xl text-fg leading-snug">
                  {s.label}
                </p>
                <p className="mt-3 text-sm text-fg-muted leading-relaxed">
                  {s.body}
                </p>
              </li>
            </Reveal>
          ))}
        </ol>

        <p className="mt-8 max-w-2xl text-xs text-fg-subtle leading-relaxed">
          Stage labels are the engine&rsquo;s published pipeline names.
          Specific models, features, and tuning parameters stay inside
          the subscriber portal where they belong.
        </p>
      </div>
    </section>
  );
}
