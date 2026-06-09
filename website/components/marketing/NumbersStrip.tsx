import { mockLatestSummary, mockTimeSeries } from "@/lib/outcome-log-mock";
import { AnimatedCounter, type CounterFormat } from "@/components/ui/AnimatedCounter";
import { Reveal } from "@/components/ui/Reveal";
import { StatMark } from "./StatMark";

type Stat = {
  label: string;
  raw: number;
  format: CounterFormat;
  sub: string;
  mark: "staircase" | "tally" | "oscillation" | "disc";
};

function buildStats(): Stat[] {
  const summary = mockLatestSummary();
  const series = mockTimeSeries();
  const last30 = series.slice(-30);
  const meanAbsErr =
    last30.reduce((a, d) => a + Math.abs(d.touch_watch_calibration_error), 0) /
    last30.length;
  return [
    {
      label: "Briefs published",
      raw: series.length,
      format: { kind: "int" },
      sub: "since launch, across equity, options, index",
      mark: "staircase",
    },
    {
      label: "Predictions audited",
      raw: summary.reduce((a, r) => a + r.n, 0),
      format: { kind: "int" },
      sub: "every claim resolved, hit or miss, on the next session",
      mark: "tally",
    },
    {
      label: "Mean calibration error",
      raw: meanAbsErr * 100,
      format: { kind: "decimal", decimals: 1, suffix: "%" },
      sub: "rolling 30 trading days, touch-watch head",
      mark: "oscillation",
    },
    {
      label: "Audit transparency",
      raw: 100,
      format: { kind: "decimal", decimals: 0, suffix: "%" },
      sub: "no claim is published without an outcome the next day",
      mark: "disc",
    },
  ];
}

export function NumbersStrip() {
  const stats = buildStats();
  return (
    <section className="bg-bg-raised/40">
      <div className="max-w-dash mx-auto px-6 py-24 md:py-28">
        <Reveal>
          <div className="flex flex-wrap items-end justify-between gap-6 mb-14">
            <div className="max-w-xl">
              <p className="text-xs uppercase tracking-[0.18em] text-accent mb-3">
                The record so far
              </p>
              <h2 className="font-serif text-3xl md:text-4xl leading-tight text-fg">
                The numbers we publish, and the numbers we don&rsquo;t hide.
              </h2>
            </div>
            <p className="text-xs text-fg-subtle max-w-xs leading-relaxed">
              A share of the early figures is drawn from retrospective
              replay of historical sessions, marked as such in every
              audit. Live-collected numbers grow each session.
            </p>
          </div>
        </Reveal>
        <dl className="grid gap-x-6 gap-y-12 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((s, i) => (
            <Reveal key={s.label} delay={i * 110}>
              <div className="group border-l-2 border-accent/40 pl-5 transition-colors hover:border-warm/60">
                <StatMark kind={s.mark} className="mb-4 transition-opacity group-hover:opacity-100 opacity-80" />
                <dd className="font-serif text-4xl md:text-5xl text-fg tabnum leading-none">
                  <AnimatedCounter value={s.raw} format={s.format} />
                </dd>
                <dt className="mt-4 text-xs uppercase tracking-[0.14em] text-fg-muted">
                  {s.label}
                </dt>
                <p className="mt-2 text-sm text-fg-subtle leading-relaxed">
                  {s.sub}
                </p>
              </div>
            </Reveal>
          ))}
        </dl>
      </div>
    </section>
  );
}
