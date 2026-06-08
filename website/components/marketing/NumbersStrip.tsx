import { mockLatestSummary, mockTimeSeries } from "@/lib/outcome-log-mock";

type Stat = { label: string; value: string; sub: string };

function buildStats(): Stat[] {
  const summary = mockLatestSummary();
  const series = mockTimeSeries();
  const last30 = series.slice(-30);
  const meanAbsErr =
    last30.reduce((a, d) => a + Math.abs(d.touch_watch_calibration_error), 0) /
    last30.length;
  const briefsPublished = series.length;
  const predictionsResolved = summary.reduce((a, r) => a + r.n, 0);
  return [
    {
      label: "Briefs published",
      value: briefsPublished.toLocaleString("en-IN"),
      sub: "since launch, across equity, options, index",
    },
    {
      label: "Predictions audited",
      value: predictionsResolved.toLocaleString("en-IN"),
      sub: "every claim resolved, hit or miss, on the next session",
    },
    {
      label: "Mean calibration error",
      value: (meanAbsErr * 100).toFixed(1) + "%",
      sub: "rolling 30 trading days, touch-watch head",
    },
    {
      label: "Audit transparency",
      value: "100%",
      sub: "no claim is published without an outcome the next day",
    },
  ];
}

export function NumbersStrip() {
  const stats = buildStats();
  return (
    <section className="border-t border-border bg-bg-raised/40">
      <div className="max-w-dash mx-auto px-6 py-16">
        <div className="flex flex-wrap items-end justify-between gap-6 mb-12">
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
        <dl className="grid gap-x-6 gap-y-10 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((s) => (
            <div key={s.label} className="border-l-2 border-accent/40 pl-5">
              <dd className="font-serif text-4xl text-fg tabnum leading-none">
                {s.value}
              </dd>
              <dt className="mt-3 text-xs uppercase tracking-wider text-fg-muted">
                {s.label}
              </dt>
              <p className="mt-2 text-sm text-fg-subtle leading-relaxed">
                {s.sub}
              </p>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
