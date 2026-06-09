import type { Metadata } from "next";
import { BucketTable } from "@/components/calibration/BucketTable";
import { CalibrationTimeSeries } from "@/components/calibration/CalibrationTimeSeries";
import { Card, CardContent, CardDescription, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/marketing/PageHeader";
import { Reveal } from "@/components/ui/Reveal";
import { AnimatedCounter, type CounterFormat } from "@/components/ui/AnimatedCounter";
import { StatMark } from "@/components/marketing/StatMark";
import {
  mockDriftFlags,
  mockLatestSummary,
  mockRetrospectiveShare,
  mockTimeSeries,
} from "@/lib/outcome-log-mock";

export const metadata: Metadata = {
  title: "Track record",
  description:
    "Live calibration dashboard. Per-bucket hit rate, calibration error over time, drift flags.",
};

// Refresh hourly via ISR. In production this is a Supabase read; in
// preview / dev it returns deterministic mock data.
export const revalidate = 3600;

type Stat = {
  label: string;
  raw: number;
  format: CounterFormat;
  mark: "staircase" | "tally" | "oscillation" | "disc";
};

function buildStats(
  series: ReturnType<typeof mockTimeSeries>,
  summary: ReturnType<typeof mockLatestSummary>,
  drift: ReturnType<typeof mockDriftFlags>,
): Stat[] {
  const last30 = series.slice(-30);
  const meanAbsErr =
    last30.reduce((a, d) => a + Math.abs(d.touch_watch_calibration_error), 0) /
    last30.length;
  return [
    {
      label: "Briefs published",
      raw: series.length,
      format: { kind: "int" },
      mark: "staircase",
    },
    {
      label: "Predictions resolved",
      raw: summary.reduce((a, r) => a + r.n, 0),
      format: { kind: "int" },
      mark: "tally",
    },
    {
      label: "Calibration error · 30d",
      raw: meanAbsErr * 100,
      format: { kind: "decimal", decimals: 1, suffix: "%" },
      mark: "oscillation",
    },
    {
      label: "Drift heads active",
      raw: drift.length,
      format: { kind: "decimal", decimals: 0 },
      mark: "disc",
    },
  ];
}

export default function TrackRecordPage() {
  const buckets = mockLatestSummary();
  const series = mockTimeSeries();
  const drift = mockDriftFlags();
  const retroShare = mockRetrospectiveShare();
  const stats = buildStats(series, buckets, drift);
  const updated = new Date();
  const updatedIST = updated.toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
    dateStyle: "medium",
    timeStyle: "short",
  });
  return (
    <>
      <PageHeader
        section="Track record"
        editionTag="LIVE"
        dateline={`UPDATED ${updatedIST} IST`}
        status={
          <span className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-calibrated">
            <span className="live-dot" />
            <span>Nightly ingest · ok</span>
          </span>
        }
        title={
          <>
            The dashboard we built to keep ourselves{" "}
            <span className="text-warm">honest</span>.
          </>
        }
        lead={
          <p>
            Every prediction every brief publishes is logged, resolved
            against actual market outcomes, and aggregated into the
            per-bucket calibration view below. No claim leaves this
            page un-audited.
          </p>
        }
        meta="Sanitised public view · per-prediction detail lives in subscriber portals"
      />

      {/* Top stats strip - high-level numbers before the visitor scrolls */}
      <section className="border-t border-border bg-bg-raised/40">
        <div className="max-w-dash mx-auto px-6 py-12 md:py-14">
          <dl className="grid gap-x-6 gap-y-10 sm:grid-cols-2 lg:grid-cols-4">
            {stats.map((s, i) => (
              <Reveal key={s.label} delay={i * 100}>
                <div className="group border-l-2 border-accent/40 pl-5 transition-colors hover:border-warm/60">
                  <StatMark kind={s.mark} className="mb-3 opacity-80" />
                  <dd className="font-serif text-3xl md:text-4xl text-fg tabnum leading-none">
                    <AnimatedCounter value={s.raw} format={s.format} />
                  </dd>
                  <dt className="mt-3 text-xs uppercase tracking-[0.14em] text-fg-muted">
                    {s.label}
                  </dt>
                </div>
              </Reveal>
            ))}
          </dl>
        </div>
      </section>

      <div className="max-w-dash mx-auto px-6 py-16 md:py-20">
        {/* Retrospective-share disclosure */}
        {retroShare > 0.1 && (
          <Reveal>
            <div className="mb-16 border border-drift/40 bg-drift/[0.06] rounded-sm overflow-hidden max-w-3xl">
              <div className="px-6 py-3 border-b border-drift/30 bg-drift/10 flex items-center gap-2.5">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-drift" />
                <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-drift">
                  Transparency note · calibration window mix
                </span>
              </div>
              <p className="px-6 py-5 text-sm text-fg leading-relaxed">
                <span className="font-mono text-fg-muted tabnum">
                  {(retroShare * 100).toFixed(0)}%
                </span>{" "}
                of the last 90 days&rsquo; resolved predictions were
                backfilled from historical bundles (retrospective
                replay), and{" "}
                <span className="font-mono text-fg-muted tabnum">
                  {((1 - retroShare) * 100).toFixed(0)}%
                </span>{" "}
                were collected live. We disclose this explicitly because
                retrospective replay can encode hindsight; treat the
                displayed numbers as a directional read of the
                engine&rsquo;s calibration, not a live-only track record.
              </p>
            </div>
          </Reveal>
        )}

        {/* Drift flags */}
        <Reveal>
          <section className="mb-20">
            <div className="flex items-baseline justify-between gap-4 mb-6">
              <h2 className="font-serif text-2xl md:text-3xl text-fg">
                Drift flags
              </h2>
              <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-fg-subtle">
                {drift.length === 0 ? "0 active" : `${drift.length} active`}
              </span>
            </div>
            {drift.length === 0 ? (
              <div className="border border-border bg-bg-raised/40 rounded-sm p-6 max-w-2xl">
                <p className="flex items-center gap-3 text-sm text-fg-muted">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-calibrated" />
                  No heads flagged as drifting in the current calibration
                  window.
                </p>
              </div>
            ) : (
              <ul className="grid gap-3 max-w-3xl">
                {drift.map((d, i) => (
                  <li
                    key={i}
                    className="border border-border bg-bg-raised/40 rounded-sm overflow-hidden"
                  >
                    <div className="px-5 py-2.5 border-b border-border bg-drift/[0.06] flex items-center gap-2.5">
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-drift" />
                      <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-drift">
                        {d.prediction_type} · {d.confidence_bucket}
                      </span>
                    </div>
                    <p className="px-5 py-3 text-sm text-fg-muted leading-relaxed">
                      {d.reason}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </Reveal>

        {/* Time series */}
        <Reveal>
          <section className="mb-20">
            <Card>
              <CardTitle>Calibration error · last 90 trading days</CardTitle>
              <CardDescription>
                Mean predicted probability minus actual hit rate, per
                prediction type. Dashed lines mark the &plusmn;8% drift
                tolerance.
              </CardDescription>
              <CardContent className="mt-6">
                <CalibrationTimeSeries data={series} />
              </CardContent>
            </Card>
          </section>
        </Reveal>

        {/* Per-bucket table */}
        <Reveal>
          <section>
            <h2 className="font-serif text-2xl md:text-3xl text-fg mb-3">
              Per-bucket calibration
            </h2>
            <p className="text-fg-muted text-sm mb-8 max-w-prose leading-relaxed">
              A bucket is calibrated when its hit rate is close to its
              mean predicted probability &mdash; the calibration error
              column should hover around zero. Errors larger than{" "}
              <span className="font-mono">|0.08|</span> trigger a drift
              flag (above) and a note in the next brief&rsquo;s
              confidence section.
            </p>
            <BucketTable rows={buckets} />
          </section>
        </Reveal>

        <Reveal>
          <p className="mt-20 pt-8 border-t border-border text-xs text-fg-subtle leading-relaxed max-w-prose">
            These numbers are computed from a sanitised public view of
            the outcome log. Specific prediction ids, symbols, and
            levels are never exposed on this surface. Subscribers see
            the full per-prediction detail in their portal.
          </p>
        </Reveal>
      </div>
    </>
  );
}
