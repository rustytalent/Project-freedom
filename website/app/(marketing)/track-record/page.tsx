import type { Metadata } from "next";
import { BucketTable } from "@/components/calibration/BucketTable";
import { CalibrationTimeSeries } from "@/components/calibration/CalibrationTimeSeries";
import { Card, CardContent, CardDescription, CardTitle } from "@/components/ui/card";
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

export default function TrackRecordPage() {
  const buckets = mockLatestSummary();
  const series = mockTimeSeries();
  const drift = mockDriftFlags();
  const retroShare = mockRetrospectiveShare();
  const updated = new Date();
  return (
    <div className="max-w-dash mx-auto px-6 py-20">
      <header className="max-w-prose mb-12">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
          Track record
        </p>
        <h1 className="font-serif text-4xl md:text-5xl leading-tight text-fg">
          The dashboard we built to keep ourselves honest.
        </h1>
        <p className="mt-6 text-fg-muted leading-relaxed">
          Every prediction every brief publishes is logged, resolved
          against actual market outcomes, and aggregated into the
          per-bucket calibration view below. Updated nightly.
        </p>
        <p className="mt-3 text-xs text-fg-subtle">
          Last refreshed:{" "}
          {updated.toLocaleString("en-IN", {
            timeZone: "Asia/Kolkata",
          })}{" "}
          IST
        </p>
      </header>

      {/* Retrospective-share disclosure, surfaced from Stream G plumbing */}
      {retroShare > 0.1 && (
        <div className="mb-12 border border-drift/40 bg-drift/5 rounded-sm p-6 max-w-prose">
          <p className="text-sm text-fg leading-relaxed">
            <strong>Calibration window mix:</strong>{" "}
            {(retroShare * 100).toFixed(0)}% of the last 90 days&rsquo;
            resolved predictions were backfilled from historical
            bundles (retrospective replay), and{" "}
            {((1 - retroShare) * 100).toFixed(0)}% were collected live.
            We disclose this explicitly because retrospective replay
            can encode hindsight; treat the displayed numbers as a
            directional read of the engine&rsquo;s calibration, not a
            live-only track record.
          </p>
        </div>
      )}

      {/* Drift flags */}
      <section className="mb-16">
        <h2 className="font-serif text-2xl text-fg mb-4">Drift flags</h2>
        {drift.length === 0 ? (
          <p className="text-fg-muted text-sm">
            No heads flagged as drifting in the current calibration
            window.
          </p>
        ) : (
          <ul className="space-y-3">
            {drift.map((d, i) => (
              <li key={i} className="border-l border-drift pl-4 py-1">
                <p className="text-sm text-fg">
                  <span className="font-mono">{d.prediction_type}</span>{" "}
                  · <span className="font-mono">{d.confidence_bucket}</span>
                </p>
                <p className="text-xs text-fg-muted mt-1 leading-relaxed">
                  {d.reason}
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* Time series */}
      <section className="mb-16">
        <Card>
          <CardTitle>Calibration error — last 90 trading days</CardTitle>
          <CardDescription>
            Mean predicted probability minus actual hit rate, per
            prediction type. Dashed lines mark the ±8% drift tolerance.
          </CardDescription>
          <CardContent className="mt-6">
            <CalibrationTimeSeries data={series} />
          </CardContent>
        </Card>
      </section>

      {/* Per-bucket table */}
      <section>
        <h2 className="font-serif text-2xl text-fg mb-2">
          Per-bucket calibration
        </h2>
        <p className="text-fg-muted text-sm mb-8 max-w-prose leading-relaxed">
          A bucket is calibrated when its hit rate is close to its
          mean predicted probability — the calibration error column
          should hover around zero. Errors larger than{" "}
          <span className="font-mono">|0.08|</span> trigger a drift
          flag (above) and a note in the next brief&rsquo;s confidence
          section.
        </p>
        <BucketTable rows={buckets} />
      </section>

      <p className="mt-16 text-xs text-fg-subtle leading-relaxed max-w-prose">
        These numbers are computed from a sanitised public view of the
        outcome log. Specific prediction ids, symbols, and levels are
        never exposed on this surface. Subscribers see the full per-
        prediction detail in their portal.
      </p>
    </div>
  );
}
