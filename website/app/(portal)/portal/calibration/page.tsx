import type { Metadata } from "next";
import { BucketTable } from "@/components/calibration/BucketTable";
import { CalibrationTimeSeries } from "@/components/calibration/CalibrationTimeSeries";
import { Card, CardContent, CardDescription, CardTitle } from "@/components/ui/card";
import {
  mockLatestSummary,
  mockTimeSeries,
} from "@/lib/outcome-log-mock";

export const metadata: Metadata = {
  title: "Calibration",
};

export default function PortalCalibrationPage() {
  const buckets = mockLatestSummary();
  const series = mockTimeSeries();
  return (
    <div className="max-w-dash mx-auto px-6 py-14">
      <header className="mb-10 max-w-prose">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-3">
          Calibration
        </p>
        <h1 className="font-serif text-3xl text-fg">
          The same dashboard, with your personal audit overlay.
        </h1>
        <p className="mt-4 text-fg-muted leading-relaxed">
          Live engine calibration on the left. Once you mark which
          calls you acted on, the personal-audit overlay (coming soon)
          will show your hit rate inside the engine&rsquo;s
          calibration.
        </p>
      </header>

      <section className="mb-12">
        <Card>
          <CardTitle>Calibration error — last 90 trading days</CardTitle>
          <CardDescription>
            Dashed lines mark the ±8% drift tolerance.
          </CardDescription>
          <CardContent className="mt-6">
            <CalibrationTimeSeries data={series} />
          </CardContent>
        </Card>
      </section>

      <section>
        <h2 className="font-serif text-2xl text-fg mb-4">
          Per-bucket calibration
        </h2>
        <BucketTable rows={buckets} />
      </section>
    </div>
  );
}
