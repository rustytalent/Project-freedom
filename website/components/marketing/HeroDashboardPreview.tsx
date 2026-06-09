import { CalibrationSparkline } from "@/components/calibration/CalibrationSparkline";
import { mockLatestSummary, mockTimeSeries } from "@/lib/outcome-log-mock";
import { buildDateline } from "@/lib/clock";

/**
 * HeroDashboardPreview: the right-side artifact that fills the hero.
 *
 * Renders a small, redacted facsimile of a live calibration panel:
 * a header strip with publish timestamp, a 30-day touch-watch
 * sparkline, a per-confidence-bucket distribution strip, three
 * status rows for the three prediction heads, and a footer line
 * with predictions resolved overnight.
 *
 * The bucket strip is the precision signal - the detail only someone
 * who actually thinks about calibration would design. It makes the
 * "this works" click in the first few seconds of looking.
 *
 * Numbers come from the same mock outcome-log view that powers
 * /track-record so the artifact and dashboard always agree.
 * Nothing exposes private technique names or per-symbol data.
 */
function Row({
  label,
  value,
  tone,
  delayMs,
}: {
  label: string;
  value: string;
  tone: "calibrated" | "drift" | "neutral";
  delayMs: number;
}) {
  const dot =
    tone === "calibrated"
      ? "bg-calibrated"
      : tone === "drift"
        ? "bg-drift"
        : "bg-fg-subtle";
  const valColor =
    tone === "drift" ? "text-drift" : tone === "calibrated" ? "text-calibrated" : "text-fg";
  return (
    <div
      className="row-fade-in flex items-center justify-between gap-4 py-2.5 border-t border-border first:border-t-0"
      style={{ ["--row-delay" as string]: `${delayMs}ms` }}
    >
      <div className="flex items-center gap-2.5 min-w-0">
        <span className={`inline-block w-1.5 h-1.5 rounded-full ${dot} shrink-0`} />
        <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-fg-muted truncate">
          {label}
        </span>
      </div>
      <span className={`font-mono text-xs tabnum ${valColor}`}>{value}</span>
    </div>
  );
}

/**
 * Per-confidence-bucket distribution strip. Four bars (very-high,
 * high, moderate, low) showing the touch-watch head's bucket sizes.
 * Height encodes N (count of predictions resolved). Bar fill colour
 * encodes whether that bucket is calibrated, drifting, or neutral
 * based on the published calibration_error vs a small tolerance.
 */
function BucketStrip() {
  const summary = mockLatestSummary();
  const tw = summary.filter((r) => r.prediction_type === "touch_watch");
  // Stable ordering for the visual; never sort by N or the bars dance.
  const order = ["very_high", "high", "moderate", "low"] as const;
  const rows = order.map((b) => tw.find((r) => r.confidence_bucket === b)!);
  const maxN = Math.max(...rows.map((r) => r.n));
  const TOL = 0.05;
  return (
    <div className="px-5 pt-4 pb-2">
      <div className="flex items-baseline justify-between mb-2.5">
        <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-fg-muted">
          Confidence buckets · touch-watch
        </p>
        <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-fg-subtle">
          n / bucket
        </p>
      </div>
      <div className="flex items-end gap-2 h-12">
        {rows.map((r, i) => {
          const h = Math.max(12, Math.round((r.n / maxN) * 100));
          const within = Math.abs(r.calibration_error) <= TOL;
          const fill = within ? "bg-calibrated/80" : "bg-drift/80";
          return (
            <div
              key={r.confidence_bucket}
              className="flex-1 flex flex-col items-center justify-end h-full gap-1.5"
            >
              <span className="font-mono text-[9px] text-fg-muted tabnum">
                {r.n}
              </span>
              <div
                className={`w-full ${fill} bar-rise transition-colors`}
                style={{
                  height: `${h}%`,
                  ["--bar-delay" as string]: `${600 + i * 60}ms`,
                }}
                aria-hidden="true"
              />
            </div>
          );
        })}
      </div>
      <div className="flex items-end gap-2 mt-1.5">
        {(["VH", "H", "M", "L"] as const).map((l) => (
          <span
            key={l}
            className="flex-1 text-center font-mono text-[9px] uppercase tracking-[0.16em] text-fg-subtle"
          >
            {l}
          </span>
        ))}
      </div>
    </div>
  );
}

export function HeroDashboardPreview() {
  const series = mockTimeSeries();
  const last30 = series.slice(-30);
  const meanAbsErr =
    last30.reduce((a, d) => a + Math.abs(d.touch_watch_calibration_error), 0) /
    last30.length;
  const dl = buildDateline();

  return (
    <div
      className="materialize-in relative rounded-sm border border-border bg-bg-raised overflow-hidden hero-artifact"
      aria-hidden="true"
    >
      <div className="absolute inset-0 panel-grid opacity-60 pointer-events-none" />
      <div className="relative">
        {/* Top strip */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-border bg-bg/60">
          <div className="flex items-center gap-2.5">
            <span className="live-dot" />
            <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-fg">
              Today&rsquo;s brief
            </span>
            <span className="font-mono text-[11px] text-fg-subtle">
              · {dl.todayShort} · 08:30 IST
            </span>
          </div>
          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-fg-subtle">
            Subscriber portal
          </span>
        </div>

        {/* Sparkline */}
        <div className="px-4 pt-4 pb-1">
          <div className="flex items-baseline justify-between mb-2 px-1">
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-fg-muted">
              Calibration error · 30d
            </p>
            <p className="font-mono text-xs text-fg tabnum">
              mean {(meanAbsErr * 100).toFixed(1)}%
            </p>
          </div>
          <CalibrationSparkline
            data={last30}
            height={100}
            showAxes={false}
            animate
            animationBeginMs={500}
            animationDurationMs={900}
          />
        </div>

        {/* Bucket distribution strip - the precision signal */}
        <div className="border-t border-border">
          <BucketStrip />
        </div>

        {/* Status rows */}
        <div className="px-5 pt-2 pb-4 border-t border-border">
          <Row label="Touch-watch head" value="On target" tone="calibrated" delayMs={1000} />
          <Row label="Avoidance head" value="Within tolerance" tone="calibrated" delayMs={1080} />
          <Row label="Options-strike head" value="Drift flagged · +11%" tone="drift" delayMs={1160} />
        </div>

        {/* Footer strip */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-border bg-bg/40">
          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-fg-subtle">
            247 predictions audited overnight
          </span>
          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-warm-glow">
            Next · {dl.nextBriefLabel}
          </span>
        </div>
      </div>
    </div>
  );
}
