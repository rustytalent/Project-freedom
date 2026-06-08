import { CalibrationSparkline } from "@/components/calibration/CalibrationSparkline";
import { mockTimeSeries } from "@/lib/outcome-log-mock";

/**
 * HeroDashboardPreview: the right-side artifact that fills the hero.
 *
 * Renders a small, redacted facsimile of a live calibration panel:
 * a header strip with publish timestamp, a 30-day touch-watch
 * sparkline, three status chips for the three prediction heads,
 * and a footer line with predictions resolved overnight.
 *
 * Numbers come from the same mock outcome-log view that powers
 * /track-record so the artifact and dashboard always agree.
 * Nothing here exposes private technique names or per-symbol data.
 */
function Row({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "calibrated" | "drift" | "neutral";
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
    <div className="flex items-center justify-between gap-4 py-2.5 border-t border-border first:border-t-0">
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

export function HeroDashboardPreview() {
  const series = mockTimeSeries();
  const last30 = series.slice(-30);
  const meanAbsErr =
    last30.reduce((a, d) => a + Math.abs(d.touch_watch_calibration_error), 0) /
    last30.length;

  return (
    <div
      className="relative rounded-sm border border-border bg-bg-raised shadow-[0_24px_64px_-32px_rgba(0,0,0,0.6)] overflow-hidden"
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
              · 06 JUN · 08:30 IST
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
              Touch-watch calibration error · 30d
            </p>
            <p className="font-mono text-xs text-fg tabnum">
              mean {(meanAbsErr * 100).toFixed(1)}%
            </p>
          </div>
          <CalibrationSparkline data={last30} height={120} showAxes={false} />
        </div>

        {/* Status rows */}
        <div className="px-5 pt-2 pb-4">
          <Row label="Touch-watch head" value="On target" tone="calibrated" />
          <Row label="Avoidance head" value="Within tolerance" tone="calibrated" />
          <Row label="Options-strike head" value="Drift flagged · +11%" tone="drift" />
        </div>

        {/* Footer strip */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-border bg-bg/40">
          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-fg-subtle">
            247 predictions audited overnight
          </span>
          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-accent">
            Next · 08:30 IST
          </span>
        </div>
      </div>
    </div>
  );
}
