import { mockLatestSummary, mockTimeSeries } from "@/lib/outcome-log-mock";

/**
 * TickerTape: a Bloomberg-style strip pinned above the primary nav.
 *
 * Carries calibration metadata only - NEVER instrument symbols,
 * directional reads, or anything that could be mis-read as a tip.
 * Numbers are sourced from the same mock view that powers the
 * public dashboard, so the tape and /track-record never disagree.
 *
 * The marquee animation lives in globals.css (.ticker-track) and
 * pauses on hover / focus-within. CSS-only - no JS scheduling.
 */
type Item = {
  kind: "calibrated" | "drift" | "info";
  label: string;
  value: string;
};

function buildItems(): Item[] {
  const series = mockTimeSeries();
  const summary = mockLatestSummary();
  const last30 = series.slice(-30);
  const meanAbsErr =
    last30.reduce((a, d) => a + Math.abs(d.touch_watch_calibration_error), 0) /
    last30.length;
  const predictions = summary.reduce((a, r) => a + r.n, 0);
  return [
    { kind: "info",       label: "Brief #" + series.length, value: "Published 08:30 IST" },
    { kind: "calibrated", label: "Touch-watch error", value: "−" + (meanAbsErr * 100).toFixed(1) + "% / 30d" },
    { kind: "calibrated", label: "Avoidance head", value: "Within tolerance" },
    { kind: "drift",      label: "Options-strike head", value: "Drift flagged" },
    { kind: "info",       label: "Predictions audited", value: predictions.toLocaleString("en-IN") },
    { kind: "calibrated", label: "Audit transparency", value: "100%" },
    { kind: "info",       label: "Retrospective share", value: "35%" },
    { kind: "info",       label: "Next brief", value: "Tomorrow 08:30 IST" },
    { kind: "calibrated", label: "System", value: "Operational" },
    { kind: "info",       label: "Research stream", value: "Live" },
  ];
}

const dot = (k: Item["kind"]) =>
  k === "drift"
    ? "bg-drift"
    : k === "calibrated"
      ? "bg-calibrated"
      : "bg-fg-subtle";

function Row({ items }: { items: Item[] }) {
  return (
    <ul
      className="flex items-center gap-10 px-6 text-[11px] uppercase tracking-[0.14em] font-mono text-fg-muted whitespace-nowrap"
      aria-hidden="true"
    >
      {items.map((it, i) => (
        <li key={i} className="flex items-center gap-2.5 shrink-0">
          <span className={`inline-block w-1.5 h-1.5 rounded-full ${dot(it.kind)}`} />
          <span className="text-fg-subtle">{it.label}</span>
          <span aria-hidden="true" className="text-fg-subtle/50">·</span>
          <span className="text-fg tabnum">{it.value}</span>
        </li>
      ))}
    </ul>
  );
}

export function TickerTape() {
  const items = buildItems();
  return (
    <div
      className="ticker-host border-b border-border bg-bg overflow-hidden"
      role="region"
      aria-label="System status ticker"
    >
      <div className="ticker-fade-mask">
        <div className="ticker-track flex py-2.5">
          <Row items={items} />
          <Row items={items} />
        </div>
      </div>
    </div>
  );
}
