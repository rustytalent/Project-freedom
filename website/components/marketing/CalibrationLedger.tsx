import { mockLatestSummary } from "@/lib/outcome-log-mock";

/**
 * CalibrationLedger: a small ledger panel showing a handful of
 * resolved predictions from yesterday's brief, hit-or-missed
 * against actual outcomes, with calibration error per row.
 *
 * Visually echoes a real receipt / accounting ledger: monospace,
 * tabular numerals, ledger lines between rows, ledger total at
 * the bottom. Stands in for "this is what an audit looks like"
 * without the noise of a full dashboard.
 *
 * Data is derived from the same mock outcome-log view that powers
 * /track-record so the panel never disagrees with the dashboard.
 * Rows show prediction type + confidence bucket + hit/miss + error,
 * all of which are fields on the SANITISED public view.
 */

const TOL = 0.05;

type LedgerRow = {
  predId: string;
  type: string;
  bucket: string;
  hit: boolean;
  err: number;
};

const TYPE_LABEL: Record<string, string> = {
  touch_watch: "touch-watch",
  avoidance: "avoidance",
  options_strike: "opt-strike",
};

const BUCKET_LABEL: Record<string, string> = {
  very_high: "V-HIGH",
  high: "HIGH",
  moderate: "MOD",
  low: "LOW",
};

function buildRows(): LedgerRow[] {
  // Take 6 rows across the three heads, derived deterministically
  // from mockLatestSummary so the panel looks like a real ledger
  // pulled at a specific instant rather than randomised.
  const summary = mockLatestSummary();
  const picks: LedgerRow[] = [];
  let predBase = 4112;
  const order = [
    { type: "touch_watch", bucket: "high" },
    { type: "touch_watch", bucket: "very_high" },
    { type: "avoidance", bucket: "moderate" },
    { type: "options_strike", bucket: "high" },
    { type: "avoidance", bucket: "high" },
    { type: "options_strike", bucket: "very_high" },
  ];
  for (const o of order) {
    const r = summary.find(
      (x) => x.prediction_type === o.type && x.confidence_bucket === o.bucket,
    );
    if (!r) continue;
    // "Hit" if the bucket's hit-rate sample lands above 50% & not
    // drifting; otherwise count as a miss. Doesn't claim a literal
    // per-prediction outcome - just shapes a plausible ledger.
    const hit =
      Math.abs(r.calibration_error) <= TOL && r.hit_rate >= 0.5;
    picks.push({
      predId: "#" + (predBase++).toString(),
      type: o.type,
      bucket: o.bucket,
      hit,
      err: r.calibration_error,
    });
  }
  return picks;
}

export function CalibrationLedger({ label }: { label: string }) {
  const rows = buildRows();
  const hits = rows.filter((r) => r.hit).length;
  return (
    <figure
      className="bg-bg-raised border border-border rounded-sm overflow-hidden"
      aria-label={`Calibration ledger - ${label}`}
    >
      {/* Header strip */}
      <div className="px-5 py-3 border-b border-border bg-bg/60 flex items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <span className="live-dot" />
          <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-fg">
            Yesterday ledger
          </span>
          <span className="font-mono text-[11px] text-fg-subtle">
            · 05 JUN
          </span>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-fg-subtle">
          public view
        </span>
      </div>

      {/* Column headers */}
      <div className="px-5 py-2 border-b border-border bg-bg/40 grid grid-cols-[auto_1fr_auto_auto_auto] gap-x-4 font-mono text-[10px] uppercase tracking-[0.16em] text-fg-subtle">
        <span>pred</span>
        <span>head · bucket</span>
        <span className="text-right">outcome</span>
        <span className="text-right">cal&nbsp;err</span>
        <span aria-hidden="true" className="w-2.5" />
      </div>

      {/* Rows */}
      <ul className="divide-y divide-border">
        {rows.map((r) => {
          const within = Math.abs(r.err) <= TOL;
          const outcomeColor = r.hit ? "text-calibrated" : "text-drift";
          const errColor = within ? "text-fg-muted" : "text-drift";
          const errSign = r.err >= 0 ? "+" : "−";
          return (
            <li
              key={r.predId}
              className="px-5 py-2.5 grid grid-cols-[auto_1fr_auto_auto_auto] gap-x-4 items-center font-mono text-xs tabnum"
            >
              <span className="text-fg-subtle">{r.predId}</span>
              <span className="text-fg-muted">
                {TYPE_LABEL[r.type]} · {BUCKET_LABEL[r.bucket]}
              </span>
              <span className={`text-right ${outcomeColor} text-[11px] uppercase tracking-[0.16em]`}>
                {r.hit ? "hit" : "miss"}
              </span>
              <span className={`text-right ${errColor}`}>
                {errSign}
                {Math.abs(r.err * 100).toFixed(2)}%
              </span>
              <span
                aria-hidden="true"
                className={`w-2.5 h-2.5 rounded-full ${
                  r.hit ? "bg-calibrated/80" : "bg-drift/80"
                }`}
              />
            </li>
          );
        })}
      </ul>

      {/* Total strip */}
      <div className="px-5 py-3 border-t border-border bg-bg/40 grid grid-cols-2 gap-4 font-mono text-[11px] uppercase tracking-[0.16em]">
        <span className="text-fg-muted">
          Resolved · <span className="text-fg tabnum">{rows.length}</span>
        </span>
        <span className="text-right text-fg-muted">
          Hits ·{" "}
          <span className="text-calibrated tabnum">
            {hits}/{rows.length}
          </span>
        </span>
      </div>
    </figure>
  );
}
