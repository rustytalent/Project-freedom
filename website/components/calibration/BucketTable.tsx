import type { CalibrationBucketRow } from "@/lib/outcome-log-mock";

export function BucketTable({ rows }: { rows: CalibrationBucketRow[] }) {
  // Group rows by prediction_type, with a fixed bucket order.
  const types = Array.from(new Set(rows.map((r) => r.prediction_type)));
  const order = ["very_high", "high", "moderate", "low"];
  return (
    <div className="space-y-8">
      {types.map((t) => {
        const sub = rows
          .filter((r) => r.prediction_type === t)
          .sort(
            (a, b) =>
              order.indexOf(a.confidence_bucket) -
              order.indexOf(b.confidence_bucket),
          );
        return (
          <div key={t}>
            <h3 className="text-sm text-fg uppercase tracking-wider mb-3">
              {t.replace("_", " ")}
            </h3>
            <div className="overflow-x-auto -mx-1">
              <table className="w-full font-mono text-xs tabnum">
                <thead>
                  <tr className="text-fg-subtle border-b border-border">
                    <th className="text-left py-2 px-3 font-normal">
                      Bucket
                    </th>
                    <th className="text-right py-2 px-3 font-normal">n</th>
                    <th className="text-right py-2 px-3 font-normal">
                      Hit rate
                    </th>
                    <th className="text-right py-2 px-3 font-normal">
                      Mean predicted
                    </th>
                    <th className="text-right py-2 px-3 font-normal">
                      Calibration error
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sub.map((r, i) => {
                    const drifting = Math.abs(r.calibration_error) > 0.08;
                    return (
                      <tr key={i} className="border-b border-border/40">
                        <td className="py-2 px-3 text-fg">
                          {r.confidence_bucket}
                        </td>
                        <td className="py-2 px-3 text-right text-fg-muted">
                          {r.n}
                        </td>
                        <td className="py-2 px-3 text-right text-fg">
                          {(r.hit_rate * 100).toFixed(0)}%
                        </td>
                        <td className="py-2 px-3 text-right text-fg-muted">
                          {(r.mean_predicted_p * 100).toFixed(0)}%
                        </td>
                        <td
                          className={`py-2 px-3 text-right ${
                            drifting ? "text-drift" : "text-calibrated"
                          }`}
                        >
                          {r.calibration_error >= 0 ? "+" : ""}
                          {r.calibration_error.toFixed(2)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}
