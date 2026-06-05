// Mock data for the calibration dashboard.
//
// In production this module reads from a SANITISED Supabase view
// (`outcome_log_public_view`) that the engine cron job populates
// nightly. For local dev / preview deploys without a Supabase
// connection, we return deterministic mock data shaped like the
// real view. The shapes match `docs/outcome_logging_schema.md`
// in the engine repo.
//
// CRITICAL: this view NEVER exposes prediction_ids, specific symbols,
// or specific levels. Only aggregate per-bucket calibration metrics.

export type CalibrationBucketRow = {
  trading_date_ist: string;       // YYYY-MM-DD
  prediction_type: string;        // proximity | avoidance | options_strike
  confidence_bucket: string;      // very_high | high | moderate | low
  n: number;
  hit_rate: number;
  mean_predicted_p: number;
  calibration_error: number;      // mean_predicted_p - hit_rate
  is_retrospective_share: number; // 0..1
};

export type DriftFlag = {
  prediction_type: string;
  confidence_bucket: string;
  reason: string;
};

// Deterministic mock generator — same seed → same numbers.
function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

const PREDICTION_TYPES = ["proximity", "avoidance", "options_strike"] as const;
const BUCKETS = ["very_high", "high", "moderate", "low"] as const;

const BUCKET_CENTERS: Record<string, number> = {
  very_high: 0.86,
  high: 0.72,
  moderate: 0.57,
  low: 0.32,
};

export function mockLatestSummary(): CalibrationBucketRow[] {
  const rand = seeded(20260605);
  const rows: CalibrationBucketRow[] = [];
  for (const ptype of PREDICTION_TYPES) {
    for (const bucket of BUCKETS) {
      const centre = BUCKET_CENTERS[bucket];
      const drift = (rand() - 0.5) * 0.10;       // +/- 5%
      const n = Math.floor(40 + rand() * 220);
      rows.push({
        trading_date_ist: "2026-06-04",
        prediction_type: ptype,
        confidence_bucket: bucket,
        n,
        hit_rate: Math.max(0.05, Math.min(0.95, centre + drift)),
        mean_predicted_p: centre,
        calibration_error: -drift,
        is_retrospective_share: 0.35,
      });
    }
  }
  return rows;
}

export function mockTimeSeries(): Array<{
  trading_date_ist: string;
  proximity_calibration_error: number;
  avoidance_calibration_error: number;
  options_calibration_error: number;
}> {
  const rand = seeded(20260114);
  const out = [];
  // Last 90 trading days, ascending.
  const end = new Date("2026-06-04T00:00:00+05:30");
  for (let i = 89; i >= 0; i--) {
    const d = new Date(end);
    d.setDate(d.getDate() - i);
    if (d.getDay() === 0 || d.getDay() === 6) continue;
    out.push({
      trading_date_ist: d.toISOString().slice(0, 10),
      proximity_calibration_error: (rand() - 0.5) * 0.16,
      avoidance_calibration_error: (rand() - 0.5) * 0.10,
      options_calibration_error: (rand() - 0.5) * 0.20,
    });
  }
  return out;
}

export function mockDriftFlags(): DriftFlag[] {
  return [
    {
      prediction_type: "options_strike",
      confidence_bucket: "very_high",
      reason: "calibration_error +0.11 over 7d window (drift threshold 0.08)",
    },
  ];
}

export function mockRetrospectiveShare(): number {
  // The Stream G flag: fraction of the last-90-day window that came
  // from retrospective replay vs live brief generation.
  return 0.35;
}
