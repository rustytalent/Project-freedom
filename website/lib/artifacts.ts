// Artifact types and tier policy.
//
// The engine (VPS) generates research artifacts (briefs, diagnosis
// reports, calibration exports). It POSTs them to the website at
// /api/v1/artifacts. The website stores them, lists them on the
// subscriber portal, and serves tier-gated download URLs.
//
// Each artifact carries an explicit `tier` — the lowest subscription
// level required to download. The portal hides artifacts above the
// reader's tier rather than showing locked padlocks.

export type ArtifactKind =
  | "daily_brief_pdf"
  | "daily_brief_email"
  | "swing_brief_pdf"
  | "diagnosis_report_pdf"
  | "calibration_summary_csv"
  | "outcome_log_export_csv"
  | "yesterday_audit_csv"
  | "options_strikes_csv"
  | "weekly_research_note_pdf";

export const ARTIFACT_KIND_LABELS: Record<ArtifactKind, string> = {
  daily_brief_pdf: "Daily Brief (PDF)",
  daily_brief_email: "Daily Brief (email archive)",
  swing_brief_pdf: "Swing Brief (PDF)",
  diagnosis_report_pdf: "Diagnosis Report",
  calibration_summary_csv: "Calibration summary",
  outcome_log_export_csv: "Outcome log export",
  yesterday_audit_csv: "Yesterday Audit",
  options_strikes_csv: "Options strikes in play",
  weekly_research_note_pdf: "Weekly research note",
};

export const ARTIFACT_KIND_DESCRIPTIONS: Record<ArtifactKind, string> = {
  daily_brief_pdf:
    "Today's structured research brief, in print-friendly PDF.",
  daily_brief_email:
    "The same brief in email format (text + light HTML).",
  swing_brief_pdf:
    "Multi-day positional research, published Monday pre-open.",
  diagnosis_report_pdf:
    "Your strategy diagnosis report.",
  calibration_summary_csv:
    "Per-bucket calibration error for the last 90 trading days.",
  outcome_log_export_csv:
    "Per-prediction outcome log for your tier window.",
  yesterday_audit_csv:
    "Calibrated hit-rate audit for the previous IST trading session.",
  options_strikes_csv:
    "Index option strikes flagged in today's brief, with proximity numbers.",
  weekly_research_note_pdf:
    "Long-form Monday note on regime structure.",
};

// Tiers used for access control.
//
// Subscribers carry a "tier" set in their Supabase row. The download
// gate compares the artifact's tier against the reader's tier-set.
// "public" artifacts (e.g. weekly note teaser, anonymised calibration
// CSV) are downloadable without authentication.
export type AccessTier =
  | "public"
  | "free_signup"
  | "paid_intraday"
  | "paid_multi_product"
  | "paid_diagnosis";

export const TIER_LABELS: Record<AccessTier, string> = {
  public: "Public",
  free_signup: "Signed-in (free 7-day window)",
  paid_intraday: "Intraday subscribers",
  paid_multi_product: "Multi-product subscribers",
  paid_diagnosis: "Diagnosis customers",
};

// Order matters: higher index = higher-privilege tier.
const TIER_RANK: Record<AccessTier, number> = {
  public: 0,
  free_signup: 1,
  paid_intraday: 2,
  paid_multi_product: 3,
  paid_diagnosis: 4,
};

export function canAccess(
  reader_tiers: ReadonlySet<AccessTier>,
  required: AccessTier,
): boolean {
  if (required === "public") return true;
  for (const t of reader_tiers) {
    if (TIER_RANK[t] >= TIER_RANK[required]) return true;
  }
  // Diagnosis is orthogonal: diagnosis customers see their own report,
  // multi-product subs do NOT inherit access to other people's reports.
  if (required === "paid_diagnosis") {
    return reader_tiers.has("paid_diagnosis");
  }
  return false;
}

export type ArtifactRecord = {
  id: string;                       // stable hash-of-content + kind
  kind: ArtifactKind;
  trading_date_ist?: string;         // for daily/swing artifacts
  generated_at_utc: string;
  filename: string;
  bytes: number;
  sha256: string;
  tier: AccessTier;
  // Storage adapter knows how to resolve this to an actual download URL.
  // For the in-memory adapter it's an opaque key.
  storage_key: string;
  description?: string;
  // Free-form engine-side context — useful for audit, never shown
  // to the customer. Same shape rule as the outcome log: never
  // contains methodology details.
  meta?: Record<string, unknown>;
};

export function isToday(trading_date_ist?: string): boolean {
  if (!trading_date_ist) return false;
  const ist = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Kolkata",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
  return trading_date_ist === ist;
}

export function relativeTimeFromNow(generated_at_utc: string): string {
  const ts = new Date(generated_at_utc).getTime();
  const now = Date.now();
  const seconds = Math.max(0, Math.floor((now - ts) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  const days = Math.floor(seconds / 86400);
  if (days < 30) return `${days} day${days === 1 ? "" : "s"} ago`;
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).format(new Date(generated_at_utc));
}
