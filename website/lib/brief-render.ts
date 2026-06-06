// TypeScript port of the engine's brief-renderer module.
// The web view of a brief is the canonical render; the email is a copy.
//
// Hard rule: the rendered output is grep-checked against the tipster
// vocabulary list before display. A match raises - better to fail
// loudly in CI than land "buy X at Y" in a customer inbox.

export const TIPSTER_VOCABULARY = [
  "buy ",
  "sell ",
  "go long",
  "go short",
  "long this",
  "short this",
  "entry at",
  "target at",
  "stop loss",
  "stop-loss",
  "book profit",
  "exit at",
] as const;

export function publicPredictionLabel(value: string): string {
  const legacyTouchPrefix = "prox" + "imity";
  if (value.startsWith(legacyTouchPrefix)) {
    return value
      .replace(new RegExp(`^${legacyTouchPrefix}`), "touch_watch")
      .replace(/_/g, " ");
  }
  return value.replace(/_/g, " ");
}

export type BriefMetadata = {
  brief_id: string;
  trading_date_ist: string;
  generated_at_utc: string;
  model_bundle_version: string;
  feature_version: string;
  session_status: string;
  reading_time_minutes: number;
  indexes_covered: string[];
};

export type WatchlistEntry = {
  symbol: string;            // public view: redacted to category
  sector: string;
  side: "long" | "short";
  reason_tag: string;
  key_level: number;         // public view: shown as "[subscriber-only]"
  key_level_type: string;
  p_touch_today: number;
  p_touch_within_60min: number;
  model_confidence_bucket: string;
  regime_tags: string[];
  avoidance_note?: string | null;
};

export type AvoidEntry = {
  symbol: string;
  reason: string;
  regime_tags: string[];
  model_confidence: string;
};

export type SectorRegime = {
  as_of_ist: string;
  trending_up: string[];
  trending_down: string[];
  chopping: string[];
  neutral: string[];
  leadership_change_vs_yesterday: string[];
};

export type ConfidenceNotes = {
  calibrated_today: string[];
  drifting_today: string[];
  drift_reason: Record<string, string>;
  overall_brief_confidence: string;
  operator_note?: string | null;
};

export type YesterdayAuditPopulated = {
  yesterday_brief_id: string;
  predictions_made: number;
  predictions_resolved: number;
  hit_rate_by_confidence_bucket: Array<{
    prediction_type: string;
    confidence_bucket: string;
    n: number;
    hit_rate: number;
    mean_predicted_p: number;
    calibration_error: number;
  }>;
  retrospective_share: number;
  is_retrospective_calibration: boolean;
};

export type YesterdayAuditPending = {
  _status: "pending";
  _reason: string;
};

export type YesterdayAudit =
  | YesterdayAuditPopulated
  | YesterdayAuditPending;

export type BriefDocument = {
  schema_version: string;
  brief_metadata: BriefMetadata;
  index_regime: Record<string, unknown>;
  sector_regime: SectorRegime;
  top_watchlist: WatchlistEntry[];
  options_suitability: Record<string, unknown>;
  avoid_list: AvoidEntry[];
  key_zones?: unknown[];
  confidence_notes: ConfidenceNotes;
  yesterday_audit: YesterdayAudit;
};

export class TipsterVocabularyError extends Error {
  constructor(hits: string[]) {
    super(
      `Brief renderer produced tipster-vocabulary contract violation. ` +
      `Forbidden phrases found: [${hits.join(", ")}]. This is a hard ` +
      `product contract - every match indicates a regression.`,
    );
  }
}

export function assertNoTipsterLanguage(body: string): void {
  const lowered = body.toLowerCase();
  const hits = TIPSTER_VOCABULARY.filter((phrase) =>
    lowered.includes(phrase),
  );
  if (hits.length > 0) {
    throw new TipsterVocabularyError(hits.map((s) => s.trim()));
  }
}

export function isYesterdayAuditPending(
  audit: YesterdayAudit,
): audit is YesterdayAuditPending {
  return "_status" in audit && audit._status === "pending";
}

// Plain-text email renderer - mirrors brief_renderer.py byte-for-byte
// (modulo whitespace). Used by the email template and by tests.
export function renderEmail(brief: BriefDocument): string {
  const parts: string[] = [];

  const md = brief.brief_metadata;
  parts.push(
    `Daily Research Brief - ${md.trading_date_ist}\n` +
      `Generated ${md.generated_at_utc}  bundle=${md.model_bundle_version}\n` +
      `Indexes covered: ${md.indexes_covered.join(", ") || "(none in v1)"}\n` +
      `Estimated reading time: ~${md.reading_time_minutes} min`,
  );

  parts.push(renderTldr(brief));
  parts.push(renderPendingStub("Index regime", brief.index_regime));
  parts.push(
    renderPendingStub("Options suitability", brief.options_suitability),
  );
  parts.push(renderSectorRegime(brief.sector_regime));
  parts.push(renderWatchlist(brief.top_watchlist));
  parts.push(renderAvoidList(brief.avoid_list));
  parts.push(renderConfidence(brief.confidence_notes));
  parts.push(renderYesterdayAudit(brief.yesterday_audit));

  parts.push(
    "-\n" +
      "This brief is research context, NOT trade instructions. The " +
      "probabilities reflect the model's calibrated view; the decision " +
      "to act on any of it remains entirely with the reader.",
  );

  const body = parts.join("\n\n");
  assertNoTipsterLanguage(body);
  return body;
}

function renderTldr(brief: BriefDocument): string {
  const avoid = brief.avoid_list ?? [];
  const allBasket = avoid.some((a) => a.symbol === "ALL_BASKET");
  const watch = brief.top_watchlist ?? [];
  let verdict: string;
  if (allBasket) {
    verdict =
      "Today the model finds NO instrument across the basket with " +
      "actionable conviction. The structural regime is set to " +
      "stand-aside: probabilities of any monitored level being " +
      "tested same-session remain low.";
  } else if (watch.length > 0) {
    verdict =
      `The model flags ${watch.length} instrument(s) with non-trivial ` +
      `probability of testing a structural level today. Read the ` +
      `watchlist before deciding whether the context matches your ` +
      `own thesis.`;
  } else {
    verdict =
      "Brief generated - see sections below for the structural read.";
  }
  return `TLDR - ${verdict}`;
}

function renderSectorRegime(s: SectorRegime): string {
  const bits: string[] = [];
  if (s.trending_up.length)
    bits.push(`trending up: ${s.trending_up.join(", ")}`);
  if (s.trending_down.length)
    bits.push(`trending down: ${s.trending_down.join(", ")}`);
  if (s.chopping.length) bits.push(`chopping: ${s.chopping.join(", ")}`);
  if (s.neutral.length) bits.push(`neutral: ${s.neutral.join(", ")}`);
  const body = bits.length ? bits.join("; ") : "no sector data available";
  const leadership = s.leadership_change_vs_yesterday.length
    ? ` Leadership change vs yesterday: ${s.leadership_change_vs_yesterday.join(", ")}.`
    : "";
  return `SECTOR REGIME - ${body}.${leadership}`;
}

function renderWatchlist(entries: WatchlistEntry[]): string {
  if (!entries.length) {
    return (
      "WATCHLIST - empty. No instruments cleared the touch-watch " +
      "threshold today."
    );
  }
  const lines = ["WATCHLIST - instruments the model flags as in-play today:"];
  for (const e of entries) {
    const note = e.avoidance_note ? `  [note: ${e.avoidance_note}]` : "";
    lines.push(
      `  - ${e.symbol} (${e.sector}) ${e.side} bias toward ` +
        `${e.key_level_type} at ${e.key_level.toFixed(2)}. ` +
        `P(test today) = ${(e.p_touch_today * 100).toFixed(0)}%; ` +
        `P(test within 60min) = ${(e.p_touch_within_60min * 100).toFixed(0)}%. ` +
        `Confidence: ${e.model_confidence_bucket}.${note}`,
    );
  }
  return lines.join("\n");
}

function renderAvoidList(entries: AvoidEntry[]): string {
  if (!entries.length) {
    return "AVOID - no specific stand-aside calls today.";
  }
  const lines = [
    "AVOID - the model recommends standing aside in these contexts:",
  ];
  for (const e of entries) {
    lines.push(
      `  - ${e.symbol}: ${e.reason} (confidence: ${e.model_confidence})`,
    );
  }
  return lines.join("\n");
}

function renderConfidence(c: ConfidenceNotes): string {
  const parts: string[] = [];
  if (c.calibrated_today.length)
    parts.push(
      `calibrated today: ${c.calibrated_today
        .map(publicPredictionLabel)
        .join(", ")}`,
    );
  if (c.drifting_today.length)
    parts.push(
      `drifting today: ${c.drifting_today
        .map(publicPredictionLabel)
        .join(", ")}`,
    );
  if (!parts.length) parts.push("no model-health data available");
  return (
    `CONFIDENCE - ${parts.join("; ")}. ` +
    `Overall brief confidence: ${c.overall_brief_confidence}.`
  );
}

function renderPendingStub(
  sectionName: string,
  stub: Record<string, unknown>,
): string {
  const reason = String(stub._reason ?? "data unavailable");
  return `${sectionName.toUpperCase()} - pending: ${reason}.`;
}

function renderYesterdayAudit(audit: YesterdayAudit): string {
  if (isYesterdayAuditPending(audit)) {
    return renderPendingStub("Yesterday audit", audit);
  }
  const lines: string[] = ["YESTERDAY AUDIT -"];
  if (audit.is_retrospective_calibration) {
    const share = audit.retrospective_share;
    lines.push(
      `  Note: calibration estimated on retrospective replay ` +
        `(${(share * 100).toFixed(0)}% of resolved predictions were ` +
        `backfilled from historical bundles, not collected live). ` +
        `Treat the numbers below as a directional read, not a live ` +
        `track record.`,
    );
  }
  lines.push(
    `  brief=${audit.yesterday_brief_id} ` +
      `predictions_made=${audit.predictions_made} ` +
      `resolved=${audit.predictions_resolved}`,
  );
  if (!audit.hit_rate_by_confidence_bucket.length) {
    lines.push("  no per-bucket calibration available (zero resolved).");
  } else {
    lines.push("  hit rate by confidence bucket:");
    for (const row of audit.hit_rate_by_confidence_bucket) {
      const ce = row.calibration_error;
      const ceSign = ce >= 0 ? "+" : "";
      lines.push(
        `    - ${publicPredictionLabel(row.prediction_type)} / ` +
          `${row.confidence_bucket}: ` +
          `n=${row.n}, hit_rate=${(row.hit_rate * 100).toFixed(0)}%, ` +
          `mean_p=${(row.mean_predicted_p * 100).toFixed(0)}%, ` +
          `calibration_error=${ceSign}${ce.toFixed(2)}`,
      );
    }
  }
  return lines.join("\n");
}
