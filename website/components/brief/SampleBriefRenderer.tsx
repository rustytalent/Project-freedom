import {
  BriefDocument,
  isYesterdayAuditPending,
  WatchlistEntry,
} from "@/lib/brief-render";

// Public-view renderer: the same brief shape, but specific levels
// rendered as ₹[subscriber-only] so prospects see the shape, not
// the actionable content. Symbols stay anonymised in the source
// JSON for sample briefs.

export function SampleBriefRenderer({ brief }: { brief: BriefDocument }) {
  return (
    <div className="space-y-10">
      <BriefHeader brief={brief} />
      <Section title="TLDR">
        <TLDRBody brief={brief} />
      </Section>
      <PendingSection
        title="Index regime"
        stubs={brief.index_regime as Record<string, { _status?: string; _reason?: string }>}
      />
      <OptionsSection options={brief.options_suitability} />
      <Section title="Sector regime">
        <SectorRegimeBody brief={brief} />
      </Section>
      <Section title="Watchlist">
        <WatchlistBody entries={brief.top_watchlist} />
      </Section>
      <Section title="Avoid">
        <AvoidBody entries={brief.avoid_list} />
      </Section>
      <Section title="Confidence">
        <ConfidenceBody brief={brief} />
      </Section>
      <Section title="Yesterday audit">
        <YesterdayAuditBody brief={brief} />
      </Section>
      <footer className="text-xs text-fg-subtle border-t border-border pt-6 leading-relaxed">
        This brief is research context, not trade instructions. The
        probabilities reflect the model&rsquo;s calibrated view; the
        decision to act on any of it remains entirely with the reader.
      </footer>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <h3 className="text-xs uppercase tracking-[0.18em] text-accent mb-3">
        {title}
      </h3>
      <div className="text-fg leading-relaxed">{children}</div>
    </section>
  );
}

function BriefHeader({ brief }: { brief: BriefDocument }) {
  const md = brief.brief_metadata;
  return (
    <header className="border-b border-border pb-6">
      <h2 className="font-serif text-2xl text-fg">
        Daily Research Brief - {md.trading_date_ist}
      </h2>
      <p className="text-xs text-fg-subtle mt-2 font-mono">
        bundle={md.model_bundle_version} · indexes covered:{" "}
        {md.indexes_covered.join(", ") || "(none)"} · est. read ~
        {md.reading_time_minutes} min
      </p>
    </header>
  );
}

function TLDRBody({ brief }: { brief: BriefDocument }) {
  const allBasket = brief.avoid_list.some((a) => a.symbol === "ALL_BASKET");
  const watchN = brief.top_watchlist.length;
  if (allBasket) {
    return (
      <p>
        Today the model finds no instrument across the basket with
        actionable conviction. The structural regime is set to
        stand-aside.
      </p>
    );
  }
  if (watchN > 0) {
    return (
      <p>
        The model flags {watchN} instrument(s) with non-trivial
        probability of testing a structural level today. Read the
        watchlist before deciding whether the context matches your own
        thesis.
      </p>
    );
  }
  return <p>Brief generated. See sections below.</p>;
}

function PendingSection({
  title,
  stubs,
}: {
  title: string;
  stubs: Record<string, { _status?: string; _reason?: string }>;
}) {
  return (
    <Section title={title}>
      <ul className="space-y-2 text-fg-muted text-sm">
        {Object.entries(stubs).map(([k, v]) => (
          <li key={k}>
            <span className="font-mono text-fg">{k}</span> - pending:{" "}
            {v._reason ?? "data unavailable"}.
          </li>
        ))}
      </ul>
    </Section>
  );
}

function OptionsSection({ options }: { options: Record<string, unknown> }) {
  return (
    <Section title="Options suitability">
      <div className="space-y-6">
        {Object.entries(options).map(([idx, payload]) => {
          if (
            payload &&
            typeof payload === "object" &&
            "_status" in payload &&
            (payload as { _status?: string })._status === "pending"
          ) {
            return (
              <p key={idx} className="text-sm text-fg-muted">
                <span className="font-mono text-fg">{idx}</span> -
                pending:{" "}
                {(payload as { _reason?: string })._reason ??
                  "data unavailable"}
                .
              </p>
            );
          }
          const p = payload as {
            directional_bias?: string;
            expected_range_today_atr?: number;
            theta_danger_score?: number;
            regime_for_premium_buyers?: string;
            regime_for_premium_sellers?: string;
            strike_levels_in_play?: Array<{
              strike: number;
              p_test_today: number;
              p_test_within_60min: number;
              key_level_type: string;
            }>;
          };
          return (
            <div key={idx} className="space-y-2">
              <p className="text-sm">
                <span className="font-mono text-fg">{idx}</span> -
                directional bias: <em>{p.directional_bias}</em>;
                expected range today:{" "}
                <strong className="text-fg">
                  {p.expected_range_today_atr?.toFixed(2)} ATR
                </strong>
                ; theta danger:{" "}
                <strong className="text-fg">
                  {((p.theta_danger_score ?? 0) * 100).toFixed(0)}%
                </strong>
                .
              </p>
              <p className="text-sm text-fg-muted">
                regime for premium buyers:{" "}
                <em>{p.regime_for_premium_buyers}</em>; for premium
                sellers: <em>{p.regime_for_premium_sellers}</em>.
              </p>
              {p.strike_levels_in_play && p.strike_levels_in_play.length > 0 && (
                <ul className="font-mono text-xs space-y-1 mt-2 text-fg-muted tabnum">
                  {p.strike_levels_in_play.map((s, i) => (
                    <li key={i}>
                      strike ₹[subscriber-only] ({s.key_level_type})
                      &nbsp;p_test_today=
                      <span className="text-fg">
                        {(s.p_test_today * 100).toFixed(0)}%
                      </span>{" "}
                      p_60min=
                      <span className="text-fg">
                        {(s.p_test_within_60min * 100).toFixed(0)}%
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </div>
    </Section>
  );
}

function SectorRegimeBody({ brief }: { brief: BriefDocument }) {
  const s = brief.sector_regime;
  return (
    <ul className="text-sm space-y-1 text-fg-muted">
      {s.trending_up.length > 0 && (
        <li>
          <span className="text-calibrated">trending up:</span>{" "}
          {s.trending_up.join(", ")}
        </li>
      )}
      {s.trending_down.length > 0 && (
        <li>
          <span className="text-drift">trending down:</span>{" "}
          {s.trending_down.join(", ")}
        </li>
      )}
      {s.chopping.length > 0 && (
        <li>chopping: {s.chopping.join(", ")}</li>
      )}
      {s.neutral.length > 0 && <li>neutral: {s.neutral.join(", ")}</li>}
      {s.leadership_change_vs_yesterday.length > 0 && (
        <li className="text-fg pt-2">
          Leadership change vs yesterday:{" "}
          {s.leadership_change_vs_yesterday.join(", ")}.
        </li>
      )}
    </ul>
  );
}

function WatchlistBody({ entries }: { entries: WatchlistEntry[] }) {
  if (!entries.length) {
    return (
      <p className="text-fg-muted text-sm">
        Empty. No instruments cleared the proximity threshold today.
      </p>
    );
  }
  return (
    <ul className="space-y-4 text-sm">
      {entries.map((e, i) => (
        <li
          key={i}
          className="border-l border-border pl-4 leading-relaxed"
        >
          <p className="text-fg">
            <span className="font-mono">{e.symbol}</span>{" "}
            <span className="text-fg-subtle">({e.sector})</span>{" "}
            <span className="text-fg-muted">{e.side} bias toward</span>{" "}
            <span className="text-accent-glow">{e.key_level_type}</span>{" "}
            at <span className="text-fg-subtle">₹[subscriber-only]</span>
          </p>
          <p className="text-fg-muted tabnum mt-1">
            P(test today) ={" "}
            <span className="text-fg">
              {(e.p_touch_today * 100).toFixed(0)}%
            </span>
            ; P(test within 60min) ={" "}
            <span className="text-fg">
              {(e.p_touch_within_60min * 100).toFixed(0)}%
            </span>
            . Confidence: <em>{e.model_confidence_bucket}</em>.
          </p>
          {e.avoidance_note && (
            <p className="text-drift mt-1 text-xs">
              note: {e.avoidance_note}
            </p>
          )}
        </li>
      ))}
    </ul>
  );
}

function AvoidBody({
  entries,
}: {
  entries: BriefDocument["avoid_list"];
}) {
  if (!entries.length) {
    return (
      <p className="text-fg-muted text-sm">
        No specific stand-aside calls today.
      </p>
    );
  }
  return (
    <ul className="space-y-2 text-sm">
      {entries.map((a, i) => (
        <li key={i} className="text-fg-muted">
          <span className="font-mono text-fg">{a.symbol}</span>: {a.reason}{" "}
          <span className="text-fg-subtle">
            (confidence: {a.model_confidence})
          </span>
        </li>
      ))}
    </ul>
  );
}

function ConfidenceBody({ brief }: { brief: BriefDocument }) {
  const c = brief.confidence_notes;
  return (
    <div className="text-sm text-fg-muted space-y-1">
      {c.calibrated_today.length > 0 && (
        <p>
          <span className="text-calibrated">calibrated today:</span>{" "}
          {c.calibrated_today.join(", ")}
        </p>
      )}
      {c.drifting_today.length > 0 && (
        <p>
          <span className="text-drift">drifting today:</span>{" "}
          {c.drifting_today.join(", ")}
        </p>
      )}
      <p className="text-fg pt-2">
        Overall brief confidence:{" "}
        <em>{c.overall_brief_confidence}</em>.
      </p>
    </div>
  );
}

function YesterdayAuditBody({ brief }: { brief: BriefDocument }) {
  const a = brief.yesterday_audit;
  if (isYesterdayAuditPending(a)) {
    return (
      <p className="text-fg-muted text-sm">
        Pending: {a._reason}.
      </p>
    );
  }
  return (
    <div className="text-sm text-fg-muted space-y-3">
      {a.is_retrospective_calibration && (
        <p className="border border-drift/40 bg-drift/5 px-4 py-3 rounded-sm text-fg leading-relaxed">
          <strong>Note:</strong> calibration estimated on retrospective
          replay ({(a.retrospective_share * 100).toFixed(0)}% of
          resolved predictions were backfilled from historical bundles,
          not collected live). Treat the numbers below as a directional
          read, not a live track record.
        </p>
      )}
      <p className="text-fg-muted font-mono text-xs">
        brief={a.yesterday_brief_id} predictions_made=
        <span className="text-fg">{a.predictions_made}</span> resolved=
        <span className="text-fg">{a.predictions_resolved}</span>
      </p>
      <div>
        <p className="text-xs uppercase tracking-wider text-fg-subtle mb-2">
          Hit rate by confidence bucket
        </p>
        <table className="w-full font-mono text-xs tabnum">
          <thead>
            <tr className="text-fg-subtle border-b border-border">
              <th className="text-left py-2 pr-4 font-normal">Type</th>
              <th className="text-left py-2 pr-4 font-normal">Bucket</th>
              <th className="text-right py-2 pr-4 font-normal">n</th>
              <th className="text-right py-2 pr-4 font-normal">Hit</th>
              <th className="text-right py-2 pr-4 font-normal">Mean p</th>
              <th className="text-right py-2 font-normal">Error</th>
            </tr>
          </thead>
          <tbody>
            {a.hit_rate_by_confidence_bucket.map((row, i) => (
              <tr key={i} className="border-b border-border/50">
                <td className="py-2 pr-4 text-fg-muted">
                  {row.prediction_type}
                </td>
                <td className="py-2 pr-4 text-fg-muted">
                  {row.confidence_bucket}
                </td>
                <td className="py-2 pr-4 text-right text-fg-muted">
                  {row.n}
                </td>
                <td className="py-2 pr-4 text-right text-fg">
                  {(row.hit_rate * 100).toFixed(0)}%
                </td>
                <td className="py-2 pr-4 text-right text-fg-muted">
                  {(row.mean_predicted_p * 100).toFixed(0)}%
                </td>
                <td
                  className={`py-2 text-right ${
                    Math.abs(row.calibration_error) > 0.08
                      ? "text-drift"
                      : "text-calibrated"
                  }`}
                >
                  {row.calibration_error >= 0 ? "+" : ""}
                  {row.calibration_error.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
