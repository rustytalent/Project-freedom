import type { Metadata } from "next";
import { BriefExcerpt } from "@/components/marketing/BriefExcerpt";
import { LinkButton } from "@/components/ui/button";

export const metadata: Metadata = {
  title: "Yesterday Audit",
  description:
    "Why every brief is followed by an audit, and how we disclose retrospective replay vs live history.",
};

const RETROSPECTIVE_AUDIT_EXCERPT = `YESTERDAY AUDIT -
  Note: calibration estimated on retrospective replay (62% of resolved
  predictions were backfilled from historical bundles, not collected
  live). Treat the numbers below as a directional read, not a live
  track record.
  brief=BRIEF_2026_05_28 predictions_made=11 resolved=9
  hit rate by confidence bucket:
    - proximity / high:     n=4, hit_rate=75%, mean_p=71%, calibration_error=-0.04
    - proximity / moderate: n=3, hit_rate=33%, mean_p=58%, calibration_error=+0.25
    - avoidance / moderate: n=2, hit_rate=100%, mean_p=100%, calibration_error=0.00`;

const LIVE_AUDIT_EXCERPT = `YESTERDAY AUDIT -
  brief=BRIEF_2026_06_04 predictions_made=12 resolved=11
  hit rate by confidence bucket:
    - proximity / very_high: n=2, hit_rate=100%, mean_p=88%, calibration_error=-0.12
    - proximity / high:      n=5, hit_rate=80%, mean_p=72%, calibration_error=-0.08
    - proximity / moderate:  n=3, hit_rate=33%, mean_p=58%, calibration_error=+0.25
    - avoidance / high:      n=1, hit_rate=100%, mean_p=100%, calibration_error=0.00`;

export default function YesterdayAuditPage() {
  return (
    <article className="max-w-prose mx-auto px-6 py-20">
      <header className="mb-12">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
          Feature - Yesterday Audit
        </p>
        <h1 className="font-serif text-4xl md:text-5xl leading-tight text-fg">
          Every brief is followed by an audit of the previous brief.
        </h1>
        <p className="mt-6 text-fg-muted leading-relaxed">
          Most research products in this space don&rsquo;t look back.
          When yesterday&rsquo;s call was wrong, it quietly disappears
          and tomorrow&rsquo;s gets fresh attention. The reader has no
          structured way to know whether the service has been right or
          wrong over time.
        </p>
        <p className="mt-3 text-fg-muted leading-relaxed">
          We do the opposite. Every brief opens with a Yesterday
          Audit: per-bucket hit rate, calibration error per
          prediction type, drift flags, retrospective-vs-live
          disclosure.
        </p>
      </header>

      <section className="mt-12 pt-12 border-t border-border">
        <h2 className="font-serif text-2xl text-fg mb-6">
          What a populated audit section looks like
        </h2>
        <BriefExcerpt
          label="Yesterday Audit, live predictions"
          body={LIVE_AUDIT_EXCERPT}
        />
        <p className="mt-4 text-fg-muted text-sm leading-relaxed">
          Each row pairs the bucket&rsquo;s mean predicted probability
          against its actual hit rate. The calibration error is the
          difference. A small positive error means we were a touch
          over-confident; a small negative error means we under-priced
          the move. Errors larger than{" "}
          <span className="font-mono text-fg">|0.08|</span> trigger a
          drift flag on the public dashboard and a note in the next
          brief&rsquo;s confidence section.
        </p>
      </section>

      <section className="mt-16 pt-12 border-t border-border">
        <h2 className="font-serif text-2xl text-fg mb-6">
          When the audit window includes retrospective replay
        </h2>
        <p className="text-fg-muted leading-relaxed mb-6">
          New customers see briefs days after we&rsquo;ve started
          publishing them, which means the live-collected outcome log
          is thin at first. To fill the gap, we replay historical
          bundles through the same brief generator to produce
          retrospective predictions, then resolve them against actual
          bar data.
        </p>
        <p className="text-fg-muted leading-relaxed mb-6">
          Retrospective replay is useful as a directional check on the
          engine&rsquo;s calibration. It is <em>not</em> the same as a
          live track record - the bundle was fit on data that
          overlaps the replayed dates, which can encode hindsight in
          subtle ways.
        </p>
        <p className="text-fg-muted leading-relaxed mb-6">
          When more than half of the resolved predictions in
          yesterday&rsquo;s audit window came from retrospective
          replay, our renderer prepends an explicit disclosure
          line:
        </p>
        <BriefExcerpt
          label="Yesterday Audit, retrospective-replay disclosure"
          body={RETROSPECTIVE_AUDIT_EXCERPT}
        />
        <p className="mt-4 text-fg-muted text-sm leading-relaxed">
          As live outcomes accumulate, the retrospective share drops.
          You can see the current share on the{" "}
          <a
            href="/track-record"
            className="text-accent underline underline-offset-4"
          >
            track record page
          </a>
          , and the disclosure line will quietly disappear from your
          briefs once the live share crosses 50%.
        </p>
      </section>

      <div className="mt-16 pt-12 border-t border-border flex flex-wrap gap-3">
        <LinkButton href="/track-record" variant="primary">
          See the public dashboard
        </LinkButton>
        <LinkButton href="/sample-brief" variant="secondary">
          View a full sample brief
        </LinkButton>
      </div>
    </article>
  );
}
