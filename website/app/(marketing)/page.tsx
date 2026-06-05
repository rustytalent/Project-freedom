import Link from "next/link";
import { Hero } from "@/components/marketing/Hero";
import { PitchSection } from "@/components/marketing/PitchSection";
import { BriefExcerpt } from "@/components/marketing/BriefExcerpt";
import { CalibrationSparkline } from "@/components/calibration/CalibrationSparkline";
import { LinkButton } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardTitle } from "@/components/ui/card";
import { mockTimeSeries } from "@/lib/outcome-log-mock";

const NO_TIPSTER_EXCERPT = `WATCHLIST — instruments the model flags as in-play today:
  - LARGE_CAP_BANK_A (BANKING) long bias toward demand pool at
    ₹[subscriber-only]. P(test today) = 62%; P(test within 60min) = 28%.
    Confidence: high.
  - INDEX_NIFTY50 short bias toward supply pool at ₹[subscriber-only].
    P(test today) = 47%; P(test within 60min) = 14%. Confidence:
    moderate.`;

const YESTERDAY_AUDIT_EXCERPT = `YESTERDAY AUDIT —
  Note: 35% of resolved predictions were backfilled from historical
  bundles, not collected live. Treat the numbers below as a directional
  read, not a live track record.
  brief=BRIEF_2026_06_03 predictions_made=14 resolved=12
  hit rate by confidence bucket:
    - proximity / very_high: n=4, hit_rate=75%, mean_p=84%, calibration_error=+0.09
    - proximity / high:      n=5, hit_rate=80%, mean_p=72%, calibration_error=-0.08
    - proximity / moderate:  n=3, hit_rate=33%, mean_p=58%, calibration_error=+0.25`;

const CALIBRATION_EXCERPT = `Last 30 days, proximity head:
    very_high (n=68):  hit_rate=78%, mean_p=84%, error +0.06
    high      (n=152): hit_rate=71%, mean_p=72%, error +0.01
    moderate  (n=204): hit_rate=58%, mean_p=57%, error −0.01
    low       (n=311): hit_rate=29%, mean_p=32%, error +0.03`;

export default function LandingPage() {
  const series = mockTimeSeries();
  return (
    <>
      <Hero />

      <PitchSection
        eyebrow="Discipline 01"
        title="We do not say buy or sell."
        body={
          <>
            <p>
              Every research line in every brief we publish describes
              regime, probability, and context. Never an entry. Never a
              stop. Never a target.
            </p>
            <p>
              The discipline is enforced at render time. Any phrase from
              a forbidden vocabulary list — &ldquo;buy&rdquo;,
              &ldquo;sell&rdquo;, &ldquo;target at&rdquo;,
              &ldquo;stop loss&rdquo;, and others — raises in our build
              pipeline before the brief reaches your inbox.
            </p>
            <p>
              The decision to act on any of our research remains
              entirely with you, the reader.{" "}
              <Link
                href="/philosophy"
                className="text-accent-glow underline underline-offset-4"
              >
                Read why this matters →
              </Link>
            </p>
          </>
        }
        artifact={
          <BriefExcerpt
            label="watchlist section, redacted"
            body={NO_TIPSTER_EXCERPT}
          />
        }
      />

      <PitchSection
        eyebrow="Discipline 02"
        title="Every brief is followed by an audit."
        reverse
        body={
          <>
            <p>
              Tomorrow&rsquo;s brief opens by auditing today&rsquo;s.
              Per-confidence-bucket hit rate. Calibration error per
              prediction type. Drift flags surfaced before they
              degrade.
            </p>
            <p>
              When the audit is drawn from a retrospective replay of
              historical bundles rather than live-collected outcomes,
              we say so explicitly — and our renderer prepends a
              disclosure line so you never confuse the two.
            </p>
            <p>
              No other research service in this space audits itself
              publicly.{" "}
              <Link
                href="/yesterday-audit"
                className="text-accent-glow underline underline-offset-4"
              >
                See how the audit works →
              </Link>
            </p>
          </>
        }
        artifact={
          <BriefExcerpt
            label="yesterday audit, real shape"
            body={YESTERDAY_AUDIT_EXCERPT}
          />
        }
      />

      <PitchSection
        eyebrow="Discipline 03"
        title="Calibration is the product."
        body={
          <>
            <p>
              A 70% probability we publish should resolve in our favour
              roughly 70 times out of 100. If it doesn&rsquo;t, the gap
              shows up on the dashboard as calibration error — and we
              flag the head as drifting until it returns to within
              tolerance.
            </p>
            <p>
              The whole engine is built around this contract. Every
              probability you read was calibrated against a hold-out
              window the model never saw during fitting.
            </p>
            <p>
              <Link
                href="/track-record"
                className="text-accent-glow underline underline-offset-4"
              >
                Live calibration dashboard →
              </Link>
            </p>
          </>
        }
        artifact={
          <div className="space-y-4">
            <Card>
              <CardTitle>Proximity head — calibration error</CardTitle>
              <CardDescription>
                Daily mean predicted probability minus actual hit rate, last
                90 trading days. Closer to zero is better.
              </CardDescription>
              <CardContent className="mt-4">
                <CalibrationSparkline data={series} showAxes height={180} />
              </CardContent>
            </Card>
            <BriefExcerpt
              label="per-bucket calibration, last 30d"
              body={CALIBRATION_EXCERPT}
            />
          </div>
        }
      />

      {/* Pricing teaser */}
      <section className="border-t border-border">
        <div className="max-w-dash mx-auto px-6 py-20">
          <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
            Pricing
          </p>
          <h2 className="font-serif text-3xl md:text-4xl leading-tight text-fg max-w-2xl">
            Sober pricing for sober readers.
          </h2>
          <p className="mt-4 text-fg-muted max-w-xl leading-relaxed">
            No freemium games, no urgency promos. Three tiers, monthly
            or annual. First seven days of the archive are free to
            browse after you create an account; after that a paid tier
            is required.
          </p>
          <div className="grid gap-4 md:grid-cols-3 mt-10">
            <Card>
              <CardTitle>Intraday</CardTitle>
              <CardDescription>
                Daily Brief, every NSE trading day before open.
              </CardDescription>
              <CardContent className="mt-6 font-mono text-2xl text-fg tabnum">
                ₹—
              </CardContent>
              <CardContent className="text-xs text-fg-subtle mt-1">
                /month (pricing finalising)
              </CardContent>
            </Card>
            <Card>
              <CardTitle>Multi-product</CardTitle>
              <CardDescription>
                Daily Brief + Swing Brief. Monday pre-open + ad-hoc on
                regime flips.
              </CardDescription>
              <CardContent className="mt-6 font-mono text-2xl text-fg tabnum">
                ₹—
              </CardContent>
              <CardContent className="text-xs text-fg-subtle mt-1">
                /month (pricing finalising)
              </CardContent>
            </Card>
            <Card>
              <CardTitle>Diagnosis</CardTitle>
              <CardDescription>
                One-off audit of your own strategy. Five trading days
                turnaround.
              </CardDescription>
              <CardContent className="mt-6 font-mono text-2xl text-fg tabnum">
                ₹—
              </CardContent>
              <CardContent className="text-xs text-fg-subtle mt-1">
                per audit (pricing finalising)
              </CardContent>
            </Card>
          </div>
          <div className="mt-10">
            <LinkButton href="/pricing" variant="secondary">
              Full pricing →
            </LinkButton>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="border-t border-border bg-bg-raised">
        <div className="max-w-prose mx-auto px-6 py-20 text-center">
          <h2 className="font-serif text-3xl md:text-4xl leading-tight text-fg">
            Read one brief before deciding.
          </h2>
          <p className="mt-4 text-fg-muted leading-relaxed">
            The sample brief is a real published artifact, redacted for
            public view. It will take six minutes to read.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <LinkButton href="/sample-brief" variant="primary">
              View sample brief
            </LinkButton>
            <LinkButton href="/contact" variant="secondary">
              Talk to us
            </LinkButton>
          </div>
        </div>
      </section>
    </>
  );
}
