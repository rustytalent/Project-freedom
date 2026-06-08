import Link from "next/link";
import { Hero } from "@/components/marketing/Hero";
import { PitchSection } from "@/components/marketing/PitchSection";
import { BriefExcerpt } from "@/components/marketing/BriefExcerpt";
import { NumbersStrip } from "@/components/marketing/NumbersStrip";
import { BuiltFor } from "@/components/marketing/BuiltFor";
import { Principles } from "@/components/marketing/Principles";
import { FAQ } from "@/components/marketing/FAQ";
import { CalibrationSparkline } from "@/components/calibration/CalibrationSparkline";
import { LinkButton } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardTitle } from "@/components/ui/card";
import { mockTimeSeries } from "@/lib/outcome-log-mock";
import { plans } from "@/lib/pricing";

const RESEARCH_EXCERPT = `WATCHLIST
  - BANKING basket: subscriber-only level cluster is near enough to
    monitor. Touch probability is moderate; reaction quality is below
    the paid-action threshold.
  - IT basket: directional read is mixed. Stand aside until the next
    update confirms whether the move is developing or fading.`;

const YESTERDAY_AUDIT_EXCERPT = `YESTERDAY AUDIT
  Note: 35% of resolved predictions were backfilled from historical
  bundles, not collected live. Treat the numbers below as a directional
  read, not a live track record.
  brief=BRIEF_2026_06_03 predictions_made=14 resolved=12
  hit rate by confidence bucket:
    - touch watch / very high: n=4, hit_rate=75%, mean_p=84%, calibration_error=+0.09
    - touch watch / high:      n=5, hit_rate=80%, mean_p=72%, calibration_error=-0.08
    - touch watch / moderate:  n=3, hit_rate=33%, mean_p=58%, calibration_error=+0.25`;

const CALIBRATION_EXCERPT = `Last 30 days, touch-watch model:
    very_high (n=68):  hit_rate=78%, mean_p=84%, error +0.06
    high      (n=152): hit_rate=71%, mean_p=72%, error +0.01
    moderate  (n=204): hit_rate=58%, mean_p=57%, error −0.01
    low       (n=311): hit_rate=29%, mean_p=32%, error +0.03`;

export default function LandingPage() {
  const series = mockTimeSeries();
  return (
    <>
      <Hero />

      <NumbersStrip />

      <PitchSection
        eyebrow="Discipline 01"
        title="A brief, not a trade call."
        body={
          <>
            <p>
              The Daily Brief gives you market structure, sector
              pressure, touch watch, reaction context, and avoidance
              flags before the session begins. It is built for a trader
              who wants better preparation, not a command to follow.
            </p>
            <p>
              Public samples redact exact levels and instruments where
              needed. Subscriber briefs carry the full view inside the
              portal, while public pages keep the product understandable
              without exposing the research machinery.
            </p>
            <p>
              The decision to act on any of our research remains
              entirely with you, the reader.{" "}
              <Link
                href="/philosophy"
                className="text-accent-glow underline underline-offset-4"
              >
                Read why this matters
              </Link>
            </p>
          </>
        }
        artifact={
          <BriefExcerpt
            label="watchlist section, redacted"
            body={RESEARCH_EXCERPT}
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
              we say so explicitly, and the brief carries a
              disclosure line so you never confuse the two.
            </p>
            <p>
              No other research service in this space audits itself
              publicly.{" "}
              <Link
                href="/yesterday-audit"
                className="text-accent-glow underline underline-offset-4"
              >
                See how the audit works
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
              shows up on the dashboard as calibration error, and we
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
                Live calibration dashboard
              </Link>
            </p>
          </>
        }
        artifact={
          <div className="space-y-4">
            <Card>
              <CardTitle>Touch watch calibration error</CardTitle>
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

      <BuiltFor />

      <Principles />

      {/* Pricing teaser */}
      <section className="border-t border-border">
        <div className="max-w-dash mx-auto px-6 py-20">
          <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
            Pricing
          </p>
          <h2 className="font-serif text-3xl md:text-4xl leading-tight text-fg max-w-2xl">
            Pricing for a research product, not a tip sheet.
          </h2>
          <p className="mt-4 text-fg-muted max-w-xl leading-relaxed">
            Google sign-in starts a five-day preview automatically.
            Paid plans run through Razorpay and keep the exact research
            archive inside the subscriber portal.
          </p>
          <div className="grid gap-4 md:grid-cols-3 mt-10">
            {(["daily", "pro", "diagnosis"] as const).map((id) => {
              const p = plans[id];
              const isDiag = id === "diagnosis";
              return (
                <Card key={id}>
                  <CardTitle>{p.name}</CardTitle>
                  <CardDescription>{p.description}</CardDescription>
                  <CardContent className="mt-6 font-mono text-2xl text-fg tabnum">
                    {p.displayMonthly}
                  </CardContent>
                  <CardContent className="text-xs text-fg-subtle mt-1">
                    {isDiag ? "base + scope" : "per month"}
                  </CardContent>
                </Card>
              );
            })}
          </div>
          <div className="mt-10">
            <LinkButton href="/pricing" variant="secondary">
              Full pricing
            </LinkButton>
          </div>
        </div>
      </section>

      <FAQ />

      {/* Final CTA */}
      <section className="border-t border-border bg-bg-raised">
        <div className="max-w-prose mx-auto px-6 py-24 text-center">
          <p className="text-xs uppercase tracking-[0.18em] text-accent mb-5">
            One brief, then decide.
          </p>
          <h2 className="font-serif text-3xl md:text-5xl leading-[1.1] text-fg">
            Read one brief before deciding.
          </h2>
          <p className="mt-5 text-fg-muted leading-relaxed">
            The sample brief is a real published artifact, redacted for
            public view. It will take six minutes to read. No card, no
            signup, no follow-up sequence.
          </p>
          <div className="mt-10 flex flex-wrap justify-center gap-3">
            <LinkButton href="/sample-brief" variant="primary">
              View sample brief
            </LinkButton>
            <LinkButton href="/contact" variant="secondary">
              Talk to the founder
            </LinkButton>
          </div>
        </div>
      </section>
    </>
  );
}
