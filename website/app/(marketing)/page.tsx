import Link from "next/link";
import { Hero } from "@/components/marketing/Hero";
import { PitchSection } from "@/components/marketing/PitchSection";
import { BriefExcerpt } from "@/components/marketing/BriefExcerpt";
import { CalibrationLedger } from "@/components/marketing/CalibrationLedger";
import { NumbersStrip } from "@/components/marketing/NumbersStrip";
import { BuiltFor } from "@/components/marketing/BuiltFor";
import { Principles } from "@/components/marketing/Principles";
import { ProcessTimeline } from "@/components/marketing/ProcessTimeline";
import { PullQuote } from "@/components/marketing/PullQuote";
import { SectionStamp } from "@/components/marketing/SectionStamp";
import { FAQ } from "@/components/marketing/FAQ";
import { CalibrationSparkline } from "@/components/calibration/CalibrationSparkline";
import { LinkButton } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardTitle } from "@/components/ui/card";
import { mockTimeSeries } from "@/lib/outcome-log-mock";
import { plans } from "@/lib/pricing";
import { Reveal } from "@/components/ui/Reveal";

// Public-facing brief excerpts. The redaction-token treatment in the
// first excerpt makes the subscriber-only material visible AS REDACTED
// rather than hidden - the visitor sees the shape of what they get.
const RESEARCH_EXCERPT = `WATCHLIST
  - BANKING basket: ₹[subscriber-only] level cluster is near enough to
    monitor. Touch probability is 62%; reaction quality is below the
    paid-action threshold at ₹[subscriber-only].
  - IT basket: directional read is mixed. Stand aside until the next
    update confirms whether the move is developing or fading.

AVOIDANCE
  - METAL basket: regime is rotating; tip-of-spear names show drift.
    Avoid the entire bucket above ₹[subscriber-only] today.`;

const CALIBRATION_EXCERPT = `Last 30 days, touch-watch head:
    very_high (n=68):  hit_rate=78%, mean_p=84%, error +0.06
    high      (n=152): hit_rate=71%, mean_p=72%, error +0.01
    moderate  (n=204): hit_rate=58%, mean_p=57%, error −0.01
    low       (n=311): hit_rate=29%, mean_p=32%, error +0.03`;

export default function LandingPage() {
  const series = mockTimeSeries();
  return (
    <>
      <Hero />

      <SectionStamp stamp="S/01" label="The record" />
      <NumbersStrip />

      <PitchSection
        stamp="S/02"
        stampLabel="Discipline · research"
        eyebrow="Discipline 01"
        title={
          <>
            A brief, not a{" "}
            <span className="text-warm">trade call</span>.
          </>
        }
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
            label="watchlist + avoidance, redacted"
            body={RESEARCH_EXCERPT}
            redactTokens={["₹[subscriber-only]"]}
          />
        }
      />

      <PullQuote
        quote="We grade ourselves out loud, every morning before the bell."
        attribution="Research desk principles · §3"
      />

      <PitchSection
        stamp="S/03"
        stampLabel="Discipline · audit"
        stampTone="warm"
        eyebrow="Discipline 02"
        title={
          <>
            Every brief is followed by an{" "}
            <span className="text-warm">audit</span>.
          </>
        }
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
              No other research desk in this space audits itself
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
          <CalibrationLedger label="6 of 12 resolved, head + bucket + error" />
        }
      />

      <PitchSection
        stamp="S/04"
        stampLabel="Discipline · calibration"
        eyebrow="Discipline 03"
        title={
          <>
            Calibration{" "}
            <span className="text-warm">is</span>
            {" "}the product.
          </>
        }
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
              window the engine never saw during fitting.
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
                Daily mean predicted probability minus actual hit rate,
                last 90 trading days. Closer to zero is better.
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

      <SectionStamp stamp="S/05" label="The engine room" />
      <ProcessTimeline />

      <SectionStamp stamp="S/06" label="Built for" />
      <BuiltFor />

      <PullQuote
        quote="The reader makes the trade. The brief makes the case."
        attribution="Research desk principles · §1"
      />

      <SectionStamp stamp="S/07" label="How we publish" tone="warm" />
      <Principles />

      <SectionStamp stamp="S/08" label="Pricing" />
      <section>
        <div className="max-w-dash mx-auto px-6 py-24 md:py-28">
          <Reveal>
            <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
              Pricing
            </p>
            <h2 className="font-serif text-3xl md:text-4xl leading-tight text-fg max-w-2xl">
              Pricing for a{" "}
              <span className="text-warm">research product</span>, not a
              tip sheet.
            </h2>
            <p className="mt-5 text-fg-muted max-w-xl leading-relaxed">
              Google sign-in starts a five-day preview automatically.
              Paid plans run through Razorpay and keep the exact
              research archive inside the subscriber portal.
            </p>
          </Reveal>
          <div className="grid gap-4 md:grid-cols-3 mt-12">
            {(["daily", "pro", "diagnosis"] as const).map((id, i) => {
              const p = plans[id];
              const isDiag = id === "diagnosis";
              return (
                <Reveal key={id} delay={i * 110}>
                  <Card>
                    <CardTitle>{p.name}</CardTitle>
                    <CardDescription>{p.description}</CardDescription>
                    <CardContent className="mt-6 font-mono text-2xl text-fg tabnum">
                      {p.displayMonthly}
                    </CardContent>
                    <CardContent className="text-xs text-fg-subtle mt-1">
                      {isDiag ? "base + scope" : "per month"}
                    </CardContent>
                  </Card>
                </Reveal>
              );
            })}
          </div>
          <div className="mt-12">
            <LinkButton href="/pricing" variant="secondary">
              Full pricing
            </LinkButton>
          </div>
        </div>
      </section>

      <SectionStamp stamp="S/09" label="Common questions" />
      <FAQ />

      <SectionStamp stamp="S/10" label="Decide" tone="warm" />
      {/* Final CTA */}
      <section className="bg-bg-raised relative overflow-hidden">
        <div className="absolute inset-0 hero-glow pointer-events-none opacity-70" aria-hidden="true" />
        <div className="relative max-w-prose mx-auto px-6 py-28 md:py-32 text-center">
          <Reveal>
            <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-warm-glow mb-6">
              Six minutes · no signup · no card
            </p>
            <h2 className="font-serif text-3xl md:text-5xl leading-[1.05] text-fg tracking-[-0.01em]">
              Read one brief
              <span className="text-warm"> before deciding</span>.
            </h2>
            <p className="mt-6 text-fg-muted leading-relaxed">
              The sample brief is a real published artifact, redacted
              for public view. No follow-up sequence, no aggressive
              retargeting. Read it, close the tab, come back when you
              want to.
            </p>
            <div className="mt-10 flex flex-wrap justify-center gap-3">
              <LinkButton href="/sample-brief" variant="primary">
                View sample brief
              </LinkButton>
              <LinkButton href="/contact" variant="secondary">
                Talk to the founder
              </LinkButton>
            </div>
          </Reveal>
        </div>
      </section>
    </>
  );
}
