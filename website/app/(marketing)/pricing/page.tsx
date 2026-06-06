import type { Metadata } from "next";
import { Card, CardContent, CardDescription, CardTitle } from "@/components/ui/card";
import { LinkButton } from "@/components/ui/button";
import { plans } from "@/lib/pricing";

export const metadata: Metadata = {
  title: "Pricing",
  description:
    "Core Research, Live Desk, and audit pricing for Indian market research.",
};

const tiers = [
  {
    name: plans.daily.name,
    monthly: plans.daily.displayMonthly,
    annual: plans.daily.displayAnnual,
    desc: plans.daily.description,
    recommended: true,
    includes: [
      "Equity, options, and index research",
      "Daily Brief by 08:30 IST on NSE trading days",
      "Subscriber-only portal archive",
      "Yesterday Audit inside every brief",
      "Email plus portal delivery",
      "No live intraday update stream",
    ],
    cta: "Subscribe with Razorpay",
    href: "/checkout?plan=daily&cycle=monthly",
  },
  {
    name: plans.pro.name,
    monthly: plans.pro.displayMonthly,
    annual: plans.pro.displayAnnual,
    desc: plans.pro.description,
    includes: [
      "Everything in Core Research",
      "Live update stream when the desk is active",
      "Equity, options, and index coverage",
      "Full brief archive",
      "Calibration and outcome dashboard",
      "Priority delivery support",
    ],
    cta: "Subscribe with Razorpay",
    href: "/checkout?plan=pro&cycle=monthly",
  },
  {
    name: plans.diagnosis.name,
    monthly: plans.diagnosis.displayMonthly,
    annual: plans.diagnosis.displayAnnual,
    desc: plans.diagnosis.description,
    includes: [
      "₹2,000 base review",
      "Scope expands by infrastructure and data size",
      "Structured PDF report",
      "Data processed for the audit only",
      "No strategy data retained after delivery unless requested",
    ],
    cta: "Request a diagnosis",
    href: "/contact",
  },
];

export default function PricingPage() {
  return (
    <div className="max-w-dash mx-auto px-6 py-20">
      <header className="max-w-prose mb-16">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
          Pricing
        </p>
        <h1 className="font-serif text-4xl md:text-5xl leading-tight text-fg">
          Research pricing that matches the current evidence.
        </h1>
        <p className="mt-6 text-fg-muted leading-relaxed">
          The current product is strongest as a daily research brief,
          avoidance layer, and calibration record. Pricing reflects
          that. It does not price the service as a trade-call desk.
        </p>
        <p className="mt-3 text-xs text-fg-subtle">
          Annual subscriptions get roughly two months free. Payments
          are processed through Razorpay. Google sign-in is the
          production account path.
        </p>
      </header>

      <section className="mb-10 rounded-lg border border-border bg-bg-raised p-6">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-3">
          Login preview
        </p>
        <h2 className="font-serif text-2xl text-fg">
          Five days start automatically after Google sign-in.
        </h2>
        <p className="mt-3 text-sm text-fg-muted leading-relaxed max-w-3xl">
          There is no public zero-price plan card. When a reader signs
          in, their preview clock starts for five days with redacted
          access to selected artifacts. The timer is account-based and
          does not require a Razorpay checkout.
        </p>
      </section>

      <div className="grid gap-6 md:grid-cols-3">
        {tiers.map((t) => (
          <Card
            key={t.name}
            className={t.recommended ? "border-accent" : undefined}
          >
            {t.recommended && (
              <p className="text-xs uppercase tracking-wider text-accent mb-3">
                Recommended
              </p>
            )}
            <CardTitle>{t.name}</CardTitle>
            <CardDescription>{t.desc}</CardDescription>
            <CardContent className="mt-6">
              <p className="font-mono text-2xl text-fg tabnum">
                {t.monthly}
              </p>
              <p className="text-xs text-fg-subtle mt-1 tabnum">
                Annual: {t.annual}
              </p>
            </CardContent>
            <CardContent className="mt-6">
              <ul className="space-y-2 text-sm text-fg-muted">
                {t.includes.map((line) => (
                  <li key={line} className="flex gap-2">
                    <span className="text-accent">+</span>
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
            <CardContent className="mt-8">
              <LinkButton
                href={t.href}
                variant={t.recommended ? "primary" : "secondary"}
              >
                {t.cta}
              </LinkButton>
            </CardContent>
          </Card>
        ))}
      </div>

      <section className="mt-20 pt-12 border-t border-border max-w-prose">
        <h2 className="font-serif text-2xl text-fg mb-4">
          Why no permanent free tier?
        </h2>
        <p className="text-fg-muted leading-relaxed">
          Exact levels, archive access, and private brief delivery are
          the paid product. The preview is intentionally limited so
          public pages cannot be reverse engineered into the subscriber
          experience. It begins on first login and expires after five
          days.
        </p>
        <h2 className="font-serif text-2xl text-fg mt-12 mb-4">
          Why no performance-fee tier?
        </h2>
        <p className="text-fg-muted leading-relaxed">
          A performance fee aligns the publisher with how aggressively
          the reader trades, not with the quality of the research. We
          want our incentives anchored to the research, so we charge a
          flat subscription.
        </p>
        <h2 className="font-serif text-2xl text-fg mt-12 mb-4">
          Cancellation
        </h2>
        <p className="text-fg-muted leading-relaxed">
          Monthly subscriptions can be cancelled any time from your
          portal. Annual subscriptions are refundable pro rata in the
          first fourteen days after billing.
        </p>
      </section>
    </div>
  );
}
