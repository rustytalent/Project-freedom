import type { Metadata } from "next";
import { Card, CardContent, CardDescription, CardTitle } from "@/components/ui/card";
import { LinkButton } from "@/components/ui/button";

export const metadata: Metadata = {
  title: "Pricing",
  description:
    "Three tiers, monthly or annual. First 7 days of the archive free after signup; paid tier required thereafter.",
};

const tiers = [
  {
    name: "Intraday",
    monthly: "₹—",
    annual: "₹— (15% off)",
    desc:
      "Daily Brief, every NSE trading day before open. Email + portal.",
    includes: [
      "Daily Brief delivered by 08:30 IST",
      "Full portal access to today's brief",
      "Last 7 days of the archive (free for everyone after signup)",
      "Yesterday Audit in every brief",
      "Calibration-drift alerts",
    ],
    cta: "Start with intraday",
  },
  {
    name: "Multi-product",
    monthly: "₹—",
    annual: "₹— (15% off)",
    desc:
      "Daily Brief + Swing Brief. Best value for traders running both intraday and positional books.",
    recommended: true,
    includes: [
      "Everything in the Intraday tier",
      "Swing Brief, Monday pre-open + ad-hoc on regime flips",
      "Multi-day proximity calls at 5/10/20 trading-day horizons",
      "Full archive access (no 7-day limit)",
      "Personal calibration view on your portal",
    ],
    cta: "Subscribe to both",
  },
  {
    name: "Diagnosis",
    monthly: "₹— per audit",
    annual: "—",
    desc:
      "One-off audit of your own strategy. Five trading days from intake to report.",
    includes: [
      "Structured PDF report (typically 12–18 pages)",
      "Regime decomposition + drawdown attribution",
      "Persistence and leakage checks",
      "Cost-sensitivity analysis",
      "Recommendation: ship / re-scope / shelve",
    ],
    cta: "Request a diagnosis",
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
          Three tiers. No urgency. No upsell.
        </h1>
        <p className="mt-6 text-fg-muted leading-relaxed">
          Annual subscriptions get a 15% discount. The first seven
          days of the brief archive are free to browse after you
          create an account; a paid tier is required thereafter.
        </p>
        <p className="mt-3 text-xs text-fg-subtle">
          Prices are finalising. We will email you once tiers are
          published. Until then, the diagnosis product is available
          immediately on request.
        </p>
      </header>

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
                    <span className="text-accent">—</span>
                    <span>{line}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
            <CardContent className="mt-8">
              <LinkButton
                href="/contact"
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
          Why no freemium tier?
        </h2>
        <p className="text-fg-muted leading-relaxed">
          Freemium attracts traders who want signals without paying for
          them. That is not who this service is for. The 7-day free
          archive after signup is a structured preview, not an ongoing
          free product.
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
          portal. Annual subscriptions are refundable pro-rata in the
          first 14 days after billing.
        </p>
      </section>
    </div>
  );
}
