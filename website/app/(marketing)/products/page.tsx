import type { Metadata } from "next";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: "Products",
  description: "Daily Brief, Swing Brief, Diagnosis, Audit Infrastructure.",
};

const products = [
  {
    href: "/products/daily-brief",
    title: "Daily Brief",
    line: "Intraday research before NSE open, every trading day.",
    desc:
      "A structured read of the day's structural levels, sector regime, " +
      "and avoidance contexts. Delivered as email plus the portal view " +
      "by 08:30 IST.",
    audience: "For intraday traders.",
  },
  {
    href: "/products/swing-brief",
    title: "Swing Brief",
    line: "Multi-day proximity calls, delivered Monday pre-open.",
    desc:
      "A multi-day research read for slower decisions. This stream is " +
      "not generally available yet and will open only after the daily " +
      "brief archive is stable.",
    audience: "For positional traders and option buyers.",
  },
  {
    href: "/products/diagnosis",
    title: "Diagnosis",
    line: "A one-off audit of your own strategy.",
    desc:
      "Submit a trade log; we return a PDF report on regime " +
      "decomposition, drawdown attribution, persistence, leakage, and " +
      "cost sensitivity. Five trading days turnaround.",
    audience: "For serious retail and small prop traders.",
  },
  {
    href: "/products/audit-infrastructure",
    title: "Audit Infrastructure",
    line: "Our audit machinery, licensed to your shop.",
    desc:
      "Outcome logging, calibration tracking, drift review, and public " +
      "reporting patterns available as a B2B engagement for teams with " +
      "their own research stack.",
    audience: "For small quant funds and prop desks.",
  },
];

export default function ProductsPage() {
  return (
    <div className="max-w-dash mx-auto px-6 py-20">
      <header className="max-w-prose mb-16">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
          Products
        </p>
        <h1 className="font-serif text-4xl md:text-5xl leading-tight text-fg">
          One engine. Four research products.
        </h1>
        <p className="mt-6 text-fg-muted leading-relaxed">
          Each product is a different read of the same calibrated
          engine, scoped to a specific horizon, customer, and price
          point. Pick the one that matches how you trade.
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-2">
        {products.map((p) => (
          <Link
            href={p.href}
            key={p.href}
            className="group block"
            aria-label={`${p.title} - ${p.line}`}
          >
            <Card className="h-full group-hover:border-accent transition-colors">
              <CardTitle className="group-hover:text-accent-glow transition-colors">
                {p.title}
              </CardTitle>
              <CardDescription>{p.line}</CardDescription>
              <CardContent className="mt-4 leading-relaxed text-fg-muted">
                {p.desc}
              </CardContent>
              <CardContent className="mt-4 text-xs text-fg-subtle">
                {p.audience}
              </CardContent>
              <CardContent className="mt-6 text-sm text-accent">
                Learn more
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
