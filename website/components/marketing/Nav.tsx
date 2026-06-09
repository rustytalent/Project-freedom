"use client";

import { useState } from "react";
import Link from "next/link";
import { brand } from "@/lib/brand";
import { plans } from "@/lib/pricing";
import { buildDateline } from "@/lib/clock";
import { ScrollProgress } from "./ScrollProgress";
import { NavMenu, type MenuItem, type MenuFeature } from "./NavMenu";

const productItems: MenuItem[] = [
  {
    href: "/products/daily-brief",
    label: "Core Research",
    desc: `Pre-market briefs, full archive. ${plans.daily.displayMonthly}.`,
  },
  {
    href: "/products/swing-brief",
    label: "Live Desk",
    desc: `Core + intraday updates, calibration dashboard. ${plans.pro.displayMonthly}.`,
  },
  {
    href: "/products/diagnosis",
    label: "Audit Base",
    desc: `One-off research or infra audit. ${plans.diagnosis.displayMonthly}.`,
  },
  {
    href: "/products/audit-infrastructure",
    label: "Audit infrastructure",
    desc: "Calibration cron, drift flags, public dashboard — for desks.",
  },
];

const productFeature: MenuFeature = {
  eyebrow: "Compare",
  title: "Pricing for a research product.",
  body: "Side-by-side comparison of all plans, billing cycles, and what each includes.",
  href: "/pricing",
  cta: "See pricing",
};

const researchItems: MenuItem[] = [
  {
    href: "/sample-brief",
    label: "Sample brief",
    desc: "A real published brief, redacted for public view. 6-minute read.",
  },
  {
    href: "/track-record",
    label: "Track record",
    desc: "Per-bucket calibration, drift flags, 90-day series. Updated nightly.",
  },
  {
    href: "/yesterday-audit",
    label: "Yesterday audit",
    desc: "How yesterday's brief resolved against actual outcomes.",
  },
  {
    href: "/philosophy",
    label: "Philosophy",
    desc: "Why we publish probabilities and audit them in public.",
  },
];

export function Nav() {
  const [open, setOpen] = useState(false);
  const dl = buildDateline();
  const researchFeature: MenuFeature = {
    eyebrow: `Edition ${dl.edition}`,
    title: `${dl.todayLong}'s brief.`,
    body: "View the redacted sample, or the live calibration dashboard.",
    href: "/sample-brief",
    cta: "View sample",
  };
  return (
    <nav
      className="relative border-b border-border bg-bg/80 backdrop-blur-md sticky top-0 z-40"
      aria-label="Primary"
    >
      <ScrollProgress />
      <div className="max-w-dash mx-auto px-6 py-4 flex items-center gap-8">
        <Link
          href="/"
          className="group flex items-center gap-2.5 text-fg hover:text-warm-glow transition-colors"
          onClick={() => setOpen(false)}
          aria-label={`${brand.name} home`}
        >
          <span className="reticle text-accent group-hover:text-warm-glow transition-colors" aria-hidden="true" />
          <span className="font-serif text-lg tracking-tight leading-none">
            {brand.name}
          </span>
        </Link>
        <ul className="hidden md:flex items-center gap-7 text-sm text-fg-muted">
          <NavMenu label="Research" items={researchItems} feature={researchFeature} />
          <NavMenu label="Products" items={productItems} feature={productFeature} />
          <li>
            <Link href="/pricing" className="hover:text-fg transition-colors">
              Pricing
            </Link>
          </li>
          <li>
            <Link href="/about" className="hover:text-fg transition-colors">
              About
            </Link>
          </li>
        </ul>
        <div className="ml-auto flex items-center gap-3">
          <Link
            href="/sign-in"
            className="text-sm text-fg-muted hover:text-fg transition-colors"
          >
            Sign in
          </Link>
          <Link
            href="/pricing"
            className="hidden sm:inline-flex items-center justify-center px-4 py-2 text-sm font-medium transition-colors duration-150 rounded-sm bg-accent text-bg hover:bg-accent-glow"
          >
            Subscribe
          </Link>
          <button
            type="button"
            className="md:hidden text-fg-muted hover:text-fg transition-colors p-1 -mr-1"
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            <svg
              width="22"
              height="22"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              {open ? (
                <>
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </>
              ) : (
                <>
                  <line x1="3" y1="6" x2="21" y2="6" />
                  <line x1="3" y1="12" x2="21" y2="12" />
                  <line x1="3" y1="18" x2="21" y2="18" />
                </>
              )}
            </svg>
          </button>
        </div>
      </div>
      {open && (
        <div className="md:hidden border-t border-border bg-bg-raised">
          <div className="px-6 py-5 space-y-6">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-fg-subtle mb-3">
                Research
              </p>
              <ul className="space-y-2.5">
                {researchItems.map((it) => (
                  <li key={it.href}>
                    <Link
                      href={it.href}
                      className="block text-fg-muted hover:text-fg transition-colors"
                      onClick={() => setOpen(false)}
                    >
                      {it.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-fg-subtle mb-3">
                Products
              </p>
              <ul className="space-y-2.5">
                {productItems.map((it) => (
                  <li key={it.href}>
                    <Link
                      href={it.href}
                      className="block text-fg-muted hover:text-fg transition-colors"
                      onClick={() => setOpen(false)}
                    >
                      {it.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
            <div className="pt-3 border-t border-border space-y-2.5">
              <Link
                href="/pricing"
                className="block text-fg-muted hover:text-fg transition-colors"
                onClick={() => setOpen(false)}
              >
                Pricing
              </Link>
              <Link
                href="/about"
                className="block text-fg-muted hover:text-fg transition-colors"
                onClick={() => setOpen(false)}
              >
                About
              </Link>
              <Link
                href="/sign-in"
                className="block text-accent hover:text-accent-glow transition-colors"
                onClick={() => setOpen(false)}
              >
                Subscriber sign in
              </Link>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}
