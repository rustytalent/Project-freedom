import Link from "next/link";
import { brand } from "@/lib/brand";

const groups: Array<{
  heading: string;
  links: Array<{ href: string; label: string }>;
}> = [
  {
    heading: "Product",
    links: [
      { href: "/products/daily-brief", label: "Core Research" },
      { href: "/products/swing-brief", label: "Live Desk" },
      { href: "/products/diagnosis", label: "Audit Base" },
      { href: "/products/audit-infrastructure", label: "Audit infrastructure" },
    ],
  },
  {
    heading: "Evidence",
    links: [
      { href: "/track-record", label: "Track record" },
      { href: "/sample-brief", label: "Sample brief" },
      { href: "/yesterday-audit", label: "Yesterday audit" },
      { href: "/philosophy", label: "Philosophy" },
    ],
  },
  {
    heading: "Company",
    links: [
      { href: "/about", label: "About" },
      { href: "/contact", label: "Contact" },
      { href: "/pricing", label: "Pricing" },
    ],
  },
  {
    heading: "Legal",
    links: [
      { href: "/legal/disclosures", label: "Disclosures" },
      { href: "/legal/terms", label: "Terms" },
      { href: "/legal/privacy", label: "Privacy" },
    ],
  },
];

const dontDoList: string[] = [
  "We do not publish entry, exit, or position-size instructions.",
  "We do not operate Telegram or WhatsApp tip channels.",
  "We do not pay or accept promoters, finfluencers, or affiliates.",
  "We do not republish other desks' research as our own.",
];

function StatusStrip() {
  return (
    <div className="border-t border-b border-border bg-bg-raised/40">
      <div className="max-w-dash mx-auto px-6 py-3 flex flex-wrap items-center gap-x-8 gap-y-2 text-[11px] font-mono uppercase tracking-[0.16em] text-fg-muted">
        <span className="flex items-center gap-2.5">
          <span className="live-dot" />
          <span className="text-fg">System operational</span>
        </span>
        <span className="text-fg-subtle">·</span>
        <span>Next brief · Tomorrow 08:30 IST</span>
        <span className="text-fg-subtle hidden md:inline">·</span>
        <span className="hidden md:inline">Audit transparency · 100%</span>
        <span className="ml-auto text-fg-subtle">
          {brand.domain}
        </span>
      </div>
    </div>
  );
}

export function Footer() {
  return (
    <footer className="mt-24">
      <StatusStrip />

      <div className="border-b border-border">
        <div className="max-w-dash mx-auto px-6 py-14 grid gap-10 lg:grid-cols-12">
          {/* Brand statement column */}
          <div className="lg:col-span-4">
            <div className="flex items-center gap-2.5 mb-4">
              <span className="reticle text-accent" aria-hidden="true" />
              <p className="font-serif text-xl text-fg">{brand.name}</p>
            </div>
            <p className="text-sm text-fg-muted leading-relaxed max-w-xs">
              {brand.tagline}
            </p>
            <p className="mt-4 text-xs text-fg-subtle leading-relaxed max-w-xs">
              Published from India for the Indian markets. The brief
              lands before NSE opens; the audit publishes the next
              session. Nothing in between pretends to be a tip.
            </p>
          </div>

          {/* Link columns */}
          {groups.map((g) => (
            <div key={g.heading} className="lg:col-span-2">
              <h4 className="text-xs uppercase tracking-[0.16em] text-fg-subtle mb-4">
                {g.heading}
              </h4>
              <ul className="space-y-2.5">
                {g.links.map((l) => (
                  <li key={l.href}>
                    <Link
                      href={l.href}
                      className="text-sm text-fg-muted hover:text-fg transition-colors"
                    >
                      {l.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* What we don't do callout */}
      <div className="border-b border-border bg-bg-raised/30">
        <div className="max-w-dash mx-auto px-6 py-10 grid gap-8 md:grid-cols-[1fr_2fr]">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-accent mb-3">
              What we don&rsquo;t do
            </p>
            <p className="font-serif text-2xl text-fg leading-snug max-w-xs">
              The four lines we&rsquo;ve decided we&rsquo;ll never cross.
            </p>
          </div>
          <ul className="grid gap-3 text-sm text-fg-muted leading-relaxed sm:grid-cols-2">
            {dontDoList.map((s) => (
              <li key={s} className="flex gap-3">
                <span aria-hidden="true" className="font-mono text-drift mt-0.5">
                  ×
                </span>
                <span>{s}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Disclosure + © strip */}
      <div className="bg-bg">
        <div className="max-w-dash mx-auto px-6 py-7 text-xs text-fg-subtle leading-relaxed flex flex-wrap items-start justify-between gap-4">
          <p className="max-w-2xl">
            {brand.sebi.shortDisclosure} See{" "}
            <Link
              href="/legal/disclosures"
              className="text-fg-muted underline underline-offset-2"
            >
              full disclosures
            </Link>
            .
          </p>
          <p className="font-mono tracking-wide">
            © {new Date().getFullYear()} {brand.legalEntity ?? brand.name}.
            All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
