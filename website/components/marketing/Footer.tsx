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

export function Footer() {
  return (
    <footer className="border-t border-border mt-24">
      <div className="max-w-dash mx-auto px-6 py-12 grid gap-10 md:grid-cols-5">
        <div className="md:col-span-1">
          <p className="font-serif text-lg text-fg">{brand.name}</p>
          <p className="text-xs text-fg-subtle mt-2 max-w-xs">
            {brand.tagline}
          </p>
        </div>
        {groups.map((g) => (
          <div key={g.heading}>
            <h4 className="text-xs uppercase tracking-wider text-fg-subtle mb-3">
              {g.heading}
            </h4>
            <ul className="space-y-2">
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
      <div className="border-t border-border">
        <div className="max-w-dash mx-auto px-6 py-6 text-xs text-fg-subtle leading-relaxed">
          <p>
            {brand.sebi.shortDisclosure} See{" "}
            <Link
              href="/legal/disclosures"
              className="text-fg-muted underline underline-offset-2"
            >
              full disclosures
            </Link>
            .
          </p>
          <p className="mt-2">
            © {new Date().getFullYear()} {brand.legalEntity}. All rights
            reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
