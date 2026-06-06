import Link from "next/link";
import { brand } from "@/lib/brand";

const portalLinks: Array<{ href: string; label: string }> = [
  { href: "/portal", label: "Today" },
  { href: "/portal/brief/today", label: "Today's brief" },
  { href: "/portal/artifacts", label: "Artifacts" },
  { href: "/portal/calibration", label: "Calibration" },
  { href: "/portal/account", label: "Account" },
];

export default function PortalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-border sticky top-0 z-40 bg-bg/90 backdrop-blur-sm">
        <div className="max-w-dash mx-auto px-6 py-4 flex items-center gap-6">
          <Link
            href="/"
            className="font-serif text-lg text-fg hover:text-accent-glow transition-colors"
          >
            {brand.name}
          </Link>
          <span className="text-fg-subtle text-xs uppercase tracking-wider">
            Subscriber portal
          </span>
          <ul className="hidden md:flex items-center gap-5 text-sm text-fg-muted ml-6">
            {portalLinks.map((l) => (
              <li key={l.href}>
                <Link
                  href={l.href}
                  className="hover:text-fg transition-colors"
                >
                  {l.label}
                </Link>
              </li>
            ))}
          </ul>
          <div className="ml-auto flex items-center gap-3">
            <span className="text-xs text-fg-subtle">account@gcruxresearch.in</span>
            <button
              className="text-xs text-fg-muted hover:text-fg transition-colors"
              type="button"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>
      <main className="flex-1">{children}</main>
      <footer className="border-t border-border">
        <div className="max-w-dash mx-auto px-6 py-6 text-xs text-fg-subtle">
          {brand.sebi.shortDisclosure}{" "}
          <Link
            href="/legal/disclosures"
            className="text-fg-muted underline underline-offset-2"
          >
            See full disclosures.
          </Link>
        </div>
      </footer>
    </div>
  );
}
