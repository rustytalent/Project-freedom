import Link from "next/link";
import { brand } from "@/lib/brand";

const links: Array<{ href: string; label: string }> = [
  { href: "/philosophy", label: "Philosophy" },
  { href: "/products", label: "Products" },
  { href: "/track-record", label: "Track record" },
  { href: "/sample-brief", label: "Sample brief" },
  { href: "/pricing", label: "Pricing" },
];

export function Nav() {
  return (
    <nav
      className="border-b border-border bg-bg/80 backdrop-blur-sm sticky top-0 z-40"
      aria-label="Primary"
    >
      <div className="max-w-dash mx-auto px-6 py-4 flex items-center gap-8">
        <Link
          href="/"
          className="font-serif text-lg tracking-tight text-fg hover:text-accent-glow transition-colors"
        >
          {brand.name}
        </Link>
        <ul className="hidden md:flex items-center gap-6 text-sm text-fg-muted">
          {links.map((l) => (
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
          <Link
            href="/portal"
            className="text-sm text-fg-muted hover:text-fg transition-colors"
          >
            Subscriber sign in
          </Link>
        </div>
      </div>
    </nav>
  );
}
