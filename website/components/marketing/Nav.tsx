"use client";

import { useState } from "react";
import Link from "next/link";
import { brand } from "@/lib/brand";
import { ScrollProgress } from "./ScrollProgress";

const links: Array<{ href: string; label: string }> = [
  { href: "/philosophy", label: "Philosophy" },
  { href: "/products", label: "Products" },
  { href: "/track-record", label: "Track record" },
  { href: "/sample-brief", label: "Sample brief" },
  { href: "/pricing", label: "Pricing" },
];

export function Nav() {
  const [open, setOpen] = useState(false);
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
          <ul className="px-6 py-4 space-y-3">
            {links.map((l) => (
              <li key={l.href}>
                <Link
                  href={l.href}
                  className="block py-1 text-fg-muted hover:text-fg transition-colors"
                  onClick={() => setOpen(false)}
                >
                  {l.label}
                </Link>
              </li>
            ))}
            <li className="pt-2 border-t border-border">
              <Link
                href="/sign-in"
                className="block py-1 text-accent hover:text-accent-glow transition-colors"
                onClick={() => setOpen(false)}
              >
                Subscriber sign in
              </Link>
            </li>
          </ul>
        </div>
      )}
    </nav>
  );
}
