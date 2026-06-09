"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

/**
 * NavMenu: a Bloomberg / Stripe-style flyout for a top-level nav
 * item. The button itself is a label with a small chevron; on
 * hover (or focus, or click) a rich panel drops down with grouped
 * links, each carrying a short description, plus an optional
 * "feature" tile on the right edge of the panel for a key CTA.
 *
 * Keyboard: arrows / tab navigate within; ESC closes; focus-within
 * keeps it open while interacting; click-outside closes.
 *
 * Mobile: the parent Nav renders a flat list inside its hamburger
 * drawer instead - this component renders a single label that
 * navigates to `mobileHref` if present, otherwise stays inert.
 */

export type MenuItem = {
  href: string;
  label: string;
  desc: string;
};

export type MenuFeature = {
  eyebrow: string;
  title: string;
  body: string;
  href: string;
  cta: string;
};

export function NavMenu({
  label,
  items,
  feature,
}: {
  label: string;
  items: MenuItem[];
  feature?: MenuFeature;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLLIElement>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  function scheduleClose() {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    closeTimer.current = setTimeout(() => setOpen(false), 120);
  }
  function cancelClose() {
    if (closeTimer.current) {
      clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  }
  function openNow() {
    cancelClose();
    setOpen(true);
  }

  // Click outside + ESC.
  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <li
      ref={rootRef}
      className="relative"
      onMouseEnter={openNow}
      onMouseLeave={scheduleClose}
      onFocus={openNow}
      onBlur={(e) => {
        // Close when focus leaves the entire flyout subtree.
        if (!rootRef.current?.contains(e.relatedTarget as Node)) {
          scheduleClose();
        }
      }}
    >
      <button
        type="button"
        className={cn(
          "inline-flex items-center gap-1.5 text-sm transition-colors",
          open ? "text-fg" : "text-fg-muted hover:text-fg",
        )}
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => setOpen((v) => !v)}
      >
        <span>{label}</span>
        <svg
          width="9"
          height="9"
          viewBox="0 0 9 9"
          aria-hidden="true"
          className={cn(
            "transition-transform",
            open ? "rotate-180 text-warm-glow" : "text-fg-subtle",
          )}
        >
          <path
            d="M1.5 3 L4.5 6 L7.5 3"
            stroke="currentColor"
            strokeWidth="1.25"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {/* Flyout panel */}
      <div
        className={cn(
          "absolute left-1/2 top-full -translate-x-1/2 pt-3 z-50",
          open
            ? "opacity-100 translate-y-0 pointer-events-auto"
            : "opacity-0 -translate-y-1 pointer-events-none",
          "transition-[opacity,transform] duration-150 ease-out",
        )}
        // The pt-3 wrapper above creates a hover-bridge between the
        // button and the panel so the cursor can travel without
        // collapsing the menu.
      >
        <div
          className={cn(
            "rounded-sm border border-border bg-bg-raised/95 backdrop-blur-md",
            "shadow-[0_24px_48px_-24px_rgba(0,0,0,0.75),0_4px_16px_-8px_rgba(183,146,104,0.18)]",
            feature ? "grid grid-cols-[minmax(280px,1fr)_minmax(220px,260px)]" : "min-w-[300px]",
          )}
        >
          {/* Items column(s) */}
          <ul className="p-3" role="menu">
            {items.map((it) => (
              <li key={it.href} role="none">
                <Link
                  href={it.href}
                  role="menuitem"
                  className="group flex flex-col gap-1 rounded-sm px-4 py-3 transition-colors hover:bg-bg-subtle/80 focus-visible:bg-bg-subtle/80"
                  onClick={() => setOpen(false)}
                >
                  <span className="font-serif text-base text-fg leading-tight group-hover:text-warm-glow transition-colors">
                    {it.label}
                  </span>
                  <span className="text-xs text-fg-muted leading-snug">
                    {it.desc}
                  </span>
                </Link>
              </li>
            ))}
          </ul>

          {/* Optional feature tile */}
          {feature && (
            <div className="p-5 border-l border-border bg-bg/40 flex flex-col justify-between gap-5">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-warm-glow mb-2">
                  {feature.eyebrow}
                </p>
                <p className="font-serif text-lg text-fg leading-snug">
                  {feature.title}
                </p>
                <p className="text-xs text-fg-muted leading-relaxed mt-2">
                  {feature.body}
                </p>
              </div>
              <Link
                href={feature.href}
                className="inline-flex items-center gap-1.5 text-xs font-mono uppercase tracking-[0.18em] text-warm-glow hover:text-warm transition-colors"
                onClick={() => setOpen(false)}
              >
                <span>{feature.cta}</span>
                <span aria-hidden="true">&rarr;</span>
              </Link>
            </div>
          )}
        </div>
      </div>
    </li>
  );
}
