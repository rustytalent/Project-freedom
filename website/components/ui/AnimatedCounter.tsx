"use client";

import { useEffect, useRef, useState } from "react";

export type CounterFormat = {
  /** "int" rounds to nearest integer; "decimal" preserves digits. */
  kind: "int" | "decimal";
  /** decimal places (only used when kind="decimal"). default 1. */
  decimals?: number;
  /** appended after the number (e.g. "%"). default "". */
  suffix?: string;
  /** prepended before the number (e.g. "₹"). default "". */
  prefix?: string;
  /** locale used for thousand grouping on integers. default "en-IN". */
  locale?: string;
};

function render(n: number, fmt: CounterFormat): string {
  const { kind, decimals = 1, suffix = "", prefix = "", locale = "en-IN" } = fmt;
  const body =
    kind === "int"
      ? Math.round(n).toLocaleString(locale)
      : n.toFixed(decimals);
  return prefix + body + suffix;
}

/**
 * AnimatedCounter: count-up from 0 to `value`, triggered on first
 * viewport entry. Format is described via a plain config object so
 * this client component can be rendered from a server component
 * (functions can't cross the RSC boundary).
 *
 * Respects prefers-reduced-motion - skips animation, renders final.
 */
export function AnimatedCounter({
  value,
  format,
  duration = 1400,
}: {
  value: number;
  format: CounterFormat;
  duration?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const [n, setN] = useState(0);
  const started = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || started.current) return;
    const reducedQ =
      typeof window !== "undefined" && "matchMedia" in window
        ? window.matchMedia("(prefers-reduced-motion: reduce)")
        : null;
    if (reducedQ?.matches) {
      started.current = true;
      setN(value);
      return;
    }
    if (typeof IntersectionObserver === "undefined") {
      started.current = true;
      setN(value);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting || started.current) continue;
          started.current = true;
          io.disconnect();
          const start = performance.now();
          const tick = (now: number) => {
            const t = Math.min(1, (now - start) / duration);
            const eased = 1 - Math.pow(1 - t, 3);
            setN(value * eased);
            if (t < 1) requestAnimationFrame(tick);
            else setN(value);
          };
          requestAnimationFrame(tick);
        }
      },
      { threshold: 0.3 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [value, duration]);

  return <span ref={ref}>{render(n, format)}</span>;
}
