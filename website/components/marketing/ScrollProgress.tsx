"use client";

import { useEffect, useState } from "react";

/**
 * ScrollProgress: a 1.5px bronze hairline pinned just under the
 * primary nav that fills left-to-right as the visitor scrolls down
 * the page. The Linear / Stripe pattern - signals depth, gives
 * scroll its own visible reward.
 *
 * Pure CSS transform on a single child so we don't reflow on every
 * scroll frame. Uses passive listeners. Respects prefers-reduced-
 * motion by disabling the transition (the bar still tracks position).
 */
export function ScrollProgress() {
  const [pct, setPct] = useState(0);

  useEffect(() => {
    const update = () => {
      const doc = document.documentElement;
      const max = doc.scrollHeight - window.innerHeight;
      const cur = window.scrollY;
      setPct(max <= 0 ? 0 : Math.min(1, Math.max(0, cur / max)));
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, []);

  return (
    <div
      aria-hidden="true"
      className="absolute left-0 right-0 bottom-0 h-px pointer-events-none"
    >
      <div
        className="h-full origin-left bg-gradient-to-r from-warm/80 via-warm to-warm-glow motion-safe:transition-transform motion-safe:duration-150 motion-safe:ease-out"
        style={{ transform: `scaleX(${pct})` }}
      />
    </div>
  );
}
