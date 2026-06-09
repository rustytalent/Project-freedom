/**
 * SectionStamp: a newspaper-style section break. A hairline with a
 * mono "S/02 · DISCIPLINE" tag pinned to the left edge of the rule.
 *
 * Replaces the plain border-t at major section transitions. Carries
 * editorial authority across the long scroll - the visitor never
 * stops feeling like they're reading a serialised publication.
 *
 * Usage: drop above a section that no longer renders its own border.
 *   <SectionStamp stamp="S/02" label="Audit" />
 *   <section className="...">...</section>
 */
export function SectionStamp({
  stamp,
  label,
  tone = "neutral",
}: {
  stamp: string;
  label: string;
  /** "warm" tints the tag bronze - reserved for special transitions. */
  tone?: "neutral" | "warm";
}) {
  const stampColor = tone === "warm" ? "text-warm-glow" : "text-fg-subtle";
  const labelColor = tone === "warm" ? "text-warm" : "text-fg-muted";
  return (
    <div className="relative border-t border-border" aria-hidden="true">
      <div className="max-w-dash mx-auto px-6">
        <span
          className={
            "absolute -top-[9px] left-6 bg-bg px-3 font-mono text-[10px] " +
            "uppercase tracking-[0.22em] whitespace-nowrap " +
            stampColor
          }
        >
          {stamp}
          <span className="text-border mx-2">·</span>
          <span className={labelColor}>{label}</span>
        </span>
      </div>
    </div>
  );
}
