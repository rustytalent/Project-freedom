import { cn } from "@/lib/utils";

/**
 * BriefExcerpt: a mono panel showing a slice of brief copy.
 *
 * When `redactTokens` is provided, any occurrence of those tokens is
 * rendered with a bronze "subscriber-only" treatment - a redaction
 * stamp aesthetic that signals the document is a real published
 * artifact with material withheld for paying readers, rather than
 * marketing prose. The default empty set leaves the body alone.
 */
function renderWithRedactions(body: string, tokens: string[]) {
  if (tokens.length === 0) return body;
  // Build a single regex that captures any of the tokens. Escape any
  // regex metachars conservatively.
  const escaped = tokens.map((t) =>
    t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
  );
  const re = new RegExp(`(${escaped.join("|")})`, "g");
  const parts = body.split(re);
  return parts.map((p, i) => {
    if (tokens.includes(p)) {
      return (
        <span
          key={i}
          className="inline-block px-1.5 -my-0.5 mx-0.5 bg-warm/15 text-warm-glow border border-warm/30 rounded-[2px] tabnum"
        >
          {p}
        </span>
      );
    }
    return <span key={i}>{p}</span>;
  });
}

export function BriefExcerpt({
  label,
  body,
  className,
  redactTokens = [],
}: {
  label: string;
  body: string;
  className?: string;
  /** Tokens to render with a bronze "subscriber-only" redaction chip. */
  redactTokens?: string[];
}) {
  return (
    <figure
      className={cn(
        "bg-bg-raised border border-border rounded-sm p-6 font-mono text-sm text-fg-muted leading-relaxed whitespace-pre-wrap tabnum",
        className,
      )}
      aria-label={`Excerpt from a published brief - ${label}`}
    >
      <figcaption className="flex items-center justify-between text-xs uppercase tracking-wider text-fg-subtle mb-4 not-italic font-sans">
        <span>Excerpt &middot; {label}</span>
        {redactTokens.length > 0 && (
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-warm-glow">
            subscriber-only · redacted
          </span>
        )}
      </figcaption>
      {renderWithRedactions(body, redactTokens)}
    </figure>
  );
}

