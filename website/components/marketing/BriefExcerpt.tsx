import { cn } from "@/lib/utils";

export function BriefExcerpt({
  label,
  body,
  className,
}: {
  label: string;
  body: string;
  className?: string;
}) {
  return (
    <figure
      className={cn(
        "bg-bg-raised border border-border rounded-sm p-6 font-mono text-sm text-fg-muted leading-relaxed whitespace-pre-wrap tabnum",
        className,
      )}
      aria-label={`Excerpt from a published brief - ${label}`}
    >
      <figcaption className="text-xs uppercase tracking-wider text-fg-subtle mb-4 not-italic font-sans">
        Excerpt - {label}
      </figcaption>
      {body}
    </figure>
  );
}
