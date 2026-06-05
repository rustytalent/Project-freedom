import { cn } from "@/lib/utils";

export function LivePulse({
  label,
  className,
}: {
  label: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2.5 text-xs text-fg-muted",
        "border border-border bg-bg-raised/60 rounded-full px-3 py-1",
        "backdrop-blur-sm",
        className,
      )}
    >
      <span className="live-dot" aria-hidden="true" />
      <span className="tabnum">{label}</span>
    </span>
  );
}
