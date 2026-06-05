import {
  ARTIFACT_KIND_DESCRIPTIONS,
  ARTIFACT_KIND_LABELS,
  TIER_LABELS,
  type ArtifactRecord,
  relativeTimeFromNow,
} from "@/lib/artifacts";

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(2)} MB`;
}

export function ArtifactList({
  records,
  emptyMessage,
}: {
  records: ArtifactRecord[];
  emptyMessage?: string;
}) {
  if (records.length === 0) {
    return (
      <p className="text-sm text-fg-muted py-8 text-center border border-dashed border-border rounded-sm">
        {emptyMessage ??
          "Nothing here yet. Latest artifacts appear here as soon as the engine publishes them."}
      </p>
    );
  }
  return (
    <ul className="space-y-2">
      {records.map((r, i) => (
        <li
          key={r.id}
          className="group bg-bg-raised border border-border rounded-sm overflow-hidden hover:border-accent/60 transition-colors"
        >
          <div className="flex flex-wrap items-start gap-4 p-5">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                {i === 0 && (
                  <span
                    className="relative flex h-2 w-2"
                    aria-label="Most recent"
                  >
                    <span className="absolute inline-flex h-full w-full rounded-full bg-accent opacity-60 animate-ping" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-accent" />
                  </span>
                )}
                <p className="font-mono text-xs text-fg-subtle uppercase tracking-wider">
                  {ARTIFACT_KIND_LABELS[r.kind]}
                </p>
                <span className="text-xs text-fg-subtle">·</span>
                <p className="text-xs text-fg-subtle">
                  {relativeTimeFromNow(r.generated_at_utc)}
                </p>
                {r.trading_date_ist && (
                  <>
                    <span className="text-xs text-fg-subtle">·</span>
                    <p className="font-mono text-xs text-fg-subtle tabnum">
                      {r.trading_date_ist}
                    </p>
                  </>
                )}
              </div>
              <p className="text-fg font-serif text-lg leading-snug">
                {r.filename}
              </p>
              {r.description && (
                <p className="text-sm text-fg-muted mt-1 leading-relaxed">
                  {r.description}
                </p>
              )}
              {!r.description && ARTIFACT_KIND_DESCRIPTIONS[r.kind] && (
                <p className="text-sm text-fg-muted mt-1 leading-relaxed">
                  {ARTIFACT_KIND_DESCRIPTIONS[r.kind]}
                </p>
              )}
              <p className="text-xs text-fg-subtle mt-2 tabnum font-mono">
                {formatBytes(r.bytes)} · {TIER_LABELS[r.tier]}
              </p>
            </div>
            <a
              href={`/api/artifacts/${r.id}/download`}
              className="inline-flex items-center gap-2 px-4 py-2 bg-accent text-bg text-sm font-medium rounded-sm hover:bg-accent-glow transition-colors whitespace-nowrap"
              download={r.filename}
            >
              Download
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            </a>
          </div>
        </li>
      ))}
    </ul>
  );
}
