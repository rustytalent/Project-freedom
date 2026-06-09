import { ReactNode } from "react";

/**
 * PageHeader: the editorial header strip used at the top of every
 * substantive marketing page. Mirrors the home hero's top strip so
 * the premium identity carries across the site rather than dropping
 * off a cliff after the landing page.
 *
 * Composition:
 *   [section label] · [edition chip (warm bronze)] · [dateline] · [live status?]
 *   <Headline focal=bronze />
 *   <Lead off-white />
 *   [Meta line, mono, fg-subtle]
 *
 * The title is passed as ReactNode so callers can mark a focal word
 * with <span className="text-warm">...</span> for the warm anchor.
 */
export function PageHeader({
  section,
  editionTag,
  dateline,
  title,
  lead,
  meta,
  status,
}: {
  section: string;
  editionTag?: string;
  dateline?: string;
  title: ReactNode;
  lead?: ReactNode;
  meta?: ReactNode;
  status?: ReactNode;
}) {
  return (
    <header className="max-w-dash mx-auto px-6 pt-12 md:pt-16 pb-10 fade-in-up">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-10 pb-5 border-b border-border/60">
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-fg-muted">
          {section}
        </span>
        {editionTag && (
          <>
            <span className="hidden sm:inline-block w-px h-3 bg-border" />
            <span
              className="font-mono text-[11px] uppercase tracking-[0.18em] text-warm-glow px-2 py-0.5 border border-warm/40 rounded-sm"
              aria-label="Edition or revision tag"
            >
              {editionTag}
            </span>
          </>
        )}
        {dateline && (
          <>
            <span className="hidden sm:inline-block w-px h-3 bg-border" />
            <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-fg-subtle">
              {dateline}
            </span>
          </>
        )}
        {status && (
          <>
            <span className="hidden sm:inline-block w-px h-3 bg-border" />
            {status}
          </>
        )}
      </div>
      <div className="max-w-3xl">
        <h1 className="font-serif text-4xl md:text-5xl lg:text-[56px] leading-[1.05] tracking-[-0.02em] text-fg">
          {title}
        </h1>
        {lead && (
          <div className="mt-7 text-lg text-fg-muted leading-relaxed max-w-2xl">
            {lead}
          </div>
        )}
        {meta && (
          <div className="mt-5 font-mono text-[11px] uppercase tracking-[0.18em] text-fg-subtle">
            {meta}
          </div>
        )}
      </div>
    </header>
  );
}
