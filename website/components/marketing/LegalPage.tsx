export function LegalPage({
  eyebrow,
  title,
  lastUpdated,
  children,
}: {
  eyebrow: string;
  title: string;
  lastUpdated: string;
  children: React.ReactNode;
}) {
  return (
    <article className="max-w-prose mx-auto px-6 py-20">
      <header className="mb-12">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
          {eyebrow}
        </p>
        <h1 className="font-serif text-4xl md:text-5xl leading-tight text-fg">
          {title}
        </h1>
        <p className="mt-4 text-xs text-fg-subtle">
          Last updated: {lastUpdated}
        </p>
      </header>
      <div className="text-fg-muted leading-relaxed space-y-6 prose prose-invert">
        {children}
      </div>
    </article>
  );
}
