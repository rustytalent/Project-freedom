import Link from "next/link";
import { LinkButton } from "@/components/ui/button";

export function ProductPage({
  eyebrow,
  title,
  oneLiner,
  intro,
  sections,
  delivery,
  cost,
  notes,
}: {
  eyebrow: string;
  title: string;
  oneLiner: string;
  intro: React.ReactNode;
  sections: Array<{ heading: string; body: React.ReactNode }>;
  delivery: string;
  cost: string;
  notes?: React.ReactNode;
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
        <p className="mt-4 text-lg text-fg-muted leading-relaxed">
          {oneLiner}
        </p>
        <div className="mt-8 text-fg-muted leading-relaxed space-y-4">
          {intro}
        </div>
      </header>

      <div className="space-y-12 border-t border-border pt-12">
        {sections.map((s) => (
          <section key={s.heading}>
            <h2 className="font-serif text-2xl text-fg mb-4">{s.heading}</h2>
            <div className="text-fg-muted leading-relaxed space-y-4">
              {s.body}
            </div>
          </section>
        ))}
      </div>

      <div className="mt-16 pt-12 border-t border-border grid gap-8 sm:grid-cols-2">
        <div>
          <h3 className="text-xs uppercase tracking-wider text-fg-subtle mb-2">
            Delivery
          </h3>
          <p className="text-fg leading-relaxed">{delivery}</p>
        </div>
        <div>
          <h3 className="text-xs uppercase tracking-wider text-fg-subtle mb-2">
            Cost
          </h3>
          <p className="text-fg leading-relaxed">{cost}</p>
        </div>
      </div>

      {notes && (
        <div className="mt-12 text-xs text-fg-subtle leading-relaxed">
          {notes}
        </div>
      )}

      <div className="mt-16 pt-12 border-t border-border flex flex-wrap gap-3">
        <LinkButton href="/sample-brief" variant="primary">
          View sample brief
        </LinkButton>
        <LinkButton href="/pricing" variant="secondary">
          Pricing
        </LinkButton>
        <Link
          href="/contact"
          className="ml-2 text-sm text-fg-muted hover:text-fg transition-colors self-center"
        >
          Or talk to us →
        </Link>
      </div>
    </article>
  );
}
