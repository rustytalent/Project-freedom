import { LinkButton } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center px-6">
      <div className="max-w-prose text-center">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
          404
        </p>
        <h1 className="font-serif text-4xl text-fg">Page not found.</h1>
        <p className="mt-4 text-fg-muted leading-relaxed">
          The page you&rsquo;re looking for either moved or never
          existed. The landing page below has links to everything we
          publish.
        </p>
        <div className="mt-8 flex justify-center gap-3">
          <LinkButton href="/" variant="primary">
            Go home
          </LinkButton>
          <LinkButton href="/sample-brief" variant="secondary">
            View sample brief
          </LinkButton>
        </div>
      </div>
    </div>
  );
}
