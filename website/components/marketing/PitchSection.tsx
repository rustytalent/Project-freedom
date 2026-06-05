import { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function PitchSection({
  eyebrow,
  title,
  body,
  artifact,
  reverse = false,
}: {
  eyebrow: string;
  title: string;
  body: ReactNode;
  artifact: ReactNode;
  reverse?: boolean;
}) {
  return (
    <section className="border-t border-border">
      <div
        className={cn(
          "max-w-dash mx-auto px-6 py-20 grid md:grid-cols-2 gap-12 items-start",
          reverse && "md:[&>div:first-child]:order-2",
        )}
      >
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
            {eyebrow}
          </p>
          <h2 className="font-serif text-3xl md:text-4xl leading-tight text-fg max-w-md">
            {title}
          </h2>
          <div className="mt-6 text-fg-muted leading-relaxed space-y-4 max-w-md">
            {body}
          </div>
        </div>
        <div>{artifact}</div>
      </div>
    </section>
  );
}
