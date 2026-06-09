import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Reveal } from "@/components/ui/Reveal";
import { SectionStamp } from "./SectionStamp";

export function PitchSection({
  eyebrow,
  title,
  body,
  artifact,
  reverse = false,
  stamp,
  stampLabel,
  stampTone,
}: {
  eyebrow: string;
  title: ReactNode;
  body: ReactNode;
  artifact: ReactNode;
  reverse?: boolean;
  stamp?: string;
  stampLabel?: string;
  stampTone?: "neutral" | "warm";
}) {
  return (
    <>
      {stamp && stampLabel ? (
        <SectionStamp stamp={stamp} label={stampLabel} tone={stampTone} />
      ) : (
        <div className="border-t border-border" aria-hidden="true" />
      )}
      <section>
        <div
          className={cn(
            "max-w-dash mx-auto px-6 py-24 md:py-28 grid md:grid-cols-2 gap-12 lg:gap-16 items-start",
            reverse && "md:[&>div:first-child]:order-2",
          )}
        >
          <Reveal>
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
                {eyebrow}
              </p>
              <h2 className="font-serif text-3xl md:text-4xl lg:text-[42px] leading-[1.08] tracking-[-0.01em] text-fg max-w-md">
                {title}
              </h2>
              <div className="mt-7 text-fg-muted leading-relaxed space-y-4 max-w-md">
                {body}
              </div>
            </div>
          </Reveal>
          <Reveal delay={140}>
            <div>{artifact}</div>
          </Reveal>
        </div>
      </section>
    </>
  );
}
