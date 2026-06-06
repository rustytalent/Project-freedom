import type { Metadata } from "next";
import { brand } from "@/lib/brand";

export const metadata: Metadata = {
  title: "Contact",
  description: `Get in touch with ${brand.name}.`,
};

export default function ContactPage() {
  return (
    <article className="max-w-prose mx-auto px-6 py-20">
      <header className="mb-12">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
          Contact
        </p>
        <h1 className="font-serif text-4xl md:text-5xl leading-tight text-fg">
          Talk to us.
        </h1>
        <p className="mt-6 text-fg-muted leading-relaxed">
          Product questions, partnership requests, or an Audit Base
          intake call - email is the simplest channel for all of them.
        </p>
      </header>

      <section className="space-y-6 text-fg-muted leading-relaxed">
        <p>
          <strong className="text-fg">Email:</strong>{" "}
          <a
            href={`mailto:${brand.contact.email}`}
            className="text-accent underline underline-offset-4"
          >
            {brand.contact.email}
          </a>
        </p>
        <p>
          For Audit Base intakes, include a one-paragraph description of
          your strategy and a rough trade count. We&rsquo;ll send you
          the CSV column spec and a 30-minute intake call link.
        </p>
        <p>
          For Audit Infrastructure engagements, mention your team size
          and the slice of the audit harness you&rsquo;re interested
          in. We&rsquo;ll send a one-page proposal within two business
          days.
        </p>
        <p>
          For Core Research or Live Desk questions, the sample brief and
          pricing page usually answer them - email if anything is
          still unclear.
        </p>
      </section>

      <section className="mt-16 pt-12 border-t border-border text-xs text-fg-subtle leading-relaxed">
        <p>
          We do not provide trade ideas, sectoral views, or stock
          recommendations over email or phone. We are not a SEBI-
          registered investment advisor. The published research is the
          product; one-on-one advisory is not in scope.
        </p>
      </section>
    </article>
  );
}
