import type { Metadata } from "next";
import { LinkButton } from "@/components/ui/button";

export const metadata: Metadata = {
  title: "Account connected",
  description: "Google sign-in callback for Aurora Research.",
};

export default function AuthCallbackPage() {
  return (
    <main className="max-w-prose mx-auto px-6 py-20">
      <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
        Account
      </p>
      <h1 className="font-serif text-4xl md:text-5xl leading-tight text-fg">
        Authentication returned to Aurora Research.
      </h1>
      <p className="mt-6 text-fg-muted leading-relaxed">
        If your browser has completed the Google sign-in flow, continue
        to the subscriber portal. Preview access starts on first login
        and lasts five days. If paid access is not active yet, use the
        same email you paid with and contact support.
      </p>
      <div className="mt-8 flex flex-wrap gap-3">
        <LinkButton href="/portal" variant="primary">
          Open portal
        </LinkButton>
        <LinkButton href="/contact" variant="secondary">
          Contact support
        </LinkButton>
      </div>
    </main>
  );
}
