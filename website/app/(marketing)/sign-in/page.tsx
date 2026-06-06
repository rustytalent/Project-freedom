import type { Metadata } from "next";
import { LinkButton } from "@/components/ui/button";
import { brand } from "@/lib/brand";

export const metadata: Metadata = {
  title: "Sign in",
  description: "Sign in to Aurora Research with Google.",
};

function siteOrigin(): string {
  const configured = process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/+$/, "");
  if (configured) return configured;
  return `https://${brand.domain}`;
}

function googleSignInUrl(): string | null {
  const rawUrl = process.env.NEXT_PUBLIC_SUPABASE_URL?.replace(/\/+$/, "");
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!rawUrl || !anonKey) return null;

  const url = new URL(`${rawUrl}/auth/v1/authorize`);
  url.searchParams.set("provider", "google");
  url.searchParams.set("redirect_to", `${siteOrigin()}/auth/callback`);
  url.searchParams.set("apikey", anonKey);
  return url.toString();
}

export default function SignInPage() {
  const href = googleSignInUrl();
  return (
    <main className="max-w-prose mx-auto px-6 py-20">
      <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
        Account access
      </p>
      <h1 className="font-serif text-4xl md:text-5xl leading-tight text-fg">
        Sign in with Google.
      </h1>
      <p className="mt-6 text-fg-muted leading-relaxed">
        Google sign-in starts your five-day preview automatically. The
        same email used at checkout should be used here so the portal
        can match your subscription tier after payment.
      </p>

      <div className="mt-10 border border-border bg-bg-raised p-6 rounded-sm">
        {href ? (
          <>
            <LinkButton href={href} variant="primary">
              Continue with Google
            </LinkButton>
            <p className="mt-4 text-xs text-fg-subtle leading-relaxed">
              You will be redirected to Google, then back to the portal
              after authentication. Preview access starts on the first
              successful login and does not reset on later logins.
            </p>
          </>
        ) : (
          <>
            <p className="text-sm text-fg-muted leading-relaxed">
              Google sign-in is not configured on this deployment yet.
              Add{" "}
              <span className="font-mono text-fg">
                NEXT_PUBLIC_SUPABASE_URL
              </span>{" "}
              and{" "}
              <span className="font-mono text-fg">
                NEXT_PUBLIC_SUPABASE_ANON_KEY
              </span>{" "}
              in Vercel, then enable Google in the Supabase Auth
              dashboard.
            </p>
            <div className="mt-5">
              <LinkButton href="/contact" variant="secondary">
                Contact support
              </LinkButton>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
