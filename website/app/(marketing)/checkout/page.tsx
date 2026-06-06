import type { Metadata } from "next";
import { CheckoutClient } from "@/components/billing/CheckoutClient";
import { parseBillingCycle, parsePlanId } from "@/lib/pricing";

export const metadata: Metadata = {
  title: "Checkout",
  description: "Subscribe to Aurora Research through Razorpay.",
};

export default async function CheckoutPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const rawPlan = Array.isArray(params.plan) ? params.plan[0] : params.plan;
  const rawCycle = Array.isArray(params.cycle) ? params.cycle[0] : params.cycle;
  const plan = parsePlanId(rawPlan) ?? "daily";
  const cycle = parseBillingCycle(rawCycle);

  return (
    <main className="max-w-dash mx-auto px-6 py-20">
      <header className="max-w-prose mb-14">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-4">
          Checkout
        </p>
        <h1 className="font-serif text-4xl md:text-5xl leading-tight text-fg">
          Subscribe through Razorpay.
        </h1>
        <p className="mt-6 text-fg-muted leading-relaxed">
          Choose a plan, confirm the billing cycle, and pay through
          Razorpay. Google sign-in is used for account access after
          payment confirmation.
        </p>
      </header>
      <CheckoutClient initialPlan={plan} initialCycle={cycle} />
    </main>
  );
}
