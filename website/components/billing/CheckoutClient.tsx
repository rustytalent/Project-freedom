"use client";

import { useMemo, useState } from "react";
import { plans, type BillingCycle, type PlanId } from "@/lib/pricing";
import { cn } from "@/lib/utils";

type RazorpayOrder = {
  id: string;
  amount: number;
  currency: string;
};

type RazorpayConstructor = new (options: {
  key: string;
  amount: number;
  currency: string;
  name: string;
  description: string;
  order_id: string;
  prefill: { email: string };
  theme: { color: string };
}) => { open: () => void };

declare global {
  interface Window {
    Razorpay?: RazorpayConstructor;
  }
}

const checkoutablePlans: PlanId[] = ["daily", "pro"];

function amountLabel(planId: PlanId, cycle: BillingCycle): string {
  const plan = plans[planId];
  return cycle === "annual" ? plan.displayAnnual : plan.displayMonthly;
}

async function ensureRazorpayScript(): Promise<boolean> {
  if (window.Razorpay) return true;
  const existing = document.querySelector<HTMLScriptElement>(
    'script[src="https://checkout.razorpay.com/v1/checkout.js"]',
  );
  if (existing) {
    await new Promise((resolve) => {
      existing.addEventListener("load", resolve, { once: true });
      existing.addEventListener("error", resolve, { once: true });
    });
    return Boolean(window.Razorpay);
  }

  const script = document.createElement("script");
  script.src = "https://checkout.razorpay.com/v1/checkout.js";
  script.async = true;
  document.body.appendChild(script);
  await new Promise((resolve) => {
    script.addEventListener("load", resolve, { once: true });
    script.addEventListener("error", resolve, { once: true });
  });
  return Boolean(window.Razorpay);
}

export function CheckoutClient({
  initialPlan,
  initialCycle,
}: {
  initialPlan: PlanId;
  initialCycle: BillingCycle;
}) {
  const [planId, setPlanId] = useState<PlanId>(
    initialPlan === "diagnosis" ? "daily" : initialPlan,
  );
  const [cycle, setCycle] = useState<BillingCycle>(initialCycle);
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const selectedPlan = plans[planId];
  const payableLabel = useMemo(
    () => amountLabel(planId, cycle),
    [planId, cycle],
  );

  async function beginCheckout() {
    setBusy(true);
    setStatus(null);
    try {
      const scriptReady = await ensureRazorpayScript();
      if (!scriptReady || !window.Razorpay) {
        setStatus("Payment script could not load. Please retry.");
        return;
      }

      const response = await fetch("/api/checkout/razorpay-order", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ plan: planId, cycle, email }),
      });
      const payload = (await response.json()) as {
        key_id?: string;
        order?: RazorpayOrder;
        error?: string;
      };

      if (!response.ok || !payload.key_id || !payload.order) {
        setStatus(
          payload.error === "razorpay_not_configured"
            ? "Checkout is not configured on this deployment yet."
            : "Could not create checkout order. Please retry.",
        );
        return;
      }

      const checkout = new window.Razorpay({
        key: payload.key_id,
        amount: payload.order.amount,
        currency: payload.order.currency,
        name: "Crux Research",
        description: `${selectedPlan.name} ${cycle}`,
        order_id: payload.order.id,
        prefill: { email },
        theme: { color: "#D6B56D" },
      });
      checkout.open();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
      <section className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          {checkoutablePlans.map((id) => {
            const p = plans[id];
            const active = id === planId;
            return (
              <button
                type="button"
                key={id}
                onClick={() => setPlanId(id)}
                className={cn(
                  "text-left border border-border bg-bg-raised p-5 rounded-sm " +
                    "transition-colors hover:border-accent",
                  active && "border-accent",
                )}
              >
                <span className="block font-serif text-2xl text-fg">
                  {p.name}
                </span>
                <span className="block text-sm text-fg-muted mt-2 leading-relaxed">
                  {p.description}
                </span>
                <span className="block font-mono text-xl text-fg mt-6">
                  {amountLabel(id, cycle)}
                </span>
              </button>
            );
          })}
        </div>

        <div className="border border-border bg-bg-raised p-5 rounded-sm">
          <p className="text-xs uppercase tracking-wider text-fg-subtle mb-3">
            Billing cycle
          </p>
          <div className="inline-flex border border-border rounded-sm overflow-hidden">
            {(["monthly", "annual"] as const).map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => setCycle(value)}
                className={cn(
                  "px-4 py-2 text-sm transition-colors",
                  cycle === value
                    ? "bg-accent text-bg"
                    : "text-fg-muted hover:text-fg",
                )}
              >
                {value === "monthly" ? "Monthly" : "Annual"}
              </button>
            ))}
          </div>
          <p className="text-xs text-fg-subtle mt-3">
            Annual plans are priced to give roughly two months free.
          </p>
        </div>
      </section>

      <aside className="border border-border bg-bg-raised p-6 rounded-sm h-fit">
        <p className="text-xs uppercase tracking-wider text-accent mb-4">
          Checkout
        </p>
        <h2 className="font-serif text-3xl text-fg">{selectedPlan.name}</h2>
        <p className="text-fg-muted mt-3 leading-relaxed">
          {selectedPlan.description}
        </p>
        <p className="font-mono text-3xl text-fg mt-8">{payableLabel}</p>
        <label className="block mt-8">
          <span className="text-xs uppercase tracking-wider text-fg-subtle">
            Email for account access
          </span>
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@example.com"
            className="mt-2 w-full border border-border bg-bg px-3 py-2 text-sm text-fg outline-none focus:border-accent"
          />
        </label>
        <button
          type="button"
          disabled={busy || !email}
          onClick={beginCheckout}
          className="mt-6 w-full inline-flex items-center justify-center px-5 py-2.5 text-sm font-medium transition-colors duration-150 rounded-sm bg-accent text-bg hover:bg-accent-glow disabled:opacity-50 disabled:pointer-events-none"
        >
          {busy ? "Preparing checkout" : "Pay with Razorpay"}
        </button>
        {status && (
          <p className="mt-4 text-sm text-fg-muted leading-relaxed">{status}</p>
        )}
        <p className="mt-6 text-xs text-fg-subtle leading-relaxed">
          Access is issued to the same email after payment confirmation.
          Production entitlement updates are handled by the Razorpay
          webhook and subscriber database.
        </p>
      </aside>
    </div>
  );
}
