export type BillingCycle = "monthly" | "annual";

export type PlanId = "daily" | "pro" | "diagnosis";

export type Plan = {
  id: PlanId;
  name: string;
  description: string;
  monthlyInr: number | null;
  annualInr: number | null;
  displayMonthly: string;
  displayAnnual: string;
};

export const plans: Record<PlanId, Plan> = {
  daily: {
    id: "daily",
    name: "Core Research",
    description:
      "Equity, options, and index research briefs with archive access. " +
      "Live intraday updates are not included.",
    monthlyInr: 6999,
    annualInr: 69999,
    displayMonthly: "₹6,999/mo",
    displayAnnual: "₹69,999/yr",
  },
  pro: {
    id: "pro",
    name: "Live Desk",
    description:
      "Core Research plus live update delivery, full archive, " +
      "calibration dashboard, and priority support.",
    monthlyInr: 14999,
    annualInr: 149999,
    displayMonthly: "₹14,999/mo",
    displayAnnual: "₹149,999/yr",
  },
  diagnosis: {
    id: "diagnosis",
    name: "Audit Base",
    description:
      "One-off research or infrastructure audit. Starts with a base " +
      "scope and expands by infrastructure size and data complexity.",
    monthlyInr: null,
    annualInr: null,
    displayMonthly: "From ₹2,000",
    displayAnnual: "Base + scope",
  },
};

export function parsePlanId(value: unknown): PlanId | null {
  if (value === "daily" || value === "pro" || value === "diagnosis") {
    return value;
  }
  return null;
}

export function parseBillingCycle(value: unknown): BillingCycle {
  return value === "annual" ? "annual" : "monthly";
}

export function amountForPlan(
  planId: PlanId,
  cycle: BillingCycle,
): number | null {
  const plan = plans[planId];
  return cycle === "annual" ? plan.annualInr : plan.monthlyInr;
}
