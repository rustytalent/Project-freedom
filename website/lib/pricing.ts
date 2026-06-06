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
    name: "Daily Brief",
    description:
      "Equity research brief before open, subscriber-only levels, " +
      "yesterday audit, and a thirty-day archive.",
    monthlyInr: 4999,
    annualInr: 49999,
    displayMonthly: "₹4,999/mo",
    displayAnnual: "₹49,999/yr",
  },
  pro: {
    id: "pro",
    name: "Pro Desk",
    description:
      "Daily Brief plus full archive, pro calibration dashboard, " +
      "priority support, and options-ready context when enabled.",
    monthlyInr: 9999,
    annualInr: 99999,
    displayMonthly: "₹9,999/mo",
    displayAnnual: "₹99,999/yr",
  },
  diagnosis: {
    id: "diagnosis",
    name: "Strategy Diagnosis",
    description:
      "One-off strategy audit with cost sensitivity, failure modes, " +
      "and a clear ship, re-scope, or shelve recommendation.",
    monthlyInr: null,
    annualInr: null,
    displayMonthly: "From ₹49,999",
    displayAnnual: "Per audit",
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
