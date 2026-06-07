import { createHmac, timingSafeEqual } from "node:crypto";
import type { AccessTier } from "@/lib/artifacts";
import {
  amountForPlan,
  plans,
  type BillingCycle,
  type PlanId,
} from "@/lib/pricing";

export type CheckoutPlanId = Exclude<PlanId, "diagnosis">;

export type RazorpayConfig = {
  keyId: string;
  keySecret: string;
};

export type RazorpayOrder = {
  id: string;
  amount: number;
  currency: string;
  receipt?: string;
  status?: string;
  notes?: Record<string, string | undefined>;
};

export type RazorpayPayment = {
  id: string;
  amount: number;
  currency: string;
  order_id?: string;
  status?: string;
  captured?: boolean;
  email?: string;
  contact?: string;
  notes?: Record<string, string | undefined>;
};

export type SubscriberSyncInput = {
  email: string;
  planId: CheckoutPlanId;
  cycle: BillingCycle;
  status: "active" | "payment_failed" | "payment_pending";
  razorpayOrderId?: string;
  razorpayPaymentId?: string;
  razorpaySignature?: string;
};

export function getRazorpayConfig(): RazorpayConfig | null {
  const keyId = process.env.RAZORPAY_KEY_ID;
  const keySecret = process.env.RAZORPAY_KEY_SECRET;
  if (!keyId || !keySecret) return null;
  return { keyId, keySecret };
}

export function basicAuth(config: RazorpayConfig): string {
  return Buffer.from(`${config.keyId}:${config.keySecret}`).toString("base64");
}

export function isCheckoutPlan(planId: PlanId | null): planId is CheckoutPlanId {
  return planId === "daily" || planId === "pro";
}

export function accessTierForPlan(planId: CheckoutPlanId): AccessTier {
  return planId === "pro" ? "paid_multi_product" : "paid_intraday";
}

export function receiptId(plan: CheckoutPlanId, cycle: BillingCycle): string {
  const suffix = Math.random().toString(36).slice(2, 10);
  return `${plan}_${cycle}_${Date.now()}_${suffix}`.slice(0, 40);
}

function hmacHex(secret: string, payload: string): string {
  return createHmac("sha256", secret).update(payload).digest("hex");
}

function safeEqualHex(a: string, b: string): boolean {
  const left = Buffer.from(a, "hex");
  const right = Buffer.from(b, "hex");
  if (left.length !== right.length) return false;
  return timingSafeEqual(left, right);
}

export function verifyCheckoutSignature(input: {
  orderId: string;
  paymentId: string;
  signature: string;
  keySecret: string;
}): boolean {
  const expected = hmacHex(
    input.keySecret,
    `${input.orderId}|${input.paymentId}`,
  );
  return safeEqualHex(expected, input.signature);
}

export function verifyWebhookSignature(input: {
  body: string;
  signature: string;
  webhookSecret: string;
}): boolean {
  const expected = hmacHex(input.webhookSecret, input.body);
  return safeEqualHex(expected, input.signature);
}

export async function createRazorpayOrder(input: {
  config: RazorpayConfig;
  planId: CheckoutPlanId;
  cycle: BillingCycle;
  email: string;
}): Promise<RazorpayOrder> {
  const amountInr = amountForPlan(input.planId, input.cycle);
  if (!amountInr) {
    throw new Error("plan_not_checkoutable");
  }

  const response = await fetch("https://api.razorpay.com/v1/orders", {
    method: "POST",
    headers: {
      authorization: `Basic ${basicAuth(input.config)}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      amount: amountInr * 100,
      currency: "INR",
      receipt: receiptId(input.planId, input.cycle),
      notes: {
        plan: input.planId,
        plan_name: plans[input.planId].name,
        cycle: input.cycle,
        email: input.email,
      },
    }),
  });

  const payload = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    const err = new Error("razorpay_order_failed");
    (err as Error & { status?: number; details?: unknown }).status =
      response.status;
    (err as Error & { status?: number; details?: unknown }).details = payload;
    throw err;
  }

  return payload as RazorpayOrder;
}

export async function fetchRazorpayOrder(
  config: RazorpayConfig,
  orderId: string,
): Promise<RazorpayOrder> {
  const response = await fetch(
    `https://api.razorpay.com/v1/orders/${encodeURIComponent(orderId)}`,
    {
      headers: {
        authorization: `Basic ${basicAuth(config)}`,
      },
    },
  );
  const payload = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    const err = new Error("razorpay_order_fetch_failed");
    (err as Error & { status?: number; details?: unknown }).status =
      response.status;
    (err as Error & { status?: number; details?: unknown }).details = payload;
    throw err;
  }
  return payload as RazorpayOrder;
}

export async function fetchRazorpayPayment(
  config: RazorpayConfig,
  paymentId: string,
): Promise<RazorpayPayment> {
  const response = await fetch(
    `https://api.razorpay.com/v1/payments/${encodeURIComponent(paymentId)}`,
    {
      headers: {
        authorization: `Basic ${basicAuth(config)}`,
      },
    },
  );
  const payload = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    const err = new Error("razorpay_payment_fetch_failed");
    (err as Error & { status?: number; details?: unknown }).status =
      response.status;
    (err as Error & { status?: number; details?: unknown }).details = payload;
    throw err;
  }
  return payload as RazorpayPayment;
}

export function planFromOrderNotes(order: RazorpayOrder): {
  planId: CheckoutPlanId | null;
  cycle: BillingCycle;
  email: string;
} {
  const rawPlan = order.notes?.plan;
  const planId =
    rawPlan === "daily" || rawPlan === "pro" ? rawPlan : null;
  const cycle = order.notes?.cycle === "annual" ? "annual" : "monthly";
  const email = String(order.notes?.email ?? "").trim().toLowerCase();
  return { planId, cycle, email };
}

export async function syncSubscriberFromPayment(
  input: SubscriberSyncInput,
): Promise<{ stored: boolean; reason?: string }> {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL?.replace(/\/+$/, "");
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const table = process.env.SUPABASE_SUBSCRIBERS_TABLE || "subscribers";
  if (!supabaseUrl || !serviceRoleKey) {
    return { stored: false, reason: "supabase_env_missing" };
  }

  const now = new Date().toISOString();
  const response = await fetch(
    `${supabaseUrl}/rest/v1/${table}?on_conflict=email`,
    {
      method: "POST",
      headers: {
        apikey: serviceRoleKey,
        authorization: `Bearer ${serviceRoleKey}`,
        "content-type": "application/json",
        prefer: "resolution=merge-duplicates,return=minimal",
      },
      body: JSON.stringify({
        email: input.email,
        tier: accessTierForPlan(input.planId),
        plan_id: input.planId,
        billing_cycle: input.cycle,
        status: input.status,
        razorpay_order_id: input.razorpayOrderId ?? null,
        razorpay_payment_id: input.razorpayPaymentId ?? null,
        razorpay_signature: input.razorpaySignature ?? null,
        paid_at: input.status === "active" ? now : null,
        updated_at: now,
      }),
    },
  );

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    return {
      stored: false,
      reason: text || response.statusText || "supabase_write_failed",
    };
  }
  return { stored: true };
}
