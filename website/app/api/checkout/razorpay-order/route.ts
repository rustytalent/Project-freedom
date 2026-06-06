import { NextResponse } from "next/server";
import {
  amountForPlan,
  parseBillingCycle,
  parsePlanId,
  plans,
  type BillingCycle,
  type PlanId,
} from "@/lib/pricing";

export const runtime = "edge";

type CheckoutRequest = {
  plan?: unknown;
  cycle?: unknown;
  email?: unknown;
};

function razorpayConfig():
  | { keyId: string; keySecret: string }
  | null {
  const keyId = process.env.RAZORPAY_KEY_ID;
  const keySecret = process.env.RAZORPAY_KEY_SECRET;
  if (!keyId || !keySecret) return null;
  return { keyId, keySecret };
}

function encodeBasicAuth(keyId: string, keySecret: string): string {
  const raw = `${keyId}:${keySecret}`;
  return btoa(raw);
}

function receiptId(plan: PlanId, cycle: BillingCycle): string {
  const suffix = Math.random().toString(36).slice(2, 10);
  return `${plan}_${cycle}_${Date.now()}_${suffix}`.slice(0, 40);
}

export async function POST(req: Request): Promise<NextResponse> {
  let body: CheckoutRequest;
  try {
    body = (await req.json()) as CheckoutRequest;
  } catch {
    return NextResponse.json({ error: "invalid_json" }, { status: 400 });
  }

  const planId = parsePlanId(body.plan);
  if (!planId || planId === "diagnosis") {
    return NextResponse.json({ error: "invalid_plan" }, { status: 400 });
  }

  const cycle = parseBillingCycle(body.cycle);
  const amountInr = amountForPlan(planId, cycle);
  if (!amountInr) {
    return NextResponse.json({ error: "plan_not_checkoutable" }, { status: 400 });
  }

  const config = razorpayConfig();
  if (!config) {
    return NextResponse.json(
      { error: "razorpay_not_configured" },
      { status: 503 },
    );
  }

  const response = await fetch("https://api.razorpay.com/v1/orders", {
    method: "POST",
    headers: {
      authorization: `Basic ${encodeBasicAuth(config.keyId, config.keySecret)}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      amount: amountInr * 100,
      currency: "INR",
      receipt: receiptId(planId, cycle),
      notes: {
        plan: planId,
        plan_name: plans[planId].name,
        cycle,
        email: typeof body.email === "string" ? body.email : "",
      },
    }),
  });

  const payload = (await response.json().catch(() => null)) as unknown;
  if (!response.ok) {
    return NextResponse.json(
      {
        error: "razorpay_order_failed",
        status: response.status,
        details: payload,
      },
      { status: 502 },
    );
  }

  return NextResponse.json({
    key_id: config.keyId,
    order: payload,
    plan: plans[planId],
    cycle,
  });
}
