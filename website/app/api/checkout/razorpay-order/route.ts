import { NextResponse } from "next/server";
import {
  parseBillingCycle,
  parsePlanId,
  plans,
} from "@/lib/pricing";
import {
  createRazorpayOrder,
  getRazorpayConfig,
  isCheckoutPlan,
} from "@/lib/razorpay";

export const runtime = "nodejs";

type CheckoutRequest = {
  plan?: unknown;
  cycle?: unknown;
  email?: unknown;
};

export async function POST(req: Request): Promise<NextResponse> {
  let body: CheckoutRequest;
  try {
    body = (await req.json()) as CheckoutRequest;
  } catch {
    return NextResponse.json({ error: "invalid_json" }, { status: 400 });
  }

  const planId = parsePlanId(body.plan);
  if (!isCheckoutPlan(planId)) {
    return NextResponse.json({ error: "invalid_plan" }, { status: 400 });
  }

  const cycle = parseBillingCycle(body.cycle);
  const email = typeof body.email === "string" ? body.email.trim() : "";

  const config = getRazorpayConfig();
  if (!config) {
    return NextResponse.json(
      { error: "razorpay_not_configured" },
      { status: 503 },
    );
  }

  try {
    const order = await createRazorpayOrder({
      config,
      planId,
      cycle,
      email,
    });

    return NextResponse.json({
      key_id: config.keyId,
      order,
      plan: plans[planId],
      cycle,
    });
  } catch (error) {
    const enriched = error as Error & { status?: number; details?: unknown };
    return NextResponse.json(
      {
        error: enriched.message || "razorpay_order_failed",
        status: enriched.status,
        details: enriched.details,
      },
      { status: 502 },
    );
  }
}
