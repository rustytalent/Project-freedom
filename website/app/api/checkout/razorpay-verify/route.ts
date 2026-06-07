import { NextResponse } from "next/server";
import { amountForPlan } from "@/lib/pricing";
import {
  fetchRazorpayOrder,
  fetchRazorpayPayment,
  getRazorpayConfig,
  planFromOrderNotes,
  syncSubscriberFromPayment,
  verifyCheckoutSignature,
} from "@/lib/razorpay";

export const runtime = "nodejs";

type VerifyRequest = {
  razorpay_order_id?: unknown;
  razorpay_payment_id?: unknown;
  razorpay_signature?: unknown;
};

function asString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

export async function POST(req: Request): Promise<NextResponse> {
  let body: VerifyRequest;
  try {
    body = (await req.json()) as VerifyRequest;
  } catch {
    return NextResponse.json({ error: "invalid_json" }, { status: 400 });
  }

  const orderId = asString(body.razorpay_order_id);
  const paymentId = asString(body.razorpay_payment_id);
  const signature = asString(body.razorpay_signature);
  if (!orderId || !paymentId || !signature) {
    return NextResponse.json({ error: "missing_payment_fields" }, { status: 400 });
  }

  const config = getRazorpayConfig();
  if (!config) {
    return NextResponse.json(
      { error: "razorpay_not_configured" },
      { status: 503 },
    );
  }

  if (
    !verifyCheckoutSignature({
      orderId,
      paymentId,
      signature,
      keySecret: config.keySecret,
    })
  ) {
    return NextResponse.json({ error: "invalid_signature" }, { status: 400 });
  }

  const [order, payment] = await Promise.all([
    fetchRazorpayOrder(config, orderId),
    fetchRazorpayPayment(config, paymentId),
  ]);

  if (payment.order_id !== orderId) {
    return NextResponse.json({ error: "payment_order_mismatch" }, { status: 400 });
  }

  if (payment.status !== "captured" && payment.captured !== true) {
    return NextResponse.json(
      { error: "payment_not_captured", payment_status: payment.status },
      { status: 202 },
    );
  }

  const { planId, cycle, email: noteEmail } = planFromOrderNotes(order);
  if (!planId) {
    return NextResponse.json({ error: "missing_order_plan" }, { status: 400 });
  }

  const expectedAmountInr = amountForPlan(planId, cycle);
  if (!expectedAmountInr || order.amount !== expectedAmountInr * 100) {
    return NextResponse.json({ error: "order_amount_mismatch" }, { status: 400 });
  }

  const email = (payment.email || noteEmail).trim().toLowerCase();
  if (!email) {
    return NextResponse.json({ error: "missing_customer_email" }, { status: 400 });
  }

  const sync = await syncSubscriberFromPayment({
    email,
    planId,
    cycle,
    status: "active",
    razorpayOrderId: orderId,
    razorpayPaymentId: paymentId,
    razorpaySignature: signature,
  });

  return NextResponse.json({
    verified: true,
    stored: sync.stored,
    storage_reason: sync.reason,
    tier: planId === "pro" ? "paid_multi_product" : "paid_intraday",
  });
}
