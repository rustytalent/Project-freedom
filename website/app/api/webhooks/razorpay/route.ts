import { NextResponse } from "next/server";
import {
  planFromOrderNotes,
  syncSubscriberFromPayment,
  type RazorpayOrder,
  type RazorpayPayment,
  verifyWebhookSignature,
} from "@/lib/razorpay";

export const runtime = "nodejs";

type RazorpayWebhook = {
  event?: string;
  payload?: {
    order?: { entity?: RazorpayOrder };
    payment?: { entity?: RazorpayPayment };
  };
};

function webhookSecret(): string | null {
  return process.env.RAZORPAY_WEBHOOK_SECRET || null;
}

export async function POST(req: Request): Promise<NextResponse> {
  const secret = webhookSecret();
  if (!secret) {
    return NextResponse.json(
      { error: "razorpay_webhook_not_configured" },
      { status: 503 },
    );
  }

  const signature = req.headers.get("x-razorpay-signature") ?? "";
  const body = await req.text();
  if (
    !signature ||
    !verifyWebhookSignature({
      body,
      signature,
      webhookSecret: secret,
    })
  ) {
    return NextResponse.json({ error: "invalid_signature" }, { status: 400 });
  }

  let event: RazorpayWebhook;
  try {
    event = JSON.parse(body) as RazorpayWebhook;
  } catch {
    return NextResponse.json({ error: "invalid_json" }, { status: 400 });
  }

  const payment = event.payload?.payment?.entity;
  const order = event.payload?.order?.entity;
  if (!payment && !order) {
    return NextResponse.json({ ok: true, ignored: "no_payment_or_order" });
  }

  if (event.event === "payment.failed") {
    const email = (payment?.email || payment?.notes?.email || "").trim().toLowerCase();
    const plan = payment?.notes?.plan === "pro" ? "pro" : "daily";
    const cycle = payment?.notes?.cycle === "annual" ? "annual" : "monthly";
    const sync = email
      ? await syncSubscriberFromPayment({
          email,
          planId: plan,
          cycle,
          status: "payment_failed",
          razorpayOrderId: payment?.order_id,
          razorpayPaymentId: payment?.id,
        })
      : { stored: false, reason: "missing_customer_email" };
    return NextResponse.json({ ok: true, event: event.event, sync });
  }

  if (
    event.event !== "payment.captured" &&
    event.event !== "order.paid"
  ) {
    return NextResponse.json({ ok: true, ignored: event.event ?? "unknown" });
  }

  const orderLike = order || {
    id: payment?.order_id ?? "",
    amount: payment?.amount ?? 0,
    currency: payment?.currency ?? "INR",
    notes: payment?.notes,
  };
  const { planId, cycle, email: noteEmail } = planFromOrderNotes(orderLike);
  if (!planId) {
    return NextResponse.json({ ok: false, error: "missing_order_plan" }, { status: 202 });
  }

  const email = (payment?.email || noteEmail).trim().toLowerCase();
  if (!email) {
    return NextResponse.json({ ok: false, error: "missing_customer_email" }, { status: 202 });
  }

  const sync = await syncSubscriberFromPayment({
    email,
    planId,
    cycle,
    status: "active",
    razorpayOrderId: orderLike.id || payment?.order_id,
    razorpayPaymentId: payment?.id,
  });

  return NextResponse.json({
    ok: true,
    event: event.event,
    stored: sync.stored,
    storage_reason: sync.reason,
  });
}
