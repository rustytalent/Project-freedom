import { createHmac } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { POST as verifyCheckout } from "@/app/api/checkout/razorpay-verify/route";
import {
  accessTierForPlan,
  verifyCheckoutSignature,
  verifyWebhookSignature,
} from "@/lib/razorpay";

const ORIGINAL_ENV = { ...process.env };

function hmac(secret: string, payload: string): string {
  return createHmac("sha256", secret).update(payload).digest("hex");
}

describe("Razorpay helpers", () => {
  it("verifies checkout signatures", () => {
    const signature = hmac("secret", "order_123|pay_123");
    expect(
      verifyCheckoutSignature({
        orderId: "order_123",
        paymentId: "pay_123",
        signature,
        keySecret: "secret",
      }),
    ).toBe(true);
    expect(
      verifyCheckoutSignature({
        orderId: "order_123",
        paymentId: "pay_456",
        signature,
        keySecret: "secret",
      }),
    ).toBe(false);
  });

  it("verifies webhook signatures", () => {
    const body = JSON.stringify({ event: "payment.captured" });
    const signature = hmac("webhook-secret", body);
    expect(
      verifyWebhookSignature({
        body,
        signature,
        webhookSecret: "webhook-secret",
      }),
    ).toBe(true);
  });

  it("maps checkout plans to access tiers", () => {
    expect(accessTierForPlan("daily")).toBe("paid_intraday");
    expect(accessTierForPlan("pro")).toBe("paid_multi_product");
  });
});

describe("checkout verification route", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    process.env = {
      ...ORIGINAL_ENV,
      RAZORPAY_KEY_ID: "rzp_test_key",
      RAZORPAY_KEY_SECRET: "secret",
      NEXT_PUBLIC_SUPABASE_URL: "https://db.example",
      SUPABASE_SERVICE_ROLE_KEY: "service-role",
    };
  });

  afterEach(() => {
    process.env = { ...ORIGINAL_ENV };
    vi.restoreAllMocks();
  });

  it("verifies captured payments and upserts the subscriber", async () => {
    const signature = hmac("secret", "order_123|pay_123");
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: "order_123",
            amount: 699900,
            currency: "INR",
            status: "paid",
            notes: {
              plan: "daily",
              cycle: "monthly",
              email: "reader@example.com",
            },
          }),
          { status: 200 },
        ),
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: "pay_123",
            order_id: "order_123",
            amount: 699900,
            currency: "INR",
            status: "captured",
            captured: true,
            email: "reader@example.com",
          }),
          { status: 200 },
        ),
      )
      .mockResolvedValueOnce(new Response("", { status: 201 }));
    vi.stubGlobal("fetch", fetchMock);

    const response = await verifyCheckout(
      new Request("https://example.test/api/checkout/razorpay-verify", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          razorpay_order_id: "order_123",
          razorpay_payment_id: "pay_123",
          razorpay_signature: signature,
        }),
      }),
    );

    const payload = (await response.json()) as {
      verified: boolean;
      stored: boolean;
      tier: string;
    };
    expect(response.status).toBe(200);
    expect(payload).toMatchObject({
      verified: true,
      stored: true,
      tier: "paid_intraday",
    });
    expect(fetchMock).toHaveBeenCalledTimes(3);
    const [, supabaseInit] = fetchMock.mock.calls[2] as unknown as [
      string,
      RequestInit,
    ];
    expect(JSON.parse(String(supabaseInit.body))).toMatchObject({
      email: "reader@example.com",
      tier: "paid_intraday",
      plan_id: "daily",
      status: "active",
    });
  });
});
