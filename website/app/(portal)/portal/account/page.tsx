import type { Metadata } from "next";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: "Account",
};

export default function AccountPage() {
  return (
    <div className="max-w-prose mx-auto px-6 py-14">
      <header className="mb-10">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-3">
          Account
        </p>
        <h1 className="font-serif text-3xl text-fg">Your subscription.</h1>
      </header>

      <div className="space-y-6">
        <Card>
          <CardTitle>Plan</CardTitle>
          <CardDescription>Pro Desk · Monthly · ₹9,999</CardDescription>
          <CardContent className="mt-4 text-sm text-fg-muted">
            <p>Includes Daily Brief, full archive, and pro calibration view.</p>
            <p className="mt-2">Next renewal on 2026-07-04.</p>
            <div className="mt-6 flex gap-3">
              <button
                type="button"
                className="text-sm text-accent underline underline-offset-4"
              >
                Switch to annual (save 15%)
              </button>
              <button
                type="button"
                className="text-sm text-fg-muted underline underline-offset-4"
              >
                Cancel subscription
              </button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardTitle>Delivery preferences</CardTitle>
          <CardContent className="mt-4 text-sm text-fg-muted space-y-3">
            <p>Email: demo@example.in</p>
            <p>
              Daily Brief delivery:{" "}
              <span className="text-fg">Email + portal</span>
            </p>
            <p>
              Swing Brief delivery:{" "}
              <span className="text-fg">Email + portal</span>
            </p>
            <p>
              Drift alerts:{" "}
              <span className="text-fg">On (email)</span>
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardTitle>Billing</CardTitle>
          <CardContent className="mt-4 text-sm text-fg-muted">
            <p>Razorpay payment method on file.</p>
            <Link
              href="/portal/account/billing"
              className="block mt-4 text-sm text-accent underline underline-offset-4"
            >
              Open billing
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardTitle>Data &amp; privacy</CardTitle>
          <CardContent className="mt-4 text-sm text-fg-muted">
            <p>
              Email{" "}
              <a
                href="mailto:hello@aurora-research.in"
                className="text-accent underline underline-offset-4"
              >
                hello@aurora-research.in
              </a>{" "}
              to export or delete your data. We respond within seven
              business days.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
