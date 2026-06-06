import type { Metadata } from "next";
import { LegalPage } from "@/components/marketing/LegalPage";
import { brand } from "@/lib/brand";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "How we handle subscriber and prospect data.",
};

export default function PrivacyPage() {
  return (
    <LegalPage
      eyebrow="Legal · Privacy"
      title="Privacy Policy"
      lastUpdated="To be confirmed before launch"
    >
      <p>
        <strong>Placeholder - finalise with counsel before public launch.</strong>{" "}
        This page sets out how {brand.name} collects, uses, and
        protects personal data.
      </p>

      <h2 className="text-xl text-fg font-serif">Data we collect</h2>
      <ul>
        <li>
          Email address - required for account creation, brief
          delivery, and billing.
        </li>
        <li>
          Subscription state - your tier, billing cycle, and renewal
          status, held in our database for the duration of the
          subscription.
        </li>
        <li>
          Aggregate analytics - page views and brief read-rate, using
          a privacy-respecting analytics provider (Plausible). We do
          not place third-party advertising cookies and we do not run
          Google Analytics.
        </li>
        <li>
          Communication content - emails you send us, kept in our
          support inbox for the purposes of responding to you.
        </li>
      </ul>

      <h2 className="text-xl text-fg font-serif">Data we do not collect</h2>
      <ul>
        <li>
          Phone number (unless you voluntarily provide it for the
          Diagnosis intake call).
        </li>
        <li>Identity documents (KYC). We are not an Investment Adviser; KYC is not required.</li>
        <li>Broker / demat account credentials. Ever.</li>
        <li>Your trading positions or P&amp;L.</li>
      </ul>

      <h2 className="text-xl text-fg font-serif">Payments</h2>
      <p>
        Payments are processed by Razorpay. We do not store card
        numbers, CVV, UPI handles, or other payment credentials on
        our servers. We retain a token from Razorpay sufficient to
        manage subscription renewal and refund.
      </p>

      <h2 className="text-xl text-fg font-serif">Diagnosis-product data</h2>
      <p>
        If you engage the Diagnosis product, the trade log you submit
        is held under NDA on encrypted storage, used only for the
        purpose of producing your report, and deleted within 90 days
        of report delivery unless you request retention.
      </p>

      <h2 className="text-xl text-fg font-serif">Third-party processors</h2>
      <ul>
        <li>Supabase - subscriber database hosting.</li>
        <li>Razorpay - payment processing.</li>
        <li>Resend - transactional and brief-delivery email.</li>
        <li>Sentry - error tracking. We strip personal data from error reports.</li>
        <li>Plausible - privacy-respecting page-view analytics.</li>
      </ul>

      <h2 className="text-xl text-fg font-serif">Your rights</h2>
      <p>
        You can request a copy of your data, request its deletion, or
        cancel your subscription at any time. Email{" "}
        <a
          href={`mailto:${brand.contact.email}`}
          className="text-accent underline underline-offset-4"
        >
          {brand.contact.email}
        </a>{" "}
        and we will respond within seven business days.
      </p>

      <h2 className="text-xl text-fg font-serif">Cookies</h2>
      <p>
        We use a single session cookie for authentication on the
        subscriber portal. No marketing, retargeting, or analytics
        cookies are set.
      </p>

      <h2 className="text-xl text-fg font-serif">Changes</h2>
      <p>
        We may update this policy from time to time. Material
        changes will be emailed to active subscribers and noted on
        this page.
      </p>
    </LegalPage>
  );
}
