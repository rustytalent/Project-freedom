import type { Metadata } from "next";
import { LegalPage } from "@/components/marketing/LegalPage";
import { brand } from "@/lib/brand";

export const metadata: Metadata = {
  title: "Terms of Service",
  description: "Terms of Service for the website and the subscription products.",
};

export default function TermsPage() {
  return (
    <LegalPage
      eyebrow="Legal · Terms of Service"
      title="Terms of Service"
      lastUpdated="To be confirmed before launch"
    >
      <p>
        <strong>Placeholder - finalise with counsel before public launch.</strong>{" "}
        These terms govern your use of {brand.name} and any
        subscription products offered through it.
      </p>

      <h2 className="text-xl text-fg font-serif">1. Acceptance</h2>
      <p>
        By creating an account, subscribing to any product, or
        otherwise using this website, you agree to be bound by these
        Terms of Service and our Privacy Policy.
      </p>

      <h2 className="text-xl text-fg font-serif">2. Subscription products</h2>
      <p>
        Active subscriptions grant access to the corresponding equity,
        options, and index research artifacts for the period paid. Live
        update delivery is included only in plans that explicitly list
        live updates. Subscriptions renew automatically at the start of
        each billing period unless cancelled in the subscriber portal
        at least 24 hours before renewal.
      </p>

      <h2 className="text-xl text-fg font-serif">3. Refunds</h2>
      <p>
        Monthly subscriptions are not refundable beyond the first 24
        hours after purchase. Annual subscriptions are refundable
        pro-rata within 14 days of billing.
      </p>

      <h2 className="text-xl text-fg font-serif">4. Acceptable use</h2>
      <p>
        Subscription content is licensed for personal use by the
        named subscriber. Bulk redistribution, public republishing,
        and resale are prohibited. Sharing of a single subscriber
        account across multiple individuals may result in suspension.
      </p>

      <h2 className="text-xl text-fg font-serif">5. No advice</h2>
      <p>
        See the{" "}
        <a href="/legal/disclosures" className="text-accent underline underline-offset-4">
          Disclosures
        </a>{" "}
        page. Nothing on this site is personal investment advice. All
        decisions to act on published research rest entirely with the
        reader.
      </p>

      <h2 className="text-xl text-fg font-serif">6. Liability</h2>
      <p>
        To the maximum extent permitted by Indian law, {brand.name},
        {brand.legalEntity}, and our affiliates shall not be liable
        for any direct, indirect, incidental, or consequential
        damages arising out of or in connection with the use of, or
        reliance on, any content published or made available through
        this service.
      </p>

      <h2 className="text-xl text-fg font-serif">7. Jurisdiction</h2>
      <p>
        These terms are governed by the laws of India. Any dispute
        arising in connection with this service shall be subject to
        the exclusive jurisdiction of the courts of (city to be
        confirmed by counsel).
      </p>

      <h2 className="text-xl text-fg font-serif">8. Changes</h2>
      <p>
        We may update these terms from time to time. Material changes
        will be emailed to active subscribers and noted on this page.
      </p>
    </LegalPage>
  );
}
