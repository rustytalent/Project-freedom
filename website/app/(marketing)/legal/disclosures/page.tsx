import type { Metadata } from "next";
import { LegalPage } from "@/components/marketing/LegalPage";
import { brand } from "@/lib/brand";

export const metadata: Metadata = {
  title: "Disclosures",
  description: "SEBI safe-harbour disclosures and regulatory framing.",
};

export default function DisclosuresPage() {
  return (
    <LegalPage
      eyebrow="Legal · Disclosures"
      title="Regulatory disclosures"
      lastUpdated="To be confirmed before launch"
    >
      <p>
        <strong>Placeholder - finalise with counsel before public launch.</strong>{" "}
        This page sets out the regulatory framing that applies to{" "}
        {brand.name}&rsquo;s published research.
      </p>

      <h2 className="text-xl text-fg font-serif">Not an investment adviser</h2>
      <p>
        {brand.name} (the {brand.legalEntity}) is{" "}
        <strong>not registered with the Securities and Exchange Board of
        India (SEBI) as an Investment Adviser</strong> under the SEBI
        (Investment Advisers) Regulations, 2013, and is not registered
        as a Research Analyst under the SEBI (Research Analysts)
        Regulations, 2014.
      </p>
      <p>
        The content published by {brand.name} is general market
        commentary and educational research only. It is not personal
        investment advice, a recommendation to buy or sell any
        security, or a solicitation of any kind.
      </p>

      <h2 className="text-xl text-fg font-serif">Not investment advice</h2>
      <p>
        Nothing published by {brand.name} should be construed as an
        offer, solicitation, or recommendation to buy, sell, hold, or
        deal in any security, derivative, or financial instrument. All
        decisions to act on any of our published research rest
        entirely with the reader. Past performance, calibration
        metrics, and back-tested results are not indicative of future
        results.
      </p>

      <h2 className="text-xl text-fg font-serif">No personal advice</h2>
      <p>
        Our published research does not take into account any
        individual&rsquo;s investment objectives, risk tolerance,
        financial situation, or specific portfolio context. Before
        acting on any market view, readers should consult a
        SEBI-registered investment adviser of their choice.
      </p>

      <h2 className="text-xl text-fg font-serif">Risk warning</h2>
      <p>
        Trading and investing in Indian equity, derivative, and
        options markets carries substantial risk of loss and is not
        suitable for every investor. Losses can exceed the initial
        capital deployed, particularly in leveraged products. Readers
        are responsible for assessing their own suitability and risk
        capacity.
      </p>

      <h2 className="text-xl text-fg font-serif">Conflicts of interest</h2>
      <p>
        The {brand.name} team may, from time to time, hold positions
        in instruments referenced in published research. Where a
        material conflict exists, it will be disclosed in the brief
        section that discusses that instrument.
      </p>
      <p>
        {brand.name} does not accept payment in cash or kind from any
        listed company, broker, exchange, or product issuer in
        exchange for coverage. Subscription revenue is the sole
        revenue source.
      </p>

      <h2 className="text-xl text-fg font-serif">Outcome-log integrity</h2>
      <p>
        Calibration metrics displayed on the{" "}
        <a href="/track-record" className="text-accent underline underline-offset-4">
          Track record
        </a>{" "}
        page and inside every brief&rsquo;s Yesterday Audit section
        are computed from an append-only outcome log. Predictions
        whose outcomes are drawn from retrospective replay of
        historical bundles are tagged as such, and the rendered
        disclosure makes the share of retrospective vs live outcomes
        explicit.
      </p>

      <h2 className="text-xl text-fg font-serif">Jurisdiction</h2>
      <p>
        This service is offered from India, primarily targeted at
        Indian-resident readers and Indian-market securities and
        derivatives. International readers should ensure their own
        local regulations permit consumption of paid market research
        of this nature.
      </p>

      <h2 className="text-xl text-fg font-serif">Updates to this page</h2>
      <p>
        The content of this page may be updated as regulations evolve
        or as the scope of our published research changes. Material
        changes will be noted in a brief and emailed to active
        subscribers.
      </p>
    </LegalPage>
  );
}
