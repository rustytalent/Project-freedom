import type { Metadata } from "next";
import { ProductPage } from "@/components/marketing/ProductPage";

export const metadata: Metadata = {
  title: "Audit Infrastructure",
  description:
    "Our internal audit harness, licensed to other quant teams.",
};

export default function AuditInfrastructurePage() {
  return (
    <ProductPage
      eyebrow="Product · Audit Infrastructure (B2B)"
      title="The audit harness we run on ourselves, licensed to your team."
      oneLiner="For small quant funds and prop desks. Low volume, high engagement."
      intro={
        <>
          <p>
            The discipline that makes our briefs honest is the operating
            layer around the research. Outcome logging, calibration
            tracking, retrospective replay disclosure, and causality
            checks are the parts other teams can use without seeing our
            private research engine.
          </p>
          <p>
            For teams that already have a research edge and want a structured
            audit layer wrapped around it, we license that harness as
            a B2B engagement.
          </p>
        </>
      }
      sections={[
        {
          heading: "What's in scope",
          body: (
            <>
              <ul className="list-disc pl-6 space-y-2">
                <li>
                  Outcome-log schema and tooling: append-only,
                  partitioned by trading date, with the retrospective
                  versus live flag built in.
                </li>
                <li>
                  Calibration dashboards: per-bucket hit rate,
                  calibration error trend, drift flags. Same shape
                  as the dashboard you can see on our public track
                  record page.
                </li>
                <li>
                  Causality test invariants: the patterns we use
                  to pin time order in our own engine, ported into
                  your repo as test scaffolding.
                </li>
                <li>
                  Client-reporting guardrails:
                  if your shop publishes any client-facing content,
                  we can wire checks that block instruction-style
                  language before it reaches customers.
                </li>
              </ul>
            </>
          ),
        },
        {
          heading: "What's not in scope",
          body: (
            <p>
              Our detectors. Our models. Our calibration parameters.
              Our feature engineering. The operating layer is the deliverable;
              the methodology that runs inside it is ours.
            </p>
          ),
        },
        {
          heading: "How an engagement works",
          body: (
            <ul className="list-disc pl-6 space-y-2">
              <li>
                Two-week scoping engagement: we read your stack, agree
                on the audit surface, write a fixed-scope proposal.
              </li>
              <li>
                Six-week implementation: we port the harness into your
                repo, write the tests, hand it over.
              </li>
              <li>
                Optional quarterly review thereafter.
              </li>
            </ul>
          ),
        },
      ]}
      delivery="On a per-engagement basis. Roughly eight weeks from contract to handover."
      cost="On request. Audit Base starts at ₹2,000 and is the smaller engagement to start with."
    />
  );
}
