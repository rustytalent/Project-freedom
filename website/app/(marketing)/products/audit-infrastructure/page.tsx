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
            The discipline that makes our briefs honest is not the
            individual detectors, it is the audit harness around them.
            Calibration tracking, drift detection, retrospective-
            replay disclosure, outcome logging, no-lookahead invariants
            pinned in tests.
          </p>
          <p>
            For teams that already have alpha and want a structured
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
                  Outcome-log schema and tooling — append-only,
                  partitioned by trading date, with the retrospective-
                  vs-live flag built in.
                </li>
                <li>
                  Calibration dashboards — per-bucket hit rate,
                  calibration error trend, drift flags. Same shape
                  as the dashboard you can see on our public track
                  record page.
                </li>
                <li>
                  No-lookahead test invariants — the patterns we use
                  to pin causality in our own engine, ported into
                  your repo as test scaffolding.
                </li>
                <li>
                  Brief-renderer-style tipster-vocabulary guardrails —
                  if your shop publishes any client-facing content,
                  we can wire the same render-time forbidden-phrase
                  check.
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
              Our feature engineering. The harness is the deliverable;
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
      cost="On request. The Diagnosis product is a useful smaller engagement to start with."
    />
  );
}
