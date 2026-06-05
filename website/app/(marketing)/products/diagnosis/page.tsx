import type { Metadata } from "next";
import { ProductPage } from "@/components/marketing/ProductPage";

export const metadata: Metadata = {
  title: "Diagnosis",
  description:
    "A one-off audit of your own strategy. You bring the trade log; we return a PDF report.",
};

export default function DiagnosisPage() {
  return (
    <ProductPage
      eyebrow="Product · Diagnosis"
      title="You bring your alpha. We audit it."
      oneLiner="A one-off, deeply-researched PDF report on what is and isn't working in your own strategy."
      intro={
        <>
          <p>
            Most retail traders never get an honest, structured second
            opinion on the strategy they trade. The friends and forums
            available to ask are usually wrong about both the strengths
            and the weaknesses. The Diagnosis product fills that gap.
          </p>
          <p>
            Submit a trade log; we return a structured report. Five
            trading days from intake to delivery. Independent — we are
            not selling you the next strategy, we are reading the one
            you already have.
          </p>
        </>
      }
      sections={[
        {
          heading: "What you submit",
          body: (
            <>
              <p>One of two artifact types is sufficient:</p>
              <ul className="list-disc pl-6 space-y-2">
                <li>
                  A backtest log — a CSV of trades with timestamps,
                  side, entry, exit, and P&amp;L, plus a one-page
                  description of the strategy logic.
                </li>
                <li>
                  A live track record — same shape, but tagged as live
                  rather than simulated. Broker statements work if you
                  can export them as trade rows.
                </li>
              </ul>
              <p>
                The CSV column spec is on a separate page we share at
                intake. NDA on request.
              </p>
            </>
          ),
        },
        {
          heading: "What the report contains",
          body: (
            <>
              <p>
                Section list, at a high level (the report is
                structured so the conclusion is reachable from the
                first page):
              </p>
              <ul className="list-disc pl-6 space-y-2">
                <li>
                  Regime decomposition — when does the strategy work,
                  and what regime feature explains the bulk of the
                  good periods.
                </li>
                <li>
                  Drawdown attribution — what specifically went wrong
                  in the worst quintile of trades.
                </li>
                <li>
                  Persistence test — does the edge appear in both
                  halves of the sample, or is it a single sub-period?
                </li>
                <li>
                  Leakage check — are any of the features the strategy
                  uses peering into the future (intentionally or
                  otherwise)?
                </li>
                <li>
                  Cost sensitivity — how the edge degrades when you
                  bump assumed slippage and brokerage by 1.5x.
                </li>
                <li>
                  Recommendation — which of (a) ship as-is, (b) ship
                  with specific guardrails, (c) re-scope to a
                  different regime, (d) shelve.
                </li>
              </ul>
            </>
          ),
        },
        {
          heading: "What we do not do",
          body: (
            <>
              <p>
                We do not provide a corrected strategy. We do not
                provide signals you can trade. We do not co-develop or
                license alpha back to you. The Diagnosis product is
                strictly the audit; the action is yours to take.
              </p>
              <p>
                We do hold every submitted strategy in confidence —
                see the disclosures page for the data-handling terms.
              </p>
            </>
          ),
        },
      ]}
      delivery="Five trading days from intake to PDF delivery. Encrypted handoff; signed NDA on request."
      cost="From ₹— per audit. See full pricing."
    />
  );
}
