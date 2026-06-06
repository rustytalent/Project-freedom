import type { Metadata } from "next";
import { ProductPage } from "@/components/marketing/ProductPage";
import { brand } from "@/lib/brand";

export const metadata: Metadata = {
  title: "Daily Brief",
  description:
    "A structured research email and portal view, every NSE trading day, before open.",
};

export default function DailyBriefPage() {
  return (
    <ProductPage
      eyebrow="Product · Daily Brief"
      title="A six-minute read, every NSE trading day, before open."
      oneLiner="The flagship product. Research context for intraday traders, audited the next morning."
      intro={
        <>
          <p>
            The Daily Brief is one structured document, generated once
            per IST trading day, delivered as both an email and a portal
            view by 08:30 IST. It is designed to be read in five to
            seven minutes by a sophisticated trader who is not paid to
            read carefully.
          </p>
          <p>
            Every section of the brief has an outcome log behind it.
            Yesterday&rsquo;s brief is audited inside today&rsquo;s, so the
            product has to show when its read was useful and when it was
            not.
          </p>
        </>
      }
      sections={[
        {
          heading: "What's inside",
          body: (
            <>
              <p>
                Every brief carries the same fixed-order sections, so
                you know where to look:
              </p>
              <ul className="list-disc pl-6 space-y-2">
                {brand.briefSections.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
              <p className="mt-4 text-sm">
                Sections that depend on data not yet wired (for example,
                index regime when the bundle has no index data) render
                as a one-line &ldquo;pending&rdquo; note so the
                contract shape stays stable across briefs.
              </p>
            </>
          ),
        },
        {
          heading: "What's not inside",
          body: (
            <>
              <p>
                No entry, target, or stop suggestions for any
                instrument. No urgency. No commands of any kind.
              </p>
              <p>
                The brief reports the model&rsquo;s view in regime,
                probability, and avoidance language. If today the model
                has no actionable conviction anywhere across the
                basket, the brief says so plainly, and the avoid list
                opens with{" "}
                <span className="font-mono text-fg">ALL_BASKET</span>{" "}
                stand-aside.
              </p>
            </>
          ),
        },
        {
          heading: "Yesterday Audit",
          body: (
            <>
              <p>
                Every Daily Brief opens with an audit of the prior
                trading day&rsquo;s brief. Per-confidence-bucket hit
                rate, per-prediction-type calibration error, drift
                flags, and the retrospective-replay disclosure when
                applicable.
              </p>
              <p>
                The same numbers appear nightly on the public{" "}
                <a
                  href="/track-record"
                  className="text-accent underline underline-offset-4"
                >
                  Track record
                </a>{" "}
                page so prospects can browse the discipline before
                subscribing.
              </p>
            </>
          ),
        },
      ]}
      delivery="Email + portal. Time: 08:30 IST, every NSE trading day. PDF available on request."
      cost="Daily Brief starts at ₹4,999 per month. See full pricing."
      notes={
        <p>
          This brief is research context, not investment advice. We are
          not SEBI-registered investment advisors. See full disclosures.
        </p>
      }
    />
  );
}
