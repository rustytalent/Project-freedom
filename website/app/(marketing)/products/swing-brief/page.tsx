import type { Metadata } from "next";
import { ProductPage } from "@/components/marketing/ProductPage";

export const metadata: Metadata = {
  title: "Live Desk",
  description:
    "Live update desk and slower-horizon research stream for subscribers.",
};

export default function SwingBriefPage() {
  return (
    <ProductPage
      eyebrow="Product · Live Desk"
      title="Core Research, with live updates when the desk is active."
      oneLiner="For readers who want the core archive plus faster update delivery."
      intro={
        <>
          <p>
            Live Desk builds on Core Research. It keeps the same
            morning brief and archive, then adds live update delivery
            when the desk is active.
          </p>
          <p>
            Slower-horizon and options notes are part of the same desk
            language, but every module stays behind an audit trail
            before it becomes a regular customer artifact.
          </p>
        </>
      }
      sections={[
        {
          heading: "Horizons covered",
          body: (
            <>
              <p>
                The proposed brief carries three slower context windows,
                each with its own audit table:
              </p>
              <ul className="list-disc pl-6 space-y-2">
                <li>5 trading days - the near horizon.</li>
                <li>10 trading days - the standard hold.</li>
                <li>
                  20 trading days - the structural horizon. Useful
                  context for longer positional bets and weekly option
                  decisions.
                </li>
              </ul>
            </>
          ),
        },
        {
          heading: "What's different from Core Research",
          body: (
            <>
              <p>
                Same research discipline, different feedback cycle.
                Each watchlist entry needs a hold-period budget so you
                know whether it is a near-term context read or a slower
                structural setup.
              </p>
              <p>
                The Yesterday Audit equivalent on slower notes runs at the
                end of each 5/10/20-day window, not nightly.
                Calibration converges more slowly on swing horizons -
                that is honest evidence of the longer feedback loop.
              </p>
            </>
          ),
        },
        {
          heading: "Why this is the most likely product line to land profitable",
          body: (
            <p>
              We are researching this stream because cost drag is often
              less punishing on slower holds. That is a hypothesis to
              validate in the archive, not a claim to sell today.
            </p>
          ),
        },
      ]}
      delivery="Email plus portal, with live update delivery only for Live Desk subscribers when the desk is active."
      cost="Live Desk is ₹14,999 per month. Core Research still includes equity, options, and index research without live updates."
    />
  );
}
