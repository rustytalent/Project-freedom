import type { Metadata } from "next";
import { ProductPage } from "@/components/marketing/ProductPage";

export const metadata: Metadata = {
  title: "Swing Brief",
  description:
    "Coming multi-day research brief for positional traders.",
};

export default function SwingBriefPage() {
  return (
    <ProductPage
      eyebrow="Product · Swing Brief"
      title="A coming multi-day research stream."
      oneLiner="For positional traders and options users who need slower context than the Daily Brief."
      intro={
        <>
          <p>
            The current production focus is the Daily Brief. Swing is
            the next research stream because slower horizons usually
            give cleaner cost arithmetic and more time for a thesis to
            develop.
          </p>
          <p>
            We will not sell this as a separate paid product until its
            archive and outcome log are stable enough to review.
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
          heading: "What's different from the Daily Brief",
          body: (
            <>
              <p>
                Same research discipline, different feedback cycle.
                Each watchlist entry needs a hold-period budget so you
                know whether it is a near-term context read or a slower
                structural setup.
              </p>
              <p>
                The Yesterday Audit equivalent on swing runs at the
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
      delivery="Not generally available yet. Planned delivery is email plus portal."
      cost="Included in Pro Desk only after the stream is validated and enabled."
    />
  );
}
