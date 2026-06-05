import type { Metadata } from "next";
import { ProductPage } from "@/components/marketing/ProductPage";

export const metadata: Metadata = {
  title: "Swing Brief",
  description:
    "Multi-day proximity calls, delivered Monday pre-open plus ad-hoc on regime flips.",
};

export default function SwingBriefPage() {
  return (
    <ProductPage
      eyebrow="Product · Swing Brief"
      title="Multi-day calls, on the horizon that actually pays."
      oneLiner="The same engine, applied to 5-, 10-, and 20-trading-day horizons. For positional traders and option buyers."
      intro={
        <>
          <p>
            Cost arithmetic favours longer holds in the Indian markets.
            Delivery carries zero brokerage on most retail brokers, no
            MIS square-off penalty, and the typical move is several
            times the typical intraday move. The same edge that
            struggles to clear intraday cost becomes structurally
            profitable on swing.
          </p>
          <p>
            The Swing Brief is the product line that exploits this. A
            structured read of multi-day proximity calls, delivered
            Monday pre-open and ad-hoc whenever the regime flips
            mid-week.
          </p>
        </>
      }
      sections={[
        {
          heading: "Horizons covered",
          body: (
            <>
              <p>
                The brief carries proximity calls at three horizons,
                each with its own calibration table:
              </p>
              <ul className="list-disc pl-6 space-y-2">
                <li>5 trading days — the near horizon.</li>
                <li>10 trading days — the standard hold.</li>
                <li>
                  20 trading days — the structural horizon. Useful
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
                Same discipline (no tipster vocabulary, calibration
                front and centre, audited next cycle) but with a
                horizon-aware schema. Each watchlist entry includes an
                expected hold-period budget so you know whether a call
                is a near-term test or a structural setup.
              </p>
              <p>
                The Yesterday Audit equivalent on swing runs at the
                end of each 5/10/20-day window, not nightly.
                Calibration converges more slowly on swing horizons —
                that is honest evidence of the longer feedback loop.
              </p>
            </>
          ),
        },
        {
          heading: "Why this is the most likely product line to land profitable",
          body: (
            <p>
              Per-trade cost as a fraction of typical move is roughly
              seven times friendlier on swing delivery than on
              intraday MIS. We have not changed the alpha; we have
              changed the cost structure the alpha runs against. The
              numbers move accordingly.
            </p>
          ),
        },
      ]}
      delivery="Email + portal. Monday pre-open + ad-hoc when the regime flips mid-week."
      cost="From ₹— per month. Bundled with the Daily Brief at a discount in the Multi-product tier."
    />
  );
}
