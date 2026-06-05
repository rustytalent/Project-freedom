import Link from "next/link";
import { Card, CardContent, CardDescription, CardTitle } from "@/components/ui/card";
import { LinkButton } from "@/components/ui/button";
import { formatDateIST } from "@/lib/utils";

export const metadata = {
  title: "Portal",
  description: "Today's brief, archive, calibration, account.",
};

export default function PortalLandingPage() {
  const today = formatDateIST(new Date());
  return (
    <div className="max-w-dash mx-auto px-6 py-14">
      <header className="mb-10">
        <p className="text-xs uppercase tracking-[0.18em] text-accent mb-3">
          Today &middot; {today}
        </p>
        <h1 className="font-serif text-3xl md:text-4xl text-fg">
          Welcome back.
        </h1>
        <p className="mt-4 text-fg-muted leading-relaxed max-w-prose">
          Today&rsquo;s Daily Brief is ready. The Yesterday Audit
          inside it covers calls made on the previous IST trading
          session.
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-3">
        <Card className="md:col-span-2 border-accent/40">
          <p className="text-xs uppercase tracking-wider text-accent mb-3">
            Primary
          </p>
          <CardTitle>Today&rsquo;s Daily Brief</CardTitle>
          <CardDescription>
            Generated 08:30 IST &middot; estimated 6 minutes to read.
          </CardDescription>
          <CardContent className="mt-6">
            <LinkButton href="/portal/brief/today" variant="primary">
              Read today&rsquo;s brief →
            </LinkButton>
          </CardContent>
        </Card>

        <Card>
          <CardTitle>Calibration this week</CardTitle>
          <CardDescription>
            Quick read on which heads are tracking.
          </CardDescription>
          <CardContent className="mt-6">
            <p className="font-mono text-xs text-fg-muted leading-relaxed tabnum">
              proximity: <span className="text-calibrated">calibrated</span>
              <br />
              avoidance: <span className="text-calibrated">calibrated</span>
              <br />
              options: <span className="text-drift">drifting +0.11</span>
            </p>
            <Link
              href="/portal/calibration"
              className="block mt-4 text-sm text-accent underline underline-offset-4"
            >
              Open full dashboard →
            </Link>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2 mt-6">
        <Card>
          <CardTitle>Recent briefs</CardTitle>
          <CardContent className="mt-4 space-y-2 text-sm">
            {[
              { date: "2026-06-04", note: "Daily" },
              { date: "2026-06-03", note: "Daily" },
              { date: "2026-06-02", note: "Daily" },
              { date: "2026-06-01", note: "Swing — week of Jun 1" },
            ].map((row) => (
              <div
                key={row.date}
                className="flex items-center justify-between border-b border-border/40 py-2 last:border-b-0"
              >
                <span className="text-fg-muted font-mono tabnum">
                  {row.date}
                </span>
                <span className="text-xs text-fg-subtle">{row.note}</span>
                <Link
                  href={`/portal/brief/${row.date}`}
                  className="text-xs text-accent underline underline-offset-4"
                >
                  Open
                </Link>
              </div>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardTitle>Your subscription</CardTitle>
          <CardDescription>Multi-product · Monthly</CardDescription>
          <CardContent className="mt-4 text-sm text-fg-muted">
            <p>Next renewal: 2026-07-04.</p>
            <p className="mt-2">Daily Brief + Swing Brief active.</p>
            <Link
              href="/portal/account"
              className="block mt-4 text-sm text-accent underline underline-offset-4"
            >
              Manage subscription →
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
