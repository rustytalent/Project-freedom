import { brand } from "@/lib/brand";
import { Reveal } from "@/components/ui/Reveal";

/**
 * FoundersLetter: a deliberately light, cream-paper interlude.
 *
 * The rest of the site is dark - the "trading terminal" register.
 * This single section flips to cream + ink, which lets the human
 * voice of the founder breathe in a different register. The Stripe
 * Press / A16Z editorial move: dark for the data, light for the
 * voice. The dark/light contrast is itself a premium signal -
 * a visitor experiences two distinct moods on one site without
 * either feeling out of place.
 *
 * Constraint: cream is the existing brand off-white (#E8E6E1) used
 * as a background; ink is the existing brand near-black (#0E0F11)
 * as the text. No new palette - just the dark theme inverted for
 * 540px of vertical space.
 */
export function FoundersLetter() {
  return (
    <section
      className="bg-fg text-bg relative overflow-hidden"
      // Inverted theme: cream paper, near-black ink. The dotted
      // backdrop gives the panel a faint "letter stationery" texture
      // without committing to a literal paper image.
      aria-label="A letter from the founder"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0 opacity-[0.07] pointer-events-none"
        style={{
          backgroundImage:
            "radial-gradient(circle at 1px 1px, rgba(14,15,17,0.6) 1px, transparent 0)",
          backgroundSize: "12px 12px",
        }}
      />
      <div className="relative max-w-prose mx-auto px-6 py-24 md:py-32">
        <Reveal>
          <p className="font-mono text-[11px] uppercase tracking-[0.24em] text-bg/60 mb-3">
            From the desk
          </p>
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-bg/40 mb-12 pb-6 border-b border-bg/15">
            Of {brand.founder.name} · founder · {brand.name}
          </p>
        </Reveal>

        <Reveal delay={120}>
          <div className="font-serif text-[19px] md:text-[21px] leading-[1.7] text-bg space-y-6">
            <p>
              <span className="font-serif text-[64px] md:text-[80px] leading-none float-left mr-3 -mt-1 text-bg/85">
                W
              </span>
              hen I started building this, I wasn&rsquo;t trying to
              invent another signal service. India already has thousands
              of those. I was trying to answer a smaller, more honest
              question: when a brief says &ldquo;70% likely to touch,&rdquo;
              does it actually happen 70 times out of 100?
            </p>
            <p>
              Most desks never check. We do, every morning. The
              dashboard you just scrolled past is the same dashboard I
              look at before I write the next brief. When a head drifts,
              you see it before I do.
            </p>
            <p>
              If that kind of accountability is what you want from your
              morning read, the sample brief is the place to start. If
              it isn&rsquo;t, no hard feelings &mdash; close the tab.
            </p>
          </div>
        </Reveal>

        <Reveal delay={240}>
          <div className="mt-12 pt-8 border-t border-bg/15 flex items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              {/* Reticle mark in ink tone - the brand reticle in
                  inverted theme. */}
              <span
                className="inline-block w-3.5 h-3.5 relative text-bg/70"
                aria-hidden="true"
                style={{ verticalAlign: "middle" }}
              >
                <span
                  className="absolute inset-0 rounded-full border"
                  style={{ borderColor: "currentColor" }}
                />
                <span
                  className="absolute left-1/2 top-[-3px] bottom-[-3px] w-px -translate-x-1/2"
                  style={{ background: "currentColor" }}
                />
                <span
                  className="absolute top-1/2 left-[-3px] right-[-3px] h-px -translate-y-1/2"
                  style={{ background: "currentColor" }}
                />
              </span>
              <p className="font-serif italic text-lg text-bg/80">
                Garvit
              </p>
            </div>
            <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-bg/45 text-right">
              Bangalore · India
              <br />
              published before the bell
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
