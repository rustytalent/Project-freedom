// Single source of truth for brand identity.
//
// All values are reachable through this module so the founder can
// rebrand in one diff. Marketing copy in MDX/page files can also
// import from here — never hardcode brand strings in JSX.

export const brand = {
  name: process.env.NEXT_PUBLIC_BRAND_NAME ?? "Aurora Research",
  domain: process.env.NEXT_PUBLIC_DOMAIN ?? "aurora-research.in",
  tagline: "Calibrated context, daily, for the Indian markets.",
  subTagline:
    "A research brief in your inbox before NSE opens, followed by a " +
    "public audit of yesterday's calls.",
  founder: {
    name: "(founder name)",
  },
  contact: {
    email: "hello@aurora-research.in",
    calendly: null as string | null,
  },
  legalEntity: "(legal entity TBD)",
  sebi: {
    // We are NOT a SEBI-registered investment advisor. The disclosures
    // page makes this explicit. Every page footer carries the short form.
    shortDisclosure:
      "Research context only — not investment advice. We are not a " +
      "SEBI-registered investment advisor.",
  },
  // Section names in the daily brief, surfaced on marketing pages.
  briefSections: [
    "TLDR",
    "Sector regime",
    "Watchlist",
    "Avoidance list",
    "Confidence notes",
    "Yesterday audit",
  ],
} as const;

export type Brand = typeof brand;
