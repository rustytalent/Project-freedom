import type { MetadataRoute } from "next";
import { brand } from "@/lib/brand";

const ROUTES = [
  "/",
  "/philosophy",
  "/products",
  "/products/daily-brief",
  "/products/swing-brief",
  "/products/diagnosis",
  "/products/audit-infrastructure",
  "/track-record",
  "/sample-brief",
  "/yesterday-audit",
  "/pricing",
  "/about",
  "/contact",
  "/legal/disclosures",
  "/legal/terms",
  "/legal/privacy",
];

export default function sitemap(): MetadataRoute.Sitemap {
  const base = `https://${brand.domain}`;
  return ROUTES.map((path) => ({
    url: `${base}${path}`,
    lastModified: new Date(),
  }));
}
