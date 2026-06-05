import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, extname } from "node:path";

// The §2 grep-pass from the website spec, enforced as a test.
//
// The website is a public surface; the engine's methodology is the
// moat. This test walks the codebase and fails loudly if any
// surface file mentions a detector, model, or feature name that
// must not appear publicly.
//
// If a new internal name needs a public-friendly alias, add it to
// the engine's vocabulary documentation — NOT to this allowlist.

const FORBIDDEN: ReadonlyArray<RegExp> = [
  // Detector names
  /\bliquidity[\s_]?sweep\b/i,
  /\bstop[\s_-]?run[\s_-]?reclaim\b/i,
  /\bSWEEP_[HL]\b/,
  /\bSR_[HL]\b/,
  /\bMBI_(bull|bear)\b/,
  /\bPD_MID\b/,
  /\bVW_SWING_[HL]\b/,
  /\bCD_DIV_[HL]\b/,
  /\bmulti[\s_-]?bar[\s_-]?imbalance\b/i,
  /\bpremium\/discount midpoint\b/i,
  /\bcumulative[\s_-]?delta[\s_-]?divergence\b/i,
  /\bvolume[\s_-]?weighted[\s_-]?swing\b/i,
  /\border[\s_-]?block\b/i,
  /\bfair[\s_-]?value[\s_-]?gap\b/i,
  /\bFVG\b/,
  /\bORB\b/,
  /\bEQH\b/,
  /\bEQL\b/,
  /\bPWH\b/,
  /\bPWL\b/,
  // Model names
  /\bPoolProximityModel\b/,
  /\bUnifiedDirection(Model)?\b/,
  /\bPoolRespectModel\b/,
  /\bPolicyOutcomeModel\b/,
  /\bPolicyReturnModel\b/,
  /\bOptionsExpectedReturnModel\b/,
  /\bSwingProximityModel\b/,
  /\bSectorMoE\b/,
  /\bLightGBM\b/,
  /\bXGBoost\b/,
  // Feature / calibration internals
  /\bpath_efficiency_30\b/,
  /\bdirection_changes_30\b/,
  /\bvol_regime_zscore_20d\b/,
  /\biv_percentile_60d\b/,
  /\bdist_atr\b/,
  /\batr_at_last\b/,
  /\bisotonic\b/i,
  /\bbucket[\s_-]?shrinkage\b/i,
  /\bembargo[\s_-]?bars\b/i,
  /\bpurged[\s_-]?walk[\s_-]?forward\b/i,
  /\bhuber loss\b/i,
  /\bwinsorization\b/i,
  // Hyperparameter giveaways
  /\bmin_atr=0\./,
  /\breclaim_window=\d+\b/,
  /\bswing_left=\d+\b/,
  /\balpha=0\.9\b/,
  // Internal agent / repo names
  /\bopus\b/i,
  /\bcodex\b/i,
  /\bclaude\b/i,
  /\bproject-freedom\b/i,
  /\bliqpool\b/,
] as const;

// Allow this test file itself to mention the forbidden terms — that's
// the only way the test can describe what it's testing.
const ALLOW_FILES = new Set<string>([
  "tests/moat.test.ts",
]);

const SCAN_EXTENSIONS = new Set([".ts", ".tsx", ".mdx", ".md", ".json"]);
const SKIP_DIRS = new Set([
  "node_modules",
  ".next",
  ".vercel",
  "coverage",
  ".turbo",
]);

function walk(dir: string, root: string, out: string[]): void {
  for (const entry of readdirSync(dir)) {
    if (SKIP_DIRS.has(entry)) continue;
    const full = join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) {
      walk(full, root, out);
    } else if (SCAN_EXTENSIONS.has(extname(entry))) {
      out.push(full.slice(root.length + 1));
    }
  }
}

describe("§2 moat guardrail", () => {
  it("no forbidden technique name appears in any public surface file", () => {
    const root = process.cwd();
    const files: string[] = [];
    walk(root, root, files);
    const violations: Array<{
      file: string;
      pattern: string;
      excerpt: string;
    }> = [];
    for (const rel of files) {
      if (ALLOW_FILES.has(rel)) continue;
      const content = readFileSync(join(root, rel), "utf8");
      for (const pat of FORBIDDEN) {
        const m = content.match(pat);
        if (m) {
          const idx = m.index ?? 0;
          const excerpt = content
            .slice(Math.max(0, idx - 40), idx + m[0].length + 40)
            .replace(/\s+/g, " ");
          violations.push({
            file: rel,
            pattern: pat.toString(),
            excerpt,
          });
        }
      }
    }
    if (violations.length > 0) {
      const msg = violations
        .map(
          (v) =>
            `  ${v.file}\n    pattern: ${v.pattern}\n    excerpt: …${v.excerpt}…`,
        )
        .join("\n\n");
      throw new Error(
        `Moat violation: ${violations.length} forbidden technique ` +
          `reference(s) found in public surface files.\n\n${msg}\n\n` +
          `Either rephrase to use the customer-facing vocabulary ` +
          `(see docs/website_codex_prompt.md §2) or, if absolutely ` +
          `necessary, add the file to ALLOW_FILES with a justifying ` +
          `comment.`,
      );
    }
  });
});
