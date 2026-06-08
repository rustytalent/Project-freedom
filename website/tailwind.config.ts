import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./content/**/*.{md,mdx,json}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Palette per §7 of the design spec.
        bg: {
          DEFAULT: "#0E0F11",     // near-black warm
          raised: "#15171A",      // cards / elevated surfaces
          subtle: "#1B1E22",      // hover / data-strip rows
        },
        fg: {
          DEFAULT: "#E8E6E1",     // warm off-white
          muted: "#9DA0A6",       // secondary text
          subtle: "#5E6168",      // captions / dividers
        },
        accent: {
          DEFAULT: "#7C9BB8",     // muted steel
          dim: "#5A7691",
          glow: "#9FB6CB",
        },
        // Deliberate warm second accent. Used sparingly - the headline
        // focal word, the edition tag, the brand reticle hover state,
        // the friction-reducing microcopy near CTAs. One warm note in
        // a cool composition. Burnished bronze, not yellow gold.
        warm: {
          DEFAULT: "#B79268",
          dim: "#8C6F4F",
          glow: "#D4B68A",
        },
        drift: "#C66B5C",         // error / drift
        calibrated: "#7DA982",    // success / calibrated
        border: "#23262B",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ["Fraunces", "ui-serif", "Georgia", "serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      maxWidth: {
        prose: "720px",
        dash: "1100px",
      },
      typography: {
        DEFAULT: {
          css: {
            "--tw-prose-body": "#E8E6E1",
            "--tw-prose-headings": "#E8E6E1",
            "--tw-prose-links": "#9FB6CB",
            "--tw-prose-bold": "#E8E6E1",
            "--tw-prose-quotes": "#9DA0A6",
            "--tw-prose-code": "#E8E6E1",
            "--tw-prose-hr": "#23262B",
          },
        },
      },
    },
  },
  plugins: [],
};

export default config;
