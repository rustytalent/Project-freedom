/**
 * StatMark: tiny per-stat SVG visual that sits above the number in
 * NumbersStrip. Each mark reinforces the metric it labels - a small
 * editorial flourish that signals deliberate design rather than a
 * generic dashboard. Pure SVG, no client JS, no animation.
 */
type Kind = "staircase" | "tally" | "oscillation" | "disc";

const W = 56;
const H = 18;

function Staircase() {
  // Rising step pattern - "things accumulating over time".
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      <path
        d="M2 16 L10 16 L10 12 L20 12 L20 9 L30 9 L30 6 L40 6 L40 3 L54 3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
        strokeLinejoin="miter"
      />
    </svg>
  );
}

function Tally() {
  // Five vertical strokes with a diagonal - tally count.
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      <g stroke="currentColor" strokeWidth="1" strokeLinecap="round">
        <line x1="6"  y1="3" x2="6"  y2="15" />
        <line x1="11" y1="3" x2="11" y2="15" />
        <line x1="16" y1="3" x2="16" y2="15" />
        <line x1="21" y1="3" x2="21" y2="15" />
        <line x1="3"  y1="14" x2="24" y2="4" />
        <line x1="32" y1="3" x2="32" y2="15" />
        <line x1="37" y1="3" x2="37" y2="15" />
        <line x1="42" y1="3" x2="42" y2="15" />
      </g>
    </svg>
  );
}

function Oscillation() {
  // Tight sine around a zero line - "calibration hovers near zero".
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      <line
        x1="2" y1="9" x2="54" y2="9"
        stroke="currentColor" strokeOpacity="0.3" strokeWidth="0.75"
        strokeDasharray="2 2"
      />
      <path
        d="M2 9 Q9 2, 16 9 T30 9 T44 9 T58 9"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
      />
    </svg>
  );
}

function Disc() {
  // A fully-filled disc inside a ring - 100% transparency.
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      <circle cx="9" cy="9" r="7" fill="none" stroke="currentColor" strokeWidth="1" strokeOpacity="0.4" />
      <circle cx="9" cy="9" r="5" fill="currentColor" />
      <line x1="22" y1="9" x2="54" y2="9" stroke="currentColor" strokeWidth="1" strokeOpacity="0.3" strokeDasharray="2 2" />
    </svg>
  );
}

export function StatMark({ kind, className = "" }: { kind: Kind; className?: string }) {
  return (
    <div className={`text-warm/70 ${className}`}>
      {kind === "staircase" && <Staircase />}
      {kind === "tally" && <Tally />}
      {kind === "oscillation" && <Oscillation />}
      {kind === "disc" && <Disc />}
    </div>
  );
}
