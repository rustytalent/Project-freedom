/**
 * IST clock utility for the marketing surface.
 *
 * Every dateline on the public site is IST-anchored: the brief
 * publishes by 08:30 IST, the audit follows the next NSE session,
 * and the calendar week skips weekends (public holidays are honoured
 * by the live Supabase view in production; the marketing surface
 * approximates with a five-day trading week).
 *
 * All math here works on YYYY-MM-DD strings ("date keys") so we
 * never have to think about timezone offsets when adding days.
 * Weekday is computed via Date.UTC because the day-of-week of a
 * Y/M/D triple is invariant under timezone.
 */

const MONTHS = [
  "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
  "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
] as const;

// Launch reference. Used for edition numbering. Update if the live
// launch date drifts.
const LAUNCH_KEY = "2026-01-15";

/** YYYY-MM-DD string for the given Date, in IST. */
export function istKey(d: Date = new Date()): string {
  // en-CA gives the locale-invariant YYYY-MM-DD format we want.
  return new Intl.DateTimeFormat("en-CA", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    timeZone: "Asia/Kolkata",
  }).format(d);
}

function parts(key: string): [number, number, number] {
  const [y, m, d] = key.split("-").map(Number);
  return [y, m, d];
}

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

/** Weekday 0-6 (Sun-Sat) for the calendar date in `key`. */
function weekday(key: string): number {
  const [y, m, d] = parts(key);
  return new Date(Date.UTC(y, m - 1, d)).getUTCDay();
}

function isWeekend(key: string): boolean {
  const w = weekday(key);
  return w === 0 || w === 6;
}

/** Add `n` calendar days to a key, return the new key. */
function addDays(key: string, n: number): string {
  const [y, m, d] = parts(key);
  const dt = new Date(Date.UTC(y, m - 1, d + n));
  return (
    dt.getUTCFullYear() +
    "-" +
    pad2(dt.getUTCMonth() + 1) +
    "-" +
    pad2(dt.getUTCDate())
  );
}

/**
 * Most recent NSE trading day (today if today is a weekday,
 * otherwise the most recent Friday).
 */
export function currentTradingDay(now: Date = new Date()): string {
  let k = istKey(now);
  while (isWeekend(k)) k = addDays(k, -1);
  return k;
}

/** Previous NSE trading day strictly before `key`. */
export function previousTradingDay(key: string): string {
  let k = addDays(key, -1);
  while (isWeekend(k)) k = addDays(k, -1);
  return k;
}

/** Next NSE trading day strictly after `key`. */
export function nextTradingDay(key: string): string {
  let k = addDays(key, 1);
  while (isWeekend(k)) k = addDays(k, 1);
  return k;
}

/** "06 JUN 2026" - mono editorial dateline. */
export function formatLong(key: string): string {
  const [y, m, d] = parts(key);
  return `${pad2(d)} ${MONTHS[m - 1]} ${y}`;
}

/** "06 JUN" - compact ticker / ledger date stamp. */
export function formatShort(key: string): string {
  const [, m, d] = parts(key);
  return `${pad2(d)} ${MONTHS[m - 1]}`;
}

/** "Tomorrow", "Today", "Mon" - relative weekday for the next brief. */
export function relativeWeekday(key: string, from: string): string {
  if (key === from) return "Today";
  if (key === addDays(from, 1)) return "Tomorrow";
  // Single weekday short name from the calendar parts.
  const [y, m, d] = parts(key);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleString("en-US", {
    weekday: "short",
    timeZone: "UTC",
  });
}

/** "Tomorrow 08:30 IST" / "Today 08:30 IST" / "Mon 08:30 IST". */
export function nextBriefLabel(now: Date = new Date()): string {
  const today = currentTradingDay(now);
  const next = nextTradingDay(today);
  return `${relativeWeekday(next, today)} 08:30 IST`;
}

/**
 * Edition number: count of NSE trading days from LAUNCH_KEY through
 * the given key (inclusive). Formatted as 3-digit "#006".
 */
export function editionNumber(key: string = currentTradingDay()): number {
  let n = 0;
  let k = LAUNCH_KEY;
  while (k <= key) {
    if (!isWeekend(k)) n++;
    k = addDays(k, 1);
  }
  return Math.max(1, n);
}

export function formatEdition(n: number): string {
  return "#" + String(n).padStart(3, "0");
}

/** Bundle the four datelines a marketing component usually needs. */
export type Dateline = {
  todayKey: string;
  yesterdayKey: string;
  nextKey: string;
  todayLong: string;       // "06 JUN 2026"
  todayShort: string;      // "06 JUN"
  yesterdayShort: string;  // "05 JUN"
  edition: string;         // "#006"
  editionNumber: number;
  nextBriefLabel: string;  // "Tomorrow 08:30 IST"
};

export function buildDateline(now: Date = new Date()): Dateline {
  const today = currentTradingDay(now);
  const yesterday = previousTradingDay(today);
  const next = nextTradingDay(today);
  const n = editionNumber(today);
  return {
    todayKey: today,
    yesterdayKey: yesterday,
    nextKey: next,
    todayLong: formatLong(today),
    todayShort: formatShort(today),
    yesterdayShort: formatShort(yesterday),
    edition: formatEdition(n),
    editionNumber: n,
    nextBriefLabel: nextBriefLabel(now),
  };
}
