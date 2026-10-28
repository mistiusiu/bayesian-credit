/** "YYYY-MM" epoch arithmetic. */

export function epochToIndex(e: string): number {
  const [y, m] = e.split('-').map(Number);
  return y * 12 + (m - 1);
}

export function indexToEpoch(i: number): string {
  const y = Math.floor(i / 12);
  const m = (i % 12) + 1;
  return `${y}-${String(m).padStart(2, '0')}`;
}

export function epochAdd(e: string, months: number): string {
  return indexToEpoch(epochToIndex(e) + months);
}

/** Months from a to b (b − a). */
export function epochDiff(a: string, b: string): number {
  return epochToIndex(b) - epochToIndex(a);
}

/** Inclusive range of epochs. */
export function epochRange(from: string, to: string): string[] {
  const out: string[] = [];
  for (let i = epochToIndex(from); i <= epochToIndex(to); i++) out.push(indexToEpoch(i));
  return out;
}

export function isEpoch(s: string | null | undefined): s is string {
  return !!s && /^\d{4}-(0[1-9]|1[0-2])$/.test(s);
}
