/** Parsing helpers for the arm-grid editor (amounts in USD, rates typed as percents). */
export function parseList(s: string): number[] {
  return s
    .split(/[\s,;]+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => Number.isFinite(n));
}

export function parseAmounts(s: string): { values: number[]; error: string | null } {
  const values = [...new Set(parseList(s).map((v) => (v < 1000 ? v * 1000 : v)))].sort((a, b) => a - b);
  if (!values.length) return { values, error: 'At least one amount' };
  if (values.some((v) => v <= 0 || v > 100000)) return { values, error: 'Amounts must be in (0, 100k]' };
  return { values, error: null };
}

/** Rates are typed as percents ("7, 10, 13"); values ≤ 1 are taken as fractions. */
export function parseRates(s: string): { values: number[]; error: string | null } {
  const values = [...new Set(parseList(s).map((v) => (v > 1 ? v / 100 : v)))].map((v) => Math.round(v * 10000) / 10000).sort((a, b) => a - b);
  if (!values.length) return { values, error: 'At least one rate' };
  if (values.some((v) => v <= 0 || v >= 1)) return { values, error: 'Rates must be in (0%, 100%)' };
  return { values, error: null };
}
