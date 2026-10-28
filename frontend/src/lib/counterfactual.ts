/**
 * Helpers for the counterfactual extras of docs/API.md §13.1 (portfolio summary, report
 * details) and the fit's prepayment table. Shared by the Shadow portfolio tab, the Bandit
 * lab and the Evaluation page.
 */
import type { InterestRealization, RiskAversionPoint, UpliftDecomposition } from '@/api/types';

/**
 * Uplift as a fraction of |historical expected profit|. Derived from the dollar amounts so
 * it is right whatever unit a block's own `uplift_pct` uses (the portfolio endpoint serves
 * fractions; the report's `details` pass the core's percent through).
 */
export function upliftFraction(uplift: number | null | undefined, historical: number | null | undefined): number | null {
  if (typeof uplift !== 'number' || typeof historical !== 'number' || !Number.isFinite(uplift) || !Number.isFinite(historical)) return null;
  return uplift / Math.max(Math.abs(historical), 1);
}

export const TERM_BUCKETS = ['36', '60'];
export const RATE_BUCKETS = ['<10%', '10-15%', '15-20%', '20-25%', '>=25%'];
export const AMOUNT_BUCKETS = ['<$10k', '$10k-20k', '$20k-30k', '>=$30k'];

/** "10-15%" → "10–15%", ">=25%" → "≥ 25%", "36" → "36m". */
export function bucketLabel(b: string): string {
  if (/^\d+$/.test(b)) return `${b}m`;
  return b.replace('>=', '≥ ').replace(/^</, '< ').replace('-', '–');
}

/** λ keys of `by_risk_aversion`, numerically sorted ("0.0", "0.5", "1.0"). */
export function lambdaKeys(r: Record<string, RiskAversionPoint> | null | undefined): string[] {
  return Object.keys(r ?? {}).sort((a, b) => Number(a) - Number(b));
}

export interface WaterfallStep {
  key: string;
  label: string;
  /** `total` bars start at 0; `delta` bars float from the previous level. */
  kind: 'total' | 'delta';
  from: number;
  to: number;
  note: string;
}

/** Historical → + repricing → + resizing → + term choice → shadow (unrestricted). */
export function decompositionSteps(d: UpliftDecomposition): WaterfallStep[] {
  const h = d.expected_profit_historical;
  const r1 = d.repricing_only.expected_profit;
  const r2 = d.repricing_and_resizing.expected_profit;
  const u = d.unrestricted.expected_profit;
  return [
    { key: 'historical', label: 'Historical', kind: 'total', from: 0, to: h, note: 'expected profit of the arms Lending Club issued' },
    { key: 'reprice', label: '+ Repricing', kind: 'delta', from: h, to: r1, note: 'best admissible rate, keeping the historical amount and term' },
    { key: 'resize', label: '+ Resizing', kind: 'delta', from: r1, to: r2, note: 'best admissible rate and amount, keeping the historical term' },
    { key: 'term', label: '+ Term choice', kind: 'delta', from: r2, to: u, note: 'term chosen too (36 or 60 months)' },
    { key: 'shadow', label: 'Shadow', kind: 'total', from: 0, to: u, note: 'unrestricted shadow recommendation' },
  ];
}

export type RealizationSource = 'cell' | 'term' | 'global' | 'contract';

/** ρ(grade, term) with the documented fallback (grade, term) → term → global (MATH §2.8). */
export function realizationFor(ir: InterestRealization, grade: string, term: string): { value: number; source: RealizationSource } {
  if (ir.prepayment_adjustment === null) return { value: 1, source: 'contract' };
  const cell = ir.by_grade_term[grade]?.[term];
  if (typeof cell === 'number') return { value: cell, source: 'cell' };
  const t = ir.by_term[term];
  if (typeof t === 'number') return { value: t, source: 'term' };
  return { value: ir.global, source: 'global' };
}
