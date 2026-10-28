/**
 * Lending guardrails G1–G4 (docs/MATH.md §2.7), re-checked in the browser to explain why an
 * arm is not admissible. The server's `eligible` flag stays authoritative: the fallback arm
 * and an admitted historical arm are eligible even when they fail a guardrail.
 */
import type { Guardrails } from '@/api/types';
import { fmt } from './format';

export interface Violation {
  key: keyof Guardrails;
  text: string;
}

const TOL = 1e-9;

export function armViolations(
  arm: { amount: number; rate: number; installment: number; is_null: boolean },
  ctx: { annual_inc: number | null; dti: number | null },
  g: Guardrails,
  referenceRate: number | null,
): Violation[] {
  if (arm.is_null) return [];
  const out: Violation[] = [];
  const inc = ctx.annual_inc;
  const knownInc = typeof inc === 'number' && Number.isFinite(inc) && inc > 0;
  if (knownInc && g.max_payment_to_income != null && arm.installment > (g.max_payment_to_income * inc) / 12 + TOL) {
    out.push({ key: 'max_payment_to_income', text: `instalment is ${fmt.pct((12 * arm.installment) / inc, 0)} of monthly income (cap ${fmt.pct(g.max_payment_to_income, 0)})` });
  }
  if (knownInc && g.max_amount_to_income != null && arm.amount > g.max_amount_to_income * inc + TOL) {
    out.push({ key: 'max_amount_to_income', text: `amount is ${fmt.pct(arm.amount / inc, 0)} of annual income (cap ${fmt.pct(g.max_amount_to_income, 0)})` });
  }
  const dti = ctx.dti;
  if (knownInc && g.max_total_dti != null && typeof dti === 'number' && Number.isFinite(dti) && dti >= 0) {
    const total = dti / 100 + (12 * arm.installment) / inc;
    if (total > g.max_total_dti + TOL) out.push({ key: 'max_total_dti', text: `total DTI with this loan ${fmt.pct(total, 0)} (ceiling ${fmt.pct(g.max_total_dti, 0)})` });
  }
  if (referenceRate !== null && g.rate_corridor != null && Math.abs(arm.rate - referenceRate) > g.rate_corridor + 1e-6) {
    out.push({ key: 'rate_corridor', text: `rate ${fmt.rate(arm.rate)} outside ${fmt.pct(referenceRate - g.rate_corridor, 1)}–${fmt.pct(referenceRate + g.rate_corridor, 1)}` });
  }
  return out;
}

/** Index range [from, to] of the sorted grid rates inside reference ± corridor (null when none or no corridor). */
export function corridorColumns(rates: number[], referenceRate: number | null, corridor: number | null): { from: number; to: number } | null {
  if (referenceRate === null || corridor === null) return null;
  const inside = rates.flatMap((r, i) => (Math.abs(r - referenceRate) <= corridor + 1e-6 ? [i] : []));
  return inside.length ? { from: inside[0], to: inside[inside.length - 1] } : null;
}

/** Number of guardrails switched on. */
export const activeGuardrails = (g: Guardrails | null | undefined) => (g ? Object.values(g).filter((v) => v !== null && v !== undefined).length : 0);
