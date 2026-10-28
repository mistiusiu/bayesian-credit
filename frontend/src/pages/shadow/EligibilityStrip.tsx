import type { ShadowReport } from '@/api/types';
import { Badge } from '@/components/ui/Badge';
import { GUARDRAIL_INFO, guardrailValue } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { armViolations } from '@/lib/guardrails';

/** Which arms the guardrails admit for this borrower (above the arm heatmap). */
export function EligibilityStrip({ report: r }: { report: ShadowReport }) {
  const e = r.eligibility;
  const ctx = { annual_inc: r.context.annual_inc, dti: r.context.dti };
  const lend = r.arms.filter((a) => !a.is_null);
  const why = new Map(lend.map((a) => [a.arm, armViolations(a, ctx, e.guardrails, e.reference_rate)]));
  const fallback = lend.filter((a) => a.eligible && a.arm !== r.historical_arm && (why.get(a.arm)?.length ?? 0) > 0);
  const c = e.guardrails.rate_corridor;
  return (
    <section aria-label="Admissible arms for this borrower" className="rounded-lg border border-line bg-surface px-4 py-2.5 text-xs">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <span>
          <span className="text-muted">Admissible</span> <strong className="num text-[15px] font-semibold text-ink">{fmt.int(e.n_eligible_lending)}</strong>
          <span className="text-muted"> of {fmt.int(lend.length)} lending arms, plus decline</span>
        </span>
        <span className="num">
          <span className="text-muted">Reference rate</span> <strong className="text-ink">{fmt.pct(e.reference_rate, 2)}</strong>
          {r.context.sub_grade && <span className="text-muted"> ({r.context.sub_grade} → grade → all)</span>}
        </span>
        {e.rate_corridor_applied && e.reference_rate !== null && c != null && (
          <span className="num">
            <span className="text-muted">Corridor</span> <strong className="text-ink">{fmt.pct(e.reference_rate - c, 1)}–{fmt.pct(e.reference_rate + c, 1)}</strong>
          </span>
        )}
        <ul className="flex flex-wrap gap-x-3 gap-y-1" aria-label="Arms each guardrail excludes">
          {GUARDRAIL_INFO.map((g) => {
            const on = e.guardrails[g.key] != null;
            const n = lend.filter((a) => why.get(a.arm)?.some((v) => v.key === g.key)).length;
            return (
              <li key={g.key} title={g.blurb} className="rounded-full border border-line px-2 py-0.5">
                <span className="text-muted">{g.short}</span> <span className="num text-ink">{guardrailValue(e.guardrails[g.key], g.unit)}</span>
                {on && <span className="num text-muted"> · excludes {fmt.int(n)}</span>}
              </li>
            );
          })}
        </ul>
        {e.historical_admitted && <Badge tone="warning" title="The arm Lending Club issued fails a guardrail; it stays eligible so the comparison is defined">Historical arm admitted despite the guardrails</Badge>}
        {fallback.length > 0 && <Badge tone="warning" title="No lending arm passes; the arm closest to the reference rate with the smallest instalment is kept">Fallback arm only: {fallback.map((a) => fmt.arm(a)).join(', ')}</Badge>}
      </div>
    </section>
  );
}
