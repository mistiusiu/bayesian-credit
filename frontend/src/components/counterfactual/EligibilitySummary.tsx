import type { PortfolioEligibility } from '@/api/types';
import { StatTile } from '@/components/ui/StatTile';
import { guardrailSummary } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { activeGuardrails } from '@/lib/guardrails';

/** Portfolio-level eligibility: guardrails in force, reference-rate coverage, admissible arms per borrower. */
export function EligibilitySummary({ e, nLendingArms }: { e: PortfolioEligibility; nLendingArms?: number | null }) {
  return (
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      <StatTile label="Guardrails in force" value={`${activeGuardrails(e.guardrails)} of 4`} hint={guardrailSummary(e.guardrails)} />
      <StatTile
        label="Reference rate found"
        value={fmt.pct(e.share_with_reference_rate, 0)}
        hint={e.reference_rates ? 'sub-grade → grade → global training mean' : 'fit has no reference rates: corridor off'}
      />
      <StatTile
        label="Admissible lending arms"
        value={fmt.num(e.mean_n_eligible_lending, 1)}
        hint={`mean per borrower${nLendingArms ? ` of ${fmt.int(nLendingArms)} lending arms` : ''}`}
      />
      <StatTile label="Only the fallback arm" value={fmt.pct(e.share_single_fallback_arm, 1)} hint="borrowers with ≤ 1 admissible lending arm" />
    </div>
  );
}
