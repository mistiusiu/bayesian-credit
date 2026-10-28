import type { ShadowReport } from '@/api/types';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { StatTile } from '@/components/ui/StatTile';
import { fmt } from '@/lib/format';

/** Recommended arm vs the historical policy (expected and, if resolved, realized). */
export function RecommendationCard({ report: r }: { report: ShadowReport }) {
  const rec = r.arms[r.recommended_arm];
  const hp = r.historical_policy;
  return (
    <Card
      title="Recommendation vs historical"
      subtitle="argmax of risk-adjusted reward among the admissible arms (guardrails applied)"
      actions={<Badge tone={r.lendable ? 'good' : 'warning'}>{r.lendable ? `Lendable (P(reward>0) ≥ ${fmt.pct(r.lendable_threshold, 0)})` : 'Not lendable at threshold'}</Badge>}
    >
      <div className="grid gap-3 md:grid-cols-3">
        <StatTile label="Shadow recommendation" value={fmt.arm(rec)} hint={`E[reward] ${fmt.usd(rec.reward_mean)} · P(>0) ${fmt.pct(rec.prob_positive, 0)}${rec.is_null ? '' : ` · ρ ${fmt.pct(rec.realization, 0)}`}`} />
        <StatTile
          label="Historical policy (Lending Club)"
          value={hp ? fmt.arm({ amount: hp.amount, rate: hp.rate, term: hp.term }) : '—'}
          hint={hp ? `snaps to arm ${r.historical_arm} · E[reward] ${fmt.usd(r.historical_expected_reward)}${r.eligibility.historical_admitted ? ' · fails the guardrails' : ''}` : 'context-only request'}
        />
        <StatTile
          label="Expected uplift"
          value={fmt.usdSigned(r.uplift_expected)}
          delta={r.uplift_expected !== null ? { text: r.uplift_expected >= 0 ? 'shadow ≥ historical' : 'historical better', good: r.uplift_expected >= 0 } : null}
        />
      </div>
      <div className="num mt-3 grid gap-x-6 gap-y-1 rounded-md bg-surface-2 px-3 py-2 text-xs md:grid-cols-2">
        <span className="flex justify-between gap-2">
          <span className="text-muted">Historical outcome</span>
          <span className="text-ink">{r.outcome ? `${r.outcome.loan_status}${r.outcome.months_on_book !== null ? ` · ${r.outcome.months_on_book} months on book` : ''}` : '—'}</span>
        </span>
        <span className="flex justify-between gap-2">
          <span className="text-muted">Realized historical reward</span>
          <span className="text-ink">{r.historical_realized_reward === null ? 'unresolved' : fmt.usdSigned(r.historical_realized_reward)}</span>
        </span>
        <span className="flex justify-between gap-2">
          <span className="text-muted">Uplift vs realized</span>
          <span className="text-ink">{fmt.usdSigned(r.uplift_vs_realized)}</span>
        </span>
        <span className="flex justify-between gap-2">
          <span className="text-muted">Recovery if defaulted</span>
          <span className="text-ink">{r.outcome?.recovery_rate != null ? fmt.pct(r.outcome.recovery_rate) : '—'}</span>
        </span>
      </div>
      <p className="mt-2 text-2xs text-muted">Realized values are post-hoc evaluation of the arm that was actually issued; counterfactual arms have predictive distributions only.</p>
    </Card>
  );
}
