import type { Fit, InclusionRow, PortfolioLoanRow, PortfolioOutcomeClass, PortfolioResult } from '@/api/types';
import { HBars } from '@/charts/Bars';
import { ChartFrame } from '@/charts/ChartFrame';
import { GroupedBars } from '@/charts/GroupedBars';
import { Legend } from '@/charts/Legend';
import { EligibilitySummary } from '@/components/counterfactual/EligibilitySummary';
import { HowToReadUplift } from '@/components/counterfactual/HowToReadUplift';
import { RecommendationMixPanel } from '@/components/counterfactual/RecommendationMixPanel';
import { RiskAversionSweep } from '@/components/counterfactual/RiskAversionSweep';
import { UpliftDecomposition } from '@/components/counterfactual/UpliftDecomposition';
import { Card, SectionTitle } from '@/components/ui/Card';
import { DataTable } from '@/components/ui/DataTable';
import { StatTile } from '@/components/ui/StatTile';
import { fmt } from '@/lib/format';
import { policyColor } from '@/lib/palette';

/** Classes in precedence order (each borrower takes the first that applies). */
const CLASSES: { key: PortfolioOutcomeClass; label: string; note: string }[] = [
  { key: 'declined', label: 'Declined', note: 'no admissible arm beats not lending' },
  { key: 'rescued', label: 'Rescued', note: 'historical loan lost money; shadow arm has positive expected reward' },
  { key: 'upsized', label: 'Upsized', note: 'larger amount than issued, positive reward' },
  { key: 'downsized', label: 'Downsized', note: 'smaller amount than issued' },
  { key: 'repriced_down', label: 'Repriced down', note: 'lower rate than issued' },
  { key: 'repriced_up', label: 'Repriced up', note: 'higher rate than issued' },
  { key: 'unchanged', label: 'Unchanged', note: 'shadow agrees with the historical arm' },
];

export function PortfolioResultView({ result, fit }: { result: PortfolioResult; fit: Fit }) {
  const s = result.summary!;
  const nLending = fit.n_arms !== null && fit.n_arms !== undefined ? fit.n_arms - (fit.config.arm_space.include_null ? 1 : 0) : null;
  const series = [
    { key: 'historical', label: 'Historical policy', color: policyColor('historical') },
    { key: 'shadow', label: 'Shadow engine', color: 'var(--series-1)' },
  ];
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-6">
        <StatTile label="Borrowers evaluated" value={fmt.int(s.n_loans)} hint={`job #${result.job.id}`} />
        <StatTile label="Lendable share" value={fmt.pct(s.share_lendable)} hint={`${fmt.int(s.n_lendable)} borrowers`} />
        <StatTile label="Expected profit · historical" value={fmt.usdCompact(s.expected_profit_historical)} />
        <StatTile label="Expected profit · shadow" value={fmt.usdCompact(s.expected_profit_shadow)} />
        <StatTile
          label="Uplift"
          value={fmt.usdCompactSigned(s.uplift)}
          delta={{ text: fmt.pctSigned(s.uplift_pct, 0), good: s.uplift >= 0 }}
          hint={s.uplift_risk_adjusted != null ? `risk-adjusted ${fmt.usdCompactSigned(s.uplift_risk_adjusted)}` : undefined}
        />
        <StatTile label="Realized · historical" value={fmt.usdCompactSigned(s.realized_profit_historical)} hint={s.n_realized != null ? `${fmt.int(s.n_realized)} resolved loans` : 'resolved loans only'} />
      </div>
      <HowToReadUplift />
      {(s.uplift_decomposition || s.by_risk_aversion) && (
        <div className="grid gap-4 2xl:grid-cols-2">
          {s.uplift_decomposition && <UpliftDecomposition d={s.uplift_decomposition} />}
          {s.by_risk_aversion && <RiskAversionSweep points={s.by_risk_aversion} configured={fit.config.reward.risk_aversion} />}
        </div>
      )}
      {s.recommendation_mix && <RecommendationMixPanel mix={s.recommendation_mix} />}
      {s.eligibility && (
        <section aria-label="Eligibility" className="space-y-2">
          <SectionTitle>Eligibility under the guardrails</SectionTitle>
          <EligibilitySummary e={s.eligibility} nLendingArms={nLending} />
        </section>
      )}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <ChartFrame
          title="Expected profit by grade"
          caption="Sum of expected risk-adjusted reward per grade: the arm Lending Club issued vs the shadow recommendation for the same borrowers."
          legend={<Legend items={series.map((x) => ({ ...x, kind: 'rect' as const }))} />}
          yLabel="USD, expected"
          xLabel="Grade"
          table={{
            columns: [{ key: 'g', label: 'Grade' }, { key: 'n', label: 'n', align: 'right' }, { key: 'h', label: 'Historical', align: 'right' }, { key: 's', label: 'Shadow', align: 'right' }, { key: 'r', label: 'Realized hist.', align: 'right' }],
            rows: result.by_grade.map((g) => ({ g: g.grade, n: fmt.int(g.n), h: fmt.usd(g.expected_profit_historical), s: fmt.usd(g.expected_profit_shadow), r: fmt.usd(g.realized_profit_historical) })),
          }}
        >
          <GroupedBars
            data={result.by_grade.map((g) => ({ x: g.grade, historical: g.expected_profit_historical, shadow: g.expected_profit_shadow }))}
            series={series}
            format={fmt.usd}
            yTick={fmt.usdCompact}
            height={260}
          />
        </ChartFrame>
        <ChartFrame
          title="What the shadow engine changed"
          caption="Each borrower is classified once, by the first class that applies in this order (declined → rescued → upsized → downsized → repriced down → repriced up → unchanged)."
          table={{ columns: [{ key: 'c', label: 'Class' }, { key: 'n', label: 'Borrowers', align: 'right' }, { key: 'd', label: 'Meaning' }], rows: CLASSES.map((c) => ({ c: c.label, n: fmt.int(s.counts[c.key] ?? 0), d: c.note })) }}
        >
          <div className="px-2 pt-1 pb-2">
            <HBars rows={CLASSES.map((c) => ({ label: c.label, value: s.counts[c.key] ?? 0, note: c.note }))} format={(v) => `${fmt.int(v)} (${fmt.pct(v / Math.max(s.n_loans, 1), 0)})`} />
          </div>
        </ChartFrame>
      </div>
      <Card title="Inclusion by grade" subtitle="Who gets credit, at what price: share lendable, decline rate and mean offered terms, historical vs shadow" bodyClassName="p-0">
        <DataTable<InclusionRow>
          caption="Inclusion table"
          rows={result.inclusion}
          rowKey={(r) => r.grade}
          columns={[
            { key: 'g', header: 'Grade', render: (r) => r.grade },
            { key: 'n', header: 'Borrowers', align: 'right', render: (r) => fmt.int(r.n) },
            { key: 'l', header: 'Lendable', align: 'right', render: (r) => fmt.pct(r.share_lendable) },
            { key: 'd', header: 'Shadow declines', align: 'right', render: (r) => fmt.pct(r.decline_rate_shadow) },
            { key: 'rh', header: 'Rate · historical', align: 'right', render: (r) => fmt.pct(r.mean_rate_historical) },
            { key: 'rs', header: 'Rate · shadow', align: 'right', render: (r) => fmt.pct(r.mean_rate_shadow) },
            { key: 'ah', header: 'Amount · historical', align: 'right', render: (r) => fmt.usd(r.mean_amount_historical) },
            { key: 'as', header: 'Amount · shadow', align: 'right', render: (r) => fmt.usd(r.mean_amount_shadow) },
          ]}
        />
      </Card>
      <Card title="Largest differences" subtitle="Top borrowers by |shadow − historical| expected reward" bodyClassName="p-0">
        <DataTable<PortfolioLoanRow>
          caption="Largest differences"
          rows={result.top_loans}
          rowKey={(r) => r.loan_id}
          maxHeight={360}
          columns={[
            { key: 'id', header: 'Loan id', render: (r) => <code className="text-xs">{r.loan_id}</code> },
            { key: 'g', header: 'Group', render: (r) => <span className="text-ink-2">{fmt.group(r.group)}</span> },
            { key: 'h', header: 'Historical arm', render: (r) => r.historical_label },
            { key: 's', header: 'Shadow arm', render: (r) => r.recommended_label },
            { key: 'eh', header: 'E[R] hist.', align: 'right', render: (r) => fmt.usd(r.historical_expected_reward) },
            { key: 'es', header: 'E[R] shadow', align: 'right', render: (r) => fmt.usd(r.shadow_expected_reward) },
            { key: 'r', header: 'Realized hist.', align: 'right', render: (r) => fmt.usdSigned(r.realized_reward) },
            { key: 'c', header: 'Class', render: (r) => fmt.label(r.class) },
          ]}
        />
      </Card>
    </div>
  );
}
