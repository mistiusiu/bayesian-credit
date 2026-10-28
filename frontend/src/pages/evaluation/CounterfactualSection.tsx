import { useFit, usePortfolioRuns } from '@/api/client';
import type { EvaluationReport, PortfolioEligibility, RecommendationMix, RiskAversionPoint, UpliftDecomposition as Decomposition } from '@/api/types';
import { EligibilitySummary } from '@/components/counterfactual/EligibilitySummary';
import { HowToReadUplift } from '@/components/counterfactual/HowToReadUplift';
import { RecommendationMixPanel } from '@/components/counterfactual/RecommendationMixPanel';
import { RiskAversionSweep } from '@/components/counterfactual/RiskAversionSweep';
import { UpliftDecomposition } from '@/components/counterfactual/UpliftDecomposition';
import { InterestRealizationCard } from '@/components/fit/InterestRealizationCard';
import { PolicyEffectCard } from '@/components/fit/PolicyEffectCard';
import { SectionTitle } from '@/components/ui/Card';
import { StatTile } from '@/components/ui/StatTile';
import { upliftFraction } from '@/lib/counterfactual';
import { fmt } from '@/lib/format';

interface Blocks {
  origin: string;
  n: number | null;
  hist: number | null;
  shadow: number | null;
  uplift: number | null;
  upliftRisk: number | null;
  mix: RecommendationMix | null;
  decomposition: Decomposition | null;
  sweep: Record<string, RiskAversionPoint> | null;
  eligibility: PortfolioEligibility | null;
}

/** Counterfactual detail of the report (docs/API.md §13.1); falls back to the fit's latest portfolio run when the report has none. */
export function CounterfactualSection({ report }: { report: EvaluationReport }) {
  const fit = useFit(report.fit).data;
  const cf = report.details?.counterfactual_evaluation;
  const runs = usePortfolioRuns(report.fit, !cf?.available && fit?.status === 'ready');
  const latest = runs.data?.find((r) => r.summary);
  let b: Blocks | null = null;
  if (cf?.available) {
    b = {
      origin: 'From the report: the shadow portfolio summary stored with the run.',
      n: cf.n_loans ?? null,
      hist: cf.expected_profit_historical ?? null,
      shadow: cf.expected_profit_shadow ?? null,
      uplift: cf.uplift ?? null,
      upliftRisk: cf.uplift_risk_adjusted ?? null,
      mix: cf.recommendation_mix ?? null,
      decomposition: cf.uplift_decomposition ?? null,
      sweep: cf.by_risk_aversion ?? null,
      eligibility: cf.eligibility ?? null,
    };
  } else if (latest?.summary) {
    const s = latest.summary;
    b = {
      origin: `The report carries no shadow portfolio summary (${cf?.note ?? 'block absent'}); shown instead: the latest portfolio evaluation of fit #${report.fit}, job #${latest.job.id} (${fmt.int(latest.request.max_loans)} borrowers, seed ${latest.request.seed}).`,
      n: s.n_loans,
      hist: s.expected_profit_historical,
      shadow: s.expected_profit_shadow,
      uplift: s.uplift,
      upliftRisk: s.uplift_risk_adjusted ?? null,
      mix: s.recommendation_mix ?? null,
      decomposition: s.uplift_decomposition ?? null,
      sweep: s.by_risk_aversion ?? null,
      eligibility: s.eligibility ?? null,
    };
  }
  const realization = cf?.interest_realization ?? fit?.metrics?.interest_realization;
  const nLending = fit?.n_arms != null ? fit.n_arms - (fit.config.arm_space.include_null ? 1 : 0) : null;
  return (
    <section aria-label="Counterfactual evaluation detail" className="space-y-3">
      <SectionTitle>Counterfactual evaluation · shadow engine detail</SectionTitle>
      {b ? (
        <>
          <p className="text-xs text-muted">{b.origin}</p>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <StatTile label="Borrowers evaluated" value={fmt.int(b.n)} />
            <StatTile label="Expected profit · historical" value={fmt.usdCompact(b.hist)} />
            <StatTile label="Expected profit · shadow" value={fmt.usdCompact(b.shadow)} />
            <StatTile label="Uplift" value={fmt.usdCompactSigned(b.uplift)} delta={b.uplift !== null ? { text: fmt.pctSigned(upliftFraction(b.uplift, b.hist), 0), good: b.uplift >= 0 } : null} />
            <StatTile label="Risk-adjusted uplift" value={fmt.usdCompactSigned(b.upliftRisk)} hint="Σ score(recommended) − score(historical)" />
          </div>
          <HowToReadUplift />
          {(b.decomposition || b.sweep) && (
            <div className="grid gap-4 2xl:grid-cols-2">
              {b.decomposition && <UpliftDecomposition d={b.decomposition} />}
              {b.sweep && <RiskAversionSweep points={b.sweep} configured={fit?.config.reward.risk_aversion} />}
            </div>
          )}
          {b.mix && <RecommendationMixPanel mix={b.mix} />}
          {b.eligibility && <EligibilitySummary e={b.eligibility} nLendingArms={nLending} />}
        </>
      ) : (
        <p className="rounded-lg border border-line bg-surface px-4 py-3 text-xs text-muted">
          The report has no shadow portfolio summary ({cf?.note ?? 'block absent'}) and fit #{report.fit} has no finished portfolio evaluation: run one on the Shadow engine page (Portfolio tab).
        </p>
      )}
      <SectionTitle>Risk-model extras · fit #{report.fit}</SectionTitle>
      <div className="grid gap-4 xl:grid-cols-2">
        <PolicyEffectCard pe={fit?.metrics?.policy_effect} />
        <InterestRealizationCard ir={realization} />
      </div>
    </section>
  );
}
