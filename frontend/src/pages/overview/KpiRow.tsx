import type { Dataset, Fit, Simulation } from '@/api/types';
import { StatTile } from '@/components/ui/StatTile';
import { fmt } from '@/lib/format';
import { LEVEL_TONE } from '@/lib/palette';

export function KpiRow({ dataset, fit, sim }: { dataset: Dataset | null; fit: Fit | null; sim: Simulation | null }) {
  const m = fit?.metrics;
  const hist = sim?.headline.find((h) => h.policy === 'historical');
  const upliftPct = sim?.uplift_vs_historical != null && hist ? sim.uplift_vs_historical / Math.abs(hist.cum_reward || 1) : null;
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
      <StatTile label="Loans" value={fmt.compact(dataset?.n_loans)} hint={dataset?.issue_min ? `${fmt.epochShort(dataset.issue_min)} – ${fmt.epochShort(dataset.issue_max)}` : dataset ? fmt.label(dataset.status) : 'no dataset'} />
      <StatTile label="Default rate (resolved)" value={fmt.pct(dataset?.default_rate)} hint={dataset?.n_resolved ? `${fmt.compact(dataset.n_resolved)} resolved (${fmt.pct(dataset.resolved_share, 0)})` : undefined} />
      <StatTile
        label="Brier score"
        value={fmt.num(m?.brier, 4)}
        delta={m ? { text: `${fmt.numSigned(m.brier - m.brier_baseline, 4)} vs base rate`, good: m.brier < m.brier_baseline } : null}
        hint={m ? undefined : fit ? fmt.label(fit.status) : 'no fit'}
      />
      <StatTile label="ROC-AUC" value={fmt.num(m?.auc, 3)} hint={m ? `holdout n ${fmt.compact(m.n_test)}` : undefined} />
      <StatTile
        label="Reward uplift vs historical"
        value={fmt.usdCompactSigned(sim?.uplift_vs_historical)}
        delta={upliftPct !== null ? { text: fmt.pctSigned(upliftPct, 0), good: upliftPct > 0 } : null}
        hint={sim ? 'cumulative' : 'no simulation'}
      />
      <StatTile
        label="PSI now"
        value={fmt.num(sim?.psi_aggregate, 3)}
        status={sim ? { tone: LEVEL_TONE[sim.sentinel_level], label: sim.sentinel_level } : null}
        hint={sim ? `${sim.current_epoch ? fmt.epoch(sim.current_epoch) : ''} · breaker ${sim.breaker_state.replace('_', '-')}` : undefined}
      />
    </div>
  );
}
