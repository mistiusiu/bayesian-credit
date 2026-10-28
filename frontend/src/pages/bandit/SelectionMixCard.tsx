import { useReport } from '@/api/client';
import type { Simulation } from '@/api/types';
import { SelectionMixTable } from '@/components/counterfactual/SelectionMixTable';
import { Card } from '@/components/ui/Card';
import { QueryState } from '@/components/ui/States';
import { fmt } from '@/lib/format';
import { selectionMixRows, windowLabel } from '@/lib/selectionMix';

/** Where each policy's recent decisions land (term / rate / amount), from the report's bandit_performance block. */
export function SelectionMixCard({ sim }: { sim: Simulation }) {
  const q = useReport(sim.id, sim);
  return (
    <QueryState query={q} loadingLabel="Loading the selection mix" height={160}>
      {(r) => {
        const rows = selectionMixRows(r, sim.config.policies);
        const w = windowLabel(r, fmt.epoch);
        return (
          <Card
            title="Selection mix, last 6 epochs"
            subtitle={`${w ? `${w}: ` : ''}lending decisions by term, rate bucket and amount bucket; top-3 share = concentration on the three most chosen arms.${r.status === 'partial' ? ' Run in progress: the window moves with it.' : ''}`}
            bodyClassName="p-3"
          >
            {rows.length ? <SelectionMixTable rows={rows} caption="Selection mix by policy, last 6 epochs" /> : <p className="py-4 text-center text-xs text-muted">The report has no bandit_performance detail yet.</p>}
          </Card>
        );
      }}
    </QueryState>
  );
}
