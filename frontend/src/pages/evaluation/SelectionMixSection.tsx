import type { EvaluationReport, Simulation } from '@/api/types';
import { SelectionMixTable } from '@/components/counterfactual/SelectionMixTable';
import { SectionTitle } from '@/components/ui/Card';
import { fmt } from '@/lib/format';
import { selectionMixRows, windowLabel } from '@/lib/selectionMix';

/** Bandit performance detail: each policy's decisions over the last 6 epochs. */
export function SelectionMixSection({ report, sim }: { report: EvaluationReport; sim: Simulation }) {
  const rows = selectionMixRows(report, sim.config.policies);
  if (!rows.length) return null;
  const w = windowLabel(report, fmt.epoch);
  return (
    <section aria-label="Selection mix" className="space-y-2">
      <SectionTitle>Bandit performance · selection mix, last 6 epochs</SectionTitle>
      <p className="text-xs text-muted">
        {w ? `${w}. ` : ''}Lending decisions by term, rate bucket and amount bucket; the top-3 share measures how concentrated each policy has become.
      </p>
      <div className="rounded-lg border border-line bg-surface p-3">
        <SelectionMixTable rows={rows} caption="Selection mix by policy, last 6 epochs" />
      </div>
    </section>
  );
}
