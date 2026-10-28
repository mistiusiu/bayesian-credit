import { useState } from 'react';
import { usePosteriors, useSimArms } from '@/api/client';
import type { Simulation } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { Legend } from '@/charts/Legend';
import { PosteriorBands } from '@/charts/PosteriorBands';
import { Select } from '@/components/ui/Field';
import { Segmented } from '@/components/ui/Segmented';
import { QueryState } from '@/components/ui/States';
import { leadPolicy } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { seriesColor } from '@/lib/palette';

/** Gaussian arm posteriors N(μ, σ²) of the selected group over time (proposal scale σ, or σ/√n). */
export function PosteriorPanel({ sim }: { sim: Simulation }) {
  const policy = leadPolicy(sim.config.policies) ?? sim.config.policies[0];
  const groups = useSimArms(sim.id, sim, { policy }).data?.groups ?? [];
  const [group, setGroup] = useState<string | null>(null);
  const [scale, setScale] = useState<'sigma' | 'se'>('se');
  const g = group ?? groups[0]?.group ?? null;
  const q = usePosteriors(sim.id, sim, { policy, group: g });
  return (
    <ChartFrame
      title="Arm posteriors for one borrower group"
      caption={
        scale === 'sigma'
          ? 'μ ± σ with the proposal scale: σ is the reward dispersion and stays wide (that width is what keeps Thompson exploring); μ moves from the risk-model warm start toward the realized rewards.'
          : 'μ ± σ/√n (standard error): uncertainty about the mean shrinks as effective observations accumulate.'
      }
      actions={
        <div className="flex flex-wrap items-center gap-2">
          <Select className="h-7 w-56 text-xs" aria-label="Borrower group" value={g ?? ''} onValue={(v) => setGroup(v)} options={groups.slice(0, 30).map((x) => ({ value: x.group, label: `${fmt.group(x.group)} (${fmt.compact(x.n)})` }))} />
          <Segmented<'sigma' | 'se'> size="sm" label="Band" value={scale} onChange={setScale} options={[{ value: 'se', label: 'σ/√n' }, { value: 'sigma', label: 'σ (proposal)' }]} />
        </div>
      }
      yLabel="Posterior mean reward (USD per loan)"
      xLabel="Epoch"
    >
      <QueryState query={q} loadingLabel="Loading posteriors">
        {(h) => {
          const epochs = [...new Set(h.rows.map((r) => r.epoch))];
          const series = h.arms.map((a, k) => {
            const rows = epochs.map((e) => h.rows.find((r) => r.epoch === e && r.arm === a.arm) ?? null);
            const band = (r: (typeof rows)[number]) => (r ? (scale === 'sigma' ? r.sigma : r.sigma / Math.sqrt(Math.max(r.n, 1))) : 0);
            return {
              key: `a${a.arm}`,
              label: fmt.arm(a),
              color: seriesColor(k + 1),
              mu: rows.map((r) => r?.mu ?? null),
              lo: rows.map((r) => (r ? r.mu - band(r) : null)),
              hi: rows.map((r) => (r ? r.mu + band(r) : null)),
            };
          });
          return (
            <div className="space-y-1">
              <Legend className="px-2" items={series.map((s) => ({ key: s.key, label: s.label, color: s.color }))} />
              <PosteriorBands epochs={epochs} series={series} height={260} />
            </div>
          );
        }}
      </QueryState>
    </ChartFrame>
  );
}
