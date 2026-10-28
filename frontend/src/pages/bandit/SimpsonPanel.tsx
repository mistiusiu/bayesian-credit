import { useSimpson } from '@/api/client';
import type { Simulation } from '@/api/types';
import { ChartFrame } from '@/charts/ChartFrame';
import { Dumbbell } from '@/charts/Dumbbell';
import { Legend } from '@/charts/Legend';
import { QueryState } from '@/components/ui/States';
import { leadPolicy } from '@/lib/domain';
import { fmt } from '@/lib/format';

/** Aggregation-bias check: pooled-best arm vs group-best arm, per group (Simpson's paradox). */
export function SimpsonPanel({ sim }: { sim: Simulation }) {
  const policy = leadPolicy(sim.config.policies) ?? sim.config.policies[0];
  const q = useSimpson(sim.id, sim, { policy });
  return (
    <QueryState query={q} loadingLabel="Comparing pooled and group-level choices">
      {(s) => {
        const rows = s.groups.slice(0, 14);
        return (
          <ChartFrame
            title="Pooled vs group-best arm (Simpson's paradox)"
            caption={`Pooling all borrowers picks ${s.pooled_best_label}; it is the best arm in only ${s.n_agree} of ${s.n_groups} groups. Applying it everywhere forfeits ${fmt.usd(s.forfeited_per_decision)} per decision; the hierarchy keeps group-level choices.`}
            legend={
              <Legend
                items={[
                  { key: 'p', label: `Pooled-best arm in the group (${s.pooled_best_label})`, color: 'var(--text-axis)', kind: 'dot' },
                  { key: 'g', label: 'Group-best arm', color: 'var(--series-1)', kind: 'dot' },
                ]}
              />
            }
            xLabel={`Posterior mean reward per loan (USD), ${fmt.epoch(s.epoch)}`}
            table={{
              columns: [
                { key: 'g', label: 'Group' },
                { key: 'n', label: 'Decisions', align: 'right' },
                { key: 'b', label: 'Group-best arm' },
                { key: 'bm', label: 'μ group-best', align: 'right' },
                { key: 'pm', label: 'μ pooled arm', align: 'right' },
                { key: 'gap', label: 'Gap', align: 'right' },
              ],
              rows: s.groups.map((r) => ({ g: fmt.group(r.group), n: fmt.int(r.n), b: r.group_best_label, bm: fmt.usd(r.group_best_mu), pm: fmt.usd(r.pooled_arm_mu), gap: fmt.usd(r.gap) })),
            }}
          >
            <div className="px-2">
              <Dumbbell rows={rows.map((r) => ({ key: r.group, label: fmt.group(r.group), a: r.pooled_arm_mu, b: r.group_best_mu, note: r.group_best_label }))} aLabel="pooled-best arm" bLabel="group-best arm" />
            </div>
          </ChartFrame>
        );
      }}
    </QueryState>
  );
}
