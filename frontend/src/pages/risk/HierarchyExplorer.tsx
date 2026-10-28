import { useState } from 'react';
import { useGroups } from '@/api/client';
import type { Fit, HierarchyRow } from '@/api/types';
import { IntervalBar } from '@/charts/IntervalBar';
import { Card } from '@/components/ui/Card';
import { Chips } from '@/components/ui/Chips';
import { DataTable } from '@/components/ui/DataTable';
import { Select } from '@/components/ui/Field';
import { Meter } from '@/components/ui/Progress';
import { QueryState } from '@/components/ui/States';
import { fmt } from '@/lib/format';
import { useParam } from '@/lib/params';
import { GroupDetail } from './GroupDetail';

const dimValues = (rows: HierarchyRow[], dim: string) => [...new Set(rows.map((r) => r.dims[dim]).filter(Boolean))].sort();

/** Global → group → leaf explorer: posterior mean with 95% HDI on a shared axis, shrinkage n/(n+κ). */
export function HierarchyExplorer({ fit }: { fit: Fit }) {
  const q = useGroups(fit.id, true);
  const [group, setGroup] = useParam('group', null);
  const [filters, setFilters] = useState<Record<string, string[]>>({});
  const dims = fit.config.segmentation.dimensions;
  return (
    <QueryState query={q} loadingLabel="Loading hierarchy">
      {(rows) => {
        const global = rows.find((r) => r.level === 'global');
        const groups = rows.filter((r) => r.level === 'group');
        const shown = groups.filter((g) => dims.every((d) => !filters[d]?.length || filters[d].includes(g.dims[d])));
        const max = Math.min(1, Math.ceil(Math.max(...groups.map((g) => g.hdi_high), 0.1) * 20) / 20);
        return (
          <div className="space-y-4">
            {global && (
              <div className="num flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border border-line bg-surface px-4 py-3 text-xs">
                <span className="font-semibold text-ink">Global p₀</span>
                <span>
                  Beta({fmt.num(global.alpha, 1)}, {fmt.num(global.beta, 1)})
                </span>
                <span>
                  mean <strong className="text-ink">{fmt.pct(global.mean, 2)}</strong> [{fmt.pct(global.hdi_low, 2)}, {fmt.pct(global.hdi_high, 2)}]
                </span>
                <span className="text-muted">
                  n {fmt.int(global.n_obs)} · {fmt.int(global.defaults)} defaults · {global.n_children} groups · recovery E[R] {fmt.pct(global.recovery?.mean)}
                </span>
              </div>
            )}
            <Card
              title="Groups"
              subtitle={`${shown.length} of ${groups.length} groups. Dot = posterior mean, bar = 95% HDI, tick = prior mean (global). Small groups are pulled toward the prior.`}
            >
              <div className="mb-3 flex flex-wrap items-end gap-4">
                {dims.map((d) => {
                  const vals = dimValues(groups, d);
                  return vals.length <= 8 ? (
                    <div key={d} className="flex flex-col gap-1">
                      <span className="text-2xs font-medium text-muted">{fmt.label(d)}</span>
                      <Chips size="sm" label={d} options={vals.map((v) => ({ value: v, label: fmt.label(v.toLowerCase()) }))} value={filters[d] ?? []} onChange={(v) => setFilters((f) => ({ ...f, [d]: v }))} />
                    </div>
                  ) : (
                    <label key={d} className="flex flex-col gap-1 text-2xs font-medium text-muted">
                      {fmt.label(d)}
                      <Select className="h-7 w-36 text-xs" value={filters[d]?.[0] ?? ''} onValue={(v) => setFilters((f) => ({ ...f, [d]: v ? [v] : [] }))} options={[{ value: '', label: 'Any' }, ...vals.map((v) => ({ value: v, label: v }))]} />
                    </label>
                  );
                })}
              </div>
              <DataTable<HierarchyRow>
                caption="Group posteriors"
                rows={shown}
                rowKey={(r) => r.key}
                selectedKey={group}
                onRowClick={(r) => setGroup(r.key === group ? null : r.key)}
                maxHeight={460}
                initialSort={{ key: 'n', dir: 'desc' }}
                columns={[
                  { key: 'group', header: 'Group', render: (r) => <span className="font-medium">{fmt.group(r.key)}</span>, sortValue: (r) => r.key },
                  { key: 'n', header: 'n', align: 'right', render: (r) => fmt.int(r.n_obs), sortValue: (r) => r.n_obs },
                  { key: 'd', header: 'Defaults', align: 'right', render: (r) => fmt.int(r.defaults), sortValue: (r) => r.defaults },
                  { key: 'emp', header: 'Empirical', align: 'right', render: (r) => fmt.pct(r.empirical_rate), sortValue: (r) => r.empirical_rate },
                  { key: 'mean', header: 'Posterior', align: 'right', render: (r) => <strong>{fmt.pct(r.mean)}</strong>, sortValue: (r) => r.mean },
                  {
                    key: 'hdi',
                    header: `95% HDI (0–${fmt.pct(max, 0)})`,
                    render: (r) => <IntervalBar mean={r.mean} low={r.hdi_low} high={r.hdi_high} prior={r.prior_mean} max={max} label={fmt.group(r.key)} />,
                    sortValue: (r) => r.hdi_high - r.hdi_low,
                  },
                  { key: 'shrink', header: 'Data weight n/(n+κ)', render: (r) => <Meter value={r.shrinkage_weight} label={`Shrinkage weight ${fmt.pct(r.shrinkage_weight)}`} />, sortValue: (r) => r.shrinkage_weight },
                  { key: 'rec', header: 'E[R | D]', align: 'right', render: (r) => fmt.pct(r.recovery?.mean), sortValue: (r) => r.recovery?.mean ?? null },
                  { key: 'leaves', header: 'Leaves', align: 'right', render: (r) => r.n_children },
                ]}
              />
            </Card>
            {group && <GroupDetail fit={fit} group={group} max={max} />}
          </div>
        );
      }}
    </QueryState>
  );
}
