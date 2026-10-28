import { useState } from 'react';
import { useVariables } from '@/api/client';
import type { Stage, VariableRow } from '@/api/types';
import { StageBadge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { Chips } from '@/components/ui/Chips';
import { DataTable } from '@/components/ui/DataTable';
import { Icon } from '@/components/ui/Icon';
import { QueryState } from '@/components/ui/States';
import { STAGE_BLURB, STAGE_LABEL, STAGES } from '@/lib/domain';
import { fmt } from '@/lib/format';

export function StageTable({ datasetId }: { datasetId: number }) {
  const q = useVariables(datasetId, true);
  const [stages, setStages] = useState<Stage[]>([]);
  return (
    <div className="space-y-4">
      <div className="grid gap-3 lg:grid-cols-5">
        {STAGES.map((s) => {
          const n = q.data?.filter((v) => v.stage === s).length ?? 0;
          return (
            <div key={s} className="rounded-lg border border-line bg-surface p-3">
              <div className="flex items-center justify-between">
                <StageBadge stage={s} />
                <span className="num text-sm font-semibold text-ink">{n}</span>
              </div>
              <p className="mt-2 text-2xs leading-snug text-muted">{STAGE_BLURB[s]}</p>
            </div>
          );
        })}
      </div>
      <p className="flex items-start gap-2 rounded-md border border-line bg-surface px-3 py-2 text-xs text-ink-2">
        <Icon name="lock" size={14} className="mt-0.5 shrink-0 text-muted" />
        LeakageGuard: a BorrowerContext may only contain pre-decision columns; anything later raises LeakageError. Policy columns are the arm, interim columns feed the MCTM and interim rewards, terminal columns only update posteriors after the outcome.
      </p>
      <Card
        title="Retained variables (Table 3)"
        subtitle="Stage, primary use, presence and missingness in this dataset"
        actions={<Chips size="sm" label="Filter by stage" options={STAGES.map((s) => ({ value: s, label: STAGE_LABEL[s] }))} value={stages} onChange={setStages} />}
      >
        <QueryState query={q} loadingLabel="Loading variables">
          {(rows) => (
            <DataTable<VariableRow>
              caption="Table 3 variables"
              rows={rows.filter((r) => !stages.length || stages.includes(r.stage))}
              rowKey={(r) => r.name}
              maxHeight={560}
              columns={[
                { key: 'name', header: 'Variable', render: (r) => <code className="text-xs text-ink">{r.name}</code>, sortValue: (r) => r.name },
                { key: 'stage', header: 'Stage', render: (r) => <StageBadge stage={r.stage} />, sortValue: (r) => STAGES.indexOf(r.stage) },
                { key: 'use', header: 'Primary use', render: (r) => <span className="text-ink-2">{r.primary_use}</span> },
                { key: 'kind', header: 'Kind', render: (r) => <span className="text-muted">{r.kind}</span> },
                {
                  key: 'present',
                  header: 'Present',
                  align: 'center',
                  render: (r) => (r.present ? <Icon name="check" size={13} aria-label="present" style={{ color: 'var(--status-good)' }} /> : <span className="text-muted">absent</span>),
                },
                { key: 'dtype', header: 'dtype', render: (r) => <code className="text-2xs text-muted">{r.dtype ?? '—'}</code> },
                {
                  key: 'null',
                  header: 'Null share',
                  align: 'right',
                  sortValue: (r) => r.null_share,
                  render: (r) => (
                    <span className="inline-flex items-center justify-end gap-2">
                      <span className="inline-block h-1.5 w-16 overflow-hidden rounded-full bg-surface-3" aria-hidden="true">
                        <span className="block h-full bg-[var(--text-axis)]" style={{ width: `${(r.null_share ?? 0) * 100}%` }} />
                      </span>
                      <span className="w-12">{fmt.pct(r.null_share, 1)}</span>
                    </span>
                  ),
                },
              ]}
            />
          )}
        </QueryState>
      </Card>
    </div>
  );
}
