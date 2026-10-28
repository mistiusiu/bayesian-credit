import { useBuckets, useSimulation } from '@/api/client';
import type { PsiBucket } from '@/api/types';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { DataTable } from '@/components/ui/DataTable';
import { Select } from '@/components/ui/Field';
import { QueryState } from '@/components/ui/States';
import { fmt } from '@/lib/format';
import { LEVEL_TONE } from '@/lib/palette';

/** Expected (reference) vs actual (window) share per bucket and each bucket's PSI contribution. */
export function BucketTable({
  simId,
  features,
  epochs,
  feature,
  epoch,
  onFeature,
  onEpoch,
}: {
  simId: number;
  features: string[];
  epochs: string[];
  feature: string;
  epoch: string;
  onFeature: (f: string) => void;
  onEpoch: (e: string) => void;
}) {
  const sim = useSimulation(simId).data;
  const q = useBuckets(simId, sim, feature, epochs.includes(epoch) ? epoch : (epochs[epochs.length - 1] ?? null));
  return (
    <Card
      title="Bucket breakdown"
      subtitle="Where the shift comes from: reference share e (training window) vs window share a"
      actions={
        <div className="flex flex-wrap gap-2">
          <Select className="h-7 w-40 text-xs" aria-label="Feature" value={feature} onValue={onFeature} options={features.map((f) => ({ value: f, label: f }))} />
          <Select className="h-7 w-32 text-xs" aria-label="Epoch" value={epochs.includes(epoch) ? epoch : (epochs[epochs.length - 1] ?? '')} onValue={onEpoch} options={epochs.map((e) => ({ value: e, label: fmt.epoch(e) }))} />
        </div>
      }
    >
      <QueryState query={q} loadingLabel="Loading buckets">
        {(b) => {
          const maxShare = Math.max(...b.buckets.flatMap((x) => [x.expected, x.actual]), 0.01);
          const maxC = Math.max(...b.buckets.map((x) => x.contribution), 1e-6);
          return (
            <div className="space-y-2">
              <p className="num flex flex-wrap items-center gap-2 text-xs text-ink-2">
                <code className="text-ink">{b.feature}</code> · {fmt.epoch(b.epoch)} · PSI <strong className="text-ink">{fmt.num(b.psi, 3)}</strong>
                <Badge tone={LEVEL_TONE[b.level]}>{b.level}</Badge>
                <span className="text-muted">
                  reference n {fmt.int(b.n_reference)} · window n {fmt.int(b.n_window)}
                </span>
              </p>
              <DataTable<PsiBucket>
                caption="PSI buckets"
                rows={b.buckets}
                rowKey={(x) => x.bucket}
                columns={[
                  { key: 'bucket', header: b.kind === 'numeric' ? 'Bin' : 'Level', render: (x) => <span className="num">{x.bucket}</span> },
                  {
                    key: 'shares',
                    header: 'Reference e vs window a',
                    render: (x) => (
                      <span className="flex flex-col gap-0.5 py-0.5" aria-hidden="true">
                        <span className="block h-1.5 rounded-r-sm bg-[var(--text-axis)]" style={{ width: `${(x.expected / maxShare) * 140}px` }} />
                        <span className="block h-1.5 rounded-r-sm bg-[var(--series-1)]" style={{ width: `${(x.actual / maxShare) * 140}px` }} />
                      </span>
                    ),
                  },
                  { key: 'e', header: 'Expected e', align: 'right', render: (x) => fmt.pct(x.expected, 1) },
                  { key: 'a', header: 'Actual a', align: 'right', render: (x) => fmt.pct(x.actual, 1) },
                  { key: 'd', header: 'a − e', align: 'right', render: (x) => fmt.pctSigned(x.actual - x.expected, 1) },
                  {
                    key: 'c',
                    header: 'Contribution',
                    align: 'right',
                    sortValue: (x) => x.contribution,
                    render: (x) => (
                      <span className="inline-flex items-center justify-end gap-2">
                        <span className="inline-block h-1.5 rounded-sm bg-[var(--series-1)]" style={{ width: `${(x.contribution / maxC) * 60}px` }} aria-hidden="true" />
                        <span className="w-14">{fmt.num(x.contribution, 4)}</span>
                      </span>
                    ),
                  },
                ]}
              />
              <p className="text-2xs text-muted">Bars: gray = reference share, blue = current window share. Contributions sum to the feature PSI.</p>
            </div>
          );
        }}
      </QueryState>
    </Card>
  );
}
