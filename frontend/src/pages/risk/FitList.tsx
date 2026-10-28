import type { Fit } from '@/api/types';
import { StatusBadge } from '@/components/ui/Badge';
import { DataTable } from '@/components/ui/DataTable';
import { Loading } from '@/components/ui/States';
import { fmt } from '@/lib/format';

export function FitList({ fits, selectedId, onSelect, loading }: { fits: Fit[]; selectedId: number | null; onSelect: (id: number) => void; loading: boolean }) {
  if (loading && !fits.length) return <Loading label="Loading fits" />;
  return (
    <DataTable<Fit>
      caption="Fits"
      rows={fits}
      rowKey={(f) => f.id}
      selectedKey={selectedId}
      onRowClick={(f) => onSelect(f.id)}
      empty="No fits on this dataset yet"
      columns={[
        { key: 'id', header: '#', render: (f) => f.id },
        { key: 'name', header: 'Name', render: (f) => <span className="font-medium">{f.name}</span> },
        { key: 'status', header: 'Status', render: (f) => <StatusBadge status={f.status} /> },
        { key: 'seg', header: 'Segmentation', render: (f) => <span className="text-ink-2">{f.config.segmentation.dimensions.join(' × ')}</span> },
        { key: 'groups', header: 'Groups', align: 'right', render: (f) => fmt.int(f.n_groups) },
        { key: 'arms', header: 'Arms', align: 'right', render: (f) => fmt.int(f.n_arms) },
        { key: 'mctm', header: 'MCTM', render: (f) => f.config.mctm.structure },
        { key: 'brier', header: 'Brier', align: 'right', render: (f) => fmt.num(f.metrics?.brier, 4), sortValue: (f) => f.metrics?.brier ?? null },
        { key: 'auc', header: 'AUC', align: 'right', render: (f) => fmt.num(f.metrics?.auc, 3), sortValue: (f) => f.metrics?.auc ?? null },
        { key: 'created', header: 'Created', render: (f) => <span className="text-muted">{fmt.dateTime(f.created_at)}</span> },
      ]}
    />
  );
}
