import type { Simulation } from '@/api/types';
import { Badge, StatusBadge } from '@/components/ui/Badge';
import { DataTable } from '@/components/ui/DataTable';
import { ProgressBar } from '@/components/ui/Progress';
import { fmt } from '@/lib/format';
import { BREAKER_LABEL, BREAKER_TONE } from '@/lib/palette';

export function RunsTable({ sims, selectedId, onSelect }: { sims: Simulation[]; selectedId: number | null; onSelect: (id: number) => void }) {
  return (
    <DataTable<Simulation>
      caption="Simulation runs"
      rows={sims}
      rowKey={(s) => s.id}
      selectedKey={selectedId}
      onRowClick={(s) => onSelect(s.id)}
      empty="No simulations for this fit yet"
      columns={[
        { key: 'id', header: '#', render: (s) => s.id },
        { key: 'name', header: 'Name', render: (s) => <span className="font-medium">{s.name}</span> },
        { key: 'status', header: 'Status', render: (s) => <StatusBadge status={s.status} /> },
        { key: 'progress', header: 'Progress', render: (s) => <ProgressBar value={s.progress} label={`Progress of ${s.name}`} className="w-36" /> },
        { key: 'epoch', header: 'Epoch', render: (s) => fmt.epochShort(s.current_epoch) },
        { key: 'policies', header: 'Policies', align: 'right', render: (s) => s.config.policies.length },
        { key: 'feedback', header: 'Feedback', render: (s) => s.config.feedback_mode },
        { key: 'breaker', header: 'Breaker', render: (s) => <Badge tone={BREAKER_TONE[s.breaker_state]}>{BREAKER_LABEL[s.breaker_state]}</Badge> },
        { key: 'uplift', header: 'Uplift vs historical', align: 'right', render: (s) => fmt.usdCompactSigned(s.uplift_vs_historical) },
        { key: 'created', header: 'Created', render: (s) => <span className="text-muted">{fmt.dateTime(s.created_at)}</span> },
      ]}
    />
  );
}
