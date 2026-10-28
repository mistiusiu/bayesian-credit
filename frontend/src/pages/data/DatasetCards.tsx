import type { Dataset } from '@/api/types';
import { Pill, StatusBadge } from '@/components/ui/Badge';
import { ProgressBar } from '@/components/ui/Progress';
import { Empty, Loading } from '@/components/ui/States';
import { cn } from '@/lib/cn';
import { fmt } from '@/lib/format';

export function DatasetCards({ datasets, selectedId, onSelect, loading }: { datasets: Dataset[]; selectedId: number | null; onSelect: (id: number) => void; loading: boolean }) {
  if (loading && !datasets.length) return <Loading label="Loading datasets" />;
  if (!datasets.length) return <Empty title="No datasets yet">Create one above: the synthetic twin needs no download.</Empty>;
  return (
    <ul className="mt-4 grid gap-3 md:grid-cols-2 2xl:grid-cols-3" aria-label="Datasets">
      {datasets.map((d) => {
        const active = d.id === selectedId;
        const building = d.status === 'queued' || d.status === 'running';
        return (
          <li key={d.id}>
            <button
              type="button"
              aria-pressed={active}
              onClick={() => onSelect(d.id)}
              className={cn(
                'w-full rounded-lg border bg-surface p-3.5 text-left transition-colors hover:bg-surface-2',
                active ? 'border-ink/60 ring-1 ring-ink/30' : 'border-line',
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="truncate text-[13px] font-semibold text-ink">
                    #{d.id} {d.name}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    <Pill>{d.source === 'synthetic' ? 'synthetic' : 'lending club'}</Pill>
                    {d.params.shift && <Pill>shift {d.params.shift.at}</Pill>}
                    <Pill>seed {d.params.seed ?? '—'}</Pill>
                  </div>
                </div>
                <StatusBadge status={d.status} />
              </div>
              {building ? (
                <div className="mt-3">
                  <ProgressBar value={d.job?.progress ?? 0} label={`Ingest progress for ${d.name}`} />
                  <p className="mt-1 truncate text-2xs text-muted">{d.job?.message}</p>
                </div>
              ) : (
                <dl className="num mt-3 grid grid-cols-4 gap-2 text-xs">
                  <div>
                    <dt className="text-2xs text-muted">Loans</dt>
                    <dd className="font-medium text-ink">{fmt.compact(d.n_loans)}</dd>
                  </div>
                  <div>
                    <dt className="text-2xs text-muted">Issued</dt>
                    <dd className="font-medium text-ink">{d.issue_min ? `${d.issue_min.slice(0, 4)}–${d.issue_max?.slice(0, 4) ?? ''}` : '—'}</dd>
                  </div>
                  <div>
                    <dt className="text-2xs text-muted">Resolved</dt>
                    <dd className="font-medium text-ink">{fmt.pct(d.resolved_share, 0)}</dd>
                  </div>
                  <div>
                    <dt className="text-2xs text-muted">Default rate</dt>
                    <dd className="font-medium text-ink">{fmt.pct(d.default_rate)}</dd>
                  </div>
                </dl>
              )}
            </button>
          </li>
        );
      })}
    </ul>
  );
}
