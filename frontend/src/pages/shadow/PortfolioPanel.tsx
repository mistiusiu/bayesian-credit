import { useState, type FormEvent } from 'react';
import { ApiError, useCreatePortfolio, usePortfolio, usePortfolioRuns } from '@/api/client';
import type { Fit } from '@/api/types';
import { StatusBadge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Chips } from '@/components/ui/Chips';
import { DataTable } from '@/components/ui/DataTable';
import { Field, FormError, NumberInput } from '@/components/ui/Field';
import { ProgressBar } from '@/components/ui/Progress';
import { Empty, QueryState } from '@/components/ui/States';
import { Toggle } from '@/components/ui/Toggle';
import { GRADES } from '@/lib/domain';
import { fmt } from '@/lib/format';
import { PortfolioResultView } from './PortfolioResultView';

export function PortfolioPanel({ fit }: { fit: Fit }) {
  const runs = usePortfolioRuns(fit.id, true);
  const create = useCreatePortfolio(fit.id);
  const [maxLoans, setMaxLoans] = useState<number | null>(5000);
  const [seed, setSeed] = useState<number | null>(7);
  const [grades, setGrades] = useState<string[]>([]);
  const [resolvedOnly, setResolvedOnly] = useState(false);
  const [picked, setPicked] = useState<number | null>(null);
  const jobId = picked ?? runs.data?.[0]?.job.id ?? null;
  const result = usePortfolio(fit.id, jobId);
  const errs = create.error instanceof ApiError ? create.error.fieldErrors : {};
  const submit = (e: FormEvent) => {
    e.preventDefault();
    create.mutate({ max_loans: maxLoans ?? 5000, seed: seed ?? 7, filters: { grade: grades.length ? grades : undefined, resolved_only: resolvedOnly } }, { onSuccess: (job) => setPicked(job.id) });
  };
  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <Card title="Evaluate a portfolio" subtitle="Shadow vs historical on a sample of holdout borrowers (background job)">
          <form onSubmit={submit} className="space-y-3" noValidate>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Borrowers" hint="100 – 50,000" error={errs.max_loans}>
                {(id, d, inv) => <NumberInput id={id} aria-describedby={d} aria-invalid={inv} min={100} step={500} value={maxLoans} onValue={setMaxLoans} />}
              </Field>
              <Field label="Seed">{(id) => <NumberInput id={id} value={seed} onValue={setSeed} />}</Field>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium text-ink-2">Grades (empty = all)</span>
              <Chips size="sm" label="Grades" options={GRADES.map((g) => ({ value: g, label: g }))} value={grades} onChange={setGrades} />
            </div>
            <Toggle checked={resolvedOnly} onChange={setResolvedOnly} label="Resolved loans only" description="so realized historical profit is defined for every loan" />
            <FormError message={create.error && !Object.keys(errs).length ? create.error.message : null} />
            <Button type="submit" variant="primary" icon="play" loading={create.isPending}>
              Run portfolio evaluation
            </Button>
          </form>
        </Card>
        <Card title="Runs" bodyClassName="p-0">
          <QueryState query={runs} isEmpty={(r) => r.length === 0} empty={<Empty title="No portfolio runs yet" />}>
            {(rows) => (
              <DataTable
                caption="Portfolio runs"
                rows={rows}
                rowKey={(r) => r.job.id}
                selectedKey={jobId}
                onRowClick={(r) => setPicked(r.job.id)}
                columns={[
                  { key: 'job', header: 'Job', render: (r) => `#${r.job.id}` },
                  { key: 'status', header: 'Status', render: (r) => (r.job.status === 'running' ? <ProgressBar value={r.job.progress} label="Portfolio progress" className="w-28" /> : <StatusBadge status={r.job.status} />) },
                  { key: 'n', header: 'Borrowers', align: 'right', render: (r) => fmt.int(r.request.max_loans) },
                  { key: 'f', header: 'Filters', render: (r) => <span className="text-ink-2">{r.request.filters?.grade?.join(', ') || 'all grades'}{r.request.filters?.resolved_only ? ' · resolved' : ''}</span> },
                  { key: 'u', header: 'Uplift', align: 'right', render: (r) => fmt.usdCompactSigned(r.summary?.uplift) },
                  { key: 'c', header: 'Created', render: (r) => <span className="text-muted">{fmt.dateTime(r.job.created_at)}</span> },
                ]}
              />
            )}
          </QueryState>
        </Card>
      </div>
      {jobId !== null && (
        <QueryState query={result} loadingLabel="Loading portfolio result">
          {(res) =>
            res.summary ? (
              <PortfolioResultView result={res} fit={fit} />
            ) : (
              <Card title={`Portfolio job #${res.job.id}`} actions={<StatusBadge status={res.job.status} />}>
                <ProgressBar value={res.job.progress} label="Portfolio progress" />
                <ul className="mt-3 space-y-0.5 font-mono text-[11.5px] text-muted">
                  {res.job.log_tail.map((l) => (
                    <li key={l}>{l}</li>
                  ))}
                </ul>
              </Card>
            )
          }
        </QueryState>
      )}
    </div>
  );
}
