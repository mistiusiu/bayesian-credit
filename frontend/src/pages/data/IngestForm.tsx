import { useState, type FormEvent } from 'react';
import { ApiError, useCreateDataset } from '@/api/client';
import type { Dataset, DataSource } from '@/api/types';
import { Button } from '@/components/ui/Button';
import { Disclosure } from '@/components/ui/Disclosure';
import { Field, FormError, NumberInput, TextInput } from '@/components/ui/Field';
import { Segmented } from '@/components/ui/Segmented';
import { Toggle } from '@/components/ui/Toggle';
import { fmt } from '@/lib/format';
import { ablationTags, matchingPreset } from '@/lib/variables';
import { VariablePicker } from './VariablePicker';

/** Lending Club accepted file: issue dates and size of the whole file. */
const LC_SPAN = { from: '2007-06', to: '2018-12', rows: 2_260_701 };

interface State {
  source: DataSource;
  name: string;
  n: number | null;
  seed: number | null;
  path: string;
  maxRows: number | null;
  fraction: number | null;
  from: string;
  to: string;
  shift: boolean;
  shiftAt: string;
  fico: number | null;
  dti: number | null;
  /** lending_club: read every row (no max_rows / sample_fraction). */
  wholeFile: boolean;
  exclude: string[];
}

const INITIAL: State = {
  source: 'synthetic',
  name: '',
  n: 200000,
  seed: 42,
  path: '',
  maxRows: 200000,
  fraction: null,
  from: '2012-01',
  to: '2018-12',
  shift: true,
  shiftAt: '2016-06',
  fico: -25,
  dti: 5,
  wholeFile: false,
  exclude: [],
};

/** Form state reproducing an existing dataset's parameters (to build an ablation of it). */
function fromDataset(d: Dataset): State {
  const p = d.params;
  const deltas = p.shift?.deltas ?? {};
  return {
    ...INITIAL,
    source: d.source,
    name: '',
    n: p.n ?? INITIAL.n,
    seed: p.seed ?? INITIAL.seed,
    path: p.path ?? '',
    maxRows: p.max_rows ?? null,
    fraction: p.sample_fraction ?? null,
    from: p.issue_from ?? INITIAL.from,
    to: p.issue_to ?? INITIAL.to,
    shift: !!p.shift,
    shiftAt: p.shift?.at ?? INITIAL.shiftAt,
    fico: deltas.fico_range_low ?? (p.shift ? null : INITIAL.fico),
    dti: deltas.dti ?? (p.shift ? null : INITIAL.dti),
    wholeFile: d.source === 'lending_club' && !p.max_rows && !p.sample_fraction,
    exclude: p.exclude_variables ?? [],
  };
}

function autoName(s: State, synthetic: boolean): string {
  const base = synthetic
    ? `Synthetic · ${Math.round((s.n ?? 0) / 1000)}k · seed ${s.seed ?? 42}`
    : `Lending Club · ${s.wholeFile || !s.maxRows ? `full ${s.from.slice(0, 4)}–${s.to.slice(0, 4)}` : `${Math.round(s.maxRows / 1000)}k rows`}`;
  if (!s.exclude.length) return base;
  return `${base} · ${matchingPreset(s.exclude, s.exclude)?.label ?? ablationTags(s.exclude).join(', ')}`;
}

export function IngestForm({ onCreated, template = null }: { onCreated: (d: Dataset) => void; template?: Dataset | null }) {
  const [s, setS] = useState<State>(() => (template ? fromDataset(template) : INITIAL));
  const create = useCreateDataset();
  const set = <K extends keyof State>(k: K, v: State[K]) => setS((p) => ({ ...p, [k]: v }));
  const errs = create.error instanceof ApiError ? create.error.fieldErrors : {};
  const synthetic = s.source === 'synthetic';

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const name = s.name.trim() || autoName(s, synthetic);
    create.mutate(
      {
        name,
        source: s.source,
        seed: s.seed,
        issue_from: s.from || null,
        issue_to: s.to || null,
        ...(synthetic
          ? { n: s.n, shift: s.shift ? { at: s.shiftAt, deltas: { ...(s.fico ? { fico_range_low: s.fico } : {}), ...(s.dti ? { dti: s.dti } : {}) } } : null }
          : { path: s.path || null, max_rows: s.wholeFile ? null : s.maxRows, sample_fraction: s.wholeFile ? null : s.fraction }),
        exclude_variables: s.exclude,
      },
      { onSuccess: (d) => onCreated(d) },
    );
  };

  return (
    <Disclosure
      title={template ? `New dataset from #${template.id}` : 'New dataset'}
      subtitle={template ? `Same source and window as ${template.name}; change the variables to build an ablation` : 'Ingest the Kaggle file or generate a synthetic twin, choosing the variables it keeps (runs as a background job)'}
      defaultOpen={!!template}
    >
      <form onSubmit={submit} className="space-y-4" noValidate>
        <div className="flex flex-wrap items-center gap-4">
          <Segmented<DataSource>
            label="Source"
            value={s.source}
            onChange={(v) => set('source', v)}
            options={[
              { value: 'synthetic', label: 'Synthetic twin' },
              { value: 'lending_club', label: 'Lending Club CSV' },
            ]}
          />
          <p className="text-xs text-muted">
            {synthetic
              ? 'LC-schema generator: sub-grade pricing, grade-driven default, delayed outcomes, optional distribution shift for the Sentinel.'
              : 'DuckDB projects only the retained columns, parses percents and Mon-YYYY dates, caches parquet.'}
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Field label="Name" error={errs.name}>
            {(id, d, inv) => <TextInput id={id} aria-describedby={d} aria-invalid={inv} value={s.name} placeholder="auto" onChange={(e) => set('name', e.target.value)} />}
          </Field>
          {synthetic ? (
            <Field label="Loans (n)" hint="1,000 – 2,500,000" error={errs.n}>
              {(id, d, inv) => <NumberInput id={id} aria-describedby={d} aria-invalid={inv} min={1000} step={1000} value={s.n} onValue={(v) => set('n', v)} />}
            </Field>
          ) : (
            <>
              <Field label="Path" hint="default: server ARTIFACT data/raw" error={errs.path}>
                {(id, d) => <TextInput id={id} aria-describedby={d} value={s.path} placeholder="accepted_2007_to_2018Q4.csv.gz" onChange={(e) => set('path', e.target.value)} />}
              </Field>
              <Field label="Max rows" hint={s.wholeFile ? 'whole file' : 'seeded random subset'} error={errs.max_rows}>
                {(id, d, inv) => <NumberInput id={id} aria-describedby={d} aria-invalid={inv} min={1000} step={1000} disabled={s.wholeFile} value={s.wholeFile ? null : s.maxRows} onValue={(v) => set('maxRows', v)} />}
              </Field>
              <Field label="Sample fraction" hint="(0, 1]" error={errs.sample_fraction}>
                {(id, d, inv) => <NumberInput id={id} aria-describedby={d} aria-invalid={inv} min={0} max={1} step={0.05} disabled={s.wholeFile} value={s.wholeFile ? null : s.fraction} onValue={(v) => set('fraction', v)} />}
              </Field>
            </>
          )}
          <Field label="Seed" error={errs.seed}>
            {(id, d) => <NumberInput id={id} aria-describedby={d} value={s.seed} onValue={(v) => set('seed', v)} />}
          </Field>
          <Field label="Issued from" hint="YYYY-MM" error={errs.issue_from}>
            {(id, d, inv) => <TextInput id={id} aria-describedby={d} aria-invalid={inv} value={s.from} pattern="\d{4}-\d{2}" onChange={(e) => set('from', e.target.value)} />}
          </Field>
          <Field label="Issued to" hint="YYYY-MM" error={errs.issue_to}>
            {(id, d, inv) => <TextInput id={id} aria-describedby={d} aria-invalid={inv} value={s.to} pattern="\d{4}-\d{2}" onChange={(e) => set('to', e.target.value)} />}
          </Field>
        </div>
        {!synthetic && (
          <div className="rounded-md border border-line bg-surface-2/60 p-3">
            <Toggle
              checked={s.wholeFile}
              onChange={(v) => setS((p) => ({ ...p, wholeFile: v, ...(v ? { from: LC_SPAN.from, to: LC_SPAN.to } : {}) }))}
              label="Entire dataset"
              description={`Every accepted loan in the issue window (the whole file spans ${LC_SPAN.from} – ${LC_SPAN.to}, about ${fmt.compact(LC_SPAN.rows)} loans) instead of a pilot subset. Ingest takes a few minutes; simulations can still sample from it.`}
            />
          </div>
        )}
        {synthetic && (
          <div className="rounded-md border border-line bg-surface-2/60 p-3">
            <Toggle checked={s.shift} onChange={(v) => set('shift', v)} label="Inject a distribution shift" description="Applied to borrowers issued on or after the shift month, so the PSI Sentinel has something to detect." />
            {s.shift && (
              <div className="mt-3 grid grid-cols-3 gap-3 lg:w-2/3">
                <Field label="Shift month" error={errs.shift}>
                  {(id, d) => <TextInput id={id} aria-describedby={d} value={s.shiftAt} onChange={(e) => set('shiftAt', e.target.value)} />}
                </Field>
                <Field label="Δ fico_range_low" hint="points">
                  {(id) => <NumberInput id={id} value={s.fico} onValue={(v) => set('fico', v)} />}
                </Field>
                <Field label="Δ dti" hint="percent points">
                  {(id) => <NumberInput id={id} value={s.dti} onValue={(v) => set('dti', v)} />}
                </Field>
              </div>
            )}
          </div>
        )}
        <fieldset className="rounded-md border border-line p-3">
          <legend className="px-1 text-xs font-semibold text-ink">Variables</legend>
          <VariablePicker value={s.exclude} onChange={(v) => set('exclude', v)} />
          {errs.exclude_variables && <p className="mt-2 text-2xs text-delta-bad">{errs.exclude_variables.join(' ')}</p>}
        </fieldset>
        <FormError message={create.error && !Object.keys(errs).length ? create.error.message : null} />
        <div className="flex items-center gap-3">
          <Button type="submit" variant="primary" icon="plus" loading={create.isPending}>
            Create dataset
          </Button>
          {create.isSuccess && <span className="text-xs text-muted">Queued: ingest job #{create.data.job?.id} — progress shows on the card below.</span>}
        </div>
      </form>
    </Disclosure>
  );
}
